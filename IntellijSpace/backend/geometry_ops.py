import os
from typing import Optional

import matplotlib
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def get_scene_align_rotation() -> np.ndarray:
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 100, degrees=True).as_matrix()
    align_rotation[:3, :3] = align_rotation[:3, :3] @ Rotation.from_euler("x", 155, degrees=True).as_matrix()
    return align_rotation


def compute_target_bbox_from_ref_mask(predictions: dict, align_transform: Optional[np.ndarray] = None):
    if not isinstance(predictions, dict):
        return None

    if "points" not in predictions or "referring_mask_pred" not in predictions:
        return None

    points = np.asarray(predictions["points"])
    ref_mask = np.asarray(predictions["referring_mask_pred"])

    if points.size == 0 or ref_mask.size == 0:
        return None

    flat_points = points.reshape(-1, 3)
    flat_mask = ref_mask.reshape(-1) > 0

    if flat_points.shape[0] != flat_mask.shape[0]:
        return None

    target_points = flat_points[flat_mask]
    if target_points.shape[0] == 0:
        return None

    if align_transform is not None and np.asarray(align_transform).shape == (4, 4):
        target_points_h = np.concatenate(
            [target_points, np.ones((target_points.shape[0], 1), dtype=target_points.dtype)], axis=1
        )
        target_points = (target_points_h @ np.asarray(align_transform).T)[:, :3]

    min_xyz = np.min(target_points, axis=0)
    max_xyz = np.max(target_points, axis=0)
    center_xyz = (min_xyz + max_xyz) / 2.0
    size_xyz = max_xyz - min_xyz

    return {
        "min": min_xyz.astype(float).tolist(),
        "max": max_xyz.astype(float).tolist(),
        "center": center_xyz.astype(float).tolist(),
        "size": size_xyz.astype(float).tolist(),
        "num_points": int(target_points.shape[0]),
    }


def _wrap_angle_pi(angle_rad: float) -> float:
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi


def _snap_yaw_to_right_angles(yaw_rad: float) -> tuple[float, bool]:
    candidates = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2], dtype=np.float64)
    diffs = np.array([abs(_wrap_angle_pi(yaw_rad - c)) for c in candidates])
    idx = int(np.argmin(diffs))
    return float(candidates[idx]), True


def estimate_target_yaw_from_mask(
    predictions: dict,
    align_transform: Optional[np.ndarray] = None,
    anisotropy_threshold: float = 1.25,
):
    if not isinstance(predictions, dict):
        return {"valid": False, "reason": "invalid_predictions"}

    if "points" not in predictions or "referring_mask_pred" not in predictions:
        return {"valid": False, "reason": "missing_points_or_mask"}

    points = np.asarray(predictions["points"])
    ref_mask = np.asarray(predictions["referring_mask_pred"])

    if points.size == 0 or ref_mask.size == 0:
        return {"valid": False, "reason": "empty_points_or_mask"}

    flat_points = points.reshape(-1, 3)
    flat_mask = ref_mask.reshape(-1) > 0
    if flat_points.shape[0] != flat_mask.shape[0]:
        return {"valid": False, "reason": "shape_mismatch"}

    target_points = flat_points[flat_mask]
    if target_points.shape[0] < 8:
        return {"valid": False, "reason": "too_few_points"}

    if align_transform is not None and np.asarray(align_transform).shape == (4, 4):
        target_points_h = np.concatenate(
            [target_points, np.ones((target_points.shape[0], 1), dtype=target_points.dtype)], axis=1
        )
        target_points = (target_points_h @ np.asarray(align_transform).T)[:, :3]

    xz = target_points[:, [0, 2]].astype(np.float64)
    xz_mean = np.mean(xz, axis=0)
    xz_centered = xz - xz_mean

    cov = np.cov(xz_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    major = eigvecs[:, 0]
    raw_yaw = float(np.arctan2(major[1], major[0]))

    l1 = float(max(eigvals[0], 1e-9))
    l2 = float(max(eigvals[1], 1e-9))
    anisotropy_ratio = float(np.sqrt(l1 / l2))

    return {
        "valid": anisotropy_ratio >= anisotropy_threshold,
        "raw_yaw_rad": raw_yaw,
        "anisotropy_ratio": anisotropy_ratio,
        "reason": "ok" if anisotropy_ratio >= anisotropy_threshold else "near_isotropic",
    }


def place_asset_into_scene(
    scene_3d: trimesh.Scene,
    selected_asset_path: str,
    target_bbox: dict,
    vertical_axis: int = 1,
    min_size_eps: float = 1e-6,
    lift_ratio: float = 0.001,
    yaw_rad: Optional[float] = None,
    fit_scale_override: Optional[float] = None,
):
    placement_info = {"status": "skipped", "reason": "unknown"}

    if scene_3d is None:
        placement_info["reason"] = "scene_none"
        return scene_3d, placement_info

    if not selected_asset_path or not os.path.exists(selected_asset_path):
        placement_info["reason"] = "asset_not_found"
        return scene_3d, placement_info

    if not isinstance(target_bbox, dict) or "size" not in target_bbox or "center" not in target_bbox:
        placement_info["reason"] = "target_bbox_missing"
        return scene_3d, placement_info

    target_size = np.asarray(target_bbox.get("size", []), dtype=np.float64)
    target_center = np.asarray(target_bbox.get("center", []), dtype=np.float64)

    if target_size.shape[0] != 3 or target_center.shape[0] != 3:
        placement_info["reason"] = "target_bbox_invalid"
        return scene_3d, placement_info

    if np.any(target_size <= min_size_eps):
        placement_info["reason"] = "target_bbox_too_small"
        return scene_3d, placement_info

    try:
        asset_loaded = trimesh.load(selected_asset_path, force="scene")
    except Exception as exc:
        placement_info["reason"] = f"asset_load_failed:{exc}"
        return scene_3d, placement_info

    asset_scene = asset_loaded.copy() if isinstance(asset_loaded, trimesh.Scene) else trimesh.Scene(asset_loaded)

    asset_bounds = asset_scene.bounds
    if asset_bounds is None or np.asarray(asset_bounds).shape != (2, 3):
        placement_info["reason"] = "asset_bounds_invalid"
        return scene_3d, placement_info

    asset_min = asset_bounds[0].astype(np.float64)
    asset_max = asset_bounds[1].astype(np.float64)
    asset_size = asset_max - asset_min
    if np.any(asset_size <= min_size_eps):
        placement_info["reason"] = "asset_size_too_small"
        return scene_3d, placement_info

    axis = 1 if vertical_axis not in (0, 1, 2) else int(vertical_axis)

    asset_bottom_center = (asset_min + asset_max) / 2.0
    asset_bottom_center[axis] = asset_min[axis]

    target_bottom_center = target_center.copy()
    target_bottom_center[axis] = target_center[axis] - target_size[axis] / 2.0

    fit_scale = float(np.min(target_size / asset_size))
    if fit_scale_override is not None:
        fit_scale = float(fit_scale_override)
    if fit_scale <= min_size_eps:
        placement_info["reason"] = "fit_scale_invalid"
        return scene_3d, placement_info

    lift = float(target_size[axis] * lift_ratio)

    scale_m = np.eye(4, dtype=np.float64)
    scale_m[:3, :3] = np.eye(3, dtype=np.float64) * fit_scale

    yaw_used = 0.0 if yaw_rad is None else float(yaw_rad)
    yaw_m = np.eye(4, dtype=np.float64)
    c = float(np.cos(yaw_used))
    s = float(np.sin(yaw_used))
    yaw_m[0, 0] = c
    yaw_m[0, 2] = -s
    yaw_m[2, 0] = s
    yaw_m[2, 2] = c

    pivot_to_origin = np.eye(4, dtype=np.float64)
    pivot_to_origin[:3, 3] = -asset_bottom_center

    to_target = np.eye(4, dtype=np.float64)
    to_target[:3, 3] = target_bottom_center
    to_target[axis, 3] += lift

    transform = to_target @ yaw_m @ scale_m @ pivot_to_origin
    asset_scene.apply_transform(transform)

    inserted_node = "inserted_asset"
    inserted_geom = "inserted_asset_geom"
    nodes_before = set(scene_3d.graph.nodes)
    try:
        scene_3d.add_geometry(
            asset_scene,
            node_name=inserted_node,
            geom_name=inserted_geom,
        )
    except Exception:
        try:
            if asset_scene.geometry:
                merged = trimesh.util.concatenate(list(asset_scene.geometry.values()))
                scene_3d.add_geometry(
                    merged,
                    node_name=inserted_node,
                    geom_name=inserted_geom,
                )
            else:
                placement_info["reason"] = "asset_empty"
                return scene_3d, placement_info
        except Exception:
            for i, (gname, geom) in enumerate(asset_scene.geometry.items()):
                suffix = gname if isinstance(gname, str) else str(i)
                scene_3d.add_geometry(
                    geom,
                    node_name=f"{inserted_node}_{suffix}",
                    geom_name=f"{inserted_geom}_{suffix}",
                )
    nodes_after = set(scene_3d.graph.nodes)
    asset_node_names = sorted(nodes_after - nodes_before)

    placement_info = {
        "status": "placed",
        "asset_path": selected_asset_path,
        "scale": fit_scale,
        "target_center": target_center.astype(float).tolist(),
        "target_size": target_size.astype(float).tolist(),
        "vertical_axis": axis,
        "lift": lift,
        "yaw_rad": yaw_used,
        "yaw_deg": float(np.degrees(yaw_used)),
        "asset_node_names": asset_node_names,
    }
    return scene_3d, placement_info


def predictions_to_glb(
    predictions: dict,
    conf_thres: float = 50.0,
    filter_by_frames: str = "all",
    show_cam: bool = True,
    apply_mask: bool = False,
    action: str = "SEGMENT",
) -> trimesh.Scene:
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    if conf_thres is None:
        conf_thres = 10

    selected_frame_idx = None
    if filter_by_frames not in {"all", "All"}:
        try:
            selected_frame_idx = int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            pass

    pred_world_points = predictions["points"]
    pred_world_points_conf = predictions.get("conf", np.ones_like(pred_world_points[..., 0]))
    images = predictions["images"]
    camera_poses = predictions["camera_poses"]

    ref_mask = predictions.get("referring_mask_pred")

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        camera_poses = camera_poses[selected_frame_idx][None]
        if ref_mask is not None:
            ref_mask = ref_mask[selected_frame_idx][None]

    vertices_3d = pred_world_points.reshape(-1, 3)
    if images.ndim == 4 and images.shape[1] == 3:
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    if ref_mask is not None:
        flat_mask = ref_mask.reshape(-1)
        target_indices = flat_mask > 0

        normalized_action = str(action).strip().upper() if action is not None else "SEGMENT"
        if normalized_action not in {"REPLACE", "REMOVE", "SEGMENT"}:
            normalized_action = "SEGMENT"

        if normalized_action in {"REPLACE", "REMOVE"}:
            keep_indices = ~target_indices
            vertices_3d = vertices_3d[keep_indices]
            colors_rgb = colors_rgb[keep_indices]
            pred_world_points_conf = pred_world_points_conf.reshape(-1)[keep_indices]
        elif apply_mask:
            colors_rgb[target_indices] = [255, 0, 0]

    conf = pred_world_points_conf.reshape(-1)
    conf_threshold = 0.0 if conf_thres == 0.0 else conf_thres / 100
    conf_mask = conf >= conf_threshold

    kept_before = int(vertices_3d.shape[0])
    vertices_try = vertices_3d[conf_mask]
    colors_try = colors_rgb[conf_mask]

    if vertices_try.shape[0] < 100:
        # Fallback: when confidence filtering removes almost everything, keep raw points
        vertices_3d = vertices_3d
        colors_rgb = colors_rgb
    else:
        vertices_3d = vertices_try
        colors_rgb = colors_try

    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")
    scene_3d = trimesh.Scene()
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)
    scene_3d.add_geometry(point_cloud_data)

    num_cameras = len(camera_poses)
    if show_cam:
        for i in range(num_cameras):
            camera_to_world = camera_poses[i]
            rgba_color = colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])
            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale)

    align_rotation = get_scene_align_rotation()
    scene_3d.apply_transform(align_rotation)

    return scene_3d


def integrate_camera_into_scene(scene: trimesh.Scene, transform: np.ndarray, face_colors: tuple, scene_scale: float):
    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    opengl_transform = get_opengl_conversion_matrix()
    complete_transform = transform @ opengl_transform @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    vertices_combined = np.concatenate(
        [
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            transform_points(slight_rotation, camera_cone_shape.vertices),
        ]
    )
    vertices_transformed = transform_points(complete_transform, vertices_combined)
    mesh_faces = compute_camera_faces(camera_cone_shape)

    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def get_opengl_conversion_matrix() -> np.ndarray:
    matrix = np.identity(4)
    matrix[1, 1] = -1
    matrix[2, 2] = -1
    return matrix


def transform_points(transformation: np.ndarray, points: np.ndarray, dim: Optional[int] = None) -> np.ndarray:
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]
    transformation = transformation.swapaxes(-1, -2)
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]
    result = points[..., :dim].reshape(*initial_shape, dim)
    return result


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    faces_list = []
    num_vertices_cone = len(cone_shape.vertices)

    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices_cone
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone

        faces_list.extend(
            [
                (v1, v2, v2_offset),
                (v1, v1_offset, v3),
                (v3_offset, v2, v3),
                (v1, v2, v2_offset_2),
                (v1, v1_offset_2, v3),
                (v3_offset_2, v2, v3),
            ]
        )

    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)
