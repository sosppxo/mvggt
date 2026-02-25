import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time
# import spaces         # only for web demo

from mvggt.utils.geometry import se3_inverse, homogenize_points, depth_edge
from mvggt.models.mvggt_training import MVGGT
from mvggt.utils.basic import load_images_as_tensor

import trimesh
import matplotlib
from scipy.spatial.transform import Rotation
from transformers import RobertaTokenizer


"""
Gradio utils
"""

def predictions_to_glb(
    predictions,
    conf_thres=50.0,
    filter_by_frames="all",
    show_cam=True,
    apply_mask=False,
) -> trimesh.Scene:
    """
    Converts VGGT predictions to a 3D scene represented as a GLB file.

    Args:
        predictions (dict): Dictionary containing model predictions with keys:
            - world_points: 3D point coordinates (S, H, W, 3)
            - world_points_conf: Confidence scores (S, H, W)
            - images: Input images (S, H, W, 3)
            - extrinsic: Camera extrinsic matrices (S, 3, 4)
        conf_thres (float): Percentage of low-confidence points to filter out (default: 50.0)
        filter_by_frames (str): Frame filter specification (default: "all")
        show_cam (bool): Include camera visualization (default: True)
        apply_mask (bool): Whether to highlight points based on referring mask (default: False)

    Returns:
        trimesh.Scene: Processed 3D scene containing point cloud and cameras

    Raises:
        ValueError: If input predictions structure is invalid
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    if conf_thres is None:
        conf_thres = 10

    print("Building GLB scene")
    selected_frame_idx = None
    if filter_by_frames != "all" and filter_by_frames != "All":
        try:
            # Extract the index part before the colon
            selected_frame_idx = int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            pass

    pred_world_points = predictions["points"]
    pred_world_points_conf = predictions.get("conf", np.ones_like(pred_world_points[..., 0]))

    # Get images from predictions
    images = predictions["images"]
    # Use extrinsic matrices instead of pred_extrinsic_list
    camera_poses = predictions["camera_poses"]

    ref_mask = None
    if apply_mask and "referring_mask_pred" in predictions:
        ref_mask = predictions["referring_mask_pred"]

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        camera_poses = camera_poses[selected_frame_idx][None]
        if ref_mask is not None:
             ref_mask = ref_mask[selected_frame_idx][None]

    vertices_3d = pred_world_points.reshape(-1, 3)
    # Handle different image formats - check if images need transposing
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    # Apply Highlight if mask is present
    if ref_mask is not None:
        flat_mask = ref_mask.reshape(-1)
        # Assuming values > 0 are the mask
        highlight_indices = flat_mask > 0
        
        # Apply red color to highlighted points
        colors_rgb[highlight_indices] = [255, 0, 0]

    conf = pred_world_points_conf.reshape(-1)
    # Convert percentage threshold to actual confidence value
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        # conf_threshold = np.percentile(conf, conf_thres)
        conf_threshold = conf_thres / 100

    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    # # Random sample 20000 points if there are too many
    # num_points = vertices_3d.shape[0]
    # if num_points > 200000:
    #     print(f"Downsampling from {num_points} to 200000 points")
    #     indices = np.random.choice(num_points, 200000, replace=False)
    #     vertices_3d = vertices_3d[indices]
    #     colors_rgb = colors_rgb[indices]

    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        # Calculate the 5th and 95th percentiles along each axis
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)

        # Calculate the diagonal length of the percentile bounding box
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

    # Initialize a 3D scene
    scene_3d = trimesh.Scene()

    # Add point cloud data to the scene
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)

    scene_3d.add_geometry(point_cloud_data)

    # Prepare 4x4 matrices for camera extrinsics
    num_cameras = len(camera_poses)

    if show_cam:
        # Add camera models to the scene
        for i in range(num_cameras):
            camera_to_world = camera_poses[i]
            rgba_color = colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])

            # integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale)
            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, 1.)          # fixed camera size

    # Rotate scene for better visualize
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 100, degrees=True).as_matrix()            # plane rotate
    align_rotation[:3, :3] = align_rotation[:3, :3] @ Rotation.from_euler("x", 155, degrees=True).as_matrix()           # roll
    scene_3d.apply_transform(align_rotation)

    print("GLB Scene built")
    return scene_3d

def integrate_camera_into_scene(scene: trimesh.Scene, transform: np.ndarray, face_colors: tuple, scene_scale: float):
    """
    Integrates a fake camera mesh into the 3D scene.

    Args:
        scene (trimesh.Scene): The 3D scene to add the camera model.
        transform (np.ndarray): Transformation matrix for camera positioning.
        face_colors (tuple): Color of the camera face.
        scene_scale (float): Scale of the scene.
    """

    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    # Create cone shape for camera
    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    opengl_transform = get_opengl_conversion_matrix()
    # Combine transformations
    complete_transform = transform @ opengl_transform @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    # Generate mesh for the camera
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

    # Add the camera mesh to the scene
    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def get_opengl_conversion_matrix() -> np.ndarray:
    """
    Constructs and returns the OpenGL conversion matrix.

    Returns:
        numpy.ndarray: A 4x4 OpenGL conversion matrix.
    """
    # Create an identity matrix
    matrix = np.identity(4)

    # Flip the y and z axes
    matrix[1, 1] = -1
    matrix[2, 2] = -1

    return matrix


def transform_points(transformation: np.ndarray, points: np.ndarray, dim: int = None) -> np.ndarray:
    """
    Applies a 4x4 transformation to a set of points.

    Args:
        transformation (np.ndarray): Transformation matrix.
        points (np.ndarray): Points to be transformed.
        dim (int, optional): Dimension for reshaping the result.

    Returns:
        np.ndarray: Transformed points.
    """
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]

    # Apply transformation
    transformation = transformation.swapaxes(-1, -2)  # Transpose the transformation matrix
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]

    # Reshape the result
    result = points[..., :dim].reshape(*initial_shape, dim)
    return result


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    """
    Computes the faces for the camera mesh.

    Args:
        cone_shape (trimesh.Trimesh): The shape of the camera cone.

    Returns:
        np.ndarray: Array of faces for the camera mesh.
    """
    # Create pseudo cameras
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


# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
# @spaces.GPU(duration=120)
def run_model(target_dir, model, text_prompt=None) -> dict:
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    # interval = 10 if target_dir.endswith('.mp4') else 1
    interval = 1
    imgs = load_images_as_tensor(os.path.join(target_dir, "images"), interval=interval).to(device) # (N, 3, H, W)

    # 3. Infer
    print("Running model inference...")

    input_ids = None
    attention_mask = None
    if text_prompt is not None and len(text_prompt) > 0:
        print(f"Tokenizing text prompt: {text_prompt}")
        text_inputs = tokenizer(text_prompt, return_tensors="pt")
        input_ids = text_inputs['input_ids'].to(device)
        attention_mask = text_inputs['attention_mask'].to(device)

    dtype = torch.bfloat16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            predictions = model(imgs[None], input_ids=input_ids, attention_mask=attention_mask) # Add batch dimension
    predictions['images'] = imgs[None].permute(0, 1, 3, 4, 2)
    predictions['conf'] = torch.sigmoid(predictions['conf'])
    edge = depth_edge(predictions['local_points'][..., 2], rtol=0.03)
    predictions['conf'][edge] = 0.0
    del predictions['local_points']

    # Remove intermediate list results that cause saving issues
    if 'layer_referring_mask_preds' in predictions:
        del predictions['layer_referring_mask_preds']
        print("del predictions['layer_referring_mask_preds']")

    # Handle referring mask prediction if present
    if 'referring_mask_pred' in predictions:
        print("Referring mask prediction found.")
        # Ensure it's on CPU and numpy for saving
        # It's typically (B, N, H, W) or similar, handled in the generic loop below
        pass

    # # transform to first camera coordinate
    # predictions['points'] = torch.einsum('bij, bnhwj -> bnhwi', se3_inverse(predictions['camera_poses'][:, 0]), homogenize_points(predictions['points']))[..., :3]
    # predictions['camera_poses'] = torch.einsum('bij, bnjk -> bnik', se3_inverse(predictions['camera_poses'][:, 0]), predictions['camera_poses'])

    # Convert tensors to numpy
    # Use list(predictions.keys()) to avoid runtime error if dict changes size, though we're just modifying values
    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension

    # Clean up
    torch.cuda.empty_cache()
    return predictions


# -------------------------------------------------------------------------
# 2) Handle uploaded video/images --> produce target_dir + images
# -------------------------------------------------------------------------
def handle_uploads(input_video, input_images, interval=-1):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images or extracted frames from video into it. Return (target_dir, image_paths).
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(target_dir_images, exist_ok=True)

    image_paths = []

    # --- Handle images ---
    if input_images is not None:
        if interval is not None and interval > 0:
            input_images = input_images[::interval]

        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)
        
    # --- Handle video ---
    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        if interval is not None and interval > 0:
            frame_interval = interval
        else:
            frame_interval = int(fps * 1)  # 1 frame/sec

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


# -------------------------------------------------------------------------
# 3) Update gallery on upload
# -------------------------------------------------------------------------
def update_gallery_on_upload(input_video, input_images, interval=-1):
    """
    Whenever user uploads or changes files, immediately handle them
    and show in the gallery. Return (target_dir, image_paths).
    If nothing is uploaded, returns "None" and empty list.
    """
    if not input_video and not input_images:
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images, interval=interval)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."


# -------------------------------------------------------------------------
# 4) Reconstruction: uses the target_dir plus any viz parameters
# -------------------------------------------------------------------------
# commit below for local demo
# @spaces.GPU(duration=120)
def gradio_demo(
    target_dir,
    conf_thres=3.0,
    frame_filter="All",
    show_cam=True,
    text_prompt=None,
    apply_mask=False,
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None, None
    
    if not text_prompt or len(text_prompt.strip()) == 0:
        return None, "Text prompt is required. Please enter a text description.", None, None

    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running run_model...")
    with torch.no_grad():
        predictions = run_model(target_dir, model, text_prompt)

    # Save predictions
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    # Handle None frame_filter
    if frame_filter is None:
        frame_filter = "All"

    # Build a GLB file name
    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_cam{show_cam}_mask{apply_mask}.glb",
    )

    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        show_cam=show_cam,
        apply_mask=apply_mask,
    )
    glbscene.export(file_obj=glbfile)

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."

    return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True)


# -------------------------------------------------------------------------
# 5) Helper functions for UI resets + re-visualization
# -------------------------------------------------------------------------
def clear_fields():
    """
    Clears the 3D viewer, the stored target_dir, and empties the gallery.
    """
    return None


def update_log():
    """
    Display a quick log message while waiting.
    """
    return "Loading and Reconstructing..."


def update_visualization(
    target_dir, conf_thres, frame_filter, show_cam, is_example, apply_mask
):
    """
    Reload saved predictions from npz, create (or reuse) the GLB for new parameters,
    and return it for the 3D viewer. If is_example == "True", skip.
    """

    # If it's an example click, skip as requested
    if is_example == "True":
        return None, "No reconstruction available. Please click the Reconstruct button first."

    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Please click the Reconstruct button first."

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first."

    key_list = [
        "images",
        "points",
        "conf",
        "camera_poses",
    ]

    loaded = np.load(predictions_path)
    # Check if referring_mask_pred exists in loaded files
    if "referring_mask_pred" in loaded:
        key_list.append("referring_mask_pred")
    
    predictions = {key: np.array(loaded[key]) for key in key_list}

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_cam{show_cam}_mask{apply_mask}.glb",
    )

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            show_cam=show_cam,
            apply_mask=apply_mask,
        )
        glbscene.export(file_obj=glbfile)

    return glbfile, "Updating Visualization"

# -------------------------------------------------------------------------
# 6) Build Gradio UI
# -------------------------------------------------------------------------

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Initializing and loading MVGGT model...")
    
    ckpt_path = 'ckpts/best/pytorch_model.bin'
    model = MVGGT(
        use_referring_segmentation=True,
        train_conf=True, 
        ckpt=ckpt_path,
    )

    print(f"Loading checkpoint from {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
        
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Load result: {msg}")
    model.eval()
    model = model.to(device)

    tokenizer = RobertaTokenizer.from_pretrained("./ckpts/roberta-base")

    theme = gr.themes.Soft()

    with gr.Blocks(
        theme=theme,
        css="""
        @import url('https://fonts.googleapis.com/css?family=Google+Sans|Noto+Sans|Castoro');
        
        body {
            font-family: 'Noto Sans', sans-serif;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }
        
        .publication-header {
            text-align: center;
            margin-bottom: 30px;
            padding-top: 20px;
        }
        
        .publication-title {
            font-family: 'Google Sans', sans-serif;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            line-height: 1.2;
            color: #363636;
        }
        
        .publication-authors {
            font-family: 'Google Sans', sans-serif;
            font-size: 1.25rem;
            margin-bottom: 1rem;
        }
        
        .author-block {
            display: inline-block;
            margin-right: 10px;
            color: #363636;
        }
        
        .author-block a {
            color: #2563eb;
            text-decoration: none;
        }
        
        .affiliations {
            font-size: 1.1rem;
            margin-bottom: 1rem;
            color: #4a4a4a;
        }
        
        .publication-links {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 20px;
        }
        
        .link-block a {
            display: inline-flex;
            align-items: center;
            padding: 8px 18px;
            background-color: #363636;
            color: #fff !important;
            border-radius: 290486px;
            font-size: 1rem;
            text-decoration: none;
            transition: background-color 0.3s ease;
        }
        
        .link-block a:hover {
            background-color: #4a4a4a;
        }
        
        .link-block .icon {
            margin-right: 5px;
        }
        
        .teaser-section {
            margin: 40px auto;
            text-align: center;
            max-width: 85%;
        }
        
        .teaser-image {
            width: 100%;
            border-radius: 5px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
        }
        
        .instructions {
            margin: 40px auto;
            max-width: 800px;
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 10px;
            text-align: left;
        }
        
        .instructions h3 {
            font-family: 'Google Sans', sans-serif;
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 10px;
        }
        
        .instructions ol {
            padding-left: 20px;
        }
        
        .instructions li {
            margin-bottom: 8px;
            font-size: 1rem;
            color: #4a4a4a;
        }
        
        /* Custom section styling to match */
        .section-header {
            font-family: 'Google Sans', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 15px;
            color: #363636;
            border-bottom: none;
        }
        """,
    ) as demo:
        # Instead of gr.State, we use a hidden Textbox:
        is_example = gr.Textbox(label="is_example", visible=False, value="None")
        num_images = gr.Textbox(label="num_images", visible=False, value="None")
        target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

        gr.HTML(
        """
        <div class="publication-header">
            <h1 class="publication-title">MVGGT: Multimodal Visual Geometry Grounded Transformer for Multiview 3D Referring Expression Segmentation</h1>
            
            <div class="publication-authors">
                <span class="author-block">Changli Wu<sup>1, 2, †</sup>,</span>
                <span class="author-block">Haodong Wang<sup>1, †</sup>,</span>
                <span class="author-block">Jiayi Ji<sup>1</sup>,</span>
                <span class="author-block">Yutian Yao<sup>5</sup>,</span>
                <br>
                <span class="author-block">Chunsai Du<sup>4</sup>,</span>
                <span class="author-block">Jihua Kang<sup>4</sup>,</span>
                <span class="author-block">Yanwei Fu<sup>3, 2</sup>,</span>
                <span class="author-block">Liujuan Cao<sup>1, *</sup></span>
            </div>
            
            <div class="affiliations">
                <span class="author-block"><sup>1</sup>Xiamen University,</span>
                <span class="author-block"><sup>2</sup>Shanghai Innovation Institute,</span>
                <span class="author-block"><sup>3</sup>Fudan University,</span>
                <br>
                <span class="author-block"><sup>4</sup>ByteDance,</span>
                <span class="author-block"><sup>5</sup>Tianjin University of Science and Technology</span>
            </div>
            
            <div class="affiliations" style="font-size: 0.9em;">
                <span class="author-block"><sup>†</sup>Equal Contribution,</span>
                <span class="author-block"><sup>*</sup>Corresponding Author</span>
            </div>
            
            <div class="publication-links">
                <span class="link-block">
                    <a href="https://arxiv.org/abs/2601.06874" target="_blank">
                        <span class="icon" style="color: white;">📄</span>
                        <span style="color: white;">Paper</span>
                    </a>
                </span>
                <span class="link-block">
                    <a href="https://github.com/sosppxo/mvggt" target="_blank">
                        <span class="icon" style="color: white;">💻</span>
                        <span style="color: white;">Code</span>
                    </a>
                </span>
                <span class="link-block">
                    <a href="https://mvggt.github.io/" target="_blank">
                        <span class="icon" style="color: white;">🌐</span>
                        <span style="color: white;">Project</span>
                    </a>
                </span>
            </div>
        </div>

        <div class="instructions">
            <h3>How to Use This Demo</h3>
            <ol>
                <li><strong>Upload Media:</strong> Upload a video or a collection of images from multiple viewpoints.</li>
                <li><strong>Text Prompt:</strong> Enter a text description of the object you want to find (e.g., "the wooden table").</li>
                <li><strong>Reconstruct:</strong> Click "Reconstruct" to run the model.</li>
                <li><strong>Interact:</strong> View the 3D result. Toggle "Apply Text Mask" to see the segmentation. Adjust "Confidence Threshold" to filter points.</li>
            </ol>
        </div>
        """
    )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("## 1. Upload Media")
                    input_video = gr.Video(label="Upload Video", interactive=True)
                    input_images = gr.File(file_count="multiple", label="Or Upload Images", interactive=True)
                    text_prompt = gr.Textbox(
                        label="Text Prompt", 
                        placeholder="e.g. 'the red chair' or 'a wooden table'", 
                        interactive=True,
                        info="Describe the object you want to segment in 3D space."
                    )
                    interval = gr.Number(
                        None, 
                        label='Frame/Image Interval', 
                        info="Sampling interval. Video default: 1 FPS. Image default: 1 (all images)."
                    )
                
                image_gallery = gr.Gallery(
                    label="Image Preview",
                    columns=4,
                    height="300px",
                    show_download_button=True,
                    object_fit="contain",
                    preview=True,
                )

            with gr.Column(scale=2):
                gr.Markdown("## 2. View Reconstruction")
                log_output = gr.Markdown("Please upload media and click Reconstruct.", elem_classes=["custom-log"])
                reconstruction_output = gr.Model3D(
                    height=480, 
                    zoom_speed=0.5, 
                    pan_speed=0.5, 
                    label="3D Output",
                    show_label=True
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Reconstruct", scale=3, variant="primary", size="lg")
                    clear_btn = gr.ClearButton(
                        scale=1,
                        size="lg"
                    )
                
                with gr.Group():
                    gr.Markdown("## 3. Adjust Visualization")
                    with gr.Row():
                        conf_thres = gr.Slider(
                            minimum=0, 
                            maximum=100, 
                            value=10, 
                            step=0.1, 
                            label="Confidence Threshold (%)",
                            info="Filter points by confidence score."
                        )
                        show_cam = gr.Checkbox(label="Show Cameras", value=True, info="Display camera poses in 3D view")
                        apply_mask = gr.Checkbox(
                            label="Apply Text Mask", 
                            value=False,
                            info="Highlight segmented object in red"
                        )
                    frame_filter = gr.Dropdown(
                        choices=["All"], 
                        value="All", 
                        label="Show Points from Frame",
                        info="Filter points by source frame"
                    )

        # Set clear button targets
        clear_btn.add([input_video, input_images, reconstruction_output, log_output, target_dir_output, image_gallery, interval, text_prompt, apply_mask])

        submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
            fn=update_log, inputs=[], outputs=[log_output]
        ).then(
            fn=gradio_demo,
            inputs=[
                target_dir_output,
                conf_thres,
                frame_filter,
                show_cam,
                text_prompt,
                apply_mask,
            ],
            outputs=[reconstruction_output, log_output, frame_filter],
        ).then(
            fn=lambda: "False", inputs=[], outputs=[is_example]  # set is_example to "False"
        )

        # -------------------------------------------------------------------------
        # Real-time Visualization Updates
        # -------------------------------------------------------------------------
        conf_thres.change(
            update_visualization,
            [
                target_dir_output,
                conf_thres,
                frame_filter,
                show_cam,
                is_example,
                apply_mask,
            ],
            [reconstruction_output, log_output],
        )
        frame_filter.change(
            update_visualization,
            [
                target_dir_output,
                conf_thres,
                frame_filter,
                show_cam,
                is_example,
                apply_mask,
            ],
            [reconstruction_output, log_output],
        )
    
        show_cam.change(
            update_visualization,
            [
                target_dir_output,
                conf_thres,
                frame_filter,
                show_cam,
                is_example,
                apply_mask,
            ],
            [reconstruction_output, log_output],
        )
        
        apply_mask.change(
            update_visualization,
            [
                target_dir_output,
                conf_thres,
                frame_filter,
                show_cam,
                is_example,
                apply_mask,
            ],
            [reconstruction_output, log_output],
        )

        # -------------------------------------------------------------------------
        # Auto-update gallery whenever user uploads or changes their files
        # -------------------------------------------------------------------------
        input_video.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images, interval],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
        )
        input_images.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images, interval],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
        )

    demo.queue(max_size=20).launch(show_error=True, share=True)
