import gc
import json
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from gradio_client import Client, handle_file
from huggingface_hub import hf_hub_download
from transformers import RobertaTokenizer

from mvggt.models.mvggt_training import MVGGT
from mvggt.utils.basic import load_images_as_tensor
from mvggt.utils.geometry import depth_edge

from .config import settings
from .geometry_ops import (
    compute_target_bbox_from_ref_mask,
    estimate_target_yaw_from_mask,
    get_scene_align_rotation,
    place_asset_into_scene,
    predictions_to_glb,
    _snap_yaw_to_right_angles,
)
from .prompt_parser import parse_user_prompt_with_llm, resolve_asset_path
from .runtime import detect_runtime_options


@dataclass
class TaskRecord:
    task_id: str
    status: str
    message: str
    target_dir: Path
    raw_prompt: str
    selected_asset_path: Optional[str] = None
    backend_mode: str = "auto"
    execution_backend: Optional[str] = None
    action: str = "SEGMENT"
    target_to_segment: str = ""
    replace_with: str = ""
    predictions_path: Optional[Path] = None
    result_keys: Optional[list[str]] = None
    predictions_data: Optional[dict] = None
    glb_path: Optional[Path] = None
    glb_nomask_path: Optional[Path] = None
    glb_mask_path: Optional[Path] = None
    orientation_candidates: Optional[list[dict]] = None


TASKS: dict[str, TaskRecord] = {}
MODEL = None
TOKENIZER = None
SAVE_PREDICTIONS_TO_DISK = False


def ensure_model():
    global MODEL, TOKENIZER
    if MODEL is not None and TOKENIZER is not None:
        return MODEL, TOKENIZER

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if settings.local_model_path.exists():
        ckpt_path = settings.local_model_path
    else:
        ckpt_path = hf_hub_download(
            repo_id=settings.model_repo,
            filename=settings.model_file,
            cache_dir=str(settings.model_cache_dir),
        )

    model = MVGGT(
        use_referring_segmentation=True,
        load_vggt=False,
        train_conf=True,
        ckpt=ckpt_path,
        use_pretrained_weights=False,
    )

    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)

    tokenizer_path = settings.tokenizer_path if settings.tokenizer_path.exists() else "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(str(tokenizer_path))

    MODEL = model
    TOKENIZER = tokenizer
    return MODEL, TOKENIZER


def warmup_local_backend():
    """Load model/tokenizer and run one tiny dummy forward to warm CUDA kernels."""
    t0 = time.perf_counter()
    if not torch.cuda.is_available():
        print(f"[timing] warmup skipped (cuda unavailable) total={time.perf_counter() - t0:.3f}s")
        return
    try:
        model, tokenizer = ensure_model()
        device = "cuda"
        dummy_imgs = torch.zeros((1, 1, 3, 224, 224), device=device, dtype=torch.float32)
        text_inputs = tokenizer("warmup", return_tensors="pt")
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _ = model(dummy_imgs, input_ids=input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize()
        print(f"[timing] warmup ensure_model+dummy_forward total={time.perf_counter() - t0:.3f}s")
    except Exception as exc:
        print(f"[warn] warmup dummy-forward failed: {exc}")


def _make_target_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = settings.workspace_dir / f"input_{timestamp}_{uuid.uuid4().hex[:6]}"
    images_dir = target_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _copy_upload_to_dir(src_path: Path, dst_dir: Path) -> Path:
    dst_path = dst_dir / src_path.name
    shutil.copy(str(src_path), str(dst_path))
    return dst_path


def _extract_frames(video_path: Path, dst_dir: Path, interval: int) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    frame_interval = interval if interval > 0 else int(fps)
    count = 0
    frame_id = 0
    saved = []
    while True:
        got, frame = cap.read()
        if not got:
            break
        count += 1
        if count % frame_interval == 0:
            frame_path = dst_dir / f"{frame_id:06}.png"
            cv2.imwrite(str(frame_path), frame)
            saved.append(frame_path)
            frame_id += 1
    return saved


def prepare_inputs(image_paths: list[Path], video_path: Optional[Path], interval: int) -> Path:
    gc.collect()
    torch.cuda.empty_cache()

    target_dir = _make_target_dir()
    images_dir = target_dir / "images"

    saved_paths: list[Path] = []
    if image_paths:
        for image_path in image_paths:
            saved_paths.append(_copy_upload_to_dir(image_path, images_dir))

    if video_path is not None:
        saved_paths.extend(_extract_frames(video_path, images_dir, interval))

    if not saved_paths:
        raise ValueError("No valid images or video frames were provided.")

    return target_dir


def run_inference(target_dir: Path, text_prompt: Optional[str]) -> tuple[Optional[Path], dict]:
    model, tokenizer = ensure_model()
    t_start = time.perf_counter()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_root = target_dir / "images"
    image_names = sorted(str(path) for path in image_root.glob("*"))
    if not image_names:
        raise ValueError("No images found in target directory.")

    t_load_images = time.perf_counter()
    imgs = load_images_as_tensor(str(image_root), interval=1).to(device)
    t_load_images = time.perf_counter() - t_load_images

    input_ids = None
    attention_mask = None
    if text_prompt:
        text_inputs = tokenizer(text_prompt, return_tensors="pt")
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

    t_forward = time.perf_counter()
    dtype = torch.bfloat16
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            predictions = model(imgs[None], input_ids=input_ids, attention_mask=attention_mask)
    t_forward = time.perf_counter() - t_forward

    t_post = time.perf_counter()
    predictions["images"] = imgs[None].permute(0, 1, 3, 4, 2)
    predictions["conf"] = torch.sigmoid(predictions["conf"])
    edge = depth_edge(predictions["local_points"][..., 2], rtol=0.03)
    predictions["conf"][edge] = 0.0
    del predictions["local_points"]

    if "layer_referring_mask_preds" in predictions:
        del predictions["layer_referring_mask_preds"]

    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    t_post = time.perf_counter() - t_post

    predictions_path: Optional[Path] = None
    t_save = 0.0
    if SAVE_PREDICTIONS_TO_DISK:
        t_save = time.perf_counter()
        predictions_path = target_dir / "predictions.npz"
        np.savez(predictions_path, **predictions)
        t_save = time.perf_counter() - t_save

    t_total = time.perf_counter() - t_start
    print(
        f"[timing] local_inference load_images={t_load_images:.3f}s "
        f"forward={t_forward:.3f}s postprocess={t_post:.3f}s "
        f"save_npz={t_save:.3f}s save_enabled={int(SAVE_PREDICTIONS_TO_DISK)} total={t_total:.3f}s"
    )

    torch.cuda.empty_cache()
    return predictions_path, predictions


def run_hf_inference_via_space(target_dir: Path, text_prompt: Optional[str]) -> Path:
    image_root = target_dir / "images"
    image_paths = sorted(str(path) for path in image_root.glob("*"))
    if not image_paths:
        raise ValueError("No images found in target directory for HF Space inference.")


    client = Client(settings.hf_space_id, hf_token=settings.hf_token or None)
    result = client.predict(
        input_images=[handle_file(p) for p in image_paths],
        text_prompt=text_prompt,
        interval=1,
        conf_thres=0,
        show_cam=False,
        apply_mask=False,
        api_name=settings.hf_space_api_name,
    )

    if not isinstance(result, (list, tuple)) or len(result) < 1:
        raise ValueError("HF Space returned invalid result format.")

    remote_predictions_path = Path(result[0])
    if not remote_predictions_path.exists():
        raise ValueError("HF Space predictions file was not downloaded locally.")

    predictions_path = target_dir / "predictions.npz"
    shutil.copy(str(remote_predictions_path), str(predictions_path))

    # Validate returned predictions before downstream GLB export.
    try:
        loaded = np.load(predictions_path, allow_pickle=True)
    except Exception as exc:
        raise ValueError(f"HF predictions.npz cannot be loaded: {exc}")

    required_keys = ["points", "conf", "images", "camera_poses"]
    missing = [k for k in required_keys if k not in loaded]
    if missing:
        raise ValueError(f"HF predictions missing required keys: {missing}")

    points = np.asarray(loaded["points"])
    conf = np.asarray(loaded["conf"])
    cams = np.asarray(loaded["camera_poses"])

    if points.size == 0 or conf.size == 0 or cams.size == 0:
        raise ValueError("HF predictions contain empty points/conf/camera_poses arrays")

    if points.shape[-1] != 3:
        raise ValueError(f"HF predictions points last dim must be 3, got {points.shape}")

    if cams.shape[-2:] != (4, 4):
        raise ValueError(f"HF predictions camera_poses must end with (4,4), got {cams.shape}")

    return predictions_path


def create_task(
    image_paths: list[Path],
    video_path: Optional[Path],
    interval: int,
    raw_prompt: str,
    selected_asset_path: Optional[str],
    backend_mode: str,
) -> TaskRecord:
    target_dir = prepare_inputs(image_paths, video_path, interval)
    task_id = uuid.uuid4().hex
    record = TaskRecord(
        task_id=task_id,
        status="queued",
        message="Queued",
        target_dir=target_dir,
        raw_prompt=raw_prompt,
        selected_asset_path=selected_asset_path,
        backend_mode=backend_mode,
    )
    if record.orientation_candidates is None:
        record.orientation_candidates = []
    TASKS[task_id] = record
    return record


def run_task(
    task_id: str,
    raw_prompt: str,
    selected_asset_path: Optional[str],
    backend_mode: str,
) -> TaskRecord:
    record = get_task(task_id)
    try:
        t_task_start = time.perf_counter()
        record.status = "running"
        record.message = "Parsing prompt"
        t_parse = time.perf_counter()
        parsed = parse_user_prompt_with_llm(raw_prompt)
        t_parse = time.perf_counter() - t_parse
        record.action = parsed.get("action", "SEGMENT")
        record.target_to_segment = parsed.get("target_to_segment", "") or raw_prompt
        record.replace_with = parsed.get("replace_with", "")

        resolved_asset = resolve_asset_path(selected_asset_path, record.replace_with)
        record.selected_asset_path = resolved_asset

        if record.action == "REPLACE" and not resolved_asset:
            raise ValueError("REPLACE action requires selecting a replacement asset.")

        t_runtime = time.perf_counter()
        runtime = detect_runtime_options()
        t_runtime = time.perf_counter() - t_runtime
        chosen_backend = backend_mode
        if backend_mode == "auto":
            chosen_backend = runtime.get("recommended_backend", "hf_api")

        if chosen_backend == "local" and not runtime.get("local_gpu_available", False):
            raise ValueError("backend_mode=local but no local GPU is available.")

        record.execution_backend = chosen_backend
        record.message = f"Running inference ({chosen_backend})"

        t_infer = time.perf_counter()
        if chosen_backend == "hf_api":
            predictions_path = run_hf_inference_via_space(record.target_dir, record.target_to_segment)
            t_load_predictions = time.perf_counter()
            loaded = np.load(predictions_path, allow_pickle=True)
            key_list = ["images", "points", "conf", "camera_poses"]
            if "referring_mask_pred" in loaded:
                key_list.append("referring_mask_pred")
            predictions = {key: np.array(loaded[key]) for key in key_list}
            t_load_predictions = time.perf_counter() - t_load_predictions
        else:
            predictions_path, predictions = run_inference(record.target_dir, record.target_to_segment)
            t_load_predictions = 0.0
        t_infer = time.perf_counter() - t_infer

        record.predictions_path = predictions_path
        record.predictions_data = predictions
        record.result_keys = sorted(predictions.keys())

        t_glb_result = time.perf_counter()
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=0,
            filter_by_frames="All",
            show_cam=False,
            apply_mask=True,
            action=record.action,
        )
        t_glb_result = time.perf_counter() - t_glb_result

        t_glb_nomask = time.perf_counter()
        recon_nomask_scene = predictions_to_glb(
            predictions,
            conf_thres=0,
            filter_by_frames="All",
            show_cam=False,
            apply_mask=False,
            action="SEGMENT",
        )
        t_glb_nomask = time.perf_counter() - t_glb_nomask

        t_glb_mask = time.perf_counter()
        recon_mask_scene = predictions_to_glb(
            predictions,
            conf_thres=0.0,
            filter_by_frames="All",
            show_cam=False,
            apply_mask=True,
            action="SEGMENT",
        )
        t_glb_mask = time.perf_counter() - t_glb_mask

        t_place = time.perf_counter()
        placement_info = {"status": "skipped", "reason": "not_replace_or_missing_bbox"}
        if record.action == "REPLACE" and record.selected_asset_path:
            align_transform = get_scene_align_rotation()
            target_bbox = compute_target_bbox_from_ref_mask(predictions, align_transform=align_transform)
            yaw_info = estimate_target_yaw_from_mask(predictions, align_transform=align_transform)
            yaw_rad = None
            yaw_mode = "fallback"
            if yaw_info.get("valid"):
                yaw_rad = yaw_info.get("raw_yaw_rad")
                yaw_mode = "pca"
            elif "raw_yaw_rad" in yaw_info:
                yaw_rad = yaw_info.get("raw_yaw_rad")
                yaw_mode = "pca_unstable"

            if yaw_rad is not None:
                snapped_yaw, _ = _snap_yaw_to_right_angles(float(yaw_rad))
                yaw_rad = snapped_yaw
                yaw_mode = f"{yaw_mode}+snap90"

            if target_bbox is not None:
                glbscene, placement_info = place_asset_into_scene(
                    scene_3d=glbscene,
                    selected_asset_path=record.selected_asset_path,
                    target_bbox=target_bbox,
                    vertical_axis=1,
                    yaw_rad=yaw_rad,
                )

            if placement_info.get("status") == "placed":
                placement_info["baseline_yaw_rad"] = float(placement_info["yaw_rad"])
                placement_info["baseline_fit_scale"] = float(placement_info["scale"])

            placement_info.update(
                {
                    "yaw_mode": yaw_mode,
                    "yaw_raw_rad": yaw_info.get("raw_yaw_rad"),
                    "yaw_raw_deg": float(np.degrees(yaw_info.get("raw_yaw_rad", 0.0))) if "raw_yaw_rad" in yaw_info else None,
                    "yaw_anisotropy_ratio": yaw_info.get("anisotropy_ratio"),
                    "yaw_valid": yaw_info.get("valid"),
                    "yaw_reason": yaw_info.get("reason"),
                }
            )
        t_place = time.perf_counter() - t_place

        t_export = time.perf_counter()
        placement_info_path = record.target_dir / "placement_info.json"
        with placement_info_path.open("w", encoding="utf-8") as f:
            import json
            json.dump(placement_info, f, ensure_ascii=False, indent=2)

        glb_path = record.target_dir / "result.glb"
        glb_nomask_path = record.target_dir / "recon_nomask.glb"
        glb_mask_path = record.target_dir / "recon_mask.glb"
        export_jobs = [
            (glbscene, glb_path),
            (recon_nomask_scene, glb_nomask_path),
            (recon_mask_scene, glb_mask_path),
        ]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(scene.export, file_obj=str(path)) for scene, path in export_jobs]
            for fut in futures:
                fut.result()

        record.glb_path = glb_path
        record.glb_nomask_path = glb_nomask_path
        record.glb_mask_path = glb_mask_path
        t_export = time.perf_counter() - t_export

        t_task_total = time.perf_counter() - t_task_start
        print(
            f"[timing] run_task task={task_id} backend={chosen_backend} "
            f"parse={t_parse:.3f}s runtime={t_runtime:.3f}s infer={t_infer:.3f}s "
            f"load_preds={t_load_predictions:.3f}s glb_result={t_glb_result:.3f}s "
            f"glb_nomask={t_glb_nomask:.3f}s glb_mask={t_glb_mask:.3f}s "
            f"place={t_place:.3f}s export={t_export:.3f}s total={t_task_total:.3f}s"
        )


        record.status = "success"
        record.message = "Completed"
    except Exception as exc:
        record.status = "failed"
        record.message = str(exc)
    return record


def get_task(task_id: str) -> TaskRecord:
    if task_id not in TASKS:
        raise KeyError(f"Task {task_id} not found")
    return TASKS[task_id]


def apply_user_transform_to_result_glb(task_id: str, yaw_deg: float, scale_mult: float) -> dict:
    record = get_task(task_id)
    if record.status != "success" or record.glb_path is None:
        raise ValueError("task result is not ready")
    if record.action != "REPLACE":
        raise ValueError("transform update only applies to REPLACE tasks")
    placement_path = record.target_dir / "placement_info.json"
    if not placement_path.is_file():
        raise ValueError("placement_info.json missing")
    with placement_path.open(encoding="utf-8") as f:
        placement_info = json.load(f)
    if placement_info.get("status") != "placed":
        raise ValueError("no placed asset to adjust")

    if record.predictions_data is not None:
        predictions = record.predictions_data
    elif record.predictions_path is not None:
        loaded = np.load(record.predictions_path, allow_pickle=True)
        key_list = ["images", "points", "conf", "camera_poses"]
        if "referring_mask_pred" in loaded:
            key_list.append("referring_mask_pred")
        predictions = {key: np.array(loaded[key]) for key in key_list}
    else:
        raise ValueError("predictions unavailable for dynamic transform")

    glbscene = predictions_to_glb(
        predictions,
        conf_thres=0,
        filter_by_frames="All",
        show_cam=False,
        apply_mask=True,
        action="REPLACE",
    )

    target_bbox = {
        "center": placement_info["target_center"],
        "size": placement_info["target_size"],
    }
    asset_path = placement_info.get("asset_path") or record.selected_asset_path
    if not asset_path:
        raise ValueError("asset path unknown")
    asset_path = str(asset_path)
    if not os.path.isfile(asset_path) and record.selected_asset_path:
        asset_path = str(record.selected_asset_path)
    if not os.path.isfile(asset_path):
        raise ValueError("asset file not found for re-export")

    baseline_yaw = placement_info.get("baseline_yaw_rad")
    baseline_fit = placement_info.get("baseline_fit_scale")
    if baseline_yaw is None or baseline_fit is None:
        last_yaw_deg = float(placement_info.get("last_user_yaw_deg") or 0.0)
        last_scale_mult = float(placement_info.get("last_user_scale_mult") or 1.0)
        cur_yaw = float(placement_info.get("yaw_rad", 0.0))
        cur_scale = float(placement_info.get("scale", 1.0))
        baseline_yaw = cur_yaw - float(np.radians(last_yaw_deg))
        baseline_fit = cur_scale / last_scale_mult if abs(last_scale_mult) > 1e-9 else cur_scale

    new_yaw = float(baseline_yaw) + float(np.radians(yaw_deg))
    new_scale = float(baseline_fit) * float(scale_mult)

    axis = int(placement_info.get("vertical_axis", 1))
    glbscene, new_placement = place_asset_into_scene(
        scene_3d=glbscene,
        selected_asset_path=asset_path,
        target_bbox=target_bbox,
        vertical_axis=axis,
        yaw_rad=new_yaw,
        fit_scale_override=new_scale,
    )
    if new_placement.get("status") != "placed":
        raise ValueError(new_placement.get("reason", "re-place failed"))

    meta_keys = (
        "yaw_mode",
        "yaw_raw_rad",
        "yaw_raw_deg",
        "yaw_anisotropy_ratio",
        "yaw_valid",
        "yaw_reason",
    )
    for mk in meta_keys:
        if mk in placement_info:
            new_placement[mk] = placement_info[mk]
    new_placement["baseline_yaw_rad"] = float(baseline_yaw)
    new_placement["baseline_fit_scale"] = float(baseline_fit)
    new_placement["last_user_yaw_deg"] = float(yaw_deg)
    new_placement["last_user_scale_mult"] = float(scale_mult)

    with placement_path.open("w", encoding="utf-8") as f:
        json.dump(new_placement, f, ensure_ascii=False, indent=2)

    glbscene.export(file_obj=str(record.glb_path))
    return new_placement
