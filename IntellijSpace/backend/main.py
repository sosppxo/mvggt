import json
from pathlib import Path
from threading import Thread
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .assets import scan_asset_library, is_asset_path_allowed
from .inference import apply_user_transform_to_result_glb, create_task, get_task, run_task, warmup_local_backend
from .prompt_parser import parse_user_prompt_with_llm
from .runtime import detect_runtime_options
from .schemas import AssetItem, ParsedInstruction, ReconstructTaskCreate, TaskCreateResponse, TaskResultResponse, TaskStatusResponse

app = FastAPI(title="MVGGT Backend API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_warmup():
    try:
        runtime = detect_runtime_options()
        if not runtime.get("local_gpu_available", False):
            print("[timing] startup warmup skipped (no local gpu)")
            return
        warmup_local_backend()
    except Exception as exc:
        print(f"[warn] startup warmup failed: {exc}")


@app.get("/api/v1/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/assets", response_model=list[AssetItem])
def list_assets():
    return [AssetItem(**item) for item in scan_asset_library()]


@app.get("/api/v1/runtime/options")
def runtime_options():
    return detect_runtime_options()


@app.post("/api/v1/prompt/parse", response_model=ParsedInstruction)
def parse_prompt(raw_prompt: str = Form(...)):
    parsed = parse_user_prompt_with_llm(raw_prompt)
    return ParsedInstruction(
        action=parsed.get("action", "SEGMENT"),
        target_to_segment=parsed.get("target_to_segment", raw_prompt),
        replace_with=parsed.get("replace_with", ""),
    )


@app.post("/api/v1/tasks/example", response_model=TaskCreateResponse)
def create_example_task(
    interval: int = Form(1),
    raw_prompt: str = Form(...),
    selected_asset_path: Optional[str] = Form(None),
    backend_mode: str = Form("auto"),
    max_images: int = Form(5),
):
    example_dir = settings.example_dir
    if not example_dir.exists():
        raise HTTPException(status_code=400, detail="Example directory not found")

    image_candidates: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_candidates.extend(sorted(example_dir.glob(ext)))

    if not image_candidates:
        raise HTTPException(status_code=400, detail="No example images found")

    picked = image_candidates[: max(1, max_images)]

    req = ReconstructTaskCreate(
        raw_prompt=raw_prompt,
        selected_asset_path=selected_asset_path,
        backend_mode=backend_mode,
        interval=interval,
    )

    if req.selected_asset_path and not is_asset_path_allowed(req.selected_asset_path):
        raise HTTPException(status_code=400, detail="Selected asset path is not allowed.")

    record = create_task(
        image_paths=picked,
        video_path=None,
        interval=req.interval,
        raw_prompt=req.raw_prompt,
        selected_asset_path=req.selected_asset_path,
        backend_mode=req.backend_mode,
    )

    worker = Thread(
        target=run_task,
        args=(record.task_id, req.raw_prompt, req.selected_asset_path, req.backend_mode),
        daemon=True,
    )
    worker.start()

    return TaskCreateResponse(task_id=record.task_id, status=record.status)


@app.post("/api/v1/tasks", response_model=TaskCreateResponse)
async def create_reconstruct_task(
    interval: int = Form(1),
    raw_prompt: str = Form(...),
    selected_asset_path: Optional[str] = Form(None),
    backend_mode: str = Form("auto"),
    images: Optional[list[UploadFile]] = File(default=None),
    video: Optional[UploadFile] = File(default=None),
):
    req = ReconstructTaskCreate(
        raw_prompt=raw_prompt,
        selected_asset_path=selected_asset_path,
        backend_mode=backend_mode,
        interval=interval,
    )

    upload_dir = settings.workspace_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    if images:
        for file in images:
            file_path = upload_dir / file.filename
            file_path.write_bytes(await file.read())
            image_paths.append(file_path)

    video_path = None
    if video is not None:
        video_path = upload_dir / video.filename
        video_path.write_bytes(await video.read())

    if not image_paths and video_path is None:
        raise HTTPException(status_code=400, detail="Please provide at least one image or one video.")

    if req.selected_asset_path and not is_asset_path_allowed(req.selected_asset_path):
        raise HTTPException(status_code=400, detail="Selected asset path is not allowed.")

    record = create_task(
        image_paths=image_paths,
        video_path=video_path,
        interval=req.interval,
        raw_prompt=req.raw_prompt,
        selected_asset_path=req.selected_asset_path,
        backend_mode=req.backend_mode,
    )

    worker = Thread(
        target=run_task,
        args=(record.task_id, req.raw_prompt, req.selected_asset_path, req.backend_mode),
        daemon=True,
    )
    worker.start()

    return TaskCreateResponse(task_id=record.task_id, status=record.status)


@app.get("/api/v1/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    try:
        record = get_task(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskStatusResponse(task_id=task_id, status=record.status, message=record.message)


@app.get("/api/v1/tasks/{task_id}/result", response_model=TaskResultResponse)
def get_task_result(task_id: str):
    try:
        record = get_task(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Task not found")

    if record.status != "success":
        raise HTTPException(status_code=400, detail="Task result is not ready")
    keys = record.result_keys or []
    if not keys and record.predictions_path is not None and record.predictions_path.is_file():
        loaded = np.load(record.predictions_path)
        keys = list(loaded.keys())

    candidates = []
    for c in (record.orientation_candidates or []):
        candidates.append(
            {
                "id": c.get("id"),
                "yaw_deg": c.get("yaw_deg"),
                "offset_deg": c.get("offset_deg"),
                "variant": c.get("variant"),
            }
        )

    has_placement = False
    base_yaw_deg = 0.0
    placement_pivot: list[float] = []
    asset_node_names: list[str] = []
    placement_path = record.target_dir / "placement_info.json"
    if placement_path.is_file():
        pinfo = json.loads(placement_path.read_text(encoding="utf-8"))
        if pinfo.get("status") == "placed":
            has_placement = True
            base_yaw_deg = float(pinfo.get("yaw_deg", 0.0))
            tc = pinfo.get("target_center")
            if isinstance(tc, list) and len(tc) == 3:
                placement_pivot = [float(v) for v in tc]
            ann = pinfo.get("asset_node_names")
            if isinstance(ann, list):
                asset_node_names = [str(n) for n in ann]

    return TaskResultResponse(
        task_id=task_id,
        predictions_path=record.predictions_path,
        glb_path=record.glb_path,
        glb_nomask_path=record.glb_nomask_path,
        glb_mask_path=record.glb_mask_path,
        orientation_candidates=candidates,
        result_keys=keys,
        action=record.action,
        target_to_segment=record.target_to_segment,
        replace_with=record.replace_with,
        selected_asset_path=record.selected_asset_path,
        has_placement=has_placement,
        base_yaw_deg=base_yaw_deg,
        placement_pivot=placement_pivot,
        asset_node_names=asset_node_names,
    )


@app.get("/api/v1/tasks/{task_id}/dynamic")
def get_dynamic_glb(task_id: str, yaw_deg: float = 0.0, scale: float = 1.0):
    try:
        apply_user_transform_to_result_glb(task_id, yaw_deg, scale)
    except KeyError:
        raise HTTPException(status_code=404, detail="Task not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    record = get_task(task_id)
    if record.glb_path is None or not Path(record.glb_path).is_file():
        raise HTTPException(status_code=400, detail="GLB not available after transform")

    return FileResponse(record.glb_path, media_type="model/gltf-binary", filename="dynamic.glb")


@app.get("/api/v1/tasks/{task_id}/download")
def download_task_glb(task_id: str):
    return download_task_glb_variant(task_id, "result")


@app.get("/api/v1/tasks/{task_id}/download/{variant}")
def download_task_glb_variant(task_id: str, variant: str):
    try:
        record = get_task(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Task not found")

    if record.status != "success":
        raise HTTPException(status_code=400, detail="GLB result is not ready")

    variant = variant.lower()
    path = None
    if variant == "result":
        path = record.glb_path
    elif variant == "nomask":
        path = record.glb_nomask_path
    elif variant == "mask":
        path = record.glb_mask_path
    elif variant.startswith("orientation_"):
        for cand in (record.orientation_candidates or []):
            if cand.get("variant") == variant:
                p = cand.get("path")
                path = Path(p) if p else None
                break

    if path is None:
        raise HTTPException(status_code=400, detail=f"GLB variant '{variant}' is not ready")

    return FileResponse(path, media_type="model/gltf-binary", filename=path.name)
