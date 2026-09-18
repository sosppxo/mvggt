from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


TaskStatus = Literal["queued", "running", "success", "failed"]
ActionType = Literal["REPLACE", "REMOVE", "SEGMENT"]
BackendMode = Literal["auto", "local", "hf_api"]


class ReconstructTaskCreate(BaseModel):
    raw_prompt: str = Field(..., description="Natural language editing prompt")
    selected_asset_path: Optional[str] = Field(default=None, description="Optional replacement asset path selected in frontend")
    backend_mode: BackendMode = Field(default="auto", description="Inference backend preference")
    interval: int = Field(default=1, ge=1, description="Frame/image sampling interval")


class TaskCreateResponse(BaseModel):
    task_id: str
    status: TaskStatus


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str


class OrientationCandidate(BaseModel):
    id: str
    yaw_deg: float
    offset_deg: float
    variant: str


class TaskResultResponse(BaseModel):
    task_id: str
    predictions_path: Optional[Path]
    glb_path: Optional[Path]
    glb_nomask_path: Optional[Path]
    glb_mask_path: Optional[Path]
    orientation_candidates: list[OrientationCandidate]
    result_keys: list[str]
    action: ActionType
    target_to_segment: str
    replace_with: str
    selected_asset_path: Optional[str]
    has_placement: bool = False
    base_yaw_deg: float = 0.0
    placement_pivot: list[float] = []
    asset_node_names: list[str] = []


class AssetItem(BaseModel):
    index: int
    name: str
    path: str


class ParsedInstruction(BaseModel):
    action: ActionType
    target_to_segment: str
    replace_with: str
