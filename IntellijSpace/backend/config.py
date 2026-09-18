import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

APP_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(APP_ROOT / ".env", override=False)


def app_path(env_name: str, default: str) -> Path:
    path = Path(os.environ.get(env_name) or default).expanduser()
    return path.resolve() if path.is_absolute() else (APP_ROOT / path).resolve()


class Settings(BaseModel):
    model_repo: str = os.getenv("MVGGT_MODEL_REPO", "sosppxo/mvggt")
    model_file: str = os.getenv("MVGGT_MODEL_FILE", "best_model/pytorch_model.bin")
    model_cache_dir: Path = app_path("MVGGT_MODEL_CACHE_DIR", "models")
    local_model_path: Path = app_path("MVGGT_LOCAL_MODEL_PATH", "models/best_model/pytorch_model.bin")
    tokenizer_path: Path = app_path("MVGGT_TOKENIZER_PATH", "ckpts/roberta-base")
    asset_root: Path = app_path("MVGGT_ASSET_ROOT", "glb")
    example_dir: Path = app_path("MVGGT_EXAMPLE_DIR", "example")
    llm_api_url: str = os.getenv("MVGGT_LLM_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    llm_api_key: str = os.getenv("MVGGT_LLM_API_KEY", "")
    llm_model_name: str = os.getenv("MVGGT_LLM_MODEL_NAME", "qwen-plus")
    hf_space_id: str = os.getenv("MVGGT_HF_SPACE_ID", "sosppxo/mvggt")
    hf_space_api_name: str = os.getenv("MVGGT_HF_SPACE_API_NAME", "/predict_remote")
    hf_token: str = os.getenv("HF_TOKEN", "")
    workspace_dir: Path = app_path("MVGGT_WORKSPACE_DIR", "backend_workspace")
    default_interval: int = 1


settings = Settings()
settings.workspace_dir.mkdir(parents=True, exist_ok=True)