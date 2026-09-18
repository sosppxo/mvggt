from pathlib import Path

from .config import settings


def scan_asset_library() -> list[dict]:
    asset_root = settings.asset_root
    assets: list[dict] = []
    if not asset_root.exists():
        return assets

    asset_files = sorted(asset_root.rglob("*.glb"))
    for idx, asset_path in enumerate(asset_files):
        rel_path = asset_path.relative_to(asset_root).as_posix()
        assets.append({
            "index": idx,
            "name": rel_path,
            "path": str(asset_path.resolve()),
        })

    return assets


def is_asset_path_allowed(path_str: str) -> bool:
    try:
        asset_root = settings.asset_root.resolve()
        path = Path(path_str).resolve()
        return asset_root in path.parents or path == asset_root
    except (OSError, RuntimeError):
        return False
