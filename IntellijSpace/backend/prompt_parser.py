import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from .config import settings


def _extract_first_json_object(text: str) -> dict:
    if not text:
        raise ValueError("Empty LLM response")

    text = text.strip()
    code_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    candidate = code_block_match.group(1).strip() if code_block_match else text

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response")

    depth = 0
    for i in range(start, len(candidate)):
        ch = candidate[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                snippet = candidate[start : i + 1]
                return json.loads(snippet)

    raise ValueError("Could not extract a complete JSON object from LLM response")


def parse_user_prompt_with_llm(raw_prompt: str) -> dict:
    if raw_prompt is None or len(raw_prompt.strip()) == 0:
        raise ValueError("Text prompt is required")

    raw_prompt = raw_prompt.strip()

    if not settings.llm_api_url:
        return {
            "action": "SEGMENT",
            "target_to_segment": raw_prompt,
            "replace_with": "",
            "parser_mode": "fallback_no_api_url",
        }

    schema = {
        "action": "REPLACE | REMOVE | SEGMENT",
        "target_to_segment": "string",
        "replace_with": "string",
    }

    system_prompt = (
        "You convert user 3D editing instructions into strict JSON. "
        "Return JSON only with keys: action, target_to_segment, replace_with. "
        "Do not include markdown or explanations."
    )
    user_prompt = (
        "Input sentence:\n"
        f"{raw_prompt}\n\n"
        "Return one JSON object.\n"
        f"Schema hint: {json.dumps(schema)}"
    )

    payload = {
        "model": settings.llm_model_name,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }

    req = urllib.request.Request(
        settings.llm_api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}),
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp_text = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
        raise ValueError(f"LLM parse HTTP error: {exc.code}, body={body[:300]}")
    except urllib.error.URLError as exc:
        raise ValueError(f"LLM parse URL error: {exc}")

    response_json = json.loads(resp_text)

    llm_content = None
    try:
        llm_content = response_json["choices"][0]["message"]["content"]
    except Exception:
        llm_content = resp_text

    parsed = _extract_first_json_object(llm_content)

    action_raw = str(parsed.get("action", "SEGMENT")).strip().upper() or "SEGMENT"
    action = action_raw if action_raw in {"REPLACE", "REMOVE", "SEGMENT"} else "SEGMENT"

    return {
        "action": action,
        "target_to_segment": str(parsed.get("target_to_segment", "")).strip(),
        "replace_with": str(parsed.get("replace_with", "")).strip(),
    }


def resolve_asset_path(selected_asset_path: str | None, replace_with: str) -> str | None:
    if selected_asset_path:
        return selected_asset_path

    if not replace_with:
        return None

    asset_root = settings.asset_root
    if not asset_root.exists():
        return None

    normalized = replace_with.replace("\\", "/").lower()
    for asset in asset_root.rglob("*.glb"):
        rel_name = asset.relative_to(asset_root).as_posix().lower()
        if normalized in rel_name:
            return str(asset.resolve())

    return None
