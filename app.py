from typing import Dict, Any
import json
import logging
import traceback
from pathlib import Path

from my_agent.agent import run_agent

LOG_DIR = Path(".generated_assets")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "adk_calls.log"

logger = logging.getLogger("adk.entry")
logging.basicConfig(level=logging.INFO)


def _norm_input(inp: Any) -> Dict[str, Any]:
    if isinstance(inp, dict):
        return inp
    if isinstance(inp, str):
        return {"task_type": "text", "payload": {"prompt": inp}}
    return {"task_type": "text", "payload": {}}


def run(input: Dict[str, Any]) -> Dict[str, Any]:
    """ADK entrypoint. Normalizes input, forwards to run_agent, logs results and errors.

    Returns a dict with either the agent result or error information so the ADK UI
    can display what happened.
    """
    try:
        normalized = _norm_input(input)
        task_type = normalized.get("task_type") or normalized.get("type") or "text"
        payload = normalized.get("payload", {})

        entry = {"time": str(Path('.').resolve()), "task_type": task_type, "payload": payload}
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "call", "entry": entry}) + "\n")

        result = run_agent(task_type, payload)

        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "result", "result": result}) + "\n")

        return result

    except Exception as exc:  # catch-all to ensure ADK UI receives diagnostics
        tb = traceback.format_exc()
        err = {"status": "error", "message": str(exc), "traceback": tb}
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "error", "error": err}) + "\n")
        logger.error("ADK run failed: %s", exc)
        return err


def build_graph() -> Dict[str, Any]:
    """Return a minimal app graph for ADK UI to display.

    This is a defensive stub so adk web can request app graphs without
    triggering server-side errors.
    """
    try:
        graph = {
            "nodes": [{"id": "run", "label": "run(input)"}],
            "edges": [],
        }
        return {"status": "success", "graph": graph}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def build_graph_image(dark_mode: bool = False) -> bytes:
    """Return a small PNG bytes image representing the app graph.

    ADK may request this to show a preview; return a tiny generated image.
    """
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGBA", (400, 200), (48, 48, 48) if dark_mode else (240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "My Agent", fill=(255, 255, 255) if dark_mode else (0, 0, 0))
        # export to bytes
        from io import BytesIO

        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:
        # return a tiny transparent PNG fallback
        from base64 import b64decode

        transparent_png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVQYV2NgYAAAAAMA" "ASsJTYQAAAAASUVORK5CYII="
        )
        try:
            return b64decode(transparent_png_b64)
        except Exception:
            return b""


def run_sse(*args, **kwargs):
    """Simple Server-Sent Events (SSE) generator used defensively by ADK.

    Yields a single ready message so the UI doesn't fail when requesting SSE.
    """
    try:
        yield "data: {\"status\": \"ready\"}\n\n"
    except Exception:
        yield "data: {\"status\": \"error\"}\n\n"


if __name__ == "__main__":
    # Quick local tests that simulate ADK calling into the app
    print(run({"task_type": "text", "payload": {"prompt": "hello from adk"}}))
    print(run("a quick prompt string"))
