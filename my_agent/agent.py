import hashlib
import json
import logging
import os
import re
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

import requests
from dotenv import load_dotenv
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

load_dotenv()

if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    for index in range(1, 15):
        numbered_key = os.getenv(f"GEMINI_API_KEY_{index}")
        if numbered_key:
            os.environ["GOOGLE_API_KEY"] = numbered_key
            break

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MEDIA_OUTPUT_DIR = Path(os.getenv("MEDIA_OUTPUT_DIR", ".generated_assets"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache"))
POLLINATIONS_BASE = "https://image.pollinations.ai/prompt"

MEDIA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

USE_CACHE = os.getenv("USE_CACHE", "true").lower() == "true"


class ContentCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def _key_path(self, key: str) -> Path:
        return self.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not USE_CACHE:
            return None
        path = self._key_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        if not USE_CACHE:
            return
        path = self._key_path(key)
        try:
            path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to write cache: %s", exc)


cache = ContentCache(CACHE_DIR)


def _asset_path(prompt: str, extension: str) -> Path:
    digest = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    return MEDIA_OUTPUT_DIR / f"image_{digest}.{extension}"


def _save_bytes(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return str(path)


def _exception_message(exc: BaseException) -> str:
    return str(exc)


def _clean_subject(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" .")
    return cleaned or "your request"


def _title_case(value: str) -> str:
    words = _clean_subject(value).split()
    return " ".join(word.capitalize() if len(word) > 3 else word.lower() for word in words)


def generate_text(prompt: str) -> Dict[str, Any]:
    result = {
        "status": "success",
        "type": "text",
        "content": f"Generated text for prompt: {prompt}",
        "prompt": prompt,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    return result


def generate_report(topic: str) -> Dict[str, Any]:
    topic = _clean_subject(topic)
    title = _title_case(topic)
    report = (
        f"# {title} Report\n\n"
        "## Executive Summary\n"
        f"{topic} has clear creative and strategic potential. The strongest approach is to define the audience, "
        "choose one sharp message, and package the output in the format the user needs.\n\n"
        "## Key Insights\n"
        f"- Audience fit: Shape the tone and visuals around who will consume {topic}.\n"
        "- Message clarity: Lead with one concrete promise instead of many competing ideas.\n"
        "- Format match: Reports should explain decisions, ads should persuade quickly, images should show the idea, "
        "and videos should create motion around one visual story.\n\n"
        "## Recommended Direction\n"
        f"Use {topic} as the central concept, then produce supporting assets from the same theme so the campaign feels "
        "consistent across report, ad, image, and video outputs.\n\n"
        "## Next Steps\n"
        "1. Confirm the target audience.\n"
        "2. Choose the primary call to action.\n"
        "3. Generate the needed creative assets.\n"
    )
    return {
        "status": "success",
        "type": "report",
        "topic": topic,
        "report": report,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def generate_ad(prompt: str, platform: str = "general") -> Dict[str, Any]:
    subject = _clean_subject(prompt)
    platform = _clean_subject(platform).lower()
    headline = f"Make {subject} impossible to ignore"
    copy = (
        f"Bring {subject} to life with a clear benefit, a striking visual, and a direct next step. "
        "Designed for fast attention and easy action."
    )
    hashtags = [
        f"#{re.sub(r'[^A-Za-z0-9]', '', word).title()}"
        for word in subject.split()[:3]
        if re.sub(r"[^A-Za-z0-9]", "", word)
    ]
    image_prompt = (
        f"professional advertising poster for {subject}, bold product-focused composition, "
        f"{platform} campaign, clean readable negative space, premium lighting, no text"
    )
    image = generate_image(image_prompt, style="commercial advertising", width=1024, height=1024)
    return {
        "status": "success",
        "type": "ad",
        "platform": platform,
        "headline": headline,
        "copy": copy,
        "call_to_action": "Learn more",
        "hashtags": hashtags or ["#Campaign"],
        "image": image,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def generate_image(
    prompt: str,
    style: str = "photorealistic",
    width: int = 1024,
    height: int = 1024,
) -> Dict[str, Any]:
    cache_key = f"image:{MEDIA_OUTPUT_DIR.resolve()}:{prompt}:{style}:{width}:{height}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    full_prompt = f"{prompt}, {style} style"
    encoded_prompt = urllib.parse.quote(full_prompt)
    url = (
        f"{POLLINATIONS_BASE}/{encoded_prompt}"
        f"?width={width}&height={height}&nologo=true&model=flux"
    )

    try:
        response = requests.get(url, timeout=120)
        if response.status_code != 200:
            result = {
                "status": "error",
                "message": f"HTTP {response.status_code}: {response.text[:200]}",
                "model": "pollinations/flux",
            }
            cache.set(cache_key, result)
            return result

        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type.lower():
            result = {
                "status": "error",
                "message": f"Unexpected content type: {content_type}",
                "model": "pollinations/flux",
            }
            cache.set(cache_key, result)
            return result

        extension = "png" if "png" in content_type.lower() else "jpg"
        filepath = _asset_path(full_prompt, extension)
        _save_bytes(filepath, response.content)

        result = {
            "status": "success",
            "asset": str(filepath),
            "style": style,
            "model": "pollinations/flux",
            "size_bytes": len(response.content),
        }
        cache.set(cache_key, result)
        return result
    except Exception as exc:
        result = {
            "status": "error",
            "message": _exception_message(exc),
            "model": "pollinations/flux",
        }
        cache.set(cache_key, result)
        return result


def generate_video(prompt: str, width: int = 512, height: int = 512, frames: int = 24) -> Dict[str, Any]:
    try:
        from PIL import Image, ImageDraw, ImageEnhance, ImageFont
        import imageio
    except ModuleNotFoundError as exc:
        return {
            "status": "error",
            "message": f"Missing dependency for video generation: {exc.name}. Run pip install -r requirements.txt.",
            "prompt": prompt,
        }

    prompt = _clean_subject(prompt)
    cache_key = f"video:v2:{MEDIA_OUTPUT_DIR.resolve()}:{prompt}:{width}:{height}:{frames}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    base_prompt = (
        f"cinematic key visual of {prompt}, dynamic motion, rich scene detail, professional lighting, no text"
    )
    base_image = generate_image(base_prompt, style="cinematic", width=max(width, 768), height=max(height, 768))
    source_path = base_image.get("asset") if base_image.get("status") == "success" else None

    font = None
    try:
        font = ImageFont.load_default()
    except Exception:
        pass

    if source_path and Path(source_path).exists():
        base = Image.open(source_path).convert("RGB")
    else:
        base = Image.new("RGB", (width, height), color=(18, 22, 34))
        draw = ImageDraw.Draw(base)
        draw.rectangle((0, height // 2, width, height), fill=(42, 64, 96))
        draw.ellipse((width // 4, height // 4, width * 3 // 4, height * 3 // 4), fill=(130, 180, 220))

    images = []
    safe_frames = max(8, min(int(frames), 96))
    caption = prompt[:70]
    for frame_idx in range(frames):
        progress = frame_idx / max(safe_frames - 1, 1)
        zoom = 1.0 + 0.12 * progress
        crop_w = int(base.width / zoom)
        crop_h = int(base.height / zoom)
        pan_x = int((base.width - crop_w) * progress)
        pan_y = int((base.height - crop_h) * (0.5 - abs(progress - 0.5)))
        frame = base.crop((pan_x, pan_y, pan_x + crop_w, pan_y + crop_h)).resize((width, height))
        frame = ImageEnhance.Color(frame).enhance(1.05 + 0.15 * progress)
        frame = ImageEnhance.Contrast(frame).enhance(1.03)

        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        band_top = int(height * 0.78)
        draw.rectangle((0, band_top, width, height), fill=(0, 0, 0, 135))
        draw.text((18, band_top + 16), caption, fill=(255, 255, 255, 255), font=font)
        draw.rectangle((18, height - 18, 18 + int((width - 36) * progress), height - 12), fill=(255, 255, 255, 220))
        frame = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
        images.append(frame)

    filepath = MEDIA_OUTPUT_DIR / f"video_{hashlib.sha256(prompt.encode()).hexdigest()[:16]}.gif"
    imageio.mimsave(str(filepath), images, fps=8)

    result = {
        "status": "success",
        "asset": str(filepath),
        "prompt": prompt,
        "format": "gif",
        "frames": safe_frames,
        "source_image": source_path,
    }
    cache.set(cache_key, result)
    return result


def run_agent(task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if task_type == "image":
        return generate_image(
            prompt=payload.get("prompt", ""),
            style=payload.get("style", "photorealistic"),
            width=int(payload.get("width", 1024)),
            height=int(payload.get("height", 1024)),
        )
    if task_type == "text":
        return generate_text(payload.get("prompt", ""))
    if task_type == "report":
        return generate_report(payload.get("topic", payload.get("prompt", "")))
    if task_type == "video":
        return generate_video(
            prompt=payload.get("prompt", ""),
            width=int(payload.get("width", 512)),
            height=int(payload.get("height", 512)),
            frames=int(payload.get("frames", 12)),
        )
    return {"status": "error", "message": "Invalid task_type"}


def _content_text(content: Optional[types.Content]) -> str:
    if not content or not content.parts:
        return ""
    chunks = []
    for part in content.parts:
        text = getattr(part, "text", None)
        if text:
            chunks.append(text)
    return " ".join(chunks).strip()


def _prompt_subject(prompt: str) -> str:
    match = re.search(r"\b(?:of|about|for)\s+(.+)$", prompt, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(" .")
    return prompt.strip() or "your request"


def _route_task(prompt: str) -> tuple[str, Dict[str, Any]]:
    lowered = prompt.lower()
    subject = _prompt_subject(prompt)
    if any(word in lowered for word in ("image", "picture", "photo", "poster", "ad creative")):
        return "image", {"prompt": subject}
    if any(word in lowered for word in ("video", "gif", "animation", "animate")):
        return "video", {"prompt": subject}
    if any(word in lowered for word in ("report", "summary", "brief")):
        return "report", {"topic": subject}
    return "text", {"prompt": prompt}


def _format_result(result: Dict[str, Any]) -> str:
    if result.get("status") == "success":
        if result.get("type") == "text":
            return str(result.get("content", "Done."))
        if result.get("type") == "report":
            return str(result.get("report", "Report generated."))
        if result.get("asset"):
            return f"Done. Asset: {result['asset']}"
        return json.dumps(result, indent=2)
    return f"Error: {result.get('message', 'Unknown error')}"


class LocalMediaAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        prompt = _content_text(ctx.user_content)
        task_type, payload = _route_task(prompt)
        result = run_agent(task_type, payload)
        response_text = _format_result(result)
        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=response_text)],
            ),
        )


root_agent = LocalMediaAgent(
    name="my_agent",
    description="Local media agent for text, reports, generated images, and GIF videos.",
)


if __name__ == "__main__":
    import json

    print(json.dumps(run_agent("image", {"prompt": "walking elephant"}), indent=2))
    print(json.dumps(run_agent("video", {"prompt": "walking elephant"}), indent=2))
    print(json.dumps(run_agent("report", {"topic": "walking elephant"}), indent=2))
