"""
Integrated Content Generation Agent
====================================
Combines:
  - Pollinations-based image generation  (reliable, no Gemini quota)
  - PIL/imageio GIF video generation     (local, no API needed)
  - Full LLM orchestration pipeline      (BriefAgent → Dispatcher → Pipelines)
  - Google Search-powered report writing (real content, not templates)
  - Multi-key Gemini rotation + retry    (handles quota limits gracefully)
"""

import asyncio
import hashlib
import itertools
import json
import logging
import os
import re
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, PrivateAttr

from google.adk.agents import Agent, BaseAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.models import Gemini
from google.adk.tools import FunctionTool
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import google_search
from google import genai
from google.genai import types
from google.genai.errors import ClientError

# =========================================================
# Bootstrap & Logging
# =========================================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# =========================================================
# Paths & Constants
# =========================================================

MEDIA_OUTPUT_DIR = Path(os.getenv("MEDIA_OUTPUT_DIR", "generated_assets"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache"))
POLLINATIONS_BASE = "https://image.pollinations.ai/prompt"

MEDIA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ORCHESTRATION_MODEL = os.getenv("ORCHESTRATION_MODEL", "gemini-2.5-flash")
CONTENT_MODEL = os.getenv("CONTENT_MODEL", "gemini-2.5-flash")

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
INITIAL_RETRY_DELAY = float(os.getenv("INITIAL_RETRY_DELAY", "1"))
MAX_RETRY_DELAY = float(os.getenv("MAX_RETRY_DELAY", "30"))
REQUEST_THROTTLE_DELAY = float(os.getenv("REQUEST_THROTTLE_DELAY", "0.5"))

USE_CACHE = os.getenv("USE_CACHE", "true").lower() == "true"

# =========================================================
# Multi-Key Rotation / Failover
# =========================================================

class GeminiKeyRotator:
    """
    Rotates through numbered GEMINI_API_KEY_1 … GEMINI_API_KEY_N keys.
    Falls back to GOOGLE_API_KEY / GEMINI_API_KEY for single-key setups.
    """

    def __init__(self, env_prefix: str = "GEMINI_API_KEY") -> None:
        numbered_keys = sorted(
            (k, v) for k, v in os.environ.items()
            if k.startswith(f"{env_prefix}_") and v
        )
        keys = [v for _, v in numbered_keys]

        if not keys:
            single = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if single:
                keys = [single]

        if not keys:
            raise RuntimeError(
                "No Gemini API keys found. Set GEMINI_API_KEY_1, GEMINI_API_KEY_2, ... "
                "in .env, or at minimum GOOGLE_API_KEY / GEMINI_API_KEY."
            )

        self.keys: List[str] = keys
        self._cycle = itertools.cycle(self.keys)
        self.current_key: str = next(self._cycle)
        self._dead_until: Dict[str, float] = {}
        logger.info("[KeyRotator] Loaded %d API key(s).", len(self.keys))

    def rotate(self) -> str:
        previous = self.current_key
        for _ in range(len(self.keys)):
            candidate = next(self._cycle)
            if self._dead_until.get(candidate, 0) <= time.time():
                self.current_key = candidate
                if self.current_key != previous:
                    logger.warning(
                        "[KeyRotator] Switched ...%s -> ...%s",
                        previous[-4:], self.current_key[-4:],
                    )
                return self.current_key
        self.current_key = next(self._cycle)
        return self.current_key

    def mark_exhausted(self, key: Optional[str] = None, cooldown_seconds: float = 60.0) -> None:
        key = key or self.current_key
        self._dead_until[key] = time.time() + cooldown_seconds

    def all_exhausted(self) -> bool:
        now = time.time()
        return all(self._dead_until.get(k, 0) > now for k in self.keys)


key_rotator = GeminiKeyRotator()


def _is_rate_limit_error(exc: BaseException) -> bool:
    if isinstance(exc, ClientError) and getattr(exc, "code", None) == 429:
        return True
    msg = str(exc).lower()
    return any(x in msg for x in ["429", "quota", "rate limit", "resource_exhausted"])


# =========================================================
# Cache
# =========================================================

class ContentCache:
    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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
            logger.warning("Cache write failed: %s", exc)


cache = ContentCache()

# =========================================================
# Shared Helpers
# =========================================================

def _asset_path(prefix: str, source: str, extension: str) -> Path:
    digest = hashlib.sha256(source.encode()).hexdigest()[:12]
    return MEDIA_OUTPUT_DIR / f"{prefix}_{digest}.{extension}"


def _save_bytes(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return str(path)


def _exception_message(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(_exception_message(e) for e in exc.exceptions) or str(exc)
    return str(exc)


def _clean_subject(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" .")
    return cleaned or "your request"


def _title_case(value: str) -> str:
    words = _clean_subject(value).split()
    return " ".join(w.capitalize() if len(w) > 3 else w.lower() for w in words)


# =========================================================
# Retry Decorator
# =========================================================

def _retry_with_backoff(func, max_retries: int = MAX_RETRIES):
    def wrapper(*args, **kwargs):
        total_attempts = max_retries * max(len(key_rotator.keys), 1)
        for attempt in range(total_attempts):
            try:
                time.sleep(REQUEST_THROTTLE_DELAY)
                return func(*args, **kwargs)
            except Exception as e:
                if _is_rate_limit_error(e) and attempt < total_attempts - 1:
                    key_rotator.mark_exhausted()
                    key_rotator.rotate()
                    wait = min(INITIAL_RETRY_DELAY * (2 ** (attempt % max_retries)), MAX_RETRY_DELAY)
                    wait += time.time() % 1
                    logger.warning(
                        "Rate limited. Retry %d/%d in %.1fs on key ...%s",
                        attempt + 1, total_attempts, wait, key_rotator.current_key[-4:],
                    )
                    time.sleep(wait)
                    continue
                raise
        return None
    return wrapper


# =========================================================
# IMAGE GENERATION  —  Pollinations (free, reliable, no Gemini quota)
# =========================================================

def generate_image(
    prompt: str,
    style: str = "photorealistic",
    width: int = 1024,
    height: int = 1024,
    aspect_ratio: str = "1:1",  # accepted for API compat, ignored
) -> Dict[str, Any]:
    """
    Generate an image via Pollinations AI (flux model).
    No Gemini quota consumed. Results are disk-cached.
    """
    cache_key = f"pollinations_image:{prompt}:{style}:{width}:{height}"
    cached = cache.get(cache_key)
    if cached:
        logger.info("[Image] Cache HIT for: %.50s", prompt)
        return cached

    full_prompt = f"{prompt}, {style} style"
    encoded = urllib.parse.quote(full_prompt)
    url = (
        f"{POLLINATIONS_BASE}/{encoded}"
        f"?width={width}&height={height}&nologo=true&model=flux"
    )

    try:
        logger.info("[Image] Generating via Pollinations: %.60s", prompt)
        response = requests.get(url, timeout=120)
        if response.status_code != 200:
            result: Dict[str, Any] = {
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
                "message": f"Unexpected content-type: {content_type}",
                "model": "pollinations/flux",
            }
            cache.set(cache_key, result)
            return result

        extension = "png" if "png" in content_type.lower() else "jpg"
        filepath = _asset_path("image", full_prompt, extension)
        _save_bytes(filepath, response.content)

        result = {
            "status": "success",
            "asset": str(filepath),
            "style": style,
            "model": "pollinations/flux",
            "size_bytes": len(response.content),
        }
        cache.set(cache_key, result)
        logger.info("[Image] Saved -> %s", filepath)
        return result

    except Exception as exc:
        result = {
            "status": "error",
            "message": _exception_message(exc),
            "model": "pollinations/flux",
        }
        cache.set(cache_key, result)
        return result


# =========================================================
# VIDEO GENERATION  —  Local PIL/imageio GIF (no external API)
# =========================================================

def generate_video(
    prompt: str = "",
    width: int = 512,
    height: int = 512,
    frames: int = 24,
    # aliases for Code-2 pipeline_c compatibility
    script: str = "",
    format: str = "short",
    duration_seconds: int = 30,
) -> Dict[str, Any]:
    """
    Generate an animated GIF using PIL + imageio.
    Base image is fetched from Pollinations, then animated with
    Ken-Burns zoom/pan + caption overlay.
    """
    try:
        from PIL import Image, ImageDraw, ImageEnhance, ImageFont
        import imageio
    except ModuleNotFoundError as exc:
        return {
            "status": "error",
            "message": f"Missing dependency: {exc.name}. Run: pip install pillow imageio",
            "prompt": prompt,
        }

    effective_prompt = _clean_subject(script or prompt)
    safe_frames = max(8, min(int(frames), 96))

    cache_key = (
        f"gif_video:v2:{MEDIA_OUTPUT_DIR.resolve()}:"
        f"{effective_prompt}:{width}:{height}:{safe_frames}"
    )
    cached = cache.get(cache_key)
    if cached:
        logger.info("[Video] Cache HIT for: %.50s", effective_prompt)
        return cached

    logger.info("[Video] Generating GIF for: %.60s", effective_prompt)

    base_prompt = (
        f"cinematic key visual of {effective_prompt}, "
        "dynamic motion, rich scene detail, professional lighting, no text"
    )
    base_img_result = generate_image(
        base_prompt, style="cinematic",
        width=max(width, 768), height=max(height, 768),
    )
    source_path = (
        base_img_result.get("asset")
        if base_img_result.get("status") == "success"
        else None
    )

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    if source_path and Path(source_path).exists():
        base = Image.open(source_path).convert("RGB")
    else:
        base = Image.new("RGB", (width, height), color=(18, 22, 34))
        draw = ImageDraw.Draw(base)
        draw.rectangle((0, height // 2, width, height), fill=(42, 64, 96))
        draw.ellipse(
            (width // 4, height // 4, width * 3 // 4, height * 3 // 4),
            fill=(130, 180, 220),
        )

    caption = effective_prompt[:70]
    images: List[Any] = []

    for idx in range(safe_frames):
        progress = idx / max(safe_frames - 1, 1)
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
        if font:
            draw.text((18, band_top + 16), caption, fill=(255, 255, 255, 255), font=font)
        draw.rectangle(
            (18, height - 18, 18 + int((width - 36) * progress), height - 12),
            fill=(255, 255, 255, 220),
        )
        frame = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
        images.append(frame)

    filepath = MEDIA_OUTPUT_DIR / f"video_{hashlib.sha256(effective_prompt.encode()).hexdigest()[:16]}.gif"
    imageio.mimsave(str(filepath), images, fps=8)

    result: Dict[str, Any] = {
        "status": "success",
        "asset": str(filepath),
        "file": str(filepath),   # alias for Code-2 compat
        "prompt": effective_prompt,
        "format": "gif",
        "frames": safe_frames,
        "source_image": source_path,
        "model": "local/pillow+imageio",
    }
    cache.set(cache_key, result)
    logger.info("[Video] Saved -> %s", filepath)
    return result


# Alias so Code-2-style calls to produce_video still work
produce_video = generate_video


# =========================================================
# REPORT / TEXT GENERATION
# =========================================================

def generate_report(topic: str) -> Dict[str, Any]:
    """
    Scaffold report — used as a fallback for direct/CLI calls.
    Full LLM-written reports are produced by pipeline_b_agent (Gemini + Search).
    """
    topic = _clean_subject(topic)
    title = _title_case(topic)
    report = (
        f"# {title} Report\n\n"
        "## Executive Summary\n"
        f"{topic} presents clear creative and strategic potential. "
        "Define the audience, choose one sharp message, and package "
        "the output in the format the user needs.\n\n"
        "## Key Insights\n"
        f"- **Audience fit**: Shape tone and visuals around who will consume {topic}.\n"
        "- **Message clarity**: Lead with one concrete promise, not many competing ideas.\n"
        "- **Format match**: Reports explain decisions; ads persuade quickly; "
        "images show the idea; GIFs create motion around one visual story.\n\n"
        "## Recommended Direction\n"
        f"Use **{topic}** as the central concept, then produce supporting assets "
        "from the same theme for a consistent multi-channel campaign.\n\n"
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


def generate_text(prompt: str) -> Dict[str, Any]:
    return {
        "status": "success",
        "type": "text",
        "content": f"Generated text for: {prompt}",
        "prompt": prompt,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def write_content(content_type: str, brief: str, tone: str) -> Dict[str, Any]:
    """Fast local content scaffold (no API call)."""
    logger.info("[ContentWriter] %s in %s tone", content_type, tone)
    draft = (
        f"[{content_type.upper()}]\n"
        f"Title: {brief}\n"
        f"Tone: {tone}\n\n"
        f"This is a professional {content_type} ready for refinement."
    )
    return {
        "status": "success",
        "content_type": content_type,
        "draft": draft,
        "word_count": len(draft.split()),
    }


def create_social_media_post(platform: str, copy: str, hashtags: list) -> Dict[str, Any]:
    logger.info("[SocialPost] %s - %d chars", platform, len(copy))
    return {
        "status": "success",
        "platform": platform,
        "copy": copy,
        "hashtags": hashtags,
        "post_url": f"draft://{platform}/{hashlib.md5(copy.encode()).hexdigest()[:8]}",
    }


def generate_ad(prompt: str, platform: str = "general") -> Dict[str, Any]:
    subject = _clean_subject(prompt)
    platform = _clean_subject(platform).lower()
    headline = f"Make {subject} impossible to ignore"
    copy = (
        f"Bring {subject} to life with a clear benefit, a striking visual, "
        "and a direct next step. Designed for fast attention and easy action."
    )
    hashtags = [
        f"#{re.sub(r'[^A-Za-z0-9]', '', w).title()}"
        for w in subject.split()[:3]
        if re.sub(r"[^A-Za-z0-9]", "", w)
    ]
    image = generate_image(
        f"professional advertising poster for {subject}, bold composition, "
        f"{platform} campaign, clean negative space, premium lighting, no text",
        style="commercial advertising", width=1024, height=1024,
    )
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


# =========================================================
# FunctionTools (ADK wrappers)
# =========================================================

image_tool   = FunctionTool(func=generate_image)
social_tool  = FunctionTool(func=create_social_media_post)
writing_tool = FunctionTool(func=write_content)
video_tool   = FunctionTool(func=generate_video)

# =========================================================
# RotatingGemini  —  key-rotation on quota hit
# =========================================================

class RotatingGemini(Gemini):
    """Gemini model wrapper that transparently rotates API keys on quota errors."""

    _client_cache: Dict[str, genai.Client] = PrivateAttr(default_factory=dict)

    @property
    def api_client(self) -> genai.Client:
        active_key = key_rotator.current_key
        if active_key in self._client_cache:
            return self._client_cache[active_key]

        base_url, api_version = self._base_url_and_api_version
        http_options: Dict[str, Any] = {
            "headers": self._tracking_headers(),
            "retry_options": self.retry_options,
            "base_url": base_url,
        }
        if api_version:
            http_options["api_version"] = api_version

        kwargs: Dict[str, Any] = {
            "api_key": active_key,
            "http_options": types.HttpOptions(**http_options),
        }
        if self.model.startswith("projects/"):
            kwargs["enterprise"] = True

        client = genai.Client(**kwargs)
        self._client_cache[active_key] = client
        return client

    async def generate_content_async(self, *args, **kwargs):
        total_attempts = MAX_RETRIES * max(len(key_rotator.keys), 1)
        last_exc: Optional[Exception] = None
        for attempt in range(total_attempts):
            try:
                async for response in super().generate_content_async(*args, **kwargs):
                    yield response
                return
            except Exception as exc:
                last_exc = exc
                if _is_rate_limit_error(exc) and attempt < total_attempts - 1:
                    key_rotator.mark_exhausted()
                    key_rotator.rotate()
                    logger.warning(
                        "[RotatingGemini] Quota hit -> switched to ...%s (attempt %d)",
                        key_rotator.current_key[-4:], attempt + 1,
                    )
                    await asyncio.sleep(
                        min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                    )
                    continue
                raise
        if last_exc:
            raise last_exc


LLM = RotatingGemini(
    model=ORCHESTRATION_MODEL,
    retry_options=types.HttpRetryOptions(
        attempts=MAX_RETRIES,
        initial_delay=INITIAL_RETRY_DELAY,
        max_delay=MAX_RETRY_DELAY,
        exp_base=2,
        jitter=1,
        http_status_codes=[408, 429, 500, 502, 503, 504],
    ),
)

LLM_CONTENT = RotatingGemini(
    model=CONTENT_MODEL,
    retry_options=types.HttpRetryOptions(
        attempts=MAX_RETRIES,
        initial_delay=INITIAL_RETRY_DELAY,
        max_delay=MAX_RETRY_DELAY,
        exp_base=2,
        jitter=1,
        http_status_codes=[408, 429, 500, 502, 503, 504],
    ),
)

# =========================================================
# Search Agent
# =========================================================

search_agent = Agent(
    name="SearchAgent",
    model=LLM,
    description="Search for recent information efficiently.",
    instruction="""
You are a Research Assistant.
1. Use the search tool ONLY for current information (news, recent data, trends).
2. Return ONLY the 3-5 most relevant facts with sources.
3. Keep your response under 150 words.
4. Do NOT search for historical facts or general knowledge you already know.
""",
    tools=[google_search],
)

search_agent_tool = AgentTool(agent=search_agent)

# =========================================================
# Pydantic Schemas
# =========================================================

class ContentBrief(BaseModel):
    topic: str = Field(..., description="Main topic")
    audience: str = Field(default="general", description="Target audience")
    tone: str = Field(default="professional", description="Writing tone")
    brand: str = Field(default="", description="Brand name (optional)")
    platform: str = Field(
        default="",
        description="Target platform: LinkedIn / Instagram / Twitter / Facebook",
    )
    deliverables: List[str] = Field(
        default_factory=list,
        description="List from: social_post, ad_images, report, story, short_video",
    )


class DispatchDecision(BaseModel):
    pipelines: List[str] = Field(
        description="Selected pipelines: pipeline_a, pipeline_b, pipeline_c"
    )
    reasoning: str = Field(default="", description="Brief explanation")


# =========================================================
# Pipeline Skip Callbacks
# =========================================================

def make_skip_callback(pipeline_name: str):
    def _skip_if_not_selected(
        callback_context: CallbackContext, **_: Any
    ) -> Optional[types.Content]:
        selected = _selected_pipelines(callback_context)
        if pipeline_name not in selected:
            logger.info("[Dispatcher] Skipping %s", pipeline_name)
            return types.Content(
                role="model",
                parts=[types.Part(text=f"[SKIPPED: {pipeline_name} not selected]")],
            )
        return None
    return _skip_if_not_selected


def _selected_pipelines(ctx: CallbackContext) -> List[str]:
    dispatch = ctx.state.get("dispatch") or {}
    if isinstance(dispatch, str):
        try:
            parsed = json.loads(dispatch)
            if isinstance(parsed, dict):
                return parsed.get("pipelines", [])
        except (json.JSONDecodeError, TypeError):
            pass
        return [dispatch]
    if isinstance(dispatch, dict):
        return dispatch.get("pipelines", [])
    return getattr(dispatch, "pipelines", [])


# =========================================================
# Brief Agent
# =========================================================

brief_agent = Agent(
    name="BriefAgent",
    model=LLM,
    description="Extract a structured content brief from the user's request.",
    instruction="""
Extract a content brief from the user's request.

Rules:
- topic: main subject (required)
- audience: who this is for (default: "general")
- tone: style (default: "professional")
- platform: where it will be posted (LinkedIn/Instagram/Twitter/Facebook, or "")
- deliverables: list ONLY what the user explicitly asked for.
  Valid values: social_post, ad_images, report, story, short_video

Be strict — only include deliverables that are clearly requested.
If the intent is ambiguous, ask for clarification.
""",
    output_schema=ContentBrief,
    output_key="brief",
)

# =========================================================
# Dispatcher Agent
# =========================================================

dispatcher_agent = Agent(
    name="DispatcherAgent",
    model=LLM,
    description="Route the content brief to the appropriate pipeline(s).",
    instruction="""
Given the content brief in {brief}, decide which pipeline(s) to run.

Mapping rules:
- social_post OR ad_images  ->  pipeline_a
- report OR story           ->  pipeline_b
- short_video               ->  pipeline_c

Include ALL pipelines that match the requested deliverables. No duplicates.
""",
    output_schema=DispatchDecision,
    output_key="dispatch",
)

# =========================================================
# Pipeline A  —  Social Media & Ad Images
# =========================================================

pipeline_a_agent = Agent(
    name="PipelineA_SocialAds",
    model=LLM_CONTENT,
    description="Generate ad images and social media posts.",
    instruction="""
You have this content brief: {brief}

1. Build an image prompt from brief.topic, brief.tone, and brief.platform.
   Dimensions:
   - Instagram / TikTok  ->  width=1080, height=1920
   - LinkedIn / Twitter  ->  width=1200, height=628
   - Default             ->  width=1024, height=1024
   Call the generate_image tool.

2. Write social post copy (max 280 chars for Twitter, longer for LinkedIn).
   Add 3-5 relevant hashtags. Call the create_social_media_post tool.

Output:
## Image
[file path and one-line description]

## Social Post
Platform: [name]
Copy: [text]
Hashtags: [list]
""",
    tools=[image_tool, social_tool],
    before_agent_callback=make_skip_callback("pipeline_a"),
    output_key="pipeline_a_result",
)

# =========================================================
# Pipeline B  —  Long-Form Content (Reports & Stories)
# =========================================================

pipeline_b_agent = Agent(
    name="PipelineB_LongForm",
    model=LLM_CONTENT,
    description="Generate rich, research-backed reports and stories.",
    instruction="""
You have this content brief: {brief}

Steps:
1. RESEARCH (if the topic needs current facts):
   Call search_agent_tool first. Wait for results before writing.
   Use only verified facts from the search.

2. WRITE the main content:
   - Type: report or story (from brief.deliverables)
   - Tone: exactly match brief.tone
   - Audience: write for brief.audience
   - Length: 300-500 words (detailed but scannable)

   REPORT structure:
     # Title
     ## Executive Summary   (2-3 sentences)
     ## Key Findings        (bullet points with data)
     ## Analysis            (2-3 paragraphs)
     ## Recommendations     (numbered list)
     ## Conclusion          (1 paragraph)

   STORY structure:
     # Title
     Opening hook (1 paragraph)
     Rising action / body (3-4 paragraphs)
     Conclusion / resolution (1 paragraph)

3. OPTIONAL extras (only if in brief.deliverables):
   - ad_images:   call generate_image with a scene from the report.
   - social_post: call create_social_media_post with a summary + hashtags.

Output:
## Report / Story
[full content here]

## Supporting Image (if generated)
[file path]

## Social Post (if generated)
[copy + hashtags]
""",
    tools=[search_agent_tool, image_tool, social_tool, writing_tool],
    before_agent_callback=make_skip_callback("pipeline_b"),
    output_key="pipeline_b_result",
)

# =========================================================
# Pipeline C  —  Short Video (Animated GIF)
# =========================================================

pipeline_c_agent = Agent(
    name="PipelineC_Video",
    model=LLM_CONTENT,
    description="Generate short-form video scripts and animated GIFs.",
    instruction="""
You have this content brief: {brief}

1. RESEARCH (if current facts are needed):
   Call search_agent_tool first.

2. WRITE a video script:
   Duration: 30-45 seconds of spoken content.
   Structure:
     Hook    (3 s)  — one striking opening line
     Content (25 s) — 3-4 punchy points
     CTA     (5 s)  — clear call to action
   Tone: match brief.tone. Language: conversational, short sentences.

3. GENERATE the video:
   Call generate_video tool. Pass the full script as `prompt`.
   Use width=512, height=512, frames=24.
   Output is an animated GIF.

Output:
## Script
[30-45 s video script]

## Video
[generated file path]
""",
    tools=[search_agent_tool, writing_tool, video_tool],
    before_agent_callback=make_skip_callback("pipeline_c"),
    output_key="pipeline_c_result",
)

# =========================================================
# Root Orchestrator
# =========================================================

root_agent = SequentialAgent(
    name="ContentOrchestrator",
    description=(
        "End-to-end content generation: "
        "BriefAgent -> DispatcherAgent -> PipelineA (images/social) | "
        "PipelineB (reports/stories via Gemini+Search) | "
        "PipelineC (GIF video via Pollinations+PIL)."
    ),
    sub_agents=[
        brief_agent,
        dispatcher_agent,
        SequentialAgent(
            name="PipelineRunner",
            description="Execute the selected content pipelines in order.",
            sub_agents=[
                pipeline_a_agent,
                pipeline_b_agent,
                pipeline_c_agent,
            ],
        ),
    ],
)

# =========================================================
# Simple Route Helper  (for direct / CLI calls)
# =========================================================

def _route_task(prompt: str) -> tuple:
    lowered = prompt.lower()
    match = re.search(r"\b(?:of|about|for)\s+(.+)$", prompt, re.IGNORECASE)
    subject = match.group(1).strip(" .") if match else prompt.strip()

    if any(w in lowered for w in ("image", "picture", "photo", "poster", "ad creative")):
        return "image", {"prompt": subject}
    if any(w in lowered for w in ("video", "gif", "animation", "animate")):
        return "video", {"prompt": subject}
    if any(w in lowered for w in ("report", "summary", "brief", "story")):
        return "report", {"topic": subject}
    return "text", {"prompt": prompt}


def run_agent(task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Direct dispatcher — bypasses LLM orchestration for quick single-task use."""
    if task_type == "image":
        return generate_image(
            prompt=payload.get("prompt", ""),
            style=payload.get("style", "photorealistic"),
            width=int(payload.get("width", 1024)),
            height=int(payload.get("height", 1024)),
        )
    if task_type == "video":
        return generate_video(
            prompt=payload.get("prompt", ""),
            width=int(payload.get("width", 512)),
            height=int(payload.get("height", 512)),
            frames=int(payload.get("frames", 24)),
        )
    if task_type == "report":
        return generate_report(payload.get("topic", payload.get("prompt", "")))
    if task_type == "text":
        return generate_text(payload.get("prompt", ""))
    return {"status": "error", "message": f"Unknown task_type: {task_type!r}"}


# =========================================================
# LocalMediaAgent  (BaseAgent shim — for ADK web runner)
# =========================================================

def _content_text(content: Optional[types.Content]) -> str:
    if not content or not content.parts:
        return ""
    return " ".join(getattr(p, "text", None) or "" for p in content.parts).strip()


class LocalMediaAgent(BaseAgent):
    """
    Thin BaseAgent shim for quick single-task requests via ADK web runner.
    For multi-deliverable requests, use orchestrate() / root_agent directly.
    """

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        prompt = _content_text(ctx.user_content)
        task_type, payload = _route_task(prompt)
        result = run_agent(task_type, payload)

        if result.get("type") == "report":
            response_text = result.get("report", "Report generated.")
        elif result.get("type") == "text":
            response_text = result.get("content", "Done.")
        elif result.get("asset") or result.get("file"):
            path = result.get("asset") or result.get("file")
            response_text = f"Done. Asset saved -> {path}"
        else:
            response_text = f"Error: {result.get('message', 'Unknown error')}"

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=response_text)],
            ),
        )


# =========================================================
# Orchestration Entry Point
# =========================================================

def orchestrate(user_request: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Main entry point for the full LLM orchestration pipeline.

    Use for complex, multi-deliverable requests.
    Use run_agent() for quick single-task direct calls.
    """
    if verbose:
        logger.info("Launching orchestration for: %s", user_request[:80])
        logger.info("Active key pool size: %d", len(key_rotator.keys))

    try:
        result = root_agent.run(user_request)
        if verbose:
            logger.info("Content generation complete.")
        return {"status": "success", "result": result, "timestamp": time.time()}

    except Exception as exc:
        if _is_rate_limit_error(exc) and key_rotator.all_exhausted():
            logger.error("All API keys exhausted.")
            return {
                "status": "error",
                "message": "All API keys hit their quota.",
                "timestamp": time.time(),
            }
        logger.error("Orchestration failed: %s", exc, exc_info=True)
        return {"status": "error", "message": str(exc), "timestamp": time.time()}


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        # Direct mode: python agent_integrated.py direct image "walking elephant"
        if mode == "direct" and len(sys.argv) >= 4:
            task = sys.argv[2]
            prompt_str = " ".join(sys.argv[3:])
            print(json.dumps(run_agent(task, {"prompt": prompt_str}), indent=2))
            sys.exit(0)

        request = " ".join(sys.argv[1:])
    else:
        request = (
            "Create a LinkedIn post about AI trends with an ad image and a full report"
        )

    print(f"\nRequest: {request}\n")
    result = orchestrate(request, verbose=True)

    if result["status"] == "success":
        print("\nRESULT:")
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\nError: {result['message']}")