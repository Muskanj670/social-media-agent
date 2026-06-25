import base64
import asyncio
import hashlib
import itertools
import json
import os
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field, PrivateAttr

from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import Gemini
from google.adk.tools.google_search_tool import google_search
from google.adk.tools import FunctionTool
from google.adk.tools.agent_tool import AgentTool
from google import genai
from google.genai import types
from google.genai.errors import ClientError

# =========================================================
# Configuration & Logging
# =========================================================

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Paths & Models
MEDIA_OUTPUT_DIR = Path(os.getenv("MEDIA_OUTPUT_DIR", "generated_assets"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache"))
IMAGE_MODEL = os.getenv("IMAGE_GENERATION_MODEL", "gemini-2.5-flash")
VIDEO_MODEL = os.getenv("VIDEO_GENERATION_MODEL", "veo-2.0-generate-001")

# Rate limiting & Retry
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
INITIAL_RETRY_DELAY = float(os.getenv("INITIAL_RETRY_DELAY", "1"))
MAX_RETRY_DELAY = float(os.getenv("MAX_RETRY_DELAY", "30"))
REQUEST_THROTTLE_DELAY = float(os.getenv("REQUEST_THROTTLE_DELAY", "0.5"))

# Token optimization
USE_CACHE = os.getenv("USE_CACHE", "true").lower() == "true"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
MIN_SEARCH_CONFIDENCE = float(os.getenv("MIN_SEARCH_CONFIDENCE", "0.6"))


# =========================================================
# Multi-Key Rotation / Failover
# =========================================================
#
# .env should contain something like:
#   GEMINI_API_KEY_1=xxxx
#   GEMINI_API_KEY_2=xxxx
#   GEMINI_API_KEY_3=xxxx
#
# (Falls back to a single GOOGLE_API_KEY / GEMINI_API_KEY if no
# numbered keys are found, so this is backward compatible.)

class GeminiKeyRotator:
    def __init__(self, env_prefix: str = "GEMINI_API_KEY"):
        numbered_keys = sorted(
            (k, v) for k, v in os.environ.items()
            if k.startswith(f"{env_prefix}_") and v
        )
        keys = [v for _, v in numbered_keys]

        # Backward-compatible fallback to a single key
        if not keys:
            single = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if single:
                keys = [single]

        if not keys:
            raise RuntimeError(
                "No API keys found. Set GEMINI_API_KEY_1, GEMINI_API_KEY_2, ... "
                "in .env, or at minimum GOOGLE_API_KEY / GEMINI_API_KEY."
            )

        self.keys: List[str] = keys
        self._cycle = itertools.cycle(self.keys)
        self.current_key: str = next(self._cycle)
        self._dead_until: Dict[str, float] = {}  # key -> unix time it can be retried
        logger.info(f"[KeyRotator] Loaded {len(self.keys)} API key(s).")

    def rotate(self) -> str:
        """Move to the next key and return it."""
        previous = self.current_key
        for _ in range(len(self.keys)):
            candidate = next(self._cycle)
            cooldown = self._dead_until.get(candidate, 0)
            if cooldown <= time.time():
                self.current_key = candidate
                if self.current_key != previous:
                    logger.warning(
                        f"[KeyRotator] Switched key ...{previous[-4:]} -> "
                        f"...{self.current_key[-4:]}"
                    )
                return self.current_key
        # Every key is on cooldown — just take the next one anyway and
        # let the caller's own retry/backoff handle it.
        self.current_key = next(self._cycle)
        return self.current_key

    def mark_exhausted(self, key: Optional[str] = None, cooldown_seconds: float = 60.0) -> None:
        """Flag a key as temporarily unusable (e.g. just hit a 429)."""
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
# Cache Management (Token Efficiency)
# =========================================================

class ContentCache:
    """Simple file-based cache for generated content."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_hash(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not USE_CACHE:
            return None

        cache_file = self.cache_dir / f"{self._key_hash(key)}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    logger.debug(f"Cache HIT: {key[:50]}")
                    return data
            except (json.JSONDecodeError, IOError):
                pass
        return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        if not USE_CACHE:
            return

        cache_file = self.cache_dir / f"{self._key_hash(key)}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(value, f)
                logger.debug(f"Cache MISS (stored): {key[:50]}")
        except IOError as e:
            logger.warning(f"Cache write failed: {e}")


cache = ContentCache()

# =========================================================
# Tools - Optimized for Token Efficiency & Speed
# =========================================================


def _media_client() -> genai.Client:
    """Get Gemini API client using whichever key is currently active."""
    return genai.Client(api_key=key_rotator.current_key)


def _asset_path(prefix: str, source: str, extension: str) -> Path:
    """Generate unique asset path based on content hash."""
    MEDIA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.encode()).hexdigest()[:12]
    return MEDIA_OUTPUT_DIR / f"{prefix}_{digest}.{extension}"


def _save_bytes(path: Path, data: bytes) -> str:
    """Save binary data to file."""
    path.write_bytes(data)
    return str(path)


def _exception_message(exc: BaseException) -> str:
    """Extract clean error message."""
    if isinstance(exc, BaseExceptionGroup):
        messages = [_exception_message(inner) for inner in exc.exceptions]
        return "; ".join(m for m in messages if m) or str(exc)
    return str(exc)


def _inline_image_bytes(inline_data: object) -> tuple[Optional[bytes], str]:
    """Extract image bytes from Gemini response."""
    data = getattr(inline_data, "data", None)
    if not data:
        return None, "jpg"

    mime_type = str(getattr(inline_data, "mime_type", "") or "").lower()
    extension = "png" if mime_type == "image/png" else "jpg"
    if isinstance(data, str):
        return base64.b64decode(data), extension
    if isinstance(data, (bytes, bytearray)):
        return bytes(data), extension
    return None, extension


def _retry_with_backoff(func, max_retries: int = MAX_RETRIES):
    """
    Decorator: Retry function with exponential backoff on rate limits.

    On a rate-limit/quota error this now ALSO rotates to the next API
    key before retrying, so a key that's exhausted doesn't keep eating
    retries — the next attempt goes out on a fresh key immediately.
    """
    def wrapper(*args, **kwargs):
        total_attempts = max_retries * max(len(key_rotator.keys), 1)
        for attempt in range(total_attempts):
            try:
                time.sleep(REQUEST_THROTTLE_DELAY)  # Pre-throttle
                return func(*args, **kwargs)
            except Exception as e:
                if _is_rate_limit_error(e) and attempt < total_attempts - 1:
                    key_rotator.mark_exhausted()
                    key_rotator.rotate()
                    # Light backoff between attempts on the *new* key too,
                    # in case it's also close to its own limit.
                    wait = min(INITIAL_RETRY_DELAY * (2 ** (attempt % max_retries)), MAX_RETRY_DELAY)
                    wait += time.time() % 1  # jitter
                    logger.warning(
                        f"Rate limited. Retry {attempt + 1}/{total_attempts} "
                        f"in {wait:.1f}s on key ...{key_rotator.current_key[-4:]}"
                    )
                    time.sleep(wait)
                    continue
                raise
        return None
    return wrapper


@_retry_with_backoff
def _save_gemini_image(prompt: str, style: str, aspect_ratio: str) -> dict:
    """Generate image via Gemini with retry + key rotation."""
    client = _media_client()
    full_prompt = f"{prompt}\nStyle: {style}\nAspect ratio: {aspect_ratio}"

    response = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        ),
    )

    image_bytes = None
    extension = "jpg"
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        parts = getattr(getattr(candidates[0], "content", None), "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline:
                image_bytes, extension = _inline_image_bytes(inline)
                if image_bytes:
                    break

    if not image_bytes:
        raise RuntimeError("No image data returned from Gemini")

    asset = _save_bytes(_asset_path("image", prompt, extension), image_bytes)
    return {"status": "success", "asset": asset, "style": style, "model": IMAGE_MODEL}


def generate_image(prompt: str, style: str = "photorealistic", aspect_ratio: str = "9:16") -> dict:
    """
    Generate image with caching.

    Tokens saved:
    - Cache hit: ~0 tokens (file lookup)
    - Cache miss: ~150 tokens (Gemini image generation)
    """
    cache_key = f"image:{prompt}:{style}:{aspect_ratio}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    try:
        result = _save_gemini_image(prompt, style, aspect_ratio)
        cache.set(cache_key, result)
        return result
    except BaseExceptionGroup as exc:
        return {
            "status": "error",
            "message": _exception_message(exc),
            "style": style,
            "model": IMAGE_MODEL
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
            "style": style,
            "model": IMAGE_MODEL
        }


def create_social_media_post(platform: str, copy: str, hashtags: list[str]) -> dict:
    """Create social media post (local, no API call)."""
    logger.info(f"[SocialPost] {platform} - {len(copy)} chars, {len(hashtags)} hashtags")
    return {
        "status": "success",
        "platform": platform,
        "copy": copy,
        "hashtags": hashtags,
        "post_url": f"draft://{platform}/{hashlib.md5(copy.encode()).hexdigest()[:8]}"
    }


def write_content(content_type: str, brief: str, tone: str) -> dict:
    """
    Create content draft.

    Token optimization:
    - Fast local generation
    - Structured output for parsing
    """
    logger.info(f"[ContentWriter] {content_type} in {tone} tone")

    templates = {
        "report": "## {title}\n\n{intro}\n\n### Key Points\n{body}\n\n### Conclusion\n{outro}",
        "story": "# {title}\n\n{intro}\n\n{body}\n\n{outro}",
        "script": "{intro}\n\n{body}\n\n{outro}",
    }

    template = templates.get(content_type, templates["report"])
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
        "tokens_estimated": len(draft.split()) * 1.3  # Rough estimate
    }


@_retry_with_backoff
def produce_video(script: str, format: str = "short", duration_seconds: int = 30) -> dict:
    """
    Produce video with retry, key rotation & timeout handling.

    Note: Video production is slow; consider caching or returning async operation.
    """
    client = _media_client()
    duration = max(1, min(duration_seconds, int(os.getenv("MAX_VIDEO_DURATION_SECONDS", "60"))))

    logger.info(f"[VideoProduction] {format} format, {duration}s duration")

    operation = client.models.generate_videos(
        model=VIDEO_MODEL,
        prompt=script,
        config=types.GenerateVideosConfig(
            duration_seconds=duration,
            aspect_ratio="9:16" if format == "short" else "16:9",
        ),
    )

    timeout_seconds = int(os.getenv("VIDEO_GENERATION_TIMEOUT_SECONDS", "600"))
    poll_interval = int(os.getenv("VIDEO_GENERATION_POLL_SECONDS", "5"))
    deadline = time.monotonic() + timeout_seconds

    while not getattr(operation, "done", False):
        if time.monotonic() >= deadline:
            logger.warning(f"Video generation timeout after {timeout_seconds}s")
            return {
                "status": "pending",
                "operation": getattr(operation, "name", None),
                "format": format,
                "duration": duration,
                "model": VIDEO_MODEL,
                "message": "Processing will continue in background"
            }
        time.sleep(poll_interval)
        operation = client.operations.get(operation)

    videos = getattr(getattr(operation, "response", None), "generated_videos", None) or []
    if not videos:
        raise RuntimeError("No videos returned from API")

    video = videos[0].video
    video_bytes = getattr(video, "video_bytes", None)
    file = _save_bytes(_asset_path("video", script, "mp4"), video_bytes) if video_bytes else getattr(video, "uri", None)

    return {
        "status": "success",
        "format": format,
        "file": file,
        "duration": duration,
        "model": VIDEO_MODEL
    }


# FunctionTools
image_tool = FunctionTool(func=generate_image)
social_tool = FunctionTool(func=create_social_media_post)
writing_tool = FunctionTool(func=write_content)
video_tool = FunctionTool(func=produce_video)

# =========================================================
# LLM Configuration - Token Optimized + Key Rotation
# =========================================================
#
# IMPORTANT: This part wraps google.adk.models.Gemini so the agent
# pipeline (BriefAgent, DispatcherAgent, PipelineA/B/C) also rotates
# keys on quota errors. It overrides generate_content_async, which is
# the method ADK's Agent runtime calls under the hood as of the
# google-adk versions this was written against.
#
# VERIFY THIS AGAINST YOUR INSTALLED VERSION:
#   python3 -c "from google.adk.models import Gemini; import inspect;
#               print(inspect.signature(Gemini.generate_content_async))"
# If ADK has changed this method's name/signature, adjust the override
# below accordingly — the rotation logic itself (rotate + retry) will
# still work once the override hooks the right method.

class RotatingGemini(Gemini):
    """Gemini model wrapper that rotates API keys on quota errors."""

    _client_cache: Dict[str, genai.Client] = PrivateAttr(default_factory=dict)

    @property
    def api_client(self) -> genai.Client:
        active_key = key_rotator.current_key
        cached_client = self._client_cache.get(active_key)
        if cached_client is not None:
            return cached_client

        base_url, api_version = self._base_url_and_api_version
        http_options: dict[str, Any] = {
            "headers": self._tracking_headers(),
            "retry_options": self.retry_options,
            "base_url": base_url,
        }
        if api_version:
            http_options["api_version"] = api_version

        kwargs: dict[str, Any] = {
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
        last_exc = None
        for attempt in range(total_attempts):
            try:
                async for response in super().generate_content_async(*args, **kwargs):
                    yield response
                return
            except Exception as e:
                last_exc = e
                if _is_rate_limit_error(e) and attempt < total_attempts - 1:
                    key_rotator.mark_exhausted()
                    key_rotator.rotate()
                    logger.warning(
                        f"[RotatingGemini] Quota hit, switched to key "
                        f"...{key_rotator.current_key[-4:]} (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY))
                    continue
                raise
        if last_exc:
            raise last_exc


# Use fast, cheap model for orchestration
ORCHESTRATION_MODEL = os.getenv("ORCHESTRATION_MODEL", "gemini-2.5-flash")
CONTENT_MODEL = os.getenv("CONTENT_MODEL", "gemini-2.5-flash")  # Same for consistency

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
# Search Agent - Token Optimized
# =========================================================

search_agent = Agent(
    name="SearchAgent",
    model=LLM,
    description="Search for recent information efficiently.",
    instruction="""
You are a Research Assistant. When asked to search:
1. Use ONLY the search tool if the topic requires current information (news, recent data, trends)
2. Return ONLY the most relevant 3-5 facts with sources
3. Keep response concise (max 150 words)
4. Do NOT search for historical facts or general knowledge
""",
    tools=[google_search],
)

search_agent_tool = AgentTool(agent=search_agent)

# =========================================================
# Pydantic Schemas - Structured Output (Reliable Parsing)
# =========================================================


class ContentBrief(BaseModel):
    """Structured content brief."""
    topic: str = Field(..., description="Main topic")
    audience: str = Field(default="general", description="Target audience")
    tone: str = Field(default="professional", description="Writing tone")
    brand: str = Field(default="", description="Brand name (optional)")
    platform: str = Field(default="", description="Target platform (LinkedIn/Instagram/Twitter/Facebook)")
    deliverables: List[str] = Field(
        default_factory=list,
        description="Comma-separated: social_post, ad_images, report, story, short_video"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "topic": "AI trends 2024",
                "audience": "Tech professionals",
                "tone": "Informative",
                "platform": "LinkedIn",
                "deliverables": ["social_post", "ad_images"]
            }
        }


class DispatchDecision(BaseModel):
    """Pipeline dispatch decision."""
    pipelines: List[str] = Field(
        description="Selected pipelines: pipeline_a, pipeline_b, pipeline_c"
    )
    reasoning: str = Field(default="", description="Brief explanation (optional)")


# =========================================================
# Brief Agent - Fast & Lightweight
# =========================================================

brief_agent = Agent(
    name="BriefAgent",
    model=LLM,
    description="Extract content brief from user request.",
    instruction="""
Extract a content brief from the user's request.

RULES:
1. topic: Main subject (required)
2. audience: Who this is for (default: "general")
3. tone: Style (default: "professional")
4. platform: Where posted (LinkedIn/Instagram/Twitter/Facebook, or "")
5. deliverables: List ONLY items user asked for. Valid: social_post, ad_images, report, story, short_video

Be strict: only include deliverables explicitly requested. If user wants "story and images", include both.
If unclear, ask for clarification instead of guessing.
""",
    output_schema=ContentBrief,
    output_key="brief",
)

# =========================================================
# Dispatcher Agent - Minimal Logic
# =========================================================

dispatcher_agent = Agent(
    name="DispatcherAgent",
    model=LLM,
    description="Route to appropriate pipelines.",
    instruction="""
Given brief: {brief}

Map to pipelines:
- social_post OR ad_images → pipeline_a
- report OR story → pipeline_b
- short_video → pipeline_c

If user wants BOTH images AND report: include both pipeline_a AND pipeline_b.

Return pipelines list. No duplicates.
""",
    output_schema=DispatchDecision,
    output_key="dispatch",
)

# =========================================================
# Pipeline Conditional Execution
# =========================================================


def make_skip_callback(pipeline_name: str):
    """Skip pipeline if not selected."""
    def _skip_if_not_selected(callback_context: CallbackContext, **_: Any) -> Optional[types.Content]:
        selected = _selected_pipelines(callback_context)
        if pipeline_name not in selected:
            logger.info(f"Skipping {pipeline_name}")
            return types.Content(
                role="model",
                parts=[types.Part(text=f"[SKIPPED: {pipeline_name} not selected]")],
            )
        return None
    return _skip_if_not_selected


def _selected_pipelines(ctx: CallbackContext) -> list[str]:
    """Extract selected pipelines from context."""
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


def _brief_value(brief: object, key: str, default: str = "") -> str:
    """Extract brief field safely."""
    if isinstance(brief, dict):
        return str(brief.get(key, default) or default)
    return str(getattr(brief, key, default) or default)


# =========================================================
# Pipeline A - Social Media & Ad Images (Parallel Execution)
# =========================================================

pipeline_a_agent = Agent(
    name="PipelineA_SocialAds",
    model=LLM_CONTENT,
    description="Generate ad images and social posts.",
    instruction="""
Brief: {brief}

You MUST create content matching the brief exactly. Do NOT use placeholder content.

Steps:
1. Generate image prompt from brief.topic, brief.tone, brief.platform
   - For Instagram/TikTok: aspect_ratio="9:16"
   - For LinkedIn/Twitter: aspect_ratio="16:9"
2. Call generate_image tool with optimized prompt
3. Create social post with copy and hashtags matching platform
4. Return both image path and post copy

Output format:
## Image
[image file path and brief description]

## Social Post
[platform]: [copy]
Hashtags: [comma-separated list]
""",
    tools=[image_tool, social_tool],
    before_agent_callback=make_skip_callback("pipeline_a"),
    output_key="pipeline_a_result",
)

# =========================================================
# Pipeline B - Long-Form Content (Reports/Stories)
# =========================================================

pipeline_b_agent = Agent(
    name="PipelineB_LongForm",
    model=LLM_CONTENT,
    description="Generate reports and stories.",
    instruction="""
Brief: {brief}

Create written content:

1. If brief requires facts (like "AI trends"):
   - Use search_agent_tool FIRST to get current data
   - Wait for results before writing
   - Only use facts that passed search

2. Write content:
   - Type: report OR story (from brief.deliverables)
   - Tone: match brief.tone exactly
   - Length: 150-200 words (concise!)
   - Structure: Title, Opening, Main Points, Closing

3. If brief.deliverables includes "ad_images":
   - Also generate supporting image(s)
   - Use same brief.platform for aspect ratio

4. If brief.deliverables includes "social_post":
   - Also create post copy with hashtags

Output format:
## Report/Story
[full content]

## Supporting Image (if requested)
[image path]

## Social Post (if requested)
[copy + hashtags]
""",
    tools=[search_agent_tool, image_tool, social_tool, writing_tool],
    before_agent_callback=make_skip_callback("pipeline_b"),
    output_key="pipeline_b_result",
)

# =========================================================
# Pipeline C - Short Video
# =========================================================

pipeline_c_agent = Agent(
    name="PipelineC_Video",
    model=LLM_CONTENT,
    description="Generate short-form video scripts and videos.",
    instruction="""
Brief: {brief}

Create a short video:

1. If content requires research:
   - Use search_agent_tool for current facts
   - Wait for results

2. Write script:
   - Duration: 30-60 seconds (brief format)
   - Structure: Hook (3s) → Content (20s) → CTA (5s)
   - Tone: match brief.tone
   - Language: conversational, scannable

3. Generate video:
   - Format: "short" (9:16 vertical)
   - Duration: 30-45 seconds
   - Script: pass exactly to produce_video tool

Output format:
## Script
[30-45s video script]

## Video
[generated video file path]
""",
    tools=[search_agent_tool, writing_tool, video_tool],
    before_agent_callback=make_skip_callback("pipeline_c"),
    output_key="pipeline_c_result",
)

# =========================================================
# Main Orchestrator - Sequential for Reliability
# =========================================================

root_agent = SequentialAgent(
    name="ContentOrchestrator",
    description="Multi-pipeline content generation system.",
    sub_agents=[
        brief_agent,           # Extract brief
        dispatcher_agent,      # Route to pipelines
        SequentialAgent(
            name="PipelineRunner",
            description="Run selected pipelines in sequence.",
            sub_agents=[
                pipeline_a_agent,  # Social/Ads
                pipeline_b_agent,  # Reports/Stories
                pipeline_c_agent,  # Videos
            ],
        ),
    ],
)

# =========================================================
# Execution Wrapper (Production Entry Point)
# =========================================================


def orchestrate(user_request: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Main entry point: Generate content from user request.

    Args:
        user_request: Natural language request
        verbose: Log detailed execution

    Returns:
        Structured result with generated content
    """
    if verbose:
        logger.info(f"🚀 Processing request: {user_request[:60]}...")
        logger.info(f"🔑 Active key pool size: {len(key_rotator.keys)}")

    try:
        # Run the orchestrator
        result = root_agent.run(user_request)

        if verbose:
            logger.info("✅ Content generation complete")

        return {
            "status": "success",
            "result": result,
            "timestamp": time.time(),
        }

    except Exception as e:
        if _is_rate_limit_error(e) and key_rotator.all_exhausted():
            logger.error("❌ All API keys exhausted — stopping.")
            return {
                "status": "error",
                "message": "All available API keys hit their quota.",
                "timestamp": time.time(),
            }
        logger.error(f"❌ Orchestration failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "timestamp": time.time(),
        }


# =========================================================
# CLI & Testing
# =========================================================

if __name__ == "__main__":
    import sys

    # Example requests
    examples = [
        "Create a LinkedIn post about AI trends in 2024 with an ad image",
        "Write a report on mental health impacts of social media for parents and educators, include an educational illustration",
        "Make a 30-second TikTok script about sustainable fashion for Gen Z",
    ]

    if len(sys.argv) > 1:
        request = " ".join(sys.argv[1:])
    else:
        request = examples[0]

    print(f"\n📝 Request: {request}\n")
    result = orchestrate(request, verbose=True)

    if result["status"] == "success":
        print("\n✨ RESULT:")
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n❌ Error: {result['message']}")
