import base64
import hashlib
import os
import time
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from google.adk.agents import Agent
from google.adk.agents import SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import Gemini
from google.adk.tools.google_search_tool import google_search
from google.adk.tools import FunctionTool
from google.adk.tools.agent_tool import AgentTool
from google import genai
from google.genai import types

load_dotenv()

# =========================================================
# Tools
# =========================================================

MEDIA_OUTPUT_DIR = Path(os.getenv("MEDIA_OUTPUT_DIR", "generated_assets"))
IMAGE_MODEL = os.getenv("IMAGE_GENERATION_MODEL", "gemini-2.5-flash-image")
VIDEO_MODEL = os.getenv("VIDEO_GENERATION_MODEL", "veo-2.0-generate-001")


def _media_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY to enable media generation.")
    return genai.Client(api_key=api_key)


def _asset_path(prefix: str, source: str, extension: str) -> Path:
    MEDIA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return MEDIA_OUTPUT_DIR / f"{prefix}_{digest}.{extension}"


def _save_bytes(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return str(path)


def _save_gemini_image(prompt: str, style: str, aspect_ratio: str) -> dict:
    """Generate an image via Gemini's native image-generation capability."""
    client = _media_client()
    # Gemini image models use generate_content with IMAGE response modality.
    # aspect_ratio is passed as a hint in the prompt since GenerateContentConfig
    # does not have a dedicated aspect_ratio field for all model versions.
    full_prompt = f"{prompt}\nAspect ratio: {aspect_ratio}"
    response = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        ),
    )

    # Extract inline image bytes from the first candidate.
    image_data = None
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        parts = getattr(getattr(candidates[0], "content", None), "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline:
                image_data = getattr(inline, "data", None)
                break

    if not image_data:
        return {"status": "error", "message": "Gemini image API returned no image data.", "style": style, "model": IMAGE_MODEL}

    image_bytes = base64.b64decode(image_data)
    asset = _save_bytes(_asset_path("image", prompt, "jpg"), image_bytes)
    return {"status": "success", "asset": asset, "style": style, "model": IMAGE_MODEL}


def _save_imagen_image(prompt: str, style: str) -> dict:
    client = _media_client()
    result = client.models.generate_images(
        model=IMAGE_MODEL,
        prompt=prompt,
        config=types.GenerateImagesConfig(number_of_images=1),
    )

    images = getattr(result, "generated_images", None) or []
    if not images:
        return {"status": "error", "message": "Image API returned no generated images.", "style": style, "model": IMAGE_MODEL}

    image = images[0].image
    image_bytes = getattr(image, "image_bytes", None)
    if not image_bytes:
        return {"status": "success", "asset": getattr(image, "uri", None), "style": style, "model": IMAGE_MODEL}

    asset = _save_bytes(_asset_path("image", prompt, "png"), image_bytes)
    return {"status": "success", "asset": asset, "style": style, "model": IMAGE_MODEL}


def generate_image(prompt: str, style: str = "photorealistic", aspect_ratio: str = "9:16") -> dict:
    """Generates an ad image based on a creative prompt."""
    try:
        full_prompt = f"{prompt}\nStyle: {style}"
        if IMAGE_MODEL.startswith("gemini-"):
            return _save_gemini_image(full_prompt, style, aspect_ratio)
        return _save_imagen_image(full_prompt, style)
    except Exception as exc:
        return {"status": "error", "message": str(exc), "style": style, "model": IMAGE_MODEL}


def create_social_media_post(platform: str, copy: str, hashtags: list[str]) -> dict:
    """Create a social media post."""
    print(f"[SocialTool] platform={platform}")
    return {"status": "success", "platform": platform, "copy": copy, "hashtags": hashtags}


def write_content(content_type: str, brief: str, tone: str) -> dict:
    """Creates a compact draft for video scripts or simple long-form content."""
    print(f"[WritingTool] content_type={content_type} tone={tone}")
    draft = (
        f"Title: {brief}\n\n"
        f"Here is a short {content_type} in a {tone} tone. "
        "It uses clear, simple language and is ready to refine into the final answer."
    )
    return {
        "status": "success",
        "content_type": content_type,
        "draft": draft,
        "word_count": len(draft.split()),
    }


def produce_video(script: str, format: str = "short", duration_seconds: int = 30) -> dict:
    """Produce a short video from a script."""
    try:
        client = _media_client()
        duration_seconds = max(1, min(duration_seconds, int(os.getenv("MAX_VIDEO_DURATION_SECONDS", "60"))))
        operation = client.models.generate_videos(
            model=VIDEO_MODEL,
            prompt=script,
            config=types.GenerateVideosConfig(
                duration_seconds=duration_seconds,
                aspect_ratio="9:16" if format == "short" else "16:9",
            ),
        )

        timeout_seconds = int(os.getenv("VIDEO_GENERATION_TIMEOUT_SECONDS", "600"))
        poll_interval_seconds = int(os.getenv("VIDEO_GENERATION_POLL_SECONDS", "5"))
        deadline = time.monotonic() + timeout_seconds
        while not getattr(operation, "done", False):
            if time.monotonic() >= deadline:
                return {
                    "status": "pending",
                    "operation": getattr(operation, "name", None),
                    "format": format,
                    "duration": duration_seconds,
                    "model": VIDEO_MODEL,
                }
            time.sleep(poll_interval_seconds)
            operation = client.operations.get(operation)

        videos = getattr(getattr(operation, "response", None), "generated_videos", None) or []
        if not videos:
            return {"status": "error", "message": "Video API returned no generated videos.", "model": VIDEO_MODEL}

        video = videos[0].video
        video_bytes = getattr(video, "video_bytes", None)
        if video_bytes:
            file = _save_bytes(_asset_path("video", script, "mp4"), video_bytes)
        else:
            file = getattr(video, "uri", None)

        return {
            "status": "success",
            "format": format,
            "file": file,
            "duration": duration_seconds,
            "model": VIDEO_MODEL,
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
            "format": format,
            "duration": duration_seconds,
            "model": VIDEO_MODEL,
        }


image_tool = FunctionTool(func=generate_image)
social_tool = FunctionTool(func=create_social_media_post)
writing_tool = FunctionTool(func=write_content)
video_tool = FunctionTool(func=produce_video)

# =========================================================
# Shared model constant
# =========================================================
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LLM_RETRY_ATTEMPTS = int(os.getenv("GEMINI_RETRY_ATTEMPTS", "5"))
LLM_RETRY_INITIAL_DELAY = float(os.getenv("GEMINI_RETRY_INITIAL_DELAY", "1"))
LLM_RETRY_MAX_DELAY = float(os.getenv("GEMINI_RETRY_MAX_DELAY", "30"))

# Gemini can occasionally return 503 UNAVAILABLE during temporary demand spikes.
# ADK passes these retry options to the underlying GenAI SDK, so transient
# capacity errors are retried before the agent run fails.
LLM = Gemini(
    model=LLM_MODEL,
    retry_options=types.HttpRetryOptions(
        attempts=LLM_RETRY_ATTEMPTS,
        initial_delay=LLM_RETRY_INITIAL_DELAY,
        max_delay=LLM_RETRY_MAX_DELAY,
        exp_base=2,
        jitter=1,
        http_status_codes=[408, 429, 500, 502, 503, 504],
    ),
)

# =========================================================
# Search Agent
# =========================================================

search_agent = Agent(
    name="search_agent",
    model=LLM,
    description="Search the web for up-to-date information on any topic.",
    instruction="""
    You are a Social Media Research Specialist.
    Search for accurate, recent, and relevant information related to the user's topic using available search tools.
    Return concise findings with key insights, statistics, trends, and sources. Do not generate social media content.
    """,
    tools=[google_search],
)

search_agent_tool = AgentTool(agent=search_agent)

# =========================================================
# Structured output schemas
# =========================================================
# CHANGE: replaced "Return ONLY valid JSON" prompt-only instructions with
# Pydantic output_schema. This is far more reliable than hoping the model
# doesn't add markdown fences / prose around the JSON.

class ContentBrief(BaseModel):
    topic: str = "unknown"
    audience: str = "unknown"
    tone: str = "unknown"
    brand: str = "unknown"
    platform: str = Field(default="unknown", description="LinkedIn | Instagram | Twitter/X | Facebook")
    deliverables: List[str] = Field(
        default_factory=list,
        # Fix 2: Added "story" so the brief agent captures it when a user asks for a story.
        # The dispatcher maps both "report" and "story" -> pipeline_b.
        description="Subset of: social_post, ad_images, report, story, short_video",
    )


class DispatchDecision(BaseModel):
    pipelines: List[str] = Field(
        description="Which pipelines to run, subset of: pipeline_a, pipeline_b, pipeline_c"
    )

# =========================================================
# Brief Agent
# =========================================================

brief_agent = Agent(
    name="BriefAgent",
    model=LLM,
    description="Extracts a structured content brief from user input.",
    instruction="""
    You are a Content Brief Specialist.
    Analyze the user's request and extract a structured content brief:
    topic, audience, tone, brand, platform, and deliverables.

    Deliverables must be chosen from this list (include ALL that apply):
    - social_post   → a caption / post for a social platform
    - ad_images     → visual ad images
    - report        → a written report or article
    - story         → a narrative story
    - short_video   → a short-form video (TikTok / Reels style)

    A single user request may result in multiple deliverables.
    If a field is missing or unclear, use "unknown".
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
    description="Decides which content pipelines should run for this brief.",
    instruction="""
    Read the brief: {brief}

    Map brief.deliverables to pipelines using these rules:
    - "social_post" or "ad_images"          -> include "pipeline_a"
    - "report" or "story"                   -> include "pipeline_b"
    - "short_video"                         -> include "pipeline_c"

    If the user asked for BOTH images/posts AND a report/story, include BOTH
    "pipeline_a" AND "pipeline_b" in the result so that each pipeline handles
    its own speciality.

    Return the list of pipelines that should run, with no duplicates.
    Do not create content yourself.
    """,
    output_schema=DispatchDecision,
    output_key="dispatch",
)

# =========================================================
# Conditional execution helper
# =========================================================
# Each pipeline gets a before_agent_callback that checks state["dispatch"].
# If this pipeline wasn't selected, the callback returns Content, which tells
# ADK to skip running that agent's model/tools.

def make_skip_callback(pipeline_name: str):
    def _skip_if_not_selected(callback_context: CallbackContext) -> Optional[types.Content]:
        selected = _selected_pipelines(callback_context)
        if pipeline_name not in selected:
            return types.Content(
                role="model",
                parts=[types.Part(text=f"Skipped: {pipeline_name} not selected by dispatcher.")],
            )
        return None  # None = proceed with normal execution
    return _skip_if_not_selected


def _selected_pipelines(callback_context: CallbackContext) -> list[str]:
    dispatch = callback_context.state.get("dispatch") or {}
    # Fix 4: When ADK serialises a Pydantic output_key value it may arrive as a
    # JSON string.  Parse it properly instead of wrapping the whole string as a
    # single list element.
    if isinstance(dispatch, str):
        import json as _json
        try:
            parsed = _json.loads(dispatch)
            if isinstance(parsed, dict):
                return parsed.get("pipelines", [])
        except (_json.JSONDecodeError, TypeError):
            pass
        return [dispatch]  # fallback: treat the raw string as a pipeline name
    if isinstance(dispatch, dict):
        return dispatch.get("pipelines", [])
    return getattr(dispatch, "pipelines", [])


def _brief_value(brief: object, key: str, default: str = "unknown") -> str:
    if isinstance(brief, dict):
        value = brief.get(key, default)
    else:
        value = getattr(brief, key, default)
    return str(value or default)


# _instagram_caption and make_pipeline_a_callback removed.
# Pipeline A is now fully LLM-driven via a brief-aware instruction + tools,
# using make_skip_callback for conditional execution (same pattern as B and C).

# =========================================================
# Pipeline A - Ad Images + Social Post
# =========================================================

pipeline_a_agent = Agent(
    name="pipelineA_AdImages",
    model=LLM,
    description="Generates ad images and social media posts for a campaign.",
    instruction="""
    You are a Social Media Ad Specialist.

    Brief: {brief}

    Your responsibilities:
    1. Use the generate_image tool to create a visually compelling ad image that matches
       brief.topic, brief.platform, brief.tone, and brief.audience.
       - For vertical/mobile platforms (Instagram, TikTok) use aspect_ratio="9:16".
       - For horizontal platforms (LinkedIn, YouTube) use aspect_ratio="16:9".
       - For square feeds use aspect_ratio="1:1".
    2. Use the create_social_media_post tool to craft an engaging post copy with relevant hashtags.
    3. Return the image file path and the full post copy.

    Always base your content exactly on what the brief specifies.
    Do not substitute your own topic, brand, or generic placeholder content.
    """,
    tools=[image_tool, social_tool],
    before_agent_callback=make_skip_callback("pipeline_a"),
    output_key="pipeline_a_result",
)

# =========================================================
# Pipeline B - Report
# =========================================================

pipeline_b_agent = Agent(
    name="pipelineB_ContentSpecialist",
    model=LLM,
    description="Writes reports/stories and optionally generates supporting images and social posts.",
    instruction="""
    You are a versatile Content Specialist who handles written content and supporting media.

    Brief: {brief}

    Work through each deliverable in brief.deliverables:

    1. REPORT or STORY
       - Use the write_content tool (content_type="report" or "story") to draft the content.
       - If the topic requires up-to-date facts, use the search_agent tool first to
         gather accurate information, then pass those findings into write_content.
       - Match brief.audience and brief.tone exactly.
       - Keep reports concise (120-180 words) unless the user asks otherwise.
       - Use a clear title, simple paragraphs or friendly bullet points.

    2. AD IMAGES (if brief.deliverables includes "ad_images" or supporting visuals are useful)
       - Use the generate_image tool to create a relevant visual.
       - Match the image style and aspect ratio to brief.platform:
           * Instagram / TikTok / Reels → aspect_ratio="9:16"
           * LinkedIn / YouTube        → aspect_ratio="16:9"
           * General / square feed     → aspect_ratio="1:1"

    3. SOCIAL POST (if brief.deliverables includes "social_post")
       - Use the create_social_media_post tool to craft platform-appropriate copy and hashtags.

    Present all outputs clearly labelled (e.g. "## Report", "## Ad Image", "## Social Post").
    Do not return raw JSON. Do not include tool status messages in your final answer.
    """,
    tools=[search_agent_tool, image_tool, social_tool, writing_tool],
    before_agent_callback=make_skip_callback("pipeline_b"),
    output_key="pipeline_b_result",
)

# =========================================================
# Pipeline C - Short Video
# =========================================================

pipeline_c_agent = Agent(
    name="pipelineC_ShortVideo",
    model=LLM,
    description="Writes a video script and produces a short video.",
    instruction="""
    You are a Video Content Specialist.

    Brief: {brief}

    Your responsibilities:
    - Create an engaging video script matching the brief's topic, audience, and tone.
    - Structure: Hook, Introduction, Main Content, Call To Action, Closing.
    - If research is required, use the search tool to find up-to-date information before writing.

    Keep scripts clear, concise, and audience-focused.
    """,
    # Fix 5: Added search_agent_tool so the video agent can research topics
    # when the brief requires up-to-date information before scripting.
    tools=[search_agent_tool, writing_tool, video_tool],
    before_agent_callback=make_skip_callback("pipeline_c"),
    output_key="pipeline_c_result",
)

# =========================================================
# Orchestration
# =========================================================

root_agent = SequentialAgent(
    name="Social_Media_Main",
    description="""
    End-to-end content pipeline: Brief -> Dispatch -> Selected Pipelines
    (only dispatcher-selected pipelines actually execute).
    Produces social content for LinkedIn, Instagram, Twitter/X, and Facebook.
    """,
    sub_agents=[
        brief_agent,
        dispatcher_agent,
        SequentialAgent(
            name="PipelineRunner",
            description="Runs dispatcher-selected pipelines; others are skipped.",
            # Fix 6: Corrected order to A (images/posts) -> B (reports) -> C (videos)
            # so pipeline execution matches the logical content creation sequence.
            sub_agents=[
                pipeline_a_agent,
                pipeline_b_agent,
                pipeline_c_agent,
            ],
        ),
    ],
)