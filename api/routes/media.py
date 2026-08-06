"""
Media Preprocessing Route — PreViral
POST /preprocess-media

Called immediately when the user uploads a thumbnail, BEFORE they click Analyze.
Runs the full vision engine and caches the result server-side.
Returns a vision_cache_id that the /analyze endpoint accepts.

This keeps CLIP off the critical path — by the time the user fills in
their caption and hits Analyze, vision features are already computed.
"""
import os
import sys
import uuid
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import APIRouter, UploadFile, File, HTTPException
from engines.vision_engine import analyze_image, no_image_defaults

router = APIRouter()

# In-memory cache: {cache_id: {features: dict, ts: float}}
# Entries expire after 30 minutes (user should have analyzed by then)
_vision_cache: dict = {}
CACHE_TTL = 1800  # 30 minutes

def _evict_expired():
    now = time.time()
    expired = [k for k, v in _vision_cache.items() if now - v["ts"] > CACHE_TTL]
    for k in expired:
        del _vision_cache[k]

@router.post("/preprocess-media")
async def preprocess_media(
    platform: str = "instagram",
    media: UploadFile = File(...)
):
    """
    Upload a thumbnail immediately on file select.
    Runs CLIP + face detection + HSV analysis in the background.
    Returns a cache_id to pass to /analyze so vision doesn't re-run.
    """
    _evict_expired()

    if not media.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    image_bytes = await media.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # Run vision analysis (this is where CLIP's 2-4s runs)
    import asyncio
    loop = asyncio.get_event_loop()
    features = await loop.run_in_executor(None, analyze_image, image_bytes, platform)

    cache_id = str(uuid.uuid4())
    _vision_cache[cache_id] = {
        "features": features,
        "ts": time.time(),
        "platform": platform,
        "filename": media.filename
    }

    return {
        "vision_cache_id": cache_id,
        "features": features,
        "message": "Vision analysis complete — thumbnail is ready for analysis"
    }


def get_cached_vision(cache_id: str) -> dict | None:
    """Called by the analyze route to retrieve pre-computed vision features."""
    if cache_id and cache_id in _vision_cache:
        return _vision_cache[cache_id]["features"]
    return None
