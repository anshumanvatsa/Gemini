"""
Hashtag Route — PreViral
GET /hashtags?q=<caption>&platform=<platform>&niche=<niche>
Returns ranked hashtag suggestions for a given caption and platform.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import APIRouter, Query
from typing import Optional
from engines.hashtag_engine import suggest_hashtags
from api.schemas import HashtagSuggestion

router = APIRouter()

@router.get("/hashtags")
async def get_hashtag_suggestions(
    q: str = Query(..., description="Caption or topic text"),
    platform: str = Query(..., description="Target platform"),
    niche: Optional[str] = Query(None, description="Content niche"),
    top_k: Optional[int] = Query(10, description="Number of suggestions to return")
):
    suggestions = suggest_hashtags(q, platform, niche, top_k)
    return {
        "query": q,
        "platform": platform,
        "niche": niche,
        "count": len(suggestions),
        "suggestions": suggestions
    }
