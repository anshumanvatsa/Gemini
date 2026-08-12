"""
Pydantic Schemas — PreViral API
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class AnalyzeRequest(BaseModel):
    caption: str = Field(..., description="Draft post caption including any hashtags")
    platform: str = Field(..., description="Target platform: instagram|tiktok|youtube|twitter|linkedin|facebook|reddit")
    post_datetime: Optional[str] = Field(None, description="ISO datetime string for planned post time. Defaults to now.")
    follower_count: Optional[int] = Field(1000, description="Account follower count")
    avg_engagement_rate: Optional[float] = Field(0.03, description="Historical avg engagement rate (0-1)")
    niche: Optional[str] = Field("tech", description="Content niche: tech|fashion|food|fitness|travel|business|entertainment|beauty")

    class Config:
        json_schema_extra = {
            "example": {
                "caption": "Just shipped our AI coding assistant! You won't believe how fast it writes Python. Tag a developer friend! #ai #coding #startup",
                "platform": "instagram",
                "post_datetime": "2026-08-05T11:00:00",
                "follower_count": 5000,
                "avg_engagement_rate": 0.04,
                "niche": "tech"
            }
        }


class HashtagSuggestion(BaseModel):
    hashtag: str
    competition_score: float
    trend_velocity: float
    trend_status: str
    relevance_score: float
    composite_score: float
    niche: str


class CounterfactualSuggestion(BaseModel):
    feature: str
    direction: str
    current_value: Optional[float]
    target_direction: str
    suggestion: str
    estimated_impact: str


class TrajectoryPoint(BaseModel):
    day: int
    low: int
    mid: int
    high: int


class AnalyzeResponse(BaseModel):
    # Core Prediction
    prediction: str                    # HIGH | LOW
    confidence: float                  # 0-1
    reach_percentile: int              # 0-100
    headline: str                      # One-line summary

    # Feature Breakdown
    nlp_features: Dict[str, Any]
    hashtag_features: Dict[str, Any]
    vision_features: Dict[str, Any]
    timing_features: Dict[str, Any]

    # Outputs
    suggestions: List[CounterfactualSuggestion]
    hashtag_suggestions: List[HashtagSuggestion]
    trajectory: List[TrajectoryPoint]

    # Meta
    platform: str
    processing_time_ms: float

    # Real-time Trend Intelligence (Gemini Search Grounding)
    trending_hashtags: Optional[Dict[str, Any]] = None


class HashtagQueryRequest(BaseModel):
    query: str
    platform: str
    niche: Optional[str] = None
    top_k: Optional[int] = 10
