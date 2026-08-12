"""
Main Analyze Route — PreViral
POST /analyze — Fans out to all 4 engines in parallel using asyncio.gather()
Assembles the unified feature vector, runs LightGBM, generates counterfactuals.
"""
import asyncio
import time
import os
import sys
import numpy as np
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

# Add parent to path so engines can import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from api.schemas import AnalyzeResponse, CounterfactualSuggestion, TrajectoryPoint, HashtagSuggestion
from engines.nlp_engine import analyze_caption
from engines.hashtag_engine import extract_hashtags, score_hashtags, suggest_hashtags
from engines.timing_engine import analyze_timing
from engines.vision_engine import analyze_image, no_image_defaults
from counterfactual.dice_engine import generate_suggestions
from api.routes.media import get_cached_vision
try:
    from engines.gemini_engine import extract_features as gemini_features, ai_content_director, suggest_trending_hashtags
    GEMINI_ENGINE_AVAILABLE = True
except ImportError:
    GEMINI_ENGINE_AVAILABLE = False
    def suggest_trending_hashtags(*a, **kw): return {"trending_now": [], "stable_performers": [], "avoid": [], "grounding_used": False}

router = APIRouter()

# ── Trajectory Generator (LSTM-powered) ──────────────────────────────────────
_lstm_model   = None
_lstm_scaler  = None
_lstm_tmax    = None

def _load_lstm():
    global _lstm_model, _lstm_scaler, _lstm_tmax
    if _lstm_model is not None:
        return True
    try:
        import torch, joblib, re
        from models.train_lstm import TrajectoryLSTM

        SAVED = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'saved')
        model = TrajectoryLSTM()
        model.load_state_dict(torch.load(
            os.path.join(SAVED, "trajectory_lstm_best.pt"),
            map_location="cpu", weights_only=True
        ))
        model.eval()
        _lstm_model  = model
        _lstm_scaler = joblib.load(os.path.join(SAVED, "trajectory_scaler.joblib"))
        _lstm_tmax   = joblib.load(os.path.join(SAVED, "trajectory_target_max.joblib"))
        return True
    except Exception:
        return False


def generate_trajectory(confidence: float, follower_count: int, platform: str,
                        feature_vector: dict = None) -> list:
    """
    Generates a 4-point impression trajectory.
    Uses the trained LSTM when available; falls back to calibrated heuristic.
    """
    # ── Try LSTM path ──────────────────────────────────────────────
    if feature_vector and _load_lstm():
        try:
            import torch, numpy as _np, re
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _v = SentimentIntensityAnalyzer()
            text = feature_vector.get("caption", "")[:300]
            s = _v.polarity_scores(text)
            feats = _np.array([[
                s['compound'], max(0, s['compound']), abs(s['compound']),
                s['pos'], s['neg'],
                min(len(text), 300) / 300,
                float('?' in text), float('!' in text),
                min(text.count('!'), 5) / 5,
                float(bool(re.search(r'how to|tutorial', text, re.I))),
                float(bool(re.search(r'best|top|worst', text, re.I))),
                float(bool(re.search(r'\d+', text))),
                float(bool(re.search(r'you|your|we', text, re.I))),
                float(bool(re.search(r'secret|hack|never|always', text, re.I))),
                float(bool(re.search(r'shorts|short|quick', text, re.I))),
                float(bool(re.search(r'full|complete', text, re.I))),
                float(bool(re.search(r'vs|versus', text, re.I))),
                float(bool(re.search(r'new|first|exclusive', text, re.I))),
                min(len(re.findall(r'[A-Z]', text[:50])), 10) / 10,
                min(len(text.split()), 20) / 20,
            ]], dtype=_np.float32)

            X_scaled = _lstm_scaler.transform(feats)
            with torch.no_grad():
                pred_norm = _lstm_model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()[0]
            pred_log_views = pred_norm * _lstm_tmax
            raw_views = _np.expm1(pred_log_views)  # [d1, d3, d7, d10]

            # Scale by follower count context
            platform_mult = {
                "tiktok": 8.0, "instagram": 4.0, "youtube": 12.0,
                "twitter": 2.5, "linkedin": 1.8, "facebook": 2.0, "reddit": 3.0
            }.get(platform.lower(), 3.0)
            scale = max(0.1, follower_count / 10000) * platform_mult * confidence

            mids = [max(100, int(v * scale * 0.01)) for v in raw_views]
            spreads = [0.25, 0.30, 0.35, 0.40]
            return [
                TrajectoryPoint(
                    day=d,
                    low=max(0, int(m * (1 - sp))),
                    mid=m,
                    high=int(m * (1 + sp))
                )
                for d, m, sp in zip([1, 3, 7, 10], mids, spreads)
            ]
        except Exception:
            pass  # Fall through to heuristic

    # ── Heuristic fallback ─────────────────────────────────────────
    base_reach = follower_count * 0.1
    platform_mult = {
        "tiktok": 8.0, "instagram": 4.0, "youtube": 12.0,
        "twitter": 2.5, "linkedin": 1.8, "facebook": 2.0, "reddit": 3.0
    }.get(platform.lower(), 3.0)

    if confidence > 0.65:
        growth = confidence * platform_mult
        mids = [int(base_reach * 1.5), int(base_reach * growth * 2.5),
                int(base_reach * growth * 4.0), int(base_reach * growth * 3.5)]
    else:
        mids = [int(base_reach * 0.6), int(base_reach * 0.8),
                int(base_reach * 0.5), int(base_reach * 0.3)]

    spreads = [0.25, 0.30, 0.35, 0.40]
    return [
        TrajectoryPoint(
            day=d,
            low=max(0, int(m * (1 - sp))),
            mid=max(0, m),
            high=int(m * (1 + sp))
        )
        for d, m, sp in zip([1, 3, 7, 10], mids, spreads)
    ]



# ── Reach Percentile ─────────────────────────────────────────────────────────
def compute_reach_percentile(confidence: float, features: dict) -> int:
    """Map the composite score to a reach percentile vs similar posts on the platform."""
    base = confidence * 60  # 0-60 from confidence
    bonus = 0
    bonus += features.get("trending_hashtag_count", 0) * 5
    bonus += features.get("peak_overlap_score", 0) * 15
    bonus += features.get("face_count", 0) * 5
    bonus += features.get("cta_present", 0) * 8
    bonus += features.get("color_vibrancy", 0) * 7
    return min(99, max(1, int(base + bonus)))

# ── LightGBM Predictor ────────────────────────────────────────────────────────
def run_lgbm(feature_vector: dict) -> tuple:
    """
    Run LightGBM prediction. Falls back to heuristic if model not trained yet.
    Returns (prediction: str, confidence: float)
    """
    # Model priority: v5 raw (84MB, deployment-safe) > v5_cal (508MB, needs 1GB+ RAM)
    # > v4 > v3 > v2 > v1
    # NOTE: Use previral_lgbm_v5_cal.joblib on servers with 2GB+ RAM for best accuracy.
    SAVED = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'saved')
    def _pick(*names):
        for n in names:
            p = os.path.join(SAVED, n)
            if os.path.exists(p): return p
        return os.path.join(SAVED, names[-1])
    MODEL_PATH = _pick('previral_lgbm_v5.joblib', 'previral_lgbm_v5_cal.joblib',
                       'previral_lgbm_v4.joblib', 'previral_lgbm_v3.joblib',
                       'previral_lgbm.joblib')
    COLS_PATH  = _pick('feature_columns_v5.joblib', 'feature_columns_v4.joblib',
                       'feature_columns_v3.joblib', 'feature_columns.joblib')

    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        feature_cols = joblib.load(COLS_PATH)

        # Build platform one-hot features
        PLATFORMS = ['youtube', 'instagram', 'tiktok', 'twitter', 'linkedin',
                     'facebook', 'reddit', 'pinterest']
        plat = feature_vector.get('platform', '').lower()
        platform_feats = {f"platform_{p}": float(plat == p) for p in PLATFORMS}

        # Merge all feature sources
        full_vector = {**feature_vector, **platform_feats}

        # Align to exact training column order (fill 0 for any missing)
        X = np.array([[full_vector.get(col, 0.0) for col in feature_cols]])
        proba = model.predict_proba(X)[0]
        high_proba = float(proba[1])

        # 3-tier threshold calibrated on v4 validation (gap=0.600):
        # HIGH   > 0.60 — confident above-median performance
        # MEDIUM  0.35–0.60 — uncertain; worth optimizing before posting
        # LOW    < 0.35 — predicted below-median performance
        if high_proba >= 0.60:
            prediction = "HIGH"
        elif high_proba >= 0.35:
            prediction = "MEDIUM"
        else:
            prediction = "LOW"

        confidence = round(high_proba, 3)
        return prediction, confidence

    except Exception:
        # Heuristic fallback while model is being trained
        score = 0.0
        score += feature_vector.get("sentiment_score", 0) * 0.10
        score += feature_vector.get("emotional_valence", 0) * 0.15
        score += feature_vector.get("clickbait_score", 0) * 0.10
        score += feature_vector.get("cta_present", 0) * 0.10
        score += (1 - feature_vector.get("avg_competition_ratio", 0.7)) * 0.20
        score += feature_vector.get("trending_hashtag_count", 0) * 0.05
        score += feature_vector.get("peak_overlap_score", 0) * 0.15
        score += feature_vector.get("color_vibrancy", 0) * 0.05
        score += feature_vector.get("face_count", 0) * 0.05
        score += feature_vector.get("avg_engagement_rate", 0) * 0.05
        confidence = max(0.35, min(0.95, score))
        prediction = "HIGH" if confidence > 0.55 else "LOW"
        return prediction, round(confidence, 3)



# ── Main Analyze Endpoint ─────────────────────────────────────────────────────
@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_post(
    caption: str = Form(...),
    platform: str = Form(...),
    post_datetime: Optional[str] = Form(None),
    follower_count: Optional[int] = Form(1000),
    avg_engagement_rate: Optional[float] = Form(0.03),
    niche: Optional[str] = Form("tech"),
    vision_cache_id: Optional[str] = Form(None),
    media: Optional[UploadFile] = File(None)
):
    start_time = time.time()

    # Parse datetime
    try:
        dt = datetime.fromisoformat(post_datetime) if post_datetime else datetime.now()
    except Exception:
        dt = datetime.now()

    # Extract hashtags from caption
    hashtags = extract_hashtags(caption)

    # Check vision cache first (pre-computed during thumbnail upload)
    # This keeps CLIP off the critical 3-second analysis path
    cached_vision = get_cached_vision(vision_cache_id) if vision_cache_id else None

    # Read image bytes only if no cache hit
    image_bytes = None
    if not cached_vision and media and media.filename:
        image_bytes = await media.read()

    # Run all engines in parallel — Gemini runs alongside local NLP
    loop = asyncio.get_event_loop()
    nlp_task     = loop.run_in_executor(None, analyze_caption, caption)
    timing_task  = loop.run_in_executor(None, analyze_timing, platform, dt)
    hashtag_task = loop.run_in_executor(None, score_hashtags, hashtags, platform, caption)

    # Gemini multimodal NLP task — runs async so latency is hidden behind LightGBM
    if GEMINI_ENGINE_AVAILABLE:
        gemini_task = loop.run_in_executor(None, gemini_features, caption, platform, image_bytes)
    else:
        async def _no_gemini():
            return {}
        gemini_task = _no_gemini()

    # Gemini Search Grounding trend task — fully parallel, 5s hard timeout
    # Never blocks the main response — if grounding is slow, returns empty gracefully
    async def _trend_with_timeout():
        try:
            raw = loop.run_in_executor(None, suggest_trending_hashtags, caption, platform, niche or "general")
            return await asyncio.wait_for(raw, timeout=5.0)
        except asyncio.TimeoutError:
            return {"trending_now": [], "stable_performers": [], "avoid": [], "grounding_used": False, "_timeout": True}
        except Exception:
            return {"trending_now": [], "stable_performers": [], "avoid": [], "grounding_used": False}
    trend_task = _trend_with_timeout()

    if cached_vision:
        async def _cached_vision_task():
            return cached_vision
        vision_task = _cached_vision_task()
    elif image_bytes:
        vision_task = loop.run_in_executor(None, analyze_image, image_bytes, platform)
    else:
        async def _no_vision():
            return no_image_defaults()
        vision_task = _no_vision()

    nlp_features, timing_features, hashtag_features, vision_features, gemini_nlp, trending_data = await asyncio.gather(
        nlp_task, timing_task, hashtag_task, vision_task, gemini_task, trend_task
    )

    # Merge Gemini features into NLP (Gemini wins on shared keys if available)
    _gemini_extras = {k: v for k, v in gemini_nlp.items() if k.startswith('_gemini_')}
    _gemini_core   = {k: v for k, v in gemini_nlp.items() if not k.startswith('_gemini_')}
    if _gemini_core:
        nlp_features = {**nlp_features, **_gemini_core}
    nlp_features["gemini_hook_strength"]    = _gemini_extras.get("_gemini_hook_strength", 0.5)
    nlp_features["gemini_visual_alignment"] = _gemini_extras.get("_gemini_visual_alignment", 0.5)
    nlp_features["gemini_viral_potential"]  = _gemini_extras.get("_gemini_viral_potential", 0.5)
    nlp_features["gemini_used"]             = bool(_gemini_extras.get("_gemini_used", False))

    # Build unified feature vector
    feature_vector = {
        **nlp_features,
        **hashtag_features,
        **vision_features,
        **timing_features,
        "follower_count": follower_count / 1_000_000,  # Normalize
        "avg_engagement_rate": avg_engagement_rate
    }

    # Run LightGBM prediction
    prediction, confidence = run_lgbm(feature_vector)

    # Generate counterfactual suggestions
    raw_suggestions = generate_suggestions(feature_vector, prediction, platform)
    suggestions = [CounterfactualSuggestion(**s) for s in raw_suggestions]

    # Get hashtag suggestions
    raw_hashtags = suggest_hashtags(caption, platform, niche, top_k=10)
    hashtag_suggestions = [HashtagSuggestion(**h) for h in raw_hashtags]

    # Generate trajectory
    trajectory = generate_trajectory(
        confidence, follower_count, platform,
        feature_vector={**feature_vector, "caption": caption}
    )


    # Reach percentile
    reach_percentile = compute_reach_percentile(confidence, feature_vector)

    # Headline
    if prediction == "HIGH":
        headline = f"This post is predicted to outperform {reach_percentile}% of similar {platform} posts."
    else:
        headline = f"This post needs work — see {len(suggestions)} specific changes below to flip it to HIGH."

    processing_time = (time.time() - start_time) * 1000

    # Strip internal keys from trending data before sending to client
    clean_trending = {k: v for k, v in trending_data.items() if not k.startswith("_")}

    return AnalyzeResponse(
        prediction=prediction,
        confidence=round(confidence, 3),
        reach_percentile=reach_percentile,
        headline=headline,
        nlp_features=nlp_features,
        hashtag_features=hashtag_features,
        vision_features=vision_features,
        timing_features=timing_features,
        suggestions=suggestions,
        hashtag_suggestions=hashtag_suggestions,
        trajectory=trajectory,
        platform=platform,
        processing_time_ms=round(processing_time, 1),
        trending_hashtags=clean_trending
    )


# ── AI Content Director Endpoint ─────────────────────────────────────────────
@router.post("/ai-director")
async def ai_director(
    caption: str = Form(...),
    platform: str = Form(...),
    follower_count: Optional[int] = Form(1000),
    avg_engagement_rate: Optional[float] = Form(0.03),
    niche: Optional[str] = Form("tech"),
    media: Optional[UploadFile] = File(None)
):
    """
    AI Content Director — Gemini Pro Vision analyzes caption + thumbnail together.
    Returns: rewritten caption, visual-caption alignment, 3 specific improvements,
             thumbnail suggestion, hook rewrite, vocabulary, predicted score lift.
    This is PreViral's flagship Gemini-native feature for the XPRIZE submission.
    """
    start_time = time.time()
    dt = datetime.now()

    image_bytes = None
    if media and media.filename:
        image_bytes = await media.read()

    # Get current prediction score first
    loop = asyncio.get_event_loop()
    hashtags = extract_hashtags(caption)
    nlp_task     = loop.run_in_executor(None, analyze_caption, caption)
    timing_task  = loop.run_in_executor(None, analyze_timing, platform, dt)
    hashtag_task = loop.run_in_executor(None, score_hashtags, hashtags, platform, caption)
    nlp_f, timing_f, hashtag_f = await asyncio.gather(nlp_task, timing_task, hashtag_task)

    feature_vector = {
        **nlp_f, **hashtag_f, **timing_f, **no_image_defaults(),
        "follower_count": follower_count / 1_000_000,
        "avg_engagement_rate": avg_engagement_rate
    }
    prediction, confidence = run_lgbm(feature_vector)

    # Run Gemini AI Content Director
    if GEMINI_ENGINE_AVAILABLE:
        director_result = await loop.run_in_executor(
            None, ai_content_director,
            caption, platform, confidence, prediction, image_bytes
        )
    else:
        director_result = {
            "alignment_assessment": "Gemini API not configured. Set GEMINI_API_KEY env var.",
            "rewritten_caption": caption,
            "specific_improvements": [
                "Add a stronger hook in your opening line.",
                "Include a clear call-to-action (comment, share, follow).",
                "Use 3-5 niche hashtags your audience follows."
            ],
            "thumbnail_suggestion": "Add a human face to thumbnail for higher click-through.",
            "predicted_score_after": round(min(confidence + 0.12, 0.95), 3),
            "hook_rewrite": caption[:80].strip(),
            "best_posting_time": "Tuesday-Thursday, 11am-1pm local time",
            "vocabulary_suggestion": "Use emotionally charged, platform-native language.",
            "_gemini_used": False
        }

    return {
        "current_prediction": prediction,
        "current_confidence": round(confidence, 3),
        "platform": platform,
        "gemini_used": director_result.get("_gemini_used", False),
        **{k: v for k, v in director_result.items() if not k.startswith("_")},
        "processing_time_ms": round((time.time() - start_time) * 1000, 1)
    }


# ── Gemini Health Check ───────────────────────────────────────────────────────
@router.get("/gemini-status")
async def gemini_status():
    """Check if Gemini API is configured and responding."""
    if not GEMINI_ENGINE_AVAILABLE:
        return {"status": "unavailable", "reason": "google-generativeai not installed"}
    try:
        from engines.gemini_engine import health_check
        return {"status": "ok", **health_check()}
    except Exception as e:
        return {"status": "error", "error": str(e)}
