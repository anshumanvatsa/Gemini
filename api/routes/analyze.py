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

            # Scale by follower count context — calibrated to real impression ranges
            platform_mult = {
                "tiktok": 8.0, "instagram": 4.0, "youtube": 12.0,
                "twitter": 2.5, "linkedin": 1.8, "facebook": 2.0, "reddit": 3.0
            }.get(platform.lower(), 3.0)
            scale = max(0.1, follower_count / 10000) * platform_mult * confidence

            # Apply tier-shaped curve multipliers (same logic as heuristic)
            raw_normalized = [max(1.0, float(v * scale * 0.01)) for v in raw_views]
            max_raw = max(raw_normalized)

            if confidence > 0.65:
                curve_mult = [0.30, 0.55, 1.00, 0.85]  # growth spike
            elif confidence > 0.40:
                curve_mult = [0.40, 0.80, 0.85, 0.70]  # plateau
            else:
                curve_mult = [1.00, 0.75, 0.42, 0.21]  # decay

            base = max(follower_count * 0.08, 400)
            mids = [max(int(base * cm), int(base * cm)) for cm in curve_mult]
            # Scale up by model signal
            signal_boost = max(1.0, max_raw / 10.0)
            mids = [max(int(m * signal_boost), int(base * cm)) for m, cm in zip(mids, curve_mult)]

            spreads = [0.20, 0.25, 0.30, 0.35]
            return [
                TrajectoryPoint(
                    day=d,
                    low=max(0, int(m * (1 - sp))),
                    mid=max(1, m),
                    high=int(m * (1 + sp))
                )
                for d, m, sp in zip([1, 3, 7, 10], mids, spreads)
            ]
        except Exception:
            pass  # Fall through to heuristic

    # ── Heuristic fallback ─────────────────────────────────────────
    base_reach = max(follower_count * 0.1, 500)  # minimum 500 so chart never looks dead
    platform_mult = {
        "tiktok": 8.0, "instagram": 4.0, "youtube": 12.0,
        "twitter": 2.5, "linkedin": 1.8, "facebook": 2.0, "reddit": 3.0
    }.get(platform.lower(), 3.0)

    if confidence > 0.65:
        # HIGH — growth curve: slow start, big spike day 7, slight pullback day 10
        growth = confidence * platform_mult
        mids = [
            int(base_reach * 1.5),
            int(base_reach * growth * 2.5),
            int(base_reach * growth * 5.0),
            int(base_reach * growth * 4.2)
        ]
    elif confidence > 0.40:
        # MEDIUM — plateau: rises to day 3, holds, slight drop
        growth = confidence * platform_mult * 0.6
        mids = [
            int(base_reach * 1.0),
            int(base_reach * growth * 1.8),
            int(base_reach * growth * 1.9),
            int(base_reach * growth * 1.5)
        ]
    else:
        # LOW — decay curve: initial burst from followers, then drops off fast
        mids = [
            int(base_reach * 1.2),   # Day 1: initial follower exposure
            int(base_reach * 0.9),   # Day 3: algorithm doesn't pick it up
            int(base_reach * 0.5),   # Day 7: fades
            int(base_reach * 0.25)   # Day 10: near zero organic reach
        ]

    spreads = [0.20, 0.25, 0.30, 0.35]
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

    # ── Build 10-Day Summary Report ────────────────────────────────────────────
    traj_mids  = [p.mid  for p in trajectory]
    traj_highs = [p.high for p in trajectory]
    traj_lows  = [p.low  for p in trajectory]

    # Interpolate days between the 4 anchor points (days 1,3,7,10)
    # Weight: day1=1, day2-3=2 days, day4-7=4 days, day8-10=3 days
    weights     = [1, 2, 4, 3]  # day spans represented by each point
    total_mid   = sum(m * w for m, w in zip(traj_mids,  weights))
    total_best  = sum(h * w for h, w in zip(traj_highs, weights))
    total_worst = sum(l * w for l, w in zip(traj_lows,  weights))

    peak_day_map = {1: "Day 1", 3: "Day 3", 7: "Day 7", 10: "Day 10"}
    peak_idx     = traj_mids.index(max(traj_mids))
    peak_day_label = ["Day 1", "Day 3", "Day 7", "Day 10"][peak_idx]

    daily_reach_rate = round((total_mid / 10) / max(follower_count, 1) * 100, 1)

    if prediction == "HIGH":
        narrative = (
            f"Strong viral trajectory — algorithm pickup expected by {peak_day_label}. "
            f"Post in the next peak window and monitor first-hour engagement. "
            f"If it exceeds your average, boost immediately."
        )
        tier_label = "Strong Growth"
    elif prediction == "MEDIUM":
        narrative = (
            f"Moderate reach expected — algorithm will show this to your core audience "
            f"but organic spread will be limited. Apply the suggested changes to push into HIGH territory."
        )
        tier_label = "Steady Reach"
    else:
        narrative = (
            f"Low organic reach predicted — the algorithm is unlikely to distribute this beyond "
            f"your existing followers. Apply at least 2 of the 3 suggestions below before posting."
        )
        tier_label = "Limited Reach"

    def _fmt(n: int) -> str:
        if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
        if n >= 1_000:     return f"{n/1_000:.0f}K"
        return str(n)

    ten_day_summary = {
        "total_impressions_mid":   total_mid,
        "total_impressions_best":  total_best,
        "total_impressions_worst": total_worst,
        "total_mid_fmt":    _fmt(total_mid),
        "total_best_fmt":   _fmt(total_best),
        "total_worst_fmt":  _fmt(total_worst),
        "peak_day":         peak_day_label,
        "daily_reach_rate": daily_reach_rate,
        "tier_label":       tier_label,
        "narrative":        narrative,
        "platform":         platform,
        "follower_count":   follower_count,
    }

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
        trending_hashtags=clean_trending,
        ten_day_summary=ten_day_summary
    )


# ── AI Content Director Endpoint ───────────────────────────────────────────────
@router.post("/ai-director")
async def ai_director(
    caption: str = Form(...),
    platform: str = Form(...),
    follower_count: Optional[int] = Form(1000),
    avg_engagement_rate: Optional[float] = Form(0.03),
    niche: Optional[str] = Form("tech"),
    media: Optional[UploadFile] = File(None)
):

    loop = asyncio.get_event_loop()

    # ── Helper: score any caption text with LightGBM ──────────────────────────
    async def _score(cap: str) -> tuple[str, float]:
        """Return (prediction, confidence) for any caption string."""
        hashtags = extract_hashtags(cap)
        nlp_f, timing_f, hashtag_f = await asyncio.gather(
            loop.run_in_executor(None, analyze_caption, cap),
            loop.run_in_executor(None, analyze_timing, platform, dt),
            loop.run_in_executor(None, score_hashtags, hashtags, platform, cap),
        )
        fv = {
            **nlp_f, **hashtag_f, **timing_f, **no_image_defaults(),
            "follower_count": follower_count / 1_000_000,
            "avg_engagement_rate": avg_engagement_rate,
        }
        return run_lgbm(fv)

    # ── Score the original caption ─────────────────────────────────────────────
    orig_prediction, orig_confidence = await _score(caption)
    orig_score_pct = round(orig_confidence * 100)

    iteration_trail = [{"label": "Original", "score": orig_score_pct, "caption": caption}]
    best_result     = None
    best_score      = orig_confidence
    current_caption = caption
    current_score   = orig_confidence

    if not GEMINI_ENGINE_AVAILABLE:
        return {
            "current_prediction": orig_prediction,
            "current_confidence": round(orig_confidence, 3),
            "platform": platform,
            "gemini_used": False,
            "alignment_assessment": "Gemini API not configured. Set GEMINI_API_KEY env var.",
            "rewritten_caption": caption,
            "specific_improvements": [
                "Add a stronger hook in your opening line.",
                "Include a clear call-to-action (comment, share, follow).",
                "Use 3-5 niche hashtags your audience follows."
            ],
            "iteration_trail": iteration_trail,
            "processing_time_ms": round((time.time() - start_time) * 1000, 1),
        }

    # ── 2-iteration model-validated loop ──────────────────────────────────────
    for iteration in range(1, 3):
        score_pct = round(current_score * 100)

        # Call Gemini with the NUMERIC score so it knows exactly how far from HIGH
        result = await loop.run_in_executor(
            None, ai_content_director,
            current_caption, platform, current_score, orig_prediction, image_bytes
        )

        # Extract the rewritten caption from Gemini's output
        rewritten = (
            result.get("rewritten_caption")          # fixer mode
            or current_caption                        # optimizer mode (no full rewrite)
        )

        # Score the rewrite with LightGBM
        iter_prediction, iter_confidence = await _score(rewritten)
        iter_score_pct = round(iter_confidence * 100)

        iteration_trail.append({
            "label": f"Iteration {iteration}",
            "score": iter_score_pct,
            "caption": rewritten,
            "prediction": iter_prediction,
        })

        # Track best — keep whichever iteration scores highest
        if iter_confidence > best_score:
            best_score  = iter_confidence
            best_result = result
            best_result["rewritten_caption"] = rewritten
            best_result["_iter_best"]        = iteration
            best_result["_iter_score"]       = iter_score_pct

        # Feed the rewrite into the next iteration
        current_caption = rewritten
        current_score   = iter_confidence

        # Early exit if already HIGH — no point iterating further
        if iter_prediction == "HIGH" and iter_confidence >= 0.75:
            break

    # Fallback if both iterations scored lower than original
    if best_result is None:
        best_result = await loop.run_in_executor(
            None, ai_content_director,
            caption, platform, orig_confidence, orig_prediction, image_bytes
        )

    return {
        "current_prediction":  orig_prediction,
        "current_confidence":  round(orig_confidence, 3),
        "platform":            platform,
        "gemini_used":         best_result.get("_gemini_used", False),
        "iteration_trail":     iteration_trail,
        "best_iteration":      best_result.get("_iter_best", 1),
        "best_score_pct":      best_result.get("_iter_score", orig_score_pct),
        **{k: v for k, v in best_result.items() if not k.startswith("_")},
        "processing_time_ms":  round((time.time() - start_time) * 1000, 1),
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
