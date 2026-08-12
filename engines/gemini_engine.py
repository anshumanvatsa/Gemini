"""
engines/gemini_engine.py
─────────────────────────
Gemini-powered multimodal analysis for PreViral.
Replaces VADER + RoBERTa + clickbait with a single Gemini Pro Vision call
that reads caption + thumbnail TOGETHER, the way an audience would.

Outputs a structured JSON with 8 NLP features that feed directly into
the LightGBM feature vector (same schema as nlp_engine.py).

Also powers the AI Content Director:
  - Visual-caption alignment score
  - Rewritten caption (preserves user voice)
  - Thumbnail composition suggestion
  - Niche-specific vocabulary recommendation
"""

import os
import json
import base64
import re
import time
import hashlib
from functools import lru_cache
from typing import Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME     = "gemini-2.0-flash"   # fast, cheap, multimodal — perfect for live demo

# ── Schema returned by Gemini (maps 1:1 to nlp_engine feature names) ──────────
DEFAULT_FEATURES = {
    "sentiment_score":    0.0,    # -1 to 1
    "emotional_valence":  0.5,    # 0 to 1
    "emotional_arousal":  0.3,    # 0 to 1
    "clickbait_score":    0.0,    # 0 to 1
    "cta_present":        0,      # 0 or 1
    "readability_grade":  0.5,    # 0 to 1 (normalized)
    "question_count":     0,      # raw count
    "exclamation_count":  0,      # raw count
    "has_url":            0,      # 0 or 1
    "emoji_count":        0,      # raw count
    "caps_ratio":         0.0,    # 0 to 1
    "avg_word_length":    5.0,    # chars
    "unique_word_ratio":  0.7,    # 0 to 1
    "text_length":        100,    # chars
    "hashtag_count_nlp":  0,      # from text
    "mention_count":      0,      # from text
}

ANALYSIS_PROMPT = """You are an expert social media analyst. Analyze the following social media post caption (and thumbnail image if provided).

Your task: return a JSON object with these exact keys and value ranges. Be precise — these values feed a machine learning model.

Caption:
{caption}

Platform: {platform}

Return ONLY valid JSON (no markdown, no explanation), with these keys:
{{
  "sentiment_score": <float -1.0 to 1.0, overall emotional tone>,
  "emotional_valence": <float 0.0 to 1.0, positivity strength>,
  "emotional_arousal": <float 0.0 to 1.0, excitement/intensity>,
  "clickbait_score": <float 0.0 to 1.0, how clickbait-y the hook is>,
  "cta_present": <0 or 1, has clear call-to-action>,
  "readability_grade": <float 0.0 to 1.0, 1.0=very easy to read>,
  "question_count": <int, number of questions>,
  "exclamation_count": <int, number of exclamation marks>,
  "has_url": <0 or 1>,
  "emoji_count": <int>,
  "caps_ratio": <float 0.0 to 1.0, proportion of uppercase letters>,
  "avg_word_length": <float, mean word length in chars>,
  "unique_word_ratio": <float 0.0 to 1.0, lexical diversity>,
  "text_length": <int, total characters>,
  "hashtag_count_nlp": <int, hashtags in caption>,
  "mention_count": <int, @ mentions>,
  "hook_strength": <float 0.0 to 1.0, how strong the opening hook is>,
  "visual_caption_alignment": <float 0.0 to 1.0, how well thumbnail matches caption, 0.5 if no image>,
  "niche_relevance": <float 0.0 to 1.0, how niche-specific the vocabulary is>,
  "viral_potential": <float 0.0 to 1.0, your overall assessment of viral potential>
}}"""

CONTENT_DIRECTOR_PROMPT = """You are PreViral's AI Content Director. A creator wants to maximize their post's engagement.

Platform: {platform}
Current Caption: {caption}
Current Score: {current_score} / 1.0 ({tier} tier)

{visual_instruction}

Analyze their content holistically and return ONLY valid JSON (no markdown):
{{
  "alignment_assessment": "<2 sentences: what's working and what's misaligned between visual and caption>",
  "rewritten_caption": "<rewritten version that fixes weaknesses while preserving creator's voice — max 280 chars for Twitter, 2200 for Instagram>",
  "specific_improvements": [
    "<concrete improvement 1>",
    "<concrete improvement 2>",
    "<concrete improvement 3>"
  ],
  "thumbnail_suggestion": "<one specific thumbnail composition change that would increase click-through, or 'Current thumbnail is strong' if good>",
  "predicted_score_after": <float 0.0 to 1.0, estimated new viral potential after applying suggestions>,
  "hook_rewrite": "<just the first sentence/hook, rewritten to be stronger>",
  "best_posting_time": "<day and time window recommendation for {platform}>",
  "vocabulary_suggestion": "<3-5 trending words or phrases your target audience uses on {platform}>"
}}"""


def _init_client():
    """Initialize Gemini client. Returns None if API key not set."""
    if not GEMINI_AVAILABLE:
        return None
    key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
    if not key:
        return None
    genai.configure(api_key=key)
    return genai.GenerativeModel(MODEL_NAME)


def _image_to_part(image_bytes: bytes):
    """Convert raw image bytes to Gemini Part."""
    import google.generativeai as genai
    return {
        "mime_type": "image/jpeg",
        "data": base64.b64encode(image_bytes).decode()
    }


def extract_features(
    caption: str,
    platform: str = "instagram",
    image_bytes: Optional[bytes] = None,
    timeout: int = 10
) -> dict:
    """
    Extract NLP features using Gemini (with fallback to nlp_engine.py).

    Returns dict with same keys as nlp_engine.py extract_features().
    Designed to be a drop-in replacement.
    """
    client = _init_client()

    # ── Fallback to local NLP if Gemini unavailable ──────────────────────────
    if client is None:
        try:
            from engines.nlp_engine import NLPEngine
            nlp = NLPEngine()
            return nlp.extract_features(caption)
        except Exception:
            return {**DEFAULT_FEATURES, "text_length": len(caption)}

    # ── Build Gemini request ──────────────────────────────────────────────────
    prompt = ANALYSIS_PROMPT.format(caption=caption[:2000], platform=platform)
    parts = [prompt]
    if image_bytes:
        parts.insert(0, _image_to_part(image_bytes))  # image first for better attention

    try:
        start = time.time()
        response = client.generate_content(
            parts,
            generation_config={"temperature": 0.1, "max_output_tokens": 512}
        )
        elapsed = time.time() - start

        raw = response.text.strip()
        # Strip any accidental markdown
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        data = json.loads(raw)
        features = {**DEFAULT_FEATURES}

        for k in DEFAULT_FEATURES:
            if k in data:
                features[k] = data[k]

        # Store Gemini-only extras (for AI Content Director, not for LightGBM)
        features["_gemini_hook_strength"]          = float(data.get("hook_strength", 0.5))
        features["_gemini_visual_alignment"]       = float(data.get("visual_caption_alignment", 0.5))
        features["_gemini_niche_relevance"]        = float(data.get("niche_relevance", 0.5))
        features["_gemini_viral_potential"]        = float(data.get("viral_potential", 0.5))
        features["_gemini_latency_ms"]             = int(elapsed * 1000)
        features["_gemini_used"]                   = True

        return features

    except json.JSONDecodeError:
        # Gemini returned non-JSON — fallback gracefully
        features = {**DEFAULT_FEATURES, "text_length": len(caption), "_gemini_used": False}
        return features
    except Exception as e:
        features = {**DEFAULT_FEATURES, "text_length": len(caption),
                    "_gemini_used": False, "_gemini_error": str(e)}
        return features


def ai_content_director(
    caption: str,
    platform: str,
    current_score: float,
    tier: str,
    image_bytes: Optional[bytes] = None
) -> dict:
    """
    The AI Content Director — PreViral's flagship Gemini-powered feature.

    Takes a caption + thumbnail and returns:
    - Visual-caption alignment assessment
    - Rewritten caption (preserves user voice)
    - 3 specific improvements
    - Thumbnail composition suggestion
    - Predicted score after improvements
    - Hook rewrite
    - Best posting time
    - Vocabulary suggestions
    """
    client = _init_client()
    if client is None:
        return {
            "alignment_assessment": "Gemini API key not configured.",
            "rewritten_caption": caption,
            "specific_improvements": [
                "Add a stronger hook in the first sentence.",
                "Include a clear call-to-action.",
                "Use 3–5 niche-specific hashtags."
            ],
            "thumbnail_suggestion": "Add a human face to the frame.",
            "predicted_score_after": min(current_score + 0.12, 0.95),
            "hook_rewrite": caption[:60],
            "best_posting_time": "Tuesday–Thursday, 11am–1pm",
            "vocabulary_suggestion": "Use platform-native trending vocabulary.",
            "_gemini_used": False
        }

    visual_instruction = (
        "A thumbnail image has been provided. Analyze visual-caption alignment carefully."
        if image_bytes else
        "No thumbnail provided. Skip visual analysis, focus on caption only."
    )

    prompt = CONTENT_DIRECTOR_PROMPT.format(
        platform=platform,
        caption=caption[:2000],
        current_score=f"{current_score:.2f}",
        tier=tier,
        visual_instruction=visual_instruction
    )

    parts = [prompt]
    if image_bytes:
        parts.insert(0, _image_to_part(image_bytes))

    try:
        response = client.generate_content(
            parts,
            generation_config={"temperature": 0.4, "max_output_tokens": 1024}
        )
        raw = response.text.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        result = json.loads(raw)
        result["_gemini_used"] = True
        return result
    except Exception as e:
        return {
            "alignment_assessment": f"Analysis temporarily unavailable: {str(e)[:60]}",
            "rewritten_caption": caption,
            "specific_improvements": [
                "Add a curiosity hook to your opening line.",
                "Include a direct call-to-action.",
                "Add 3 niche hashtags relevant to your audience."
            ],
            "thumbnail_suggestion": "Ensure your thumbnail has a clear focal point and human element.",
            "predicted_score_after": min(current_score + 0.10, 0.95),
            "hook_rewrite": caption[:80].strip() + "...",
            "best_posting_time": "Tuesday–Thursday, 11am–1pm",
            "vocabulary_suggestion": "Use emotional, active language specific to your niche.",
            "_gemini_used": False,
            "_error": str(e)
        }


def health_check() -> dict:
    """Quick health check — returns Gemini availability status."""
    client = _init_client()
    if client is None:
        return {"gemini_available": False, "reason": "API key not set or library missing"}
    try:
        r = client.generate_content("Reply with: OK",
            generation_config={"max_output_tokens": 5})
        return {"gemini_available": True, "model": MODEL_NAME, "response": r.text.strip()}
    except Exception as e:
        return {"gemini_available": False, "error": str(e)}
