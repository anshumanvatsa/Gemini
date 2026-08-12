"""
engines/gemini_engine.py
─────────────────────────
Gemini-powered multimodal analysis for PreViral.
Uses the NEW google-genai SDK (google.generativeai is deprecated).

Replaces VADER + RoBERTa + clickbait scorer with a single Gemini call
that reads caption + thumbnail TOGETHER, the way an audience would.

Powers two features:
  1. Enriched NLP features (feeds LightGBM — same schema as nlp_engine.py)
  2. AI Content Director (flagship Gemini-native hackathon feature)
"""

import os
import json
import base64
import re
import time
from typing import Optional

# ── New SDK (google-genai package) ────────────────────────────────────────────
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
# Use gemini-flash-latest — confirmed working on free API keys
# (gemini-2.5-flash is restricted for new API keys as of Aug 2026)
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# ── Default features (fallback when Gemini unavailable) ──────────────────────
DEFAULT_FEATURES = {
    "sentiment_score":    0.0,
    "emotional_valence":  0.5,
    "emotional_arousal":  0.3,
    "clickbait_score":    0.0,
    "cta_present":        0,
    "readability_grade":  0.5,
    "question_count":     0,
    "exclamation_count":  0,
    "has_url":            0,
    "emoji_count":        0,
    "caps_ratio":         0.0,
    "avg_word_length":    5.0,
    "unique_word_ratio":  0.7,
    "text_length":        100,
    "hashtag_count_nlp":  0,
    "mention_count":      0,
}

ANALYSIS_PROMPT = """You are an expert social media analyst. Analyze this social media caption (and thumbnail if provided).

Platform: {platform}
Caption: {caption}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "sentiment_score": <float -1.0 to 1.0, overall emotional tone>,
  "emotional_valence": <float 0.0 to 1.0, positivity strength>,
  "emotional_arousal": <float 0.0 to 1.0, excitement intensity>,
  "clickbait_score": <float 0.0 to 1.0, clickbait level>,
  "cta_present": <0 or 1, clear call-to-action present>,
  "readability_grade": <float 0.0 to 1.0, 1.0=very easy>,
  "question_count": <int>,
  "exclamation_count": <int>,
  "has_url": <0 or 1>,
  "emoji_count": <int>,
  "caps_ratio": <float 0.0 to 1.0>,
  "avg_word_length": <float, mean chars per word>,
  "unique_word_ratio": <float 0.0 to 1.0, lexical diversity>,
  "text_length": <int, total characters>,
  "hashtag_count_nlp": <int, # hashtags in caption>,
  "mention_count": <int, @ mentions>,
  "hook_strength": <float 0.0 to 1.0, opening hook quality>,
  "visual_caption_alignment": <float 0.0 to 1.0, 0.5 if no image provided>,
  "niche_relevance": <float 0.0 to 1.0, niche-specific vocabulary>,
  "viral_potential": <float 0.0 to 1.0, your overall viral assessment>
}}"""

DIRECTOR_PROMPT = """You are PreViral's AI Content Director. Help this creator maximize engagement.

Platform: {platform}
Current Caption: {caption}
Current Score: {current_score}/1.0 ({tier})
{visual_note}

Return ONLY valid JSON (no markdown):
{{
  "alignment_assessment": "<2 sentences: what works and what's misaligned>",
  "rewritten_caption": "<improved version preserving creator's voice>",
  "specific_improvements": ["<improvement 1>", "<improvement 2>", "<improvement 3>"],
  "thumbnail_suggestion": "<one specific change to increase click-through, or 'Thumbnail looks strong'>",
  "predicted_score_after": <float 0.0-1.0, estimated score after applying suggestions>,
  "hook_rewrite": "<just the opening line, rewritten stronger>",
  "best_posting_time": "<specific day and time window for {platform}>",
  "vocabulary_suggestion": "<3-5 trending words your audience uses on {platform}>"
}}"""


def _get_client():
    """Return a configured Gemini client, or None if unavailable."""
    if not GEMINI_AVAILABLE:
        return None
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        return None
    return genai.Client(api_key=key)


def _clean_json(raw: str) -> str:
    """Strip markdown fences from Gemini response."""
    raw = raw.strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'^```\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return raw.strip()


def _build_contents(prompt: str, image_bytes: Optional[bytes] = None) -> list:
    """Build content parts list for Gemini request."""
    parts = []
    if image_bytes:
        parts.append(types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg"
        ))
    parts.append(types.Part.from_text(text=prompt))
    return [types.Content(role="user", parts=parts)]


def extract_features(
    caption: str,
    platform: str = "instagram",
    image_bytes: Optional[bytes] = None,
) -> dict:
    """
    Extract NLP features via Gemini multimodal.
    Drop-in replacement for nlp_engine.analyze_caption().
    Falls back to local NLP engine if Gemini unavailable.
    """
    client = _get_client()

    if client is None:
        try:
            from engines.nlp_engine import analyze_caption
            feats = analyze_caption(caption)
            feats["_gemini_used"] = False
            return feats
        except Exception:
            return {**DEFAULT_FEATURES, "text_length": len(caption), "_gemini_used": False}

    prompt = ANALYSIS_PROMPT.format(caption=caption[:2000], platform=platform)

    try:
        t0 = time.time()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=_build_contents(prompt, image_bytes),
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=512,
            )
        )
        elapsed_ms = int((time.time() - t0) * 1000)

        data = json.loads(_clean_json(response.text))

        features = {**DEFAULT_FEATURES}
        for k in DEFAULT_FEATURES:
            if k in data:
                features[k] = data[k]

        # Gemini-only extras (not fed to LightGBM, used by AI Director)
        features["_gemini_hook_strength"]      = float(data.get("hook_strength", 0.5))
        features["_gemini_visual_alignment"]   = float(data.get("visual_caption_alignment", 0.5))
        features["_gemini_niche_relevance"]    = float(data.get("niche_relevance", 0.5))
        features["_gemini_viral_potential"]    = float(data.get("viral_potential", 0.5))
        features["_gemini_latency_ms"]         = elapsed_ms
        features["_gemini_used"]               = True
        return features

    except json.JSONDecodeError:
        return {**DEFAULT_FEATURES, "text_length": len(caption), "_gemini_used": False}
    except Exception as e:
        try:
            # Gemini failed — fall back to local NLP silently
            from engines.nlp_engine import analyze_caption
            feats = analyze_caption(caption)
            feats["_gemini_used"] = False
            feats["_gemini_error"] = str(e)[:80]
            return feats
        except Exception:
            return {**DEFAULT_FEATURES, "text_length": len(caption),
                    "_gemini_used": False, "_gemini_error": str(e)[:80]}


def ai_content_director(
    caption: str,
    platform: str,
    current_score: float,
    tier: str,
    image_bytes: Optional[bytes] = None,
) -> dict:
    """
    AI Content Director — PreViral's flagship Gemini feature.
    Analyzes caption + thumbnail together and returns a complete content upgrade.
    """
    client = _get_client()
    fallback = {
        "alignment_assessment": "Add your GEMINI_API_KEY to enable AI Content Director.",
        "rewritten_caption": caption,
        "specific_improvements": [
            "Add a stronger hook in your opening line.",
            "Include a clear call-to-action (comment, share, follow).",
            "Use 3-5 niche-specific hashtags.",
        ],
        "thumbnail_suggestion": "Add a human face for higher click-through rate.",
        "predicted_score_after": round(min(current_score + 0.12, 0.95), 3),
        "hook_rewrite": caption[:80].strip(),
        "best_posting_time": "Tuesday–Thursday, 11am–1pm local time",
        "vocabulary_suggestion": "Use emotionally charged, platform-native language.",
        "_gemini_used": False,
    }

    if client is None:
        return fallback

    visual_note = (
        "A thumbnail image is provided — analyze visual-caption alignment carefully."
        if image_bytes else
        "No thumbnail provided — focus on caption only."
    )

    prompt = DIRECTOR_PROMPT.format(
        platform=platform,
        caption=caption[:2000],
        current_score=f"{current_score:.2f}",
        tier=tier,
        visual_note=visual_note,
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=_build_contents(prompt, image_bytes),
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=1024,
            )
        )
        result = json.loads(_clean_json(response.text))
        result["_gemini_used"] = True
        return result
    except Exception as e:
        fallback["_gemini_error"] = str(e)[:100]
        return fallback


def health_check() -> dict:
    """Quick liveness check for the /gemini-status endpoint."""
    client = _get_client()
    if client is None:
        key_set = bool(os.getenv("GEMINI_API_KEY", ""))
        return {
            "gemini_available": False,
            "reason": "API key not set" if not key_set else "google-genai not installed",
            "sdk": "google-genai (new)",
        }
    try:
        r = client.models.generate_content(
            model=MODEL_NAME,
            contents="Reply with exactly: PREVIRAL_OK",
            config=types.GenerateContentConfig(max_output_tokens=10)
        )
        return {
            "gemini_available": True,
            "model": MODEL_NAME,
            "sdk": "google-genai (new)",
            "response": r.text.strip(),
        }
    except Exception as e:
        return {"gemini_available": False, "error": str(e)[:120], "sdk": "google-genai (new)"}
