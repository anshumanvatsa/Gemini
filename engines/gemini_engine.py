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

# ── Load .env automatically ───────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass  # python-dotenv not installed — rely on env vars set by shell

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
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")

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

DIRECTOR_FIXER_PROMPT = """You are PreViral's AI Content Director. You receive BOTH the caption text AND the thumbnail image in this single multimodal call. Analyze them together.

Platform: {platform}
Caption: {caption}
LightGBM Score: {score_pct} out of 100 — needs to reach 75+ to be HIGH
{visual_note}

Diagnosis from our ML model (DICE-ML counterfactuals — these are the SPECIFIC issues to fix):
{directives_section}

Your rewrite must address the diagnosed issues above. Be specific, not generic.
Return ONLY valid JSON (no markdown, keep all string values SHORT — max 120 chars each):
{{
  "alignment_assessment": "<2 short sentences: how well caption and thumbnail work together>",
  "rewritten_caption": "<improved version — preserves creator voice, adds hook+CTA+hashtags, fixes diagnosed issues>",
  "specific_improvements": ["<fix 1 addressing a diagnosed issue>", "<fix 2>", "<fix 3>"],
  "thumbnail_suggestion": "<one specific visual change based on multimodal analysis>",
  "predicted_score_after": <float 0.0-1.0>,
  "hook_rewrite": "<just the new opening line>",
  "best_posting_time": "<specific day + time for {platform}>",
  "vocabulary_suggestion": "<3-4 trending words for this niche on {platform}>",
  "mode": "fixer"
}}"""

DIRECTOR_OPTIMIZER_PROMPT = """You are PreViral's AI Content Director running in OPTIMIZER MODE.
You receive BOTH the caption text AND the thumbnail image in this single multimodal call. Analyze them together.
This caption already scores HIGH. Your job is NOT to rewrite it — find the marginal gains
that platform algorithm intelligence reveals.

Platform: {platform}
Caption: {caption}
LightGBM Score: {score_pct} out of 100 (already HIGH — push toward 95+)
{visual_note}

Fine-tuning signals from our ML model:
{directives_section}

Deliver what Claude, ChatGPT, and other AIs CANNOT:
1. A/B hook variants — 3 alternative opening lines (curiosity / authority / story angles)
2. Micro-improvements — 3 surgical word-level tweaks (not a full rewrite)
3. Hashtag velocity — which to ADD (rising) and which to REMOVE (dying/oversaturated)
4. Algorithm timing edge — exact window on {platform} where THIS content type peaks
5. Competitive gap — what top 1% posts in this niche do that this caption still lacks

Return ONLY valid JSON (no markdown, keep all string values SHORT — max 150 chars each):
{{
  "alignment_assessment": "<1-2 sentences: what is already excellent, and how thumbnail and caption work together>",
  "hook_variants": [
    {{"variant": "<hook option A — curiosity angle>", "angle": "Curiosity"}},
    {{"variant": "<hook option B — authority angle>", "angle": "Authority"}},
    {{"variant": "<hook option C — story angle>", "angle": "Story"}}
  ],
  "micro_improvements": ["<surgical tweak 1>", "<surgical tweak 2>", "<surgical tweak 3>"],
  "hashtags_to_add": ["#tag1", "#tag2", "#tag3"],
  "hashtags_to_remove": ["#tag1", "#tag2"],
  "timing_edge": "<exact day + hour + why this content type peaks then on {platform}>",
  "competitive_gap": "<the one thing top 1% posts do that this caption still lacks>",
  "predicted_score_after": <float 0.0-1.0>,
  "vocabulary_suggestion": "<3-4 words currently trending in this niche on {platform}>",
  "mode": "optimizer"
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


def _repair_json(raw: str) -> str:
    """
    Best-effort repair of truncated/malformed JSON from LLM responses.
    Handles trailing commas, unclosed brackets, and other common issues.
    """
    raw = raw.strip()
    # Remove trailing comma before closing brace/bracket
    raw = re.sub(r',\s*([}\]])', r'\1', raw)
    # Try to close unclosed structures
    opens  = raw.count('{') - raw.count('}')
    aopens = raw.count('[') - raw.count(']')
    if opens > 0 or aopens > 0:
        raw = raw.rstrip().rstrip(',')
        raw += ']' * aopens + '}' * opens
    return raw


def _build_contents(prompt: str, image_bytes: Optional[bytes] = None) -> list:
    """
    Build a single multimodal Gemini request containing BOTH image and text.
    This is a genuine one-call multimodal fusion — not two separate API calls.
    Gemini receives image bytes + caption text simultaneously in one request,
    enabling cross-modal analysis that neither text-only nor image-only can produce.
    """
    parts = []
    if image_bytes:
        parts.append(types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg"
        ))
    parts.append(types.Part.from_text(text=prompt))
    return [types.Content(role="user", parts=parts)]


def _get_text(response) -> str:
    """
    Extract text from Gemini response robustly.
    gemini-flash-latest is a thinking model — finish_reason=MAX_TOKENS is common
    because thinking tokens eat the budget before the actual answer.
    r.text returns None in that case, so we read from candidates directly.
    """
    # Try r.text first (works when finish_reason=STOP)
    if response.text is not None:
        return response.text
    # Fall back: read text parts from candidates, skipping thought_signature parts
    try:
        for candidate in (response.candidates or []):
            for part in (candidate.content.parts or []):
                if hasattr(part, 'text') and part.text:
                    return part.text
    except Exception:
        pass
    return ""


def extract_features(
    caption: str,
    platform: str = "instagram",
    image_bytes: Optional[bytes] = None,
    _unused: None = None,
    content_type: Optional[str] = None,
    visual_description: Optional[str] = None,
) -> dict:
    """
    Extract NLP features via Gemini multimodal.
    - image_bytes: actual image upload (highest priority)
    - visual_description: text fallback when creator doesn't upload image
    - content_type: platform content type (reel, tweet, etc.) for context
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

    # Build context-aware prompt
    visual_context = ""
    if visual_description:
        visual_context = f"\nVisual Description (creator described their image): {visual_description}"
    if content_type:
        visual_context += f"\nContent Type: {content_type}"

    prompt = ANALYSIS_PROMPT.format(
        caption=caption[:2000] + visual_context,
        platform=platform
    )

    try:
        t0 = time.time()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=_build_contents(prompt, image_bytes),
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=6144,  # thinking model needs headroom before producing JSON
            )
        )
        elapsed_ms = int((time.time() - t0) * 1000)

        data = json.loads(_clean_json(_get_text(response)))

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
    directives: Optional[str] = None,
) -> dict:
    """
    AI Content Director — two modes:
    FIXER (LOW/MEDIUM): Full rewrite + 3 improvements
    OPTIMIZER (HIGH): A/B hook variants + micro-tweaks + hashtag velocity + timing edge
    The OPTIMIZER is PreViral's differentiator vs Claude/GPT — platform algorithm intelligence.
    """
    client = _get_client()
    is_high = current_score >= 0.60 or tier == "HIGH"
    score_pct = round(current_score * 100)

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
        "mode": "optimizer" if is_high else "fixer",
        "_gemini_used": False,
    }

    if client is None:
        return fallback

    visual_note = (
        "A thumbnail image is provided — analyze visual-caption alignment carefully."
        if image_bytes else
        "No thumbnail provided — focus on caption only."
    )

    # Format DICE-ML directives for the prompt
    if directives and directives.strip():
        directives_section = directives.strip()
    else:
        directives_section = "No specific directives provided — use your own analysis."

    # Choose prompt based on confidence tier
    if is_high:
        prompt = DIRECTOR_OPTIMIZER_PROMPT.format(
            platform=platform,
            caption=caption[:2000],
            score_pct=score_pct,
            visual_note=visual_note,
            directives_section=directives_section,
        )
    else:
        prompt = DIRECTOR_FIXER_PROMPT.format(
            platform=platform,
            caption=caption[:2000],
            score_pct=score_pct,
            visual_note=visual_note,
            directives_section=directives_section,
        )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=_build_contents(prompt, image_bytes),
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=8192,  # thinking model needs headroom for reasoning tokens
            )
        )
        raw_text = _get_text(response) or ""
        if not raw_text:
            # Thinking model ran out of tokens before producing output
            finish = None
            try:
                finish = response.candidates[0].finish_reason
            except Exception:
                pass
            raise ValueError(f"Empty response from Gemini (finish_reason={finish}). Thinking tokens may have exhausted the budget.")
        cleaned  = _clean_json(raw_text)
        repaired = _repair_json(cleaned)
        result   = json.loads(repaired)
        result["_gemini_used"] = True
        result["mode"]         = "optimizer" if is_high else "fixer"
        return result
    except Exception as e:
        err_str = str(e)
        import sys
        print(f"[AI Director ERROR] {err_str[:200]}", file=sys.stderr)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            fallback["alignment_assessment"] = (
                "Gemini quota reached for today. Your content analysis and score are still running. "
                "AI rewrite will be available after your quota resets (usually midnight US Pacific time)."
            )
            fallback["_quota_exhausted"] = True
        elif "API_KEY" in err_str or "api key" in err_str.lower():
            fallback["alignment_assessment"] = "Add your GEMINI_API_KEY to enable AI Content Director."
        else:
            fallback["alignment_assessment"] = f"Gemini temporarily unavailable — your score analysis still ran."
        fallback["_gemini_error"] = err_str[:100]
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
            config=types.GenerateContentConfig(max_output_tokens=128)
        )
        return {
            "gemini_available": True,
            "model": MODEL_NAME,
            "sdk": "google-genai (new)",
            "response": _get_text(r).strip(),
        }
    except Exception as e:
        return {"gemini_available": False, "error": str(e)[:120], "sdk": "google-genai (new)"}


# ── Trend Intelligence via Direct Gemini ─────────
TREND_PROMPT = """You are a hashtag strategist.
Platform: {platform} | Niche: {niche} | Topic: {topic}
Generate 15 hashtags.
Return JSON: {"trending_now": [{"tag": "#...", "why": "...", "velocity": "high"}], "stable_performers": [{"tag": "#...", "why": "..."}], "avoid": [{"tag": "#...", "reason": "..."}], "topic_detected": "..."}"""


def suggest_trending_hashtags(
    caption: str,
    platform: str = "instagram",
    niche: str = "general",
) -> dict:
    """
    Fast Gemini hashtag intelligence.
    """
    fallback = {
        "trending_now": [],
        "stable_performers": [
            {"tag": f"#{niche}", "why": "Core niche tag"},
            {"tag": f"#{platform}creator", "why": "Platform creator community"},
        ],
        "avoid": [
            {"tag": "#viral", "reason": "Too generic"},
        ],
        "topic_detected": niche,
        "_gemini_used": False,
    }

    client = _get_client()
    if client is None:
        return fallback

    topic_hint = caption[:300].strip()
    prompt = TREND_PROMPT.format(platform=platform, niche=niche, topic=topic_hint)

    try:
        t0 = time.time()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=4096,
            )
        )
        elapsed_ms = int((time.time() - t0) * 1000)

        raw = _get_text(response) or ""
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if not json_match:
            fallback["_error"] = "No JSON in response"
            return fallback

        repaired = _repair_json(json_match.group(0))
        data = json.loads(repaired)
        data["_gemini_used"] = True
        data["_latency_ms"] = elapsed_ms
        return data

    except Exception as e:
        fallback["_error"] = str(e)[:120]
        fallback["_gemini_used"] = False
        return fallback
