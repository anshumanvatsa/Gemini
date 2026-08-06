"""
NLP Engine — PreViral
Runs VADER + RoBERTa + Clickbait Scorer + CTA Detector on caption text.
Returns 6 features: sentiment_score, emotional_valence, emotional_arousal,
clickbait_score, cta_present, readability_grade
"""
import re
import math
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Lazy-load heavy models so the API starts fast
_roberta_pipeline = None

def _get_roberta():
    global _roberta_pipeline
    if _roberta_pipeline is None:
        try:
            from transformers import pipeline
            _roberta_pipeline = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment",
                return_all_scores=True
            )
        except Exception as e:
            # Graceful fallback — RoBERTa requires PyTorch >= 2.6 with safetensors.
            # VADER provides the fallback sentiment scores in analyze_caption().
            print(f"[NLP Engine] RoBERTa unavailable (will use VADER fallback): {type(e).__name__}")
            _roberta_pipeline = "UNAVAILABLE"
    return _roberta_pipeline if _roberta_pipeline != "UNAVAILABLE" else None


# ── CTA keyword patterns ──────────────────────────────────────────────────────
CTA_PATTERNS = [
    r"\blink in bio\b", r"\bswipe up\b", r"\bclick (here|below|the link)\b",
    r"\bcomment (below|down|your)\b", r"\bshare this\b", r"\btag (a friend|someone|your)\b",
    r"\bfollow (us|me|for)\b", r"\bcheck out\b", r"\bdon'?t miss\b",
    r"\bsave this\b", r"\blike if\b", r"\bwatch (till|to|the)\b",
    r"\bshop (now|the|our)\b", r"\bjoin (us|now|the)\b", r"\bsign up\b",
    r"\bget (yours|started|it)\b", r"\btry (it|now|this)\b"
]
CTA_RE = re.compile("|".join(CTA_PATTERNS), re.IGNORECASE)

# ── Clickbait trigger words ───────────────────────────────────────────────────
CLICKBAIT_TRIGGERS = [
    "you won't believe", "shocking", "mind-blowing", "insane", "secret",
    "nobody talks about", "this changes everything", "warning", "exposed",
    "they don't want you to know", "viral", "blew up", "trending",
    "jaw-dropping", "unbelievable", "must see", "gone wrong", "emotional",
    "wait for it", "what happened next", "life-changing", "ultimate guide",
    "the truth about", "finally revealed", "breaking"
]

# ── Flesch-Kincaid Grade Level ────────────────────────────────────────────────
def _flesch_kincaid_grade(text: str) -> float:
    sentences = max(1, len(re.split(r'[.!?]+', text)))
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    words_per_sentence = len(words) / sentences
    syllables_per_word = syllables / len(words)
    grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    return round(max(0, min(grade, 18)), 2)

def _count_syllables(word: str) -> int:
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

# ── Clickbait Score ───────────────────────────────────────────────────────────
def _clickbait_score(text: str) -> float:
    text_lower = text.lower()
    hits = sum(1 for trigger in CLICKBAIT_TRIGGERS if trigger in text_lower)
    # Also check for excessive punctuation / ALL CAPS words
    caps_words = len(re.findall(r'\b[A-Z]{3,}\b', text))
    exclamations = text.count('!')
    questions = text.count('?')
    raw = hits * 0.15 + caps_words * 0.1 + exclamations * 0.05 + questions * 0.03
    return round(min(raw, 1.0), 3)

# ── CTA Detector ─────────────────────────────────────────────────────────────
def _cta_present(text: str) -> int:
    return 1 if CTA_RE.search(text) else 0

# ── Main Engine ───────────────────────────────────────────────────────────────
def analyze_caption(caption: str) -> dict:
    """
    Main NLP analysis function. Returns the 6-feature dict.
    """
    if not caption or len(caption.strip()) < 3:
        return {
            "sentiment_score": 0.0,
            "emotional_valence": 0.0,
            "emotional_arousal": 0.0,
            "clickbait_score": 0.0,
            "cta_present": 0,
            "readability_grade": 0.0
        }

    # 1. VADER Sentiment (fast, runs locally, no GPU needed)
    vader = SentimentIntensityAnalyzer()
    vader_scores = vader.polarity_scores(caption)
    sentiment_score = round(vader_scores['compound'], 4)

    # 2. RoBERTa Emotional Valence + Arousal (heavy model, lazy-loaded)
    emotional_valence = 0.0
    emotional_arousal = 0.0
    try:
        roberta = _get_roberta()
        # Truncate to 512 tokens for RoBERTa
        trunc_caption = caption[:512]
        results = roberta(trunc_caption)[0]
        scores = {r['label']: r['score'] for r in results}
        # Map label names (model uses LABEL_0=neg, LABEL_1=neu, LABEL_2=pos)
        pos = scores.get('LABEL_2', scores.get('positive', 0))
        neg = scores.get('LABEL_0', scores.get('negative', 0))
        neu = scores.get('LABEL_1', scores.get('neutral', 0))
        # Valence: -1 (very negative) to +1 (very positive)
        emotional_valence = round(pos - neg, 4)
        # Arousal: how extreme is the emotion (not neutral)
        emotional_arousal = round(1.0 - neu, 4)
    except Exception:
        # Graceful fallback if model not downloaded yet
        emotional_valence = sentiment_score
        emotional_arousal = abs(sentiment_score)

    # 3. Clickbait Score
    clickbait = _clickbait_score(caption)

    # 4. CTA Present
    cta = _cta_present(caption)

    # 5. Readability Grade
    readability = _flesch_kincaid_grade(caption)

    return {
        "sentiment_score": sentiment_score,
        "emotional_valence": emotional_valence,
        "emotional_arousal": emotional_arousal,
        "clickbait_score": clickbait,
        "cta_present": cta,
        "readability_grade": readability
    }


if __name__ == "__main__":
    test_captions = [
        "You WON'T believe what happened at this restaurant!! Tag a friend who needs to see this! Link in bio!",
        "Our quarterly earnings report is now available for download.",
        "Just another day. Nothing special.",
    ]
    for cap in test_captions:
        print(f"\nCaption: {cap[:60]}...")
        result = analyze_caption(cap)
        for k, v in result.items():
            print(f"  {k}: {v}")
