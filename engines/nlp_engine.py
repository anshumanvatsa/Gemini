"""
NLP Engine — PreViral
Runs VADER + RoBERTa + Clickbait Scorer + CTA Detector on caption text.
Returns 16 features aligned with v3 LightGBM training features.
Top model predictors: text_length, caps_ratio, unique_word_ratio,
                      readability_grade, sentiment_score, avg_word_length.
"""
import re
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
    grade = 0.39 * (len(words)/sentences) + 11.8 * (syllables/len(words)) - 15.59
    return round(max(0, min(grade, 18)), 2)

def _count_syllables(word: str) -> int:
    word = word.lower()
    count, prev_vowel = 0, False
    for char in word:
        is_v = char in "aeiouy"
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

# ── Clickbait Score ───────────────────────────────────────────────────────────
def _clickbait_score(text: str) -> float:
    tl = text.lower()
    hits = sum(1 for t in CLICKBAIT_TRIGGERS if t in tl)
    caps = len(re.findall(r'\b[A-Z]{3,}\b', text))
    raw = hits * 0.15 + caps * 0.1 + text.count('!') * 0.05 + text.count('?') * 0.03
    return round(min(raw, 1.0), 3)

# ── CTA Detector ─────────────────────────────────────────────────────────────
def _cta_present(text: str) -> int:
    return 1 if CTA_RE.search(text) else 0

# ── Main Engine ───────────────────────────────────────────────────────────────
def analyze_caption(caption: str) -> dict:
    """
    Returns 16-feature dict aligned with v3 LightGBM training.
    All features are in [0, 1] range or small positive floats.
    """
    EMPTY = {
        "sentiment_score": 0.0, "emotional_valence": 0.0,
        "emotional_arousal": 0.0, "clickbait_score": 0.0,
        "cta_present": 0, "readability_grade": 0.5,
        "text_length": 0.0, "has_url": 0.0,
        "question_count": 0.0, "exclamation_count": 0.0,
        "emoji_count": 0.0, "hashtag_count_nlp": 0.0,
        "mention_count": 0.0, "caps_ratio": 0.0,
        "avg_word_length": 0.5, "unique_word_ratio": 0.5,
    }
    if not caption or len(caption.strip()) < 3:
        return EMPTY

    # ── Structural stats (no models needed) ──────────────────────────────────
    words = caption.split()
    alpha = [c for c in caption if c.isalpha()]
    caps  = [c for c in alpha if c.isupper()]

    tags     = re.findall(r'#\w+', caption)
    mentions = re.findall(r'@\w+', caption)
    emojis   = re.findall(r'[^\w\s,.\-!?#@\'"()\[\]{}:;/\\]', caption)
    has_url  = float(bool(re.search(r'https?://', caption)))

    avg_wl          = sum(len(w) for w in words) / max(len(words), 1)
    caps_ratio      = len(caps) / max(len(alpha), 1)
    unique_wr       = len(set(w.lower() for w in words)) / max(len(words), 1)

    # Readability: lower FK grade = more readable = closer to 1.0
    fk = _flesch_kincaid_grade(caption)
    readability_norm = max(0.0, 1.0 - fk / 18.0)

    # ── VADER Sentiment ───────────────────────────────────────────────────────
    vader = SentimentIntensityAnalyzer()
    vs = vader.polarity_scores(caption)
    sentiment_score = round(vs['compound'], 4)

    # ── RoBERTa Valence + Arousal ─────────────────────────────────────────────
    emotional_valence = sentiment_score          # fallback
    emotional_arousal = abs(sentiment_score)     # fallback
    try:
        roberta = _get_roberta()
        if roberta:
            results = roberta(caption[:512])[0]
            scores  = {r['label']: r['score'] for r in results}
            pos = scores.get('LABEL_2', scores.get('positive', 0))
            neg = scores.get('LABEL_0', scores.get('negative', 0))
            neu = scores.get('LABEL_1', scores.get('neutral',  0))
            emotional_valence = round(pos - neg, 4)
            emotional_arousal = round(1.0 - neu, 4)
    except Exception:
        pass

    return {
        # Sentiment (3)
        "sentiment_score":    sentiment_score,
        "emotional_valence":  round(max(0.0, emotional_valence), 4),
        "emotional_arousal":  round(emotional_arousal, 4),
        # Intent (2)
        "clickbait_score":    _clickbait_score(caption),
        "cta_present":        _cta_present(caption),
        # Readability (1)
        "readability_grade":  round(readability_norm, 4),
        # Text stats — v3 top features (6)
        "text_length":        round(min(len(caption), 3000) / 3000, 4),
        "caps_ratio":         round(min(caps_ratio, 0.5) / 0.5, 4),
        "unique_word_ratio":  round(min(unique_wr, 1.0), 4),
        "avg_word_length":    round(min(avg_wl, 12) / 12, 4),
        "question_count":     round(min(caption.count('?'), 5) / 5, 4),
        "exclamation_count":  round(min(caption.count('!'), 5) / 5, 4),
        # Social signals (4)
        "has_url":            has_url,
        "emoji_count":        round(min(len(emojis), 15) / 15, 4),
        "hashtag_count_nlp":  float(min(len(tags), 30)),
        "mention_count":      round(min(len(mentions), 10) / 10, 4),
    }


if __name__ == "__main__":
    tests = [
        "You WON'T believe what happened!! Tag a friend! Link in bio! #viral #food",
        "Our quarterly earnings report is now available for download. Metrics improved.",
        "Just another day. Nothing special.",
        "HUGE SALE TODAY ONLY!! Don't miss out!! Fire emoji #sale #fashion",
    ]
    for cap in tests:
        print(f"\n{cap[:70]}")
        r = analyze_caption(cap)
        for k, v in r.items():
            print(f"  {k:25s} {v}")
