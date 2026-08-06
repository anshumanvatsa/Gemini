"""
Hashtag Engine — PreViral
Queries the hashtag trend database to score a set of hashtags and
return 4 features for the unified feature vector, plus ranked suggestions.
"""
import os
import sqlite3
import re
from typing import List, Dict

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "hashtag_db", "hashtags.db")

# Lazy-load sentence transformer
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def _cosine_similarity(v1, v2) -> float:
    import numpy as np
    v1, v2 = np.array(v1), np.array(v2)
    norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm == 0:
        return 0.0
    return float(np.dot(v1, v2) / norm)

def extract_hashtags(text: str) -> List[str]:
    """Extract hashtags from caption text (with or without # symbol)."""
    # Find explicit #hashtags
    tagged = re.findall(r'#(\w+)', text)
    return [t.lower() for t in tagged]

def _get_db():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Hashtag DB not found at {DB_PATH}. Run: python hashtag_db/build_db.py"
        )
    return sqlite3.connect(DB_PATH)

def score_hashtags(hashtags: List[str], platform: str, caption: str = "") -> dict:
    """
    Scores a list of hashtags and returns the 4 features for the feature vector.
    """
    if not hashtags:
        return {
            "hashtag_count": 0,
            "avg_competition_ratio": 0.8,
            "niche_hashtag_ratio": 0.0,
            "trending_hashtag_count": 0
        }

    platform = platform.lower()
    conn = _get_db()
    cur = conn.cursor()

    competition_scores = []
    trending_count = 0
    found_count = 0

    for tag in hashtags:
        cur.execute("""
            SELECT competition_ratio, trend_velocity, trend_status
            FROM hashtags
            WHERE hashtag = ? AND platform = ?
        """, (tag.lower().strip('#'), platform))
        row = cur.fetchone()

        if row:
            found_count += 1
            comp_ratio, velocity, status = row
            competition_scores.append(comp_ratio)
            if status == "rising" or velocity >= 0.75:
                trending_count += 1
        else:
            # Unknown hashtag - assume medium competition
            competition_scores.append(0.6)

    conn.close()

    avg_competition = sum(competition_scores) / len(competition_scores) if competition_scores else 0.7
    niche_ratio = found_count / len(hashtags) if hashtags else 0.0

    return {
        "hashtag_count": len(hashtags),
        "avg_competition_ratio": round(avg_competition, 3),
        "niche_hashtag_ratio": round(niche_ratio, 3),
        "trending_hashtag_count": trending_count
    }

def suggest_hashtags(caption: str, platform: str, niche: str = None, top_k: int = 10) -> List[Dict]:
    """
    Returns top_k ranked hashtag suggestions based on:
    - Low competition ratio (< 0.5)
    - High trend velocity (> 0.6)
    - Semantic relevance to caption (using sentence embeddings)
    """
    platform = platform.lower()
    conn = _get_db()
    cur = conn.cursor()

    # Query low-competition, rising hashtags
    query = """
        SELECT hashtag, competition_ratio, trend_velocity, trend_status, niche
        FROM hashtags
        WHERE platform = ?
        AND competition_ratio < 0.6
        AND trend_velocity > 0.5
    """
    params = [platform]
    if niche:
        query += " AND niche = ?"
        params.append(niche.lower())

    query += " ORDER BY trend_velocity DESC LIMIT 200"
    cur.execute(query, params)
    candidates = cur.fetchall()
    conn.close()

    if not candidates:
        return []

    # Semantic similarity scoring
    try:
        embedder = _get_embedder()
        caption_emb = embedder.encode(caption)
        tag_texts = [row[0].replace("_", " ") for row in candidates]
        tag_embs = embedder.encode(tag_texts)

        scored = []
        for i, row in enumerate(candidates):
            tag, comp, velocity, status, tag_niche = row
            relevance = _cosine_similarity(caption_emb, tag_embs[i])
            # Composite score: 40% relevance + 40% velocity + 20% low competition
            composite = (relevance * 0.4) + (velocity * 0.4) + ((1 - comp) * 0.2)
            scored.append({
                "hashtag": f"#{tag}",
                "competition_score": round(comp, 3),
                "trend_velocity": round(velocity, 3),
                "trend_status": status,
                "relevance_score": round(relevance, 3),
                "composite_score": round(composite, 3),
                "niche": tag_niche
            })

        scored.sort(key=lambda x: x["composite_score"], reverse=True)
        return scored[:top_k]

    except Exception:
        # Fallback without embeddings: just return by velocity
        result = []
        for row in candidates[:top_k]:
            tag, comp, velocity, status, tag_niche = row
            result.append({
                "hashtag": f"#{tag}",
                "competition_score": round(comp, 3),
                "trend_velocity": round(velocity, 3),
                "trend_status": status,
                "relevance_score": 0.0,
                "composite_score": round(velocity, 3),
                "niche": tag_niche
            })
        return result


if __name__ == "__main__":
    caption = "Just launched my new AI-powered coding assistant! It writes Python code 10x faster. #ai #coding"
    hashtags = extract_hashtags(caption)
    print(f"Extracted hashtags: {hashtags}")

    features = score_hashtags(hashtags, "instagram", caption)
    print(f"\nHashtag features: {features}")

    suggestions = suggest_hashtags(caption, "instagram", "tech", top_k=5)
    print(f"\nTop suggestions:")
    for s in suggestions:
        print(f"  {s['hashtag']} | comp={s['competition_score']} | velocity={s['trend_velocity']} | {s['trend_status']}")
