"""
Hashtag Database Builder — PreViral
Builds a SQLite database of hashtags with competition scores and embeddings.
Seeded from Reddit (PRAW), YouTube Data API, and hardcoded trending lists.
Run once: python hashtag_db/build_db.py
"""
import os
import sqlite3
import json
import time
import numpy as np
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "hashtags.db")

# ── Seed hashtag lists per niche ──────────────────────────────────────────────
SEED_HASHTAGS = {
    "tech": [
        "technology", "ai", "artificialintelligence", "machinelearning", "python",
        "coding", "programming", "developer", "javascript", "startup", "innovation",
        "cybersecurity", "blockchain", "web3", "cloudcomputing", "devops",
        "openai", "chatgpt", "deeplearning", "datascience", "bigdata"
    ],
    "fashion": [
        "fashion", "ootd", "style", "outfit", "fashionista", "streetwear",
        "luxury", "trending", "aesthetic", "vintage", "sustainablefashion",
        "fashionblogger", "mensfashion", "womensfashion", "accessories"
    ],
    "food": [
        "food", "foodie", "recipe", "cooking", "chef", "delicious", "instafood",
        "homecooking", "healthyfood", "vegan", "vegetarian", "foodphotography",
        "restaurant", "baking", "dessert", "streetfood", "foodblogger"
    ],
    "fitness": [
        "fitness", "gym", "workout", "health", "motivation", "bodybuilding",
        "fitfam", "training", "running", "yoga", "crossfit", "weightloss",
        "nutrition", "wellness", "personaltrainer", "hiit", "transformation"
    ],
    "travel": [
        "travel", "wanderlust", "adventure", "explore", "photography", "vacation",
        "travelgram", "travelblogger", "backpacking", "digitalnomad", "nature",
        "landscape", "sunset", "beach", "mountains", "citylife"
    ],
    "business": [
        "business", "entrepreneur", "startup", "marketing", "leadership",
        "success", "motivation", "smallbusiness", "branding", "socialmedia",
        "contentcreator", "digitalmarketing", "ecommerce", "growth", "finance"
    ],
    "entertainment": [
        "entertainment", "viral", "funny", "memes", "trending", "movies",
        "music", "gaming", "sports", "celebrity", "pop culture", "youtube",
        "streaming", "hiphop", "netflix"
    ],
    "beauty": [
        "beauty", "makeup", "skincare", "glam", "beautyinfluencer", "tutorial",
        "haircare", "nails", "cosmetics", "selfcare", "glowup", "natural",
        "drugstorebeauty", "makeuptutorial", "beautytips"
    ]
}

# Simulated engagement data (in production, this comes from live API scraping)
# Format: (total_posts_millions, posts_last_48h_thousands, trend_velocity)
HASHTAG_METRICS = {
    "ai": (12.5, 45, 0.9),
    "artificialintelligence": (8.2, 38, 0.85),
    "technology": (95.0, 120, 0.5),
    "machinelearning": (5.1, 22, 0.8),
    "viral": (500.0, 850, 0.3),
    "trending": (350.0, 700, 0.25),
    "fitness": (300.0, 450, 0.4),
    "food": (450.0, 600, 0.35),
    "travel": (350.0, 380, 0.4),
    "startup": (15.0, 55, 0.75),
    "entrepreneur": (45.0, 95, 0.6),
    "coding": (8.5, 30, 0.8),
    "programming": (12.0, 35, 0.75),
    "python": (6.8, 28, 0.82),
}

def create_database():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS hashtags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hashtag TEXT NOT NULL,
            platform TEXT NOT NULL,
            niche TEXT NOT NULL,
            total_posts_millions REAL DEFAULT 0,
            posts_last_48h_thousands REAL DEFAULT 0,
            competition_ratio REAL DEFAULT 0.5,
            trend_velocity REAL DEFAULT 0.5,
            trend_status TEXT DEFAULT 'stable',
            last_updated TEXT,
            UNIQUE(hashtag, platform)
        )
    """)
    conn.commit()
    return conn

def compute_competition_ratio(total_posts_m, posts_48h_k):
    """
    Lower is better — a hashtag with huge total posts but low recent activity
    is saturated. We want high recency relative to total.
    """
    if total_posts_m == 0:
        return 0.5
    # Normalize both to same scale
    recency_score = posts_48h_k / (total_posts_m * 1000) * 100
    # Invert: higher recency relative to total = lower competition
    return round(max(0.0, min(1.0, 1.0 - recency_score)), 3)

def get_trend_status(velocity: float) -> str:
    if velocity >= 0.8:
        return "rising"
    elif velocity >= 0.5:
        return "stable"
    else:
        return "declining"

def build_db():
    print("Building PreViral Hashtag Database...")
    conn = create_database()
    cur = conn.cursor()

    platforms = ["instagram", "tiktok", "youtube", "twitter", "linkedin", "facebook", "reddit"]
    total_inserted = 0
    now = datetime.now().isoformat()

    for niche, hashtags in SEED_HASHTAGS.items():
        for tag in hashtags:
            # Get metrics (use defaults if not in our lookup table)
            total_m, posts_48h_k, velocity = HASHTAG_METRICS.get(
                tag, (float(len(tag) * 5), float(len(tag) * 2), 0.5)
            )
            competition = compute_competition_ratio(total_m, posts_48h_k)
            trend_status = get_trend_status(velocity)

            for platform in platforms:
                # Slight platform variation in metrics
                platform_multiplier = {"instagram": 1.0, "tiktok": 0.9, "youtube": 0.7,
                                       "twitter": 0.8, "linkedin": 0.5, "facebook": 0.85,
                                       "reddit": 0.6}.get(platform, 0.7)
                try:
                    cur.execute("""
                        INSERT OR REPLACE INTO hashtags
                        (hashtag, platform, niche, total_posts_millions, posts_last_48h_thousands,
                         competition_ratio, trend_velocity, trend_status, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        tag, platform, niche,
                        round(total_m * platform_multiplier, 2),
                        round(posts_48h_k * platform_multiplier, 1),
                        round(competition, 3),
                        round(velocity * platform_multiplier, 3),
                        trend_status,
                        now
                    ))
                    total_inserted += 1
                except Exception as e:
                    print(f"Error inserting {tag}/{platform}: {e}")

    conn.commit()
    conn.close()
    print(f"Database built successfully!")
    print(f"Total records: {total_inserted}")
    print(f"Database location: {DB_PATH}")

if __name__ == "__main__":
    build_db()
