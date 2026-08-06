"""
PreViral — LightGBM Training Pipeline
======================================
Trains the HIGH/LOW viral prediction model on 80,000 rows across 7 platforms.

Feature extraction strategy:
- NLP features: VADER (fast, <1ms/sample) — RoBERTa used at inference only
- Hashtag features: regex extraction + SQLite DB lookup
- Timing features: synthetic (no timestamps in dataset, use platform peak defaults)
- Vision features: excluded from training (no image data in text dataset)
  → Vision features are bonus at inference time via CLIP

Target label:
- Platform-normalized engagement percentile
- Top 40% = HIGH (1), bottom 60% = LOW (0)
- This gives ~40K positive samples — well-balanced for LightGBM

Expected outcome: F1 ~0.74–0.80 (vs 0.63 pre-publish baseline)
"""

import os
import sys
import re
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, classification_report, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "..", "scraper_pipeline", "data_exports", "multi_modal_dataset_70k.csv"
)
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ── Feature extraction helpers ────────────────────────────────────────────────
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
_vader = SentimentIntensityAnalyzer()

def _extract_hashtags(text: str):
    if not isinstance(text, str):
        return []
    return re.findall(r'#(\w+)', text.lower())

def _extract_nlp_features(texts: list) -> pd.DataFrame:
    """Fast VADER-based NLP features for 80K rows. ~2 min on CPU."""
    print(f"  Extracting NLP features from {len(texts):,} texts...")
    rows = []
    for i, text in enumerate(texts):
        if i % 10000 == 0:
            print(f"    {i:,}/{len(texts):,}")
        if not isinstance(text, str):
            text = ""
        # Truncate to first 512 chars for speed
        text_short = text[:512]

        scores = _vader.polarity_scores(text_short)
        compound = scores['compound']

        # CTA detection
        cta_patterns = [
            r'\b(comment|share|like|follow|subscribe|tag|save|click|watch|visit|join|sign.?up)\b',
            r'\?(.*?)\?',  # Questions
            r'!\s*$'  # Ends in exclamation
        ]
        cta_present = int(any(re.search(p, text_short, re.I) for p in cta_patterns))

        # Clickbait markers
        clickbait_words = [
            'you won', 'won\'t believe', 'shocked', 'amazing', 'incredible',
            'secret', 'hack', 'trick', 'must', 'never', 'always', 'insane',
            'mind.?blow', 'viral', 'trend'
        ]
        clickbait_score = min(1.0, sum(
            1 for w in clickbait_words if re.search(w, text_short, re.I)
        ) / 3)

        # Readability (word length proxy — short words = more readable)
        words = text_short.split()
        avg_word_len = np.mean([len(w) for w in words]) if words else 5
        readability = max(0, min(1, 1 - (avg_word_len - 3) / 10))

        # Hashtag features
        tags = _extract_hashtags(text)

        rows.append({
            "sentiment_score": float(compound),
            "emotional_valence": float(max(0, compound)),       # positive part
            "emotional_arousal": float(abs(compound)),          # intensity
            "clickbait_score": float(clickbait_score),
            "cta_present": float(cta_present),
            "readability_grade": float(readability),
            "hashtag_count": float(min(len(tags), 30)),
            "avg_competition_ratio": float(0.6),  # neutral default (no live DB at train time)
            "niche_hashtag_ratio": float(min(len(tags) / 10, 1.0)),
            "trending_hashtag_count": float(min(len(tags) // 3, 5)),
            "text_length": float(min(len(text), 2000) / 2000),
            "has_url": float(1 if re.search(r'https?://', text) else 0),
            "question_count": float(min(text.count('?'), 5) / 5),
            "exclamation_count": float(min(text.count('!'), 5) / 5),
            "emoji_count": float(min(len(re.findall(r'[^\w\s,.]', text_short)), 10) / 10),
        })

    return pd.DataFrame(rows)


def _extract_platform_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode platform + add platform-specific priors."""
    PLATFORMS = ['youtube', 'instagram', 'tiktok', 'twitter', 'linkedin', 'facebook', 'reddit']
    # Peak overlap scores by platform (general priors)
    PLATFORM_PEAK_SCORE = {
        'youtube': 0.72, 'instagram': 0.68, 'tiktok': 0.75,
        'twitter': 0.65, 'linkedin': 0.60, 'facebook': 0.55, 'reddit': 0.70
    }

    rows = []
    for _, row in df.iterrows():
        platform = str(row['platform']).lower()
        feat = {
            f"platform_{p}": float(platform == p) for p in PLATFORMS
        }
        feat["peak_overlap_score"] = PLATFORM_PEAK_SCORE.get(platform, 0.6)
        feat["day_of_week_score"] = 0.6   # Neutral (no timestamp)
        feat["audience_active_pct"] = 0.5  # Neutral
        feat["follower_count"] = 0.005      # Normalized ~5K median
        feat["avg_engagement_rate"] = 0.035 # Platform median
        feat["face_count"] = 0.0            # No image data
        feat["face_prominence_score"] = 0.0
        feat["text_density"] = 0.0
        feat["brightness_score"] = 0.5
        feat["color_vibrancy"] = 0.5
        feat["clip_semantic_score"] = 0.5
        feat["scene_cut_count"] = 0.0
        rows.append(feat)

    return pd.DataFrame(rows)


def _create_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Platform-normalized engagement percentile labeling.
    Within each platform: top 40% posts = HIGH (1), rest = LOW (0).
    This controls for YouTube's naturally higher raw engagement vs TikTok.
    """
    labels = np.zeros(len(df), dtype=int)
    for platform in df['platform'].unique():
        mask = df['platform'] == platform
        threshold = df.loc[mask, 'engagement_score'].quantile(0.60)
        labels[mask & (df['engagement_score'] >= threshold).values] = 1
    return labels


# ── Main Training Function ────────────────────────────────────────────────────
def train():
    print("="*60)
    print("PreViral — LightGBM Training Pipeline")
    print("="*60)

    # 1. Load dataset
    print(f"\n[1/6] Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=['raw_text', 'engagement_score'])
    df['platform'] = df['platform'].str.lower().str.strip()
    print(f"  Loaded: {len(df):,} rows across {df['platform'].nunique()} platforms")
    print(f"  Platform distribution:\n{df['platform'].value_counts().to_string()}")

    # 2. Create labels
    print(f"\n[2/6] Creating platform-normalized labels...")
    y = _create_labels(df)
    print(f"  HIGH (1): {y.sum():,} ({y.mean()*100:.1f}%)")
    print(f"  LOW  (0): {(1-y).sum():,} ({(1-y.mean())*100:.1f}%)")

    # 3. Extract features
    print(f"\n[3/6] Extracting NLP features (VADER)...")
    t0 = time.time()
    nlp_feats = _extract_nlp_features(df['raw_text'].tolist())
    print(f"  NLP done in {time.time()-t0:.1f}s")

    print(f"\n[4/6] Extracting platform + timing features...")
    platform_feats = _extract_platform_features(df)

    # Combine all features
    X = pd.concat([nlp_feats, platform_feats], axis=1)
    print(f"  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Features: {list(X.columns)}")

    # 4. Train with 5-fold cross-validation
    print(f"\n[5/6] Training LightGBM with 5-fold StratifiedKFold...")
    lgbm_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_f1s, fold_aucs = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**lgbm_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )

        preds = model.predict(X_val)
        proba = model.predict_proba(X_val)[:, 1]
        f1 = f1_score(y_val, preds)
        auc = roc_auc_score(y_val, proba)
        fold_f1s.append(f1)
        fold_aucs.append(auc)
        print(f"  Fold {fold}: F1={f1:.4f}  AUC={auc:.4f}  Trees={model.best_iteration_}")

    print(f"\n  Cross-validation results:")
    print(f"  Mean F1:  {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print(f"  Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    # 5. Train final model on full data
    print(f"\n[6/6] Training final model on full dataset...")
    final_model = lgb.LGBMClassifier(**lgbm_params)
    final_model.fit(X, y, callbacks=[lgb.log_evaluation(100)])

    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n  Top 15 most important features:")
    print(importance.head(15).to_string(index=False))

    # Save model + feature columns
    model_path = os.path.join(MODEL_SAVE_DIR, "previral_lgbm.joblib")
    cols_path  = os.path.join(MODEL_SAVE_DIR, "feature_columns.joblib")
    joblib.dump(final_model, model_path)
    joblib.dump(list(X.columns), cols_path)

    print(f"\n  Model saved: {model_path}")
    print(f"  Feature columns saved: {cols_path}")

    # Final evaluation summary
    final_preds = final_model.predict(X)
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (full training set)")
    print(f"{'='*60}")
    print(classification_report(y, final_preds, target_names=["LOW", "HIGH"]))
    print(f"  Cross-val F1 (unbiased): {np.mean(fold_f1s):.4f}")

    return np.mean(fold_f1s), np.mean(fold_aucs)


if __name__ == "__main__":
    mean_f1, mean_auc = train()
    print(f"\nDone! Cross-val F1={mean_f1:.4f}, AUC={mean_auc:.4f}")
