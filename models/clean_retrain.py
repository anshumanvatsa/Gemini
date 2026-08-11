"""
CLEAN REBUILD — No leakage, proper pre-publish features only.

Key insight from leakage audit:
- combined_dataset has no real follower counts for YouTube/TikTok (median=0)
- The existing engagement_class is already correct (ER-normalized, 98.9% match)
- Our Phase 1 label IS valid — we just can't use post-publish engagement as features

FEATURE RULES (strict pre-publish only):
  OK  : caption_length, hashtag_count, post_hour, post_weekday, has_media, is_video
  OK  : platform one-hot, peak_overlap, day_of_week_score
  OK  : follower_count (known before posting)
  NO  : total_likes, total_comments, total_shares, total_views (post-publish)
  NO  : avg_engagement_rate derived from this post (leakage)
  NO  : normalized_er, raw_er (leakage — computed from post-publish counts)

LABEL: engagement_class from combined_dataset (already ER-normalized, verified correct)

RUN: python models/clean_retrain.py
"""

import os, sys, re, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, 'd:/dg-social/previral')

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report

SAVED = 'd:/dg-social/previral/models/saved'
os.makedirs(SAVED, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
print("=" * 65)
print("LOADING 800K combined dataset")
print("=" * 65)
COMBINED = 'd:/dg-social/phase2/data/combined_dataset.csv'
df = pd.read_csv(COMBINED, encoding='latin1', on_bad_lines='skip')
df.columns = [c.lower().strip() for c in df.columns]
print(f"Loaded {len(df):,} rows")
print(f"Platforms: {df['platform'].value_counts().to_dict()}")

# ── Label ─────────────────────────────────────────────────────────────────────
# engagement_class is already ER-normalized (verified: 98.9% match with our computed label)
labels = df['engagement_class'].values
print(f"Label balance: HIGH={labels.mean()*100:.1f}%")

# ── Stratified sampling — balanced across platforms ───────────────────────────
print("\n" + "=" * 65)
print("STRATIFIED SAMPLING")
print("=" * 65)

TARGET_PER_CLASS_PER_PLAT = 15000
sampled = []

for plat in df['platform'].unique():
    sub = df[df['platform'] == plat].copy()
    high = sub[sub['engagement_class'] == 1]
    low  = sub[sub['engagement_class'] == 0]
    n = min(len(high), len(low), TARGET_PER_CLASS_PER_PLAT)
    s_high = high.sample(n=n, random_state=42)
    s_low  = low.sample(n=n,  random_state=42)
    sampled.append(pd.concat([s_high, s_low]))
    print(f"  {plat:<12} HIGH={n:>6,}  LOW={n:>6,}  total={2*n:,}")

train_df = pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=42)
print(f"\nFinal set: {len(train_df):,} rows, {train_df['engagement_class'].mean()*100:.1f}% HIGH")

# ── Feature engineering — STRICT pre-publish only ─────────────────────────────
print("\n" + "=" * 65)
print("FEATURE ENGINEERING (pre-publish only, no leakage)")
print("=" * 65)

PEAK_HOURS = {
    'youtube':   [(14, 22)],
    'tiktok':    [(6, 10), (19, 23)],
    'twitter':   [(8, 10), (12, 13), (17, 18)],
    'facebook':  [(13, 16)],
    'instagram': [(11, 13), (19, 21)],
    'linkedin':  [(8, 10), (17, 18)],
    'reddit':    [(12, 14), (18, 22)],
    'pinterest': [(20, 23)],
}
PEAK_DAYS = {
    'youtube':   [0.6, 0.7, 0.75, 0.75, 0.8, 0.7, 0.65],
    'tiktok':    [0.6, 0.65, 0.7, 0.7, 0.75, 0.8, 0.8],
    'twitter':   [0.7, 0.75, 0.8, 0.75, 0.7, 0.5, 0.5],
    'facebook':  [0.6, 0.7, 0.75, 0.7, 0.65, 0.5, 0.5],
    'instagram': [0.65, 0.7, 0.75, 0.7, 0.65, 0.6, 0.6],
    'linkedin':  [0.7, 0.8, 0.85, 0.8, 0.7, 0.3, 0.2],
    'reddit':    [0.6, 0.65, 0.7, 0.75, 0.7, 0.65, 0.6],
    'pinterest': [0.5, 0.6, 0.55, 0.6, 0.7, 0.8, 0.75],
}
PLATFORMS = ['youtube', 'instagram', 'tiktok', 'twitter', 'linkedin', 'facebook', 'reddit', 'pinterest']

def peak_overlap(hour, plat):
    ranges = PEAK_HOURS.get(plat, [(12, 20)])
    for s, e in ranges:
        if s <= hour < e:
            centre = (s + e) / 2
            half   = (e - s) / 2 + 1
            return float(min(1.0, max(0.0, 1.0 - abs(hour - centre) / half)))
    return 0.1

def build_features(row):
    plat = str(row['platform']).lower()
    hour = int(row['post_hour'])   if pd.notna(row['post_hour'])    else 12
    wday = int(row['post_weekday']) if pd.notna(row['post_weekday']) else 2
    wday = wday % 7

    # Timing
    po   = peak_overlap(hour, plat)
    ds   = PEAK_DAYS.get(plat, [0.6]*7)[wday]

    # Content metadata (pre-publish, available before posting)
    cap_len = float(min(int(row.get('caption_length', 0)), 2000) / 2000)
    ht_cnt  = float(min(int(row.get('hashtag_count',  0)), 30))
    media   = float(int(row.get('has_media',   0)))
    video   = float(int(row.get('is_video',    0)))
    paid    = float(int(row.get('is_paid',     0)))
    weekend = float(int(row.get('is_weekend',  0)))
    is_peak = float(int(row.get('is_peak_hour', 0)))

    # Follower count — known pre-publish (use 0 for missing, model learns to handle)
    fc = float(min(int(row.get('follower_count', 0)), 10_000_000) / 10_000_000)

    # Hashtag-derived
    niche_ratio = float(min(ht_cnt / 10, 1.0))
    trending_ht = float(min(ht_cnt // 3, 5))

    return {
        # Content signals
        'caption_length':        cap_len,
        'hashtag_count':         ht_cnt,
        'niche_hashtag_ratio':   niche_ratio,
        'trending_hashtag_count':trending_ht,
        'avg_competition_ratio': 0.6,     # populated at inference from hashtag_db
        'has_media':             media,
        'is_video':              video,
        'is_paid':               paid,
        # Timing signals
        'peak_overlap_score':    po,
        'day_of_week_score':     ds,
        'audience_active_pct':   float(0.4 + po * 0.4),
        'is_weekend':            weekend,
        'is_peak_hour':          is_peak,
        'post_hour_sin':         float(np.sin(2 * np.pi * hour / 24)),
        'post_hour_cos':         float(np.cos(2 * np.pi * hour / 24)),
        'post_wday_sin':         float(np.sin(2 * np.pi * wday / 7)),
        'post_wday_cos':         float(np.cos(2 * np.pi * wday / 7)),
        # Account signal (pre-publish)
        'follower_count':        fc,
        # NLP placeholders (0 when no raw text; real values at inference)
        'sentiment_score':       0.0,
        'emotional_valence':     0.0,
        'emotional_arousal':     0.0,
        'clickbait_score':       0.0,
        'cta_present':           0.0,
        'readability_grade':     0.5,
        'text_length':           cap_len,  # same as caption_length
        'has_url':               0.0,
        'question_count':        0.0,
        'exclamation_count':     0.0,
        'emoji_count':           0.0,
        # Vision placeholders (0 when no thumbnail; real values at inference)
        'face_count':            0.0,
        'face_prominence_score': 0.0,
        'text_density':          0.0,
        'brightness_score':      0.5,
        'color_vibrancy':        0.5,
        'clip_semantic_score':   0.5,
        'scene_cut_count':       video,
        # Platform OHE
        **{f'platform_{p}': float(plat == p) for p in PLATFORMS},
    }

print("Building feature matrix...")
t0 = time.time()
feat_rows = [build_features(row) for _, row in train_df.iterrows()]
X_df = pd.DataFrame(feat_rows)
y    = train_df['engagement_class'].values
print(f"Done in {time.time()-t0:.1f}s. Shape: {X_df.shape}")

feature_cols = list(X_df.columns)
X = X_df.values

# ── Verify zero leakage ────────────────────────────────────────────────────────
print("\n=== Leakage Check: correlations with label ===")
corr = pd.Series(
    [abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])],
    index=feature_cols
).sort_values(ascending=False)
print(corr.head(10))
print(f"Max correlation: {corr.max():.4f}  (should be <0.5 to confirm no leakage)")

# ── Train/val split ────────────────────────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain: {len(y_tr):,}  Val: {len(y_val):,}")

# ── Phase 5 — Train with corrected parameters ──────────────────────────────────
print("\n" + "=" * 65)
print("PHASE 5 — LightGBM with corrected parameters (no leakage)")
print("=" * 65)

params = {
    'objective':         'binary',
    'metric':            'binary_logloss',
    'n_estimators':      2000,
    'learning_rate':     0.02,
    'num_leaves':        127,
    'min_child_samples': 50,
    'feature_fraction':  0.8,
    'bagging_fraction':  0.8,
    'bagging_freq':      5,
    'reg_alpha':         0.1,
    'reg_lambda':        0.1,
    'class_weight':      'balanced',
    'random_state':      42,
    'n_jobs':            -1,
    'verbosity':         -1,
}

model = lgb.LGBMClassifier(**params)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)]
)

# ── Evaluate ───────────────────────────────────────────────────────────────────
preds = model.predict(X_val)
proba = model.predict_proba(X_val)[:, 1]
f1    = f1_score(y_val, preds)
auc   = roc_auc_score(y_val, proba)

print(f"\n{'='*65}")
print(f"CLEAN HOLD-OUT RESULTS ({len(y_val):,} samples)")
print(f"{'='*65}")
print(f"F1 (HIGH class): {f1:.4f}")
print(f"AUC-ROC:         {auc:.4f}")
print(f"Best iteration:  {model.best_iteration_}")
print()
print(classification_report(y_val, preds, target_names=["LOW", "HIGH"]))

# Per-platform
_, val_idx = train_test_split(np.arange(len(train_df)), test_size=0.20, random_state=42, stratify=y)
plat_val = train_df['platform'].values[val_idx]
print("Per-platform F1:")
for plat in np.unique(plat_val):
    mask = plat_val == plat
    if mask.sum() < 10: continue
    p_f1  = f1_score(y_val[mask], preds[mask], zero_division=0)
    p_auc = roc_auc_score(y_val[mask], proba[mask]) if len(np.unique(y_val[mask])) > 1 else 0.5
    print(f"  {plat:<12} n={mask.sum():>6,}  F1={p_f1:.3f}  AUC={p_auc:.3f}")

# Feature importance
fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"\nTop 15 features by importance:")
for feat, imp in fi.head(15).items():
    print(f"  {feat:<30} {imp:>6.0f}")

# Confidence gap check (intuition test)
high_proba_on_high = proba[y_val == 1].mean()
high_proba_on_low  = proba[y_val == 0].mean()
print(f"\nConfidence gap:")
print(f"  Avg P(HIGH) on actual HIGH posts: {high_proba_on_high:.3f}")
print(f"  Avg P(HIGH) on actual LOW posts:  {high_proba_on_low:.3f}")
print(f"  Separation: {high_proba_on_high - high_proba_on_low:.3f}  (target: >0.2)")

# ── Save ───────────────────────────────────────────────────────────────────────
print("\nSaving artifacts...")
model_full = lgb.LGBMClassifier(**{**params, 'n_estimators': model.best_iteration_ + 100})
model_full.fit(X, y, callbacks=[lgb.log_evaluation(0)])

joblib.dump(model_full,  f'{SAVED}/previral_lgbm_v2.joblib')
joblib.dump(feature_cols, f'{SAVED}/feature_columns_v2.joblib')

print(f"  previral_lgbm_v2.joblib:   {os.path.getsize(SAVED+'/previral_lgbm_v2.joblib')//1024}KB")
print(f"  feature_columns_v2.joblib: {os.path.getsize(SAVED+'/feature_columns_v2.joblib')} bytes")
print(f"\nFINAL: F1={f1:.4f}  AUC={auc:.4f}")
