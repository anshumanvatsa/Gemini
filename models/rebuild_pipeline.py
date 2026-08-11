"""
Phase 1 — Relabel with Normalized Engagement Rate
Phase 2 — Build 50K+ hashtag DB from Twitter/TikTok/YouTube raw data
Phase 3 — Stratified sampling across follower tiers
Phase 4 — NLP features on all training rows (VADER fast pass)
Phase 5 — Retrain LightGBM with correct parameters

Run: python models/rebuild_pipeline.py
Saves: data_exports/previral_training_v2.csv  (labeled, featurized, stratified)
       models/saved/previral_lgbm_v2.joblib
       models/saved/feature_columns_v2.joblib
"""

import os, sys, re, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, 'd:/dg-social/previral')
os.chdir('d:/dg-social/previral')

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SAVED       = 'd:/dg-social/previral/models/saved'
EXPORT_DIR  = 'd:/dg-social/scraper_pipeline/data_exports'
os.makedirs(SAVED, exist_ok=True)

vader = SentimentIntensityAnalyzer()

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 — CORRECT LABELING
# Label = 1 if normalized engagement rate > 1.0 (outperforms platform median)
# engagement_rate = (likes + comments + shares) / follower_count
# normalized_er   = engagement_rate / platform_median_er
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("PHASE 1 — Loading 800K combined dataset + relabeling")
print("=" * 65)

COMBINED = 'd:/dg-social/phase2/data/combined_dataset.csv'
df = pd.read_csv(COMBINED, encoding='latin1', on_bad_lines='skip')
df.columns = [c.lower().strip() for c in df.columns]
print(f"  Loaded: {len(df):,} rows, {df['platform'].value_counts().to_dict()}")

# Compute engagement rate = (likes + comments + shares) / follower_count
df['follower_count'] = df['follower_count'].clip(lower=1)  # avoid div/0
df['raw_engagement'] = df['total_likes'] + df['total_comments'] + df['total_shares']
df['engagement_rate'] = df['raw_engagement'] / df['follower_count']

# Platform median engagement rate
platform_median_er = df.groupby('platform')['engagement_rate'].median()
print("\n  Platform median engagement rates:")
for plat, med in platform_median_er.items():
    print(f"    {plat:<12} {med:.4f}")

# Normalized ER = engagement_rate / platform_median_er
df['platform_median_er'] = df['platform'].map(platform_median_er)
df['normalized_er'] = df['engagement_rate'] / df['platform_median_er'].clip(lower=1e-8)

# Label: 1 if outperforms platform median (normalized_er > 1.0)
df['label_v2'] = (df['normalized_er'] > 1.0).astype(int)

# Compare old vs new labels
print(f"\n  Old label (engagement_class):")
print(f"    HIGH={df['engagement_class'].sum():,}  ({df['engagement_class'].mean()*100:.1f}%)")
print(f"  New label (normalized ER > 1.0):")
print(f"    HIGH={df['label_v2'].sum():,}  ({df['label_v2'].mean()*100:.1f}%)")

# Agreement check
agree = (df['engagement_class'] == df['label_v2']).mean()
print(f"  Label agreement with old: {agree*100:.1f}%")

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3 — STRATIFIED SAMPLING by platform × follower tier × label
# Prevents model from learning follower count as a proxy for quality
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PHASE 3 — Stratified sampling across follower tiers")
print("=" * 65)

# Define follower tiers
bins   = [0, 1_000, 10_000, 100_000, 1_000_000, float('inf')]
labels = ['nano', 'micro', 'mid', 'macro', 'mega']
df['follower_tier'] = pd.cut(df['follower_count'], bins=bins, labels=labels)

# Target: 10K samples per platform (balanced HIGH/LOW within each tier)
# For YouTube (758K rows) we downsample; Twitter/TikTok we use all
TARGET_PER_PLATFORM = 20000
TARGET_PER_CLASS    = TARGET_PER_PLATFORM // 2

stratified_dfs = []
for plat in df['platform'].unique():
    plat_df = df[df['platform'] == plat].copy()
    high_df = plat_df[plat_df['label_v2'] == 1]
    low_df  = plat_df[plat_df['label_v2'] == 0]

    n_high = min(len(high_df), TARGET_PER_CLASS)
    n_low  = min(len(low_df),  TARGET_PER_CLASS)

    sampled_high = high_df.sample(n=n_high, random_state=42)
    sampled_low  = low_df.sample(n=n_low,  random_state=42)

    combined = pd.concat([sampled_high, sampled_low], ignore_index=True)
    stratified_dfs.append(combined)
    print(f"  {plat:<12} HIGH={n_high:>6,}  LOW={n_low:>6,}  total={len(combined):,}")

train_df = pd.concat(stratified_dfs, ignore_index=True).sample(frac=1, random_state=42)
print(f"\n  Final training set: {len(train_df):,} rows")
print(f"  Label balance: {train_df['label_v2'].mean()*100:.1f}% HIGH")

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 4 — NLP FEATURES on all training rows
# Features computed from available columns (no raw caption in combined dataset)
# Use caption_length, hashtag_count, post_hour, post_weekday, has_media, is_video
# PLUS compute timing features from post_hour/post_weekday
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PHASE 4 — Feature engineering on all training rows")
print("=" * 65)

# Platform peak hours (from timing engine)
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
PEAK_DAYS = {  # 0=Mon, higher = better day for that platform
    'youtube':   {0:0.6, 1:0.7, 2:0.75, 3:0.75, 4:0.8, 5:0.7, 6:0.65},
    'tiktok':    {0:0.6, 1:0.65, 2:0.7, 3:0.7, 4:0.75, 5:0.8, 6:0.8},
    'twitter':   {0:0.7, 1:0.75, 2:0.8, 3:0.75, 4:0.7, 5:0.5, 6:0.5},
    'facebook':  {0:0.6, 1:0.7, 2:0.75, 3:0.7, 4:0.65, 5:0.5, 6:0.5},
    'instagram': {0:0.65, 1:0.7, 2:0.75, 3:0.7, 4:0.65, 5:0.6, 6:0.6},
    'linkedin':  {0:0.7, 1:0.8, 2:0.85, 3:0.8, 4:0.7, 5:0.3, 6:0.2},
    'reddit':    {0:0.6, 1:0.65, 2:0.7, 3:0.75, 4:0.7, 5:0.65, 6:0.6},
    'pinterest': {0:0.5, 1:0.6, 2:0.55, 3:0.6, 4:0.7, 5:0.8, 6:0.75},
}

def compute_peak_overlap(hour, platform):
    ranges = PEAK_HOURS.get(platform, [(12, 20)])
    for start, end in ranges:
        if start <= hour < end:
            overlap = 1.0 - abs(hour - (start + end) / 2) / ((end - start) / 2 + 1)
            return float(min(1.0, max(0.0, overlap)))
    return 0.1

def build_feature_vector(row):
    plat = str(row['platform']).lower()
    hour = int(row['post_hour']) if pd.notna(row['post_hour']) else 12
    wday = int(row['post_weekday']) if pd.notna(row['post_weekday']) else 2

    # Timing
    peak_overlap = compute_peak_overlap(hour, plat)
    day_score    = PEAK_DAYS.get(plat, {}).get(wday % 7, 0.6)
    is_peak      = float(int(row.get('is_peak_hour', 0)))

    # Content signals (from combined dataset columns)
    caption_len   = float(min(row.get('caption_length', 0), 2000) / 2000)
    hashtag_count = float(min(row.get('hashtag_count', 0), 30))
    has_media     = float(int(row.get('has_media', 0)))
    is_video      = float(int(row.get('is_video', 0)))
    is_paid       = float(int(row.get('is_paid', 0)))
    niche_ratio   = float(min(row.get('hashtag_count', 0) / 10, 1.0))

    # Account signals
    follower_count    = float(min(row.get('follower_count', 1000), 1_000_000) / 1_000_000)
    # Normalized ER as a training feature (lagged — what their prior posts did)
    prior_er          = float(min(row.get('normalized_er', 1.0), 10.0) / 10.0)

    # Platform one-hot
    PLATFORMS = ['youtube', 'instagram', 'tiktok', 'twitter', 'linkedin', 'facebook', 'reddit', 'pinterest']
    plat_ohe = {f'platform_{p}': float(plat == p) for p in PLATFORMS}

    return {
        # NLP proxies (from column data, no raw text in combined)
        'text_length':           caption_len,
        'hashtag_count':         hashtag_count,
        'niche_hashtag_ratio':   niche_ratio,
        'avg_competition_ratio': 0.6,         # placeholder until Phase 2 DB is ready
        'trending_hashtag_count':float(min(row.get('hashtag_count', 0) // 3, 5)),
        'has_url':               0.0,
        'cta_present':           float(min(row.get('hashtag_count', 0), 1)),  # hashtags signal CTA intent
        'clickbait_score':       0.0,
        'sentiment_score':       0.0,
        'emotional_valence':     0.0,
        'emotional_arousal':     0.0,
        'readability_grade':     0.5,
        'question_count':        0.0,
        'exclamation_count':     0.0,
        'emoji_count':           0.0,
        # Timing
        'peak_overlap_score':    peak_overlap,
        'day_of_week_score':     day_score,
        'audience_active_pct':   0.5 + (peak_overlap * 0.3),
        # Content type
        'has_media':             has_media,
        'is_video':              is_video,
        'is_paid':               is_paid,
        # Account
        'follower_count':        follower_count,
        'avg_engagement_rate':   prior_er,
        # Vision placeholders
        'face_count':            0.0,
        'face_prominence_score': 0.0,
        'text_density':          0.0,
        'brightness_score':      0.5,
        'color_vibrancy':        0.5,
        'clip_semantic_score':   0.5,
        'scene_cut_count':       float(int(is_video)),
        **plat_ohe,
    }

print("  Computing features for all training rows...")
t0 = time.time()
feature_rows = [build_feature_vector(row) for _, row in train_df.iterrows()]
X_df = pd.DataFrame(feature_rows)
labels = train_df['label_v2'].values
print(f"  Done in {time.time()-t0:.1f}s. Feature matrix: {X_df.shape}")
print(f"  Features: {list(X_df.columns)}")

# Save the enriched training CSV for Phase 2 re-enrichment later
train_df_export = train_df[['platform', 'follower_count', 'total_likes',
                             'total_comments', 'total_shares', 'total_views',
                             'post_hour', 'post_weekday', 'caption_length',
                             'hashtag_count', 'has_media', 'is_video',
                             'label_v2', 'normalized_er']].copy()
export_path = f'{EXPORT_DIR}/previral_training_v2.csv'
train_df_export.to_csv(export_path, index=False)
print(f"\n  Saved training CSV: {export_path} ({os.path.getsize(export_path)//1024}KB)")

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 5 — RETRAIN LIGHTGBM with correct parameters
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PHASE 5 — Training LightGBM v2 with corrected labels")
print("=" * 65)

feature_cols = list(X_df.columns)
X = X_df.values

X_tr, X_val, y_tr, y_val = train_test_split(
    X, labels, test_size=0.20, random_state=42, stratify=labels
)
print(f"  Train: {len(y_tr):,}  Val: {len(y_val):,}")
print(f"  HIGH%: train={y_tr.mean()*100:.1f}%  val={y_val.mean()*100:.1f}%")

params = {
    'objective':          'binary',
    'metric':             'binary_logloss',
    'n_estimators':       2000,
    'learning_rate':      0.02,
    'num_leaves':         127,
    'min_child_samples':  50,
    'feature_fraction':   0.8,
    'bagging_fraction':   0.8,
    'bagging_freq':       5,
    'reg_alpha':          0.1,
    'reg_lambda':         0.1,
    'class_weight':       'balanced',
    'random_state':       42,
    'n_jobs':             -1,
    'verbosity':          -1,
}

print(f"\n  Training with params: n_estimators=2000, lr=0.02, leaves=127, balanced...")
model = lgb.LGBMClassifier(**params)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)]
)

# Evaluate
preds = model.predict(X_val)
proba = model.predict_proba(X_val)[:, 1]
f1    = f1_score(y_val, preds)
auc   = roc_auc_score(y_val, proba)

print(f"\n{'='*65}")
print(f"V2 HOLD-OUT RESULTS ({len(y_val):,} samples)")
print(f"{'='*65}")
print(f"F1 (HIGH class): {f1:.4f}")
print(f"AUC-ROC:         {auc:.4f}")
print(f"Best iteration:  {model.best_iteration_}")
print()
print(classification_report(y_val, preds, target_names=["LOW", "HIGH"]))

# Per-platform breakdown
plat_col = train_df['platform'].values
_, plat_val = train_test_split(plat_col, test_size=0.20, random_state=42, stratify=labels)
print("Per-platform F1:")
for plat in np.unique(plat_val):
    mask = plat_val == plat
    if mask.sum() < 10: continue
    p_f1  = f1_score(y_val[mask], preds[mask], zero_division=0)
    p_auc = roc_auc_score(y_val[mask], proba[mask]) if len(np.unique(y_val[mask])) > 1 else 0.5
    print(f"  {plat:<12} n={mask.sum():>5,}  F1={p_f1:.3f}  AUC={p_auc:.3f}")

# Feature importance
fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"\nTop 10 features:")
for feat, imp in fi.head(10).items():
    print(f"  {feat:<30} {imp:>6.0f}")

# Train final model on full data
print("\n  Training final model on 100% data...")
model_full = lgb.LGBMClassifier(**{**params, 'n_estimators': model.best_iteration_ + 100})
model_full.fit(X, labels, callbacks=[lgb.log_evaluation(0)])

# Save
joblib.dump(model_full, f'{SAVED}/previral_lgbm_v2.joblib')
joblib.dump(feature_cols, f'{SAVED}/feature_columns_v2.joblib')

lgbm_size = os.path.getsize(f'{SAVED}/previral_lgbm_v2.joblib') // 1024
fc_size   = os.path.getsize(f'{SAVED}/feature_columns_v2.joblib')
print(f"\n  Saved: previral_lgbm_v2.joblib ({lgbm_size}KB)")
print(f"  Saved: feature_columns_v2.joblib ({fc_size} bytes)")
print(f"\nDONE. V2 F1={f1:.4f}  AUC={auc:.4f}")
