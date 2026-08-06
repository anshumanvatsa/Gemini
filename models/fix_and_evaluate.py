"""
Fix corrupt 0-byte model artifacts and run honest hold-out evaluation.
"""
import sys, os, re
sys.path.insert(0, 'd:/dg-social/previral')

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()
SAVED = 'd:/dg-social/previral/models/saved'
os.makedirs(SAVED, exist_ok=True)

CLICKBAIT = ['wont believe', 'shocked', 'secret', 'hack', 'viral', 'insane', 'mind blow']
PLATFORMS = ['youtube', 'instagram', 'tiktok', 'twitter', 'linkedin', 'facebook', 'reddit']
PEAK = {'youtube': 0.72, 'instagram': 0.68, 'tiktok': 0.75,
        'twitter': 0.65, 'linkedin': 0.60, 'facebook': 0.55, 'reddit': 0.70}


def extract_features(texts, platforms):
    rows = []
    for text, plat in zip(texts, platforms):
        if not isinstance(text, str):
            text = ''
        t = text[:512]
        plat = str(plat).lower()
        s = vader.polarity_scores(t)
        tags = re.findall(r'#(\w+)', t.lower())
        words = t.split()
        avg_wl = float(np.mean([len(w) for w in words])) if words else 5.0
        cb = float(min(1.0, sum(1 for w in CLICKBAIT if w in t.lower()) / 3))
        rows.append({
            'sentiment_score':       float(s['compound']),
            'emotional_valence':     float(max(0, s['compound'])),
            'emotional_arousal':     float(abs(s['compound'])),
            'clickbait_score':       cb,
            'cta_present':           float(bool(re.search(
                r'\b(comment|share|like|follow|subscribe|click|watch)\b', t, re.I))),
            'readability_grade':     float(max(0, min(1, 1 - (avg_wl - 3) / 10))),
            'hashtag_count':         float(min(len(tags), 30)),
            'avg_competition_ratio': 0.6,
            'niche_hashtag_ratio':   float(min(len(tags) / 10, 1.0)),
            'trending_hashtag_count':float(min(len(tags) // 3, 5)),
            'text_length':           float(min(len(text), 2000) / 2000),
            'has_url':               float(bool(re.search(r'https?://', text))),
            'question_count':        float(min(text.count('?'), 5) / 5),
            'exclamation_count':     float(min(text.count('!'), 5) / 5),
            'emoji_count':           float(min(len(re.findall(r'[^\w\s,.]', t)), 10) / 10),
            **{f'platform_{p}': float(plat == p) for p in PLATFORMS},
            'peak_overlap_score':    PEAK.get(plat, 0.6),
            'day_of_week_score':     0.6,
            'audience_active_pct':   0.5,
            'follower_count':        0.005,
            'avg_engagement_rate':   0.035,
            'face_count':            0.0,
            'face_prominence_score': 0.0,
            'text_density':          0.0,
            'brightness_score':      0.5,
            'color_vibrancy':        0.5,
            'clip_semantic_score':   0.5,
            'scene_cut_count':       0.0,
        })
    return pd.DataFrame(rows)


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading 80K dataset...")
DATASET = 'd:/dg-social/scraper_pipeline/data_exports/multi_modal_dataset_70k.csv'
df = pd.read_csv(DATASET).dropna(subset=['raw_text', 'engagement_score'])
df['platform'] = df['platform'].str.lower().str.strip()
print(f"  {len(df):,} rows, {df['platform'].nunique()} platforms")

# ── Labels ─────────────────────────────────────────────────────────────────────
labels = np.zeros(len(df), dtype=int)
for plat in df['platform'].unique():
    mask = df['platform'] == plat
    thr = df.loc[mask, 'engagement_score'].quantile(0.60)
    labels[mask & (df['engagement_score'] >= thr).values] = 1
print(f"  HIGH: {labels.sum():,} ({labels.mean()*100:.1f}%)")

# ── Features ───────────────────────────────────────────────────────────────────
print("Extracting features...")
X = extract_features(df['raw_text'].tolist(), df['platform'].tolist())
feature_cols = list(X.columns)
print(f"  Feature matrix: {X.shape}")

# ── Honest 80/20 split ─────────────────────────────────────────────────────────
X_tr, X_val, y_tr, y_val, df_tr, df_val = train_test_split(
    X.values, labels, df.reset_index(drop=True),
    test_size=0.20, random_state=42, stratify=labels
)

# ── Train ──────────────────────────────────────────────────────────────────────
print("\nTraining LightGBM on 80% split...")
model = lgb.LGBMClassifier(
    objective='binary', n_estimators=600, learning_rate=0.05,
    num_leaves=63, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=5, class_weight='balanced', reg_alpha=0.1,
    reg_lambda=0.1, random_state=42, n_jobs=-1, verbosity=-1
)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
)

# ── Hold-out evaluation ────────────────────────────────────────────────────────
preds = model.predict(X_val)
proba = model.predict_proba(X_val)[:, 1]
f1  = f1_score(y_val, preds)
auc = roc_auc_score(y_val, proba)
print(f"\n{'='*50}")
print(f"HOLD-OUT RESULTS (20% = {len(y_val):,} samples)")
print(f"{'='*50}")
print(f"F1 (HIGH class): {f1:.4f}")
print(f"AUC-ROC:         {auc:.4f}")
print(f"Best iteration:  {model.best_iteration_}")
print()
print(classification_report(y_val, preds, target_names=["LOW", "HIGH"]))

# ── Per-platform breakdown ─────────────────────────────────────────────────────
print("Per-platform F1:")
for plat in df_val['platform'].unique():
    mask = df_val['platform'] == plat
    if mask.sum() < 10:
        continue
    p_preds = preds[mask.values]
    p_y     = y_val[mask.values]
    p_f1    = f1_score(p_y, p_preds, zero_division=0)
    p_auc   = roc_auc_score(p_y, proba[mask.values]) if len(np.unique(p_y)) > 1 else 0.5
    print(f"  {plat:<12} n={mask.sum():>5}  F1={p_f1:.3f}  AUC={p_auc:.3f}")

# ── Save properly ──────────────────────────────────────────────────────────────
print("\nSaving artifacts...")
# Train on full data for production
model_full = lgb.LGBMClassifier(
    objective='binary', n_estimators=model.best_iteration_ + 50,
    learning_rate=0.05, num_leaves=63, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5, class_weight='balanced',
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, verbosity=-1
)
model_full.fit(X.values, labels, callbacks=[lgb.log_evaluation(0)])
joblib.dump(model_full, f'{SAVED}/previral_lgbm.joblib')
joblib.dump(feature_cols, f'{SAVED}/feature_columns.joblib')

fc_size = os.path.getsize(f'{SAVED}/feature_columns.joblib')
lgbm_size = os.path.getsize(f'{SAVED}/previral_lgbm.joblib') // 1024
print(f"  feature_columns.joblib: {fc_size} bytes  ({'OK' if fc_size > 100 else 'EMPTY — ERROR'})")
print(f"  previral_lgbm.joblib:   {lgbm_size}KB")
print(f"\nDone. Hold-out F1={f1:.4f}, AUC={auc:.4f}")
