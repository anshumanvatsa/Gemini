"""
Fast-track trainer — loads previral_training_v3.csv and trains LightGBM.
Skips NLP extraction since the CSV was already built by master_train.py.
Run: python models/train_from_csv.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report

SAVED      = 'd:/dg-social/previral/models/saved'
EXPORT_DIR = 'd:/dg-social/scraper_pipeline/data_exports'
CSV_PATH   = f'{EXPORT_DIR}/previral_training_v3.csv'

print("=" * 65)
print("Loading pre-built training CSV...")
print("=" * 65)

df = pd.read_csv(CSV_PATH)
print(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")

# Separate features and label
y = df['label'].values
plat = df['platform'].values
feature_cols = [c for c in df.columns if c not in ['label', 'platform']]
X = df[feature_cols].values

print(f"Features: {len(feature_cols)}")
print(f"Label balance: HIGH={y.mean()*100:.1f}%")
print(f"Platforms: {dict(zip(*np.unique(plat, return_counts=True)))}")

# Quick leakage check
corrs = abs(np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]))
max_corr = float(np.nanmax(corrs))
top_feat = feature_cols[int(np.nanargmax(corrs))]
print(f"\nMax feature-label correlation: {max_corr:.4f} ({top_feat}) -- must be <0.5")

# Train/val split
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
_, val_idx = train_test_split(np.arange(len(df)), test_size=0.20, random_state=42, stratify=y)
plat_val = plat[val_idx]

print(f"\nTrain: {len(y_tr):,}   Val: {len(y_val):,}")

# Phase 5 params (exact spec)
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

print("\nTraining LightGBM v3...")
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
print(f"V3 HOLD-OUT RESULTS  ({len(y_val):,} samples)")
print(f"{'='*65}")
print(f"F1 (HIGH class): {f1:.4f}")
print(f"AUC-ROC:         {auc:.4f}")
print(f"Best iteration:  {model.best_iteration_}")
print()
print(classification_report(y_val, preds, target_names=["LOW", "HIGH"]))

print("Per-platform F1:")
for p in np.unique(plat_val):
    mask = plat_val == p
    if mask.sum() < 20: continue
    pf1  = f1_score(y_val[mask], preds[mask], zero_division=0)
    pauc = roc_auc_score(y_val[mask], proba[mask]) if len(np.unique(y_val[mask])) > 1 else 0.5
    print(f"  {p:<12}  n={mask.sum():>6,}  F1={pf1:.3f}  AUC={pauc:.3f}")

gap = proba[y_val == 1].mean() - proba[y_val == 0].mean()
print(f"\nConfidence gap: {gap:.3f}  (target >0.20)")

fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"\nTop 15 features:")
for feat, imp in fi.head(15).items():
    print(f"  {feat:<35} {imp:>6.0f}")

# Save production artifacts
print("\nSaving v3 artifacts...")
model_full = lgb.LGBMClassifier(**{**params, 'n_estimators': model.best_iteration_ + 100})
model_full.fit(X, y, callbacks=[lgb.log_evaluation(0)])

joblib.dump(model_full,   f'{SAVED}/previral_lgbm_v3.joblib')
joblib.dump(feature_cols, f'{SAVED}/feature_columns_v3.joblib')

print(f"  previral_lgbm_v3.joblib:   {os.path.getsize(SAVED+'/previral_lgbm_v3.joblib')//1024}KB")
print(f"  feature_columns_v3.joblib: {os.path.getsize(SAVED+'/feature_columns_v3.joblib')} bytes")
print(f"\nFINAL: F1={f1:.4f}  AUC={auc:.4f}  Gap={gap:.3f}")
