import pandas as pd, numpy as np

df = pd.read_csv('d:/dg-social/phase2/data/combined_dataset.csv', encoding='latin1', on_bad_lines='skip')
df.columns = [c.lower().strip() for c in df.columns]

print('=== engagement_class distribution ===')
print(df['engagement_class'].value_counts())
print()

# Per-platform stats to understand what's in the data
print('=== Per-platform stats ===')
for plat in df['platform'].unique():
    sub = df[df['platform'] == plat]
    fc_med = sub['follower_count'].median()
    lk_med = sub['total_likes'].median()
    vw_med = sub['total_views'].median()
    er = (sub['total_likes'] + sub['total_comments'] + sub['total_shares']) / sub['follower_count'].clip(1)
    print(f'{plat}: n={len(sub):,}  follower_median={fc_med:.0f}')
    print(f'  likes_median={lk_med:.1f}  views_median={vw_med:.1f}  raw_ER_median={er.median():.4f}')
    print()

# Correlation of raw features with label to find leakage
df['follower_clip'] = df['follower_count'].clip(lower=1)
df['raw_er'] = (df['total_likes'] + df['total_comments'] + df['total_shares']) / df['follower_clip']
plat_med = df.groupby('platform')['raw_er'].median()
df['norm_er'] = df['raw_er'] / df['platform'].map(plat_med).clip(1e-8)
df['our_label'] = (df['norm_er'] > 1.0).astype(int)

print('=== Old vs new label agreement ===')
agree = (df['engagement_class'] == df['our_label']).mean()
print(f'Agreement: {agree*100:.1f}%')
print()

print('=== Feature correlation with label (leakage check) ===')
numeric = df.select_dtypes(include=[np.number])
corr = numeric.corrwith(df['our_label']).abs().sort_values(ascending=False)
print(corr)
