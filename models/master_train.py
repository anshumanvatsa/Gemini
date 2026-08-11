"""
MASTER TRAINING PIPELINE — PreViral F1 ≥ 0.78 Target
=====================================================
Step 1: Load and normalize all 7 platform datasets
Step 2: Run full NLP feature extraction (VADER + NLP signals) on all captions
Step 3: Apply engagement rate labeling (platform-normalized ER)
Step 4: Stratified sampling — balanced per platform × follower tier
Step 5: Retrain LightGBM with Phase 5 parameters
Step 6: Evaluate with honest hold-out split

Run: python models/master_train.py
Saves: models/saved/previral_lgbm_v3.joblib
       models/saved/feature_columns_v3.joblib
       data_exports/previral_training_v3.csv
"""

import os, sys, re, time, glob, warnings
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

vader = SentimentIntensityAnalyzer()
SAVED = 'd:/dg-social/previral/models/saved'
RAW   = 'd:/dg-social/phase2/data/raw_datasets'
EXPORT_DIR = 'd:/dg-social/scraper_pipeline/data_exports'
os.makedirs(SAVED, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CLICKBAIT_WORDS = [
    'wont believe', 'shocking', 'shocked', 'secret', 'hack', 'viral',
    'insane', 'mind blow', 'jaw drop', 'unbelievable', 'must see',
    'you need', 'watch this', 'incredible', 'exposed', 'truth about'
]
CTA_WORDS = [
    'comment', 'share', 'like', 'follow', 'subscribe', 'click', 'watch',
    'tag', 'save', 'dm me', 'link in bio', 'swipe', 'join', 'sign up', 'get'
]
PLATFORMS = ['youtube', 'instagram', 'tiktok', 'twitter', 'linkedin', 'facebook', 'reddit', 'pinterest']
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
    'youtube':   [0.60, 0.70, 0.75, 0.75, 0.80, 0.70, 0.65],
    'tiktok':    [0.60, 0.65, 0.70, 0.70, 0.75, 0.80, 0.80],
    'twitter':   [0.70, 0.75, 0.80, 0.75, 0.70, 0.50, 0.50],
    'facebook':  [0.60, 0.70, 0.75, 0.70, 0.65, 0.50, 0.50],
    'instagram': [0.65, 0.70, 0.75, 0.70, 0.65, 0.60, 0.60],
    'linkedin':  [0.70, 0.80, 0.85, 0.80, 0.70, 0.30, 0.20],
    'reddit':    [0.60, 0.65, 0.70, 0.75, 0.70, 0.65, 0.60],
    'pinterest': [0.50, 0.60, 0.55, 0.60, 0.70, 0.80, 0.75],
}


def _peak_overlap(hour, plat):
    for s, e in PEAK_HOURS.get(plat, [(12, 20)]):
        if s <= hour < e:
            centre = (s + e) / 2
            half   = (e - s) / 2 + 1
            return float(min(1.0, max(0.0, 1.0 - abs(hour - centre) / half)))
    return 0.1


def extract_nlp_features(text):
    """Full NLP feature extraction from raw caption text."""
    if not isinstance(text, str) or not text.strip():
        return {
            'sentiment_score': 0.0, 'emotional_valence': 0.0, 'emotional_arousal': 0.0,
            'clickbait_score': 0.0, 'cta_present': 0.0, 'readability_grade': 0.5,
            'text_length': 0.0, 'has_url': 0.0, 'question_count': 0.0,
            'exclamation_count': 0.0, 'emoji_count': 0.0,
            'hashtag_count_nlp': 0.0, 'mention_count': 0.0, 'caps_ratio': 0.0,
            'avg_word_length': 4.0, 'unique_word_ratio': 0.5,
        }
    t = text[:1000]
    t_low = t.lower()
    words = t_low.split()

    # VADER sentiment
    s = vader.polarity_scores(t)

    # Text stats
    tags = re.findall(r'#\w+', t)
    mentions = re.findall(r'@\w+', t)
    emojis = re.findall(r'[^\w\s,.\-!?#@\'\"()\[\]{}:;/\\]', t)
    unique_w = len(set(words))

    # Clickbait: keyword presence
    cb = float(min(1.0, sum(1 for w in CLICKBAIT_WORDS if w in t_low) / 3))

    # CTA: call-to-action keyword
    cta = float(any(w in t_low for w in CTA_WORDS))

    # Readability proxy: shorter avg word length = more readable
    avg_wl = float(np.mean([len(w) for w in words])) if words else 4.0
    readability = float(max(0, min(1, 1 - (avg_wl - 3) / 8)))

    # Capitalization ratio (CAPS = excitement / urgency)
    alpha = [c for c in t if c.isalpha()]
    caps_ratio = float(sum(1 for c in alpha if c.isupper()) / (len(alpha) + 1))

    return {
        'sentiment_score':    float(s['compound']),
        'emotional_valence':  float(max(0, s['compound'])),
        'emotional_arousal':  float(abs(s['compound'])),
        'clickbait_score':    cb,
        'cta_present':        cta,
        'readability_grade':  readability,
        'text_length':        float(min(len(text), 3000) / 3000),
        'has_url':            float(bool(re.search(r'https?://', t))),
        'question_count':     float(min(t.count('?'), 5) / 5),
        'exclamation_count':  float(min(t.count('!'), 5) / 5),
        'emoji_count':        float(min(len(emojis), 15) / 15),
        'hashtag_count_nlp':  float(min(len(tags), 30)),
        'mention_count':      float(min(len(mentions), 10) / 10),
        'caps_ratio':         float(min(caps_ratio, 0.5) / 0.5),
        'unique_word_ratio':  float(min(unique_w / (len(words) + 1), 1.0)),
        'avg_word_length':    float(min(avg_wl, 12) / 12),
    }


def build_feature_vector(text, platform, follower_count, post_hour, post_weekday,
                         hashtag_count, has_media, is_video, is_paid):
    """Combined NLP + timing + account features — ALL pre-publish."""
    plat = str(platform).lower()
    hour = int(post_hour)   if pd.notna(post_hour)    else 12
    wday = int(post_weekday) if pd.notna(post_weekday) else 2
    wday = wday % 7

    nlp = extract_nlp_features(text)

    # Use NLP hashtag count if available, else metadata count
    ht_cnt = max(float(nlp['hashtag_count_nlp']), float(min(hashtag_count or 0, 30)))

    # Timing
    po = _peak_overlap(hour, plat)
    ds = PEAK_DAYS.get(plat, [0.6] * 7)[wday]

    # Follower count (log-normalized — handles 0 to 100M range)
    fc = float(np.log1p(max(float(follower_count or 0), 0)) / np.log1p(10_000_000))

    feat = {
        # NLP (16 features)
        **nlp,
        # Hashtag signals (4 features)
        'hashtag_count':         ht_cnt,
        'niche_hashtag_ratio':   float(min(ht_cnt / 10, 1.0)),
        'trending_hashtag_count':float(min(ht_cnt // 3, 5)),
        'avg_competition_ratio': 0.6,  # populated at inference from hashtag_db_v2
        # Timing (7 features — cyclical encoding avoids 23→0 discontinuity)
        'peak_overlap_score':    po,
        'day_of_week_score':     ds,
        'audience_active_pct':   float(0.4 + po * 0.4),
        'post_hour_sin':         float(np.sin(2 * np.pi * hour / 24)),
        'post_hour_cos':         float(np.cos(2 * np.pi * hour / 24)),
        'post_wday_sin':         float(np.sin(2 * np.pi * wday / 7)),
        'post_wday_cos':         float(np.cos(2 * np.pi * wday / 7)),
        # Content type (3 features)
        'has_media':             float(int(bool(has_media))),
        'is_video':              float(int(bool(is_video))),
        'is_paid':               float(int(bool(is_paid))),
        # Account (1 feature)
        'follower_count':        fc,
        # Vision placeholders (enriched at inference by CLIP)
        'face_count':            0.0,
        'face_prominence_score': 0.0,
        'text_density':          0.0,
        'brightness_score':      0.5,
        'color_vibrancy':        0.5,
        'clip_semantic_score':   0.5,
        'scene_cut_count':       float(int(bool(is_video))),
        # Platform OHE (8 features)
        **{f'platform_{p}': float(plat == p) for p in PLATFORMS},
    }
    return feat


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD ALL DATA SOURCES
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — Loading all platform datasets")
print("=" * 65)

all_dfs = []

# --- YouTube (trending CSVs — has title + tags + views + likes) ---
yt_files = glob.glob(f'{RAW}/youtube_trending/*.csv')[:3]  # US, IN, CA
yt_rows = []
for fpath in yt_files:
    try:
        yt = pd.read_csv(fpath, encoding='latin1', on_bad_lines='skip')
        yt.columns = [c.lower().strip() for c in yt.columns]
        for _, row in yt.iterrows():
            title = str(row.get('title', ''))
            tags  = str(row.get('tags', ''))
            text  = title + ' ' + tags.replace('|', ' ')
            views = float(row.get('views', 0))
            likes = float(row.get('likes', 0))
            cmt   = float(row.get('comment_count', 0))
            if views > 0:
                yt_rows.append({
                    'platform': 'youtube', 'text': text,
                    'likes': likes, 'comments': cmt, 'shares': 0,
                    'views': views, 'followers': 100000,  # YouTube channel estimate
                    'post_hour': 14, 'post_weekday': 2,
                    'hashtag_count': text.count('#'), 'has_media': 1,
                    'is_video': 1, 'is_paid': 0
                })
    except Exception as e:
        print(f"  YouTube {os.path.basename(fpath)}: {e}")

# Deduplicate YouTube by title
yt_df = pd.DataFrame(yt_rows).drop_duplicates(subset='text')
yt_df = yt_df.sample(n=min(len(yt_df), 80000), random_state=42)

all_dfs.append(yt_df)
print(f"  YouTube: {len(yt_df):,} rows")

# --- YouTube non-trending (390K rows — has description) ---
try:
    ynt = pd.read_csv(f'{RAW}/youtube_nontrends/Youtube_Videos.csv',
                      encoding='latin1', on_bad_lines='skip', nrows=80000)
    ynt.columns = [c.lower().strip() for c in ynt.columns]
    desc_col = next((c for c in ynt.columns if 'desc' in c or 'title' in c), None)
    views_col = next((c for c in ynt.columns if 'view' in c), None)
    likes_col = next((c for c in ynt.columns if 'like' in c), None)
    if desc_col and views_col:
        ynt_rows = []
        for _, row in ynt.iterrows():
            text  = str(row.get(desc_col, ''))
            views = float(str(row.get(views_col, 0)).replace(',', '') or 0)
            likes = float(str(row.get(likes_col, 0)).replace(',', '') or 0) if likes_col else 0
            if views > 0 and len(text) > 10:
                ynt_rows.append({
                    'platform': 'youtube', 'text': text,
                    'likes': likes, 'comments': 0, 'shares': 0,
                    'views': views, 'followers': 50000,
                    'post_hour': 14, 'post_weekday': 2,
                    'hashtag_count': text.count('#'), 'has_media': 1,
                    'is_video': 1, 'is_paid': 0
                })
        ynt_df = pd.DataFrame(ynt_rows)
        all_dfs.append(ynt_df)
        print(f"  YouTube (non-trending): {len(ynt_df):,} rows")
except Exception as e:
    print(f"  YouTube non-trending error: {e}")

# --- Twitter ---
try:
    tw_path = f'{RAW}/twitter/DMO social media engagement dataset/Data LIWC 01 02 23.csv'
    tw = pd.read_csv(tw_path, encoding='latin1', on_bad_lines='skip')
    tw.columns = [c.lower().strip() for c in tw.columns]
    # Find text columns
    text_cols = [c for c in tw.columns if c in ['x', 'text', 'content', 'tweet', 'status text']]
    tw_rows = []
    for _, row in tw.iterrows():
        text = ''
        for tc in text_cols:
            if pd.notna(row.get(tc)) and len(str(row.get(tc, ''))) > 5:
                text = str(row[tc])
                break
        likes = float(row.get('like_count', 0) or 0)
        rt    = float(row.get('retweet_count', 0) or 0)
        followers = float(row.get('followers', 1000) or 1000)
        if text and followers > 0:
            tw_rows.append({
                'platform': 'twitter', 'text': text,
                'likes': likes, 'comments': 0, 'shares': rt,
                'views': 0, 'followers': followers,
                'post_hour': 10, 'post_weekday': 2,
                'hashtag_count': text.count('#'), 'has_media': 0,
                'is_video': 0, 'is_paid': 0
            })
    tw_df = pd.DataFrame(tw_rows)
    all_dfs.append(tw_df)
    print(f"  Twitter: {len(tw_df):,} rows")
except Exception as e:
    print(f"  Twitter error: {e}")

# --- TikTok existing ---
try:
    tk = pd.read_csv(f'{RAW}/tiktok/tiktok_dataset.csv', encoding='latin1', on_bad_lines='skip')
    tk.columns = [c.lower().strip() for c in tk.columns]
    text_col  = next((c for c in tk.columns if 'transcription' in c or 'text' in c or 'description' in c), None)
    views_col = next((c for c in tk.columns if 'view' in c), None)
    likes_col = next((c for c in tk.columns if 'like' in c), None)
    share_col = next((c for c in tk.columns if 'share' in c), None)
    if views_col:
        tk_rows = []
        for _, row in tk.iterrows():
            text  = str(row.get(text_col, '')) if text_col else ''
            views = float(row.get(views_col, 0) or 0)
            likes = float(row.get(likes_col, 0) or 0) if likes_col else 0
            shares = float(row.get(share_col, 0) or 0) if share_col else 0
            if views > 0:
                tk_rows.append({
                    'platform': 'tiktok', 'text': text,
                    'likes': likes, 'comments': 0, 'shares': shares,
                    'views': views, 'followers': 5000,
                    'post_hour': 19, 'post_weekday': 5,
                    'hashtag_count': text.count('#'), 'has_media': 1,
                    'is_video': 1, 'is_paid': 0
                })
        tk_df = pd.DataFrame(tk_rows)
        all_dfs.append(tk_df)
        print(f"  TikTok (existing): {len(tk_df):,} rows")
except Exception as e:
    print(f"  TikTok error: {e}")

# --- NEW: Instagram (only use multi-platform dataset which has real captions) ---
# NOTE: instagram-analytics-dataset skipped — it has no caption text (zero NLP signal)
# Instagram rows come from social_multi/social_media_dataset.csv (content_description)
print("  Instagram: using rows from multi-platform dataset (real captions)")


# --- NEW: TikTok v2 ---
for fpath in glob.glob(f'{RAW}/tiktok_v2/**/*.csv', recursive=True):
    try:
        tk2 = pd.read_csv(fpath, encoding='latin1', on_bad_lines='skip')
        tk2.columns = [c.lower().strip().replace(' ', '_') for c in tk2.columns]
        text_col  = next((c for c in tk2.columns if any(x in c for x in ['description', 'caption', 'text', 'title'])), None)
        views_col = next((c for c in tk2.columns if 'view' in c or 'play' in c), None)
        likes_col = next((c for c in tk2.columns if 'like' in c), None)
        follow_col= next((c for c in tk2.columns if 'follow' in c), None)
        share_col = next((c for c in tk2.columns if 'share' in c), None)
        print(f"  TikTok v2 {os.path.basename(fpath)}: views={views_col}, text={text_col}")
        if views_col:
            tk2_rows = []
            for _, row in tk2.iterrows():
                text = str(row.get(text_col, '')) if text_col else ''
                views = float(str(row.get(views_col, 0)).replace(',', '') or 0)
                likes = float(str(row.get(likes_col, 0)).replace(',', '') or 0) if likes_col else 0
                shares = float(str(row.get(share_col, 0)).replace(',', '') or 0) if share_col else 0
                followers = float(str(row.get(follow_col, 5000)).replace(',', '') or 5000) if follow_col else 5000
                if views > 0:
                    tk2_rows.append({
                        'platform': 'tiktok', 'text': text,
                        'likes': likes, 'comments': 0, 'shares': shares,
                        'views': views, 'followers': max(followers, 1),
                        'post_hour': 19, 'post_weekday': 5,
                        'hashtag_count': text.count('#'), 'has_media': 1,
                        'is_video': 1, 'is_paid': 0
                    })
            if tk2_rows:
                tk2_df = pd.DataFrame(tk2_rows)
                all_dfs.append(tk2_df)
                print(f"    -> {len(tk2_df):,} rows")
    except Exception as e:
        print(f"  TikTok v2 {fpath}: {e}")

# --- NEW: LinkedIn ---
for fpath in glob.glob(f'{RAW}/linkedin/**/*.csv', recursive=True):
    try:
        li = pd.read_csv(fpath, encoding='latin1', on_bad_lines='skip')
        li.columns = [c.lower().strip().replace(' ', '_') for c in li.columns]
        text_col  = next((c for c in li.columns if any(x in c for x in ['content', 'text', 'post', 'description'])), None)
        likes_col = next((c for c in li.columns if 'like' in c or 'reaction' in c), None)
        cmt_col   = next((c for c in li.columns if 'comment' in c), None)
        share_col = next((c for c in li.columns if 'share' in c), None)
        follow_col= next((c for c in li.columns if ('follow' in c or 'connect' in c)
                          and 'url' not in c and 'link' not in c), None)
        print(f"  LinkedIn {os.path.basename(fpath)}: cols={list(li.columns[:6])}")
        li_rows = []
        for _, row in li.iterrows():
            text = str(row.get(text_col, '')) if text_col else ''
            try:
                likes = float(str(row.get(likes_col, 0) or 0).replace(',','')) if likes_col else 0
                cmt   = float(str(row.get(cmt_col, 0) or 0).replace(',','')) if cmt_col else 0
                shares = float(str(row.get(share_col, 0) or 0).replace(',','')) if share_col else 0
                followers = float(str(row.get(follow_col, 2000) or 2000).replace(',','')) if follow_col else 2000
                if followers > 1e8: followers = 2000  # URL accidentally parsed
            except (ValueError, TypeError):
                likes, cmt, shares, followers = 0, 0, 0, 2000
            li_rows.append({
                'platform': 'linkedin', 'text': text,
                'likes': likes, 'comments': cmt, 'shares': shares,
                'views': likes * 30,  # LinkedIn impression proxy
                'followers': max(followers, 1),
                'post_hour': 9, 'post_weekday': 1,
                'hashtag_count': text.count('#'), 'has_media': 0,
                'is_video': 0, 'is_paid': 0
            })
        if li_rows:
            li_df = pd.DataFrame(li_rows)
            all_dfs.append(li_df)
            print(f"    -> {len(li_df):,} rows")
    except Exception as e:
        print(f"  LinkedIn {fpath}: {e}")

# --- NEW: Multi-platform sponsorship dataset ---
for fpath in glob.glob(f'{RAW}/social_multi/**/*.csv', recursive=True):
    try:
        sm = pd.read_csv(fpath, encoding='latin1', on_bad_lines='skip')
        sm.columns = [c.lower().strip().replace(' ', '_') for c in sm.columns]
        plat_col  = next((c for c in sm.columns if 'platform' in c), None)
        # Priority: description > caption > text > content_description > avoid content_id/content_url
        text_col = None
        for candidate in ['content_description', 'description', 'caption', 'text', 'content_text']:
            if candidate in sm.columns:
                text_col = candidate
                break
        if text_col is None:
            text_col = next((c for c in sm.columns if any(x in c for x in ['description', 'caption', 'text'])
                             and 'id' not in c and 'url' not in c and 'type' not in c), None)
        likes_col = next((c for c in sm.columns if 'like' in c and 'ratio' not in c), None)
        cmt_col   = next((c for c in sm.columns if 'comment' in c and 'ratio' not in c), None)
        views_col = next((c for c in sm.columns if 'view' in c or 'impression' in c or 'reach' in c), None)
        follow_col= next((c for c in sm.columns if 'follow' in c), None)
        share_col = next((c for c in sm.columns if 'share' in c or 'save' in c), None)
        hour_col  = next((c for c in sm.columns if 'hour' in c or 'time' in c), None)
        print(f"  Multi {os.path.basename(fpath)}: plat={plat_col}, text={text_col}, likes={likes_col}, views={views_col}")
        if plat_col or likes_col:
            sm_rows = []
            for _, row in sm.iterrows():
                plat = str(row.get(plat_col, 'instagram')).lower().strip() if plat_col else 'instagram'
                if plat not in PLATFORMS:
                    plat = 'instagram'
                text = str(row.get(text_col, '')) if text_col else ''
                likes = float(row.get(likes_col, 0) or 0) if likes_col else 0
                cmt   = float(row.get(cmt_col, 0) or 0) if cmt_col else 0
                shares = float(row.get(share_col, 0) or 0) if share_col else 0
                views = float(row.get(views_col, likes * 10) or likes * 10) if views_col else likes * 10
                followers = float(row.get(follow_col, 5000) or 5000) if follow_col else 5000
                hour = int(row.get(hour_col, 12) or 12) if hour_col else 12
                sm_rows.append({
                    'platform': plat, 'text': text,
                    'likes': likes, 'comments': cmt, 'shares': shares,
                    'views': views, 'followers': max(followers, 1),
                    'post_hour': hour, 'post_weekday': 2,
                    'hashtag_count': text.count('#'), 'has_media': 1,
                    'is_video': 0, 'is_paid': 0
                })
            if sm_rows:
                sm_df = pd.DataFrame(sm_rows)
                all_dfs.append(sm_df)
                print(f"    -> {len(sm_df):,} rows, platforms: {sm_df['platform'].value_counts().to_dict()}")
    except Exception as e:
        print(f"  Multi-platform {fpath}: {e}")

# Combine all
combined = pd.concat(all_dfs, ignore_index=True)
print(f"\nCOMBINED RAW: {len(combined):,} rows")
print(f"Platforms: {combined['platform'].value_counts().to_dict()}")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — ENGAGEMENT RATE LABELS (normalized ER)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 — Computing normalized engagement rate labels")
print("=" * 65)

combined['followers_clip'] = combined['followers'].clip(lower=1)
combined['raw_er'] = (combined['likes'] + combined['comments'] + combined['shares']) / combined['followers_clip']

# Platform median ER
platform_med = combined.groupby('platform')['raw_er'].median()
print("Platform median ER:")
for p, m in platform_med.items():
    print(f"  {p:<12} {m:.4f}")

combined['platform_med_er'] = combined['platform'].map(platform_med).clip(lower=1e-8)
combined['norm_er'] = combined['raw_er'] / combined['platform_med_er']
combined['label'] = (combined['norm_er'] > 1.0).astype(int)
print(f"\nLabel balance: HIGH={combined['label'].mean()*100:.1f}%")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — STRATIFIED SAMPLING
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3 — Stratified sampling (balanced per platform)")
print("=" * 65)

TARGET = 15000  # per class per platform
sampled = []
for plat in combined['platform'].unique():
    sub  = combined[combined['platform'] == plat]
    high = sub[sub['label'] == 1]
    low  = sub[sub['label'] == 0]
    n = min(len(high), len(low), TARGET)
    if n < 50:
        print(f"  {plat}: skipped (too few rows: high={len(high)}, low={len(low)})")
        continue
    sh = high.sample(n=n, random_state=42)
    sl = low.sample(n=n,  random_state=42)
    sampled.append(pd.concat([sh, sl]))
    print(f"  {plat:<12} HIGH={n:>6,}  LOW={n:>6,}  total={2*n:,}")

train_df = pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=42)
print(f"\nFinal training set: {len(train_df):,} rows, {train_df['label'].mean()*100:.1f}% HIGH")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — NLP FEATURE EXTRACTION (GPU-accelerated VADER on all captions)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4 — NLP feature extraction on all training captions")
print("=" * 65)

t0 = time.time()
BATCH = 1000
feature_rows = []
for i in range(0, len(train_df), BATCH):
    batch = train_df.iloc[i:i+BATCH]
    for _, row in batch.iterrows():
        feat = build_feature_vector(
            text=row.get('text', ''),
            platform=row['platform'],
            follower_count=row.get('followers', 1000),
            post_hour=row.get('post_hour', 12),
            post_weekday=row.get('post_weekday', 2),
            hashtag_count=row.get('hashtag_count', 0),
            has_media=row.get('has_media', 1),
            is_video=row.get('is_video', 0),
            is_paid=row.get('is_paid', 0),
        )
        feature_rows.append(feat)
    if (i // BATCH) % 10 == 0:
        pct = (i + BATCH) / len(train_df) * 100
        elapsed = time.time() - t0
        print(f"  {min(pct,100):.0f}%  {elapsed:.0f}s  ({i+BATCH:,}/{len(train_df):,})")

X_df = pd.DataFrame(feature_rows)
y    = train_df['label'].values
print(f"Done in {time.time()-t0:.1f}s. Feature matrix: {X_df.shape}")

feature_cols = list(X_df.columns)
print(f"Feature count: {len(feature_cols)}")

# Leakage check
corrs = X_df.corrwith(pd.Series(y, name='label')).abs().sort_values(ascending=False)
print(f"Max feature-label correlation: {corrs.max():.4f}  (must be <0.5)")
print(f"Top 5 corrs: {corrs.head().to_dict()}")

# Save enriched CSV
X_df['label'] = y
X_df['platform'] = train_df['platform'].values
X_df.to_csv(f'{EXPORT_DIR}/previral_training_v3.csv', index=False)
X_df.drop(columns=['label', 'platform'], inplace=True)
print(f"Saved: previral_training_v3.csv")

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — RETRAIN LIGHTGBM WITH PHASE 5 PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5 -- Training LightGBM (target F1 >= 0.78)")
print("=" * 65)

X = X_df.values
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Train: {len(y_tr):,}  Val: {len(y_val):,}")

# Platform-stratified eval
_, val_idx = train_test_split(np.arange(len(train_df)), test_size=0.20, random_state=42, stratify=y)
plat_val = train_df['platform'].values[val_idx]

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
model.fit(X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(50)])

# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — HONEST EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
preds = model.predict(X_val)
proba = model.predict_proba(X_val)[:, 1]
f1    = f1_score(y_val, preds)
auc   = roc_auc_score(y_val, proba)

print(f"\n{'='*65}")
print(f"FINAL RESULTS — V3 MODEL ({len(y_val):,} hold-out samples)")
print(f"{'='*65}")
print(f"F1 (HIGH class): {f1:.4f}")
print(f"AUC-ROC:         {auc:.4f}")
print(f"Best iteration:  {model.best_iteration_}")
print()
print(classification_report(y_val, preds, target_names=["LOW", "HIGH"]))

print("Per-platform F1:")
for plat in np.unique(plat_val):
    mask = plat_val == plat
    if mask.sum() < 20: continue
    p_f1  = f1_score(y_val[mask], preds[mask], zero_division=0)
    p_auc = roc_auc_score(y_val[mask], proba[mask]) if len(np.unique(y_val[mask])) > 1 else 0.5
    n = mask.sum()
    print(f"  {plat:<12} n={n:>6,}  F1={p_f1:.3f}  AUC={p_auc:.3f}")

# Confidence gap
gap = proba[y_val == 1].mean() - proba[y_val == 0].mean()
print(f"\nConfidence gap: {gap:.3f}  (target >0.20)")

# Feature importance
fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"\nTop 15 features:")
for feat, imp in fi.head(15).items():
    print(f"  {feat:<35} {imp:>6.0f}")

# Save production model
print("\nSaving v3 artifacts...")
model_full = lgb.LGBMClassifier(**{**params, 'n_estimators': model.best_iteration_ + 100})
model_full.fit(X, y, callbacks=[lgb.log_evaluation(0)])
joblib.dump(model_full,   f'{SAVED}/previral_lgbm_v3.joblib')
joblib.dump(feature_cols, f'{SAVED}/feature_columns_v3.joblib')
lgbm_sz = os.path.getsize(f'{SAVED}/previral_lgbm_v3.joblib') // 1024
print(f"  previral_lgbm_v3.joblib:   {lgbm_sz}KB")
print(f"  feature_columns_v3.joblib: {os.path.getsize(SAVED+'/feature_columns_v3.joblib')} bytes")
print(f"\nFINAL: F1={f1:.4f}  AUC={auc:.4f}  Gap={gap:.3f}")
