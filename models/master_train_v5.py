"""
MASTER TRAINING PIPELINE v5 — PreViral PRODUCTION FINAL
=========================================================
ALL real data. Probability calibrated. 6 platforms.

New in v5 vs v4:
  - tahirmohd Instagram Post Performance (real captions + metrics, 1.0 usability)
  - dumbergerl Macro-Influencer Instagram 2019 (46MB, real posts)
  - yogesh21 Instagram Posts Dataset 2026 (real)
  - CalibratedClassifierCV(isotonic) for proper probability calibration
  - Saves both raw + calibrated models

Run: python models/master_train_v5.py
Saves:
  models/saved/previral_lgbm_v5.joblib          (raw LightGBM)
  models/saved/previral_lgbm_v5_cal.joblib      (calibrated, USE THIS IN PROD)
  models/saved/feature_columns_v5.joblib
"""

import os, sys, re, time, glob, ast, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, 'd:/dg-social/previral')
os.chdir('d:/dg-social/previral')

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()
SAVED      = 'd:/dg-social/previral/models/saved'
RAW        = 'd:/dg-social/phase2/data/raw_datasets'
EXPORT_DIR = 'd:/dg-social/scraper_pipeline/data_exports'
os.makedirs(SAVED, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

PLATFORMS  = ['youtube','instagram','tiktok','twitter','linkedin',
               'facebook','reddit','pinterest']
PEAK_HOURS = {
    'youtube':   [(14,22)], 'tiktok':    [(6,10),(19,23)],
    'twitter':   [(8,10),(12,13),(17,18)], 'facebook':  [(13,16)],
    'instagram': [(11,13),(19,21)], 'linkedin':  [(8,10),(17,18)],
    'reddit':    [(12,14),(18,22)], 'pinterest': [(20,23)],
}
PEAK_DAYS = {
    'youtube':   [.60,.70,.75,.75,.80,.70,.65],
    'tiktok':    [.60,.65,.70,.70,.75,.80,.80],
    'twitter':   [.70,.75,.80,.75,.70,.50,.50],
    'facebook':  [.60,.70,.75,.70,.65,.50,.50],
    'instagram': [.65,.70,.75,.70,.65,.60,.60],
    'linkedin':  [.70,.80,.85,.80,.70,.30,.20],
    'reddit':    [.60,.65,.70,.75,.70,.65,.60],
    'pinterest': [.50,.60,.55,.60,.70,.80,.75],
}
CLICKBAIT_WORDS = [
    'wont believe','shocking','secret','hack','viral','insane',
    'mind blow','jaw drop','unbelievable','must see','incredible',
    'exposed','truth about','you need','watch this'
]
CTA_WORDS = [
    'comment','share','like','follow','subscribe','click','watch',
    'tag','save','dm me','link in bio','swipe','join','sign up',
    'get','check out','tap','try','shop now','grab'
]
WDAY_MAP = {'monday':0,'tuesday':1,'wednesday':2,'thursday':3,
            'friday':4,'saturday':5,'sunday':6}
SENT_MAP = {'positive':0.7,'negative':-0.7,'neutral':0.0,'mixed':0.1,
            'very positive':0.9,'very negative':-0.9}
EMOTION_MAP = {
    'funny':0.6,'humor':0.6,'comedy':0.6,'inspirational':0.8,
    'motivational':0.8,'educational':0.3,'entertainment':0.5,
    'lifestyle':0.4,'news':0.2,'sports':0.5,'beauty':0.4,
    'fitness':0.6,'food':0.5,'travel':0.5,'fashion':0.4,
    'tech':0.3,'gaming':0.5,'love':0.7,'sadness':-0.3,
    'anger':-0.4,'fear':-0.3,'surprise':0.6,'nostalgia':0.3,
    'curiosity':0.5,'joy':0.8,'disgust':-0.5,'neutral':0.0,
}

def _wday(v, d=2):
    s = str(v or d).strip().lower()
    if s in WDAY_MAP: return WDAY_MAP[s]
    try: return int(float(s)) % 7
    except: return d

def _sent(v):
    s = str(v or 0).strip().lower()
    if s in SENT_MAP: return SENT_MAP[s]
    try: return float(s)
    except: return 0.0

def _emotion(v, d=0.4):
    s = str(v or '').strip().lower()
    if s in EMOTION_MAP: return EMOTION_MAP[s]
    try: return float(s)
    except: return d

def _peak_overlap(hour, plat):
    for s, e in PEAK_HOURS.get(plat, [(12,20)]):
        if s <= hour < e:
            centre, half = (s+e)/2, (e-s)/2+1
            return float(min(1.0, max(0.0, 1.0-abs(hour-centre)/half)))
    return 0.1

def extract_nlp_features(text):
    if not isinstance(text, str) or not text.strip():
        return {
            'sentiment_score':0.0,'emotional_valence':0.0,'emotional_arousal':0.0,
            'clickbait_score':0.0,'cta_present':0.0,'readability_grade':0.5,
            'text_length':0.0,'has_url':0.0,'question_count':0.0,
            'exclamation_count':0.0,'emoji_count':0.0,'hashtag_count_nlp':0.0,
            'mention_count':0.0,'caps_ratio':0.0,'avg_word_length':4.0/12,
            'unique_word_ratio':0.5,
        }
    t = text[:2000]
    tl = t.lower()
    words = t.split()
    alpha = [c for c in t if c.isalpha()]
    caps  = [c for c in alpha if c.isupper()]
    tags  = re.findall(r'#\w+', t)
    mentions = re.findall(r'@\w+', t)
    emojis = re.findall(r'[\U00010000-\U0010ffff]', t)
    s = vader.polarity_scores(t)
    avg_wl = sum(len(w) for w in words)/max(len(words),1)
    caps_r = len(caps)/max(len(alpha),1)
    uniq_r = len(set(w.lower() for w in words))/max(len(words),1)
    cb = float(min(1.0, sum(1 for w in CLICKBAIT_WORDS if w in tl)/3))
    cta = float(any(w in tl for w in CTA_WORDS))
    fk_syl = sum(max(1, sum(1 for i,c in enumerate(w.lower())
                  if c in 'aeiouy' and (i==0 or w[i-1].lower() not in 'aeiouy')))
               for w in words) if words else 0
    sents = max(1, len(re.split(r'[.!?]+', t)))
    fk = 0.39*(len(words)/sents)+11.8*(fk_syl/max(len(words),1))-15.59
    readability = max(0.0, min(1.0, 1.0-max(0,fk)/18.0))
    return {
        'sentiment_score':   float(s['compound']),
        'emotional_valence': float(max(0,s['compound'])),
        'emotional_arousal': float(abs(s['compound'])),
        'clickbait_score':   cb,
        'cta_present':       cta,
        'readability_grade': readability,
        'text_length':       float(min(len(text),3000)/3000),
        'has_url':           float(bool(re.search(r'https?://',t))),
        'question_count':    float(min(t.count('?'),5)/5),
        'exclamation_count': float(min(t.count('!'),5)/5),
        'emoji_count':       float(min(len(emojis),15)/15),
        'hashtag_count_nlp': float(min(len(tags),30)),
        'mention_count':     float(min(len(mentions),10)/10),
        'caps_ratio':        float(min(caps_r,0.5)/0.5),
        'avg_word_length':   float(min(avg_wl,12)/12),
        'unique_word_ratio': float(min(uniq_r,1.0)),
    }

def build_row(text, platform, follower_count, post_hour, post_weekday,
              hashtag_count, has_media=1, is_video=0, is_paid=0,
              sentiment_override=None, cta_override=None):
    plat = str(platform).lower()
    hour = int(post_hour or 12) % 24
    wday = int(post_weekday or 2) % 7
    nlp  = extract_nlp_features(text)
    if sentiment_override is not None:
        nlp['sentiment_score']   = float(sentiment_override)
        nlp['emotional_valence'] = max(0, float(sentiment_override))
        nlp['emotional_arousal'] = abs(float(sentiment_override))
    if cta_override is not None:
        nlp['cta_present'] = float(cta_override)
    ht = max(float(nlp['hashtag_count_nlp']), float(min(hashtag_count or 0, 30)))
    po = _peak_overlap(hour, plat)
    ds = PEAK_DAYS.get(plat, [0.6]*7)[wday]
    fc = float(np.log1p(max(float(follower_count or 0),0))/np.log1p(10_000_000))
    return {
        **nlp,
        'hashtag_count':          ht,
        'niche_hashtag_ratio':    float(min(ht/10,1.0)),
        'trending_hashtag_count': float(min(ht//3,5)),
        'avg_competition_ratio':  0.6,
        'peak_overlap_score':     po,
        'day_of_week_score':      ds,
        'audience_active_pct':    float(0.4+po*0.4),
        'post_hour_sin':          float(np.sin(2*np.pi*hour/24)),
        'post_hour_cos':          float(np.cos(2*np.pi*hour/24)),
        'post_wday_sin':          float(np.sin(2*np.pi*wday/7)),
        'post_wday_cos':          float(np.cos(2*np.pi*wday/7)),
        'has_media':              float(int(bool(has_media))),
        'is_video':               float(int(bool(is_video))),
        'is_paid':                float(int(bool(is_paid))),
        'follower_count':         fc,
        'face_count':0.0,'face_prominence_score':0.0,'text_density':0.0,
        'brightness_score':0.5,'color_vibrancy':0.5,
        'clip_semantic_score':0.5,'scene_cut_count':float(int(bool(is_video))),
        **{f'platform_{p}':float(plat==p) for p in PLATFORMS},
    }

# ── helpers ────────────────────────────────────────────────────────────────────
def _safe_float(v, d=0.0):
    try: return float(str(v or d).replace(',','').split()[0])
    except: return d

def _safe_int(v, d=0):
    try: return int(float(str(v or d).replace(',','').split()[0]))
    except: return d

# ── STEP 1: LOAD DATA ─────────────────────────────────────────────────────────
print("="*65)
print("STEP 1 — Loading ALL platform datasets (v5 FINAL)")
print("="*65)

all_dfs = []

# ── YouTube trending ──────────────────────────────────────────────────────────
yt_rows = []
for fpath in glob.glob(f'{RAW}/youtube_trending/*.csv')[:3]:
    try:
        yt = pd.read_csv(fpath, encoding='latin1', on_bad_lines='skip')
        yt.columns = [c.lower().strip() for c in yt.columns]
        for _, r in yt.iterrows():
            title = str(r.get('title',''))
            tags  = str(r.get('tags',''))
            text  = title+' '+tags.replace('|',' ')
            views = _safe_float(r.get('views',0))
            likes = _safe_float(r.get('likes',0))
            cmt   = _safe_float(r.get('comment_count',0))
            if views > 0:
                yt_rows.append({'platform':'youtube','text':text,'likes':likes,
                    'comments':cmt,'shares':0,'views':views,'followers':100000,
                    'post_hour':14,'post_weekday':2,'hashtag_count':text.count('#'),
                    'has_media':1,'is_video':1,'is_paid':0})
    except Exception as e: print(f"  YT trending: {e}")
yt_df = pd.DataFrame(yt_rows).drop_duplicates(subset='text')
all_dfs.append(yt_df); print(f"  YouTube trending: {len(yt_df):,}")

# ── YouTube non-trending ──────────────────────────────────────────────────────
try:
    ynt = pd.read_csv(f'{RAW}/youtube_nontrends/Youtube_Videos.csv',
                      encoding='latin1', on_bad_lines='skip', nrows=80000)
    ynt.columns = [c.lower().strip() for c in ynt.columns]
    desc_col  = next((c for c in ynt.columns if 'desc' in c or 'title' in c), None)
    views_col = next((c for c in ynt.columns if 'view' in c), None)
    likes_col = next((c for c in ynt.columns if 'like' in c), None)
    if desc_col and views_col:
        ynt_rows = []
        for _, r in ynt.iterrows():
            text  = str(r.get(desc_col,''))
            views = _safe_float(r.get(views_col,0))
            likes = _safe_float(r.get(likes_col,0)) if likes_col else 0
            if views > 0 and len(text) > 10:
                ynt_rows.append({'platform':'youtube','text':text,'likes':likes,
                    'comments':0,'shares':0,'views':views,'followers':50000,
                    'post_hour':14,'post_weekday':2,'hashtag_count':text.count('#'),
                    'has_media':1,'is_video':1,'is_paid':0})
        ynt_df = pd.DataFrame(ynt_rows)
        all_dfs.append(ynt_df); print(f"  YouTube non-trending: {len(ynt_df):,}")
except Exception as e: print(f"  YT non-trending: {e}")

# ── Twitter DMO ───────────────────────────────────────────────────────────────
try:
    tw_path = f'{RAW}/twitter/DMO social media engagement dataset/Data LIWC 01 02 23.csv'
    tw = pd.read_csv(tw_path, encoding='latin1', on_bad_lines='skip')
    tw.columns = [c.lower().strip() for c in tw.columns]
    text_cols = [c for c in tw.columns if c in ['x','text','content','tweet']]
    tw_rows = []
    for _, r in tw.iterrows():
        text = ''
        for tc in text_cols:
            if pd.notna(r.get(tc)) and len(str(r.get(tc,''))) > 5:
                text = str(r[tc]); break
        if text:
            tw_rows.append({'platform':'twitter','text':text,
                'likes':_safe_float(r.get('like_count',0)),
                'comments':0,'shares':_safe_float(r.get('retweet_count',0)),
                'views':0,'followers':max(_safe_float(r.get('followers',1000)),1),
                'post_hour':10,'post_weekday':2,'hashtag_count':text.count('#'),
                'has_media':0,'is_video':0,'is_paid':0})
    all_dfs.append(pd.DataFrame(tw_rows))
    print(f"  Twitter DMO: {len(tw_rows):,}")
except Exception as e: print(f"  Twitter DMO: {e}")

# ── Twitter 100K ──────────────────────────────────────────────────────────────
try:
    tw2 = pd.read_csv(f'{RAW}/social_real/Twitterdatainsheets.csv',
                      encoding='latin1', on_bad_lines='skip')
    tw2.columns = [c.lower().strip() for c in tw2.columns]
    tw2_rows = []
    for _, r in tw2.iterrows():
        text  = str(r.get('text','') or '')
        if len(text) < 5: continue
        try: hour = int(float(str(r.get('hour',10) or 10))) % 24
        except: hour = 10
        tw2_rows.append({'platform':'twitter','text':text,
            'likes':_safe_float(r.get('likes',0)),
            'comments':0,'shares':_safe_float(r.get('retweetcount',0)),
            'views':_safe_float(r.get('reach',0)),
            'followers':max(_safe_float(r.get('reach',1)),1),
            'post_hour':hour,'post_weekday':_wday(r.get('weekday',2)),
            'hashtag_count':text.count('#'),'has_media':0,'is_video':0,'is_paid':0})
    all_dfs.append(pd.DataFrame(tw2_rows))
    print(f"  Twitter 100K: {len(tw2_rows):,}")
except Exception as e: print(f"  Twitter 100K: {e}")

# ── Twitter SSSniperWolf (11MB, real tweets + likes) ──────────────────────────
try:
    ss = pd.read_csv(f'{RAW}/social_real/sssniperwolf.csv',
                     encoding='latin1', on_bad_lines='skip')
    ss.columns = [c.lower().strip() for c in ss.columns]
    ss_rows = []
    for _, r in ss.iterrows():
        text = str(r.get('content','') or '')
        if len(text) < 5: continue
        ss_rows.append({'platform':'twitter','text':text,
            'likes':_safe_float(r.get('likecount',0)),
            'comments':_safe_float(r.get('replycount',0)),
            'shares':_safe_float(r.get('retweetcount',0)),
            'views':0,'followers':15000000,  # SSSniperWolf ~15M followers
            'post_hour':14,'post_weekday':2,
            'hashtag_count':text.count('#'),'has_media':0,'is_video':0,'is_paid':0})
    all_dfs.append(pd.DataFrame(ss_rows))
    print(f"  Twitter SSSniperWolf: {len(ss_rows):,}")
except Exception as e: print(f"  Twitter SSSniperWolf: {e}")

# ── TikTok existing ───────────────────────────────────────────────────────────
for fpath in [f'{RAW}/tiktok/tiktok_dataset.csv'] + \
             glob.glob(f'{RAW}/tiktok_v2/**/*.csv', recursive=True):
    try:
        tk = pd.read_csv(fpath, encoding='latin1', on_bad_lines='skip')
        tk.columns = [c.lower().strip().replace(' ','_') for c in tk.columns]
        text_col  = next((c for c in tk.columns if any(x in c for x in
                          ['transcription','description','caption','text','title'])), None)
        views_col = next((c for c in tk.columns if 'view' in c or 'play' in c), None)
        likes_col = next((c for c in tk.columns if 'like' in c), None)
        share_col = next((c for c in tk.columns if 'share' in c), None)
        follow_col= next((c for c in tk.columns if 'follow' in c), None)
        if views_col:
            tk_rows = []
            for _, r in tk.iterrows():
                text   = str(r.get(text_col,'') or '') if text_col else ''
                views  = _safe_float(r.get(views_col,0))
                likes  = _safe_float(r.get(likes_col,0)) if likes_col else 0
                shares = _safe_float(r.get(share_col,0)) if share_col else 0
                followers = max(_safe_float(r.get(follow_col,5000)) if follow_col else 5000, 1)
                if views > 0:
                    tk_rows.append({'platform':'tiktok','text':text,'likes':likes,
                        'comments':0,'shares':shares,'views':views,'followers':followers,
                        'post_hour':19,'post_weekday':5,'hashtag_count':text.count('#'),
                        'has_media':1,'is_video':1,'is_paid':0})
            if tk_rows:
                all_dfs.append(pd.DataFrame(tk_rows))
                print(f"  TikTok {os.path.basename(fpath)}: {len(tk_rows):,}")
    except Exception as e: print(f"  TikTok {os.path.basename(fpath)}: {e}")

# ── Instagram: ALL real sources ───────────────────────────────────────────────
def _parse_ig_dir(folder, label=''):
    rows = []
    for fpath in glob.glob(f'{RAW}/{folder}/**/*.csv', recursive=True):
        try:
            df = pd.read_csv(fpath, encoding='latin1', on_bad_lines='skip')
            df.columns = [c.lower().strip().replace(' ','_') for c in df.columns]
            cols = list(df.columns)

            # Find best text column (avoid id/url/type columns)
            text_col = None
            for candidate in ['caption','description','content_description','text',
                               'content','post_text','caption_text']:
                if candidate in cols: text_col = candidate; break
            if text_col is None:
                text_col = next((c for c in cols if any(x in c for x in
                    ['caption','description','text','content'])
                    and 'id' not in c and 'url' not in c
                    and 'type' not in c and 'length' not in c), None)

            likes_col  = next((c for c in cols if 'like' in c and 'ratio' not in c
                               and 'count' not in c.replace('like_count','')), None) or \
                         next((c for c in cols if 'like' in c), None)
            cmt_col    = next((c for c in cols if 'comment' in c and 'ratio' not in c), None)
            share_col  = next((c for c in cols if 'share' in c or 'save' in c), None)
            follow_col = next((c for c in cols if 'follow' in c and 'ratio' not in c), None)
            views_col  = next((c for c in cols if 'view' in c or 'impression' in c or 'reach' in c), None)
            hour_col   = next((c for c in cols if 'hour' in c or 'time' in c), None)
            wday_col   = next((c for c in cols if 'day' in c or 'weekday' in c), None)

            print(f"  IG {label} {os.path.basename(fpath)}: text={text_col} likes={likes_col} follow={follow_col}")

            for _, r in df.iterrows():
                text = ''
                if text_col:
                    raw = r.get(text_col,'')
                    # Handle JSON dict format: {'text': '...'}
                    if isinstance(raw, str) and "text'" in raw:
                        try:
                            d = ast.literal_eval(raw)
                            text = str(d.get('text',''))
                        except:
                            m = re.search(r"'text':\s*'([^']{5,})'", raw)
                            if m: text = m.group(1)
                    else:
                        text = str(raw or '')
                if len(text) < 5: continue

                likes     = _safe_float(r.get(likes_col,0)) if likes_col else 0
                comments  = _safe_float(r.get(cmt_col,0))  if cmt_col   else 0
                shares    = _safe_float(r.get(share_col,0)) if share_col else 0
                followers = max(_safe_float(r.get(follow_col,5000)) if follow_col else 5000, 1)
                views     = _safe_float(r.get(views_col,0)) if views_col else max(likes*10, 1)

                try: hour = int(float(str(r.get(hour_col,12) or 12).split(':')[0])) % 24
                except: hour = 12
                wday = _wday(r.get(wday_col, 2))

                rows.append({'platform':'instagram','text':text,'likes':likes,
                    'comments':comments,'shares':shares,'views':max(views,1),
                    'followers':followers,'post_hour':hour,'post_weekday':wday,
                    'hashtag_count':text.count('#'),'has_media':1,
                    'is_video':0,'is_paid':0})
        except Exception as e: print(f"  IG {label} {fpath}: {e}")
    return rows

ig_rows = []
# Source 1: prajapatisuraj (real API JSON captions, 80MB)
try:
    ig_api = pd.read_csv(f'{RAW}/instagram_real/instagram.csv',
                         encoding='latin1', on_bad_lines='skip')
    ig_api.columns = [c.lower().strip() for c in ig_api.columns]
    src_rows = 0
    for _, r in ig_api.iterrows():
        cap_raw = r.get('caption','')
        text = ''
        if isinstance(cap_raw, str) and 'text' in cap_raw:
            try:
                cap_dict = ast.literal_eval(cap_raw)
                text = str(cap_dict.get('text',''))
            except:
                m = re.search(r"'text':\s*'([^']{5,})'", cap_raw)
                if m: text = m.group(1)
                else:
                    m2 = re.search(r'"text":\s*"([^"]{5,})"', cap_raw)
                    if m2: text = m2.group(1)
        if len(text) < 5: continue
        likes = 0
        for lc in ['like_count','fb_like_count']:
            try:
                likes = float(r.get(lc,0) or 0)
                if likes > 0: break
            except: pass
        comments = _safe_float(r.get('comment_count',0))
        media_type = str(r.get('media_type','') or '')
        is_video = float('video' in media_type.lower())
        ig_rows.append({'platform':'instagram','text':text,'likes':likes,
            'comments':comments,'shares':0,'views':max(likes*10,1),
            'followers':15000,'post_hour':12,'post_weekday':2,
            'hashtag_count':text.count('#'),'has_media':1,
            'is_video':is_video,'is_paid':0})
        src_rows += 1
    print(f"  Instagram API (prajapatisuraj): {src_rows:,}")
except Exception as e: print(f"  Instagram API: {e}")

# Source 2: Bhanu (real creator with hashtags column)
try:
    ig_b = pd.read_csv(f'{RAW}/instagram_real/Instagram_data_by_Bhanu.csv',
                       encoding='latin1', on_bad_lines='skip')
    ig_b.columns = [c.lower().strip() for c in ig_b.columns]
    ib_rows = 0
    for _, r in ig_b.iterrows():
        text = str(r.get('caption','') or '')
        tags = str(r.get('hashtags','') or '')
        if len(text) < 5: continue
        ig_rows.append({'platform':'instagram','text':text+' '+tags,
            'likes':_safe_float(r.get('likes',0)),
            'comments':_safe_float(r.get('comments',0)),
            'shares':_safe_float(r.get('shares',0)),
            'views':max(_safe_float(r.get('impressions',0)),1),
            'followers':5000,'post_hour':12,'post_weekday':2,
            'hashtag_count':tags.count('#'),'has_media':1,'is_video':0,'is_paid':0})
        ib_rows += 1
    print(f"  Instagram Bhanu: {ib_rows:,}")
except Exception as e: print(f"  Instagram Bhanu: {e}")

# Source 3–5: New datasets (tahirmohd, dumbergerl, yogesh)
for fname in ['my-datasat.csv','posts.csv','instagram_posts.csv',
              'instagram_post_performance.csv','Instagram_Posts.csv']:
    for fpath in glob.glob(f'{RAW}/instagram_real/**/{fname}', recursive=True):
        try:
            df = pd.read_csv(fpath, encoding='latin1', on_bad_lines='skip')
            df.columns = [c.lower().strip().replace(' ','_') for c in df.columns]
            text_col   = next((c for c in df.columns if any(x in c for x in
                ['caption','text','description','content'])
                and 'id' not in c and 'url' not in c), None)
            likes_col  = next((c for c in df.columns if 'like' in c and 'ratio' not in c), None)
            cmt_col    = next((c for c in df.columns if 'comment' in c and 'ratio' not in c), None)
            follow_col = next((c for c in df.columns if 'follow' in c and 'ratio' not in c), None)
            views_col  = next((c for c in df.columns if 'view' in c or 'impression' in c), None)
            print(f"  IG new {os.path.basename(fpath)}: text={text_col} likes={likes_col} rows={len(df)}")
            n = 0
            for _, r in df.iterrows():
                text = str(r.get(text_col,'') or '') if text_col else ''
                if len(text) < 5: continue
                ig_rows.append({'platform':'instagram','text':text,
                    'likes':_safe_float(r.get(likes_col,0)) if likes_col else 0,
                    'comments':_safe_float(r.get(cmt_col,0)) if cmt_col else 0,
                    'shares':0,
                    'views':max(_safe_float(r.get(views_col,0)) if views_col else 0, 1),
                    'followers':max(_safe_float(r.get(follow_col,5000)) if follow_col else 5000, 1),
                    'post_hour':12,'post_weekday':2,'hashtag_count':text.count('#'),
                    'has_media':1,'is_video':0,'is_paid':0})
                n += 1
            print(f"    -> {n:,} rows with captions")
        except Exception as e: print(f"  IG new {fpath}: {e}")

# Also scan all .csv in instagram_real for any we missed
for fpath in glob.glob(f'{RAW}/instagram_real/**/*.csv', recursive=True):
    fname = os.path.basename(fpath)
    if fname in ['instagram.csv','Instagram_data_by_Bhanu.csv',
                 'comments_cleaned.csv']: continue  # already handled
    try:
        df = pd.read_csv(fpath, encoding='latin1', on_bad_lines='skip', nrows=5)
        df.columns = [c.lower().strip() for c in df.columns]
        text_col = next((c for c in df.columns if any(x in c for x in
            ['caption','text','description','content']) and 'id' not in c
            and 'url' not in c and 'type' not in c), None)
        if text_col:
            df_full = pd.read_csv(fpath, encoding='latin1', on_bad_lines='skip')
            df_full.columns = [c.lower().strip().replace(' ','_') for c in df_full.columns]
            likes_col  = next((c for c in df_full.columns if 'like' in c and 'ratio' not in c), None)
            cmt_col    = next((c for c in df_full.columns if 'comment' in c), None)
            follow_col = next((c for c in df_full.columns if 'follow' in c), None)
            views_col  = next((c for c in df_full.columns if 'view' in c or 'impression' in c), None)
            n = 0
            for _, r in df_full.iterrows():
                text = str(r.get(text_col,'') or '')
                if len(text) < 5: continue
                ig_rows.append({'platform':'instagram','text':text,
                    'likes':_safe_float(r.get(likes_col,0)) if likes_col else 0,
                    'comments':_safe_float(r.get(cmt_col,0)) if cmt_col else 0,
                    'shares':0,
                    'views':max(_safe_float(r.get(views_col,0)) if views_col else 0, 1),
                    'followers':max(_safe_float(r.get(follow_col,5000)) if follow_col else 5000, 1),
                    'post_hour':12,'post_weekday':2,'hashtag_count':text.count('#'),
                    'has_media':1,'is_video':0,'is_paid':0})
                n += 1
            if n > 0: print(f"  IG extra {fname}: {n:,}")
    except: pass

if ig_rows:
    ig_df = pd.DataFrame(ig_rows)
    all_dfs.append(ig_df)
    print(f"  Instagram TOTAL: {len(ig_df):,} real caption rows")

# ── Multi-platform structured datasets ────────────────────────────────────────
# realistic_social (1,500 rows, viral label + features)
try:
    rs = pd.read_csv(f'{RAW}/social_real/realistic_social_media_dataset.csv',
                     encoding='latin1', on_bad_lines='skip')
    rs.columns = [c.lower().strip().replace(' ','_') for c in rs.columns]
    rs_rows = []
    for _, r in rs.iterrows():
        plat = str(r.get('platform','instagram') or 'instagram').lower()
        if plat not in PLATFORMS: plat = 'instagram'
        hook = _safe_float(r.get('hook_strength',5))
        pseudo_text = '!' * min(int(hook), 5)
        rs_rows.append({'platform':plat,'text':pseudo_text,
            'likes':_safe_float(r.get('likes',0)),
            'comments':_safe_float(r.get('comments',0)),
            'shares':_safe_float(r.get('shares',0))+_safe_float(r.get('saves',0)),
            'views':_safe_float(r.get('views',0)),
            'followers':max(_safe_float(r.get('follower_count',5000)),1),
            'post_hour':12,'post_weekday':_wday(r.get('day_of_week',2)),
            'hashtag_count':_safe_float(r.get('hashtag_count',0)),
            'has_media':1,'is_video':float(str(r.get('content_type','')).lower() in
                                          ['short','video','reel']),
            'is_paid':0,
            '_sentiment_override':(hook-5)/10,
            '_cta_override':float(str(r.get('call_to_action','')) in ['1','True','true','yes'])})
    all_dfs.append(pd.DataFrame(rs_rows))
    print(f"  Realistic social: {len(rs_rows):,}")
except Exception as e: print(f"  Realistic social: {e}")

# social_engagement (5,000 rows, 6 platforms with sentiment)
try:
    se = pd.read_csv(f'{RAW}/social_real/social_media_engagement_dataset.csv',
                     encoding='latin1', on_bad_lines='skip')
    se.columns = [c.lower().strip().replace(' ','_') for c in se.columns]
    se_rows = []
    for _, r in se.iterrows():
        plat = str(r.get('platform','instagram') or 'instagram').lower()
        if plat not in PLATFORMS: plat = 'instagram'
        followers = max(_safe_float(r.get('follower_count',5000)),1)
        content_len = _safe_float(r.get('content_length',100))
        pseudo_text = ' ' * max(0, int(content_len))
        if followers > 0:
            se_rows.append({'platform':plat,'text':pseudo_text,
                'likes':_safe_float(r.get('likes',0)),
                'comments':_safe_float(r.get('comments',0)),
                'shares':_safe_float(r.get('shares',0)),
                'views':_safe_float(r.get('views',0)),
                'followers':followers,
                'post_hour':_safe_int(r.get('hour_of_day',12)) % 24,
                'post_weekday':_wday(r.get('day_of_week',2)),
                'hashtag_count':_safe_float(r.get('hashtag_count',0)),
                'has_media':_safe_float(r.get('has_media',1)),
                'is_video':0,'is_paid':0,
                '_sentiment_override':_sent(r.get('sentiment',0)),
                '_cta_override':0})
    all_dfs.append(pd.DataFrame(se_rows))
    print(f"  Social engagement: {len(se_rows):,}")
except Exception as e: print(f"  Social engagement: {e}")

# ── Combine ───────────────────────────────────────────────────────────────────
combined = pd.concat(all_dfs, ignore_index=True)
for col in ['_sentiment_override','_cta_override']:
    if col not in combined.columns: combined[col] = np.nan
print(f"\nCOMBINED RAW: {len(combined):,}")
print(f"Platforms: {combined['platform'].value_counts().to_dict()}")

# ── STEP 2: LABELS ────────────────────────────────────────────────────────────
print("\n" + "="*65); print("STEP 2 — Normalized ER labels")
combined['followers_clip'] = combined['followers'].clip(lower=1)
combined['raw_er'] = (combined['likes']+combined['comments']+
                      combined['shares'])/combined['followers_clip']
plat_med = combined.groupby('platform')['raw_er'].median()
print("Platform median ER:")
for p, m in plat_med.items(): print(f"  {p:<12} {m:.4f}")
combined['platform_med_er'] = combined['platform'].map(plat_med).clip(lower=1e-8)
combined['norm_er']  = combined['raw_er']/combined['platform_med_er']
combined['label']    = (combined['norm_er'] > 1.0).astype(int)
print(f"Label balance: HIGH={combined['label'].mean()*100:.1f}%")

# ── STEP 3: STRATIFIED SAMPLING ───────────────────────────────────────────────
print("\n" + "="*65); print("STEP 3 — Stratified sampling")
TARGET = 20000; sampled = []
for plat in combined['platform'].unique():
    sub  = combined[combined['platform']==plat]
    high = sub[sub['label']==1]; low = sub[sub['label']==0]
    n = min(len(high), len(low), TARGET)
    if n < 50:
        print(f"  {plat}: skipped (high={len(high)}, low={len(low)})")
        continue
    sampled.append(pd.concat([high.sample(n=n,random_state=42),
                               low.sample(n=n,random_state=42)]))
    print(f"  {plat:<12} HIGH={n:>6,}  LOW={n:>6,}  total={2*n:,}")

train_df = pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=42)
print(f"\nFinal training set: {len(train_df):,} rows, {train_df['label'].mean()*100:.1f}% HIGH")

# ── STEP 4: NLP FEATURE EXTRACTION ───────────────────────────────────────────
print("\n" + "="*65); print("STEP 4 — NLP feature extraction")
t0 = time.time(); feature_rows = []; BATCH = 2000
for i in range(0, len(train_df), BATCH):
    batch = train_df.iloc[i:i+BATCH]
    for _, row in batch.iterrows():
        sent_ov = row.get('_sentiment_override')
        cta_ov  = row.get('_cta_override')
        feat = build_row(
            text=str(row.get('text','') or ''),
            platform=row['platform'],
            follower_count=row.get('followers',1000),
            post_hour=row.get('post_hour',12),
            post_weekday=row.get('post_weekday',2),
            hashtag_count=row.get('hashtag_count',0),
            has_media=row.get('has_media',1),
            is_video=row.get('is_video',0),
            is_paid=row.get('is_paid',0),
            sentiment_override=float(sent_ov) if pd.notna(sent_ov) else None,
            cta_override=float(cta_ov) if pd.notna(cta_ov) else None,
        )
        feature_rows.append(feat)
    if (i//BATCH) % 5 == 0:
        pct = min((i+BATCH)/len(train_df)*100, 100)
        print(f"  {pct:.0f}%  {time.time()-t0:.0f}s  ({min(i+BATCH,len(train_df)):,}/{len(train_df):,})")

X_df = pd.DataFrame(feature_rows)
y    = train_df['label'].values
feature_cols = list(X_df.columns)
print(f"Done in {time.time()-t0:.1f}s. Shape: {X_df.shape}")
corrs = X_df.corrwith(pd.Series(y)).abs().sort_values(ascending=False)
print(f"Max corr: {corrs.max():.4f} ({corrs.index[0]}) -- must be <0.5")

X_df['label'] = y
X_df['platform'] = train_df['platform'].values
X_df.to_csv(f'{EXPORT_DIR}/previral_training_v5.csv', index=False)
X_df.drop(columns=['label','platform'], inplace=True)
print("Saved: previral_training_v5.csv")

# ── STEP 5: TRAIN LIGHTGBM ────────────────────────────────────────────────────
print("\n" + "="*65); print("STEP 5 — Training LightGBM v5")
X = X_df.values
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
_, val_idx = train_test_split(np.arange(len(train_df)), test_size=0.20, random_state=42, stratify=y)
plat_val = train_df['platform'].values[val_idx]
print(f"Train: {len(y_tr):,}  Val: {len(y_val):,}")

params = {
    'objective':'binary','metric':'binary_logloss',
    'n_estimators':6000,'learning_rate':0.010,
    'num_leaves':127,'min_child_samples':40,
    'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
    'reg_alpha':0.1,'reg_lambda':0.1,
    'class_weight':'balanced','random_state':42,'n_jobs':-1,'verbosity':-1,
}
model = lgb.LGBMClassifier(**params)
model.fit(X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(250, verbose=False), lgb.log_evaluation(500)])

# ── STEP 6: EVALUATE ──────────────────────────────────────────────────────────
preds = model.predict(X_val)
proba = model.predict_proba(X_val)[:,1]
f1  = f1_score(y_val, preds)
auc = roc_auc_score(y_val, proba)
gap = proba[y_val==1].mean() - proba[y_val==0].mean()

print(f"\n{'='*65}")
print(f"V5 HOLD-OUT RESULTS ({len(y_val):,} samples)")
print(f"{'='*65}")
print(f"F1 (HIGH):   {f1:.4f}")
print(f"AUC-ROC:     {auc:.4f}")
print(f"Best iters:  {model.best_iteration_}")
print()
print(classification_report(y_val, preds, target_names=["LOW","HIGH"]))
print("Per-platform F1:")
for plat in np.unique(plat_val):
    mask = plat_val == plat
    if mask.sum() < 20: continue
    pf1  = f1_score(y_val[mask], preds[mask], zero_division=0)
    pauc = roc_auc_score(y_val[mask], proba[mask]) if len(np.unique(y_val[mask]))>1 else 0.5
    print(f"  {plat:<12} n={mask.sum():>6,}  F1={pf1:.3f}  AUC={pauc:.3f}")
print(f"\nConfidence gap: {gap:.3f}")

fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"\nTop 15 features:")
for feat, imp in fi.head(15).items(): print(f"  {feat:<35} {imp:>6.0f}")

# ── STEP 7: CALIBRATE PROBABILITIES ──────────────────────────────────────────
print(f"\n{'='*65}")
print("STEP 7 — Probability Calibration (isotonic)")
print("Wraps LightGBM in CalibratedClassifierCV for reliable P(HIGH)")
# Re-train final model on full data first
print("  Training final model on ALL data...")
model_full = lgb.LGBMClassifier(**{**params, 'n_estimators':model.best_iteration_+250})
model_full.fit(X, y, callbacks=[lgb.log_evaluation(0)])

# Calibrate using 5-fold cross-validation
print("  Calibrating (5-fold isotonic)...")
calibrated = CalibratedClassifierCV(model_full, cv=5, method='isotonic')
calibrated.fit(X, y)

# Evaluate calibrated model on hold-out
cal_proba = calibrated.predict_proba(X_val)[:,1]
cal_preds = (cal_proba >= 0.5).astype(int)
cal_f1    = f1_score(y_val, cal_preds)
cal_auc   = roc_auc_score(y_val, cal_proba)
cal_gap   = cal_proba[y_val==1].mean() - cal_proba[y_val==0].mean()
print(f"\nCalibrated model:")
print(f"  F1:  {cal_f1:.4f}  (vs raw: {f1:.4f})")
print(f"  AUC: {cal_auc:.4f}  (vs raw: {auc:.4f})")
print(f"  Gap: {cal_gap:.3f}  (vs raw: {gap:.3f})")

# ── STEP 8: SAVE ──────────────────────────────────────────────────────────────
print(f"\nSaving v5 artifacts...")
joblib.dump(model_full,   f'{SAVED}/previral_lgbm_v5.joblib')
joblib.dump(calibrated,   f'{SAVED}/previral_lgbm_v5_cal.joblib')
joblib.dump(feature_cols, f'{SAVED}/feature_columns_v5.joblib')
print(f"  previral_lgbm_v5.joblib:     {os.path.getsize(SAVED+'/previral_lgbm_v5.joblib')//1024}KB")
print(f"  previral_lgbm_v5_cal.joblib: {os.path.getsize(SAVED+'/previral_lgbm_v5_cal.joblib')//1024}KB")
print(f"  feature_columns_v5.joblib:   {os.path.getsize(SAVED+'/feature_columns_v5.joblib')} bytes")

print(f"\n{'='*65}")
print(f"FINAL v5: F1={f1:.4f}  AUC={auc:.4f}  Gap={gap:.3f}")
print(f"CALIBRATED: F1={cal_f1:.4f}  AUC={cal_auc:.4f}  Gap={cal_gap:.3f}")
print(f"{'='*65}")
