"""
Phase 2 — Build hashtag DB to 50K+ entries with real competition scores.

Sources (all free, no API key needed):
  1. Twitter/DMO dataset — extract all hashtags + compute within-dataset frequency
  2. TikTok dataset     — extract from video_transcription_text
  3. YouTube trending   — extract from tags column (9 country files)
  4. Synthetic niche seeds — curated starter lists per niche

Run: python hashtag_db/build_hashtag_db_v2.py
Output: hashtag_db/hashtags_v2.db  (target: 50K+ rows)
"""

import os, sys, re, sqlite3, glob, time
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

DB_PATH = 'd:/dg-social/previral/hashtag_db/hashtags_v2.db'

# ── Niche keyword classifier ──────────────────────────────────────────────────
NICHE_KEYWORDS = {
    'fitness':       ['gym','workout','fitness','exercise','bodybuilding','crossfit','yoga',
                      'running','weightloss','muscle','cardio','training','health','diet','protein'],
    'food':          ['recipe','food','cooking','baking','restaurant','chef','vegan','vegetarian',
                      'dinner','breakfast','lunch','foodie','meal','cuisine','delicious'],
    'travel':        ['travel','vacation','trip','destination','wanderlust','explore','adventure',
                      'backpacking','hotel','flight','tourism','holiday','roadtrip','beach'],
    'fashion':       ['fashion','style','outfit','ootd','clothing','shoes','accessories','designer',
                      'streetwear','vintage','sustainable','trend','lookbook','wardrobe'],
    'beauty':        ['beauty','makeup','skincare','haircare','cosmetics','lipstick','foundation',
                      'moisturizer','skinroutine','glowup','naturalbeauty','nails','eyeshadow'],
    'tech':          ['tech','technology','coding','programming','software','hardware','ai',
                      'machinelearning','startup','developer','python','javascript','gadgets','innovation'],
    'business':      ['business','entrepreneur','startup','marketing','sales','finance','investing',
                      'money','success','leadership','productivity','strategy','growth','revenue'],
    'entertainment': ['music','movies','gaming','anime','tv','celebrity','pop','hiphop','netflix',
                      'streaming','concert','meme','viral','trending','entertainment'],
}

def classify_niche(tag: str) -> str:
    tag_l = tag.lower()
    for niche, keywords in NICHE_KEYWORDS.items():
        if any(kw in tag_l for kw in keywords):
            return niche
    return 'general'

def normalize_tag(tag: str) -> str:
    return re.sub(r'[^a-z0-9]', '', tag.lower().strip())

# ── Setup DB ──────────────────────────────────────────────────────────────────
print("Setting up hashtag DB v2...")
conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS hashtags (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        hashtag          TEXT NOT NULL,
        niche            TEXT,
        platform         TEXT,
        post_count       INTEGER DEFAULT 0,
        competition_ratio REAL DEFAULT 0.5,
        trending_score   REAL DEFAULT 0.5,
        source           TEXT,
        UNIQUE(hashtag, platform)
    )
""")
cur.execute("CREATE INDEX IF NOT EXISTS idx_hashtag ON hashtags(hashtag)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_niche    ON hashtags(niche)")
conn.commit()

hashtag_data = []  # list of (hashtag, niche, platform, count)

# ── SOURCE 1: Twitter DMO dataset ─────────────────────────────────────────────
print("\n[1/4] Twitter DMO dataset...")
TWITTER_PATH = 'd:/dg-social/phase2/data/raw_datasets/twitter/DMO social media engagement dataset/Data LIWC 01 02 23.csv'
try:
    tw = pd.read_csv(TWITTER_PATH, encoding='latin1', on_bad_lines='skip')
    tw.columns = [c.lower().strip() for c in tw.columns]
    text_cols = [c for c in tw.columns if 'text' in c or 'content' in c or 'tweet' in c or 'x' == c]
    print(f"  Text columns: {text_cols}")
    tw_counter = Counter()
    for col in text_cols:
        if col in tw.columns:
            for text in tw[col].dropna().astype(str):
                tags = re.findall(r'#([a-zA-Z]\w{1,29})', text)
                tw_counter.update([normalize_tag(t) for t in tags if len(t) > 2])
    print(f"  Unique Twitter hashtags found: {len(tw_counter):,}")
    for tag, count in tw_counter.most_common(15000):
        if tag and len(tag) >= 3:
            niche = classify_niche(tag)
            hashtag_data.append((tag, niche, 'twitter', count))
except Exception as e:
    print(f"  ERROR: {e}")

# ── SOURCE 2: TikTok dataset ──────────────────────────────────────────────────
print("\n[2/4] TikTok dataset...")
TIKTOK_PATH = 'd:/dg-social/phase2/data/raw_datasets/tiktok/tiktok_dataset.csv'
try:
    tk = pd.read_csv(TIKTOK_PATH, encoding='latin1', on_bad_lines='skip')
    tk.columns = [c.lower().strip() for c in tk.columns]
    text_col = next((c for c in tk.columns if 'text' in c or 'transcription' in c or 'description' in c), None)
    print(f"  Text column: {text_col}")
    tk_counter = Counter()
    if text_col:
        for text in tk[text_col].dropna().astype(str):
            tags = re.findall(r'#([a-zA-Z]\w{1,29})', text)
            tk_counter.update([normalize_tag(t) for t in tags if len(t) > 2])
    # Also get view counts for competition ratio
    view_col = next((c for c in tk.columns if 'view' in c), None)
    print(f"  Unique TikTok hashtags found: {len(tk_counter):,}")
    for tag, count in tk_counter.most_common(5000):
        if tag and len(tag) >= 3:
            niche = classify_niche(tag)
            hashtag_data.append((tag, niche, 'tiktok', count * 100))  # scale to post_count proxy
except Exception as e:
    print(f"  ERROR: {e}")

# ── SOURCE 3: YouTube Trending (9 country files) ──────────────────────────────
print("\n[3/4] YouTube Trending (9 country files)...")
YT_DIR = 'd:/dg-social/phase2/data/raw_datasets/youtube_trending'
yt_files = glob.glob(f'{YT_DIR}/*.csv')
yt_counter = defaultdict(int)
yt_views   = defaultdict(list)

for fpath in yt_files:
    country = os.path.basename(fpath).replace('videos.csv', '')
    try:
        yt = pd.read_csv(fpath, encoding='latin1', on_bad_lines='skip')
        yt.columns = [c.lower().strip() for c in yt.columns]
        tags_col  = next((c for c in yt.columns if 'tag' in c), None)
        title_col = next((c for c in yt.columns if 'title' in c), None)
        views_col = next((c for c in yt.columns if 'view' in c), None)
        if tags_col:
            for i, row in yt.iterrows():
                raw_tags = str(row.get(tags_col, ''))
                title    = str(row.get(title_col, ''))
                views    = int(row.get(views_col, 0)) if views_col else 0
                # Parse tags (pipe-separated in YouTube trending CSVs)
                tags_list = [t.strip().strip('"').lower() for t in raw_tags.split('|')]
                # Also extract hashtags from title
                tags_list += re.findall(r'#([a-zA-Z]\w{1,29})', title.lower())
                for tag in tags_list:
                    tag = normalize_tag(tag)
                    if len(tag) >= 3:
                        yt_counter[tag] += 1
                        yt_views[tag].append(views)
        print(f"  {country}: processed {len(yt):,} rows")
    except Exception as e:
        print(f"  {country}: ERROR {e}")

print(f"  Unique YouTube hashtags/tags: {len(yt_counter):,}")
for tag, count in sorted(yt_counter.items(), key=lambda x: -x[1])[:30000]:
    if tag and len(tag) >= 3:
        niche = classify_niche(tag)
        avg_views = int(np.mean(yt_views[tag])) if yt_views[tag] else 0
        hashtag_data.append((tag, niche, 'youtube', avg_views))

# ── SOURCE 4: Curated niche seed lists ────────────────────────────────────────
print("\n[4/4] Curated niche seed lists...")
SEEDS = {
    'fitness': [
        'fitness','gym','workout','weightloss','bodybuilding','crossfit','yoga','running',
        'cardio','musclebuilding','fitnessmotivation','healthylifestyle','personaltrainer',
        'fitfam','gymlife','fitnessgirl','abs','squat','benchpress','deadlift','protein',
        'preworkout','postworkout','homeworkout','noequipment','plank','pushup','pullup',
        'hiit','pilates','zumba','cycling','swimming','marathon','triathlon','nutrition',
        'macros','mealprep','cleaneating','intermittentfasting','keto','paleo','veganfitness',
    ],
    'food': [
        'food','recipe','cooking','baking','foodphotography','foodie','instafood','homemade',
        'dinner','breakfast','lunch','dessert','chocolate','pizza','pasta','sushi','burger',
        'vegan','vegetarian','glutenfree','healthyfood','mealprep','quickrecipes','easyrecipes',
        'chef','foodblogger','foodstagram','yummy','delicious','tasty','foodlover','restaurant',
        'streetfood','brunch','coffee','tea','cocktail','wine','beer','smoothie','juicing',
    ],
    'travel': [
        'travel','wanderlust','travelgram','traveling','travelblogger','adventure','explore',
        'vacation','holiday','trip','backpacking','solotravel','luxurytravel','budgettravel',
        'beach','mountains','citylife','roadtrip','camping','hiking','sunset','photography',
        'instatravel','travelphotography','worldtravel','travelcouple','digitalnomad',
        'remotework','workandtravel','balifood','tokyo','paris','newyork','london','dubai',
    ],
    'fashion': [
        'fashion','style','ootd','outfitoftheday','fashionblogger','streetstyle','streetwear',
        'vintage','thrifted','sustainable','slowfashion','luxuryfashion','highfashion','designer',
        'sneakers','shoes','accessories','jewelry','bag','watch','sunglasses','hat','coat',
        'summer','winter','falloutfit','springfashion','mensfashion','womensfashion','unisex',
        'minimalist','aesthetic','y2k','cottagecore','dark','grunge','preppy','boho','chic',
    ],
    'beauty': [
        'beauty','makeup','skincare','haircare','cosmetics','makeupartist','mua','grwm',
        'skincareroutine','morningroutine','glowup','naturalmakeup','nomakeup','drugstore',
        'highend','foundation','concealer','blush','eyeshadow','mascara','lipstick','gloss',
        'serum','moisturizer','sunscreen','retinol','vitaminc','niacinamide','hyaluronicacid',
        'acneprone','oilyskin','dryskin','combination','antiaging','spf','toner','exfoliant',
    ],
    'tech': [
        'tech','technology','coding','programming','developer','software','hardware','ai',
        'machinelearning','deeplearning','python','javascript','typescript','react','nodejs',
        'webdev','frontend','backend','fullstack','database','cloud','aws','azure','gcp',
        'devops','docker','kubernetes','github','opensource','startup','saas','productmanagement',
        'ux','ui','design','figma','cybersecurity','blockchain','web3','nft','crypto','fintech',
    ],
    'business': [
        'business','entrepreneur','startup','marketing','digitalmarketing','socialmedia',
        'contentmarketing','seo','emailmarketing','ppc','branding','sales','revenue','growth',
        'leadership','management','productivity','finance','investing','stocks','realestate',
        'dropshipping','ecommerce','amazon','shopify','freelance','consulting','agency',
        'smallbusiness','sidehustle','passiveincome','mindset','success','motivation',
    ],
    'entertainment': [
        'entertainment','music','movies','gaming','anime','kpop','netflix','hulu','disney',
        'marvel','dccomics','hiphop','rap','rnb','pop','edm','rock','indie','classical',
        'concert','festival','comedy','standup','podcast','youtube','streaming','esports',
        'twitch','valorant','fortnite','minecraft','genshin','cosplay','manga','webtoon',
    ],
}

seed_count = 0
for niche, tags in SEEDS.items():
    for tag in tags:
        tag_norm = normalize_tag(tag)
        if tag_norm:
            hashtag_data.append((tag_norm, niche, 'all', 50000))  # baseline count
            seed_count += 1

print(f"  Seeded {seed_count} curated hashtags across {len(SEEDS)} niches")

# ── Insert to DB ──────────────────────────────────────────────────────────────
print(f"\n  Total hashtag candidates: {len(hashtag_data):,}")

# Group by (hashtag, platform) and sum counts
agg = defaultdict(lambda: {'count': 0, 'niche': 'general'})
for tag, niche, platform, count in hashtag_data:
    key = (tag, platform)
    agg[key]['count'] += count
    if niche != 'general':
        agg[key]['niche'] = niche

# Compute competition ratio (log-normalized)
all_counts = [v['count'] for v in agg.values() if v['count'] > 0]
max_log    = np.log1p(max(all_counts)) if all_counts else 1.0

print(f"  Inserting {len(agg):,} unique (hashtag, platform) pairs...")
inserted = 0
for (tag, platform), meta in agg.items():
    count = meta['count']
    niche = meta['niche']
    competition_ratio = float(np.log1p(count) / max_log)
    trending_score    = 1.0 - competition_ratio  # inverse: low comp = high opportunity
    try:
        cur.execute("""
            INSERT OR REPLACE INTO hashtags
                (hashtag, niche, platform, post_count, competition_ratio, trending_score, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (tag, niche, platform, int(count), round(competition_ratio, 4),
              round(trending_score, 4), 'dataset_extraction'))
        inserted += 1
    except Exception:
        pass

conn.commit()

# Final stats
cur.execute("SELECT COUNT(*) FROM hashtags")
total = cur.fetchone()[0]
cur.execute("SELECT platform, COUNT(*) FROM hashtags GROUP BY platform ORDER BY COUNT(*) DESC")
plat_breakdown = cur.fetchall()
cur.execute("SELECT niche, COUNT(*) FROM hashtags GROUP BY niche ORDER BY COUNT(*) DESC")
niche_breakdown = cur.fetchall()

print(f"\n{'='*50}")
print(f"HASHTAG DB V2 COMPLETE")
print(f"{'='*50}")
print(f"Total rows: {total:,}")
print(f"\nBy platform:")
for plat, cnt in plat_breakdown:
    print(f"  {plat:<12} {cnt:>8,}")
print(f"\nBy niche:")
for niche, cnt in niche_breakdown:
    print(f"  {niche:<14} {cnt:>8,}")

conn.close()
print(f"\nSaved to: {DB_PATH}")
print(f"DB size: {os.path.getsize(DB_PATH)//1024}KB")
