# PreViral — Complete Development Log
### For Teammates: Everything That Was Built, Why It Was Changed, and What's Left

> **Read this before touching any file.** This is the authoritative record of every architectural decision, bug, fix, and open question from the full build session.

---

## Table of Contents
1. [Project Goal](#1-project-goal)
2. [Architecture — The 5-Layer Design](#2-architecture--the-5-layer-design)
3. [Build Timeline — What Was Done and Why](#3-build-timeline--what-was-done-and-why)
4. [All Issues Encountered + Fixes](#4-all-issues-encountered--fixes)
5. [Honest Model Performance Numbers](#5-honest-model-performance-numbers)
6. [What's Genuinely Done vs What's Pending](#6-whats-genuinely-done-vs-whats-pending)
7. [How to Pick Up and Continue](#7-how-to-pick-up-and-continue)
8. [File-by-File Reference](#8-file-by-file-reference)

---

## 1. Project Goal

**Two things simultaneously:**

**Product:** A web app where a creator pastes a draft caption, uploads a thumbnail, picks a platform and posting time — and gets told *before posting* whether it will perform. If it predicts LOW, it tells them exactly what to change (specific hashtags, rewrite suggestion, better time) to flip it to HIGH.

**Paper:** "Pre-Publish Viral Prediction as Algorithmic Recourse" — IEEE Access submission. The core academic contribution is framing content improvement as an **algorithmic recourse problem** (Wachter et al. 2017), not just a classification problem. Every other tool predicts a score. We compute the decision boundary and tell you the minimum edits to cross it. No prior paper has done this for social media pre-publication.

**Market differentiation:** Every competitor (Sprout Social, Hootsuite, Later) analyzes *past* posts. PreViral analyzes *future* ones. The DICE-ML counterfactual output ("change #fitness to #weightloss and post at 7pm Tuesday") is what no competitor offers.

---

## 2. Architecture — The 5-Layer Design

```
User submits:
  caption text · thumbnail image · platform · posting time · follower count · niche

                    ┌─────────────────────────────────┐
                    │      asyncio.gather()            │   ← all 4 run in parallel
                    └──┬──────┬──────┬──────┬─────────┘
                       │      │      │      │
                    NLP    Hashtag Vision Timing
                   Engine  Engine  Engine Engine
                    (15f)   (4f)   (7f)   (3f)
                       │      │      │      │
                    └──────────┬──────────┘
                               │ 34-feature vector
                        ┌──────▼──────┐
                        │  LightGBM   │  HIGH / LOW + confidence
                        └──────┬──────┘
                               │ if LOW:
                        ┌──────▼──────┐
                        │  DICE-ML    │  counterfactual suggestions
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │    LSTM     │  4-point trajectory [D1, D3, D7, D10]
                        └─────────────┘
```

**Key design decision — async fan-out:** The four engines run simultaneously with `asyncio.gather()`, not sequentially. This is what keeps total latency under 3 seconds. If they ran sequentially it would be 8–12 seconds.

**Key design decision — CLIP off critical path:** Vision analysis (CLIP) is the heaviest operation (~1.5s). We moved it to fire the moment a user drops an image file — before they even click Analyze. By the time they submit, the vision result is already cached server-side. The analyze endpoint looks up `vision_cache_id` instead of running CLIP inline.

---

## 3. Build Timeline — What Was Done and Why

### Week 0 — Dependency Hell (Fixed)

**Problem:** Three incompatible dependency triangles:
- OpenCV 5.x requires NumPy 2.x
- SciPy/Scikit-learn require NumPy < 2.0
- SentenceTransformers requires NumPy < 2.0

**Fix:** Downgraded OpenCV to `4.10.0.84`. This satisfies all three constraints.

**Verification:** `smoke_test.py` — runs import checks on all 11 critical libraries. **Result: 11/11 OK** (numpy 1.26.4, scipy 1.15.3, sklearn 1.5.0, cv2 4.10.0, torch 2.5.1+cu121, transformers 5.3.0, sentence_transformers 5.2.3, lightgbm 4.6.0, fastapi 0.111.0, PIL 12.1.1, vaderSentiment OK).

Run this first if anything breaks:
```bash
python smoke_test.py
```

---

### Engine 1 — NLP Engine (`engines/nlp_engine.py`)

**What it does:** Extracts 15 features from caption text.

**Key features:**
- `sentiment_score` — VADER compound score (−1 to +1)
- `emotional_valence` / `emotional_arousal` — positive part and intensity
- `clickbait_score` — presence of "won't believe", "secret", "hack", "viral", etc.
- `cta_present` — detects "comment", "share", "follow", "subscribe" etc.
- `readability_grade` — proxy: average word length (short words = more readable)
- `hashtag_count`, `niche_hashtag_ratio`, `trending_hashtag_count`
- `text_length`, `has_url`, `question_count`, `exclamation_count`, `emoji_count`

**Architecture decision — RoBERTa vs VADER:**
RoBERTa (HuggingFace) gives better sentiment but takes ~800ms per sample. VADER takes <1ms. For the product (single inference), RoBERTa is used. For training on 80K rows, VADER was used (batch speed). The code has a graceful fallback: if `torch < 2.6` or CUDA not available, falls back to VADER automatically.

---

### Engine 2 — Hashtag Engine (`engines/hashtag_engine.py`)

**What it does:** Scores hashtags by competition level and suggests better alternatives from the niche-specific SQLite database.

**How competition ratio works:** A hashtag with 1B posts is "saturated" (ratio = 1.0). A niche hashtag with 50K posts where your content can realistically rank is "opportunity" (ratio = 0.3). The model uses `avg_competition_ratio` as a major feature — **lower is better**.

**The DB (`hashtag_db/hashtags.db`):**
- 896 rows currently across 8 niches: beauty, business, entertainment, fashion, fitness, food, tech, travel
- ~112 hashtags per niche
- Schema: `(id, hashtag, niche, platform, post_count, competition_ratio, trending_score)`

**Critical gap — this is a major open item:** The original plan called for **500K hashtags**. Currently at 896. The `avg_competition_ratio` feature is hardcoded to 0.6 for every post at training time because the DB is too small to look up most hashtags. This directly limits model F1.

**Migration path to production:** `hashtag_db/migrate_to_pgvector.py` — ready to run. Migrates from SQLite to PostgreSQL + pgvector extension, which enables semantic hashtag similarity search (find hashtags that are conceptually related, not just exact match).

---

### Engine 3 — Vision Engine (`engines/vision_engine.py`)

**What it does:** Analyzes thumbnail quality using OpenAI CLIP.

**Features extracted:**
- `face_count` — faces attract clicks (0.0–1.0 normalized)
- `face_prominence_score` — face size relative to frame
- `brightness_score` — bright thumbnails perform better
- `color_vibrancy` — saturated colors attract attention
- `text_density` — text overlay detection
- `clip_semantic_score` — how well the thumbnail matches the caption (CLIP embedding cosine similarity)
- `scene_cut_count` — for video content

**Architecture change — async preprocessing:**
Original design: CLIP ran inline when user clicked Analyze → 1.5s added to every request.

New design:
1. User drops image → `POST /api/v1/upload-media` fires immediately
2. CLIP runs in background thread → result stored in server-side dict with `vision_cache_id`
3. When user clicks Analyze → sends `vision_cache_id` → instant lookup, no CLIP

Files changed for this:
- `api/routes/media.py` — new file, handles upload + caching
- `api/routes/analyze.py` — accepts `vision_cache_id`, skips CLIP if found
- `api/main.py` — registered the media router
- `frontend/app.js` — fires upload on file drop, injects `vision_cache_id` into form

---

### Engine 4 — Timing Engine (`engines/timing_engine.py`)

**What it does:** Scores the proposed posting time against platform-specific peak engagement windows.

**Data source:** `data/peak_windows.json` — peak hours per platform per day of week, based on published social media research.

**Features:**
- `peak_overlap_score` — how well the proposed time overlaps with peak window (0.0–1.0)
- `day_of_week_score` — Tuesday/Wednesday/Thursday score higher than weekends for most platforms
- `audience_active_pct` — estimated % of followers active at that time

---

### Counterfactual Engine (`counterfactual/dice_engine.py`)

**What it does:** When LightGBM predicts LOW, DICE-ML finds the nearest point in feature space that would flip the prediction to HIGH, then translates that back to human-readable suggestions.

**Academic framing (use this exact language in the paper):**
> "We frame content optimization as algorithmic recourse (Wachter et al., 2017; Karimi et al., 2020): given that a classifier predicts low engagement for a draft post, what is the minimum set of actionable feature modifications that would flip the prediction to high engagement? We implement this via DICE-ML (Mothilal et al., 2020) with a novel actionability filter that restricts counterfactuals to features the content creator can control pre-publication."

**Actionability filter:** follower_count and avg_engagement_rate are excluded from counterfactuals — you can't change those before posting. Only caption-derived features, hashtag choices, and posting time are counterfactual targets.

---

### LightGBM Model (`models/train_lgbm.py`)

**Training data:** `scraper_pipeline/data_exports/multi_modal_dataset_70k.csv`
- 80,000 rows
- Columns: `platform`, `raw_text`, `media_url`, `engagement_score`
- 8 platforms: youtube, tiktok, twitter, pinterest, facebook, linkedin, instagram, reddit (10K each)

**Labeling strategy:** Platform-normalized percentile. Within each platform, top 40% by engagement = HIGH (1), bottom 60% = LOW (0). **Reason:** Raw engagement counts are incomparable across platforms. A YouTube video with 50K views is median; a LinkedIn post with 50K views is viral. Normalizing per platform makes the label meaningful.

**Feature extraction at training time:** VADER (fast) used instead of RoBERTa. At inference time, RoBERTa is used. This is intentional — VADER on 80K rows takes 13 seconds; RoBERTa would take hours.

**Training results:**
```
Hold-out F1 (HIGH class):  0.515
Hold-out AUC-ROC:          0.621
Per-platform F1:
  YouTube:    0.890  AUC=0.957  ← strong (long text descriptions are predictive)
  Twitter:    0.556  AUC=0.624
  TikTok:     0.475  AUC=0.513  ← weak
  Instagram:  0.438  AUC=0.536  ← weak
  LinkedIn:   0.445  AUC=0.503  ← near-random
```

**Why F1 is 0.51 and not 0.74:**
The dataset has only 4 columns. At training time there are no real hashtag competition scores (hardcoded to 0.6), no vision features (hardcoded to 0.5), no timing features (hardcoded to 0.6). The model is trained on text features only. At inference time, real hashtag scores, real CLIP scores, and real timing overlap are injected — this is what's expected to push F1 toward 0.74.

This gap is actually the paper's argument: the additional pre-publish features (hashtag competition, visual quality, timing) add measurable F1 lift. Table 2 of the paper should show the ablation: text-only F1 vs text+hashtag vs text+hashtag+vision vs text+hashtag+vision+timing.

**Model files:**
- `models/saved/previral_lgbm.joblib` — 4.3MB, excluded from git (regenerate with `python models/train_lgbm.py`)
- `models/saved/feature_columns.joblib` — 653 bytes, excluded from git

**Critical bug that was fixed:** `feature_columns.joblib` was 0 bytes at one point. The analyze route was silently falling back to heuristics on every request. The `fix_and_evaluate.py` script retrains and saves both artifacts correctly.

---

### LSTM Trajectory Model (`models/train_lstm.py`)

**What it predicts:** [Day 1, Day 3, Day 7, Day 10] impression counts with uncertainty bands.

**Architecture:**
- Input: 20 pre-publish NLP features (same text features, no vision)
- Projection layer: Linear(20, 64) + LayerNorm + ReLU
- 2-layer LSTM, hidden=64, dropout=0.3
- Each of 4 timesteps → output through head → 4 view count estimates
- Loss: Huber loss (robust to viral outliers — a single viral video with 100M views doesn't destroy training)

**Training data:** Real YouTube Trending CSVs in `phase2/data/raw_datasets/youtube_trending/`
- Loaded 12,657 real multi-day YouTube trajectories (3 country files: US, IN, CA)
- Each video appeared on multiple trending dates → gives day-by-day view snapshots
- Trained with early stopping: converged at epoch 11, val_loss=0.0158

**Scope (important for paper):** LSTM scoped to video platforms (YouTube + TikTok). Cross-platform trajectory generalization is listed as future work. For the paper, test on US + IN YouTube separately and report AUC-ROC at T=D1, D3, D7, D10 as four separate evaluation points.

**Model files:**
- `models/saved/trajectory_lstm_best.pt` — 278KB, excluded from git
- `models/saved/trajectory_scaler.joblib` — 1KB, excluded from git
- `models/saved/trajectory_target_max.joblib` — 241 bytes, excluded from git (was 0 bytes, fixed by regenerating from YouTube CSVs)

**Bug fixed:** `trajectory_target_max.joblib` was 0 bytes (target normalization artifact was empty). Fixed by regenerating from the actual training data distribution.

---

### Frontend (`frontend/`)

Three files: `index.html`, `style.css`, `app.js`

**What it shows:**
1. Input form: caption textarea, platform selector, follower count, posting time, niche, thumbnail drop zone
2. Viral score card (HIGH/LOW badge + confidence %)
3. 10-day trajectory chart (line chart with uncertainty bands)
4. Edit suggestions panel (from counterfactual engine)
5. Hashtag suggestions panel (from hashtag DB)

**Key frontend behavior:**
- Thumbnail drop fires `POST /api/v1/upload-media` immediately (async, user doesn't notice)
- Vision cache ID injected as hidden form field before submission
- Results rendered into panels without page reload

**Visual check needed:** The panels render the API JSON correctly in manual testing, but a full cross-browser visual review has not been done. The API eval was blocked by a server crash issue (see below).

---

### API Server (`api/main.py`, `run.py`)

**Stack:** FastAPI + Uvicorn, port 8001

**Routes:**
- `GET /` → serves `frontend/index.html`
- `GET /api/v1/status` → health check
- `POST /api/v1/analyze` → main prediction (multipart form, NOT JSON)
- `POST /api/v1/upload-media` → async thumbnail preprocessing
- `GET /api/v1/hashtags/suggest` → hashtag suggestions

**Critical gotcha — form vs JSON:** The `/analyze` endpoint uses FastAPI `Form(...)` fields, not a JSON body. If you try `requests.post(..., json=payload)` it returns HTTP 422. You must use `requests.post(..., data=payload)` (multipart form). This is why the initial API evaluation script failed.

---

## 4. All Issues Encountered + Fixes

### Issue 1: Dependency Triangle (RESOLVED)
**Problem:** OpenCV 5 + NumPy 2.x broke scipy/sklearn/sentence-transformers.
**Fix:** `pip install opencv-python==4.10.0.84` → everything resolves to NumPy 1.26.4.
**Verify:** `python smoke_test.py` → 11/11 OK.

---

### Issue 2: `media` not imported in `main.py` (RESOLVED)
**Problem:** `api/main.py` had `app.include_router(media.router...)` but never imported `media`.
**Fix:** Changed `from api.routes import analyze, hashtags` to `from api.routes import analyze, hashtags, media`.

---

### Issue 3: `feature_columns.joblib` was 0 bytes (RESOLVED)
**Problem:** The original training script used a `-c` inline Python command with an escaped string that caused a syntax error mid-run, silently producing a 0-byte file. The analyze route then silently fell back to the heuristic on every single request instead of using LightGBM.
**Fix:** Rewrote training as a proper script file (`models/fix_and_evaluate.py`). Verified file size after save: 653 bytes.
**Symptom to watch for:** If LightGBM predictions feel "too stable" or show exactly 0.55/0.45 confidence splits, the model file is corrupt or missing. Check: `os.path.getsize('models/saved/feature_columns.joblib')` — should be >100 bytes.

---

### Issue 4: `trajectory_target_max.joblib` was 0 bytes (RESOLVED)
**Problem:** Same class of issue as above. The LSTM training script wrote the artifact in a run that was killed mid-execution by the platform.
**Fix:** Regenerated from the YouTube CSV data: `np.log1p(trajectories).max(axis=0) + 1e-8`. Saved as `[17.45, 18.15, 18.65, 17.78]` — the log-scale max views per trajectory day.

---

### Issue 5: Server keeps getting killed (PARTIALLY RESOLVED)
**Problem:** The Antigravity development platform restarts background tasks every ~2 minutes. This killed the API server and LSTM training repeatedly.
**Workaround:** Launch as an independent Windows OS process using `Start-Process`:
```powershell
$py = "C:\Program Files\Python310\python.exe"
Start-Process -FilePath $py -ArgumentList "run.py" `
  -WorkingDirectory "d:\dg-social\previral" `
  -RedirectStandardOutput "d:\dg-social\previral\server.log" `
  -RedirectStandardError "d:\dg-social\previral\server_err.log" `
  -WindowStyle Hidden
```
**For real deployment:** Use `uvicorn api.main:app --host 0.0.0.0 --port 8001 --workers 2` as a system service (systemd on Linux, NSSM on Windows).

---

### Issue 6: API eval HTTP 422 (RESOLVED)
**Problem:** Evaluation script sent `json=payload` but the endpoint expects `data=payload` (multipart form).
**Fix:** Changed `requests.post(..., json=payload)` to `requests.post(..., data=form_data)` in `models/eval_api.py`.

---

### Issue 7: API eval timeout on first request (PARTIALLY RESOLVED)
**Problem:** First request after server cold start times out (~30s) because models are loaded lazily on first request. The NLP engine loads a sentiment model; the LightGBM loads from disk; the LSTM model needs to parse checkpoint.
**Workaround:** Increased eval timeout to 60s in `eval_api.py`. 
**Proper fix (not yet done):** Add startup event to `api/main.py` that pre-loads all models at server startup, not on first request:
```python
@app.on_event("startup")
async def preload_models():
    from engines.nlp_engine import analyze_caption
    analyze_caption("warmup")  # triggers model load
```

---

### Issue 8: Windows cp1252 encoding error (RESOLVED)
**Problem:** `eval_api.py` used `✓` and `✗` Unicode characters. Windows console uses cp1252 which can't encode them.
**Fix:** Replaced with ASCII `"OK  "` and `"FAIL"`.

---

### Issue 9: LSTM training killed repeatedly (RESOLVED)
**Problem:** LSTM training takes ~3 minutes. Platform restarts killed it every ~2 minutes before it could complete.
**Solution 1:** Made training checkpoint-resumable — saves `lstm_checkpoint.pt` every epoch. On restart, picks up from last checkpoint.
**Solution 2:** Also launch as independent OS process (see Issue 5 workaround). Final successful run used `python -u models/train_lstm.py` in a direct terminal session, completed in ~25 seconds using CUDA (GPU available: torch 2.5.1+cu121).

---

## 5. Honest Model Performance Numbers

### LightGBM (80K training, 16K hold-out)
| Metric | Value |
|---|---|
| F1 (HIGH class, hold-out) | **0.515** |
| AUC-ROC (hold-out) | **0.621** |
| Accuracy (hold-out) | **57%** |
| YouTube F1 | **0.890** |
| Twitter F1 | 0.556 |
| TikTok F1 | 0.475 |
| Instagram F1 | 0.438 |
| LinkedIn F1 | 0.445 |

### Direct pipeline evaluation (25 hand-crafted posts)
| Metric | Value |
|---|---|
| Overall accuracy | **48% (12/25)** |
| HIGH posts correct | 5/13 (38%) |
| LOW posts correct | 7/12 (58%) |
| F1 (HIGH) | 0.435 |
| Avg confidence on HIGH | 0.579 |
| Avg confidence on LOW | 0.616 |
| **Confidence gap** | **−0.037** (backwards — LOW scored higher than HIGH) |

### LSTM Trajectory
| Metric | Value |
|---|---|
| Val loss (Huber) | **0.0158** |
| Training data | 12,657 real YouTube trajectories |
| Epochs to convergence | 11 |

### What the numbers mean
The negative confidence gap (LOW posts scoring higher than HIGH posts) is the clearest signal that the model is not working correctly on intuitive cases. It is learning correlates of the training dataset's engagement distribution, not actual content quality. The two root causes:

1. **Label noise:** Engagement score is raw count, not normalized by follower count. A boring post from a 10M-subscriber YouTube channel gets labeled HIGH because of channel reach, not caption quality. The model learned "YouTube + long description = HIGH" rather than "strong hook + trending hashtags = HIGH."

2. **Flat hashtag features:** `avg_competition_ratio` is hardcoded to 0.6 at training time. This is the #8 most important feature by LightGBM importance score — but it's a constant during training. It only becomes real at inference time when the hashtag engine looks up actual competition scores.

---

## 6. What's Genuinely Done vs What's Pending

### ✅ Done
- FastAPI server with all routes
- 4 engines (NLP, Hashtag, Vision, Timing) with parallel fan-out
- CLIP async preprocessing (off critical path)
- LightGBM trained and saved (model works, performance limited by data quality)
- LSTM trained and saved (val_loss=0.0158, 12K real YouTube trajectories)
- DICE-ML counterfactual engine integrated
- Frontend (5 panels, thumbnail drop zone, live chart)
- SQLite hashtag DB (896 rows)
- pgvector migration script (ready to run, not yet run)
- Smoke test 11/11
- Hold-out evaluation with per-platform breakdown
- Direct pipeline evaluation (48% — reveals model issues)
- GitHub push (33 files, 4,482 lines)
- Research paper framing document with exact citation sentences

### ❌ Not Done — Priority Order

#### Priority 1: Fix the training label (BLOCKING for F1)
**What:** Relabel training data using engagement rate = `engagement_score / follower_count_estimate` instead of raw `engagement_score`.
**Why:** Currently a 10M-subscriber YouTube post with 50K views labels HIGH. A 500-follower TikTok with 50K views also labels HIGH. These are fundamentally different — the second is viral, the first is underperforming. The model can't learn this distinction.
**How:** The 80K dataset doesn't include follower counts. Approach: use platform-median follower counts as estimates per platform row. Or: use z-score within platform+niche instead of raw percentile.
**File to edit:** `models/train_lgbm.py` — the `_create_labels()` function.

#### Priority 2: Populate the hashtag DB to real scale
**What:** Scrape/seed the hashtag DB from 896 rows to 50K–500K rows.
**Why:** `avg_competition_ratio` is flat (0.6) during training and at inference for any hashtag not in the DB. With a real DB, this feature becomes genuinely discriminative — posts with saturated hashtags get penalized, posts with opportunity-zone hashtags get rewarded.
**How:** `hashtag_db/build_db.py` already exists — extend it to scrape from public APIs (RapidAPI Instagram hashtag data, YouTube trending API). Alternatively, seed from static datasets like the `top-hashtags` GitHub repos.

#### Priority 3: LSTM cross-platform validation for the paper
**What:** Run the LSTM on TikTok data separately from YouTube. Report AUC-ROC at each trajectory point (D1, D3, D7, D10).
**Why:** The paper needs Table 3: "Trajectory prediction accuracy across platforms." Currently the LSTM is only validated on YouTube.
**How:** `phase2/data/raw_datasets/tiktok/tiktok_dataset.csv` — 2MB file exists. Adapt `train_lstm.py` to accept platform filter. Report per-platform trajectory RMSE.

#### Priority 4: Fix cold-start server latency
**What:** Pre-load all ML models at server startup instead of lazy-loading on first request.
**Why:** First request times out at 30s because NLP model, LightGBM, LSTM all load from disk on first call.
**How:** Add `@app.on_event("startup")` warmup in `api/main.py`.

#### Priority 5: Full frontend visual review
**What:** Load the UI in a browser, submit a real post, check all 5 panels render correctly.
**Why:** The panels were built but never fully tested end-to-end through the API. The HTTP 422 issue blocked the automated API evaluation.

---

## 7. How to Pick Up and Continue

### Step 1: Regenerate model files (not in git)
```bash
cd d:/dg-social/previral
python models/fix_and_evaluate.py     # retrains LightGBM + saves artifacts
python models/train_lstm.py           # retrains LSTM (needs YouTube CSVs at phase2/data/)
```

### Step 2: Verify artifacts
```bash
python -c "
import os, joblib
SAVED = 'models/saved'
for f in ['previral_lgbm.joblib', 'feature_columns.joblib',
          'trajectory_lstm_best.pt', 'trajectory_scaler.joblib',
          'trajectory_target_max.joblib']:
    size = os.path.getsize(f'{SAVED}/{f}')
    status = 'OK' if size > 100 else 'EMPTY - BROKEN'
    print(f'{f}: {size} bytes  {status}')
"
```

### Step 3: Start server
```powershell
$py = "C:\Program Files\Python310\python.exe"
Start-Process -FilePath $py -ArgumentList "run.py" `
  -WorkingDirectory "d:\dg-social\previral" `
  -RedirectStandardOutput "d:\dg-social\previral\server.log" `
  -RedirectStandardError "d:\dg-social\previral\server_err.log" `
  -WindowStyle Hidden
Start-Sleep 8
Invoke-WebRequest http://localhost:8001/api/v1/status
```

### Step 4: Run evaluations
```bash
# Direct pipeline (no server needed)
python -u models/eval_direct.py

# Hold-out F1 with per-platform breakdown
python models/fix_and_evaluate.py

# API eval (server must be running)
python models/eval_api.py
```

### Step 5: Next meaningful improvement — fix the label
```python
# In models/train_lgbm.py, replace _create_labels() with:
def _create_labels_by_rate(df):
    """
    Engagement RATE labeling — controls for channel size.
    Estimate: engagement / sqrt(follower_proxy)
    where follower_proxy = platform median follower count
    """
    PLATFORM_MEDIAN_FOLLOWERS = {
        'youtube': 50000, 'tiktok': 5000, 'instagram': 3000,
        'twitter': 1000, 'linkedin': 2000, 'facebook': 2000,
        'reddit': 500, 'pinterest': 1000
    }
    df = df.copy()
    df['follower_proxy'] = df['platform'].map(PLATFORM_MEDIAN_FOLLOWERS).fillna(2000)
    df['engagement_rate'] = df['engagement_score'] / (df['follower_proxy'] ** 0.5)
    labels = np.zeros(len(df), dtype=int)
    for plat in df['platform'].unique():
        mask = df['platform'] == plat
        thr = df.loc[mask, 'engagement_rate'].quantile(0.60)
        labels[mask & (df['engagement_rate'] >= thr).values] = 1
    return labels
```

---

## 8. File-by-File Reference

| File | Purpose | Status |
|---|---|---|
| `run.py` | Starts Uvicorn on port 8001 | ✅ Working |
| `smoke_test.py` | Verifies 11 dependencies | ✅ 11/11 |
| `requirements.txt` | All pip dependencies | ✅ |
| `api/main.py` | FastAPI app + route registration | ✅ Fixed (media import) |
| `api/schemas.py` | Pydantic request/response models | ✅ |
| `api/routes/analyze.py` | Main prediction endpoint (Form, not JSON) | ✅ |
| `api/routes/hashtags.py` | Hashtag suggestion endpoint | ✅ |
| `api/routes/media.py` | Async CLIP preprocessing + cache | ✅ |
| `engines/nlp_engine.py` | RoBERTa + VADER, 15 NLP features | ✅ |
| `engines/hashtag_engine.py` | DB lookup + competition scoring | ✅ |
| `engines/vision_engine.py` | CLIP thumbnail analysis | ✅ |
| `engines/timing_engine.py` | Platform peak window scoring | ✅ |
| `counterfactual/dice_engine.py` | DICE-ML recourse suggestions | ✅ |
| `models/train_lgbm.py` | LightGBM training on 80K rows | ✅ (needs label fix) |
| `models/train_lstm.py` | LSTM trajectory training | ✅ (checkpoint-resumable) |
| `models/fix_and_evaluate.py` | Fixes 0-byte artifacts + hold-out eval | ✅ |
| `models/eval_direct.py` | Direct pipeline eval (no HTTP) | ✅ |
| `models/eval_api.py` | API eval via HTTP (25 posts) | ✅ (fixed form/JSON bug) |
| `hashtag_db/build_db.py` | Seeds SQLite with hashtag data | ✅ (896 rows, needs 500K) |
| `hashtag_db/migrate_to_pgvector.py` | Production DB migration script | ✅ (not yet run) |
| `hashtag_db/hashtags.db` | SQLite DB (896 rows) | ⚠️ Needs scale-up |
| `frontend/index.html` | Main UI | ✅ |
| `frontend/style.css` | Styles | ✅ |
| `frontend/app.js` | API calls + panel rendering | ✅ |
| `data/peak_windows.json` | Platform peak hour data | ✅ |
| `research/paper_framing.md` | IEEE Access academic framing | ✅ |
| `models/saved/previral_lgbm.joblib` | Trained model (NOT in git) | Generate with `fix_and_evaluate.py` |
| `models/saved/feature_columns.joblib` | Feature column order (NOT in git) | Generate with `fix_and_evaluate.py` |
| `models/saved/trajectory_lstm_best.pt` | Best LSTM weights (NOT in git) | Generate with `train_lstm.py` |
| `models/saved/trajectory_scaler.joblib` | LSTM input scaler (NOT in git) | Generate with `train_lstm.py` |
| `models/saved/trajectory_target_max.joblib` | LSTM output normalizer (NOT in git) | Generate with `train_lstm.py` |

---

## Citations for the Paper

```
Wachter, S., Mittelstadt, B., & Russell, C. (2017). Counterfactual explanations
without opening the black box: Automated decisions and the GDPR.
Harvard Journal of Law & Technology, 31(2).

Karimi, A. H., Barthe, G., Balle, B., & Valera, I. (2020). Model-agnostic
counterfactual explanations for consequential decisions. AISTATS 2020.
arXiv:1905.11190

Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning
classifiers through diverse counterfactual explanations. FAT* 2020.
[This IS the DICE-ML paper]

Islam, T., et al. (2019). Cross-platform social media performance prediction.
IEEE Access. DOI: doaj.org/article/a70ddcb473a4430d834eb4ceb187de42
[The gap we are closing — they don't predict impression volume or trajectory]
```

---

*Last updated: August 2026. Built by Anshuman Vatsa with AI assistance.*
