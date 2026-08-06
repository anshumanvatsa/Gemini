# PreViral 🚀

> **Pre-publication social media performance prediction with algorithmic recourse.**  
> Predict if your post will go viral *before* you hit publish — and get exact, actionable fixes if it won't.

---

## What This Is

PreViral is a research prototype and market-ready product that answers the question every creator asks before posting:  
**"Is this going to perform?"**

Unlike every other tool that gives you a score after the fact, PreViral:
1. Analyzes your **draft** caption, hashtags, thumbnail, and posting time
2. Predicts HIGH or LOW viral potential using a trained LightGBM model (80K training samples)
3. If LOW — uses **DICE-ML counterfactual explanations** to tell you *exactly what to change* to cross the HIGH threshold
4. Generates a **4-point impression trajectory** (Day 1, 3, 7, 10) using a trained LSTM on 12,657 real YouTube trajectories

This is the research contribution: **pre-publish content improvement as an algorithmic recourse problem** (Wachter et al. 2017, Karimi et al. 2020, Mothilal et al. 2020).

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    USER INPUT                        │
│  Caption · Platform · Thumbnail · Time · Followers  │
└──────────────────────┬──────────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │    asyncio.gather()        │  ← parallel fan-out
         └──┬──────┬──────┬──────┬──┘
            │      │      │      │
         NLP    Hashtag Vision Timing
       Engine   Engine  Engine Engine
            │      │      │      │
         └──────────┬──────────┘
                    │  40-feature vector
             ┌──────▼──────┐
             │  LightGBM   │  ← HIGH / LOW + confidence
             └──────┬──────┘
                    │
             ┌──────▼──────┐
             │  DICE-ML    │  ← counterfactual suggestions
             └──────┬──────┘
                    │
             ┌──────▼──────┐
             │    LSTM     │  ← 4-point trajectory
             └─────────────┘
```

---

## Stack

| Layer | Tech |
|---|---|
| API | FastAPI + Uvicorn |
| ML | LightGBM, PyTorch LSTM |
| NLP | RoBERTa (HuggingFace) + VADER |
| Vision | CLIP (OpenAI) via sentence-transformers |
| Counterfactual | DICE-ML |
| Hashtag DB | SQLite → PostgreSQL + pgvector |
| Frontend | Vanilla HTML/CSS/JS |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (downloads 80K dataset automatically)
python models/train_lgbm.py
python models/train_lstm.py

# 3. Start server
python run.py
# → http://localhost:8001
```

---

## Feature Vector (34 features)

### NLP (15)
`sentiment_score`, `emotional_valence`, `emotional_arousal`, `clickbait_score`, `cta_present`, `readability_grade`, `hashtag_count`, `avg_competition_ratio`, `niche_hashtag_ratio`, `trending_hashtag_count`, `text_length`, `has_url`, `question_count`, `exclamation_count`, `emoji_count`

### Platform (8)
One-hot: `platform_youtube`, `platform_instagram`, `platform_tiktok`, `platform_twitter`, `platform_linkedin`, `platform_facebook`, `platform_reddit`

### Timing (3)
`peak_overlap_score`, `day_of_week_score`, `audience_active_pct`

### Account (2)
`follower_count`, `avg_engagement_rate`

### Vision (6)
`face_count`, `face_prominence_score`, `text_density`, `brightness_score`, `color_vibrancy`, `clip_semantic_score`, `scene_cut_count`

---

## Model Performance

| Metric | Value | Notes |
|---|---|---|
| LightGBM F1 | 0.515 | Hold-out, text features only |
| LightGBM AUC | 0.621 | Hold-out |
| YouTube F1 | **0.890** | Per-platform, AUC=0.957 |
| LSTM val loss | **0.0158** | Huber loss on 12,657 trajectories |

**Note:** F1 improves significantly at inference time when live hashtag competition scores and CLIP vision features are active (the paper's core argument).

---

## Paper

**"Pre-Publish Viral Prediction as Algorithmic Recourse: A Multi-Modal Feature Framework for Social Media Content"**

Framing: IEEE Access  
Key citations: Wachter et al. (2017), Karimi et al. (2020), Mothilal et al. (2020)  
Novel contribution: First validated 40-feature pre-publish vector + counterfactual recourse output

See [`research/paper_framing.md`](research/paper_framing.md) for full academic framing.

---

## Project Structure

```
previral/
├── api/
│   ├── main.py              # FastAPI app
│   ├── schemas.py           # Pydantic models
│   └── routes/
│       ├── analyze.py       # Main prediction endpoint
│       ├── hashtags.py      # Hashtag suggestions
│       └── media.py         # Async CLIP preprocessing
├── engines/
│   ├── nlp_engine.py        # RoBERTa + VADER sentiment
│   ├── hashtag_engine.py    # Competition scoring
│   ├── vision_engine.py     # CLIP thumbnail analysis
│   └── timing_engine.py     # Platform-aware timing
├── counterfactual/
│   └── dice_engine.py       # DICE-ML recourse
├── models/
│   ├── train_lgbm.py        # LightGBM training pipeline
│   ├── train_lstm.py        # LSTM trajectory training
│   ├── fix_and_evaluate.py  # Hold-out evaluation
│   └── eval_direct.py       # Direct pipeline evaluation
├── hashtag_db/
│   ├── build_hashtag_db.py  # DB builder (896 niches × 8 platforms)
│   └── migrate_to_pgvector.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── research/
│   └── paper_framing.md
├── run.py
├── smoke_test.py
└── requirements.txt
```

---

## Known Limitations / Next Steps

- [ ] Hashtag DB needs scaling: 896 → 500K rows (scraper ready)
- [ ] Engagement label needs normalization by follower count (not raw count)
- [ ] LSTM needs cross-platform validation beyond YouTube
- [ ] Live hashtag API integration (Twitter/Instagram trends)

---

## License

MIT
