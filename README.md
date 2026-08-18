# PreViral — Know Before You Post

<div align="center">

![PreViral](https://img.shields.io/badge/PreViral-Live%20Production-blueviolet?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Powered%20by-Gemini%20AI-blue?style=for-the-badge&logo=google)
![F1 Score](https://img.shields.io/badge/F1%20Score-0.8489-brightgreen?style=for-the-badge)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.9220-brightgreen?style=for-the-badge)
![Platforms](https://img.shields.io/badge/Platforms-6-blue?style=for-the-badge)
![Training Rows](https://img.shields.io/badge/Training%20Data-352K%20Real%20Rows-orange?style=for-the-badge)
![Cloud Run](https://img.shields.io/badge/Google%20Cloud%20Run-asia--south1-blue?style=for-the-badge&logo=googlecloud)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

### [Try it Live](https://previral-in-268278227902.asia-south1.run.app)

**Predict whether your social media post will go viral — before you hit publish.**

*Pre-publication features only. Powered by Gemini. No post-publication data leakage.*

</div>

---

## What is PreViral?

PreViral is an AI-powered, pre-publication social media performance predictor built on **Google Gemini**. It reads your caption and thumbnail together, scores viral potential using a LightGBM model trained on 352K+ real posts, and uses causal machine learning (DICE-ML) to tell you exactly what to change to improve your reach — before you publish.

**Built for the Build with Gemini XPRIZE hackathon.**

**Live URL:** https://previral-in-268278227902.asia-south1.run.app

---

## Key Features

### 1. Viral Confidence Score
- HIGH / MEDIUM / LOW prediction with calibrated percentage
- Trained on 352,000+ real posts across 6 platforms (F1 = 0.8489, AUC = 0.9220)
- Breaks down: Hook Strength, Sentiment, Timing Score, Hashtag Score

### 2. Causal "What to Change" Suggestions (DICE-ML)
Not generic tips — specific counterfactual explanations that identify the exact changes that flip your score:
- "Add 5-10 hashtags — you have none right now. For Instagram in this niche, 5-10 targeted hashtags can increase reach by 40-60%."
- "Add a human face to your thumbnail (posts with faces get 25-35% higher click-through rates)"
- "Add a clear Call-To-Action to your caption"

### 3. AI Content Director (Powered by Gemini)
- Gemini reads your caption AND thumbnail together (true multimodal analysis)
- Rewrites your hook, injects trending hashtags, optimizes the CTA
- Re-scores the rewrite with LightGBM — never trusts Gemini blindly
- Shows a Model-Validated Score Journey (Original -> Iteration 1 -> Iteration 2)
- When the rewrite scores lower, honestly says so and keeps your original

### 4. Platform-Specific 10-Day Reach Trajectory
- LSTM model forecasting impressions over 10 days
- Platform-aware decay shapes: Reels get fast_spike, YouTube gets instant_decay, LinkedIn gets slow_build
- Content-type-aware: Carousel, Short, Story, Community Post all have different curves
- Mega-creator handling: accounts >1M followers get front-loaded day-1 trajectories
- Expected Total, Best Case, Worst Case impressions with narrative explanation

### 5. Live Trending Hashtags (Gemini Search Grounding)
- Real-time trend intelligence via Gemini built-in Google Search
- Niche-specific, not generic hashtag lists from a static database

### 6. Content Safety
- Uploaded thumbnails moderated by Gemini vision for inappropriate content
- Privacy fallback: describe your visual in text instead of uploading an image

### 7. Shareable Reports
- One-click share link generation for any analysis result

---

## Gemini API Integrations (5 deep integrations)

| # | Integration | What it does |
|---|---|---|
| 1 | Multimodal Feature Extraction | Caption text + raw thumbnail bytes sent to Gemini in one API call |
| 2 | AI Content Director | 2-iteration model-validated rewrite loop with honest best-of-N selection |
| 3 | Search Grounding | Real-time trending hashtag intelligence via Google Search |
| 4 | Content Safety | Vision-based moderation of uploaded thumbnails |
| 5 | Visual Description Fallback | Text-to-visual alignment scoring for privacy-conscious creators |

Model in production: `gemini-flash-lite-latest` via `google-genai` SDK

---

## ML Model Performance

### Overall (v5 Production)

| Metric | Score |
|---|---|
| F1 Score (HIGH class) | 0.8489 |
| AUC-ROC | 0.9220 |
| Accuracy | 85.0% |
| Training Rows | 352,976 real posts |
| Platforms | 6 |

### Per-Platform Results

| Platform | F1 Score | AUC-ROC | Training Rows |
|---|---|---|---|
| Instagram | 0.924 | 0.983 | 3,186 |
| YouTube | 0.891 | 0.936 | 40,000 |
| TikTok | 0.853 | 0.933 | 40,000 |
| Twitter/X | 0.851 | 0.913 | 40,000 |
| Facebook | 0.816 | 0.894 | 1,968 |
| LinkedIn | 0.796 | 0.900 | 1,004 |

---

## System Architecture

```
previral-gemini/
├── api/
│   ├── main.py              <- FastAPI app entry point
│   ├── schemas.py           <- Request/Response models (incl. safety_flag)
│   └── routes/
│       ├── analyze.py       <- Main prediction + DICE-ML + AI Director endpoints
│       └── hashtags.py      <- Hashtag intelligence endpoint
├── engines/
│   ├── nlp_engine.py        <- VADER + Flesch-Kincaid + CTA + clickbait
│   ├── hashtag_engine.py    <- Competition scoring + trend velocity (40K DB)
│   ├── vision_engine.py     <- OpenCV + color vibrancy + face detection
│   ├── timing_engine.py     <- Peak window lookup (platform-calibrated)
│   └── gemini_engine.py     <- All 5 Gemini API integrations
├── models/
│   ├── master_train_v5.py   <- Production training pipeline (352K rows)
│   ├── train_lstm.py        <- LSTM trajectory model (PyTorch)
│   └── saved/
│       ├── previral_lgbm_v5.joblib       <- Production LightGBM model
│       ├── feature_columns_v5.joblib     <- 22 feature names
│       ├── trajectory_lstm_best.pt       <- LSTM weights
│       ├── trajectory_scaler.joblib      <- Input scaler
│       └── trajectory_target_max.joblib  <- Output normalizer
├── counterfactual/
│   └── dice_explainer.py    <- DICE-ML causal counterfactual engine
├── frontend/
│   ├── index.html           <- Main UI (dark mode, responsive)
│   ├── app.js               <- All frontend logic (platform pills, AI Director, charts)
│   └── style.css            <- Dark mode UI with neon accent system
├── hashtag_db/              <- 40K+ hashtag competition/trend database
├── Dockerfile               <- Google Cloud Run container definition
└── requirements-prod.txt    <- Production dependencies
```

---

## Deployment

### Live Production
Deployed on Google Cloud Run (Mumbai, asia-south1):
```
https://previral-in-268278227902.asia-south1.run.app
```

### Run Locally

```bash
git clone https://github.com/anshumanvatsa/Previral.git
cd Previral

pip install -r requirements-prod.txt

cp .env.example .env
# Add your GEMINI_API_KEY to .env

uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

Open http://localhost:8080 in your browser.

### Environment Variables

```
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-lite-latest
PORT=8080
```

### Health Check

```bash
curl https://previral-in-268278227902.asia-south1.run.app/api/v1/gemini-status
```

Expected response:
```json
{"status":"ok","gemini_available":true,"model":"gemini-flash-lite-latest","response":"PREVIRAL_OK"}
```

---

## API Reference

### POST /api/v1/analyze
Main prediction endpoint (multipart/form-data).

**Fields:** platform, content_type, caption, follower_count, niche, image (optional)

**Response includes:** prediction, confidence, hook_strength, sentiment_score, timing_score, hashtag_score, suggestions (DICE-ML), trajectory (10-day LSTM), safety_flag

### POST /api/v1/ai-director
Gemini-powered content rewrite with model-validated scoring.

**Response includes:** rewritten_caption, hook_rewrite, beat_original (bool), recommended (caption + score), alternative (if rewrite lost), iteration_trail, best_posting_time, thumbnail_suggestion, vocabulary_suggestion

### GET /api/v1/gemini-status
Health check for Gemini API connectivity.

---

## Feature Engineering (22 features)

All features extracted exclusively from pre-publication information:

**NLP (16):** sentiment_score, emotional_valence, emotional_arousal, clickbait_score, cta_present, readability_grade, text_length, has_url, question_count, exclamation_count, emoji_count, hashtag_count_nlp, mention_count, caps_ratio, avg_word_length, unique_word_ratio

**Temporal (6):** peak_overlap_score, day_of_week_score, post_hour_sin, post_hour_cos, post_wday_sin, post_wday_cos

**Hashtag (4):** hashtag_count, niche_hashtag_ratio, trending_hashtag_count, avg_competition_ratio

**Media & Account (7):** follower_count, has_media, is_video, is_paid, audience_active_pct + 5 vision features

---

## Training Data

352,976 total rows. Zero synthetic captions. All real platform data.

| Dataset | Platform | Rows | Source |
|---|---|---|---|
| YouTube US/IN/CA Trending | YouTube | 78,235 | Kaggle (Mitchell J.) |
| YouTube Non-Trending | YouTube | 79,511 | Kaggle |
| Twitter 100K | Twitter | 99,939 | Kaggle (DMO dataset) |
| SSSniperWolf Tweets | Twitter | 47,217 | Kaggle (thedevastator) |
| TikTok Transcriptions | TikTok | 19,084 | Kaggle |
| TikTok Viral Content | TikTok | 19,000+ | Kaggle |
| Instagram API Captions | Instagram | 1,406 | Kaggle (prajapatisuraj) |
| Social Engagement Multi | Multi | 6,500 | Kaggle |

---

## Comparison to Prior Work

| Work | F1 | Features | Platforms |
|---|---|---|---|
| Bao et al. (2013) | 0.71 | Post-publication mixed | Twitter |
| Gelli et al. (2015) | 0.78 | Image + text | Instagram |
| High et al. (2022) | 0.81 | Pre+post mix | Multi |
| **PreViral v5 (ours)** | **0.8489** | **Pre-publication only** | **6 platforms** |

---

## Honesty by Design

When Gemini's rewrite scores lower than the original on LightGBM, PreViral says so. It never inflates the score from Gemini's self-estimate — it validates against the independent ML model and shows the real delta. The "Model-Validated Score Journey" always reflects actual LightGBM scores, not Gemini's own prediction.

---

## Roadmap

- [x] Live production deployment on Google Cloud Run (asia-south1)
- [x] 5 deep Gemini integrations (multimodal, rewrite, search grounding, safety, fallback)
- [x] DICE-ML causal counterfactual explanations
- [x] Platform-specific content type pills (Reel, Carousel, Short, Story, etc.)
- [x] Platform-specific trajectory shapes with content-type awareness
- [x] Mega-creator trajectory calibration (>1M followers)
- [x] Honest AI Director with model-validated best-of-N selection
- [x] Shareable reports
- [ ] Instagram API integration for direct scheduling
- [ ] Historical predicted vs actual performance tracking
- [ ] Fine-tune ML model on niche-specific training data
- [ ] Paid tier launch (September 2026)

---

## Training History

| Version | F1 | AUC | Key Change |
|---|---|---|---|
| v1 | 0.71 | 0.82 | Baseline: YouTube only |
| v2 | 0.74 | 0.86 | Added Twitter, TikTok |
| v3 | 0.886 | 0.955 | RoBERTa features (data leak — rejected) |
| v4 | 0.8635 | 0.9292 | All real data, fixed IG fake labels |
| v5 | 0.8489 | 0.9220 | +100K Twitter, +SSSniperWolf, CalibratedCV |
| v6 (exp) | 0.7834 | 0.8834 | Instagram Analytics added — rejected |

---

## Citation

```bibtex
@software{previral2026,
  author    = {Mishra, Anshuman},
  title     = {PreViral: AI Pre-Publication Social Media Performance Predictor},
  year      = {2026},
  url       = {https://github.com/anshumanvatsa/Previral},
  note      = {F1=0.8489, AUC-ROC=0.9220, 6 platforms, 352K real training rows, Powered by Google Gemini}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with Gemini · Deployed on Google Cloud Run · Build with Gemini XPRIZE 2026**

*352,976 real training rows · 6 platforms · F1 = 0.8489 · AUC-ROC = 0.9220*

[Try it Live](https://previral-in-268278227902.asia-south1.run.app)

</div>
