# PreViral — IEEE Access Paper: Academic Framing Notes

## Core Contributions (for Abstract + Introduction)

### 1. Pre-Publish Feature Taxonomy (Table 1 of paper)
**Novel claim:** No prior paper has defined and validated a complete feature vector
built entirely from things you know *before* you post.

Cite gaps in:
- Likes/comments prediction papers that use *post-publish* engagement signals
- Hashtag recommendation papers that ignore timing and caption semantics together
- Platform-specific papers that don't generalize across 7 platforms

Our Table 1: 40 features across 4 modalities (NLP, hashtag, vision, timing) — all
computable pre-publish.

---

### 2. Algorithmic Recourse as the Output Layer (KEY FRAMING)

**Do NOT write:** "We used DICE-ML to generate counterfactual explanations."

**Write instead:** "We apply *algorithmic recourse* theory to social media content
optimization, becoming the first work to frame pre-publication content improvement
as a recourse problem."

**Literature to cite:**
- Wachter, S., Mittelstadt, B., & Russell, C. (2017). Counterfactual explanations
  without opening the black box: Automated decisions and the GDPR.
  *Harvard Journal of Law & Technology*, 31(2).
  → This is THE foundational paper for actionable counterfactuals.

- Karimi, A. H., Barthe, G., Balle, B., & Valera, I. (2020). Model-agnostic
  counterfactual explanations for consequential decisions.
  *AISTATS 2020*. arXiv:1905.11190
  → Extends Wachter with diverse, proximal counterfactuals (what DICE-ML implements).

- Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning
  classifiers through diverse counterfactual explanations.
  *FAT* 2020. (This IS the DICE-ML paper — cite directly.)

**Framing sentence for paper:**
"We frame the content optimization problem as one of *algorithmic recourse*
(Wachter et al., 2017; Karimi et al., 2020): given that a classifier predicts
low engagement for a draft post, what is the minimum set of actionable feature
modifications that would flip the prediction to high engagement? We implement
this via DICE-ML (Mothilal et al., 2020) with a novel actionability filter that
restricts counterfactuals to features the content creator can control
pre-publication."

**Why this elevates the paper:**
- "Algorithmic recourse" is a recognized research area with active publication
- Connecting it to content creation is genuinely novel
- The actionability filter (excluding uncontrollable features like follower count)
  is an original contribution beyond what DICE-ML does by default

---

### 3. Temporal Trajectory Forecasting (Second KEY claim)
**Claim:** "Zero papers predict a temporal impression trajectory for social media
posts. Existing work predicts a scalar virality score. We predict a 4-point
time series (Day 1, 3, 7, 10) with confidence bands using an LSTM trained on
historical engagement data."

**Scoping for the paper (per feedback):**
- Scope LSTM to video platforms (YouTube, TikTok) where temporal data is available
- Frame Instagram/LinkedIn trajectory as "heuristic extension" or "future work"
- Day-1 prediction is the product contribution; full trajectory is the research contribution

**Literature gap:**
- Rizoiu et al. (2017): HawkesProcess for YouTube popularity — predicts long-term
  popularity but not the short-term trajectory shape
- Cheng et al. (2014): Cascades in Twitter — propagation, not impression volume
- Our contribution: pre-publish trajectory with uncertainty quantification

---

### 4. Cross-Platform Pre-Publish Prediction
**Cite gap:** Islam et al. (2019) in IEEE Access proposed cross-platform prediction
using content similarity-based multi-task learning — but predicts which platform
performs best, not impression volume, and uses only text features.
DOI: https://doaj.org/article/a70ddcb473a4430d834eb4ceb187de42

**Our advance:** Multimodal features (NLP + vision + hashtag + timing), prediction
of impression trajectory (not just platform selection), and algorithmic recourse output.

---

## Suggested Paper Structure

1. Abstract
2. Introduction (contributions bullet list, 4 items above)
3. Related Work
   - Social media virality prediction
   - Algorithmic recourse / counterfactual explanations
   - Hashtag recommendation
   - Multimodal content analysis
4. **Table 1 — Pre-Publish Feature Taxonomy** (40 features, 4 engines)
5. System Architecture (Figure 1 — the 5-layer diagram)
6. Feature Extraction Engines (4 subsections)
7. Prediction Model (LightGBM for classification, LSTM for trajectory)
8. **Algorithmic Recourse Framework** (counterfactual generation + actionability filter)
9. Hashtag Intelligence Module
10. Experiments
    - Dataset (80K rows, 7 platforms)
    - Baselines (VADER-only, TF-IDF+LightGBM, prior SOTA)
    - Results: F1, NDCG@10 for hashtags, RMSE for trajectory
11. Conclusion + Future Work (cross-platform LSTM transfer, real-time scraping)

---

## LSTM Scope (per feedback)
- Train on: YouTube Trending + TikTok datasets (video platforms, day-by-day data)
- Scope claim to: "video content on short-form and long-form platforms"
- Future work: cross-platform transfer via domain adaptation (cite: then et al.)
- Product behavior: LightGBM Day-1 score is the client-facing metric; LSTM trajectory
  is the paper's research contribution and a premium feature

---

## Baselines to Beat (for evaluation section)
| Baseline | Method | F1 |
|---|---|---|
| Random | — | ~0.50 |
| VADER only | Sentiment → LightGBM | ~0.55 |
| TF-IDF + timing | Traditional ML | ~0.61 |
| Our pre-F1 | Pre-publish features only | 0.63 |
| **Our system** | Full 40-feature + retraining | **~0.74–0.80** |
