"""
Counterfactual Suggestion Engine — PreViral
Uses DICE-ML to generate diverse counterfactual explanations:
"What minimal feature changes would flip this prediction from LOW to HIGH?"
Then translates feature deltas into plain English suggestions.
"""
import numpy as np
from typing import List, Dict

# ── Suggestion Templates ──────────────────────────────────────────────────────
# Maps feature_name -> (condition_fn, suggestion_text_template)
SUGGESTION_TEMPLATES = {
    "sentiment_score": {
        "increase": lambda delta: (
            f"Your caption has a neutral or negative tone. Inject positive, energetic language — "
            f"words like 'excited', 'thrilled', or 'incredible' boost engagement by up to 22% on average."
        ),
        "decrease": lambda delta: (
            f"Your caption is overly positive which can read as inauthentic. "
            f"Balance it with a genuine, relatable observation."
        )
    },
    "emotional_valence": {
        "increase": lambda delta: (
            f"Add emotionally charged words to your caption. Posts with strong positive emotional language "
            f"receive significantly more saves and shares. Try: 'This changed everything for me...' or "
            f"'I'm genuinely proud to share...'"
        ),
        "decrease": lambda delta: (
            f"Consider a more balanced emotional tone. Overly promotional language can reduce trust."
        )
    },
    "clickbait_score": {
        "increase": lambda delta: (
            f"Add a curiosity hook to your opening line. Instead of stating the result, tease it. "
            f"Example: Instead of 'Our new product is here', try "
            f"'We built something that our team couldn't stop using — dropping this Friday.'"
        ),
        "decrease": lambda delta: (
            f"Reduce clickbait language — your audience may feel misled if the content doesn't "
            f"match the hook. This can hurt your completion rate and future algorithmic reach."
        )
    },
    "cta_present": {
        "increase": lambda delta: (
            f"Add a clear Call-To-Action to your caption. Posts with CTAs get 3x more engagement. "
            f"Try: 'Drop your thoughts below', 'Save this for later', or 'Tag someone who needs this.'"
        )
    },
    "readability_grade": {
        "decrease": lambda delta: (
            f"Simplify your caption language. Posts written at a Grade 6-8 reading level perform best "
            f"across all platforms. Break long sentences into short, punchy lines."
        ),
        "increase": lambda delta: (
            f"Your caption is very simple. For LinkedIn and professional platforms, adding more "
            f"substance and industry depth increases credibility and saves."
        )
    },
    "hashtag_count": {
        "increase": lambda delta: (
            f"You're using too few hashtags. For Instagram, 8-15 hashtags is optimal. "
            f"For TikTok, 3-7 targeted hashtags outperform large sets. See the Hashtag Panel below."
        ),
        "decrease": lambda delta: (
            f"You're using too many hashtags — this signals spam to the algorithm. "
            f"Trim to 8-12 on Instagram and 3-5 on TikTok for best results."
        )
    },
    "avg_competition_ratio": {
        "decrease": lambda delta: (
            f"Replace 3-4 of your high-competition hashtags (#viral, #trending, #fyp) with "
            f"niche-specific alternatives. These mega-tags have millions of posts and your content "
            f"will be buried within minutes. See the Hashtag Panel for 10 better alternatives."
        )
    },
    "trending_hashtag_count": {
        "increase": lambda delta, fv=None: (
            (
                f"Add hashtags to your caption — you have none right now. "
                f"For {fv.get('_platform','Instagram')} in this niche, 5-10 targeted hashtags can "
                f"increase reach by 40-60%. See the Hashtag Panel below for the best ones."
            ) if (fv and fv.get('hashtag_count_nlp', fv.get('hashtag_count', 0)) == 0) else (
                f"None of your hashtags are gaining traction right now. "
                f"Swap 2-3 of your existing ones with currently rising hashtags from the panel below."
            )
        )
    },
    "face_count": {
        "increase": lambda delta: (
            f"Add a human face to your thumbnail. Posts and thumbnails featuring faces have "
            f"25-35% higher click-through rates. Even a partial face in the corner helps."
        )
    },
    "face_prominence_score": {
        "increase": lambda delta: (
            f"Make the face in your thumbnail larger and more centered. The face should occupy "
            f"at least 20-30% of the frame for maximum emotional connection."
        )
    },
    "brightness_score": {
        "increase": lambda delta: (
            f"Your thumbnail is too dark. Increase brightness — high-performing thumbnails have "
            f"above-average brightness. Try increasing exposure by +30-50 in your editing tool."
        ),
        "decrease": lambda delta: (
            f"Your thumbnail is overexposed (too bright). Reduce exposure slightly to restore "
            f"detail and make it easier to read text overlays."
        )
    },
    "color_vibrancy": {
        "increase": lambda delta: (
            f"Increase color saturation in your thumbnail. More vibrant thumbnails stop the scroll "
            f"more effectively. Try boosting saturation by 20-30% in your editing tool."
        )
    },
    "peak_overlap_score": {
        "increase": lambda delta: (
            f"You're posting during off-peak hours for this platform. Shift your posting time "
            f"to one of the peak windows shown above — this alone can increase initial impressions "
            f"by 40-80% by getting your post in front of more active users in the first hour."
        )
    },
    "day_of_week_score": {
        "increase": lambda delta: (
            f"This is not the best day to post on this platform. "
            f"Consider posting on a higher-traffic day — see the timing panel for the optimal day."
        )
    },
    "clip_semantic_score": {
        "increase": lambda delta: (
            f"Your thumbnail doesn't align strongly with what performs well on this platform visually. "
            f"Study the top 10 posts in your niche and mirror their visual style and composition."
        )
    }
}

def generate_suggestions(
    feature_vector: dict,
    prediction: str,
    platform: str,
    max_suggestions: int = 3
) -> List[Dict]:
    """
    Generates actionable counterfactual suggestions without requiring DICE-ML.
    Uses heuristic rules based on feature thresholds to identify the biggest
    levers the user can pull to flip their prediction from LOW to HIGH.
    """
    suggestions = []

    # Feature thresholds — values below/above these are suboptimal
    thresholds = {
        "sentiment_score": ("increase", 0.2),
        "emotional_valence": ("increase", 0.3),
        "clickbait_score": ("increase", 0.25),
        "cta_present": ("increase", 0.5),
        "avg_competition_ratio": ("decrease", 0.45),
        "trending_hashtag_count": ("increase", 1),
        "face_count": ("increase", 0.5),
        "peak_overlap_score": ("increase", 0.7),
        "color_vibrancy": ("increase", 0.4),
        "brightness_score": ("increase", 0.35),
    }

    # Score each feature by how far it is from the desired threshold
    feature_gaps = []
    for feature, (direction, threshold) in thresholds.items():
        value = feature_vector.get(feature, None)
        if value is None:
            continue
        if direction == "increase":
            gap = max(0, threshold - value)
        else:
            gap = max(0, value - threshold)
        if gap > 0:
            feature_gaps.append((gap, feature, direction))

    # Sort by largest gap (biggest opportunity)
    feature_gaps.sort(reverse=True)

    for gap, feature, direction in feature_gaps[:max_suggestions]:
        template = SUGGESTION_TEMPLATES.get(feature, {}).get(direction)
        if template:
            try:
                suggestion_text = template(gap, fv=feature_vector)
            except TypeError:
                suggestion_text = template(gap)
            impact_pct = min(35, int(gap * 80))  # Estimate impact %
            suggestions.append({
                "feature": feature,
                "direction": direction,
                "current_value": round(feature_vector.get(feature, 0), 3),
                "target_direction": f"Increase to > {thresholds[feature][1]}" if direction == "increase" else f"Decrease to < {thresholds[feature][1]}",
                "suggestion": suggestion_text,
                "estimated_impact": f"+{impact_pct}% reach potential"
            })

    # If prediction is HIGH, give reinforcement suggestions
    if prediction == "HIGH" and not suggestions:
        suggestions.append({
            "feature": "general",
            "direction": "maintain",
            "current_value": None,
            "target_direction": "Keep current approach",
            "suggestion": "Your post is predicted to perform well! For maximum impact, post at the optimal time shown in the timing panel and monitor your first-hour engagement — if it exceeds your average, boost the post immediately.",
            "estimated_impact": "+15-30% additional reach if boosted in first hour"
        })

    return suggestions


if __name__ == "__main__":
    # Test with a low-performing post
    test_features = {
        "sentiment_score": 0.05,
        "emotional_valence": 0.1,
        "clickbait_score": 0.1,
        "cta_present": 0,
        "readability_grade": 14.0,
        "hashtag_count": 2,
        "avg_competition_ratio": 0.85,
        "trending_hashtag_count": 0,
        "face_count": 0,
        "face_prominence_score": 0.0,
        "peak_overlap_score": 0.2,
        "color_vibrancy": 0.2,
    }
    suggestions = generate_suggestions(test_features, "LOW", "instagram")
    print(f"Generated {len(suggestions)} suggestions:\n")
    for i, s in enumerate(suggestions, 1):
        print(f"[{i}] Feature: {s['feature']} | Impact: {s['estimated_impact']}")
        print(f"    {s['suggestion']}\n")
