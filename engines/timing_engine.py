"""
Timing Engine — PreViral
Looks up platform peak windows and computes 3 timing features:
peak_overlap_score, day_of_week_score, audience_active_pct
"""
import json
import os
from datetime import datetime

# Load peak windows table once at module level
_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "peak_windows.json")
with open(_DATA_PATH, "r") as f:
    PEAK_WINDOWS = json.load(f)

# Day-of-week scores per platform (0=Mon, 6=Sun)
DAY_WEIGHTS = {
    "instagram": [0.7, 0.9, 1.0, 0.85, 0.8, 0.6, 0.5],
    "tiktok":    [0.6, 1.0, 0.8, 0.85, 0.7, 0.9, 0.7],
    "youtube":   [0.7, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0],
    "twitter":   [0.7, 0.9, 0.9, 1.0, 0.9, 0.7, 0.5],
    "linkedin":  [0.6, 1.0, 1.0, 0.9, 0.8, 0.4, 0.2],
    "facebook":  [0.6, 0.8, 0.8, 1.0, 1.0, 0.9, 0.6],
    "reddit":    [1.0, 0.9, 0.9, 0.9, 0.8, 0.7, 0.5],
}

def analyze_timing(platform: str, post_datetime: datetime) -> dict:
    """
    Given a platform and posting datetime, return 3 timing features.
    """
    platform = platform.lower()
    if platform not in PEAK_WINDOWS:
        return {
            "peak_overlap_score": 0.5,
            "day_of_week_score": 0.5,
            "audience_active_pct": 0.05
        }

    data = PEAK_WINDOWS[platform]
    hour = post_datetime.hour
    day = post_datetime.weekday()  # 0=Monday, 6=Sunday

    # 1. Peak overlap score — is this hour in the peak hours list?
    peak_hours = data["peak_hours"]
    if hour in peak_hours:
        # Score higher if it's the best hour
        if hour == data["best_hour"]:
            peak_overlap_score = 1.0
        else:
            peak_overlap_score = 0.8
    else:
        # How close to nearest peak?
        distances = [abs(hour - ph) for ph in peak_hours]
        min_dist = min(distances)
        peak_overlap_score = max(0.0, 1.0 - (min_dist * 0.15))

    # 2. Day of week score
    weights = DAY_WEIGHTS.get(platform, [0.7] * 7)
    day_of_week_score = weights[day]

    # 3. Audience active percentage at this exact hour
    curve = data.get("audience_curve", {})
    audience_active_pct = curve.get(str(hour), 0.05)

    return {
        "peak_overlap_score": round(peak_overlap_score, 3),
        "day_of_week_score": round(day_of_week_score, 3),
        "audience_active_pct": round(audience_active_pct, 4)
    }


def get_best_time(platform: str) -> dict:
    """Return the best posting time metadata for a platform."""
    platform = platform.lower()
    if platform not in PEAK_WINDOWS:
        return {}
    data = PEAK_WINDOWS[platform]
    return {
        "best_day": data["best_day"],
        "best_hour": data["best_hour"],
        "peak_hours": data["peak_hours"]
    }


if __name__ == "__main__":
    platforms = ["instagram", "tiktok", "youtube", "linkedin"]
    # Test at different times
    test_times = [
        datetime(2026, 8, 5, 8, 0),   # Tuesday 8am
        datetime(2026, 8, 6, 3, 0),   # Wednesday 3am
        datetime(2026, 8, 8, 19, 0),  # Friday 7pm
    ]
    for plat in platforms:
        print(f"\n--- {plat.upper()} ---")
        for t in test_times:
            result = analyze_timing(plat, t)
            print(f"  {t.strftime('%A %H:00')} -> {result}")
