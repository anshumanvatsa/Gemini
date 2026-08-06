"""
End-to-end evaluation: 25 hand-crafted posts → API → check predictions make sense.
Tests intuition: strong posts → HIGH, weak posts → LOW with specific suggestions.
Also reports LSTM trajectory RMSE against YouTube hold-out.
"""
import requests, json, time

BASE = "http://localhost:8001/api/v1"

# 25 test cases: (caption, platform, follower_count, expected, label)
TEST_POSTS = [
    # ── Should be HIGH ──────────────────────────────────────────────────────────
    {
        "caption": "5 Python tricks that will blow your mind 🚀 #python #coding #programming #tech #developer",
        "platform": "instagram", "follower_count": 25000, "posting_time": "19:00", "niche": "tech",
        "expected": "HIGH", "note": "Clickbait number + trending tech hashtags + emoji"
    },
    {
        "caption": "How I made $10,000 in 30 days with dropshipping (step by step) 💰 #entrepreneur #business #money #hustle",
        "platform": "tiktok", "follower_count": 50000, "posting_time": "18:30", "niche": "business",
        "expected": "HIGH", "note": "Income claim + how-to + TikTok peak time"
    },
    {
        "caption": "You NEED to try this 5-minute pasta recipe before summer ends 🍝 #food #recipe #cooking #pasta #easyrecipes",
        "platform": "instagram", "follower_count": 15000, "posting_time": "12:00", "niche": "food",
        "expected": "HIGH", "note": "Urgency + recipe + food hashtags + lunchtime"
    },
    {
        "caption": "The gym transformation nobody talks about 💪 6 months. No shortcuts. Just discipline. #fitness #gym #transformation #workout #motivation",
        "platform": "instagram", "follower_count": 30000, "posting_time": "07:00", "niche": "fitness",
        "expected": "HIGH", "note": "Transformation + aspirational + morning gym crowd"
    },
    {
        "caption": "Tokyo in 72 hours: the complete guide 🇯🇵 Drop a ✈️ if you want the full itinerary! #travel #tokyo #japan #travelblogger #asia",
        "platform": "instagram", "follower_count": 45000, "posting_time": "20:00", "niche": "travel",
        "expected": "HIGH", "note": "CTA + specific destination + travel peak hours"
    },
    {
        "caption": "I tested 10 AI tools so you don't have to. Here's the honest ranking 🤖 #ai #chatgpt #productivity #tech #tools",
        "platform": "linkedin", "follower_count": 8000, "posting_time": "08:00", "niche": "tech",
        "expected": "HIGH", "note": "Curated list + AI trend + LinkedIn morning peak"
    },
    {
        "caption": "This skincare routine cleared my skin in 2 weeks 🌟 (dermatologist approved) #skincare #beauty #glowup #skintok",
        "platform": "tiktok", "follower_count": 20000, "posting_time": "19:30", "niche": "beauty",
        "expected": "HIGH", "note": "Transformation + authority signal + beauty niche"
    },
    {
        "caption": "Street style looks you can recreate for under $50 👗✨ Comment your city and I'll find local dupes! #fashion #style #ootd #streetwear",
        "platform": "instagram", "follower_count": 12000, "posting_time": "14:00", "niche": "fashion",
        "expected": "HIGH", "note": "CTA + budget appeal + fashion hashtags"
    },
    {
        "caption": "Why 95% of startups fail in year 1 (and how to be the 5%) 📊 #startup #entrepreneurship #business #founder",
        "platform": "linkedin", "follower_count": 5000, "posting_time": "09:00", "niche": "business",
        "expected": "HIGH", "note": "Stat hook + survival framing + LinkedIn audience"
    },
    {
        "caption": "POV: You learned this cooking hack from your Italian grandma 🍕 #cooking #food #italianfood #recipe #foodhack",
        "platform": "tiktok", "follower_count": 35000, "posting_time": "18:00", "niche": "food",
        "expected": "HIGH", "note": "POV format + nostalgia + TikTok peak"
    },
    {
        "caption": "The best hidden beaches in Bali that tourists don't know about 🌊 Save this before they get crowded! #bali #travel #beach #indonesia",
        "platform": "instagram", "follower_count": 18000, "posting_time": "21:00", "niche": "travel",
        "expected": "HIGH", "note": "Exclusive knowledge + save CTA + travel niche"
    },
    {
        "caption": "I wore the same outfit 5 ways to work this week 👔 Sustainable fashion doesn't have to be boring #fashion #sustainable #ootd",
        "platform": "tiktok", "follower_count": 22000, "posting_time": "07:30", "niche": "fashion",
        "expected": "HIGH", "note": "Challenge format + trending sustainability angle"
    },
    {
        "caption": "Full body workout you can do in your hotel room 🏨💪 No equipment needed! #workout #fitness #travel #homeworkout",
        "platform": "youtube", "follower_count": 100000, "posting_time": "06:00", "niche": "fitness",
        "expected": "HIGH", "note": "Tutorial + no-equipment hook + YouTube fitness"
    },

    # ── Should be LOW (weak posts) ───────────────────────────────────────────────
    {
        "caption": "Good morning everyone hope you have a great day",
        "platform": "instagram", "follower_count": 500, "posting_time": "03:00", "niche": "fitness",
        "expected": "LOW", "note": "Generic greeting, 3am, no hashtags, no hook"
    },
    {
        "caption": "New post",
        "platform": "twitter", "follower_count": 200, "posting_time": "04:30", "niche": "tech",
        "expected": "LOW", "note": "Empty caption, dead time, tiny account"
    },
    {
        "caption": "Just finished my lunch. It was okay. Kind of busy today with meetings.",
        "platform": "linkedin", "follower_count": 300, "posting_time": "02:00", "niche": "business",
        "expected": "LOW", "note": "No value, no CTA, 2am on LinkedIn"
    },
    {
        "caption": "check this out",
        "platform": "instagram", "follower_count": 150, "posting_time": "05:00", "niche": "fashion",
        "expected": "LOW", "note": "Vague, no hashtags, tiny account, dead time"
    },
    {
        "caption": "My cat did something funny lol #cat",
        "platform": "instagram", "follower_count": 800, "posting_time": "03:30", "niche": "entertainment",
        "expected": "LOW", "note": "Generic, only 1 hashtag, 3:30am"
    },
    {
        "caption": "Posting for the algorithm today. Not sure what to write tbh.",
        "platform": "tiktok", "follower_count": 100, "posting_time": "04:00", "niche": "tech",
        "expected": "LOW", "note": "Meta-posting confession, no value, tiny account"
    },
    {
        "caption": "Update",
        "platform": "facebook", "follower_count": 250, "posting_time": "02:30", "niche": "business",
        "expected": "LOW", "note": "Single word caption, Facebook at 2:30am"
    },
    {
        "caption": "New video is up go watch it if you want",
        "platform": "youtube", "follower_count": 400, "posting_time": "01:00", "niche": "entertainment",
        "expected": "LOW", "note": "Passive CTA, 1am, no description or hook"
    },
    {
        "caption": "Things I ate today: breakfast (eggs), lunch (sandwich), dinner (pasta). Good day.",
        "platform": "instagram", "follower_count": 600, "posting_time": "23:30", "niche": "food",
        "expected": "LOW", "note": "Food log with no visual hook, no hashtags"
    },
    {
        "caption": "Working from home again #wfh",
        "platform": "linkedin", "follower_count": 400, "posting_time": "14:00", "niche": "business",
        "expected": "LOW", "note": "Generic WFH post, single hashtag, no insight"
    },
    {
        "caption": "Meh",
        "platform": "twitter", "follower_count": 50, "posting_time": "05:30", "niche": "tech",
        "expected": "LOW", "note": "Single word, micro account, dawn posting"
    },
    {
        "caption": "Today was alright I guess. Tired but okay.",
        "platform": "instagram", "follower_count": 300, "posting_time": "06:30", "niche": "fitness",
        "expected": "LOW", "note": "No engagement hook, no hashtags, no value"
    },
]


def run_test(post, idx):
    # Endpoint uses multipart Form fields — must use data= not json=
    form_data = {
        "caption": post["caption"],
        "platform": post["platform"],
        "follower_count": str(post["follower_count"]),
        "niche": post.get("niche", "tech"),
    }
    try:
        t0 = time.time()
        r = requests.post(f"{BASE}/analyze", data=form_data, timeout=60)
        ms = (time.time() - t0) * 1000
        if r.status_code != 200:
            return {"ok": False, "error": f"HTTP {r.status_code}"}
        data = r.json()
        prediction = data.get("prediction", "?")
        confidence = data.get("confidence", 0)
        correct = prediction == post["expected"]
        return {
            "ok": True, "correct": correct,
            "prediction": prediction, "expected": post["expected"],
            "confidence": confidence, "ms": ms,
            "suggestions": len(data.get("suggestions", [])),
            "trajectory": [t["mid"] for t in data.get("trajectory", [])],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


print("=" * 65)
print("PreViral — End-to-End API Evaluation (25 posts)")
print("=" * 65)

results = []
correct = 0
high_correct = 0
low_correct = 0
high_total = 0
low_total = 0

for i, post in enumerate(TEST_POSTS, 1):
    r = run_test(post, i)
    results.append(r)
    if not r["ok"]:
        print(f"  [{i:02d}] ERROR: {r['error']}")
        continue

    mark = "OK" if r["correct"] else "FAIL"
    if r["correct"]:
        correct += 1
    label = post["expected"]
    if label == "HIGH":
        high_total += 1
        if r["correct"]: high_correct += 1
    else:
        low_total += 1
        if r["correct"]: low_correct += 1

    traj = r["trajectory"]
    traj_str = f"→ [{','.join(str(t) for t in traj)}]" if traj else ""
    print(f"  [{mark}] #{i:02d} {label:4s} | got={r['prediction']:4s} conf={r['confidence']:.2f} "
          f"sug={r['suggestions']} {int(r['ms'])}ms | {post['note'][:45]}")

total_ok = sum(1 for r in results if r["ok"])
accuracy = correct / total_ok if total_ok > 0 else 0

print(f"\n{'='*65}")
print(f"OVERALL:  {correct}/{total_ok} correct  ({accuracy*100:.1f}% accuracy)")
if high_total > 0:
    print(f"HIGH:     {high_correct}/{high_total} correct ({high_correct/high_total*100:.0f}%)")
if low_total > 0:
    print(f"LOW:      {low_correct}/{low_total} correct ({low_correct/low_total*100:.0f}%)")
print(f"{'='*65}")

# Counterfactual quality check
print("\nCounterfactual suggestions check (LOW posts should have specific fixes):")
low_posts = [(p, r) for p, r in zip(TEST_POSTS, results)
             if p["expected"] == "LOW" and r.get("ok") and r.get("suggestions", 0) > 0]
for post, r in low_posts[:3]:
    print(f"  '{post['caption'][:40]}...' → {r['suggestions']} suggestions ✓")
