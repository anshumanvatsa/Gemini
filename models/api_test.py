import requests

BASE = 'http://localhost:8001/api/v1'

tests = [
    {
        'name': 'High-quality Instagram (fitness)',
        'data': {
            'caption': 'Just crushed my morning workout! 5am club, no excuses. Drop a fire emoji if you trained today! Link in bio for my full 30-day program. #fitness #gym #motivation #workout #health #gains #bodybuilding',
            'platform': 'instagram',
            'post_datetime': '2025-08-12T11:30:00',
            'follower_count': '25000',
            'niche': 'fitness',
        }
    },
    {
        'name': 'Low-effort post (3am off-peak)',
        'data': {
            'caption': 'Good morning.',
            'platform': 'instagram',
            'post_datetime': '2025-08-12T03:00:00',
            'follower_count': '500',
            'niche': 'general',
        }
    },
    {
        'name': 'Twitter viral thread hook',
        'data': {
            'caption': 'I studied 500 viral tweets in the last 30 days. Here is EXACTLY what made them go viral (most people get this completely wrong): THREAD',
            'platform': 'twitter',
            'post_datetime': '2025-08-12T09:00:00',
            'follower_count': '8000',
            'niche': 'business',
        }
    },
    {
        'name': 'YouTube SEO video',
        'data': {
            'caption': '10 Python tricks that will BLOW YOUR MIND in 2025 | Advanced Tutorial for Developers - Complete guide to mastering Python performance, memory management, and async programming. Subscribe for weekly tutorials!',
            'platform': 'youtube',
            'post_datetime': '2025-08-12T16:00:00',
            'follower_count': '45000',
            'niche': 'tech',
        }
    },
]

print("=" * 65)
print("LIVE API TEST — PreViral v3 (F1=0.88, AUC=0.96)")
print("=" * 65)

for t in tests:
    try:
        r = requests.post(f'{BASE}/analyze', data=t['data'], timeout=30)
        if r.status_code == 200:
            d = r.json()
            pred = d.get('prediction', '?')
            conf = d.get('confidence', '?')
            reach = d.get('reach_percentile', '?')
            model = d.get('model_version', '?')
            print(f"\n{t['name']}")
            print(f"  Prediction: {pred}  |  Confidence: {conf}  |  Reach: {reach}th pct")
            if 'suggestions' in d and d['suggestions']:
                print(f"  Top suggestion: {d['suggestions'][0].get('action', '')[:80]}")
        else:
            print(f"\n{t['name']}: HTTP {r.status_code}")
            print(f"  {r.text[:300]}")
    except Exception as e:
        print(f"\n{t['name']}: ERROR {e}")
