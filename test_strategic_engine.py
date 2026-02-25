import requests
import json

URL = "http://localhost:8001/predict/full"

payload = {
    "property_id": "test-prop-123",
    "property_name": "Test Property",
    "period_start": "2026-01-01",
    "period_end": "2026-01-31",
    "historical_data": [
        {
            "date": f"2026-01-{i+1:02d}",
            "users": 100 + i,
            "new_users": 80,
            "returning_users": 20 + i,
            "sessions": 120 + i,
            "conversions": 5 + (i // 5),
            "bounce_rate": 65.5,
            "avg_session_duration": 145.0,
            "channels": {
                "Organic Search": {"users": 50, "conversions": 2},
                "Paid Search": {"users": 30, "conversions": 3}
            },
            "sources": {"google": 80, "bing": 20}
        } for i in range(20)
    ],
    "google_ads_data": [
        {
            "name": "Spring Sale / Google",
            "total_cost": 500.0,
            "total_clicks": 200,
            "total_impressions": 5000,
            "total_conversions": 15,
            "keywords": ["sale", "discount"]
        }
    ],
    "top_queries": [
        {"name": "high intent shoes", "clicks": 100, "impressions": 1000, "position": 2.1, "ctr": 0.1},
        {"name": "cheap sneakers", "clicks": 20, "impressions": 500, "position": 5.5, "ctr": 0.04}
    ],
    "by_city": [
        {"name": "Nairobi", "activeUsers": 500},
        {"name": "Mombasa", "activeUsers": 200}
    ],
    "config": {"forecast_days": 90}
}

try:
    print("Sending test request to Strategic Engine...")
    response = requests.post(URL, json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(response.text)
except Exception as e:
    print(f"Error: {e}")
