import requests, json, sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

BASE = "http://localhost:8000"

print("=== Sending SOS via /api/send-sos ===")
try:
    r = requests.post(f"{BASE}/api/send-sos", json={
        "fatigueScore": 95.0,
        "ear": 0.12,
        "mar": 0.88,
        "blinkCount": 50,
        "yawnCount": 15,
        "distractionDuration": 12.0,
        "latitude": 23.2599,
        "longitude": 77.4126,
        "customApiKey": ""
    })
    print(f"Status Code: {r.status_code}")
    print(f"Response JSON:\n{json.dumps(r.json(), indent=2, ensure_ascii=False)}")
except Exception as e:
    print(f"Request failed: {e}")
