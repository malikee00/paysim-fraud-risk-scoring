import json
import time
import statistics
from pathlib import Path
import requests

API_URL = "http://127.0.0.1:8000/predict"
N_RUNS = 50
PAYLOAD_PATH = Path("ml/inference/sample_request.json")

payload = json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))

latencies = []
for _ in range(N_RUNS):
    start = time.time()
    r = requests.post(API_URL, json=payload, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Request failed: {r.status_code} | {r.text}")
    latencies.append((time.time() - start) * 1000)

latencies.sort()
p50 = statistics.median(latencies)
p95 = latencies[int(len(latencies) * 0.95)]

print(f"p50 latency: {p50:.2f} ms")
print(f"p95 latency: {p95:.2f} ms")
