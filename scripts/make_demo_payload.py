import json
import joblib
from pathlib import Path

# load model v2
model = joblib.load("ml/models/v2_improved/model.pkl")

# create dummy feature-level payload
payload = {c: 0 for c in model.feature_names_in_}

# ensure docs folder exists
Path("docs").mkdir(exist_ok=True)

# write demo payload
out_path = Path("docs/demo_payload.json")
with out_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print(f"[OK] Wrote {out_path}")
print(payload)
