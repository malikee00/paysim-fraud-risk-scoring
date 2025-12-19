import joblib
from ml.inference.predict import load_artifacts_from_registry, predict_single

print("== Check model ==")
model = joblib.load("ml/models/v2_improved/model.pkl")
print("n_features:", len(model.feature_names_in_))

print("\n== Check raw payload mismatch ==")
raw_payload = {
  "step":180,
  "type":"CASH_OUT",
  "amount":114664.45,
  "nameOrig":"C1911904419",
  "oldbalanceOrg":0.0,
  "newbalanceOrig":0.0,
  "nameDest":"C763019604",
  "oldbalanceDest":163667.97,
  "newbalanceDest":278332.42
}

missing = [c for c in model.feature_names_in_ if c not in raw_payload]
print("missing:", len(missing))

print("\n== Smoke test inference ==")
art = load_artifacts_from_registry()
dummy = {c: 0 for c in art.model.feature_names_in_}
print(predict_single(dummy, art))
