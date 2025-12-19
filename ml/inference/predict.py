from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml


# =========================
# Types
# =========================
@dataclass(frozen=True)
class LoadedArtifacts:
    model: object
    thresholds: Dict
    metadata: Dict
    model_version: str
    model_dir: str
    thresholds_version: str


# =========================
# IO helpers
# =========================
def load_yaml(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_json(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def get_thresholds_version(thresholds_path: str) -> str:
    return Path(thresholds_path).stem

# =========================
# Registry parsing
# =========================
def parse_registry_current(registry_md: str = "ml/models/registry.md") -> Dict[str, str]:
    """
    Extract current pointers from registry.md formatted like:
    - **key**: `value`  OR  - **key**: **value**
    """
    p = Path(registry_md)
    if not p.exists():
        raise FileNotFoundError(f"Registry not found: {p}")

    text = p.read_text(encoding="utf-8")

    def grab(key: str) -> str:
        m = re.search(
            rf"\*\*{re.escape(key)}\*\*:\s*(?:`([^`]+)`|\*\*([^*]+)\*\*)",
            text,
        )
        if not m:
            raise ValueError(f"Cannot find `{key}` in {registry_md}")
        return (m.group(1) or m.group(2)).strip()

    return {
        "current_version": grab("current_version"),
        "current_model_dir": grab("current_model_dir"),
        "thresholds_config": grab("thresholds_config"),
        # optional but recommended (you already have these in registry.md)
        "model_file": grab("model_file") if re.search(r"\*\*model_file\*\*", text) else "model.pkl",
        "metadata_file": grab("metadata_file") if re.search(r"\*\*metadata_file\*\*", text) else "metadata.json",
    }


def load_artifacts_from_registry(registry_md: str = "ml/models/registry.md") -> LoadedArtifacts:
    cur = parse_registry_current(registry_md)

    model_dir = Path(cur["current_model_dir"])
    thresholds_path = Path(cur["thresholds_config"])

    model_path = model_dir / cur.get("model_file", "model.pkl")
    metadata_path = model_dir / cur.get("metadata_file", "metadata.json")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not thresholds_path.exists():
        raise FileNotFoundError(f"Thresholds file not found: {thresholds_path}")

    model = joblib.load(model_path)
    thresholds = load_yaml(str(thresholds_path))
    metadata = load_json(str(metadata_path)) if metadata_path.exists() else {}

    return LoadedArtifacts(
        model=model,
        thresholds=thresholds,
        metadata=metadata,
        model_version=cur["current_version"],
        model_dir=str(model_dir),
        thresholds_version=get_thresholds_version(str(thresholds_path)), 
    )


# =========================
# Inference helpers
# =========================
def score(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def align_features_with_model(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Force X to have EXACT feature columns used during training
    (order + names).
    """
    if not hasattr(model, "feature_names_in_"):
        raise AttributeError(
            "Model does not have feature_names_in_. "
            "Train model with sklearn >=1.0 using pandas DataFrame."
        )

    trained_features = list(model.feature_names_in_)

    missing = [c for c in trained_features if c not in X.columns]
    extra = [c for c in X.columns if c not in trained_features]

    if missing:
        raise ValueError(f"Missing required features at inference: {missing}")

    if extra:
        X = X.drop(columns=extra)

    return X[trained_features]


def score_to_bucket_action(score_value: float, t1: float, t2: float) -> Tuple[str, str]:
    """
    bucket: low / medium / high
    action: approve / review / block
    """
    if score_value < t1:
        return "low", "approve"
    if score_value < t2:
        return "medium", "review"
    return "high", "block"


def predict_single(payload: dict, artifacts: Optional[LoadedArtifacts] = None) -> dict:
    """
    Returns:
    {"risk_score": float, "bucket": str, "action": str}
    """
    artifacts = artifacts or load_artifacts_from_registry()

    X = pd.DataFrame([payload])

    # Drop columns consistently with training
    drop_cols = artifacts.metadata.get("drop_cols", [])
    if "isfraud" in X.columns:
        drop_cols = list(drop_cols) + ["isfraud"]

    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")

    X = align_features_with_model(artifacts.model, X)

    risk_score = float(score(artifacts.model, X)[0])
    t1 = float(artifacts.thresholds["t1"])
    t2 = float(artifacts.thresholds["t2"])

    bucket, action = score_to_bucket_action(risk_score, t1, t2)

    return {
        "risk_score": risk_score, 
        "bucket": bucket, 
        "action": action,
        "thresholds_version": artifacts.thresholds_version 
    }


def predict_df(
    X: pd.DataFrame,
    artifacts: LoadedArtifacts,
    *,
    return_debug: bool = False,
) -> pd.DataFrame:
    thr = artifacts.thresholds or {}
    t1 = float(thr.get("t1"))
    t2 = float(thr.get("t2"))

    y_score = score(artifacts.model, X).astype(float)

    buckets: list[str] = []
    actions: list[str] = []
    for s in y_score:
        b, a = score_to_bucket_action(float(s), t1, t2)
        buckets.append(b)
        actions.append(a)

    out = pd.DataFrame({"risk_score": y_score, "bucket": buckets, "action": actions})

    if return_debug:
        out["t1"] = t1
        out["t2"] = t2
        out["model_version"] = artifacts.model_version
        out["model_dir"] = artifacts.model_dir

    return out


# =========================
# CLI
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference - load current model from ml/models/registry.md"
    )
    parser.add_argument("--input", type=str, required=True, help="Path to parquet/csv input features")
    parser.add_argument("--format", type=str, default="parquet", choices=["parquet", "csv"])
    parser.add_argument("--registry", type=str, default="ml/models/registry.md")
    parser.add_argument("--out", type=str, default="ml/inference/predictions.csv")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    artifacts = load_artifacts_from_registry(args.registry)

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    if args.format == "parquet":
        X = pd.read_parquet(in_path)
    else:
        X = pd.read_csv(in_path)

    # Drop columns consistently with training (prevents sklearn feature-name mismatch)
    drop_cols = []
    dc = artifacts.metadata.get("drop_cols")
    if isinstance(dc, list):
        drop_cols.extend([str(c) for c in dc])
    drop_cols.append("isfraud")

    present = [c for c in sorted(set(drop_cols)) if c in X.columns]
    if present:
        X = X.drop(columns=present)

    X = align_features_with_model(artifacts.model, X)

    preds = predict_df(X, artifacts, return_debug=args.debug)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.out, index=False)

    print("[OK] Loaded current model:", artifacts.model_version)
    print("[OK] Wrote predictions:", args.out)
    if args.debug:
        print(
            "[OK] thresholds:",
            {"t1": float(artifacts.thresholds.get("t1")), "t2": float(artifacts.thresholds.get("t2"))},
        )


if __name__ == "__main__":
    main()