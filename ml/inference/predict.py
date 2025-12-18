from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class LoadedArtifacts:
    model: object
    thresholds: Dict
    metadata: Dict
    model_version: str
    model_dir: str


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


def load_current_artifacts(current_yaml: str = "ml/models/current.yaml") -> LoadedArtifacts:
    cfg = load_yaml(current_yaml)

    cur = cfg.get("current", {}) or {}
    thr = cfg.get("thresholds", {}) or {}

    model_version = str(cur.get("version", "unknown"))
    model_dir = str(cur.get("model_dir"))
    model_path = str(cur.get("model_path"))
    metadata_path = str(cur.get("metadata_path"))
    thresholds_path = str(thr.get("path"))

    if not model_path:
        raise ValueError("current.model_path is missing in current.yaml")
    if not thresholds_path:
        raise ValueError("thresholds.path is missing in current.yaml")

    model = joblib.load(model_path)

    thresholds = load_yaml(thresholds_path)
    metadata = load_json(metadata_path) if metadata_path else {}

    return LoadedArtifacts(
        model=model,
        thresholds=thresholds,
        metadata=metadata,
        model_version=model_version,
        model_dir=model_dir,
    )


def score(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def decide(score_value: float, t1: float, t2: float) -> str:
    # APPROVE / REVIEW / BLOCK
    if score_value >= t2:
        return "BLOCK"
    if score_value >= t1:
        return "REVIEW"
    return "APPROVE"


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
    actions = [decide(s, t1, t2) for s in y_score]

    out = pd.DataFrame({"score": y_score, "action": actions})
    if return_debug:
        out["t1"] = t1
        out["t2"] = t2
        out["model_version"] = artifacts.model_version
        out["model_dir"] = artifacts.model_dir
    return out

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference - load current model from ml/models/current.yaml")
    parser.add_argument("--input", type=str, required=True, help="Path to parquet/csv input features")
    parser.add_argument("--format", type=str, default="parquet", choices=["parquet", "csv"])
    parser.add_argument("--out", type=str, default="ml/inference/predictions.csv")
    parser.add_argument("--current", type=str, default="ml/models/current.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    artifacts = load_current_artifacts(args.current)

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    if args.format == "parquet":
        X = pd.read_parquet(in_path)
    else:
        X = pd.read_csv(in_path)

    # Drop columns consistently with training (prevents sklearn feature-name mismatch)
    drop_cols = []

    if artifacts.metadata:
        dc = artifacts.metadata.get("drop_cols")
        if isinstance(dc, list):
            drop_cols.extend([str(c) for c in dc])

    drop_cols.append("isfraud")

    drop_cols = sorted(set(drop_cols))
    present = [c for c in drop_cols if c in X.columns]
    if present:
        X = X.drop(columns=present)

    X = align_features_with_model(artifacts.model, X)

    preds = predict_df(X, artifacts, return_debug=args.debug)   

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.out, index=False)

    print("[OK] Loaded current model:", artifacts.model_version)
    print("[OK] Wrote predictions:", args.out)
    if args.debug:
        print("[OK] thresholds:", {"t1": float(artifacts.thresholds.get("t1")), "t2": float(artifacts.thresholds.get("t2"))})


if __name__ == "__main__":
    main()