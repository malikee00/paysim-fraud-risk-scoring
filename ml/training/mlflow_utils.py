from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow


def setup_mlflow(cfg: Dict[str, Any], default_run_name: str) -> Dict[str, Any]:
    """
    Returns mlflow config dict. If disabled, returns {"enable": False}.
    """
    mlcfg = cfg.get("mlflow", {}) or {}
    enable = bool(mlcfg.get("enable", False))
    if not enable:
        return {"enable": False}

    tracking_uri = str(mlcfg.get("tracking_uri", "mlruns"))
    
    abs_path = Path(tracking_uri).resolve()
    mlflow_tracking_uri = abs_path.as_uri() 

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    experiment_name = str(mlcfg.get("experiment_name", "default"))
    mlflow.set_experiment(experiment_name)

    return {
        "enable": True,
        "run_name": str(mlcfg.get("run_name", default_run_name)),
        "tags": dict(mlcfg.get("tags", {}) or {}),
        "tracking_uri": mlflow_tracking_uri,
        "experiment_name": experiment_name,
    }

def log_params_flat(params: Dict[str, Any], prefix: Optional[str] = None) -> None:
    flat = {}
    for k, v in params.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        # MLflow params must be string-ish
        flat[key] = json.dumps(v) if isinstance(v, (dict, list)) else v
    mlflow.log_params(flat)


def log_metrics_flat(metrics: Dict[str, float], prefix: Optional[str] = None) -> None:
    for k, v in metrics.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        try:
            mlflow.log_metric(key, float(v))
        except Exception:
            pass


def log_artifact_if_exists(path: str) -> None:
    p = Path(path)
    if p.exists() and p.is_file():
        mlflow.log_artifact(str(p.resolve()))
