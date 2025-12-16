from __future__ import annotations
from pathlib import Path

ROOT = Path.cwd()

DIRS = [
    "docs",
    "data/raw",
    "data/processed",
    "pipelines/ingest",
    "pipelines/transform/sql",
    "pipelines/transform/tests",
    "pipelines/features",
    "analytics/powerbi",
    "analytics/notebooks",
    "ml/training",
    "ml/models",
    "ml/evaluation/reports",
    "ml/inference",
    "app/api/tests",
    "app/web",
    "app/mobile",
    "ops/docker",
    "ops/ci",
    "ops/infra",
    "ops/monitoring",
    "tests",
    "scripts",
]


FILES = [
    "README.md",
    "requirements.txt",
    "Makefile",
    "docs/00_problem.md",
    "docs/01_data.md",
    "docs/02_etl.md",
    "docs/03_modeling.md",
    "docs/04_evaluation.md",
    "docs/05_deployment.md",
    "docs/06_monitoring.md",
    "docs/architecture.png",  
    "docs/demo.gif",          
    "data/README.md",
    "pipelines/ingest/fetch_data.py",
    "pipelines/ingest/validate_schema.py",
    "pipelines/transform/spark_job.py",
    "pipelines/features/build_features.py",
    "pipelines/features/feature_spec.yaml",
    "analytics/powerbi/dashboard.pbix",  
    "analytics/powerbi/dax_measures.md",
    "analytics/notebooks/eda_insights.ipynb",  
    "analytics/metrics_definition.md",
    "ml/training/train.py",
    "ml/training/tune.py",
    "ml/training/config.yaml",
    "ml/models/model.py",
    "ml/models/registry.md",
    "ml/evaluation/eval.py",
    "ml/inference/predict.py",
    "ml/inference/schema.py",
    "app/api/main.py",
    "app/api/routes.py",
    "app/api/deps.py",
    "ops/docker/Dockerfile",
    "ops/docker/docker-compose.yml",
    "ops/ci/github-actions.yml",
    "ops/infra/deploy_notes.md",
    "ops/monitoring/logging.md",
    "ops/monitoring/drift_checks.py",
    "tests/test_data.py",
    "tests/test_features.py",
    "tests/test_api.py",
    "scripts/run_local.sh",
    "scripts/run_etl.sh",
    "scripts/train_model.sh",
    "scripts/serve_api.sh",
]

TEMPLATES = {
    "README.md": "# PaySim Fraud Risk Scoring\n\n> ML Engineer portfolio project.\n",
    "data/README.md": "# Data\n\nFull dataset is not committed. Use sample + scripts.\n",
    "pipelines/ingest/fetch_data.py": '"""Fetch or document how to obtain the PaySim dataset."""\n\n',
    "pipelines/ingest/validate_schema.py": '"""Validate schema for raw PaySim CSV."""\n\n',
    "pipelines/transform/spark_job.py": '"""Transform raw -> processed (Spark or pandas)."""\n\n',
    "pipelines/features/build_features.py": '"""Build feature table from processed data."""\n\n',
    "ml/training/train.py": '"""Train baseline and improved models."""\n\n',
    "ml/evaluation/eval.py": '"""Evaluate model (PR-AUC primary, threshold analysis)."""\n\n',
    "app/api/main.py": '"""FastAPI entrypoint."""\n\nfrom fastapi import FastAPI\n\napp = FastAPI(title="PaySim Risk Scoring API")\n\n@app.get("/health")\ndef health():\n    return {"status": "ok"}\n',
    "requirements.txt": "pandas\nnumpy\nscikit-learn\nfastapi\nuvicorn\npydantic\npytest\nruff\n",
}

def touch_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(TEMPLATES.get(path.as_posix(), ""), encoding="utf-8")

def main() -> None:
    for d in DIRS:
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    for f in FILES:
        touch_file(ROOT / f)

    print("Repo skeleton created.")
    print(f"Root: {ROOT}")

if __name__ == "__main__":
    main()