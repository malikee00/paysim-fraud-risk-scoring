import json
from pathlib import Path
import pandas as pd

TRAIN_STATS_PATH = Path("data/processed_local/summary_v2_full_sakti.json")
RECENT_DATA_PATH = Path("ml/inference/recent_samples.csv")
OUTPUT_PATH = Path("ml/reports/drift_report.md")

FEATURES_TO_CHECK = [
    "log_amount",
    "org_delta_mismatch",
    "dest_delta_mismatch",
    "dest_txn_count_w10",
    "ratio_amt_to_dest_mean_w10",
]

DRIFT_THRESHOLD = 0.3  


def load_train_feature_means(summary: dict) -> dict:
    feats = summary.get("features")

    means = {}

    if isinstance(feats, list):
        for item in feats:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not name:
                continue
            if "mean" in item:
                means[name] = float(item["mean"])
            elif "avg" in item:
                means[name] = float(item["avg"])

    elif isinstance(feats, dict):
        for name, stats in feats.items():
            if isinstance(stats, dict):
                if "mean" in stats:
                    means[name] = float(stats["mean"])
                elif "avg" in stats:
                    means[name] = float(stats["avg"])

    return means


def relative_change(a, b):
    if a == 0:
        return None
    return abs(a - b) / abs(a)


def main():
    summary = json.load(open(TRAIN_STATS_PATH, "r", encoding="utf-8"))
    train_means = load_train_feature_means(summary)

    recent_df = pd.read_csv(RECENT_DATA_PATH)

    lines = []
    lines.append("# Drift Report\n")
    lines.append("Feature drift check using mean shift (proxy recent samples).\n")
    lines.append(f"- Drift threshold: {DRIFT_THRESHOLD:.0%} relative mean change\n")
    lines.append(f"- Train rows: {summary.get('n_rows')}\n")

    checked_any = False

    for feat in FEATURES_TO_CHECK:
        if feat not in train_means:
            lines.append(f"## {feat}")
            lines.append("- Status: **SKIPPED** (not found in training summary)\n")
            continue

        if feat not in recent_df.columns:
            lines.append(f"## {feat}")
            lines.append("- Status: **SKIPPED** (not found in recent_samples.csv)\n")
            continue

        train_mean = train_means[feat]
        recent_mean = float(recent_df[feat].mean())
        change = relative_change(train_mean, recent_mean)

        status = "OK"
        if change is not None and change > DRIFT_THRESHOLD:
            status = "POTENTIAL DRIFT"

        lines.append(f"## {feat}")
        lines.append(f"- Train mean: {train_mean:.6f}")
        lines.append(f"- Recent mean: {recent_mean:.6f}")
        lines.append(f"- Relative change: {change:.2%}" if change is not None else "- Relative change: N/A")
        lines.append(f"- Status: **{status}**\n")

        checked_any = True

    if not checked_any:
        lines.append("## Note")
        lines.append("No features were checked. Verify FEATURES_TO_CHECK matches both training summary and recent samples.\n")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Drift report saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()