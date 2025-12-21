import pandas as pd
from pathlib import Path

FEATURES_PATH = Path(
    "data/processed_local/features_v2_full_sakti.parquet"
)

OUTPUT_PATH = Path(
    "ml/inference/recent_samples.csv"
)

N_SAMPLES = 1000

def main():
    df = pd.read_parquet(FEATURES_PATH)

    recent_df = (
        df.sort_values("step")
          .tail(N_SAMPLES)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    recent_df.to_csv(OUTPUT_PATH, index=False)

    print(f"[OK] Recent inference samples saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
