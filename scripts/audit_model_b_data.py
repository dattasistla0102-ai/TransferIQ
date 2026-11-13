
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "Raw transfer files": ROOT / "data" / "raw" / "transfers",
    "Historical player stats": ROOT / "data" / "raw" / "historical_player_stats",
    "Premier League stats": ROOT / "data" / "raw" / "Premleg_23_24",
    "Transfer history dataset": ROOT / "data" / "processed" / "transfer_history_all_seasons.csv",
    "Transfer model dataset": ROOT / "data" / "processed" / "transfer_model_data_2020_2025.csv",
    "Market value dataset": ROOT / "data" / "processed" / "market_value_dataset.csv",
    "Historical player features": ROOT / "data" / "processed" / "historical_player_season_features.csv",
}

KEYWORDS = [
    "contract", "expiry", "expiration", "salary", "wage",
    "market_value", "transfer", "player_id", "player_name",
    "season", "date", "fee", "club", "age", "position",
]

print("=" * 75)
print("TRANSFERIQ — MODEL B DATA AUDIT")
print("=" * 75)

for label, path in FILES.items():
    print(f"\n{'-' * 75}\n{label}\nPath: {path}")

    if not path.exists():
        print("NOT FOUND")
        continue

    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        print(f"CSV files found: {len(files)}")
        for file in files[:12]:
            print(f"  - {file.name}")
        if len(files) > 12:
            print(f"  ... and {len(files) - 12} more")
        continue

    try:
        df = pd.read_csv(path, nrows=5)
        full_df = pd.read_csv(path, usecols=lambda c: True, nrows=0)
        columns = full_df.columns.tolist()

        print(f"Columns ({len(columns)}):")
        print(", ".join(columns))

        relevant = [
            c for c in columns
            if any(word in c.lower() for word in KEYWORDS)
        ]
        print("\nPotentially useful columns:")
        print(", ".join(relevant) if relevant else "None identified")

        print("\nSample rows:")
        print(df.head(3).to_string(index=False, max_cols=15))

    except Exception as exc:
        print(f"Could not inspect file: {exc}")

print("\n" + "=" * 75)
print("AUDIT COMPLETE — NO FILES WERE MODIFIED")
print("=" * 75)
