
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

df = pd.read_csv(
    PROCESSED_DIR / "players_2023_24_combined.csv"
)

print("=" * 55)
print("POSITION COVERAGE")
print("=" * 55)

print("\nTotal player-team records:", len(df))

print("\nPosition counts:")
print(df["position"].value_counts(dropna=False).to_string())

print("\nRecords with a known position:")
print(df["position"].notna().sum())

print("\nRecords with a missing position:")
print(df["position"].isna().sum())

print("\nExamples of records with missing positions:")
print(
    df.loc[
        df["position"].isna(),
        ["name_key", "team_key"]
    ]
    .head(20)
    .to_string(index=False)
)

print("\nCheck complete.")
