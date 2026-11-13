
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "transfer_history_all_seasons.csv"
)

df = pd.read_csv(DATA_PATH)

print("========== 1. TRANSFER DIRECTIONS ==========")
print(df["movement"].value_counts(dropna=False))

print("\n========== 2. LOAN STATUS ==========")
print(df["is_loan"].value_counts(dropna=False))

print("\n========== 3. TRANSFER WINDOWS ==========")
print(df["window"].value_counts(dropna=False))

print("\n========== 4. FEE DATA TYPES ==========")
print("Fee data type:", df["fee"].dtype)
print("Example fee values:")
print(df["fee"].dropna().head(15).to_list())

print("\n========== 5. MISSING FEES BY SEASON ==========")
print(
    df.groupby("source_season")["fee"]
    .agg(
        total_records="size",
        missing_fees=lambda s: s.isna().sum(),
    )
    .to_string()
)

print("\n========== 6. POSSIBLE DUPLICATES ==========")

# This is only a diagnostic key, not proof of a duplicate.
key_columns = [
    "source_season",
    "window",
    "club",
    "movement",
    "player_id",
]

duplicate_mask = df.duplicated(
    subset=key_columns,
    keep=False,
)

print("Records sharing the same diagnostic key:", duplicate_mask.sum())

if duplicate_mask.any():
    print("\nExample records to investigate:")
    print(
        df.loc[
            duplicate_mask,
            key_columns + ["player_name", "fee", "is_loan"],
        ]
        .sort_values(key_columns)
        .head(20)
        .to_string(index=False)
    )

print("\n========== 7. RECENT TRANSFER EXAMPLES ==========")
columns = [
    "source_season",
    "window",
    "club",
    "movement",
    "player_name",
    "position",
    "market_value",
    "fee",
    "is_loan",
]

print(
    df.loc[df["source_season"].between(2023, 2025), columns]
    .head(20)
    .to_string(index=False)
)

print("\n========== VALIDATION COMPLETE ==========")
