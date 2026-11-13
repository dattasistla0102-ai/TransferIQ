
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

# Recent incoming transfers only: 2020/21 through 2025/26
recent = df[
    (df["movement"] == "in")
    & (df["source_season"].between(2020, 2025))
].copy()

print("========== RECENT INCOMING TRANSFERS ==========")
print("Total records:", len(recent))

print("\nFee categories by season:")
recent["fee_category"] = recent["fee"].apply(
    lambda x: "Missing" if pd.isna(x)
    else "Zero" if x == 0
    else "Positive"
)

print(
    pd.crosstab(
        recent["source_season"],
        recent["fee_category"],
    ).to_string()
)

print("\n========== ZERO-FEE, NON-LOAN ARRIVALS ==========")

zero_permanent = recent[
    (recent["fee"] == 0)
    & (recent["is_loan"] == 0)
].copy()

print("Count:", len(zero_permanent))

columns = [
    "source_season",
    "window",
    "club",
    "player_name",
    "age",
    "position",
    "market_value",
    "fee",
    "is_loan",
]

print(
    zero_permanent[columns]
    .sort_values(["source_season", "club"])
    .head(40)
    .to_string(index=False)
)

print("\n========== POSITIVE-FEE, NON-LOAN ARRIVALS ==========")

paid_permanent = recent[
    (recent["fee"] > 0)
    & (recent["is_loan"] == 0)
].copy()

print("Count:", len(paid_permanent))
print("\nFee summary:")
print(paid_permanent["fee"].describe().to_string())

print("\nRecords by season:")
print(paid_permanent.groupby("source_season").size().to_string())

print("\nNo files were modified.")
