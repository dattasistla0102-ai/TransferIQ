
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

# 1. Inspect fees and loan status together
print("\n========== 1. FEE AND LOAN BREAKDOWN ==========")

print(
    pd.crosstab(
        df["is_loan"],
        df["fee"].apply(
            lambda x: "missing" if pd.isna(x)
            else "zero" if x == 0
            else "positive"
        ),
        margins=True,
    )
)

# 2. Inspect incoming transfers separately
incoming = df[df["movement"] == "in"].copy()

print("\n========== 2. INCOMING TRANSFERS ==========")
print("Total incoming records:", len(incoming))
print("Missing fees:", incoming["fee"].isna().sum())
print("Zero fees:", (incoming["fee"] == 0).sum())
print("Loan records:", (incoming["is_loan"] == 1).sum())

print("\nIncoming fee/loan breakdown:")
print(
    pd.crosstab(
        incoming["is_loan"],
        incoming["fee"].apply(
            lambda x: "missing" if pd.isna(x)
            else "zero" if x == 0
            else "positive"
        ),
        margins=True,
    )
)

# 3. Check exact duplicate rows across all columns
print("\n========== 3. EXACT DUPLICATES ==========")

exact_duplicate_mask = df.duplicated(keep=False)
print("Rows belonging to exact duplicate groups:",
      exact_duplicate_mask.sum())

if exact_duplicate_mask.any():
    print("\nExample exact duplicates:")
    print(
        df.loc[exact_duplicate_mask]
        .sort_values(["source_season", "player_id"])
        .head(12)
        .to_string(index=False)
    )

# 4. Inspect incoming zero-fee records
print("\n========== 4. INCOMING ZERO-FEE EXAMPLES ==========")

columns = [
    "source_season", "window", "club", "player_name",
    "position", "fee", "market_value", "is_loan"
]

zero_fee_incoming = incoming[incoming["fee"] == 0]

print("Count:", len(zero_fee_incoming))
print(
    zero_fee_incoming[columns]
    .head(20)
    .to_string(index=False)
)

# 5. Save nothing: this is a diagnostic-only script
print("\n========== CHECK COMPLETE ==========")
print("No records were deleted or modified.")
