
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"

input_path = DATA_DIR / "transfer_model_data_2020_2025.csv"
output_path = DATA_DIR / "transfer_model_data_imputed.csv"

df = pd.read_csv(input_path)

missing_before = df["market_value"].isna().sum()

print("Missing market values before imputation:", missing_before)

# Calculate medians using records with known market values.
overall_median = df["market_value"].median()
position_medians = df.groupby("position")["market_value"].median()

# Fill missing values using the median for the player's position.
df["market_value"] = df["market_value"].fillna(
    df["position"].map(position_medians)
)

# Fallback in case a position has no known market values.
df["market_value"] = df["market_value"].fillna(overall_median)

missing_after = df["market_value"].isna().sum()

print("\nOverall market-value median:", overall_median)
print("\nPosition medians:")
print(position_medians.to_string())

print("\nMissing market values after imputation:", missing_after)
print("Records retained:", len(df))

df.to_csv(output_path, index=False)

print("\nSaved to:", output_path)
print("Original dataset remains unchanged.")
