
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "transfer_history_all_seasons.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)

# Keep recent incoming transfers from seasons 2020–2025.
# The season number is the year the season started.
df = df[
    (df["movement"] == "in")
    & (df["source_season"].between(2020, 2025))
].copy()

records_before = len(df)

# Remove only rows identical across every column.
df = df.drop_duplicates().copy()

exact_duplicates_removed = records_before - len(df)

# First prototype: positive-fee, non-loan arrivals only.
model_data = df[
    (df["fee"] > 0)
    & (df["is_loan"] == 0)
].copy()

# Keep records with a known target and basic player information.
model_data = model_data.dropna(
    subset=["fee", "player_name", "position", "age"]
).copy()

# Save the prepared dataset.
output_path = OUTPUT_DIR / "transfer_model_data_2020_2025.csv"
model_data.to_csv(output_path, index=False)

# Save separate chronological periods for future evaluation.
# Do not train or evaluate a model yet.
train = model_data[model_data["source_season"].between(2020, 2023)]
validation = model_data[model_data["source_season"] == 2024]
test = model_data[model_data["source_season"] == 2025]

train.to_csv(
    OUTPUT_DIR / "transfer_model_train_2020_2023.csv",
    index=False,
)
validation.to_csv(
    OUTPUT_DIR / "transfer_model_validation_2024.csv",
    index=False,
)
test.to_csv(
    OUTPUT_DIR / "transfer_model_test_2025.csv",
    index=False,
)

print("========== MODELLING DATA PREPARATION ==========")
print("Recent incoming records before deduplication:", records_before)
print("Exact duplicate rows removed:", exact_duplicates_removed)
print("Positive-fee, non-loan records:", len(model_data))

print("\nRecords per season:")
print(model_data.groupby("source_season").size().to_string())

print("\nChronological split:")
print("Training (2020–2023):", len(train))
print("Validation (2024):", len(validation))
print("Test (2025):", len(test))

print("\nMissing values in candidate features:")
for col in ["age", "position", "market_value", "club", "fee"]:
    if col in model_data.columns:
        print(f"{col}: {model_data[col].isna().sum()}")

print("\nSaved modelling datasets in:", OUTPUT_DIR)
print("Master transfer history remains unchanged.")
