
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"

files = [
    DATA_DIR / "transfer_model_train_2020_2023.csv",
    DATA_DIR / "transfer_model_validation_2024.csv",
]

for file_path in files:
    print("\n" + "=" * 70)
    print(f"FILE: {file_path.name}")
    print("=" * 70)

    df = pd.read_csv(file_path)

    print("\nAll columns:")
    print(df.columns.tolist())

    # Display columns that might help establish when market value
    # and the transfer fee were recorded.
    keywords = [
        "player", "name", "market", "value", "fee",
        "date", "season", "club", "team", "source"
    ]

    relevant_columns = [
        col for col in df.columns
        if any(word in col.lower() for word in keywords)
    ]

    print("\nRelevant columns and sample records:")
    print(df[relevant_columns].head(10).to_string(index=False))

    print("\nMarket value summary:")
    if "market_value" in df.columns:
        print(df["market_value"].describe())
        print("Missing values:", df["market_value"].isna().sum())
        print("Zero values:", (df["market_value"] == 0).sum())
    else:
        print("No column named 'market_value' was found.")

print("\nInspection complete.")
