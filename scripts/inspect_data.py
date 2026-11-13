
from pathlib import Path
import pandas as pd

# Find the project's raw data folder
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"

print("=" * 60)
print("TRANSFERIQ - DATASET INSPECTION")
print("=" * 60)

# Find all CSV files, including those in subfolders
csv_files = list(DATA_DIR.rglob("*.csv"))

print(f"\nCSV files found: {len(csv_files)}")

for file in csv_files:
    print("\n" + "=" * 60)
    print(f"FILE: {file.relative_to(DATA_DIR)}")

    try:
        df = pd.read_csv(file)

        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")
        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nFirst 3 rows:")
        print(df.head(3).to_string(index=False))

        print("\nMissing values:")
        print(df.isnull().sum().to_string())

    except Exception as error:
        print(f"Could not read this file: {error}")

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
