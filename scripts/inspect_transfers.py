
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
TRANSFER_DIR = PROJECT_DIR / "data" / "raw" / "transfers"

files = sorted(TRANSFER_DIR.glob("*.csv"))

if not files:
    print("ERROR: No transfer CSV files found.")
    raise SystemExit(1)

total_rows = 0

print("=" * 70)
print("TRANSFER DATASET INSPECTION")
print("=" * 70)

for file in files:
    try:
        df = pd.read_csv(file)
    except Exception as error:
        print(f"\nERROR reading {file.name}: {error}")
        continue

    total_rows += len(df)

    print(f"\n{'-' * 60}")
    print(f"File: {file.name}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print("Column names:")
    print(df.columns.tolist())

    if file.name in [
        "premier_league_1992.csv",
        "premier_league_2023.csv",
        "premier_league_2024.csv",
        "premier_league_2025.csv",
    ]:
        print("\nFirst 3 records:")
        print(df.head(3).to_string(index=False))

        print("\nMissing values:")
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        print(missing.head(15).to_string() if len(missing) else "None")

print("\n" + "=" * 70)
print("INSPECTION SUMMARY")
print("=" * 70)
print("CSV files found:", len(files))
print("Total rows across files:", total_rows)
print("\nInspection complete. No files were modified.")
