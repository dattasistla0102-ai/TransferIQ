
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

files = [
    ROOT / "data/raw/roster_clean.csv",
    ROOT / "data/raw/Premleg_23_24",
    ROOT / "data/processed/historical_player_season_features.csv",
]

print("\n=== TRANSFERIQ: MARKET VALUE DATA AUDIT ===")

for path in files:
    if not path.exists():
        print(f"\nNOT FOUND: {path.relative_to(ROOT)}")
        continue

    if path.is_file() and path.suffix.lower() == ".csv":
        df = pd.read_csv(path, nrows=5)
        print(f"\nFILE: {path.relative_to(ROOT)}")
        print("Columns:", df.columns.tolist())
        print(df.head(2).to_string(index=False))

    elif path.is_dir():
        csv_files = list(path.glob("*.csv"))
        print(f"\nFOLDER: {path.relative_to(ROOT)}")
        print("CSV files:", len(csv_files))

        for csv_path in csv_files[:3]:
            df = pd.read_csv(csv_path, nrows=3)
            print(f"\nSample file: {csv_path.name}")
            print("Columns:", df.columns.tolist())
            print(df.head(2).to_string(index=False))

print("\n=== END AUDIT ===")
