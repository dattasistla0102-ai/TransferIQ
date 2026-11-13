
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
FILE = PROJECT_DIR / "data" / "processed" / "players_2023_24_combined.csv"

df = pd.read_csv(FILE)

print("=" * 55)
print("TRANSFERIQ - DATASET VALIDATION")
print("=" * 55)

print(f"\nTotal rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

if "roster_match" in df.columns:
    print("\nRoster matching:")
    print(df["roster_match"].value_counts(dropna=False))

if "position" in df.columns:
    print("\nPosition distribution:")
    print(df["position"].value_counts(dropna=False))

if "Player" in df.columns:
    print("\nUnique player names in statistics:", df["Player"].nunique())

if "name_key" in df.columns:
    print("Unique normalized names:", df["name_key"].nunique())

if {"name_key", "team_key"}.issubset(df.columns):
    duplicates = df.duplicated(["name_key", "team_key"]).sum()
    print("Duplicate player/team keys:", duplicates)

print("\nMissing values in the 15 most incomplete columns:")
missing = (df.isna().mean() * 100).sort_values(ascending=False)
print(missing.head(15).round(1).astype(str).add("%").to_string())

print("\nSample rows:")
print(df.head(5).to_string(index=False, max_cols=12))

print("\nValidation complete.")
