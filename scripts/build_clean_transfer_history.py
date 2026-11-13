
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

SOURCE = DATA / "transfer_history_all_seasons.csv"
OUTPUT = DATA / "transfer_history_clean.csv"

df = pd.read_csv(SOURCE)

print(f"Original rows: {len(df)}")

# The source records transfers from both clubs' perspectives.
# Keep incoming transfers so each deal is represented once.
df["movement"] = df["movement"].astype(str).str.strip().str.lower()
df = df[df["movement"] == "in"].copy()

print(f"Incoming-transfer rows: {len(df)}")

# Remove rows that are identical across every column.
before = len(df)
df = df.drop_duplicates()
print(f"Exact duplicate rows removed: {before - len(df)}")

# Keep a stable chronological order.
if "season" in df.columns:
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df = df.sort_values(["season", "player_name"], na_position="last")

df.to_csv(OUTPUT, index=False)

print(f"\nSaved: {OUTPUT}")
print(f"Cleaned rows: {len(df)}")

print("\nAlexander Isak check:")
if "player_name" in df.columns:
    cols = [
        c for c in [
            "season", "player_name", "movement", "club",
            "dealing_club", "fee", "market_value"
        ] if c in df.columns
    ]
    print(
        df[df["player_name"].astype(str).str.contains(
            "Alexander Isak", case=False, na=False
        )][cols].to_string(index=False)
    )
