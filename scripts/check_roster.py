
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

roster = pd.read_csv(DATA_DIR / "roster_clean.csv")
combined = pd.read_csv(
    PROCESSED_DIR / "players_2023_24_combined.csv"
)

unmatched = combined.loc[
    ~combined["roster_match"].astype(str).str.lower().eq("true")
].copy()

print("=" * 60)
print("ROSTER COVERAGE CHECK")
print("=" * 60)

print("\nRoster columns:")
print(roster.columns.tolist())

print("\nCombined dataset columns related to identity:")
print([
    col for col in combined.columns
    if col in [
        "Player", "Team", "name_key", "team_key",
        "player_name", "position", "teamID", "roster_match"
    ]
])

print("\nTotal roster rows:", len(roster))
print("Unique roster player names:", roster["player_name"].nunique())
print("Unmatched statistics rows:", len(unmatched))

print("\nFirst 25 unmatched names:")
print(
    unmatched[["name_key", "team_key"]]
    .head(25)
    .to_string(index=False)
)

print("\nFirst 25 roster records:")
print(
    roster.head(25).to_string(index=False)
)

print("\nCheck complete.")
