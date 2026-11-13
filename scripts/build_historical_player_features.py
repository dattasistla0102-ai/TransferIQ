
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_DIR / "data" / "raw" / "historical_player_stats"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = [
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]

# Columns identifying a player or a gameweek, rather than
# performance totals to aggregate.
IDENTIFIER_COLUMNS = {
    "name",
    "element",
    "GW",
    "round",
    "fixture",
    "kickoff_time",
    "opponent_team",
    "was_home",
    "team_a_score",
    "team_h_score",
    "value",  # FPL price: NOT transfer market value
    "selected",
    "transfers_balance",
    "transfers_in",
    "transfers_out",
}

# These fields are counts or performance totals that can be summed
# across gameweeks.
SUM_COLUMNS = {
    "assists",
    "bonus",
    "bps",
    "clean_sheets",
    "goals_conceded",
    "goals_scored",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "red_cards",
    "saves",
    "total_points",
    "yellow_cards",
    "minutes",
}

all_seasons = []

print("=" * 65)
print("TRANSFERIQ: BUILD HISTORICAL PLAYER FEATURES")
print("=" * 65)

for season in SEASONS:
    filename = f"{season}_all_players_gw.csv"
    input_path = INPUT_DIR / filename

    if not input_path.exists():
        print(f"\nSkipping missing file: {filename}")
        continue

    print(f"\nProcessing {season}...")

    df = pd.read_csv(input_path)

    required = {"name", "element", "GW"}
    missing = required - set(df.columns)

    if missing:
        print(f"Skipping {season}; missing columns: {sorted(missing)}")
        continue

    # Remove rows without a player identifier.
    df = df.dropna(subset=["name", "element"]).copy()

    # Convert available performance columns to numeric.
    for column in df.columns:
        if column not in {"name", "kickoff_time"}:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Aggregate only known additive performance totals.
    available_sum_columns = [
        column for column in SUM_COLUMNS
        if column in df.columns
    ]

    # Preserve the number of recorded gameweeks for quality checks.
    grouped = df.groupby(
        ["element", "name"],
        dropna=False
    )

    features = grouped[available_sum_columns].sum(min_count=1)
    features["gameweeks_recorded"] = grouped["GW"].nunique()

    # Keep season and player identifiers.
    features = features.reset_index()
    features.insert(0, "season", season)

    # Deliberately exclude FPL price and cumulative/rate fields
    # until their aggregation meaning has been verified.
    all_seasons.append(features)

    print(f"Input gameweek rows: {len(df)}")
    print(f"Unique player-season records: {len(features)}")
    print(f"Aggregated metrics: {available_sum_columns}")

if not all_seasons:
    raise RuntimeError(
        "No seasons were processed. Check the input folder and CSV files."
    )

result = pd.concat(all_seasons, ignore_index=True)

output_path = OUTPUT_DIR / "historical_player_season_features.csv"
result.to_csv(output_path, index=False)

print("\n" + "=" * 65)
print("BUILD COMPLETE")
print("=" * 65)
print(f"Player-season rows: {len(result)}")
print(f"Columns: {len(result.columns)}")
print(f"Saved to: {output_path}")

print("\nRows per season:")
print(result.groupby("season").size().to_string())

print("\nSample records:")
print(result.head(5).to_string(index=False))

print(
    "\nReminder: FPL prices were excluded. "
    "These are candidate performance features, not market values."
)
