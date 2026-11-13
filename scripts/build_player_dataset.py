
from pathlib import Path
import pandas as pd
import re
import unicodedata

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize(value):
    """Normalize text for matching player and club names."""
    if pd.isna(value):
        return ""

    value = str(value).strip().casefold()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
    )
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


# Load the cleaned roster
roster_file = DATA_DIR / "roster_clean.csv"
roster = pd.read_csv(roster_file)

roster["name_key"] = roster["player_name"].map(normalize)

# Load the CSV files containing player statistics.
# Exclude team-level statistics and the roster itself.
csv_files = [
    path for path in DATA_DIR.rglob("*.csv")
    if path.name != "roster_clean.csv"
]

player_tables = []

for file in csv_files:
    try:
        df = pd.read_csv(file)
    except Exception as error:
        print(f"Skipping {file.name}: {error}")
        continue

    required = {"Player", "Team"}
    if not required.issubset(df.columns):
        continue

    df["name_key"] = df["Player"].map(normalize)
    df["team_key"] = df["Team"].map(normalize)

    # Keep a single row per player and club in each source.
    df = df.drop_duplicates(subset=["name_key", "team_key"])

    # Prefix metric names with the source filename to avoid collisions.
    metric_columns = [
        col for col in df.columns
        if col not in {
            "Player", "Team", "Country",
            "Rank", "name_key", "team_key"
        }
    ]

    df = df[
        ["name_key", "team_key"] + metric_columns
    ].copy()

    df = df.rename(columns={
        col: f"{file.stem}__{col}"
        for col in metric_columns
    })

    player_tables.append(df)

print(f"Player-statistic files loaded: {len(player_tables)}")

if not player_tables:
    raise ValueError(
        "No player-statistic CSV files found. "
        "Check the data/raw folder."
    )

# Merge each statistics source using normalized player and club names.
combined = player_tables[0]

for table in player_tables[1:]:
    combined = combined.merge(
        table,
        on=["name_key", "team_key"],
        how="outer",
        validate="one_to_one"
    )

# Match the combined statistics to the roster.
roster["team_key"] = roster["teamID"].astype(str)

# Team IDs are not club names, so do not merge team_key from
# the roster. Match by player name first, then inspect ambiguity.
roster_counts = roster.groupby("name_key")["teamID"].nunique()
ambiguous_names = roster_counts[roster_counts > 1].index

print(f"Roster names matching multiple team IDs: {len(ambiguous_names)}")

# Name-only roster lookup is used for diagnostics, not final identity.
roster_lookup = roster.drop_duplicates("name_key").copy()

combined = combined.merge(
    roster_lookup[
        ["name_key", "player_name", "position", "birthDate"]
    ],
    on="name_key",
    how="left",
    validate="many_to_one"
)

combined["roster_match"] = combined["player_name"].notna()

output_file = OUTPUT_DIR / "players_2023_24_combined.csv"
combined.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\n" + "=" * 55)
print("DATASET BUILD SUMMARY")
print("=" * 55)
print(f"Combined player rows: {len(combined)}")
print(f"Columns: {len(combined.columns)}")
print(f"Matched to roster: {combined['roster_match'].sum()}")
print(f"Not matched to roster: {(~combined['roster_match']).sum()}")
print("\nMatched positions:")
print(combined["position"].value_counts(dropna=False))
print(f"\nSaved to: {output_file}")
