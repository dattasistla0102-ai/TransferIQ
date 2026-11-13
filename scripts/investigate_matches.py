
from pathlib import Path
import pandas as pd
import re
import unicodedata

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


def normalize_name(value):
    """Normalize names consistently for comparison."""
    if pd.isna(value):
        return ""

    value = str(value).casefold().strip()

    # Convert accented characters to their basic forms.
    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
    )

    # Keep letters and numbers; standardize punctuation and spaces.
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


combined = pd.read_csv(
    PROCESSED_DIR / "players_2023_24_combined.csv"
)

roster = pd.read_csv(
    DATA_DIR / "roster_clean.csv"
)

# Recalculate normalized names using the same method.
combined["check_name"] = combined["name_key"].apply(normalize_name)
roster["check_name"] = roster["player_name"].apply(normalize_name)

roster_names = set(
    roster.loc[roster["check_name"] != "", "check_name"]
)

combined["name_found_in_roster"] = (
    combined["check_name"].isin(roster_names)
)

unmatched = combined.loc[
    ~combined["roster_match"].astype(str).str.lower().eq("true")
].copy()

unmatched["name_found_in_roster"] = (
    unmatched["check_name"].isin(roster_names)
)

print("=" * 60)
print("CORRECTED NAME-MATCH INVESTIGATION")
print("=" * 60)

print(f"\nTotal combined player-team rows: {len(combined)}")
print(f"Rows originally marked unmatched: {len(unmatched)}")

print("\nName matches among the originally unmatched rows:")
print(
    unmatched["name_found_in_roster"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nFirst 40 originally unmatched records:")
print(
    unmatched[
        ["name_key", "team_key", "name_found_in_roster"]
    ]
    .head(40)
    .to_string(index=False)
)

print("\nExamples of roster names for manual comparison:")
print(
    roster[["player_name", "position"]]
    .head(25)
    .to_string(index=False)
)

print("\nInvestigation complete.")

