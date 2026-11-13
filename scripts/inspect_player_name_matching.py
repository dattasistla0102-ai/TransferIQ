
from pathlib import Path
import re
import unicodedata

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"

TRANSFER_PATH = DATA_DIR / "transfer_model_data_2020_2025.csv"
STATS_PATH = DATA_DIR / "historical_player_season_features.csv"


def normalize_name(value):
    """Normalize names for candidate matching, not identity proof."""
    if pd.isna(value):
        return ""

    name = str(value).strip()

    # Remove the trailing FPL identifier, e.g. Hector_Bellerin_2.
    name = re.sub(r"_\d+$", "", name)

    # Convert underscores to spaces.
    name = name.replace("_", " ")

    # Remove accents for comparison.
    name = unicodedata.normalize("NFKD", name)
    name = "".join(
        character for character in name
        if not unicodedata.combining(character)
    )

    # Normalize case and punctuation.
    name = name.casefold()
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


print("=" * 65)
print("TRANSFERIQ: PLAYER MATCHING AUDIT")
print("=" * 65)

transfers = pd.read_csv(TRANSFER_PATH)
stats = pd.read_csv(STATS_PATH)

required_transfer = {"player_name", "season"}
required_stats = {"name", "season"}

if not required_transfer.issubset(transfers.columns):
    raise ValueError(
        f"Transfer data missing: {required_transfer - set(transfers.columns)}"
    )

if not required_stats.issubset(stats.columns):
    raise ValueError(
        f"Stats data missing: {required_stats - set(stats.columns)}"
    )

transfers["match_name"] = transfers["player_name"].map(normalize_name)
stats["match_name"] = stats["name"].map(normalize_name)

# Transfer season 2020 is compared with performance from 2019-20,
# transfer season 2021 with 2020-21, and so on.
transfers["previous_stats_season"] = (
    (pd.to_numeric(transfers["season"], errors="coerce") - 1).astype("Int64")
    .astype("string")
    + "-"
    + pd.to_numeric(transfers["season"], errors="coerce")
    .astype("Int64")
    .astype("string").str[-2:]
)

stats["stats_season"] = stats["season"].astype(str)

# Name-season pairs available in the historical statistics.
stats_pairs = set(
    zip(stats["match_name"], stats["stats_season"])
)

transfers["candidate_match"] = [
    (name, season) in stats_pairs
    for name, season in zip(
        transfers["match_name"],
        transfers["previous_stats_season"]
    )
]

matched = transfers[transfers["candidate_match"]]
unmatched = transfers[~transfers["candidate_match"]]

print(f"\nTransfer records: {len(transfers)}")
print(f"Candidate name + previous-season matches: {len(matched)}")
print(f"Unmatched transfer records: {len(unmatched)}")
print(
    "Candidate match rate: "
    f"{len(matched) / len(transfers) * 100:.1f}%"
    if len(transfers) else "No transfer records found."
)

print("\nCandidate matches by transfer season:")
print(
    transfers.groupby("season", dropna=False)["candidate_match"]
    .agg(["sum", "count"])
    .to_string()
)

print("\nSample candidate matches:")
print(
    matched[
        [
            "player_name",
            "season",
            "previous_stats_season",
            "player_id",
        ]
    ].head(20).to_string(index=False)
)

print("\nSample unmatched records:")
print(
    unmatched[
        ["player_name", "season", "previous_stats_season"]
    ].head(25).to_string(index=False)
)

# Save an audit report, not a merged training dataset.
report_path = DATA_DIR / "player_matching_audit.csv"

transfers[
    [
        "player_name",
        "season",
        "previous_stats_season",
        "player_id",
        "candidate_match",
    ]
].to_csv(report_path, index=False)

print("\nSaved matching audit to:")
print(report_path)

print(
    "\nReminder: matching names are candidates, not proof of identity. "
    "Review the match rate and unmatched examples before joining data."
)
