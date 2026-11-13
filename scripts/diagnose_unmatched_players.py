
from pathlib import Path
from difflib import get_close_matches
import re
import unicodedata

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"

TRANSFER_PATH = DATA_DIR / "transfer_model_data_2020_2025.csv"
STATS_PATH = DATA_DIR / "historical_player_season_features.csv"


def normalize_name(value):
    if pd.isna(value):
        return ""

    name = str(value).strip()
    name = re.sub(r"_\d+$", "", name)
    name = name.replace("_", " ")

    name = unicodedata.normalize("NFKD", name)
    name = "".join(
        char for char in name
        if not unicodedata.combining(char)
    )

    name = name.casefold()
    name = re.sub(r"[^a-z0-9 ]", " ", name)

    return re.sub(r"\s+", " ", name).strip()


# --------------------------------------------------
# 1. Load datasets
# --------------------------------------------------

transfers = pd.read_csv(TRANSFER_PATH)
stats = pd.read_csv(STATS_PATH)

transfers["match_name"] = transfers["player_name"].map(normalize_name)
stats["match_name"] = stats["name"].map(normalize_name)

stats_names = stats[
    ["match_name", "name", "season"]
].drop_duplicates()

all_names = sorted(
    name
    for name in stats_names["match_name"].unique()
    if name
)


# --------------------------------------------------
# 2. Find the preceding season for each transfer
# --------------------------------------------------

transfers["season_number"] = pd.to_numeric(
    transfers["season"],
    errors="coerce"
)

transfers["previous_stats_season"] = (
    (transfers["season_number"] - 1)
    .astype("Int64")
    .astype("string")
    + "-"
    + transfers["season_number"]
    .astype("Int64")
    .astype("string")
    .str[-2:]
)


# --------------------------------------------------
# 3. Build the set of available player-season pairs
# --------------------------------------------------

stats_pairs = set(
    zip(
        stats["match_name"],
        stats["season"].astype(str)
    )
)


# FIX: Convert the list into a pandas Boolean Series
# before using the ~ operator.

matched_mask = pd.Series(
    [
        (name, season) in stats_pairs
        for name, season in zip(
            transfers["match_name"],
            transfers["previous_stats_season"]
        )
    ],
    index=transfers.index,
    dtype=bool
)

unmatched = transfers[~matched_mask].copy()


# --------------------------------------------------
# 4. Diagnose each unmatched transfer
# --------------------------------------------------

rows = []

for _, transfer in unmatched.iterrows():
    name = transfer["match_name"]

    seasons_found = sorted(
        stats_names.loc[
            stats_names["match_name"] == name,
            "season"
        ].astype(str).unique()
    )

    suggestions = get_close_matches(
        name,
        all_names,
        n=3,
        cutoff=0.78
    )

    suggestion_details = []

    for suggestion in suggestions:
        examples = stats_names[
            stats_names["match_name"] == suggestion
        ]

        original_name = examples.iloc[0]["name"]

        seasons = sorted(
            examples["season"].astype(str).unique()
        )

        suggestion_details.append(
            f"{original_name} [{', '.join(seasons)}]"
        )

    if seasons_found:
        status = "Name found, but not in preceding season"
    else:
        status = "Name absent from all available seasons"

    rows.append({
        "transfer_season": transfer["season"],
        "player_name": transfer["player_name"],
        "normalized_name": name,
        "previous_stats_season": transfer["previous_stats_season"],
        "status": status,
        "seasons_found": ", ".join(seasons_found),
        "possible_name_matches": "; ".join(suggestion_details),
        "buying_club": transfer["club"],
        "transfer_fee": transfer["fee"],
    })


# --------------------------------------------------
# 5. Save and display the report
# --------------------------------------------------

report = pd.DataFrame(rows)

output_path = DATA_DIR / "unmatched_player_diagnosis.csv"
report.to_csv(output_path, index=False)

print("\n" + "=" * 70)
print("UNMATCHED PLAYER DIAGNOSIS")
print("=" * 70)

print(f"\nTotal transfer records: {len(transfers)}")
print(f"Matched to preceding-season stats: {matched_mask.sum()}")
print(f"Unmatched records: {len(unmatched)}")

print("\nDiagnosis by status:")

if not report.empty:
    print(report["status"].value_counts().to_string())

    print("\nFirst 40 unmatched records:")
    print(
        report[
            [
                "transfer_season",
                "player_name",
                "status",
                "seasons_found",
                "possible_name_matches",
            ]
        ].head(40).to_string(index=False)
    )
else:
    print("No unmatched records found.")

print(f"\nFull diagnosis saved to:\n{output_path}")

print(
    "\nIMPORTANT: Similar-name suggestions are candidates only. "
    "They are not confirmed player identity matches."
)

