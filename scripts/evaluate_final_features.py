
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


# =========================================================
# 1. PATHS AND SETTINGS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "players_2023_24_combined.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "feature_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_FEATURE_COVERAGE = 0.35
MIN_REAL_FEATURES = 0.50
HIGH_CORRELATION = 0.85
MIN_PAIR_SAMPLES = 30


# =========================================================
# 2. POSITION-SPECIFIC FEATURE SHORTLISTS
# =========================================================

FEATURES = {
    "Forward": [
        "player_expected_goals__Expected Goals (xG)",
        "player_total_scoring_attempts__Shot Conversion Rate (%)",
        "player_total_scoring_attempts__Shots per 90",
        "player_on_target_scoring_attempts__Shot Accuracy (%)",
        "player_expected_goals_on_target__Expected Goals on Target (xGOT)",
        "player_expected_goals_per_90__Goals per 90",
        "player_expected_assists__Expected Assists (xA)",
        "player_total_assists_in_attack__Chances Created per 90",
        "player_big_chances_created__Big Chances Created",
        "player_expected_assists__Actual Assists",
        "player_contests_won__Successful Dribbles per 90",
        "player_contests_won__Dribble Success Rate (%)",
        "player_accurate_passes__Pass Success (%)",
        "player_possessions_won_attacking_third__Possessions Won Midfield per 90",
        "player_player_ratings__FotMob Rating",
    ],

    "Midfielder": [
        "player_expected_assists__Expected Assists (xA)",
        "player_total_assists_in_attack__Chances Created per 90",
        "player_big_chances_created__Big Chances Created",
        "player_expected_goals__Expected Goals (xG)",
        "player_total_scoring_attempts__Shots per 90",
        "player_total_scoring_attempts__Shot Conversion Rate (%)",
        "player_accurate_passes__Pass Success (%)",
        "player_accurate_long_balls__Accurate Long Balls per 90",
        "player_accurate_long_balls__Successful Long Balls (%)",
        "player_contests_won__Successful Dribbles per 90",
        "player_contests_won__Dribble Success Rate (%)",
        "player_tackles_won__Tackles per 90",
        "player_interceptions__Interceptions per 90",
        "player_outfielder_blocks__Blocks per 90",
        "player_player_ratings__FotMob Rating",
    ],

    "Defender": [
        "player_tackles_won__Tackles per 90",
        "player_tackles_won__Tackle Success Rate (%)",
        "player_interceptions__Interceptions per 90",
        "player_outfielder_blocks__Blocks per 90",
        "player_effective_clearances__Clearances per 90",
        "player_fouls_committed__Fouls Committed per 90",
        "player_accurate_passes__Pass Success (%)",
        "player_accurate_long_balls__Accurate Long Balls per 90",
        "player_accurate_long_balls__Successful Long Balls (%)",
        "player_contests_won__Successful Dribbles per 90",
        "player_expected_assists__Expected Assists (xA)",
        "player_total_assists_in_attack__Chances Created per 90",
        "player_expected_goals__Expected Goals (xG)",
        "player_effective_clearances__Total Clearances",
        "player_player_ratings__FotMob Rating",
    ],

    "Goalkeeper": [
        "player_saves_made__Saves per 90",
        "player_saves_made__Total Saves",
        "player_goals_conceded__Goals Conceded per 90",
        "player_goals_conceded__Total Goals Conceded",
        "player_clean_sheets__Clean Sheets",
        "player_clean_sheets__Goals Conceded",
        "player_accurate_passes__Pass Success (%)",
        "player_accurate_passes__Accurate Passes per 90",
        "player_player_ratings__FotMob Rating",
    ],
}


# =========================================================
# 3. LOAD AND CHECK DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

required_columns = ["player_name", "position"]
missing_required = [c for c in required_columns if c not in df.columns]

if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

position_map = {
    "forward": "Forward",
    "midfielder": "Midfielder",
    "defender": "Defender",
    "goalkeeper": "Goalkeeper",
}

df["position_group"] = (
    df["position"]
    .astype("string")
    .str.strip()
    .str.lower()
    .map(position_map)
)

print("=" * 75)
print("TRANSFERIQ - FINAL FEATURE AND ELIGIBILITY AUDIT")
print("=" * 75)

print(f"Dataset rows: {len(df)}")
print(f"Dataset columns: {len(df.columns)}")

print("\nPlayers by position:")
print(df["position_group"].value_counts(dropna=False).to_string())


# =========================================================
# 4. CHECK FEATURE AVAILABILITY
# =========================================================

feature_rows = []
eligible_players = []
correlation_rows = []
position_summaries = []

for position, proposed_features in FEATURES.items():

    group = df[df["position_group"] == position].copy()
    total_players = len(group)

    print("\n" + "=" * 75)
    print(position.upper())
    print("=" * 75)

    available_features = [
        feature for feature in proposed_features
        if feature in df.columns
    ]

    missing_columns = [
        feature for feature in proposed_features
        if feature not in df.columns
    ]

    if missing_columns:
        print("\nWARNING - feature columns not found:")
        for feature in missing_columns:
            print(f"  {feature}")

    if not available_features:
        print("No proposed features were found.")
        continue

    # Numeric conversion without modifying the original dataset.
    X = group[available_features].apply(
        pd.to_numeric, errors="coerce"
    )

    # Initial feature screening.
    usable_features = []

    for feature in available_features:
        values = X[feature]
        coverage = values.notna().mean() if total_players else 0
        unique_values = values.nunique(dropna=True)

        std = values.std()
        variable = pd.notna(std) and std > 0

        passes = (
            coverage >= MIN_FEATURE_COVERAGE
            and unique_values >= 5
            and variable
        )

        if passes:
            usable_features.append(feature)

        feature_rows.append({
            "position": position,
            "feature": feature,
            "players_in_position": total_players,
            "non_missing": int(values.notna().sum()),
            "coverage_percent": round(coverage * 100, 1),
            "unique_values": int(unique_values),
            "passes_screen": passes,
        })

    print(f"Players in position: {total_players}")
    print(f"Proposed features: {len(proposed_features)}")
    print(f"Existing columns: {len(available_features)}")
    print(f"Features passing coverage/variation screen: {len(usable_features)}")

    # Eligibility is assessed against the usable selected feature set.
    # Every eligible player must have at least half observed.
    if usable_features:
        X_usable = X[usable_features]
        required_real = max(
            2,
            int(np.ceil(len(usable_features) * MIN_REAL_FEATURES))
        )

        real_counts = X_usable.notna().sum(axis=1)
        eligible_mask = real_counts >= required_real

        print(f"Features used for eligibility: {len(usable_features)}")
        print(f"Real features required per player: {required_real}")
        print(f"Eligible players: {int(eligible_mask.sum())}/{total_players}")

        for idx, row in group.iterrows():
            count = int(real_counts.loc[idx])
            eligible = bool(eligible_mask.loc[idx])

            eligible_players.append({
                "position": position,
                "player_name": row.get("player_name"),
                "team": row.get("team", row.get("team_key", "")),
                "real_features": count,
                "features_required": required_real,
                "eligible": eligible,
                "missing_selected_features": "; ".join(
                    feature for feature in usable_features
                    if pd.isna(X_usable.loc[idx, feature])
                ),
            })

        # Correlation audit among usable features.
        for feature_a, feature_b in combinations(usable_features, 2):
            pair = X_usable[[feature_a, feature_b]].dropna()

            if len(pair) < MIN_PAIR_SAMPLES:
                continue

            corr = pair[feature_a].corr(pair[feature_b])

            if pd.notna(corr) and abs(corr) >= HIGH_CORRELATION:
                correlation_rows.append({
                    "position": position,
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "correlation": round(float(corr), 3),
                    "players_compared": len(pair),
                })

    else:
        required_real = 0
        eligible_mask = pd.Series(False, index=group.index)
        print("No features passed the initial screen.")

    eligible_count = int(eligible_mask.sum())

    position_summaries.append({
        "position": position,
        "players_in_position": total_players,
        "proposed_features": len(proposed_features),
        "usable_features": len(usable_features),
        "real_features_required": required_real,
        "eligible_players": eligible_count,
        "eligibility_percent": (
            round(eligible_count / total_players * 100, 1)
            if total_players else 0
        ),
    })


# =========================================================
# 5. SAVE REPORTS
# =========================================================

coverage_df = pd.DataFrame(feature_rows)
eligibility_df = pd.DataFrame(eligible_players)
correlation_df = pd.DataFrame(correlation_rows)
summary_df = pd.DataFrame(position_summaries)

coverage_path = OUTPUT_DIR / "final_feature_coverage.csv"
eligibility_path = OUTPUT_DIR / "player_eligibility.csv"
correlation_path = OUTPUT_DIR / "final_feature_correlations.csv"
summary_path = OUTPUT_DIR / "final_feature_summary.csv"

coverage_df.to_csv(coverage_path, index=False)
eligibility_df.to_csv(eligibility_path, index=False)
correlation_df.to_csv(correlation_path, index=False)
summary_df.to_csv(summary_path, index=False)

print("\n" + "=" * 75)
print("POSITION SUMMARY")
print("=" * 75)
print(summary_df.to_string(index=False))

print("\n" + "=" * 75)
print("HIGH-CORRELATION FEATURE PAIRS")
print("=" * 75)

if correlation_df.empty:
    print("No pairs crossed the correlation threshold.")
else:
    print(correlation_df.to_string(index=False))

print("\nReports saved:")
print(coverage_path)
print(eligibility_path)
print(correlation_path)
print(summary_path)

print("\nAudit complete. Existing KNN model was not modified.")
