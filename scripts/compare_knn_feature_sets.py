
from pathlib import Path
import math
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "players_2023_24_combined.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "feature_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SET A: CURRENT FEATURE SETS
# These are the feature lists from our previous audit.
# ============================================================

CURRENT = {
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


# ============================================================
# 3. SET B: BALANCED FEATURE GROUPS
#
# Features within a group describe a related ability.
# Each group receives equal total distance weight.
#
# This is a candidate design to TEST, not a claim that
# these are definitively the best football features.
# ============================================================

BALANCED_GROUPS = {
    "Forward": {
        "Finishing": [
            "player_expected_goals__Expected Goals (xG)",
            "player_total_scoring_attempts__Shots per 90",
            "player_on_target_scoring_attempts__Shot Accuracy (%)",
        ],
        "Creativity": [
            "player_expected_assists__Expected Assists (xA)",
            "player_total_assists_in_attack__Chances Created per 90",
            "player_expected_assists__Actual Assists",
        ],
        "Dribbling": [
            "player_contests_won__Successful Dribbles per 90",
            "player_contests_won__Dribble Success Rate (%)",
        ],
        "Passing": [
            "player_accurate_passes__Pass Success (%)",
        ],
        "Overall": [
            "player_player_ratings__FotMob Rating",
        ],
    },

    "Midfielder": {
        "Creativity": [
            "player_expected_assists__Expected Assists (xA)",
            "player_total_assists_in_attack__Chances Created per 90",
        ],
        "Scoring": [
            "player_expected_goals__Expected Goals (xG)",
            "player_total_scoring_attempts__Shots per 90",
        ],
        "Passing": [
            "player_accurate_passes__Pass Success (%)",
            "player_accurate_long_balls__Successful Long Balls (%)",
        ],
        "Carrying": [
            "player_contests_won__Successful Dribbles per 90",
        ],
        "Defending": [
            "player_tackles_won__Tackles per 90",
            "player_interceptions__Interceptions per 90",
            "player_outfielder_blocks__Blocks per 90",
        ],
        "Overall": [
            "player_player_ratings__FotMob Rating",
        ],
    },

    "Defender": {
        "Defending": [
            "player_tackles_won__Tackles per 90",
            "player_interceptions__Interceptions per 90",
            "player_outfielder_blocks__Blocks per 90",
            "player_effective_clearances__Clearances per 90",
        ],
        "Defensive Discipline": [
            "player_tackles_won__Tackle Success Rate (%)",
            "player_fouls_committed__Fouls Committed per 90",
        ],
        "Passing": [
            "player_accurate_passes__Pass Success (%)",
            "player_accurate_long_balls__Successful Long Balls (%)",
        ],
        "Carrying": [
            "player_contests_won__Successful Dribbles per 90",
        ],
        "Attacking Contribution": [
            "player_expected_assists__Expected Assists (xA)",
            "player_expected_goals__Expected Goals (xG)",
        ],
        "Overall": [
            "player_player_ratings__FotMob Rating",
        ],
    },

    "Goalkeeper": {
        "Shot Stopping": [
            "player_saves_made__Saves per 90",
            "player_goals_conceded__Goals Conceded per 90",
        ],
        "Clean Sheets": [
            "player_clean_sheets__Clean Sheets",
        ],
        "Distribution": [
            "player_accurate_passes__Pass Success (%)",
            "player_accurate_passes__Accurate Passes per 90",
        ],
        "Overall": [
            "player_player_ratings__FotMob Rating",
        ],
    },
}


# ============================================================
# 4. PLAYER EXAMPLES
# ============================================================

TEST_PLAYERS = {
    "Midfielder": [
        "Bruno Fernandes",
        "Kevin De Bruyne",
        "Moisés Caicedo",
        "Casemiro",
    ],
    "Forward": ["Erling Haaland"],
    "Defender": ["William Saliba"],
    "Goalkeeper": ["Alisson Becker"],
}

MIN_COVERAGE = 0.35
MIN_UNIQUE_VALUES = 5
NEIGHBORS_TO_SHOW = 5


# ============================================================
# 5. HELPERS
# ============================================================

def normalize_name(value):
    if pd.isna(value):
        return ""

    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(
        char for char in value if not unicodedata.combining(char)
    )
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)

    return " ".join(value.split())


def prepare_position(data, position):
    return data[data["position_group"] == position].copy()


def feature_is_usable(series):
    numeric = pd.to_numeric(series, errors="coerce")
    coverage = numeric.notna().mean()

    return (
        coverage >= MIN_COVERAGE
        and numeric.nunique(dropna=True) >= MIN_UNIQUE_VALUES
        and numeric.std(skipna=True) > 0
    )


def filter_features(group_data, proposed_features):
    usable = []

    for feature in proposed_features:
        if feature not in group_data.columns:
            continue

        if feature_is_usable(group_data[feature]):
            usable.append(feature)

    return usable


def eligible_rows(group_data, features):
    if not features:
        return group_data.iloc[0:0].copy(), 0

    values = group_data[features].apply(
        pd.to_numeric, errors="coerce"
    )

    # A player must have genuine observed values in at least
    # half the usable features. Median-imputed values do NOT
    # count as genuine observations for eligibility.
    minimum_real_values = max(2, math.ceil(len(features) / 2))

    mask = values.notna().sum(axis=1) >= minimum_real_values

    result = group_data.loc[mask].copy()
    result[features] = values.loc[mask]

    return result, minimum_real_values


def fit_distances(group_data, features, feature_groups=None):
    eligible, minimum_real_values = eligible_rows(
        group_data, features
    )

    if len(eligible) < 2:
        return None

    X = eligible[features].apply(
        pd.to_numeric, errors="coerce"
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    # Set A: ordinary standardized Euclidean distance.
    # Set B: equal total weighting per feature group.
    if feature_groups is not None:
        feature_to_group = {}

        for group_name, group_features in feature_groups.items():
            for feature in group_features:
                if feature in features:
                    feature_to_group[feature] = group_name

        group_sizes = {}

        for feature in features:
            group_name = feature_to_group[feature]
            group_sizes[group_name] = (
                group_sizes.get(group_name, 0) + 1
            )

        for index, feature in enumerate(features):
            group_name = feature_to_group[feature]
            X_scaled[:, index] /= math.sqrt(
                group_sizes[group_name]
            )

    distances = pairwise_distances(
        X_scaled, metric="euclidean"
    )

    return {
        "players": eligible.reset_index(drop=True),
        "features": features,
        "distances": distances,
        "minimum_real_values": minimum_real_values,
    }


def find_player_index(model, search_name):
    players = model["players"]
    query = normalize_name(search_name)

    names = players["player_name"].fillna("").map(normalize_name)

    exact = players.index[names == query].tolist()

    if exact:
        return exact[0], None

    partial = players.index[
        names.map(
            lambda name: query in name or name in query
            if name and query else False
        )
    ].tolist()

    if len(partial) == 1:
        return partial[0], None

    if len(partial) > 1:
        matches = players.loc[
            partial, ["player_name", "team_key"]
        ].drop_duplicates()

        return None, matches.to_string(index=False)

    return None, "No eligible player match found."


def show_neighbors(model, search_name, set_name, position):
    result = find_player_index(model, search_name)

    print(f"\n  {search_name} | {set_name}")

    if result[0] is None:
        print(f"    {result[1]}")
        return

    player_index = result[0]
    players = model["players"]
    distances = model["distances"][player_index].copy()

    # Remove the player's distance to themselves.
    distances[player_index] = np.inf

    order = np.argsort(distances)
    shown = 0

    for neighbor_index in order:
        if not np.isfinite(distances[neighbor_index]):
            continue

        neighbor = players.iloc[neighbor_index]
        name = neighbor.get("player_name", "Unknown")
        team = neighbor.get("team_key", "")
        distance = distances[neighbor_index]

        print(
            f"    {shown + 1}. {name} | {team} "
            f"| distance = {distance:.3f}"
        )

        shown += 1

        if shown >= NEIGHBORS_TO_SHOW:
            break

    if shown == 0:
        print("    No other eligible players available.")


# ============================================================
# 6. LOAD AND PREPARE DATA
# ============================================================

print("=" * 72)
print("TRANSFERIQ: KNN FEATURE-SET COMPARISON")
print("=" * 72)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

required_columns = {"player_name", "position"}

missing_required = required_columns - set(df.columns)

if missing_required:
    raise ValueError(
        f"Dataset is missing required columns: {missing_required}"
    )

df["position_group"] = (
    df["position"]
    .astype("string")
    .str.strip()
    .str.lower()
    .map({
        "forward": "Forward",
        "attacker": "Forward",
        "midfielder": "Midfielder",
        "midfield": "Midfielder",
        "defender": "Defender",
        "defence": "Defender",
        "defense": "Defender",
        "goalkeeper": "Goalkeeper",
        "keeper": "Goalkeeper",
    })
)

# Keep the dataset rows as supplied. Do not automatically merge
# or fuzzy-match player identities in this evaluation.
print(f"\nDataset rows: {len(df)}")
print("\nPosition counts:")
print(df["position_group"].value_counts(dropna=False).to_string())


# ============================================================
# 7. COMPARE ELIGIBILITY AND NEIGHBORS
# ============================================================

summary_rows = []
neighbor_rows = []

for position in ["Forward", "Midfielder", "Defender", "Goalkeeper"]:
    group_data = prepare_position(df, position)

    if group_data.empty:
        print(f"\nWARNING: No rows for {position}.")
        continue

    print("\n" + "=" * 72)
    print(f"POSITION: {position} | Rows: {len(group_data)}")
    print("=" * 72)

    current_features = filter_features(
        group_data, CURRENT[position]
    )

    proposed_groups = BALANCED_GROUPS[position]

    usable_balanced_groups = {}
    for group_name, features in proposed_groups.items():
        usable = filter_features(group_data, features)

        if usable:
            usable_balanced_groups[group_name] = usable

    balanced_features = [
        feature
        for features in usable_balanced_groups.values()
        for feature in features
    ]

    # Remove duplicate columns while preserving their order.
    balanced_features = list(dict.fromkeys(balanced_features))

    feature_sets = {
        "A_Current": (current_features, None),
        "B_Balanced": (
            balanced_features,
            usable_balanced_groups,
        ),
    }

    models = {}

    for set_name, (features, feature_groups) in feature_sets.items():
        eligible, minimum_real_values = eligible_rows(
            group_data, features
        )

        coverage_pct = (
            len(eligible) / len(group_data) * 100
            if len(group_data) else 0
        )

        print(f"\n{set_name}")
        print(f"  Usable features: {len(features)}")
        print(f"  Minimum real features per player: {minimum_real_values}")
        print(
            f"  Eligible players: {len(eligible)}/{len(group_data)} "
            f"({coverage_pct:.1f}%)"
        )

        print("  Features:")
        for feature in features:
            print(f"    - {feature}")

        summary_rows.append({
            "position": position,
            "feature_set": set_name,
            "usable_features": len(features),
            "minimum_real_features": minimum_real_values,
            "eligible_players": len(eligible),
            "total_players": len(group_data),
            "eligible_percent": round(coverage_pct, 2),
        })

        model = fit_distances(
            group_data,
            features,
            feature_groups=feature_groups,
        )

        if model is not None:
            models[set_name] = model
        else:
            print("  Not enough eligible players to calculate distances.")

    for player_name in TEST_PLAYERS.get(position, []):
        print(f"\n--- Test player: {player_name} ---")

        for set_name in ["A_Current", "B_Balanced"]:
            if set_name not in models:
                print(f"\n  {set_name}: model unavailable.")
                continue

            model = models[set_name]
            result = find_player_index(model, player_name)

            show_neighbors(
                model, player_name, set_name, position
            )

            if result[0] is None:
                continue

            player_index = result[0]
            distances = model["distances"][player_index].copy()
            distances[player_index] = np.inf

            order = np.argsort(distances)[:NEIGHBORS_TO_SHOW]

            for rank, neighbor_index in enumerate(order, start=1):
                if not np.isfinite(distances[neighbor_index]):
                    continue

                neighbor = model["players"].iloc[neighbor_index]

                neighbor_rows.append({
                    "position": position,
                    "test_player": player_name,
                    "feature_set": set_name,
                    "rank": rank,
                    "neighbor": neighbor.get("player_name", ""),
                    "neighbor_team": neighbor.get("team_key", ""),
                    "distance": round(
                        float(distances[neighbor_index]), 5
                    ),
                })


# ============================================================
# 8. SAVE REPORTS
# ============================================================

summary_df = pd.DataFrame(summary_rows)
neighbors_df = pd.DataFrame(neighbor_rows)

summary_path = OUTPUT_DIR / "feature_set_comparison_summary.csv"
neighbors_path = OUTPUT_DIR / "feature_set_neighbor_comparison.csv"

summary_df.to_csv(summary_path, index=False)
neighbors_df.to_csv(neighbors_path, index=False)

print("\n" + "=" * 72)
print("COMPARISON COMPLETE")
print("=" * 72)

if not summary_df.empty:
    print("\nEligibility summary:")
    print(summary_df.to_string(index=False))

print(f"\nSaved summary: {summary_path}")
print(f"Saved neighbors: {neighbors_path}")

print(
    "\nImportant: closer neighbors are not automatically better. "
    "We must inspect whether the results make football sense."
)
