from pathlib import Path
import math
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "players_2023_24_combined.csv"
)

FC24_DATA_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "ea_fc24_premier_league_players.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "scout_players_2023_24.csv"

TOP_N = 7
MIN_FEATURES = 2
MIN_FEATURE_COVERAGE = 0.35
MIN_UNIQUE_VALUES = 5

# ============================================================
# 2. POSITION-SPECIFIC FEATURES AND ABILITY GROUPS
# ============================================================
POSITION_FEATURE_GROUPS = {
    "Forward": {
        "Finishing": [
            "player_expected_goals__Expected Goals (xG)",
            "player_total_scoring_attempts__Shots per 90",
            "player_total_scoring_attempts__Shot Conversion Rate (%)",
            "player_on_target_scoring_attempts__Shot Accuracy (%)",
            "player_expected_goals_per_90__Goals per 90",
        ],
        "Creativity": [
            "player_expected_assists__Expected Assists (xA)",
            "player_big_chances_created__Big Chances Created",
        ],
        "Dribbling": [
            "player_contests_won__Successful Dribbles per 90",
            "player_contests_won__Dribble Success Rate (%)",
        ],
        "Passing": [
            "player_accurate_passes__Pass Success (%)"
        ],
        "Pressing": [
            "player_possessions_won_attacking_third__Possessions Won Midfield per 90",
        ],
        "Overall": [
            "player_player_ratings__FotMob Rating"
        ],
    },
    "Midfielder": {
        "Creativity": [
            "player_expected_assists__Expected Assists (xA)"
        ],
        "Scoring": [
            "player_expected_goals__Expected Goals (xG)",
            "player_total_scoring_attempts__Shots per 90",
            "player_total_scoring_attempts__Shot Conversion Rate (%)",
        ],
        "Passing": [
            "player_accurate_passes__Pass Success (%)",
            "player_accurate_long_balls__Accurate Long Balls per 90",
            "player_accurate_long_balls__Successful Long Balls (%)",
        ],
        "Carrying": [
            "player_contests_won__Successful Dribbles per 90",
            "player_contests_won__Dribble Success Rate (%)",
        ],
        "Defending": [
            "player_tackles_won__Tackles per 90",
            "player_tackles_won__Tackle Success Rate (%)",
            "player_interceptions__Interceptions per 90",
            "player_outfielder_blocks__Blocks per 90",
        ],
        "Overall": [
            "player_player_ratings__FotMob Rating"
        ],
    },
    "Defender": {
        "Defending": [
            "player_tackles_won__Tackles per 90",
            "player_tackles_won__Tackle Success Rate (%)",
            "player_interceptions__Interceptions per 90",
            "player_outfielder_blocks__Blocks per 90",
            "player_effective_clearances__Clearances per 90",
        ],
        "Discipline": [
            "player_fouls_committed__Fouls Committed per 90"
        ],
        "Passing and Distribution": [
            "player_accurate_passes__Pass Success (%)",
            "player_accurate_long_balls__Accurate Long Balls per 90",
            "player_accurate_long_balls__Successful Long Balls (%)",
        ],
        "Carrying": [
            "player_contests_won__Successful Dribbles per 90"
        ],
        "Attacking Contribution": [
            "player_expected_assists__Expected Assists (xA)",
            "player_expected_goals__Expected Goals (xG)",
        ],
        "Overall": [
            "player_player_ratings__FotMob Rating"
        ],
    },
    "Goalkeeper": {
        "Shot Stopping": [
            "player_saves_made__Saves per 90",
            "player_goals_conceded__Goals Conceded per 90",
        ],
        "Clean Sheets": [
            "player_clean_sheets__Clean Sheets"
        ],
        "Distribution": [
            "player_accurate_passes__Pass Success (%)",
            "player_accurate_passes__Accurate Passes per 90",
        ],
        "Overall": [
            "player_player_ratings__FotMob Rating"
        ],
    },
}

COMPACT_POSITION_FEATURE_GROUPS = {
    "Forward": {
        "Finishing": [
            "player_expected_goals__Expected Goals (xG)"
        ],
        "Creativity": [
            "player_expected_assists__Expected Assists (xA)"
        ],
        "Dribbling": [
            "player_contests_won__Successful Dribbles per 90"
        ],
        "Passing": [
            "player_accurate_passes__Pass Success (%)"
        ],
        "Overall": [
            "player_player_ratings__FotMob Rating"
        ],
    },
    "Midfielder": {
        "Creativity": [
            "player_expected_assists__Expected Assists (xA)"
        ],
        "Scoring": [
            "player_expected_goals__Expected Goals (xG)"
        ],
        "Passing": [
            "player_accurate_passes__Pass Success (%)"
        ],
        "Carrying": [
            "player_contests_won__Successful Dribbles per 90"
        ],
        "Defending": [
            "player_tackles_won__Tackles per 90"
        ],
        "Overall": [
            "player_player_ratings__FotMob Rating"
        ],
    },
    "Defender": {
        "Defending": [
            "player_tackles_won__Tackles per 90"
        ],
        "Passing": [
            "player_accurate_passes__Pass Success (%)"
        ],
        "Carrying": [
            "player_contests_won__Successful Dribbles per 90"
        ],
        "Attacking Contribution": [
            "player_expected_assists__Expected Assists (xA)"
        ],
        "Overall": [
            "player_player_ratings__FotMob Rating"
        ],
    },
    "Goalkeeper": {
        "Shot Stopping": [
            "player_saves_made__Saves per 90"
        ],
        "Clean Sheets": [
            "player_clean_sheets__Clean Sheets"
        ],
        "Distribution": [
            "player_accurate_passes__Pass Success (%)"
        ],
        "Overall": [
            "player_player_ratings__FotMob Rating"
        ],
    },
}

COMPACT_MIN_REAL_FEATURES = 2
COMPACT_MIN_ABILITY_GROUPS = 2

GENERAL_FEATURE_GROUPS = {
    "Scoring": [
        "player_expected_goals__Expected Goals (xG)",
        "player_total_scoring_attempts__Shots per 90",
    ],
    "Creativity": [
        "player_expected_assists__Expected Assists (xA)"
    ],
    "Passing": [
        "player_accurate_passes__Pass Success (%)"
    ],
    "Carrying": [
        "player_contests_won__Successful Dribbles per 90"
    ],
    "Defending": [
        "player_tackles_won__Tackles per 90",
        "player_interceptions__Interceptions per 90",
        "player_effective_clearances__Clearances per 90",
    ],
    "Overall": [
        "player_player_ratings__FotMob Rating"
    ],
}

# ============================================================
# 3. LOAD AND CLEAN KNN DATA
# ============================================================
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Expected data/processed/players_2023_24_combined.csv."
    )

df = pd.read_csv(DATA_PATH)

required_columns = ["name_key", "team_key", "player_name"]
missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    raise ValueError(
        f"Required columns missing from dataset: {missing_columns}"
    )

if "position" not in df.columns:
    df["position"] = pd.NA

df = df.dropna(subset=["name_key", "team_key"]).copy()

for column in ["player_name", "name_key", "team_key"]:
    df[column] = df[column].fillna("").astype(str).str.strip()

df = df[
    (df["player_name"] != "")
    & (df["name_key"] != "")
    & (df["team_key"] != "")
].copy()

df["position"] = df["position"].astype("string").str.strip()
df.loc[df["position"] == "", "position"] = pd.NA

all_candidate_features = []

for groups in POSITION_FEATURE_GROUPS.values():
    for features in groups.values():
        all_candidate_features.extend(features)

for features in GENERAL_FEATURE_GROUPS.values():
    all_candidate_features.extend(features)

all_candidate_features = list(dict.fromkeys(all_candidate_features))

for feature in all_candidate_features:
    if feature in df.columns:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
        df[feature] = df[feature].replace(
            [np.inf, -np.inf], np.nan
        )

# ============================================================
# 4. NAME NORMALIZATION AND POSITION GROUPING
# ============================================================
def normalize_name(value):
    """Normalize accents, capitalization and punctuation."""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(
        c for c in value if not unicodedata.combining(c)
    )
    value = value.casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def normalize_position(value):
    """Map known position labels to broad scouting categories."""
    if pd.isna(value):
        return pd.NA

    label = normalize_name(value)

    if label in {
        "forward", "attacker", "striker", "centre forward",
        "center forward", "winger",
    }:
        return "Forward"

    if label in {
        "midfielder", "midfield", "attacking midfielder",
        "central midfielder", "defensive midfielder",
    }:
        return "Midfielder"

    if label in {
        "defender", "defence", "defense", "centre back",
        "center back", "full back", "fullback", "wing back",
    }:
        return "Defender"

    if label in {"goalkeeper", "keeper", "goalie"}:
        return "Goalkeeper"

    return pd.NA


df["_position_group"] = df["position"].map(normalize_position)
df["_search_name"] = df["player_name"].map(normalize_name)
df["_search_key"] = df["name_key"].map(normalize_name)

# ============================================================
# 5. EA FC 24 PLAYER PROFILE LOOKUP
# DISPLAY ONLY — NEVER USED IN KNN CALCULATIONS
# ============================================================
FC24_DISPLAY_COLUMNS = {
    "FC24 Overall": "overall",
    "FC24 Potential": "potential",
    "FC24 Positions": "player_positions",
    "FC24 Club": "club_name",
    "FC24 Age": "age",
    "FC24 Value (€)": "value_eur",
    "FC24 Wage (€)": "wage_eur",
    "FC24 Pace": "pace",
    "FC24 Shooting": "shooting",
    "FC24 Passing": "passing",
    "FC24 Dribbling": "dribbling",
    "FC24 Defending": "defending",
    "FC24 Physical": "physic",
    "FC24 Weak Foot": "weak_foot",
    "FC24 Skill Moves": "skill_moves",
    "FC24 Preferred Foot": "preferred_foot",
    "FC24 Work Rate": "work_rate",
    "FC24 Height (cm)": "height_cm",
    "FC24 Weight (kg)": "weight_kg",
    "FC24 International Reputation": "international_reputation",
}

# Additional attributes available in the FC24 dataset.
FC24_EXTRA_ATTRIBUTES = [
    "attacking_crossing",
    "attacking_finishing",
    "attacking_heading_accuracy",
    "attacking_short_passing",
    "attacking_volleys",
    "skill_dribbling",
    "skill_curve",
    "skill_fk_accuracy",
    "skill_long_passing",
    "skill_ball_control",
    "movement_acceleration",
    "movement_sprint_speed",
    "movement_agility",
    "movement_reactions",
    "movement_balance",
    "power_shot_power",
    "power_jumping",
    "power_stamina",
    "power_strength",
    "power_long_shots",
    "mentality_aggression",
    "mentality_interceptions",
    "mentality_positioning",
    "mentality_vision",
    "mentality_penalties",
    "mentality_composure",
    "defending_marking_awareness",
    "defending_standing_tackle",
    "defending_sliding_tackle",
]

FC24_GOALKEEPER_ATTRIBUTES = [
    "goalkeeping_diving",
    "goalkeeping_handling",
    "goalkeeping_kicking",
    "goalkeeping_positioning",
    "goalkeeping_reflexes",
]

fc24_df = pd.DataFrame()
fc24_lookup = {}


def add_fc24_lookup_key(name, row):
    """Add a name only when it maps unambiguously to one FC24 player."""
    if pd.isna(name):
        return

    key = normalize_name(name)
    if not key:
        return

    # If a name belongs to multiple players, mark it ambiguous.
    if key not in fc24_lookup:
        fc24_lookup[key] = row
    elif fc24_lookup[key] is not None:
        existing_id = fc24_lookup[key].get("player_id")
        current_id = row.get("player_id")
        if str(existing_id) != str(current_id):
            fc24_lookup[key] = None


if FC24_DATA_PATH.exists():
    fc24_df = pd.read_csv(FC24_DATA_PATH)

    for _, fc24_player in fc24_df.iterrows():
        add_fc24_lookup_key(
            fc24_player.get("short_name"), fc24_player
        )
        add_fc24_lookup_key(
            fc24_player.get("long_name"), fc24_player
        )

    matched_names = sum(
        row is not None for row in fc24_lookup.values()
    )

    print("\n" + "=" * 72)
    print("EA FC 24 DISPLAY DATASET")
    print("=" * 72)
    print(f"Players loaded: {len(fc24_df)}")
    print("Purpose: display-only player information")
    print("KNN features and rankings are unchanged.")
else:
    print("\nWARNING: EA FC 24 dataset not found.")
    print(f"Expected path: {FC24_DATA_PATH}")
    print("KNN will continue without FC24 profile details.")



def get_fc24_profile(player_name):
    """Get FC24 attributes using exact normalized-name lookup."""

    key = normalize_name(player_name)
    row = fc24_lookup.get(key)

# Fallback: handle middle names in the FC 24 dataset
    if row is None and key:
        search_tokens = set(key.split())
        matches = []

        for _, candidate in fc24_df.iterrows():
            short_name = normalize_name(candidate.get("short_name", ""))
            long_name = normalize_name(candidate.get("long_name", ""))

            short_tokens = set(short_name.split())
            long_tokens = set(long_name.split())

            if (
                search_tokens.issubset(long_tokens)
                or search_tokens.issubset(short_tokens)
            ):
                matches.append(candidate)

    # Use the fallback only when exactly one player matches
        unique_matches = {
            str(candidate.get("player_id")): candidate
            for candidate in matches
        }

        if len(unique_matches) == 1:
            row = next(iter(unique_matches.values()))

    if row is None:
        return None

    profile = {}

    for display_name, source_column in FC24_DISPLAY_COLUMNS.items():
        value = row.get(source_column, pd.NA)

        if pd.isna(value):
            profile[display_name] = "N/A"
        elif source_column in {"value_eur", "wage_eur"}:
            profile[display_name] = f"€{float(value):,.0f}"
        elif source_column in {
            "overall", "potential", "age", "pace", "shooting",
            "passing", "dribbling", "defending", "physic",
            "weak_foot", "skill_moves", "height_cm", "weight_kg",
            "international_reputation",
        }:
            profile[display_name] = (
                int(value) if float(value).is_integer() else value
            )
        else:
            profile[display_name] = str(value)

    extra_attributes = list(FC24_EXTRA_ATTRIBUTES)

    if str(row.get("player_positions", "")).strip().upper() == "GK":
        extra_attributes.extend(FC24_GOALKEEPER_ATTRIBUTES)

    for attribute in extra_attributes:
        if attribute not in fc24_df.columns:
            continue

        value = row.get(attribute, pd.NA)

        if pd.isna(value):
            profile[attribute] = "N/A"
        else:
            profile[attribute] = (
                int(value)
                if isinstance(value, (int, float, np.integer, np.floating))
                and float(value).is_integer()
                else value
            )

    return profile



def print_fc24_profile(player_name, team_name=None):
    """Print a player's FC24 profile without affecting KNN ranking."""
    profile = get_fc24_profile(player_name)

    print(f"\n{player_name}" + (
        f" | {team_name}" if team_name else ""
    ))
    print("-" * 72)

    if profile is None:
        print("FC24 profile: no unique exact name match found.")
        print("KNN similarity results are unaffected.")
        return

    print("MAIN PROFILE")
    main_keys = list(FC24_DISPLAY_COLUMNS.keys())

    for key in main_keys:
        print(f"{key}: {profile.get(key, 'N/A')}")

    print("\nINDIVIDUAL ATTRIBUTES")
    for key in profile:
        if key not in FC24_DISPLAY_COLUMNS:
            print(f"{key}: {profile[key]}")


# ============================================================
# 6. FEATURE SELECTION, ELIGIBILITY AND GROUP WEIGHTS
# ============================================================
def feature_is_usable(group, feature):
    if feature not in group.columns:
        return False

    values = pd.to_numeric(group[feature], errors="coerce")

    return (
        values.notna().mean() >= MIN_FEATURE_COVERAGE
        and values.nunique(dropna=True) >= MIN_UNIQUE_VALUES
        and values.std(skipna=True) > 0
    )


def select_feature_groups(group, candidate_groups):
    """Keep usable columns without duplicating columns across groups."""
    selected = {}
    already_selected = set()

    for group_name, candidates in candidate_groups.items():
        usable = []

        for feature in candidates:
            if (
                feature not in already_selected
                and feature_is_usable(group, feature)
            ):
                usable.append(feature)
                already_selected.add(feature)

        if usable:
            selected[group_name] = usable

    return selected


def flatten_groups(feature_groups):
    return [
        feature
        for features in feature_groups.values()
        for feature in features
    ]


def get_eligible_players(
    group,
    features,
    feature_groups,
    minimum_observed=None,
    minimum_groups=1,
):
    """Count genuine values before imputation."""
    if len(features) < MIN_FEATURES:
        return group.iloc[0:0].copy(), 0

    if minimum_observed is None:
        minimum_observed = max(
            MIN_FEATURES, math.ceil(len(features) / 2)
        )

    mask = group[features].notna().sum(axis=1) >= minimum_observed

    if minimum_groups > 1:
        observed_group_count = pd.Series(
            0, index=group.index, dtype=int
        )

        for group_features in feature_groups.values():
            present = [
                feature for feature in group_features
                if feature in group.columns and feature in features
            ]

            if present:
                observed_group_count += (
                    group[present].notna().any(axis=1).astype(int)
                )

        mask &= observed_group_count >= minimum_groups

    eligible = group.loc[mask].copy()
    eligible = eligible.drop_duplicates(
        subset=["name_key", "team_key"]
    ).reset_index(drop=True)

    return eligible, minimum_observed


def calculate_group_weights(features, feature_groups):
    """Give each ability group equal total squared-distance weight."""
    weights = []

    for feature in features:
        group_size = next(
            len(items)
            for items in feature_groups.values()
            if feature in items
        )
        weights.append(1.0 / math.sqrt(group_size))

    return np.asarray(weights, dtype=float)


def build_knn(
    group,
    candidate_groups,
    minimum_observed_override=None,
    minimum_groups=1,
):
    """Build KNN with imputation, scaling and group balancing."""
    feature_groups = select_feature_groups(
        group, candidate_groups
    )
    features = flatten_groups(feature_groups)

    if len(features) < MIN_FEATURES:
        return None, None, None, None

    eligible, minimum_observed = get_eligible_players(
        group,
        features,
        feature_groups,
        minimum_observed=minimum_observed_override,
        minimum_groups=minimum_groups,
    )

    if len(eligible) < 3:
        return None, None, None, None

    X = eligible[features].astype(float)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    weights = calculate_group_weights(
        features, feature_groups
    )
    X_weighted = X_scaled * weights

    knn = NearestNeighbors(
        metric="euclidean", algorithm="brute"
    )
    knn.fit(X_weighted)

    model = {
        "imputer": imputer,
        "scaler": scaler,
        "knn": knn,
        "weights": weights,
        "features": features,
        "feature_groups": feature_groups,
        "minimum_observed": minimum_observed,
        "minimum_groups": minimum_groups,
    }

    return model, eligible, features, minimum_observed


def transform_query(model, query_features):
    values = model["imputer"].transform(
        query_features[model["features"]]
    )
    values = model["scaler"].transform(values)
    return values * model["weights"]


# ============================================================
# 7. BUILD FULL + COMPACT POSITION MODELS
# ============================================================
position_models = {}
position_data = {}
compact_position_models = {}
compact_position_data = {}

print("\n" + "=" * 72)
print("TRANSFERIQ: TIER 1 — FULL POSITION-SPECIFIC KNN")
print("=" * 72)

for position, candidate_groups in POSITION_FEATURE_GROUPS.items():
    group = df[df["_position_group"] == position].copy()

    model, eligible, features, minimum_observed = build_knn(
        group, candidate_groups
    )

    if model is None:
        print(f"\n{position}: unavailable with current feature coverage.")
        continue

    position_models[position] = model
    position_data[position] = eligible

    print(
        f"\n{position}: {len(eligible)}/{len(group)} eligible; "
        f"{len(features)} features; minimum "
        f"{minimum_observed} real values"
    )


print("\n" + "=" * 72)
print("TRANSFERIQ: TIER 2 — COMPACT POSITION-SPECIFIC KNN")
print("=" * 72)

for position, candidate_groups in COMPACT_POSITION_FEATURE_GROUPS.items():
    group = df[df["_position_group"] == position].copy()

    model, eligible, features, minimum_observed = build_knn(
        group,
        candidate_groups,
        minimum_observed_override=COMPACT_MIN_REAL_FEATURES,
        minimum_groups=COMPACT_MIN_ABILITY_GROUPS,
    )

    if model is None:
        print(f"\n{position}: unavailable with current feature coverage.")
        continue

    compact_position_models[position] = model
    compact_position_data[position] = eligible

    print(
        f"\n{position}: {len(eligible)}/{len(group)} eligible; "
        f"{len(features)} features; minimum {minimum_observed} "
        f"real values across {COMPACT_MIN_ABILITY_GROUPS} ability groups"
    )
    print("  Features: " + ", ".join(features))


# ============================================================
# 8. BUILD GENERAL FALLBACK MODEL
# ============================================================
print("\n" + "=" * 72)
print("TRANSFERIQ: GENERAL KNN FALLBACK")
print("=" * 72)

general_pool = df[df["_position_group"] != "Goalkeeper"].copy()

general_model, general_data, general_features, general_minimum = build_knn(
    general_pool, GENERAL_FEATURE_GROUPS
)

if general_model is None:
    raise ValueError(
        "General KNN could not be built. Check feature names and data coverage."
    )

print(f"Eligible players: {len(general_data)}")
print(f"Usable features: {len(general_features)}")
print(f"Minimum real features per player: {general_minimum}")

for group_name, group_features in general_model["feature_groups"].items():
    print(f"  {group_name}: {', '.join(group_features)}")


scouting_pools = (
    [general_data]
    + list(position_data.values())
    + list(compact_position_data.values())
)

all_eligible = pd.concat(scouting_pools, ignore_index=True)
all_eligible = all_eligible.drop_duplicates(
    subset=["name_key", "team_key"]
)

all_eligible.drop(
    columns=["_search_name", "_search_key", "_position_group"],
    errors="ignore",
).to_csv(OUTPUT_PATH, index=False)

print(f"\nEligible scouting records saved to: {OUTPUT_PATH}")


# ============================================================
# 9. COVERAGE COMPARISON
# ============================================================
def record_key(row):
    return (str(row["name_key"]), str(row["team_key"]))


full_keys = {
    pos: {
        record_key(row)
        for _, row in pool.iterrows()
    }
    for pos, pool in position_data.items()
}

compact_keys = {
    pos: {
        record_key(row)
        for _, row in pool.iterrows()
    }
    for pos, pool in compact_position_data.items()
}

general_keys = {
    record_key(row)
    for _, row in general_data.iterrows()
}

coverage_rows = []

for _, row in df.drop_duplicates(
    subset=["name_key", "team_key"]
).iterrows():
    pos = row["_position_group"]
    key = record_key(row)

    before_route = "Unavailable"
    after_route = "Unavailable"

    if pd.notna(pos) and pos in full_keys and key in full_keys[pos]:
        before_route = f"{pos} full model"
    elif pos != "Goalkeeper" and key in general_keys:
        before_route = "General fallback"

    if pd.notna(pos) and pos in full_keys and key in full_keys[pos]:
        after_route = f"{pos} full model"
    elif pd.notna(pos) and pos in compact_keys and key in compact_keys[pos]:
        after_route = f"{pos} compact fallback"
    elif pos != "Goalkeeper" and key in general_keys:
        after_route = "General fallback (last resort)"

    coverage_rows.append({
        "player_name": row["player_name"],
        "team_key": row["team_key"],
        "position": pos if pd.notna(pos) else "Unknown",
        "before_route": before_route,
        "after_route": after_route,
        "gained_output": (
            before_route == "Unavailable"
            and after_route != "Unavailable"
        ),
    })


coverage_df = pd.DataFrame(coverage_rows)
coverage_path = (
    OUTPUT_DIR / "knn_tiered_fallback_coverage_comparison.csv"
)
coverage_df.to_csv(coverage_path, index=False)

before_count = int(
    (coverage_df["before_route"] != "Unavailable").sum()
)
after_count = int(
    (coverage_df["after_route"] != "Unavailable").sum()
)
total_count = len(coverage_df)

print("\n" + "=" * 72)
print("BEFORE vs AFTER COVERAGE")
print("=" * 72)
print(f"Unique player-team records: {total_count}")
print(
    f"Before: {before_count}/{total_count} "
    f"({before_count / max(total_count, 1) * 100:.2f}%)"
)
print(
    f"After:  {after_count}/{total_count} "
    f"({after_count / max(total_count, 1) * 100:.2f}%)"
)
print(f"Net additional players with output: {after_count - before_count}")

compact_gains = (
    coverage_df["after_route"].str.contains(
        "compact fallback", case=False
    )
    & coverage_df["gained_output"]
).sum()

print(
    "New coverage gained specifically from compact position models: "
    f"{int(compact_gains)}"
)
print("\nAfter-route counts:")
print(coverage_df["after_route"].value_counts().to_string())
print(f"\nCoverage comparison saved to: {coverage_path}")


# ============================================================
# 10. SMART PLAYER SEARCH
# ============================================================
def search_player(player_name):
    """Search exact names first, then safe partial matches."""
    query = normalize_name(player_name)

    if not query:
        print("Please enter a player name.")
        return None

    exact = df[
        (df["_search_name"] == query)
        | (df["_search_key"] == query)
    ]

    if exact.empty:
        tokens = set(query.split())

        mask = df.apply(
            lambda row: (
                query in row["_search_name"]
                or query in row["_search_key"]
                or tokens.issubset(
                    set(row["_search_name"].split())
                )
                or tokens.issubset(
                    set(row["_search_key"].split())
                )
            ),
            axis=1,
        )

        matches = df[mask].copy()
    else:
        matches = exact.copy()

    matches = matches.drop_duplicates(
        subset=["name_key", "team_key"]
    )

    if matches.empty:
        print("\nPlayer not found in the original dataset.")
        print("Check the spelling or try the player's full name.")
        return None

    if len(matches) == 1:
        return matches.iloc[0]

    print("\nMultiple players match your search:")
    matches = matches.reset_index(drop=True)

    for index, row in matches.iterrows():
        position = (
            row["position"]
            if pd.notna(row["position"])
            else "Unknown"
        )
        print(
            f"{index + 1}. {row['player_name']} | "
            f"{row['team_key']} | {position}"
        )

    choice = input(
        "\nEnter the number of the correct player (or 0 to cancel): "
    ).strip()

    if not choice.isdigit():
        print("Invalid selection.")
        return None

    selected = int(choice)

    if selected == 0:
        print("Search cancelled.")
        return None

    if selected < 1 or selected > len(matches):
        print("Selection out of range.")
        return None

    return matches.iloc[selected - 1]


# ============================================================
# 11. FIND SIMILAR PLAYERS + DISPLAY EA FC 24 DETAILS
# ============================================================
def find_similar_players(player_name, top_n=TOP_N):
    """Find similar players and attach FC24 display data afterward."""
    player = search_player(player_name)

    if player is None:
        return None

    position = player["_position_group"]
    key = (str(player["name_key"]), str(player["team_key"]))

    model = None
    candidate_pool = None
    model_name = None

    if pd.notna(position) and position in position_models:
        pool = position_data[position]

        pool_keys = set(
            zip(
                pool["name_key"].astype(str),
                pool["team_key"].astype(str),
            )
        )

        if key in pool_keys:
            model = position_models[position]
            candidate_pool = pool
            model_name = f"{position} Full KNN"

    if model is None and pd.notna(position):
        if position in compact_position_models:
            pool = compact_position_data[position]

            pool_keys = set(
                zip(
                    pool["name_key"].astype(str),
                    pool["team_key"].astype(str),
                )
            )

            if key in pool_keys:
                model = compact_position_models[position]
                candidate_pool = pool
                model_name = (
                    f"{position} Compact KNN (fallback)"
                )

    if model is None:
        if position == "Goalkeeper":
            print(
                f"\n{player['player_name']} has insufficient "
                "goalkeeper statistics for both goalkeeper tiers."
            )
            print("No outfield fallback is used for goalkeepers.")
            return None

        pool = general_data
        pool_keys = set(
            zip(
                pool["name_key"].astype(str),
                pool["team_key"].astype(str),
            )
        )

        if key in pool_keys:
            model = general_model
            candidate_pool = pool
            model_name = "General KNN (last-resort fallback)"

    if model is None:
        print(
            f"\n{player['player_name']} is in the dataset but "
            "does not qualify for either position-specific tier."
        )
        print("No recommendation generated; no values were invented.")
        return None

    query_record = candidate_pool[
        (candidate_pool["name_key"].astype(str) == key[0])
        & (candidate_pool["team_key"].astype(str) == key[1])
    ]

    if query_record.empty:
        print("Could not locate the selected player in the KNN pool.")
        return None

    query_features = query_record[model["features"]]

    observed_count = int(
        query_features.notna().sum(axis=1).iloc[0]
    )

    transformed_query = transform_query(model, query_features)

    distances, indices = model["knn"].kneighbors(
        transformed_query,
        n_neighbors=len(candidate_pool),
    )

    results = []

    for distance, index in zip(distances[0], indices[0]):
        candidate = candidate_pool.iloc[index]

        # Exclude the queried player.
        if (
            str(candidate["name_key"]) == key[0]
            and str(candidate["team_key"]) == key[1]
        ):
            continue

        results.append({
            "Player": candidate["player_name"],
            "Team": candidate["team_key"],
            "Position": (
                candidate["position"]
                if pd.notna(candidate["position"])
                else "Unknown"
            ),
            "Distance": round(float(distance), 3),
            "Observed features": int(
                candidate[model["features"]].notna().sum()
            ),
        })

        if len(results) >= max(1, int(top_n)):
            break

    print("\n" + "=" * 90)
    print(
        f"Player: {player['player_name']} | "
        f"Team: {player['team_key']}"
    )
    print(
        f"Position: {position if pd.notna(position) else 'Unknown'}"
    )
    print(f"Model selected: {model_name}")
    print(
        f"Real features: {observed_count}/{len(model['features'])}; "
        f"minimum required: {model['minimum_observed']}"
    )

    if model.get("minimum_groups", 1) > 1:
        print(
            f"Minimum distinct ability groups: "
            f"{model['minimum_groups']}"
        )

    print("=" * 90)

    result_df = pd.DataFrame(results)

    if result_df.empty:
        print("No alternative players found.")
        return result_df

    print("\nMOST SIMILAR PLAYERS")
    print("-" * 100)

    main_columns = [
        "Player",
        "Team",
        "Position",
        "Distance",
        "Observed features",
    ]

    print(
        result_df[main_columns].to_string(
            index=False
        )
    )

    # Attach FC24 details only after KNN has produced its ranking.
    # The FC24 dataset never enters the model or distance calculation.
    print("\n" + "=" * 90)
    print("EA FC 24 PLAYER PROFILES")
    print("=" * 90)

    for _, result in result_df.iterrows():
        print_fc24_profile(
            result["Player"],
            result["Team"],
        )

    print("\n" + "=" * 90)
    print(
        "Lower KNN distance means closer standardized performance "
        "statistics. Distance is not a probability."
    )
    print(
        "EA FC 24 attributes are display-only and do not affect "
        "the similarity ranking."
    )

    # Return original KNN columns plus FC24 display attributes.
    # A missing or ambiguous FC24 match is represented as "Not found".
    for display_column in FC24_DISPLAY_COLUMNS:
        result_df[display_column] = result_df["Player"].map(
            lambda name: (
                get_fc24_profile(name) or {}
            ).get(display_column, "Not found")
        )

    return result_df


# ============================================================
# 12. API FUNCTION FOR THE TRANSFERIQ WEBSITE
# Add this above: if __name__ == "__main__":
# ============================================================

def find_similar_players_for_api(player_name, top_n=TOP_N):
    """
    Return clean player data for the React website.
    Existing KNN ranking logic remains unchanged.
    """

    # Reuse the existing KNN function.
    results = find_similar_players(player_name, top_n=top_n)

    if results is None:
        return {
            "success": False,
            "message": (
                "Player not found, ambiguous, or unavailable "
                "for the current KNN model."
            ),
            "players": [],
        }

    players = []

    for _, result in results.iterrows():
        name = str(result["Player"])
        team = str(result["Team"])
        position = str(result["Position"])

        profile = get_fc24_profile(name)

        if profile is None:
            profile = {}

        is_goalkeeper = (
            normalize_position(position) == "Goalkeeper"
            or str(profile.get("FC24 Positions", "")).strip().upper() == "GK"
        )

        if is_goalkeeper:
            attributes = {
                "Diving": profile.get("goalkeeping_diving"),
                "Handling": profile.get("goalkeeping_handling"),
                "Kicking": profile.get("goalkeeping_kicking"),
                "Reflexes": profile.get("goalkeeping_reflexes"),
                "Speed": profile.get("FC24 Pace"),
                "Positioning": profile.get("goalkeeping_positioning"),
            }
        else:
            attributes = {
                "Pace": profile.get("FC24 Pace"),
                "Shooting": profile.get("FC24 Shooting"),
                "Passing": profile.get("FC24 Passing"),
                "Dribbling": profile.get("FC24 Dribbling"),
                "Defending": profile.get("FC24 Defending"),
                "Physical": profile.get("FC24 Physical"),
            }

        # Find a matching EA FC 24 record for its player ID.
        fc_row = fc24_lookup.get(normalize_name(name))
        player_id = None

        if fc_row is not None:
            raw_id = fc_row.get("player_id")

            if pd.notna(raw_id):
                player_id = str(raw_id).split(".")[0]

        # Missing attributes remain null; never invent ratings.
        clean_attributes = {}

        for key, value in attributes.items():
            if value is None or pd.isna(value):
                clean_attributes[key] = None
            else:
                clean_attributes[key] = value

        players.append({
            "name": name,
            "team": team,
            "position": position,
            "player_id": player_id,
            "is_goalkeeper": is_goalkeeper,
            "attributes": clean_attributes,
        })

    return {
        "success": True,
        "searched_player": player_name,
        "count": len(players),
        "players": players,
    }



# ============================================================
# 13. TERMINAL INTERFACE
# ============================================================
if __name__ == "__main__":
    print("\nWelcome to TransferIQ Player Scouting")
    print("Search by first name, surname, or full name.")
    print("The system returns up to 7 similar players.")
    print("EA FC 24 data is used only for additional player details.")
    print("Examples: Alisson, Alisson Becker, Bruno Fernandes")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nPlayer name: ").strip()

        if query.casefold() == "exit":
            print("TransferIQ scouting search closed.")
            break

        if query:
            find_similar_players(query)





