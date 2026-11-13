
from pathlib import Path
import unicodedata
import re

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "market_value_dataset.csv"
MODEL_PATH = ROOT / "models" / "market_value_model.joblib"
FC24_DATA_PATH = (
    ROOT / "data" / "processed"
    / "ea_fc24_premier_league_players.csv"
)

# Preserve the existing Model A target season.
TARGET_SEASON = 2024

PERFORMANCE_COLUMNS = [
    "minutes_played",
    "goals",
    "assists",
    "yellow_cards",
    "red_cards",
    "appearances",
]

# ============================================================
# EA FC 24 PROFILE FIELDS — DISPLAY ONLY
# These fields do not enter Model A's prediction.
# ============================================================
FC24_DISPLAY_COLUMNS = {
    "Overall rating": "overall",
    "Potential": "potential",
    "FC 24 positions": "player_positions",
    "FC 24 club": "club_name",
    "FC 24 age": "age",
    "FC 24 value": "value_eur",
    "FC 24 wage": "wage_eur",
    "Pace": "pace",
    "Shooting": "shooting",
    "Passing": "passing",
    "Dribbling": "dribbling",
    "Defending": "defending",
    "Physicality": "physic",
    "Weak foot": "weak_foot",
    "Skill moves": "skill_moves",
    "Preferred foot": "preferred_foot",
    "Work rate": "work_rate",
    "Height (cm)": "height_cm",
    "Weight (kg)": "weight_kg",
    "International reputation": "international_reputation",
}

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


def normalize_name(value):
    """Normalize accents, capitalization and punctuation for lookup."""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
    )
    value = value.casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def load_fc24_data():
    """Load FC 24 profiles for display; never used by Model A."""
    if not FC24_DATA_PATH.exists():
        print(
            "\nWARNING: EA FC 24 dataset not found."
            f"\nExpected: {FC24_DATA_PATH}"
            "\nPredictions will continue without FC 24 details."
        )
        return pd.DataFrame(), {}

    fc24_df = pd.read_csv(FC24_DATA_PATH)
    lookup = {}

    for _, row in fc24_df.iterrows():
        for column in ("short_name", "long_name"):
            name = row.get(column)

            if pd.isna(name):
                continue

            key = normalize_name(name)

            if not key:
                continue

            if key not in lookup:
                lookup[key] = row
            elif lookup[key] is not None:
                # Do not choose arbitrarily if names are ambiguous.
                existing_id = lookup[key].get("player_id")
                current_id = row.get("player_id")

                if str(existing_id) != str(current_id):
                    lookup[key] = None

    print(f"EA FC 24 display profiles loaded: {len(fc24_df)}")
    return fc24_df, lookup




def get_fc24_profile(player_name, lookup, club_name=None):
    """Find an FC 24 player using their name and, when available, club."""

    key = normalize_name(player_name)

    # First try an exact name match.
    row = lookup.get(key)

    # If the exact name is ambiguous or unavailable,
    # search the lookup for possible name matches.
    if row is None and key:
        search_tokens = set(key.split())
        matches = {}

        for lookup_key, candidate in lookup.items():

            # Skip ambiguous entries.
            if candidate is None:
                continue

            candidate_tokens = set(normalize_name(lookup_key).split())

            # Allow additional middle names in either dataset.
            if not (
                search_tokens.issubset(candidate_tokens)
                or candidate_tokens.issubset(search_tokens)
            ):
                continue

            # Use the club to distinguish players with similar names.
            if club_name:
                candidate_club = normalize_name(
                    candidate.get("club_name", "")
                )
                requested_club = normalize_name(club_name)

                if candidate_club != requested_club:
                    continue

            player_id = candidate.get("player_id")

            if pd.notna(player_id):
                matches[str(player_id)] = candidate

        # Accept only one unique player.
        if len(matches) == 1:
            row = next(iter(matches.values()))

    if row is None:
        return None

    profile = {}

    for label, source_column in FC24_DISPLAY_COLUMNS.items():
        value = row.get(source_column, pd.NA)

        if pd.isna(value):
            profile[label] = "N/A"
        elif source_column in {"value_eur", "wage_eur"}:
            profile[label] = f"€{float(value):,.0f}"
        elif source_column in {
            "overall", "potential", "age", "pace", "shooting",
            "passing", "dribbling", "defending", "physic",
            "weak_foot", "skill_moves", "height_cm", "weight_kg",
            "international_reputation",
        }:
            profile[label] = (
                int(value) if float(value).is_integer() else value
            )
        else:
            profile[label] = str(value)

    extra_attributes = list(FC24_EXTRA_ATTRIBUTES)

    positions = str(row.get("player_positions", "")).upper()
    if "GK" in positions.split(", "):
        extra_attributes.extend(FC24_GOALKEEPER_ATTRIBUTES)

    for attribute in extra_attributes:
        if attribute not in row.index:
            continue

        value = row.get(attribute, pd.NA)

        if pd.isna(value):
            profile[attribute] = "N/A"
        elif isinstance(value, (int, float)):
            profile[attribute] = (
                int(value) if float(value).is_integer() else value
            )
        else:
            profile[attribute] = value

    return profile


def get_fc24_profile_from_df(player_name, fc24_df, club_name=None):
    """Find a player's FC24 profile using name and club from the original dataset."""

    if fc24_df is None or fc24_df.empty:
        return None

    target_name = normalize_name(player_name)
    target_tokens = set(target_name.split())

    candidates = fc24_df.copy()

    # Filter by club first to avoid matching another player with the same name.
    if club_name and "club_name" in candidates.columns:
        target_club = normalize_name(club_name)

        candidates = candidates[
            candidates["club_name"].apply(
                lambda value: normalize_name(value) == target_club
                if pd.notna(value) else False
            )
        ]

    # Check both short_name and long_name.
    def matches_name(row):
        for column in ("short_name", "long_name"):
            value = row.get(column)

            if pd.isna(value):
                continue

            candidate_name = normalize_name(value)
            candidate_tokens = set(candidate_name.split())

            if (
                candidate_name == target_name
                or target_tokens.issubset(candidate_tokens)
                or candidate_tokens.issubset(target_tokens)
            ):
                return True

        return False

    matches = candidates[candidates.apply(matches_name, axis=1)]

    # Avoid duplicate records for the same player.
    matches = matches.drop_duplicates(subset=["player_id"])

    # Only return a result when the match is unambiguous.
    if len(matches) != 1:
        return None

    row = matches.iloc[0]

    # Reuse the existing profile formatter.
    temp_lookup = {}

    for column in ("short_name", "long_name"):
        value = row.get(column)

        if pd.notna(value):
            temp_lookup[normalize_name(value)] = row

    return get_fc24_profile(
        player_name,
        temp_lookup,
        club_name=club_name
    )

def print_fc24_profile(player_name, lookup):
    """Print extra FC 24 details without affecting the prediction."""
    print("\n" + "-" * 60)
    print("EA SPORTS FC 24 PLAYER PROFILE")
    print("-" * 60)

    profile = get_fc24_profile(player_name, lookup)

    if profile is None:
        print("FC 24 profile: no unique exact name match found.")
        print("The Model A prediction is still available.")
        return

    print("\nMAIN PROFILE")
    for label in FC24_DISPLAY_COLUMNS:
        print(f"{label}: {profile.get(label, 'N/A')}")

    print("\nINDIVIDUAL PLAYER ATTRIBUTES")
    for label, value in profile.items():
        if label not in FC24_DISPLAY_COLUMNS:
            print(f"{label}: {value}")


# ============================================================
# MODEL A DATA PREPARATION — PRESERVED
# ============================================================
def prepare_data():
    df = pd.read_csv(DATA_PATH)

    df["date_of_birth"] = pd.to_datetime(
        df["date_of_birth"], errors="coerce"
    )
    df["valuation_date"] = pd.to_datetime(
        df["valuation_date"], errors="coerce"
    )

    df = df.sort_values(
        ["player_id", "season"]
    ).reset_index(drop=True)

    df["age"] = (
        (df["valuation_date"] - df["date_of_birth"]).dt.days
        / 365.25
    )

    grouped = df.groupby("player_id", sort=False)

    for col in PERFORMANCE_COLUMNS:
        df[f"previous_{col}"] = grouped[col].shift(1)

    df["previous_season"] = grouped["season"].shift(1)
    df["previous_market_value"] = grouped[
        "market_value_in_eur"
    ].shift(1)
    df["previous_valuation_date"] = grouped[
        "valuation_date"
    ].shift(1)

    gap_days = (
        df["valuation_date"] - df["previous_valuation_date"]
    ).dt.days

    consecutive = df["season"] - df["previous_season"] == 1

    valid_previous_value = (
        df["previous_market_value"].gt(0)
        & gap_days.gt(0)
        & gap_days.le(548)
    )

    df = df[
        consecutive
        & valid_previous_value
        & df["market_value_in_eur"].gt(0)
        & df["age"].notna()
    ].copy()

    return df


# ============================================================
# TERMINAL INTERFACE
# ============================================================
def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained Model A not found: {MODEL_PATH}"
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Market value dataset not found: {DATA_PATH}"
        )

    print("Loading TransferIQ Model A...")

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    features = artifact["features"]

    df = prepare_data()
    season_data = df[df["season"] == TARGET_SEASON].copy()

    if season_data.empty:
        print(
            f"No eligible records found for season {TARGET_SEASON}."
        )
        return

    # Load display information separately from Model A.
    _, fc24_lookup = load_fc24_data()

    print("Model loaded successfully.")
    print(
        f"Eligible players for season {TARGET_SEASON}: "
        f"{len(season_data)}"
    )
    print("\nType a player name, or type 'exit' to quit.")

    while True:
        query = input("\nPlayer name: ").strip()

        if query.lower() == "exit":
            print("Market value prediction closed.")
            break

        if not query:
            continue

        matches = season_data[
            season_data["player_name"].str.contains(
                query,
                case=False,
                na=False,
                regex=False,
            )
        ]

        if matches.empty:
            print("Player not found for this season.")
            continue

        if len(matches) > 1:
            print("\nMultiple matches:")
            for name in matches["player_name"].drop_duplicates():
                print(f"- {name}")
            print("Type a more specific name.")
            continue

        player = matches.iloc[[0]]
        X = player[features]

        # Original Model A prediction logic is unchanged.
        predicted_value = float(model.predict(X)[0])
        actual_value = float(
            player["market_value_in_eur"].iloc[0]
        )

        player_name = player["player_name"].iloc[0]

        print("\n" + "=" * 60)
        print("TRANSFERIQ — MARKET VALUE PREDICTION")
        print("=" * 60)
        print(f"Player: {player_name}")
        print(f"Target season: {TARGET_SEASON}")
        print(f"Age at valuation: {player['age'].iloc[0]:.1f}")
        print(
            f"Predicted market value: €{predicted_value:,.0f}"
        )
        print(
            f"Recorded market value:  €{actual_value:,.0f}"
        )
        print(
            f"Absolute error:          "
            f"€{abs(predicted_value - actual_value):,.0f}"
        )
        print("=" * 60)
        print(
            "The predicted value is Model A's estimate. "
            "The recorded value is the dataset reference."
        )

        # Extra player details are displayed after prediction.
        # They do not affect Model A.
        print_fc24_profile(player_name, fc24_lookup)


if __name__ == "__main__":
    main()

