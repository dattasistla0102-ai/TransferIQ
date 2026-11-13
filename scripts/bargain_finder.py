
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import unicodedata
import re

import joblib
import pandas as pd


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

MODEL_PATH = ROOT / "models" / "market_value_model.joblib"

TARGET_SEASON = 2024
TOP_N = 7


# Import existing models. Their model logic is not changed.
with redirect_stdout(StringIO()):
    import build_knn_scout as knn
    import demo_market_value_prediction as model_a


# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def normalize_name(name):
    if name is None or pd.isna(name):
        return ""

    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(
        char for char in name
        if not unicodedata.combining(char)
    )

    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", " ", name)

    return " ".join(name.split())


def money(value):
    if value is None or pd.isna(value):
        return "Unavailable"

    return f"€{float(value):,.0f}"


def heading(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def get_value(row, columns, default=None):
    for column in columns:
        try:
            value = row.get(column)

            if value is not None and not pd.isna(value):
                return value
        except (AttributeError, TypeError, ValueError):
            continue

    return default


# ============================================================
# 3. EA FC 24 PROFILE
# Display ONLY six requested attributes.
# ============================================================

FC24_ATTRIBUTE_KEYS = {
    "Pace": ["fc24 pace", "pace"],
    "Shooting": ["fc24 shooting", "shooting", "shotting"],
    "Passing": ["fc24 passing", "passing"],
    "Dribbling": ["fc24 dribbling", "dribbling"],
    "Defending": ["fc24 defending", "defending", "defence", "defense"],
    "Physical": ["fc24 physical", "physical", "physic", "physicality"],
}


def print_fc24_profile(player_name):
    if not hasattr(knn, "get_fc24_profile"):
        print("FC 24 profile lookup is unavailable.")
        return

    try:
        profile = knn.get_fc24_profile(player_name)
    except Exception as error:
        print(f"FC 24 profile unavailable: {error}")
        return

    if profile is None:
        print("FC 24 profile: no unique matching player found.")
        return

    if isinstance(profile, pd.DataFrame):
        if len(profile) != 1:
            print("FC 24 profile: no unique matching player found.")
            return
        profile = profile.iloc[0]

    if isinstance(profile, pd.Series):
        profile = profile.to_dict()

    if not isinstance(profile, dict):
        print("FC 24 profile is unavailable.")
        return

    # Normalize keys while preserving the FC24 prefix.
    normalized_profile = {
        re.sub(r"[^a-z0-9]+", "", str(key).lower()): value
        for key, value in profile.items()
    }

    print("\nEA SPORTS FC 24 ATTRIBUTES")

    displayed = 0

    for label, aliases in FC24_ATTRIBUTE_KEYS.items():
        value = None

        for alias in aliases:
            key = re.sub(r"[^a-z0-9]+", "", alias.lower())

            if key in normalized_profile:
                candidate = normalized_profile[key]

                if candidate is not None and not pd.isna(candidate):
                    value = candidate
                    break

        if value is not None:
            print(f"  {label}: {value}")
            displayed += 1

    if displayed == 0:
        print("The six attributes could not be matched to this profile.")


# ============================================================
# 4. LOAD MODEL A
# ============================================================

def load_market_value_model():
    data_path = (
        ROOT / "data" / "processed" / "market_value_dataset.csv"
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model A not found: {MODEL_PATH}")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Market value dataset not found: {data_path}"
        )

    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    features = artifact["features"]

    # Reuse the original Model A feature preparation.
    prepared_data = model_a.prepare_data()

    market_data = prepared_data[
        prepared_data["season"] == TARGET_SEASON
    ].copy()

    market_data["_lookup_name"] = (
        market_data["player_name"].map(normalize_name)
    )

    return model, features, market_data


def predict_player_value(player_name, model, features, market_data):
    normalized = normalize_name(player_name)

    matches = market_data[
        market_data["_lookup_name"] == normalized
    ]

    if len(matches) != 1:
        return None

    row = matches.iloc[[0]]
    prediction = float(model.predict(row[features])[0])

    return {
        "predicted_value": prediction,
        "actual_value": float(row["market_value_in_eur"].iloc[0]),
        "season": int(row["season"].iloc[0]),
        "age": float(row["age"].iloc[0]),
    }


# ============================================================
# 5. PERFORMANCE STATISTICS
# Uses the ACTUAL KNN dataset column names.
# ============================================================

PERFORMANCE_COLUMNS = {
    "Appearances": [
        "player_player_ratings__Matches",
    ],
    "Minutes Played": [
        "player_player_ratings__Minutes",
    ],
    "Goals": [
        "player_top_scorers__Goals",
        "player_goals_per_90__Total Goals",
    ],
    "Assists": [
        "player_top_assists__Assists",
        "player_big_chances_created__Total Assists",
    ],
    "Expected Goals (xG)": [
        "player_expected_goals__Expected Goals (xG)",
    ],
    "Expected Assists (xA)": [
        "player_expected_assists__Expected Assists (xA)",
    ],
    "Goals per 90": [
        "player_goals_per_90__Goals per 90",
    ],
    "Successful Dribbles per 90": [
        "player_contests_won__Successful Dribbles per 90",
    ],
    "Pass Success (%)": [
        "player_accurate_passes__Pass Success (%)",
    ],
    "Tackles per 90": [
        "player_tackles_won__Tackles per 90",
    ],
    "Interceptions per 90": [
        "player_interceptions__Interceptions per 90",
    ],
    "Clearances per 90": [
        "player_effective_clearances__Clearances per 90",
    ],
    "FotMob Rating": [
        "player_player_ratings__FotMob Rating",
    ],
}


def get_knn_performance_stats(player_name, team_key=None):
    data = knn.df
    normalized = normalize_name(player_name)

    # The diagnostic confirmed _search_name is available.
    if "_search_name" in data.columns:
        matches = data[
            data["_search_name"].map(normalize_name) == normalized
        ]
    elif "player_name" in data.columns:
        matches = data[
            data["player_name"].map(normalize_name) == normalized
        ]
    else:
        return None

    # If the requested team matches, use that player's team row.
    if team_key is not None and "team_key" in data.columns:
        normalized_team = normalize_name(team_key)

        team_matches = matches[
            matches["team_key"].map(normalize_name) == normalized_team
        ]

        if not team_matches.empty:
            matches = team_matches

    if matches.empty:
        return None

    return matches.iloc[0]


def print_performance_stats(player_name, team_key=None):
    player = get_knn_performance_stats(player_name, team_key)

    print("\nPERFORMANCE STATISTICS")

    if player is None:
        print("No matching performance statistics found.")
        return

    displayed = 0

    for label, possible_columns in PERFORMANCE_COLUMNS.items():
        value = get_value(player, possible_columns)

        if value is None:
            continue

        # Avoid showing a statistic when the source value is missing.
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            continue

        # Keep counts readable.
        if label in {"Appearances", "Minutes Played", "Goals", "Assists"}:
            value = int(round(float(value)))

        elif isinstance(value, (int, float)):
            value = round(float(value), 2)

        print(f"  {label}: {value}")
        displayed += 1

    if displayed == 0:
        print("No recognized performance statistics found.")


# ============================================================
# 6. PLAYER SELECTION AND KNN
# ============================================================

def select_target_player(query):
    player = knn.search_player(query)

    if player is None:
        return None

    if isinstance(player, pd.DataFrame):
        if player.empty:
            return None

        if len(player) != 1:
            print("Multiple players found. Enter a more specific name.")
            return None

        player = player.iloc[0]

    if isinstance(player, pd.Series):
        return player.to_dict()

    if isinstance(player, dict):
        return player

    return None


def get_similar_players(target_player):
    original_search_player = knn.search_player

    try:
        # Reuse the selected player without prompting a second time.
        knn.search_player = lambda _query: target_player

        # Prevent duplicate output from the original KNN script.
        with redirect_stdout(StringIO()):
            similar = knn.find_similar_players(
                target_player["player_name"],
                top_n=TOP_N,
            )

    finally:
        knn.search_player = original_search_player

    if similar is None:
        return pd.DataFrame()

    return similar


# ============================================================
# 7. TARGET PLAYER DETAILS
# ============================================================

def display_target_player(target_player, target_value):
    name = target_player["player_name"]
    team = target_player.get("team_key", "Unknown")
    position = target_player.get("position", "Unknown")

    heading("TARGET PLAYER")

    print(f"Player: {name}")
    print(f"Team: {team}")
    print(f"Position: {position}")
    print(f"Valuation season: {TARGET_SEASON}")

    print(
        "Model A predicted market value: "
        f"{money(target_value['predicted_value'])}"
    )

    print(
        "Recorded market value in dataset: "
        f"{money(target_value['actual_value'])}"
    )

    print_performance_stats(name, team)
    print_fc24_profile(name)


# ============================================================
# 8. BARGAIN ANALYSIS
# ============================================================

def analyze_bargains(
    target_player,
    target_value,
    similar_players,
    model,
    features,
    market_data,
):
    target_name = target_player["player_name"]
    target_predicted = target_value["predicted_value"]

    results = []
    skipped = []

    for _, row in similar_players.iterrows():
        name = get_value(
            row,
            ["Player", "player_name", "short_name"],
        )

        if name is None:
            continue

        name = str(name)

        if normalize_name(name) == normalize_name(target_name):
            continue

        team = get_value(row, ["Team", "team_key", "team"], "Unknown")
        position = get_value(row, ["Position", "position"], "Unknown")
        distance = get_value(row, ["Distance", "distance"])

        comparable_value = predict_player_value(
            name,
            model,
            features,
            market_data,
        )

        if comparable_value is None:
            skipped.append(name)
            continue

        difference = target_predicted - comparable_value["predicted_value"]

        results.append({
            "Player": name,
            "Team": team,
            "Position": position,
            "KNN Distance": distance,
            "Predicted Value (€)": comparable_value["predicted_value"],
            "Cheaper By (€)": difference,
            "Team Key": row.get("Team", team),
        })

    heading("BARGAIN ANALYSIS")
    print(f"Target: {target_name}")
    print(f"Target predicted value: {money(target_predicted)}")
    print(f"Valuation season: {TARGET_SEASON}")

    if not results:
        print("No comparable players had eligible Model A valuations.")

        if skipped:
            print("\nPlayers without eligible valuations:")
            for name in skipped:
                print(f"  - {name}")

        return

    results_df = pd.DataFrame(results)

    cheaper = results_df[results_df["Cheaper By (€)"] > 0].copy()
    expensive = results_df[results_df["Cheaper By (€)"] < 0].copy()
    equal_value = results_df[results_df["Cheaper By (€)"] == 0].copy()

    print("\nALL VALUED COMPARABLES")

    summary = results_df[
        [
            "Player",
            "Team",
            "Position",
            "KNN Distance",
            "Predicted Value (€)",
            "Cheaper By (€)",
        ]
    ].copy()

    summary["Predicted Value (€)"] = summary[
        "Predicted Value (€)"
    ].map(money)

    summary["Cheaper By (€)"] = summary[
        "Cheaper By (€)"
    ].map(money)

    print(summary.to_string(index=False))

    if not cheaper.empty:
        cheaper = cheaper.sort_values(
            "Predicted Value (€)",
            ascending=True,
        )

        heading("LOWER-VALUE ALTERNATIVES")

        if len(cheaper) == len(results_df):
            print("All eligible comparables are cheaper than the target.")
        else:
            print("Cheaper comparable players, ranked by predicted value:")

        for rank, (_, player) in enumerate(cheaper.iterrows(), start=1):
            name = player["Player"]

            heading(f"ALTERNATIVE {rank}: {name}")

            print(f"Team: {player['Team']}")
            print(f"Position: {player['Position']}")
            print(f"KNN distance: {player['KNN Distance']}")
            print(
                "Predicted market value: "
                f"{money(player['Predicted Value (€)'])}"
            )
            print(
                "Predicted value below target by: "
                f"{money(player['Cheaper By (€)'])}"
            )

            print_performance_stats(name, player["Team Key"])
            print_fc24_profile(name)

    elif not expensive.empty and equal_value.empty:
        heading("TARGET IS THE LOWEST-VALUE OPTION")
        print(
            "All eligible comparables have higher predicted market "
            "values than the target."
        )
        print(f"{target_name} is the lowest-valued option in this group.")

    elif not equal_value.empty and expensive.empty:
        heading("VALUE COMPARISON")
        print(
            "No comparable has a lower predicted value. "
            "At least one has the same predicted value."
        )

    else:
        heading("NO CHEAPER ALTERNATIVES FOUND")
        print(
            "None of the eligible comparable players has a lower "
            "predicted market value than the target."
        )

    if skipped:
        heading("COMPARABLES WITHOUT ELIGIBLE VALUATIONS")
        for name in skipped:
            print(f"- {name}")

        print(
            f"These players were excluded because Model A could not "
            f"produce an eligible valuation for {TARGET_SEASON}."
        )

    heading("NOTES")
    print(
        "- KNN distance is a distance in the existing feature space, "
        "not a probability."
    )
    print(
        "- Predicted market values are estimates, not guaranteed "
        "transfer fees or asking prices."
    )
    print(
        "- EA FC 24 attributes are display-only and do not affect "
        "KNN rankings or Model A predictions."
    )



def run_bargain_finder(query):
    """Run Bargain Finder for one player and return its report."""

    output = StringIO()

    with redirect_stdout(output):
        model, features, market_data = load_market_value_model()

        if market_data.empty:
            print(f"No eligible market value records for {TARGET_SEASON}.")
        else:
            target_player = select_target_player(query.strip())

            if target_player is None:
                print("Player not found or selection was ambiguous.")
            else:
                target_name = target_player.get("player_name")

                if not target_name:
                    print("The selected player record has no player name.")
                else:
                    target_value = predict_player_value(
                        target_name,
                        model,
                        features,
                        market_data,
                    )

                    if target_value is None:
                        print(
                            f"Model A cannot produce an eligible "
                            f"{TARGET_SEASON} valuation for {target_name}."
                        )
                    else:
                        display_target_player(target_player, target_value)

                        similar_players = get_similar_players(target_player)

                        if similar_players.empty:
                            print("No similar players were returned by KNN.")
                        else:
                            analyze_bargains(
                                target_player=target_player,
                                target_value=target_value,
                                similar_players=similar_players,
                                model=model,
                                features=features,
                                market_data=market_data,
                            )

    return output.getvalue()



def get_bargain_finder_data(query):
    """Return structured Bargain Finder data for the React frontend."""

    model, features, market_data = load_market_value_model()

    if market_data.empty:
        raise ValueError(f"No eligible market value records for {TARGET_SEASON}.")

    with redirect_stdout(StringIO()):
        target_player = select_target_player(query.strip())

    if target_player is None:
        raise ValueError("Player not found or selection was ambiguous.")

    target_name = target_player.get("player_name")

    if not target_name:
        raise ValueError("Selected player has no player name.")

    target_value = predict_player_value(
        target_name, model, features, market_data
    )

    if target_value is None:
        raise ValueError(
            f"Model A cannot produce an eligible {TARGET_SEASON} valuation."
        )

    similar_players = get_similar_players(target_player)

    def get_player_details(name, team):
        row = get_knn_performance_stats(name, team)

        performance = {}

        if row is not None:
            for label, columns in PERFORMANCE_COLUMNS.items():
                value = get_value(row, columns)

                if value is not None:
                    try:
                        if not pd.isna(value):
                            performance[label] = float(value)
                    except (TypeError, ValueError):
                        pass

        attributes = {}

        try:
            profile = knn.get_fc24_profile(name)
            if isinstance(profile, pd.Series):
                profile = profile.to_dict()

            if isinstance(profile, dict):
                normalized = {
                    re.sub(r"[^a-z0-9]+", "", str(k).lower()): v
                    for k, v in profile.items()
                }

                for label, aliases in FC24_ATTRIBUTE_KEYS.items():
                    for alias in aliases:
                        key = re.sub(r"[^a-z0-9]+", "", alias.lower())
                        value = normalized.get(key)

                        if value is not None and not pd.isna(value):
                            attributes[label] = float(value)
                            break
        except Exception:
            pass

        return {
            "performance": performance,
            "general_stats": attributes,
        }

    target_team = target_player.get("team_key", "Unknown")

    target = {
        "name": target_name,
        "team": str(target_team),
        "position": str(target_player.get("position", "Unknown")),
        "predicted_value": target_value["predicted_value"],
        **get_player_details(target_name, target_team),
    }

    alternatives = []

    for _, row in similar_players.iterrows():
        name = get_value(row, ["Player", "player_name", "short_name"])

        if name is None or normalize_name(name) == normalize_name(target_name):
            continue

        team = get_value(row, ["Team", "team_key", "team"], "Unknown")
        position = get_value(row, ["Position", "position"], "Unknown")
        distance = get_value(row, ["Distance", "distance"])

        comparable_value = predict_player_value(
            str(name), model, features, market_data
        )

        if comparable_value is None:
            continue

        predicted = comparable_value["predicted_value"]
        savings = target_value["predicted_value"] - predicted

        alternatives.append({
            "name": str(name),
            "team": str(team),
            "position": str(position),
            "predicted_value": predicted,
            "savings": savings,
            "cheaper_percent": (
                savings / target_value["predicted_value"] * 100
                if target_value["predicted_value"] > 0 else 0
            ),
            "knn_distance": (
                float(distance) if distance is not None else None
            ),
            **get_player_details(str(name), team),
        })

    alternatives.sort(key=lambda player: player["predicted_value"])

    return {
        "success": True,
        "target": target,
        "target_season": TARGET_SEASON,
        "alternatives": alternatives,
    }




# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():
    heading("TRANSFERIQ — BARGAIN FINDER")
    print("Loading existing KNN and Model A components...")

    model, features, market_data = load_market_value_model()

    if market_data.empty:
        print(f"No eligible market value records for {TARGET_SEASON}.")
        return

    print("Models loaded.")
    print(f"Valuation season: {TARGET_SEASON}")
    print(f"Eligible valuation records: {len(market_data)}")

    while True:
        query = input(
            "\nEnter a Premier League player name "
            "(or type 'exit' to quit): "
        ).strip()

        if query.lower() == "exit":
            print("TransferIQ closed.")
            break

        if not query:
            print("Please enter a player name.")
            continue

        try:
            target_player = select_target_player(query)
        except Exception as error:
            print(f"Could not select player: {error}")
            continue

        if target_player is None:
            print("Player not found or selection was ambiguous.")
            continue

        target_name = target_player.get("player_name")

        if not target_name:
            print("The selected player record has no player name.")
            continue

        target_value = predict_player_value(
            target_name,
            model,
            features,
            market_data,
        )

        if target_value is None:
            print(
                f"Model A cannot produce an eligible "
                f"{TARGET_SEASON} valuation for {target_name}."
            )
            print("Try another player.")
            continue

        display_target_player(target_player, target_value)

        try:
            similar_players = get_similar_players(target_player)
        except Exception as error:
            print(f"Could not retrieve KNN comparables: {error}")
            continue

        if similar_players.empty:
            print("No similar players were returned by KNN.")
            continue

        analyze_bargains(
            target_player=target_player,
            target_value=target_value,
            similar_players=similar_players,
            model=model,
            features=features,
            market_data=market_data,
        )


if __name__ == "__main__":
    main()


