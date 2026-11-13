import joblib
import pandas as pd

from scripts.demo_market_value_prediction import (
    prepare_data,
    load_fc24_data,
    get_fc24_profile,
    get_fc24_profile_from_df,
    normalize_name,
    TARGET_SEASON,
    MODEL_PATH,
)

from pathlib import Path
import sys

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PROJECT_DIR = Path(__file__).resolve().parent

# Allow Python to import your existing scripts.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.build_knn_scout import find_similar_players_for_api

from scripts.bargain_finder import run_bargain_finder

from scripts.bargain_finder import get_bargain_finder_data
from pathlib import Path

app = FastAPI(title="TransferIQ API")

# React development server access.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "TransferIQ"}


@app.get("/api/scout")
def scout(player_name: str = Query(min_length=1, max_length=100)):
    try:
        result = find_similar_players_for_api(
            player_name.strip(),
            top_n=7,
        )

        if not result["success"]:
            return JSONResponse(
                status_code=404,
                content=result,
            )

        return result

    except Exception as exc:
        # Details are printed in the backend terminal for debugging.
        print(f"Player Scout error: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": (
                    "The scouting request failed. "
                    "Check the backend terminal for details."
                ),
                "players": [],
            },
        )
# ============================================================
# MARKET VALUE PREDICTION
# ============================================================

market_artifact = joblib.load(MODEL_PATH)

market_model = market_artifact["model"]
market_features = market_artifact["features"]

market_df = prepare_data()
market_season_df = market_df[
    market_df["season"] == TARGET_SEASON
].copy()

fc24_df, fc24_lookup = load_fc24_data()




def get_display_attributes(player_name, profile):
    """Return only the relevant FC 24 attributes for display."""

    if not profile:
        return {}

    positions = str(profile.get("FC 24 positions", "")).upper()
    is_goalkeeper = "GK" in positions.split(", ")

    if is_goalkeeper:
        return {
            "Diving": profile.get("goalkeeping_diving", "N/A"),
            "Handling": profile.get("goalkeeping_handling", "N/A"),
            "Kicking": profile.get("goalkeeping_kicking", "N/A"),
            "Reflexes": profile.get("goalkeeping_reflexes", "N/A"),
            "Speed": profile.get("movement_sprint_speed", "N/A"),
            "Position": profile.get("goalkeeping_positioning", "N/A"),
        }

    return {
        "Pace": profile.get("Pace", "N/A"),
        "Shooting": profile.get("Shooting", "N/A"),
        "Passing": profile.get("Passing", "N/A"),
        "Dribbling": profile.get("Dribbling", "N/A"),
        "Defending": profile.get("Defending", "N/A"),
        "Physical": profile.get("Physicality", "N/A"),
    }




@app.get("/api/market-value")
def predict_market_value(player_name: str):
    query = player_name.strip()

    if not query:
        return {"error": "Enter a player name."}

    matches = market_season_df[
        market_season_df["player_name"].str.contains(
            query,
            case=False,
            na=False,
            regex=False,
        )
    ]

    if matches.empty:
        return {
            "error": (
                f"No eligible player found for target season "
                f"{TARGET_SEASON}."
            )
        }

    unique_names = matches["player_name"].drop_duplicates()

    if len(unique_names) > 1:
        return {
            "error": "Multiple players found. Enter a more specific name.",
            "matches": unique_names.tolist(),
        }

    player = matches.iloc[[0]]
    player_name = str(player["player_name"].iloc[0])

    # Predict using the saved model and its original features.
    predicted_value = float(
        market_model.predict(player[market_features])[0]
    )

    # Load EA FC 24 attributes for display only.
    profile = get_fc24_profile(player_name, fc24_lookup)

    # Never return actual market value or recorded-value comparisons.
    return {
        "player_name": player_name,
        "target_season": int(TARGET_SEASON),
        "predicted_market_value_eur": max(0, round(predicted_value)),
        "club": profile.get("FC 24 club", "N/A") if profile else "N/A",
        "positions": (
            profile.get("FC 24 positions", "N/A")
            if profile else "N/A"
        ),
        "overall": profile.get("Overall rating", "N/A") if profile else "N/A",
        "player_id": (
            profile.get("player_id", "N/A")
            if profile else "N/A"
        ),
        "attributes": get_display_attributes(player_name, profile),
    }



@app.get("/api/bargain")
def bargain_finder(player_name: str = Query(min_length=1, max_length=100)):
    try:
        result = get_bargain_finder_data(player_name.strip())
        return result

    except ValueError as exc:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": str(exc),
            },
        )

    except Exception as exc:
        print(f"Bargain Finder error: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Bargain Finder failed. Check the backend terminal.",
            },
        )


# ============================================================
# PERFORMANCE LAB
# ============================================================

PERFORMANCE_LEADERBOARD_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "performance_leaderboards.csv"
)


@app.get("/api/performance")
def get_performance_data():
    try:
        if not PERFORMANCE_LEADERBOARD_PATH.exists():
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": (
                        "Performance data not found. "
                        "Run build_performance_charts.py first."
                    ),
                },
            )

        df = pd.read_csv(PERFORMANCE_LEADERBOARD_PATH)

        leaderboards = {}

        for metric, group in df.groupby("metric", sort=False):
            leaderboards[metric] = (
                group[
                    ["rank", "player_name", "team", "value"]
                ]
                .to_dict(orient="records")
            )

        return {
            "success": True,
            "season": "2023/24",
            "leaderboards": leaderboards,
        }

    except Exception as exc:
        print(f"Performance Lab error: {exc}")

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Performance Lab failed. Check the backend terminal.",
            },
        )



# ============================================================
# UNIVERSAL PLAYER PROFILE
# Actual Premier League 2023/24 statistics only
# ============================================================

PLAYER_STATS_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "players_2023_24_combined.csv"
)

# Load the real performance dataset independently.
player_stats_df = pd.read_csv(PLAYER_STATS_PATH)

# Each display label is mapped to its original dataset column.
# No values are predicted or calculated here.
PLAYER_PROFILE_STATS = {
    "Overview": {
        "Appearances": "player_player_ratings__Matches",
        "Minutes Played": "player_player_ratings__Minutes",
        "FotMob Rating": "player_player_ratings__FotMob Rating",
        "Player of the Match Awards": "player_player_ratings__Player of the Match Awards",
    },

    "Attacking": {
        "Goals": "player_top_scorers__Goals",
        "Assists": "player_top_assists__Assists",
        "Expected Goals (xG)": "player_expected_goals__Expected Goals (xG)",
        "Expected Goals on Target (xGOT)": "player_expected_goals_on_target__Expected Goals on Target (xGOT)",
        "Goals per 90": "player_goals_per_90__Goals per 90",
        "Expected Goals per 90": "player_expected_goals_per_90__Expected Goals per 90",
        "Shots per 90": "player_total_scoring_attempts__Shots per 90",
        "Shots on Target per 90": "player_on_target_scoring_attempts__Shots on Target per 90",
        "Shot Accuracy (%)": "player_on_target_scoring_attempts__Shot Accuracy (%)",
        "Shot Conversion Rate (%)": "player_total_scoring_attempts__Shot Conversion Rate (%)",
        "Big Chances Missed": "player_big_chances_missed__Big Chances Missed",
        "Penalties Scored": "player_top_scorers__Penalties",
    },

    "Creativity": {
        "Expected Assists (xA)": "player_expected_assists__Expected Assists (xA)",
        "Expected Assists per 90": "player_expected_assists_per_90__Expected Assists per 90",
        "Chances Created": "player_total_assists_in_attack__Chances Created",
        "Chances Created per 90": "player_total_assists_in_attack__Chances Created per 90",
        "Secondary Assists": "player_top_assists__Secondary Assists",
        "Big Chances Created": "player_big_chances_created__Big Chances Created",
    },

    "Passing": {
        "Pass Success (%)": "player_accurate_passes__Pass Success (%)",
        "Accurate Passes per 90": "player_accurate_passes__Accurate Passes per 90",
        "Accurate Long Balls per 90": "player_accurate_long_balls__Accurate Long Balls per 90",
        "Successful Long Balls (%)": "player_accurate_long_balls__Successful Long Balls (%)",
    },

    "Dribbling": {
        "Successful Dribbles per 90": "player_contests_won__Successful Dribbles per 90",
        "Dribble Success Rate (%)": "player_contests_won__Dribble Success Rate (%)",
    },

    "Defending": {
        "Tackles per 90": "player_tackles_won__Tackles per 90",
        "Tackle Success Rate (%)": "player_tackles_won__Tackle Success Rate (%)",
        "Interceptions per 90": "player_interceptions__Interceptions per 90",
        "Total Interceptions": "player_interceptions__Total Interceptions",
        "Clearances per 90": "player_effective_clearances__Clearances per 90",
        "Total Clearances": "player_effective_clearances__Total Clearances",
        "Blocks per 90": "player_outfielder_blocks__Blocks per 90",
        "Total Blocks": "player_outfielder_blocks__Total Blocks",
        "Possessions Won in Final Third per 90": "player_possessions_won_attacking_third__Possessions Won in Final 3rd per 90",
        "Possessions Won in Midfield per 90": "player_possessions_won_attacking_third__Possessions Won Midfield per 90",
    },

    "Goalkeeping": {
        "Saves per 90": "player_saves_made__Saves per 90",
        "Total Saves": "player_saves_made__Total Saves",
        "Clean Sheets": "player_clean_sheets__Clean Sheets",
        "Goals Conceded per 90": "player_goals_conceded__Goals Conceded per 90",
        "Total Goals Conceded": "player_goals_conceded__Total Goals Conceded",
    },

    "Discipline": {
        "Yellow Cards": "player_yellow_cards__Yellow Cards",
        "Red Cards": "player_red_cards__Red Cards",
        "Fouls Committed per 90": "player_fouls_committed__Fouls Committed per 90",
        "Penalties Conceded": "player_penalties_conceded__Penalties Conceded",
        "Penalties Won": "player_penalties_won__Penalties Won",
        "Fouls Won per 90": "player_penalties_won__Fouls Won per 90",
    },
}


@app.get("/api/player-profile")
def get_player_profile(
    player_name: str = Query(min_length=1, max_length=100),
    team: str = Query(min_length=1, max_length=100),
):
    try:
        # Normalize the requested player and club names.
        name_key = normalize_name(player_name)
        team_key = normalize_name(team)

        # Match using BOTH player name and team.
        matches = player_stats_df[
            (player_stats_df["name_key"] == name_key)
            & (player_stats_df["team_key"] == team_key)
        ]

        if matches.empty:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": "Player not found for the selected team in the 2023/24 dataset.",
                },
            )

        if len(matches) > 1:
            return JSONResponse(
                status_code=409,
                content={
                    "success": False,
                    "message": "Multiple records found for this player and team.",
                },
            )

        player = matches.iloc[0]


        # Get the player's FC24 profile for their image.
        fc24_profile = get_fc24_profile_from_df(
            str(player["player_name"]),
            fc24_df,
            club_name=str(player["team_key"])
        )

        player_id = (
            fc24_profile.get("player_id")
            if fc24_profile
            else None
        )

        statistics = {}

        for category, metrics in PLAYER_PROFILE_STATS.items():
            category_stats = {}

            for label, column in metrics.items():
                if column not in player.index:
                    continue

                value = player[column]

                # Skip missing values; never invent statistics.
                if pd.isna(value):
                    continue

                # Convert NumPy/Pandas numbers into JSON-safe values.
                category_stats[label] = float(value)

            if category_stats:
                statistics[category] = category_stats

        return {
            "success": True,
            "season": "2023/24",
            "player_name": str(player["player_name"]),
            "team": str(player["team_key"]),
            "position": (
                str(player["position"])
                if pd.notna(player["position"])
                else "N/A"
            ),
            "player_id": str(player_id) if player_id else None,
            "statistics": statistics,
        }

    except Exception as exc:
        print(f"Player Profile error: {exc}")

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Could not load player profile. Check the backend terminal.",
            },
        )
