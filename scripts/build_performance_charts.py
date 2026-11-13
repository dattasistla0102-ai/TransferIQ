
from pathlib import Path
import re
import sys

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# TransferIQ - Feature 4: Performance Analytics
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "players_2023_24_combined.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "performance_charts"
)

TOP_N = 10
MIN_MATCHES_FOR_RATING = 10


# Each metric: chart title, dataset column, aggregation method, unit
METRICS = [
    {
        "title": "Top Goal Scorers",
        "column": "player_top_scorers__Goals",
        "aggregation": "sum",
        "unit": "Goals",
    },
    {
        "title": "Top Assist Providers",
        "column": "player_top_assists__Assists",
        "aggregation": "sum",
        "unit": "Assists",
    },
    {
        "title": "Highest-Rated Players",
        "column": "player_player_ratings__FotMob Rating",
        "aggregation": "mean",
        "unit": "FotMob Rating",
    },
    {
        "title": "Expected Goals (xG)",
        "column": "player_expected_goals__Expected Goals (xG)",
        "aggregation": "sum",
        "unit": "Expected Goals",
    },
    {
        "title": "Expected Assists (xA)",
        "column": "player_expected_assists__Expected Assists (xA)",
        "aggregation": "sum",
        "unit": "Expected Assists",
    },
    {
        "title": "Successful Dribbles per 90",
        "column": "player_contests_won__Successful Dribbles per 90",
        "aggregation": "mean",
        "unit": "Dribbles per 90",
    },
    {
        "title": "Pass Success Percentage",
        "column": "player_accurate_passes__Pass Success (%)",
        "aggregation": "mean",
        "unit": "Pass Success (%)",
    },
    {
        "title": "Tackles per 90",
        "column": "player_tackles_won__Tackles per 90",
        "aggregation": "mean",
        "unit": "Tackles per 90",
    },
    {
        "title": "Interceptions per 90",
        "column": "player_interceptions__Interceptions per 90",
        "aggregation": "mean",
        "unit": "Interceptions per 90",
    },
    {
        "title": "Clearances per 90",
        "column": "player_effective_clearances__Clearances per 90",
        "aggregation": "mean",
        "unit": "Clearances per 90",
    },
]


def load_data():
    """Load the combined Premier League player dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}\n"
            "Check that the CSV exists in data/processed/."
        )

    df = pd.read_csv(DATA_PATH)

    # Use readable names when available; otherwise fall back to name_key.
    if "player_name" in df.columns:
        name_column = "player_name"
    elif "name_key" in df.columns:
        name_column = "name_key"
    else:
        raise KeyError(
            "Could not find player_name or name_key in the dataset.\n"
            f"Available columns:\n{df.columns.tolist()}"
        )

    if "team_key" not in df.columns:
        raise KeyError("The dataset is missing the team_key column.")

    df = df.copy()
    df["display_name"] = df[name_column].astype("string").str.strip()
    df["display_team"] = df["team_key"].astype("string").str.strip()

    df = df[
        df["display_name"].notna()
        & (df["display_name"] != "")
    ].copy()

    print("\n" + "=" * 65)
    print("TRANSFERIQ - PERFORMANCE ANALYTICS")
    print("=" * 65)
    print(f"Dataset: {DATA_PATH.name}")
    print(f"Players/records loaded: {len(df)}")
    print(f"Player-name column: {name_column}")

    return df


def prepare_metric_data(df, metric):
    """Prepare the top players for one metric."""

    column = metric["column"]

    if column not in df.columns:
        print(f"\nSkipping '{metric['title']}'")
        print(f"Missing dataset column: {column}")
        return None

    data = df[
        ["display_name", "display_team", column]
    ].copy()

    data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=[column])

    # Rating comparisons should exclude players with very few matches.
    if metric["title"] == "Highest-Rated Players":
        matches_column = "player_player_ratings__Matches"

        if matches_column in df.columns:
            matches = pd.to_numeric(
                df[matches_column], errors="coerce"
            )
            data = data.loc[
                matches.reindex(data.index).fillna(0)
                >= MIN_MATCHES_FOR_RATING
            ]

    if data.empty:
        print(f"\nSkipping '{metric['title']}': no usable values.")
        return None

    # Combine duplicate player/team rows if the source contains any.
    if metric["aggregation"] == "sum":
        data = (
            data.groupby(
                ["display_name", "display_team"],
                as_index=False,
                dropna=False,
            )[column]
            .sum()
        )
    else:
        data = (
            data.groupby(
                ["display_name", "display_team"],
                as_index=False,
                dropna=False,
            )[column]
            .mean()
        )

    # Highest values first.
    data = data.sort_values(column, ascending=False)
    data = data.head(TOP_N).reset_index(drop=True)

    if data.empty:
        return None

    return data


def safe_filename(title):
    """Convert a chart title into a safe filename."""

    filename = re.sub(r"[^a-zA-Z0-9]+", "_", title.lower())
    return filename.strip("_") + ".png"


def create_line_chart(data, metric):
    """Plot and save one line chart."""

    title = metric["title"]
    column = metric["column"]
    unit = metric["unit"]

    # Keep labels short enough to read on the chart.
    labels = data["display_name"].astype(str).tolist()
    values = data[column].tolist()

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        range(len(labels)),
        values,
        marker="o",
        markersize=7,
        linewidth=2,
    )

    ax.set_title(
        f"{title} - Premier League 2023/24",
        fontsize=15,
        pad=15,
    )
    ax.set_xlabel("Players", fontsize=11)
    ax.set_ylabel(unit, fontsize=11)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")

    ax.grid(True, linestyle="--", alpha=0.4)

    # Add each player's statistic beside their point.
    for index, value in enumerate(values):
        ax.annotate(
            f"{value:.2f}" if isinstance(value, float) else str(value),
            (index, value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    fig.tight_layout()

    output_path = OUTPUT_DIR / safe_filename(title)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")

    print(f"Saved chart: {output_path.relative_to(PROJECT_ROOT)}")

    # Keep the figure open so plt.show() displays it later.
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = load_data()
    except (FileNotFoundError, KeyError) as error:
        print(f"\nERROR: {error}")
        sys.exit(1)

    leaderboard_tables = []
    figures = []
    charts_created = 0

    for metric in METRICS:
        data = prepare_metric_data(df, metric)

        if data is None:
            continue

        # Store each metric in a combined CSV.
        leaderboard = data.copy()
        leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))
        leaderboard.insert(1, "metric", metric["title"])
        leaderboard = leaderboard.rename(
            columns={
                "display_name": "player_name",
                "display_team": "team",
                metric["column"]: "value",
            }
        )
        leaderboard_tables.append(leaderboard)

        # Print a short preview in the terminal.
        print(f"\n{metric['title']}")
        print("-" * 50)
        print(
            leaderboard[
                ["rank", "player_name", "team", "value"]
            ].to_string(index=False)
        )

        figures.append(create_line_chart(data, metric))
        charts_created += 1

    if leaderboard_tables:
        leaderboard_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "performance_leaderboards.csv"
        )

        pd.concat(leaderboard_tables, ignore_index=True).to_csv(
            leaderboard_path, index=False
        )

        print(f"\nLeaderboard CSV saved: {leaderboard_path}")

    print("\n" + "=" * 65)
    print(f"CHARTS CREATED: {charts_created}")
    print(f"CHART IMAGE FOLDER: {OUTPUT_DIR}")
    print("=" * 65)

    if figures:
        print("\nOpening charts... Close the chart windows to finish.")
        plt.show()
    else:
        print(
            "\nNo charts were created. Check the missing-column messages above."
        )


if __name__ == "__main__":
    main()

