
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "processed" / "market_value_dataset.csv"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

BASE_URL = (
    "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/"
)

con = duckdb.connect()
con.execute("INSTALL httpfs")
con.execute("LOAD httpfs")

competitions = f"read_csv_auto('{BASE_URL}competitions.csv.gz')"
games = f"read_csv_auto('{BASE_URL}games.csv.gz')"
appearances = f"read_csv_auto('{BASE_URL}appearances.csv.gz')"
valuations = f"read_csv_auto('{BASE_URL}player_valuations.csv.gz')"
players = f"read_csv_auto('{BASE_URL}players.csv.gz')"

print("Building Premier League-only market value dataset...")

query = f"""
WITH pl_games AS (
    SELECT
        game_id,
        season,
        CAST(date AS DATE) AS game_date
    FROM {games}
    WHERE competition_id = 'GB1'
),
season_stats AS (
    SELECT
        pg.season,
        a.player_id,
        SUM(a.minutes_played) AS minutes_played,
        SUM(a.goals) AS goals,
        SUM(a.assists) AS assists,
        SUM(a.yellow_cards) AS yellow_cards,
        SUM(a.red_cards) AS red_cards,
        COUNT(DISTINCT a.game_id) AS appearances,
        MAX(pg.game_date) AS last_match_date
    FROM pl_games pg
    JOIN {appearances} a
        ON a.game_id = pg.game_id
    GROUP BY pg.season, a.player_id
),
season_dates AS (
    SELECT
        season,
        MIN(game_date) AS season_start,
        MAX(game_date) AS season_end
    FROM pl_games
    GROUP BY season
),
season_valuations AS (
    SELECT
        s.season,
        s.player_id,
        s.minutes_played,
        s.goals,
        s.assists,
        s.yellow_cards,
        s.red_cards,
        s.appearances,
        s.last_match_date,
        sd.season_start,
        sd.season_end,
        v.market_value_in_eur,
        v.date AS valuation_date,
        ROW_NUMBER() OVER (
            PARTITION BY s.season, s.player_id
            ORDER BY v.date DESC
        ) AS valuation_rank
    FROM season_stats s
    JOIN season_dates sd
        ON sd.season = s.season
    JOIN {valuations} v
        ON v.player_id = s.player_id
       AND CAST(v.date AS DATE) <= sd.season_start
)
SELECT
    sv.season,
    sv.player_id,
    p.name AS player_name,
    p.position,
    p.sub_position,
    p.date_of_birth,
    sv.minutes_played,
    sv.goals,
    sv.assists,
    sv.yellow_cards,
    sv.red_cards,
    sv.appearances,
    sv.market_value_in_eur,
    sv.valuation_date,
    sv.season_start,
    sv.season_end
FROM season_valuations sv
JOIN {players} p
    ON p.player_id = sv.player_id
WHERE sv.valuation_rank = 1
ORDER BY sv.season, sv.player_id
"""

df = con.execute(query).df()
con.close()

df.to_csv(OUTPUT, index=False)

print("\nDATASET CREATED")
print(f"Rows: {len(df):,}")
print(f"Unique players: {df['player_id'].nunique():,}")
print(f"Seasons: {df['season'].nunique():,}")
print(f"Saved to: {OUTPUT}")

print("\nRows per season:")
print(df.groupby("season").size().to_string())

print("\nSample:")
print(df.head(10).to_string(index=False))

print("\nMissing values:")
print(df.isna().sum().to_string())

