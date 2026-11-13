
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "data" / "raw" / "historical_player_stats"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "imadeddine-belkat/Premier-League-Stats/"
    "main/fpl_scraper/fpl_stats/_merged/players/"
)

SEASONS = [
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
]

print("=" * 65)
print("TRANSFERIQ: HISTORICAL PLAYER STATISTICS")
print("=" * 65)

successful = []
failed = []

for season in SEASONS:
    filename = f"{season}_all_players_gw.csv"
    url = BASE_URL + filename

    print(f"\nDownloading season {season}...")

    try:
        df = pd.read_csv(url)

        if df.empty:
            raise ValueError("Downloaded dataset is empty.")

        output_path = OUTPUT_DIR / filename
        df.to_csv(output_path, index=False)

        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"Saved: {output_path}")

        successful.append((season, len(df), len(df.columns)))

    except Exception as error:
        print(f"FAILED: {season}")
        print(f"Reason: {error}")
        failed.append((season, str(error)))

print("\n" + "=" * 65)
print("DOWNLOAD SUMMARY")
print("=" * 65)

for season, rows, columns in successful:
    print(f"{season}: {rows} rows, {columns} columns")

if failed:
    print("\nFailed seasons:")
    for season, reason in failed:
        print(f"{season}: {reason}")

print(f"\nSuccessful seasons: {len(successful)}/{len(SEASONS)}")
print(f"Output folder: {OUTPUT_DIR}")

if successful:
    sample_path = OUTPUT_DIR / f"{successful[0][0]}_all_players_gw.csv"
    sample = pd.read_csv(sample_path, nrows=5)

    print("\nSample columns:")
    print(sample.columns.tolist())

print("\nDownload and inspection complete.")
