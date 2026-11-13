
from pathlib import Path
import pandas as pd
import json

# Project paths
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "kaggle_fc24" / "male_players.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT = OUTPUT_DIR / "ea_fc24_premier_league_players.csv"
REPORT = OUTPUT_DIR / "ea_fc24_premier_league_report.json"

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Source file not found: {SOURCE}\n"
            "Check that male_players.csv was downloaded from Kaggle."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading EA SPORTS FC 24 player data...")
    df = pd.read_csv(SOURCE, low_memory=False)

    required = {
        "player_id",
        "fifa_version",
        "fifa_update",
        "update_as_of",
        "league_name",
        "club_name",
        "player_positions",
        "overall",
        "potential",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")

    source_rows = len(df)

    # Keep only EA SPORTS FC 24 records.
    df = df[df["fifa_version"] == 24].copy()

    if df.empty:
        raise ValueError("No records found for fifa_version = 24.")

    # Select each player's latest available FC 24 update first.
    # This avoids keeping multiple versions of the same player.
    df["update_as_of"] = pd.to_datetime(
        df["update_as_of"], errors="coerce"
    )

    df = df.sort_values(
        ["fifa_update", "update_as_of"],
        na_position="first",
    )

    df = df.drop_duplicates(subset=["player_id"], keep="last")

    # Use the league recorded in the EA FC 24 source itself.
    df = df[
        df["league_name"].astype("string").str.strip()
        == "Premier League"
    ].copy()

    # Exclude invalid player identifiers.
    df = df[df["player_id"].notna()].copy()

    if df.empty:
        raise ValueError(
            "No Premier League players found after filtering."
        )

    # Keep all source columns, including detailed attributes,
    # player traits, player tags and position-rating fields.
    df = df.sort_values(
        ["club_name", "overall", "long_name"],
        ascending=[True, False, True],
        na_position="last",
    )

    # Report checks
    duplicate_ids = int(df["player_id"].duplicated().sum())
    if duplicate_ids:
        raise ValueError(
            f"Duplicate player IDs remain: {duplicate_ids}"
        )

    goalkeepers = (
        df["player_positions"]
        .fillna("")
        .str.split(",")
        .apply(lambda positions: "GK" in [p.strip() for p in positions])
    )

    # Image URLs are not present as a dedicated face-image field
    # in the inspected male_players.csv schema.
    image_columns = [
        col for col in df.columns
        if "image_url" in col.lower()
        or "face_url" in col.lower()
        or "player_face" in col.lower()
    ]

    report = {
        "source_file": str(SOURCE.relative_to(ROOT)),
        "output_file": str(OUTPUT.relative_to(ROOT)),
        "source_rows_all_versions": source_rows,
        "source_rows_fc24": int(
            pd.read_csv(
                SOURCE,
                usecols=["fifa_version"],
            )["fifa_version"].eq(24).sum()
        ),
        "players_after_latest_update_deduplication": int(
            len(df)
        ),
        "premier_league_players": int(len(df)),
        "goalkeepers": int(goalkeepers.sum()),
        "outfield_players": int((~goalkeepers).sum()),
        "column_count": int(len(df.columns)),
        "duplicate_player_ids": duplicate_ids,
        "missing_values_by_column": {
            col: int(count)
            for col, count in df.isna().sum().items()
            if count > 0
        },
        "image_url_columns_found": image_columns,
        "image_url_note": (
            "No dedicated face-image URL field was found "
            "in the inspected source schema."
        ),
        "filtering": (
            "fifa_version = 24; retain latest available update "
            "per player_id; then filter league_name = "
            "'Premier League'."
        ),
    }

    # Save the complete player dataset.
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    with open(REPORT, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    # Reopen the saved CSV to verify the deliverable.
    saved = pd.read_csv(OUTPUT, low_memory=False)

    assert len(saved) == len(df), "Saved row count does not match."
    assert saved["player_id"].is_unique, "Player IDs are not unique."
    assert saved["league_name"].eq("Premier League").all()
    assert saved["fifa_version"].eq(24).all()

    print("\nDATASET CREATED AND VERIFIED")
    print(f"Output: {OUTPUT}")
    print(f"Report: {REPORT}")
    print(f"Players: {len(saved):,}")
    print(f"Goalkeepers: {int(goalkeepers.sum()):,}")
    print(f"Outfield players: {int((~goalkeepers).sum()):,}")
    print(f"Columns: {len(saved.columns)}")
    print(f"Duplicate IDs: {saved['player_id'].duplicated().sum()}")
    print(f"Image URL columns: {image_columns}")
    print("\nFirst 5 players:")
    display_columns = [
        col for col in [
            "player_id", "long_name", "club_name",
            "player_positions", "overall", "potential"
        ]
        if col in saved.columns
    ]
    print(saved[display_columns].head().to_string(index=False))

if __name__ == "__main__":
    main()
