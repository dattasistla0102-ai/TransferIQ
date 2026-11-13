
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "data" / "processed" / "transfer_history_all_seasons.csv"

df = pd.read_csv(path)

print("=== MOVEMENT VALUES ===")
if "movement" in df.columns:
    print(df["movement"].value_counts(dropna=False).to_string())
else:
    print("No movement column found.")

print("\n=== ISAK RECORDS ===")
if "player_name" in df.columns:
    isak = df[
        df["player_name"].astype(str).str.contains(
            "Alexander Isak", case=False, na=False
        )
    ]
    cols = [
        c for c in [
            "season", "player_name", "player_id", "movement",
            "club", "dealing_club", "fee", "market_value", "window"
        ] if c in df.columns
    ]
    print(isak[cols].to_string(index=False))

print("\n=== POTENTIAL MIRRORED TRANSFERS ===")
if {"season", "player_name", "club", "dealing_club", "fee"} <= set(df.columns):
    paid = df[pd.to_numeric(df["fee"], errors="coerce") > 0].copy()
    paid["fee"] = pd.to_numeric(paid["fee"], errors="coerce")

    # Same player, season and fee appearing with the clubs reversed.
    paid["club_pair"] = paid.apply(
        lambda r: " | ".join(sorted([
            str(r["club"]), str(r["dealing_club"])
        ])),
        axis=1,
    )

    counts = paid.groupby(
        ["season", "player_name", "fee", "club_pair"],
        dropna=False
    ).size().reset_index(name="record_count")

    print(
        counts[counts["record_count"] > 1]
        .sort_values("record_count", ascending=False)
        .head(25)
        .to_string(index=False)
    )
