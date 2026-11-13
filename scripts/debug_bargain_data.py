
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_knn_scout as knn

print("\n========== KNN DATASET COLUMNS ==========")
print(knn.df.columns.tolist())

print("\n========== FC24 DATASET COLUMNS ==========")

fc24_path = (
    ROOT
    / "data"
    / "processed"
    / "ea_fc24_premier_league_players.csv"
)

fc24 = pd.read_csv(fc24_path, nrows=5)
print(fc24.columns.tolist())

print("\n========== BRUNO FERNANDES FC24 MATCH ==========")

if hasattr(knn, "get_fc24_profile"):
    profile = knn.get_fc24_profile("Bruno Fernandes")
    print(profile)
else:
    print("get_fc24_profile function not found.")

print("\n========== PERFORMANCE DATA SAMPLE ==========")

if "_search_name" in knn.df.columns:
    matches = knn.df[
        knn.df["_search_name"].astype(str).str.contains(
            "bruno fernandes",
            case=False,
            na=False,
        )
    ]

    print(matches.head(3).to_string(index=False))
else:
    print("The _search_name column was not found.")
