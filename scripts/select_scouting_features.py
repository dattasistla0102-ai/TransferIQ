
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "players_2023_24_combined.csv"
)

df = pd.read_csv(DATA_PATH)

print("========== TRANSFERIQ SCOUTING FEATURES ==========")
print("Dataset shape:", df.shape)

# Show columns that have usable data for a meaningful portion
# of the player dataset.
coverage = pd.DataFrame({
    "feature": df.columns,
    "non_missing": df.notna().sum().values,
    "coverage_percent": (
        df.notna().mean().values * 100
    ).round(1),
    "data_type": df.dtypes.astype(str).values,
})

print("\n========== PLAYER IDENTITY COLUMNS ==========")
identity_terms = [
    "player", "name", "team", "club", "position",
    "nation", "birth", "age"
]

identity = coverage[
    coverage["feature"].str.lower().apply(
        lambda col: any(term in col for term in identity_terms)
    )
]

print(identity.to_string(index=False))

print("\n========== BEST-COVERED NUMERIC FEATURES ==========")
numeric_columns = df.select_dtypes(include="number").columns

numeric_coverage = coverage[
    coverage["feature"].isin(numeric_columns)
].sort_values("coverage_percent", ascending=False)

print(numeric_coverage.head(50).to_string(index=False))

print("\n========== LOW-COVERAGE NUMERIC FEATURES ==========")
print(
    numeric_coverage[
        numeric_coverage["coverage_percent"] < 50
    ].to_string(index=False)
)

print("\nNo datasets were modified.")
