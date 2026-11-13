
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "players_2023_24_combined.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "feature_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_COVERAGE = 0.35
MIN_CORRELATION_SAMPLES = 30
HIGH_CORRELATION = 0.85


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 75)
print("TRANSFERIQ - POSITION-WISE FEATURE EVALUATION")
print("=" * 75)

print(f"\nDataset shape: {df.shape}")
print(f"Output folder: {OUTPUT_DIR}")

if "position" not in df.columns:
    raise ValueError("The dataset does not contain a 'position' column.")

df["position"] = df["position"].astype("string").str.strip().str.lower()

position_aliases = {
    "forward": "Forward",
    "midfielder": "Midfielder",
    "defender": "Defender",
    "goalkeeper": "Goalkeeper",
}

df["position_group"] = df["position"].map(position_aliases)

print("\nPlayers available by position:")
print(
    df["position_group"]
    .value_counts(dropna=False)
    .to_string()
)


# --------------------------------------------------
# 3. IDENTIFY NUMERIC PERFORMANCE FEATURES
# --------------------------------------------------

numeric_columns = df.select_dtypes(include="number").columns.tolist()

# These are useful for understanding data quality, but should not
# directly influence player similarity as ordinary performance stats.
excluded_words = [
    "rank",
    "minutes",
    "matches",
    "season",
    "shirt",
    "birth",
    "year",
    "age",
]

candidate_features = []

for col in numeric_columns:
    lower = col.lower()

    if any(word in lower for word in excluded_words):
        continue

    # Exclude columns that are entirely empty.
    if df[col].notna().sum() == 0:
        continue

    candidate_features.append(col)

print(f"\nNumeric performance candidates: {len(candidate_features)}")


# --------------------------------------------------
# 4. FEATURE COVERAGE AND VARIATION
# --------------------------------------------------

coverage_reports = []
usable_features = {}

for position in [
    "Forward",
    "Midfielder",
    "Defender",
    "Goalkeeper",
]:
    group = df[df["position_group"] == position].copy()
    n_players = len(group)

    rows = []
    accepted = []

    for feature in candidate_features:
        values = pd.to_numeric(group[feature], errors="coerce")

        non_missing = int(values.notna().sum())
        coverage = non_missing / n_players if n_players else 0.0
        unique_values = int(values.nunique(dropna=True))

        std = values.std()
        has_variation = pd.notna(std) and std > 0

        # Coverage is a first-pass screening threshold, not proof
        # that the feature is good.
        usable = (
            coverage >= MIN_COVERAGE
            and unique_values >= 5
            and has_variation
        )

        if usable:
            accepted.append(feature)

        rows.append({
            "position": position,
            "feature": feature,
            "players_in_position": n_players,
            "non_missing": non_missing,
            "coverage_percent": round(coverage * 100, 1),
            "unique_values": unique_values,
            "standard_deviation": (
                round(float(std), 4) if pd.notna(std) else np.nan
            ),
            "passes_initial_screen": usable,
        })

    coverage_reports.extend(rows)
    usable_features[position] = accepted

coverage_df = pd.DataFrame(coverage_reports)

coverage_path = OUTPUT_DIR / "feature_coverage_by_position.csv"
coverage_df.to_csv(coverage_path, index=False)

print("\nSaved feature coverage report:")
print(coverage_path)


# --------------------------------------------------
# 5. DISPLAY THE BEST-COVERED CANDIDATES
# --------------------------------------------------

for position in [
    "Forward",
    "Midfielder",
    "Defender",
    "Goalkeeper",
]:
    print("\n" + "=" * 75)
    print(f"{position.upper()} - TOP FEATURE CANDIDATES")
    print("=" * 75)

    report = coverage_df[
        (coverage_df["position"] == position)
        & (coverage_df["passes_initial_screen"])
    ].copy()

    report = report.sort_values(
        ["coverage_percent", "unique_values"],
        ascending=[False, False],
    )

    print(f"Players in group: {len(df[df['position_group'] == position])}")
    print(f"Features passing initial screen: {len(report)}")

    if report.empty:
        print("No features passed the current screening rules.")
        continue

    print(
        report[
            [
                "feature",
                "coverage_percent",
                "unique_values",
                "standard_deviation",
            ]
        ].head(30).to_string(index=False)
    )


# --------------------------------------------------
# 6. FIND HIGHLY CORRELATED / POTENTIALLY REDUNDANT FEATURES
# --------------------------------------------------

correlation_reports = []

for position in [
    "Forward",
    "Midfielder",
    "Defender",
    "Goalkeeper",
]:
    group = df[df["position_group"] == position].copy()
    features = usable_features[position]

    if len(features) < 2:
        continue

    # Pairwise correlations use available observations.
    # Record sample counts so small comparisons aren't mistaken
    # for strong evidence.
    for i, feature_a in enumerate(features):
        for feature_b in features[i + 1:]:
            pair = group[[feature_a, feature_b]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()

            pair_count = len(pair)

            if pair_count < MIN_CORRELATION_SAMPLES:
                continue

            corr = pair[feature_a].corr(pair[feature_b])

            if pd.notna(corr) and abs(corr) >= HIGH_CORRELATION:
                correlation_reports.append({
                    "position": position,
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "correlation": round(float(corr), 3),
                    "players_compared": pair_count,
                })

correlation_df = pd.DataFrame(
    correlation_reports,
    columns=[
        "position",
        "feature_a",
        "feature_b",
        "correlation",
        "players_compared",
    ],
)

correlation_path = OUTPUT_DIR / "high_correlation_pairs.csv"
correlation_df.to_csv(correlation_path, index=False)

print("\n" + "=" * 75)
print("POTENTIALLY REDUNDANT FEATURES")
print("=" * 75)

if correlation_df.empty:
    print("No pairs crossed the current correlation threshold.")
else:
    print(
        correlation_df.sort_values(
            ["position", "correlation"],
            key=lambda s: s.abs() if s.name == "correlation" else s,
            ascending=False,
        ).to_string(index=False)
    )

print(f"\nSaved correlation report:\n{correlation_path}")


# --------------------------------------------------
# 7. SUMMARY
# --------------------------------------------------

print("\n" + "=" * 75)
print("EVALUATION COMPLETE")
print("=" * 75)

for position in [
    "Forward",
    "Midfielder",
    "Defender",
    "Goalkeeper",
]:
    print(
        f"{position:12s}: "
        f"{len(usable_features[position]):3d} initial candidates"
    )

print("\nReview both CSV reports before choosing the final features.")
print("Do not automatically use every feature that passes the screen.")
