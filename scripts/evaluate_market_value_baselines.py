
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "market_value_dataset.csv"
OUTPUT_PATH = (
    ROOT / "data" / "processed"
    / "market_value_baseline_evaluation.csv"
)

TARGET = "market_value_in_eur"

NUMERIC = [
    "age",
    "previous_minutes",
    "previous_goals",
    "previous_assists",
    "previous_yellow_cards",
    "previous_red_cards",
    "previous_appearances",
]

CATEGORICAL = ["position", "sub_position"]


def calculate_metrics(name, actual, predicted):
    predicted = np.maximum(predicted, 0)
    actual = np.asarray(actual)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    # Avoid unstable percentage errors for tiny values.
    mask = actual >= 1_000_000
    mape = (
        np.mean(np.abs(actual[mask] - predicted[mask]) / actual[mask]) * 100
        if mask.any()
        else np.nan
    )

    return {
        "model": name,
        "rows": len(actual),
        "MAE_EUR": mae,
        "RMSE_EUR": rmse,
        "R2": r2,
        "MAPE_percent_players_over_1M": mape,
    }


def main():
    df = pd.read_csv(DATA_PATH)

    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["date_of_birth"] = pd.to_datetime(
        df["date_of_birth"], errors="coerce"
    )
    df["valuation_date"] = pd.to_datetime(
        df["valuation_date"], errors="coerce"
    )

    df["age"] = (
        (df["valuation_date"] - df["date_of_birth"]).dt.days / 365.2425
    )
    df["age"] = df["age"].where(df["age"].between(15, 50))

    df = df.sort_values(["player_id", "season"]).copy()

    stat_mapping = {
        "minutes_played": "previous_minutes",
        "goals": "previous_goals",
        "assists": "previous_assists",
        "yellow_cards": "previous_yellow_cards",
        "red_cards": "previous_red_cards",
        "appearances": "previous_appearances",
    }

    for original, previous in stat_mapping.items():
        df[previous] = df.groupby("player_id")[original].shift(1)

    previous_season = df.groupby("player_id")["season"].shift(1)
    consecutive = df["season"].eq(previous_season + 1)

    performance_features = list(stat_mapping.values())
    df.loc[~consecutive, performance_features] = np.nan

    df["previous_market_value"] = (
        df.groupby("player_id")[TARGET].shift(1)
    )
    df["previous_valuation_date"] = (
        df.groupby("player_id")["valuation_date"].shift(1)
    )

    days_between = (
        df["valuation_date"] - df["previous_valuation_date"]
    ).dt.days

    valid_previous_value = days_between.between(1, 548)
    df.loc[~valid_previous_value, "previous_market_value"] = np.nan

    # Keep the same eligible population for every comparison.
    df = df.dropna(
        subset=performance_features + [TARGET, "season"]
    ).copy()
    df = df[df[TARGET] > 0].copy()

    train = df[df["season"] <= 2022].copy()
    val = df[df["season"] == 2023].copy()

    if train.empty or val.empty:
        raise ValueError("Training or validation split is empty.")

    # 1. Simple baseline: previous value alone.
    baseline_rows = val["previous_market_value"].notna()
    baseline_val = val.loc[baseline_rows]

    results = [
        calculate_metrics(
            "Previous market value only",
            baseline_val[TARGET],
            baseline_val["previous_market_value"],
        )
    ]

    # 2. Model with performance and player attributes.
    # 3. Same model plus previous market value.
    for name, numeric_features in [
        ("Performance + attributes", NUMERIC),
        (
            "Performance + attributes + previous value",
            NUMERIC + ["previous_market_value"],
        ),
    ]:
        preprocessor = ColumnTransformer([
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CATEGORICAL,
            ),
        ])

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )),
        ])

        model.fit(train[numeric_features + CATEGORICAL], train[TARGET])

        predictions = model.predict(
            val[numeric_features + CATEGORICAL]
        )

        results.append(
            calculate_metrics(name, val[TARGET], predictions)
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nMARKET VALUE BASELINE EVALUATION")
    print(f"Training rows: {len(train)}")
    print(f"Validation rows: {len(val)}")
    print(
        "Previous-value baseline rows: "
        f"{int(baseline_rows.sum())}"
    )
    print("Training seasons: up to 2022")
    print("Validation season: 2023")
    print("2024–2025 test set remains untouched.")

    for _, row in results_df.iterrows():
        print(f"\n{row['model']}")
        print(f"MAE:  €{row['MAE_EUR']:,.0f}")
        print(f"RMSE: €{row['RMSE_EUR']:,.0f}")
        print(f"R²:   {row['R2']:.4f}")
        print(
            "MAPE (players worth at least €1M): "
            f"{row['MAPE_percent_players_over_1M']:.2f}%"
        )

    print(f"\nSaved results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
