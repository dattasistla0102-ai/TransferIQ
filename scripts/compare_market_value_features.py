
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
    / "market_value_feature_comparison_strict.csv"
)

TARGET = "market_value_in_eur"

BASE_NUMERIC = [
    "age",
    "previous_minutes",
    "previous_goals",
    "previous_assists",
    "previous_yellow_cards",
    "previous_red_cards",
    "previous_appearances",
]

BASE_CATEGORICAL = ["position", "sub_position"]


def evaluate(name, train, validation, numeric_features, categorical_features):
    features = numeric_features + categorical_features

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
            categorical_features,
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

    model.fit(train[features], train[TARGET])
    predictions = np.maximum(model.predict(validation[features]), 0)

    return {
        "model": name,
        "validation_rows": len(validation),
        "MAE_EUR": mean_absolute_error(validation[TARGET], predictions),
        "RMSE_EUR": np.sqrt(
            mean_squared_error(validation[TARGET], predictions)
        ),
        "R2": r2_score(validation[TARGET], predictions),
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
        (df["valuation_date"] - df["date_of_birth"]).dt.days
        / 365.2425
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

    # Previous market value and its valuation date.
    df["previous_market_value"] = (
        df.groupby("player_id")["market_value_in_eur"].shift(1)
    )
    df["previous_valuation_date"] = (
        df.groupby("player_id")["valuation_date"].shift(1)
    )

    # Require a strictly earlier valuation no more than 18 months old.
    age_of_valuation = (
        df["valuation_date"] - df["previous_valuation_date"]
    ).dt.days

    valid_previous_value = (
        df["previous_valuation_date"].notna()
        & age_of_valuation.gt(0)
        & age_of_valuation.le(548)
    )

    df.loc[~valid_previous_value, "previous_market_value"] = np.nan

    # Keep the same eligible population for both comparisons.
    df = df.dropna(
        subset=performance_features + [TARGET, "season"]
    ).copy()
    df = df[df[TARGET] > 0].copy()

    train = df[df["season"] <= 2022].copy()
    validation = df[df["season"] == 2023].copy()

    if train.empty or validation.empty:
        raise ValueError("Training or validation split is empty.")

    print(f"Training rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")
    print(
        "Validation rows with eligible previous market value: "
        f"{validation['previous_market_value'].notna().sum()}"
    )

    results = [
        evaluate(
            "Baseline: performance + attributes",
            train,
            validation,
            BASE_NUMERIC,
            BASE_CATEGORICAL,
        ),
        evaluate(
            "Strict: baseline + recent previous market value",
            train,
            validation,
            BASE_NUMERIC + ["previous_market_value"],
            BASE_CATEGORICAL,
        ),
    ]

    results_df = pd.DataFrame(results).sort_values("MAE_EUR")
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nSTRICT MARKET VALUE FEATURE COMPARISON")
    print("Training seasons: up to 2022")
    print("Validation season: 2023")
    print("Test seasons 2024–2025: untouched")

    for _, row in results_df.iterrows():
        print(f"\n{row['model']}")
        print(f"MAE:  €{row['MAE_EUR']:,.0f}")
        print(f"RMSE: €{row['RMSE_EUR']:,.0f}")
        print(f"R²:   {row['R2']:.4f}")

    print(f"\nSaved comparison: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

