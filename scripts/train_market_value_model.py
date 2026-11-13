
from pathlib import Path

import joblib
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
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "data" / "processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Loading market value dataset...")
df = pd.read_csv(DATA_PATH)

df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
df["valuation_date"] = pd.to_datetime(df["valuation_date"], errors="coerce")

df = df.sort_values(["player_id", "season"]).reset_index(drop=True)

# Calculate age on the valuation date.
df["age"] = (
    (df["valuation_date"] - df["date_of_birth"]).dt.days / 365.25
)

# Previous-season performance.
performance_columns = [
    "minutes_played",
    "goals",
    "assists",
    "yellow_cards",
    "red_cards",
    "appearances",
]

grouped = df.groupby("player_id", sort=False)

for col in performance_columns:
    df[f"previous_{col}"] = grouped[col].shift(1)

df["previous_season"] = grouped["season"].shift(1)

# Only use performance from the immediately preceding season.
consecutive = df["season"] - df["previous_season"] == 1

# Previous market value and its valuation date.
df["previous_market_value"] = grouped["market_value_in_eur"].shift(1)
df["previous_valuation_date"] = grouped["valuation_date"].shift(1)

valuation_gap_days = (
    df["valuation_date"] - df["previous_valuation_date"]
).dt.days

# Strict eligibility rule from our earlier feature experiment:
# previous valuation must be earlier and no more than 548 days old.
valid_previous_value = (
    df["previous_market_value"].gt(0)
    & valuation_gap_days.gt(0)
    & valuation_gap_days.le(548)
)

# Require consecutive performance history and an eligible previous value.
df = df[
    consecutive & valid_previous_value
].copy()

df = df[
    df["market_value_in_eur"].gt(0)
    & df["age"].notna()
].copy()

numeric_features = [
    "age",
    "previous_minutes_played",
    "previous_goals",
    "previous_assists",
    "previous_yellow_cards",
    "previous_red_cards",
    "previous_appearances",
    "previous_market_value",
]

categorical_features = ["position", "sub_position"]
features = numeric_features + categorical_features
target = "market_value_in_eur"

# Chronological evaluation:
# Train: up to 2022
# Validation: 2023
# Test: 2024–2025
train = df[df["season"] <= 2022].copy()
valid = df[df["season"] == 2023].copy()
test = df[df["season"].between(2024, 2025)].copy()

if train.empty or valid.empty or test.empty:
    raise ValueError(
        "A data split is empty. Check the dataset seasons and eligibility rules."
    )

print("\nDATA SPLIT")
print(f"Training rows (up to 2022): {len(train)}")
print(f"Validation rows (2023): {len(valid)}")
print(f"Test rows (2024–2025): {len(test)}")
print("Previous market value is included as an input feature.")

# Preprocessing is fitted on training data only.
preprocessor = ColumnTransformer([
    (
        "numeric",
        SimpleImputer(strategy="median"),
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
    ("regressor", RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )),
])

print("\nTraining improved Model A...")
model.fit(train[features], train[target])

def evaluate(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print(
        f"{name}: "
        f"MAE €{mae:,.0f} | "
        f"RMSE €{rmse:,.0f} | "
        f"R² {r2:.4f}"
    )

    return mae

validation_predictions = model.predict(valid[features])

print("\nVALIDATION RESULTS")
validation_mae = evaluate(
    "Model A — 2023 validation",
    valid[target],
    validation_predictions,
)

# Evaluate once on the later test seasons.
test_predictions = model.predict(test[features])

print("\nTEST RESULTS")
test_mae = evaluate(
    "Model A — 2024–2025 test",
    test[target],
    test_predictions,
)

# Save the updated model using the same artifact path.
# The artifact is now a dictionary containing the pipeline and metadata.
artifact = {
    "model": model,
    "features": features,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "target": target,
    "validation_mae": validation_mae,
    "includes_previous_market_value": True,
}

model_path = MODEL_DIR / "market_value_model.joblib"
joblib.dump(artifact, model_path)

# Save predictions for later inspection.
prediction_output = test[
    ["season", "player_id", "player_name", "market_value_in_eur"]
].copy()

prediction_output["predicted_market_value"] = test_predictions
prediction_output["absolute_error"] = np.abs(
    prediction_output["market_value_in_eur"]
    - prediction_output["predicted_market_value"]
)

prediction_path = OUTPUT_DIR / "market_value_test_predictions.csv"
prediction_output.to_csv(prediction_path, index=False)

print("\nSAVED")
print(f"Model: {model_path}")
print(f"Test predictions: {prediction_path}")
print("\nModel A training complete.")

