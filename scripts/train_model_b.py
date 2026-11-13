
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from lightgbm import LGBMRegressor
except ImportError:
    raise SystemExit(
        "LightGBM is not installed. Run:\n"
        ".\\.venv\\Scripts\\python.exe -m pip install lightgbm"
    )

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

SOURCE = DATA / "transfer_history_clean.csv"
MODEL_PATH = MODELS / "transfer_fee_model.joblib"
METRICS_PATH = DATA / "model_b_evaluation.csv"
PREDICTIONS_PATH = DATA / "model_b_test_predictions.csv"

# --------------------------------------------------
# 1. Load and prepare transfer history
# --------------------------------------------------
df = pd.read_csv(SOURCE)

required = [
    "season", "club", "window", "player_id",
    "age", "position", "market_value", "fee", "is_loan"
]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing expected columns: {missing}\n"
        f"Available columns: {df.columns.tolist()}"
    )

df["season"] = pd.to_numeric(df["season"], errors="coerce")
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce")
df["fee"] = pd.to_numeric(df["fee"], errors="coerce")

# Keep paid, permanent transfers with a known positive fee.
loan_text = df["is_loan"].astype(str).str.strip().str.lower()
is_permanent = loan_text.isin(["false", "0", "no"])

df = df[
    is_permanent
    & (df["fee"] > 0)
    & df["season"].notna()
].copy()

# Use the known selling-club column when available.
if "dealing_club" in df.columns:
    df["selling_club"] = df["dealing_club"]
else:
    df["selling_club"] = "Unknown"

df["buying_club"] = df["club"]

# Restrict to records with a usable market-value input.
# This is a transfer-time valuation, not necessarily a pre-transfer valuation.
df = df[df["market_value"] > 0].copy()

# Remove impossible or corrupted values.
df = df[
    (df["age"].between(15, 45) | df["age"].isna())
    & (df["fee"] < 500_000_000)
    & (df["market_value"] < 500_000_000)
].copy()

if df.empty:
    raise ValueError("No usable transfer records remain after filtering.")

# --------------------------------------------------
# 2. Define features and target
# --------------------------------------------------
numeric_features = ["age", "market_value", "season"]

categorical_features = [
    "position",
    "buying_club",
    "selling_club",
    "window",
]

features = numeric_features + categorical_features
target = "fee"

# Exclude player_name and player_id: they are identifiers,
# not useful generalizable predictors for this first model.
X = df[features].copy()
y = df[target].copy()

for col in categorical_features:
    X[col] = X[col].fillna("Unknown").astype(str)

# --------------------------------------------------
# 3. Time-based train / validation / test split
# --------------------------------------------------
train_mask = df["season"] <= 2023
val_mask = df["season"] == 2024
test_mask = df["season"] == 2025

X_train, y_train = X.loc[train_mask], y.loc[train_mask]
X_val, y_val = X.loc[val_mask], y.loc[val_mask]
X_test, y_test = X.loc[test_mask], y.loc[test_mask]

if min(len(X_train), len(X_val), len(X_test)) == 0:
    raise ValueError(
        "One split is empty. Check the season values in the source CSV."
    )

print("Model B dataset")
print(f"Training rows (up to 2023): {len(X_train)}")
print(f"Validation rows (2024):    {len(X_val)}")
print(f"Test rows (2025):          {len(X_test)}")
print(f"Features: {features}")

# --------------------------------------------------
# 4. Build LightGBM pipeline
# --------------------------------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

regressor = LGBMRegressor(
    objective="regression",
    n_estimators=700,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    verbosity=-1,
    n_jobs=-1,
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", regressor),
])

# --------------------------------------------------
# 5. Train and evaluate
# --------------------------------------------------
model.fit(X_train, y_train)

def evaluate(name, X_part, y_part):
    predictions = np.maximum(model.predict(X_part), 0)

    mae = mean_absolute_error(y_part, predictions)
    rmse = np.sqrt(mean_squared_error(y_part, predictions))
    r2 = r2_score(y_part, predictions)

    print(f"\n{name}")
    print(f"MAE:  €{mae:,.0f}")
    print(f"RMSE: €{rmse:,.0f}")
    print(f"R²:   {r2:.4f}")

    return predictions, {
        "split": name,
        "rows": len(y_part),
        "MAE_eur": mae,
        "RMSE_eur": rmse,
        "R2": r2,
    }

val_predictions, val_metrics = evaluate(
    "2024 validation", X_val, y_val
)

# Refit using all data available through 2024 after validation.
# Keep 2025 held out for final testing.
train_val_mask = df["season"] <= 2024
X_train_val = X.loc[train_val_mask]
y_train_val = y.loc[train_val_mask]

model.fit(X_train_val, y_train_val)

test_predictions, test_metrics = evaluate(
    "2025 final test", X_test, y_test
)

# --------------------------------------------------
# 6. Save metrics, predictions, and model
# --------------------------------------------------
pd.DataFrame([val_metrics, test_metrics]).to_csv(
    METRICS_PATH, index=False
)

test_output = df.loc[
    test_mask,
    ["player_name", "season", "club", "dealing_club", "fee"]
].copy() if "player_name" in df.columns else df.loc[
    test_mask,
    ["season", "club", "dealing_club", "fee"]
].copy()

test_output["predicted_fee_eur"] = test_predictions
test_output["absolute_error_eur"] = np.abs(
    test_output["fee"].to_numpy() - test_predictions
)
test_output.to_csv(PREDICTIONS_PATH, index=False)

artifact = {
    "model": model,
    "features": features,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "target": target,
    "training_years": "up to 2024",
    "test_year": 2025,
    "note": (
        "Market value is taken from the transfer-history source and may "
        "reflect a valuation recorded around the transfer. Check its "
        "timestamp before interpreting this as a strictly pre-transfer model."
    ),
}

joblib.dump(artifact, MODEL_PATH)

print("\nSaved:")
print(MODEL_PATH)
print(METRICS_PATH)
print(PREDICTIONS_PATH)
