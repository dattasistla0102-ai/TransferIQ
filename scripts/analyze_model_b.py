
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

SOURCE = DATA / "transfer_history_all_seasons.csv"
PREDICTIONS = DATA / "model_b_test_predictions.csv"

df = pd.read_csv(SOURCE)
pred = pd.read_csv(PREDICTIONS)

# Apply the same filters as train_model_b.py.
df["season"] = pd.to_numeric(df["season"], errors="coerce")
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce")
df["fee"] = pd.to_numeric(df["fee"], errors="coerce")

loan_text = df["is_loan"].astype(str).str.strip().str.lower()
is_permanent = loan_text.isin(["false", "0", "no"])

df = df[
    is_permanent
    & (df["fee"] > 0)
    & df["season"].notna()
].copy()

df = df[df["market_value"] > 0].copy()

df = df[
    (df["age"].between(15, 45) | df["age"].isna())
    & (df["fee"] < 500_000_000)
    & (df["market_value"] < 500_000_000)
].copy()

# Recreate the same time split used by Model B.
train = df[df["season"] <= 2024].copy()
test = df[df["season"] == 2025].copy()

# Align rows with saved test predictions.
# The training script preserves row order.
if len(test) != len(pred):
    raise ValueError(
        f"Test rows ({len(test)}) != prediction rows ({len(pred)}). "
        "Check whether the source dataset changed after training."
    )

actual = pred["fee"].to_numpy(dtype=float)
lgbm = pred["predicted_fee_eur"].to_numpy(dtype=float)

# Baseline 1: predict the median historical transfer fee.
median_fee = train["fee"].median()
baseline_median = np.full(len(test), median_fee)

# Baseline 2: learn a typical fee / market-value ratio
# using training data only, then apply it to test market values.
train_ratio = (
    train["fee"] / train["market_value"]
).replace([np.inf, -np.inf], np.nan).dropna()

median_ratio = train_ratio.median()
baseline_ratio = (
    test["market_value"].to_numpy(dtype=float) * median_ratio
)

def metrics(name, y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "model": name,
        "MAE_eur": mae,
        "RMSE_eur": rmse,
        "R2": r2,
    }

results = pd.DataFrame([
    metrics("Median fee baseline", actual, baseline_median),
    metrics("Median fee/value ratio baseline", actual, baseline_ratio),
    metrics("LightGBM", actual, lgbm),
])

results.to_csv(DATA / "model_b_baseline_comparison.csv", index=False)

print("\n=== BASELINE COMPARISON (2025) ===")
print(results.to_string(index=False, formatters={
    "MAE_eur": lambda x: f"€{x:,.0f}",
    "RMSE_eur": lambda x: f"€{x:,.0f}",
    "R2": lambda x: f"{x:.4f}",
}))

# Detailed error analysis.
errors = pred.copy()
errors["absolute_error_eur"] = np.abs(actual - lgbm)
errors["percentage_error"] = (
    errors["absolute_error_eur"] / np.maximum(actual, 1)
) * 100

# Use actual fee bands to see where errors are largest.
errors["fee_band"] = pd.cut(
    actual,
    bins=[0, 10_000_000, 30_000_000, 60_000_000, np.inf],
    labels=["Under €10M", "€10M–€30M", "€30M–€60M", "Over €60M"],
    include_lowest=True,
)

print("\n=== ERROR BY ACTUAL TRANSFER FEE ===")
grouped = errors.groupby("fee_band", observed=False).agg(
    transfers=("fee", "size"),
    mean_actual_fee=("fee", "mean"),
    MAE=("absolute_error_eur", "mean"),
    median_absolute_error=("absolute_error_eur", "median"),
)
print(grouped.to_string(formatters={
    "mean_actual_fee": lambda x: f"€{x:,.0f}",
    "MAE": lambda x: f"€{x:,.0f}",
    "median_absolute_error": lambda x: f"€{x:,.0f}",
}))

# Largest absolute errors.
columns = [
    c for c in [
        "player_name", "season", "club", "dealing_club",
        "fee", "predicted_fee_eur",
        "absolute_error_eur", "percentage_error"
    ] if c in errors.columns
]

worst = errors.sort_values(
    "absolute_error_eur", ascending=False
)[columns].head(20)

worst.to_csv(DATA / "model_b_worst_predictions.csv", index=False)

print("\n=== 20 LARGEST ABSOLUTE ERRORS ===")
print(worst.to_string(index=False, formatters={
    "fee": lambda x: f"€{x:,.0f}",
    "predicted_fee_eur": lambda x: f"€{x:,.0f}",
    "absolute_error_eur": lambda x: f"€{x:,.0f}",
    "percentage_error": lambda x: f"{x:.1f}%",
}))

print("\nSaved:")
print(DATA / "model_b_baseline_comparison.csv")
print(DATA / "model_b_worst_predictions.csv")

print(
    "\nNote: both baselines use the source market_value field. "
    "If it reflects a valuation recorded at the transfer, neither "
    "baseline is a strictly pre-transfer benchmark."
)
