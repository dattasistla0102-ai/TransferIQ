
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

print("=" * 60)
print("MODEL A ARTIFACT")
print("=" * 60)

model_a_path = ROOT / "models" / "market_value_model.joblib"

if model_a_path.exists():
    model_a = joblib.load(model_a_path)
    print("Type:", type(model_a).__name__)

    if isinstance(model_a, dict):
        print("Keys:", list(model_a.keys()))
        for key, value in model_a.items():
            if key in ["model", "pipeline"]:
                print(key, "type:", type(value).__name__)
            elif key in ["features", "feature_names"]:
                print(key, ":", value)
            else:
                print(key, ":", value)
    else:
        print("Features:", getattr(model_a, "feature_names_in_", "Not exposed"))
else:
    print("Model A artifact not found.")

print("\n" + "=" * 60)
print("MODEL B ARTIFACT")
print("=" * 60)

model_b_path = ROOT / "models" / "transfer_ratio_model.joblib"

if model_b_path.exists():
    model_b = joblib.load(model_b_path)
    print("Type:", type(model_b).__name__)
    if isinstance(model_b, dict):
        print("Keys:", list(model_b.keys()))
        print("Features:", model_b.get("features"))
        print("Selected model:", model_b.get("best_model_name"))
else:
    print("Model B artifact not found.")

print("\n" + "=" * 60)
print("MARKET VALUE DATASET")
print("=" * 60)

dataset_path = ROOT / "data" / "processed" / "market_value_dataset.csv"

if dataset_path.exists():
    df = pd.read_csv(dataset_path)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nLatest seasons:")
    print(df["season"].value_counts().sort_index().tail(5).to_string())
    print("\nSample rows:")
    print(df.head(3).to_string(index=False))
else:
    print("Market value dataset not found.")
