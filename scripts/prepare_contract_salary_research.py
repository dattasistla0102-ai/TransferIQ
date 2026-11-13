
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data" / "processed" / "transfer_model_data_2020_2025.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT = OUTPUT_DIR / "contract_salary_research.csv"

df = pd.read_csv(SOURCE)

# Keep paid, permanent transfers with a valid fee.
df["fee"] = pd.to_numeric(df["fee"], errors="coerce")

df = df[
    (df["fee"] > 0)
    & (df["is_loan"].astype(str).str.lower().isin(["false", "0", "no"]))
].copy()

# One research row per player, prioritizing the largest transfer fee.
df = df.sort_values("fee", ascending=False)
players = df.drop_duplicates(subset=["player_id"], keep="first").copy()

columns = [
    "player_id",
    "player_name",
    "season",
    "club",
    "dealing_club",
    "fee",
]

research = players[columns].rename(columns={
    "season": "transfer_season",
    "club": "buying_club",
    "dealing_club": "selling_club",
    "fee": "transfer_fee_eur",
})

# Research fields: leave unknown values blank, never guess.
research["transfer_date"] = ""
research["contract_expiry_date"] = ""
research["contract_months_remaining"] = ""
research["annual_salary_original_currency"] = ""
research["salary_currency"] = ""
research["annual_salary_eur"] = ""
research["contract_source_url"] = ""
research["salary_source_url"] = ""
research["contract_confidence"] = ""
research["salary_confidence"] = ""
research["research_notes"] = ""

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
research.to_csv(OUTPUT, index=False)

print(f"Research file created: {OUTPUT}")
print(f"Unique players: {len(research)}")
print("\nFirst 10 players to research:")
print(research[
    ["player_name", "transfer_season", "selling_club",
     "buying_club", "transfer_fee_eur"]
].head(10).to_string(index=False))
