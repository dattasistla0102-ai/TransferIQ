
from pathlib import Path
import csv
import re
import ast
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"

sql_files = list(DATA_DIR.rglob("*.sql"))

if not sql_files:
    raise FileNotFoundError("No SQL file found in data/raw")

sql_file = sql_files[0]
print(f"Reading: {sql_file.name}")

text = sql_file.read_text(encoding="utf-8", errors="replace")

# Extract each INSERT statement's VALUES section.
pattern = re.compile(
    r"INSERT\s+INTO\s+`?player`?\s*"
    r"\((.*?)\)\s*VALUES\s*(.*?);",
    re.IGNORECASE | re.DOTALL,
)

matches = pattern.findall(text)

if not matches:
    raise ValueError("No player INSERT statements were found.")

records = []

for columns_text, values_text in matches:
    columns = re.findall(r"`([^`]+)`", columns_text)

    # Parse SQL tuples while respecting quoted strings and escaped quotes.
    tuples = re.findall(r"\((.*?)\)(?:\s*,|$)", values_text.strip(), re.DOTALL)

    for tuple_text in tuples:
        values = []
        current = ""
        in_string = False
        i = 0

        while i < len(tuple_text):
            char = tuple_text[i]

            if char == "'" and in_string and i + 1 < len(tuple_text) and tuple_text[i + 1] == "'":
                current += "'"
                i += 2
                continue

            if char == "\\" and in_string and i + 1 < len(tuple_text):
                current += tuple_text[i + 1]
                i += 2
                continue

            if char == "'":
                in_string = not in_string
            elif char == "," and not in_string:
                values.append(current.strip())
                current = ""
            else:
                current += char

            i += 1

        values.append(current.strip())

        if len(values) != len(columns):
            continue

        record = {}
        for column, value in zip(columns, values):
            if value.upper() == "NULL":
                record[column] = None
            else:
                record[column] = value

        records.append(record)

roster = pd.DataFrame(records)

if roster.empty:
    raise ValueError("No player records could be parsed.")

# Clean names and remove exact duplicate rows.
for col in ["firstName", "middleName", "lastName"]:
    roster[col] = roster[col].fillna("").astype(str).str.strip()

roster["player_name"] = (
    roster["firstName"] + " " +
    roster["middleName"] + " " +
    roster["lastName"]
).str.replace(r"\s+", " ", regex=True).str.strip()

roster["birthDate"] = pd.to_datetime(
    roster["birthDate"], errors="coerce"
)

roster = roster.drop_duplicates()

output_file = DATA_DIR / "roster_clean.csv"
roster.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"\nParsed records: {len(records)}")
print(f"Unique rows after exact deduplication: {len(roster)}")
print(f"Unique player names: {roster['player_name'].nunique()}")
print(f"Missing birthdates: {roster['birthDate'].isna().sum()}")
print("\nPosition counts:")
print(roster["position"].value_counts(dropna=False))
print("\nSample players:")
print(roster[["player_name", "position", "birthDate", "teamID"]].head(10).to_string(index=False))
print(f"\nSaved roster to: {output_file}")
