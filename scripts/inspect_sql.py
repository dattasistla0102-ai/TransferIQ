
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"

sql_files = list(DATA_DIR.rglob("*.sql"))

if not sql_files:
    print("No SQL files found in data/raw.")
else:
    for file in sql_files:
        print("\n" + "=" * 60)
        print(f"SQL FILE: {file.name}")
        print("=" * 60)

        text = file.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        print(f"Total lines: {len(lines)}")

        print("\nTable definitions and insert statements:")
        matches = [
            (i, line.strip())
            for i, line in enumerate(lines, start=1)
            if any(word in line.upper() for word in (
                "CREATE TABLE",
                "INSERT INTO",
                "COPY ",
                "CREATE DATABASE"
            ))
        ]

        for line_number, line in matches[:25]:
            print(f"Line {line_number}: {line[:250]}")

        print("\nFirst 15 lines of the file:")
        for line in lines[:15]:
            print(line[:250])
