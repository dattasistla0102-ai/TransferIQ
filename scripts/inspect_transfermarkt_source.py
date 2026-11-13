
import duckdb

BASE_URL = (
    "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/"
)

con = duckdb.connect()

con.execute("INSTALL httpfs")
con.execute("LOAD httpfs")

for filename in [
    "competitions.csv.gz",
    "players.csv.gz",
    "appearances.csv.gz",
    "games.csv.gz",
    "player_valuations.csv.gz",
]:
    url = BASE_URL + filename

    print(f"\n{'=' * 60}")
    print(f"FILE: {filename}")
    print("=" * 60)

    try:
        result = con.execute(
            f"SELECT * FROM read_csv_auto('{url}') LIMIT 3"
        ).df()

        print("Columns:")
        print(result.columns.tolist())
        print("\nSample:")
        print(result.to_string(index=False))

        if filename == "competitions.csv.gz":
            competitions = con.execute(
                f"SELECT * FROM read_csv_auto('{url}')"
            ).df()

            print("\nPremier League competition candidates:")
            mask = (
                competitions.astype(str)
                .apply(
                    lambda col: col.str.contains(
                        "premier league",
                        case=False,
                        na=False
                    )
                )
                .any(axis=1)
            )

            print(competitions[mask].to_string(index=False))

    except Exception as exc:
        print(f"Could not inspect this file: {exc}")

con.close()
