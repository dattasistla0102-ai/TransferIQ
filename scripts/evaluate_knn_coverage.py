from pathlib import Path
import sys
import pandas as pd

# Import the exact models and eligibility pools used by build_knn_scout.py.
# Run this script from the TransferIQ project root:
# python scripts/evaluate_knn_coverage.py

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "feature_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
import build_knn_scout as knn  # noqa: E402


def record_keys(frame):
    return set(
        zip(
            frame["name_key"].astype(str),
            frame["team_key"].astype(str),
        )
    )


def make_key(row):
    return str(row["name_key"]), str(row["team_key"])


# Count unique player-team records after the same cleaning used by the model.
players = knn.df.drop_duplicates(
    subset=["name_key", "team_key"]
).copy()

specialized_keys = {
    position: record_keys(pool)
    for position, pool in knn.position_data.items()
}
general_keys = record_keys(knn.general_data)

report_rows = []
failure_rows = []

for _, player in players.iterrows():
    key = make_key(player)
    position = player.get("_position_group")
    display_position = (
        str(position) if pd.notna(position) else "Unknown"
    )

    if pd.notna(position) and position == "Goalkeeper":
        if key in specialized_keys.get("Goalkeeper", set()):
            status = "SUCCESS"
            route = "Goalkeeper KNN"
            reason = "Eligible for goalkeeper model"
        else:
            status = "FAIL"
            route = "None"
            reason = (
                "Not eligible for goalkeeper model; goalkeeper "
                "fallback to outfield model is intentionally disabled"
            )
    elif (
        pd.notna(position)
        and position in specialized_keys
        and key in specialized_keys[position]
    ):
        status = "SUCCESS"
        route = f"{position} KNN"
        reason = "Eligible for specialized model"
    elif key in general_keys:
        status = "SUCCESS"
        route = "General KNN fallback"
        reason = (
            "Not eligible for specialized model or position unknown; "
            "eligible for general outfield model"
        )
    else:
        status = "FAIL"
        route = "None"
        reason = (
            "Does not meet the real-feature eligibility threshold "
            "for the available model pools"
        )

    report_rows.append({
        "player_name": player["player_name"],
        "team": player["team_key"],
        "position": display_position,
        "status": status,
        "model_route": route,
        "reason": reason,
    })

    if status == "FAIL":
        failure_rows.append(report_rows[-1])

report = pd.DataFrame(report_rows)
failures = pd.DataFrame(failure_rows)

summary_path = OUTPUT_DIR / "knn_output_coverage_summary.csv"
details_path = OUTPUT_DIR / "knn_output_coverage_by_player.csv"
failures_path = OUTPUT_DIR / "knn_players_without_output.csv"

report.to_csv(details_path, index=False)
failures.to_csv(failures_path, index=False)

total = len(report)
successes = int((report["status"] == "SUCCESS").sum())
failed = total - successes
rate = (successes / total * 100) if total else 0.0

print("\n" + "=" * 72)
print("TRANSFERIQ: KNN OUTPUT COVERAGE AUDIT")
print("=" * 72)
print(f"Unique player-team records assessed: {total}")
print(f"Players for whom recommendations can be generated: {successes}")
print(f"Players for whom recommendations cannot be generated: {failed}")
print(f"Overall output coverage: {rate:.2f}%")
print(f"Output failure rate: {(100 - rate):.2f}%")

print("\nCoverage by recorded broad position:")
by_position = (
    report.groupby("position", dropna=False)
    .agg(
        total_players=("status", "size"),
        outputs_available=("status", lambda s: (s == "SUCCESS").sum()),
    )
    .reset_index()
)
by_position["coverage_percent"] = (
    by_position["outputs_available"] / by_position["total_players"] * 100
).round(2)
print(by_position.to_string(index=False))

print("\nSuccessful routes:")
print(report.loc[report["status"] == "SUCCESS", "model_route"]
      .value_counts().to_string())

print("\nFailure reasons:")
if failures.empty:
    print("No failures.")
else:
    print(failures["reason"].value_counts().to_string())

print("\nExample players without output (up to 25):")
if failures.empty:
    print("None.")
else:
    print(
        failures[["player_name", "team", "position", "reason"]]
        .head(25)
        .to_string(index=False)
    )

summary = pd.DataFrame([{
    "unique_player_team_records": total,
    "outputs_available": successes,
    "outputs_unavailable": failed,
    "coverage_percent": round(rate, 2),
    "failure_percent": round(100 - rate, 2),
}])
summary.to_csv(summary_path, index=False)

print("\nSaved reports:")
print(f"  Summary: {summary_path}")
print(f"  Per-player audit: {details_path}")
print(f"  Players without output: {failures_path}")
print(
    "\nInterpretation: this measures whether the current code can produce "
    "a recommendation for each player. It does not measure whether the "
    "recommendations are football-accurate or useful."
)
