"""Tune the agent human-review confidence threshold on one saved JSONL run."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="saved agent run JSONL")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--min-coverage", type=float, default=0.90)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    score_path = root / "data" / "processed" / f"scores_{args.split}.csv"
    labels = {row["trans_num"]: int(row["is_fraud"]) for row in csv.DictReader(score_path.open())}

    rows = []
    errors = 0
    skipped = 0
    for line in Path(args.run).open():
        record = json.loads(line)
        decision = record.get("decision")
        if decision is None:
            errors += 1
            continue
        predicted = decision["decision"]
        if predicted not in ("fraud", "legitimate"):
            continue
        label = labels.get(record["case_id"])
        if label is None:
            skipped += 1
            continue
        confidence = float(decision["confidence"])
        correct = (predicted == "fraud" and label == 1) or (
            predicted == "legitimate" and label == 0
        )
        rows.append((confidence, correct))

    if not rows:
        raise SystemExit(f"No usable decisions in {args.run}; errors={errors}")

    candidates = []
    for threshold in (step / 100 for step in range(50, 101)):
        auto = [correct for confidence, correct in rows if confidence >= threshold]
        coverage = len(auto) / len(rows)
        accuracy = sum(auto) / len(auto) if auto else 0.0
        candidates.append((coverage, accuracy, threshold))

    eligible = [row for row in candidates if row[0] >= args.min_coverage]
    if not eligible:
        raise SystemExit(f"No threshold reaches {args.min_coverage:.0%} coverage")

    coverage, accuracy, threshold = max(eligible, key=lambda row: (row[1], row[0], -row[2]))
    print(f"usable decisions: {len(rows)}")
    print(f"errors: {errors}")
    print(f"missing labels: {skipped}")
    print(f"balanced threshold: {threshold:.2f}")
    print(f"coverage: {coverage:.1%}")
    print(f"auto-decision accuracy: {accuracy:.1%}")
    print(f"routed to human review: {(1 - coverage):.1%}")

    print("\nthreshold  coverage  auto_accuracy")
    for threshold in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        auto = [correct for confidence, correct in rows if confidence >= threshold]
        coverage = len(auto) / len(rows)
        accuracy = sum(auto) / len(auto) if auto else 0.0
        print(f"{threshold:.2f}      {coverage:.1%}     {accuracy:.1%}")


if __name__ == "__main__":
    main()
