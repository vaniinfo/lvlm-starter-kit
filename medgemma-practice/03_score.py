"""Step 4: score the model's answers against the real labels.

Prints the "caught it / missed it / false alarm" table plus accuracy,
sensitivity and specificity.

Example:
    python 03_score.py
"""

import argparse
import csv


def percent(part, whole):
    return f"{100 * part / whole:.0f}%" if whole else "n/a"


def main():
    parser = argparse.ArgumentParser(description="Score yes/no answers.")
    parser.add_argument("--answers", default="results/answers.csv")
    args = parser.parse_args()

    with open(args.answers, newline="") as f:
        rows = list(csv.DictReader(f))

    caught = sum(r["truth"] == "yes" and r["answer"] == "yes" for r in rows)
    missed = sum(r["truth"] == "yes" and r["answer"] == "no" for r in rows)
    false_alarm = sum(r["truth"] == "no" and r["answer"] == "yes" for r in rows)
    cleared = sum(r["truth"] == "no" and r["answer"] == "no" for r in rows)
    unclear = sum(r["answer"] == "unclear" for r in rows)

    real_yes = sum(r["truth"] == "yes" for r in rows)
    real_no = sum(r["truth"] == "no" for r in rows)

    print()
    print(f"{'':16}{'model: yes':>12}{'model: no':>12}")
    print(f"{'really yes':16}{caught:>12}{missed:>12}   (caught / MISSED)")
    print(f"{'really no':16}{false_alarm:>12}{cleared:>12}   (FALSE ALARM / correct)")
    print()
    print(f"Accuracy:     {percent(caught + cleared, len(rows))}  ({caught + cleared} of {len(rows)} right)")
    print(f"Sensitivity:  {percent(caught, real_yes)}  (real cases caught: {caught} of {real_yes})")
    print(f"Specificity:  {percent(cleared, real_no)}  (healthy images cleared: {cleared} of {real_no})")
    if unclear:
        print(f"Unclear:      {unclear} answers weren't a clear yes/no. They count as wrong above. "
              "Read them in the raw_reply column.")


if __name__ == "__main__":
    main()
