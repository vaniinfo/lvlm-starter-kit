"""Step 2: build a small, balanced test set from ChestX-ray14.

Picks N images that have the finding and N images labeled "No Finding",
using only images you have actually downloaded. Saves data/test_set.csv.

Example:
    python 01_pick_images.py --finding Effusion --n 10
"""

import argparse
import csv
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build a balanced yes/no test set.")
    parser.add_argument("--labels", default="data/Data_Entry_2017_v2020.csv",
                        help="ChestX-ray14 label file")
    parser.add_argument("--images", default="data/images",
                        help="folder with the unpacked .png images")
    parser.add_argument("--finding", default="Effusion",
                        help="finding name exactly as written in the label file")
    parser.add_argument("--n", type=int, default=10,
                        help="how many 'yes' and how many 'no' images to pick")
    parser.add_argument("--out", default="data/test_set.csv")
    parser.add_argument("--seed", type=int, default=42,
                        help="fixed seed so you get the same images every run")
    args = parser.parse_args()

    images_dir = Path(args.images)
    downloaded = {p.name for p in images_dir.glob("*.png")}
    if not downloaded:
        raise SystemExit(f"No .png images found in {images_dir}. Did you unpack the download?")

    yes_rows, no_rows = [], []
    with open(args.labels, newline="") as f:
        for row in csv.DictReader(f):
            name = row["Image Index"]
            if name not in downloaded:
                continue
            findings = row["Finding Labels"].split("|")
            if args.finding in findings:
                yes_rows.append((name, "yes", row["Finding Labels"]))
            elif findings == ["No Finding"]:
                no_rows.append((name, "no", row["Finding Labels"]))

    print(f"Found {len(yes_rows)} images with {args.finding} "
          f"and {len(no_rows)} with No Finding among {len(downloaded)} downloaded images.")
    if len(yes_rows) < args.n or len(no_rows) < args.n:
        raise SystemExit("Not enough images for that --n. Download another batch or lower --n.")

    random.seed(args.seed)
    chosen = random.sample(yes_rows, args.n) + random.sample(no_rows, args.n)
    random.shuffle(chosen)  # mix yes and no so the order gives nothing away

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "truth", "all_labels"])
        writer.writerows(chosen)
    print(f"Saved {len(chosen)} images to {out}")


if __name__ == "__main__":
    main()
