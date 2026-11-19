"""Assign tanks to train/val/test so frames never leak across splits.

All frames from one tank stay in the same split.  By default a fixed number of
tanks is held out for test, and the rest are split train/val.

Usage:
    python split_by_tank.py --manifest manifest.csv --num-test-tanks 2 --val-ratio 0.15 --seed 2026

Or name the held-out tanks explicitly:
    python split_by_tank.py --manifest manifest.csv --test-tanks tank03 tank07
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("split.csv"))
    parser.add_argument("--num-test-tanks", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-tanks", nargs="*", default=None)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8-sig")))
    if not rows:
        raise SystemExit(f"empty manifest: {args.manifest}")
    tanks = sorted({row["tank"] for row in rows})

    if args.test_tanks:
        test_tanks = [t for t in args.test_tanks]
        unknown = [t for t in test_tanks if t not in tanks]
        if unknown:
            raise SystemExit(f"test tanks not in manifest: {unknown}")
    else:
        rng = random.Random(args.seed)
        shuffled = list(tanks)
        rng.shuffle(shuffled)
        test_tanks = shuffled[: min(args.num_test_tanks, len(tanks))]

    rest = [t for t in tanks if t not in set(test_tanks)]
    rng = random.Random(args.seed)
    shuffled_rest = list(rest)
    rng.shuffle(shuffled_rest)
    val_count = max(1, round(len(rest) * args.val_ratio)) if rest else 0
    val_tanks = set(shuffled_rest[:val_count])

    split_by_tank = {t: "test" if t in set(test_tanks) else ("val" if t in val_tanks else "train") for t in tanks}
    with args.out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tank", "frame", "video", "split"])
        for row in rows:
            writer.writerow([row["tank"], row["frame"], row["video"], split_by_tank[row["tank"]]])

    counts = defaultdict(int)
    for row in rows:
        counts[split_by_tank[row["tank"]]] += 1
    print(f"tanks: {len(tanks)}, test tanks: {test_tanks}, val tanks: {sorted(val_tanks)}")
    print(f"frames by split: {dict(counts)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
