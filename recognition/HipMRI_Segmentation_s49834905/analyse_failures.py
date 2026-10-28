"""
analyse_failures.py — Reads per_slice.json from all six runs and categorises
the prostate slices into four groups based on the actual number of pixels
occupied by the prostate. For each group it reports the number of slices, the
mean Dice score and the failure rate, to help determine whether failures are
concentrated among slices where the prostate is very small.

Usage:
    python analyse_failures.py
"""

import json
import statistics as stats
from pathlib import Path

RUNS = {
    "unet": ["unet", "unet_s1", "unet_s2"],
    "can": ["can30", "can_s1", "can_s2"],
}

# Groups by how many ground-truth prostate pixels a slice contains
BINS = [(1, 50), (51, 150), (151, 300), (301, 100000)]


def load_records(run):
    """Read outputs/<run>/eval/per_slice.json and return the list of slices."""
    path = Path("outputs") / run / ("eval") / ("per_slice.json")
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    for model, runs in RUNS.items():
        print(model)
        for lo, hi in BINS:
            dices = []
            for run in runs:
                for r in load_records(run):
                    if lo <= r["gt_pixels"] <= hi:
                        dices.append(r["dice"])
            if not dices:
                continue
            n_slices = len(dices) // len(runs)
            mean_dice = stats.mean(dices)
            fail_rate = sum(1 for d in dices if d < 0.5) / len(dices)
            print(f"  {lo:>4}-{hi:<6} px: {n_slices:>3} slices  "
                    f"mean Dice {mean_dice:.3f}  fail rate {fail_rate:.0%}")