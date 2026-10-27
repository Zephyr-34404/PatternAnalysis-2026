"""
compare.py — Aggregates the evaluation results (metrics.json) from all six
experiments and prints the mean and standard deviation for each model, to
account for potential random fluctuations between seeds.

Usage:
    python compare.py
"""

import json
import statistics as stats
from pathlib import Path

# Which run folders belong to which model (seeds 42, 1, 2)
RUNS = {
    "unet": ["unet", "unet_s1", "unet_s2"],
    "can": ["can30", "can_s1", "can_s2"],
}


def load_metrics(run):
    """Read outputs/<run>/eval/metrics.json and return it as a dict."""
    path = Path("outputs") / run / ("eval") / ("metrics.json")
    with open(path) as f:
        return json.load(f)


def summarise(values):
    """Format a list of numbers as 'mean ± sd' with 3 decimals."""
    mean =  stats.mean(values)
    sd = stats.stdev(values)
    return f"{mean:.3f} ± {sd:.3f}"


if __name__ == "__main__":
    for model, runs in RUNS.items():
        all_metrics = [load_metrics(r) for r in runs]

        prostate = [m["dice"]["prostate"] for m in all_metrics]
        edge = [m["prostate_dice_extreme_slices"] for m in all_metrics]
        worst = [m["worst_volume_prostate_dice"] for m in all_metrics]

        print(model)
        print("  prostate Dice :", summarise(prostate))
        print("  edge slices   :", summarise(edge))
        print("  worst volume  :", summarise(worst))