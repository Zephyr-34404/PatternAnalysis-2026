"""
predict.py — Evaluate a trained HipMRI 2D segmentation model and visualise its errors.

Loads a saved checkpoint, predicts every slice of one split (test by default)
and reports the metrics needed for the open research dilemma:

    - per-class Dice and IoU, pooled over the whole split
    - per-volume prostate Dice (worst case, not just the average)
    - prostate Dice on the extreme slices of each volume (base / apex)
        versus the central slices
    - leakage: share of predicted-prostate pixels inside ground-truth
        rectum or bladder
    - spurious detections: slices with predicted prostate but none in
        the ground truth
    - parameters, inference time per 60-slice volume and peak inference VRAM

It also saves figures: example predictions and the worst failure cases,
each as input / ground truth / prediction / error overlay.

Outputs (written to --out, default outputs/<model>/eval, git-ignored):
    metrics.json        all numbers above
    examples.png        a few representative slices
    failure_<k>.png     the k-th worst prostate slices

Usage:
    python predict.py --model unet
    python predict.py --model can --checkpoint outputs/can30/best_model.pt
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from dataset import LABEL_NAMES, NUM_CLASSES, PROSTATE_LABEL, HipMRIDataset
from modules import count_parameters
from train import build_model

BLADDER_LABEL = 3
RECTUM_LABEL = 4
EXTREME_SLICES = 2    # slices at each end of the prostate counted as base/apex

# Fixed colours per class for mask plots: background, body, bone, bladder, rectum, prostate
MASK_CMAP = ListedColormap(["#000000", "#5b5b5b", "#e8e8e8", "#3b82f6", "#f59e0b", "#ef4444"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_name(path):
    """'case_004_week_0_slice_30.nii.gz' -> ('case_004_week_0', 30)."""
    match = re.match(r"(.+)_slice_(\d+)\.nii\.gz$", Path(path).name)
    return match.group(1), int(match.group(2))


def dice(pred, true):
    """Binary Dice of two boolean arrays; NaN when both are empty."""
    denom = pred.sum() + true.sum()
    return float("nan") if denom == 0 else 2.0 * (pred & true).sum() / denom


def iou(pred, true):
    """Binary IoU of two boolean arrays; NaN when both are empty."""
    union = (pred | true).sum()
    return float("nan") if union == 0 else (pred & true).sum() / union


def for_display(arr):
    """Rotate a (256, 128) slice so left-right runs horizontally on screen."""
    return np.rot90(arr)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_split(model, dataset, device, batch_size=60):
    """Return predicted labels (N, H, W) as uint8 and seconds per 60 slices."""
    model.eval()
    images = dataset.images.unsqueeze(1)  # (N, 1, H, W)
    preds = []

    # Warm-up pass so the timing excludes one-off CUDA start-up costs.
    model(images[:batch_size].to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)
        preds.append(model(batch).argmax(dim=1).to(torch.uint8).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    seconds_per_volume = elapsed / len(images) * 60
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    return torch.cat(preds).numpy(), seconds_per_volume, peak_vram


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred, true, names):
    """All dilemma metrics for one split. pred and true are (N, H, W) uint8."""
    metrics = {}

    # 1. Pooled per-class Dice and IoU over every pixel in the split.
    metrics["dice"] = {LABEL_NAMES[c]: dice(pred == c, true == c) for c in range(NUM_CLASSES)}
    metrics["iou"] = {LABEL_NAMES[c]: iou(pred == c, true == c) for c in range(NUM_CLASSES)}

    # Group slice indices by volume, ordered by slice number.
    volumes = defaultdict(list)
    for idx, name in enumerate(names):
        vol, slice_no = parse_name(name)
        volumes[vol].append((slice_no, idx))
    volumes = {v: [i for _, i in sorted(s)] for v, s in volumes.items()}

    p_pred, p_true = pred == PROSTATE_LABEL, true == PROSTATE_LABEL

    # 2. Per-volume prostate Dice.
    per_volume = {v: dice(p_pred[idx], p_true[idx]) for v, idx in volumes.items()}
    metrics["per_volume_prostate_dice"] = per_volume
    valid = [d for d in per_volume.values() if not np.isnan(d)]
    metrics["worst_volume_prostate_dice"] = min(valid) if valid else float("nan")

    # 3. Extreme (base/apex) versus central prostate slices.
    extreme, central = [], []
    for idx in volumes.values():
        with_prostate = [i for i in idx if p_true[i].any()]
        if len(with_prostate) <= 2 * EXTREME_SLICES:
            extreme += with_prostate
            continue
        extreme += with_prostate[:EXTREME_SLICES] + with_prostate[-EXTREME_SLICES:]
        central += with_prostate[EXTREME_SLICES:-EXTREME_SLICES]
    metrics["prostate_dice_extreme_slices"] = dice(p_pred[extreme], p_true[extreme])
    metrics["prostate_dice_central_slices"] = dice(p_pred[central], p_true[central])
    metrics["n_extreme_slices"], metrics["n_central_slices"] = len(extreme), len(central)

    # 4. Leakage into neighbouring organs.
    predicted = p_pred.sum()
    metrics["leak_into_rectum"] = float((p_pred & (true == RECTUM_LABEL)).sum() / max(predicted, 1))
    metrics["leak_into_bladder"] = float((p_pred & (true == BLADDER_LABEL)).sum() / max(predicted, 1))

    # 5. Spurious prostate: predicted on slices whose ground truth has none.
    no_prostate = ~p_true.reshape(len(p_true), -1).any(1)
    spurious = p_pred.reshape(len(p_pred), -1).any(1) & no_prostate
    metrics["spurious_prostate_slices"] = int(spurious.sum())
    metrics["slices_without_prostate"] = int(no_prostate.sum())

    return metrics, volumes


def slice_scores(pred, true, min_pixels=20):
    """Prostate Dice per slice, only for slices with a non-trivial prostate."""
    scores = []
    for i in range(len(true)):
        t = true[i] == PROSTATE_LABEL
        p = pred[i] == PROSTATE_LABEL
        if t.sum() >= min_pixels or p.sum() >= min_pixels:
            scores.append((dice(p, t), i))
    return sorted(scores)


def slice_records(pred, true, names, volumes):
    """
    One record per slice whose ground truth or prediction contains prostate.

    Saved as per_slice.json so failures can be grouped and analysed across
    models: prostate size in the slice, edge versus central position, and
    how many missed prostate pixels were labelled as rectum or bladder.
    """
    edge = set()
    for idx in volumes.values():
        with_prostate = [i for i in idx if (true[i] == PROSTATE_LABEL).any()]
        edge.update(with_prostate[:EXTREME_SLICES] + with_prostate[-EXTREME_SLICES:])

    records = []
    for i, name in enumerate(names):
        t = true[i] == PROSTATE_LABEL
        p = pred[i] == PROSTATE_LABEL
        if not t.any() and not p.any():
            continue
        vol, slice_no = parse_name(name)
        records.append({
            "volume": vol,
            "slice": slice_no,
            "gt_pixels": int(t.sum()),
            "pred_pixels": int(p.sum()),
            "dice": float(dice(p, t)),
            "edge": i in edge,
            "missed_as_rectum": int((t & (pred[i] == RECTUM_LABEL)).sum()),
            "missed_as_bladder": int((t & (pred[i] == BLADDER_LABEL)).sum()),
        })
    return records

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def error_overlay(image, pred, true):
    """RGB image: grey MRI with prostate false positives red, false negatives blue."""
    grey = (image - image.min()) / (np.ptp(image) + 1e-8)
    rgb = np.stack([grey] * 3, axis=-1)
    p, t = pred == PROSTATE_LABEL, true == PROSTATE_LABEL
    rgb[p & t] = [0.2, 0.8, 0.2]  # true positive: green
    rgb[p & ~t] = [0.9, 0.1, 0.1]  # false positive: red
    rgb[~p & t] = [0.1, 0.4, 1.0]  # false negative: blue
    return rgb


def plot_slice(ax_row, image, true, pred, title):
    """Fill one row of 4 axes: input, ground truth, prediction, error overlay."""
    panels = [
        (for_display(image), "gray", "input"),
        (for_display(true), MASK_CMAP, "ground truth"),
        (for_display(pred), MASK_CMAP, "prediction"),
        (for_display(error_overlay(image, pred, true)), None, "prostate errors"),
    ]
    for ax, (data, cmap, label) in zip(ax_row, panels):
        kwargs = {"cmap": cmap, "interpolation": "nearest"}
        if cmap is MASK_CMAP:
            kwargs.update(vmin=0, vmax=NUM_CLASSES - 1)
        ax.imshow(data, **kwargs)
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    ax_row[0].set_title(f"{title}\ninput", fontsize=8)


def save_examples(images, true, pred, indices, names, path):
    fig, axes = plt.subplots(len(indices), 4, figsize=(10, 2.6 * len(indices)), squeeze=False)
    for row, i in zip(axes, indices):
        plot_slice(row, images[i], true[i], pred[i], Path(names[i]).name.replace(".nii.gz", ""))
    fig.text(0.5, 0.005, "errors: green = correct prostate, red = false positive, blue = false negative",
            ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained HipMRI segmentation model.")
    parser.add_argument("--root", default=r"D:\datasets\HipMRI\keras_slices_data")
    parser.add_argument("--model", default="unet", help="model name (see train.build_model)")
    parser.add_argument("--checkpoint", default=None, help="default outputs/<model>/best_model.pt")
    parser.add_argument("--split", default="test", choices=["validate", "test"])
    parser.add_argument("--out", default=None, help="default outputs/<model>/eval")
    parser.add_argument("--n-failures", type=int, default=5)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint or f"outputs/{args.model}/best_model.pt")
    out_dir = Path(args.out or checkpoint.parent / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HipMRIDataset(args.root, args.split, augment=False)
    names = [str(img) for img, _ in dataset.pairs]
    images = dataset.images.numpy()
    true = dataset.masks.numpy()

    model = build_model(args.model).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    pred, sec_per_volume, peak_vram = predict_split(model, dataset, device)

    metrics, volumes = compute_metrics(pred, true, names)
    metrics.update({
        "model": args.model,
        "checkpoint": str(checkpoint),
        "split": args.split,
        "parameters": count_parameters(model),
        "inference_seconds_per_volume": sec_per_volume,
        "peak_inference_vram_gb": peak_vram,
    })

    # Figures: representative central slices and the worst prostate slices.
    scores = slice_scores(pred, true)
    failures = [i for d, i in scores[:args.n_failures]]
    for k, i in enumerate(failures, start=1):
        save_examples(images, true, pred, [i], names, out_dir / f"failure_{k}.png")
    metrics["failure_cases"] = [
        {"rank": k, "slice": Path(names[i]).name, "prostate_dice": d}
        for k, (d, i) in enumerate(scores[:args.n_failures], start=1)
    ]

    middle_slices = [idx[len(idx) // 2] for idx in list(volumes.values())[:3]]
    save_examples(images, true, pred, middle_slices, names, out_dir / "examples.png")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    records = slice_records(pred, true, names, volumes)
    (out_dir / "per_slice.json").write_text(json.dumps(records, indent=1))

    # ----- Summary -----
    print(f"\n=== {args.model} on {args.split} ({len(dataset)} slices, {len(volumes)} volumes) ===")
    print("Dice: " + "  ".join(f"{k}={v:.3f}" for k, v in metrics["dice"].items()))
    print("IoU : " + "  ".join(f"{k}={v:.3f}" for k, v in metrics["iou"].items()))
    print("\nper-volume prostate Dice:")
    for v, d in metrics["per_volume_prostate_dice"].items():
        print(f"  {v}: {d:.3f}")
    print(f"worst volume: {metrics['worst_volume_prostate_dice']:.3f}")
    print(f"\nprostate Dice, extreme slices ({metrics['n_extreme_slices']}): "
        f"{metrics['prostate_dice_extreme_slices']:.3f}")
    print(f"prostate Dice, central slices ({metrics['n_central_slices']}): "
        f"{metrics['prostate_dice_central_slices']:.3f}")
    print(f"leakage into rectum: {metrics['leak_into_rectum']:.2%}   "
        f"into bladder: {metrics['leak_into_bladder']:.2%}")
    print(f"spurious prostate on {metrics['spurious_prostate_slices']} of "
        f"{metrics['slices_without_prostate']} prostate-free slices")
    print(f"\nparameters: {metrics['parameters']:,}   "
        f"inference: {sec_per_volume:.3f} s / 60-slice volume   peak VRAM: {peak_vram:.2f} GB")
    print("worst slices:")
    for f in metrics["failure_cases"]:
        print(f"  #{f['rank']} {f['slice']}: Dice {f['prostate_dice']:.3f}")
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()