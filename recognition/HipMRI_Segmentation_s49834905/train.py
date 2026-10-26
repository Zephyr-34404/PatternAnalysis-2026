"""
train.py — Train, validate, test and save a HipMRI 2D segmentation model.

Training uses a combined loss (cross-entropy + soft Dice). After every epoch
the model is evaluated on the validation split; the checkpoint with the best
validation prostate Dice is kept. At the end, that checkpoint is evaluated
once on the held-out test split.

Outputs (written to --out, which is git-ignored):
    best_model.pt     weights of the best validation epoch
    history.json      per-epoch losses and Dice scores
    curves.png        loss and Dice curves
    test_results.json per-class test Dice of the best checkpoint

Usage:
    python train.py --epochs 30
    python train.py --root D:\\datasets\\HipMRI\\keras_slices_data --epochs 30 --out outputs/unet
"""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to file without opening a window
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import LABEL_NAMES, NUM_CLASSES, PROSTATE_LABEL, get_dataloaders
from modules import UNet, count_parameters


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed):
    """Fix random seeds so runs are repeatable (up to GPU non-determinism)."""
    # Fixed random seed: shuffling, flipping, and parameter initialization are the same every timeand
    # the results are reproducible
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def build_model(name):
    """Return a fresh model by name. New architectures are added here."""
    # When adding Improved U-Net later, just add an extra branch here and select --model improved on the command line.
    if name == "unet":
        return UNet(in_channels=1, num_classes=NUM_CLASSES, base_channels=32)
    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DiceCELoss(nn.Module):
    """
    Cross-entropy plus multi-class soft Dice loss.

    Cross-entropy gives stable per-pixel gradients early in training; soft
    Dice weights every class equally regardless of size, which counters the
    heavy imbalance (prostate is about 0.4% of training pixels).

    Soft Dice per class c, over the whole batch:
        Dice_c = (2 * sum(p_c * g_c) + eps) / (sum(p_c) + sum(g_c) + eps)
    where p_c is the softmax probability and g_c the one-hot ground truth.
    The loss is 1 - mean_c(Dice_c).
    """

    def __init__(self, num_classes, dice_weight=1.0, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        probs = F.softmax(logits.float(), dim=1)
        onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * onehot).sum(dims)
        denominator = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * intersection + self.eps) / (denominator + self.eps)

        return ce + self.dice_weight * (1 - dice.mean())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class DiceMeter:
    """
    Accumulates hard (argmax) Dice per class over an entire split.

    Intersections and sizes are summed across all batches before dividing,
    so each class's Dice reflects the whole split rather than an average
    of per-batch scores (which would be unstable for small structures).
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.intersection = torch.zeros(num_classes)
        self.pred_sum = torch.zeros(num_classes)
        self.true_sum = torch.zeros(num_classes)

    def update(self, logits, target):
        pred = logits.argmax(dim=1)  # (B, H, W)
        for c in range(self.num_classes):
            p = pred == c
            t = target == c
            self.intersection[c] += (p & t).sum().item()
            self.pred_sum[c] += p.sum().item()
            self.true_sum[c] += t.sum().item()

    def dice(self):
        """Per-class Dice as a list of floats."""
        denom = self.pred_sum + self.true_sum
        scores = torch.where(denom > 0, 2 * self.intersection / denom.clamp(min=1), torch.ones_like(denom))
        return scores.tolist()


# ---------------------------------------------------------------------------
# One pass over the data
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Train for one epoch; return the mean training loss."""
    model.train()
    total_loss, n_batches = 0.0, 0

    for images, masks in loader:
        # Move data to gpu
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate without updating weights; return (mean loss, per-class Dice)."""
    model.eval()
    meter = DiceMeter(NUM_CLASSES)
    total_loss, n_batches = 0.0, 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)
        total_loss += loss.item()
        n_batches += 1
        meter.update(logits.float().cpu(), masks.cpu())

    return total_loss / n_batches, meter.dice()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_curves(history, path):
    """Save training/validation loss and validation Dice curves."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("Dice + CE loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [h["val_dice"][PROSTATE_LABEL] for h in history],
                 label="prostate", linewidth=2)
    axes[1].plot(epochs, [sum(h["val_dice"][1:]) / (NUM_CLASSES - 1) for h in history],
                 label="mean (excl. background)", linestyle="--")
    axes[1].axhline(0.75, color="grey", linestyle=":", label="target 0.75")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Validation Dice")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def format_dice(scores):
    return "  ".join(f"{name}={d:.3f}" for name, d in zip(LABEL_NAMES, scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a HipMRI 2D segmentation model.")
    parser.add_argument("--root", default=r"D:\datasets\HipMRI\keras_slices_data",
                        help="folder containing the keras_slices_* subfolders")
    parser.add_argument("--model", default="unet", help="model name (see build_model)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="output folder (default outputs/<model>)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out or f"outputs/{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}  output: {out_dir}")

    train_loader, val_loader, test_loader = get_dataloaders(args.root, args.batch_size)

    model = build_model(args.model).to(device)
    print(f"model: {args.model}, {count_parameters(model):,} trainable parameters")

    criterion = DiceCELoss(NUM_CLASSES)
    # Adam: The most commonly used optimizer automatically adjusts the step size for each parameter.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    history = []
    best_dice = -1.0
    best_path = out_dir / "best_model.pt"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_dice = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_dice": val_dice, "seconds": elapsed})

        prostate = val_dice[PROSTATE_LABEL]
        flag = ""
        # Only save when the validation set prostate Dice hits a new high, getting the best on the verification set
        if prostate > best_dice:
            best_dice = prostate
            torch.save(model.state_dict(), best_path)
            flag = "  * saved"
        print(f"epoch {epoch:3d}/{args.epochs}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"prostate Dice {prostate:.3f}  ({elapsed:.0f}s){flag}")

        # Save history and curves every epoch so an interrupted run still has them.
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        plot_curves(history, out_dir / "curves.png")

    peak_vram = (torch.cuda.max_memory_allocated() / 1024**3) if device.type == "cuda" else 0.0

    # ----- Final test on the best checkpoint -----
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_loss, test_dice = evaluate(model, test_loader, criterion, device)

    results = {
        "model": args.model,
        "parameters": count_parameters(model),
        "epochs": args.epochs,
        "best_val_prostate_dice": best_dice,
        "test_loss": test_loss,
        "test_dice": dict(zip(LABEL_NAMES, test_dice)),
        "peak_train_vram_gb": peak_vram,
        "mean_epoch_seconds": sum(h["seconds"] for h in history) / len(history),
        "args": vars(args),
    }
    (out_dir / "test_results.json").write_text(json.dumps(results, indent=2))

    print("\n=== Test results (best validation checkpoint) ===")
    print(format_dice(test_dice))
    print(f"prostate Dice: {test_dice[PROSTATE_LABEL]:.3f}  (target >= 0.75)")
    print(f"peak training VRAM: {peak_vram:.2f} GB")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()