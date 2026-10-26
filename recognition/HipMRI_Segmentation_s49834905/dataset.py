"""
dataset.py — Data loading and preprocessing for HipMRI 2D prostate segmentation.

Loads the HipMRI Study 2D MRI slices (keras_slices_data) and their
segmentation masks, normalises the images, and serves them through
PyTorch DataLoaders for training, validation and testing.

Dataset facts (verified during the data audit):
    - 38 patients, 211 3D volumes, 60 axial slices per volume
    - Provided split is patient-disjoint: train 31 / validate 4 / test 3 patients
    - Most slices are 256 x 128; a few wider volumes (e.g. 256 x 144) are centre-cropped
    - Labels: 0 background, 1 body, 2 bone, 3 bladder, 4 rectum, 5 prostate

Usage:
    from dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders("path/to/keras_slices_data")

    # Self-test:
    python dataset.py --root path/to/keras_slices_data
"""

import argparse
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 6          # background, body, bone, bladder, rectum, prostate
PROSTATE_LABEL = 5       # the class the assignment target (Dice >= 0.75) refers to
IMAGE_SHAPE = (256, 128) # (height, width) of every slice
LABEL_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]

SPLITS = ("train", "validate", "test")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def get_pairs(root, split):
    """
    Return a sorted list of (image_path, mask_path) tuples for one split.

    Image files are named case_<patient>_week_<n>_slice_<k>.nii.gz and the
    matching masks seg_<patient>_week_<n>_slice_<k>.nii.gz, so swapping the
    prefix finds the partner file.

    Args:
        root (str | Path): folder containing the keras_slices_* subfolders.
        split (str): one of "train", "validate", "test".

    Returns:
        list[tuple[Path, Path]]

    Raises:
        FileNotFoundError: if a folder is missing, a mask is missing,
            or no files are found at all.
    """
    root = Path(root)
    img_dir = root / f"keras_slices_{split}"
    seg_dir = root / f"keras_slices_seg_{split}"

    for d in (img_dir, seg_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Folder not found: {d}")

    pairs = []
    for img_path in sorted(img_dir.glob("*.nii.gz")):
        seg_path = seg_dir / img_path.name.replace("case_", "seg_", 1)
        if not seg_path.exists():
            raise FileNotFoundError(f"Missing mask for {img_path.name}")
        pairs.append((img_path, seg_path))

    if not pairs:
        raise FileNotFoundError(f"No .nii.gz files found in {img_dir}")
    return pairs

# ---------------------------------------------------------------------------
# Loading and preprocessing
# ---------------------------------------------------------------------------
def load_nifti_2d(path):
    """
    Load one NIfTI slice as a 2D float64 NumPy array.

    Some HipMRI files carry an extra trailing dimension; it is dropped here
    (same handling as load_data_2D in the assessment specification, Appendix B).

    Args:
        path (str | Path): path to a .nii.gz file.

    Returns:
        np.ndarray of shape (H, W).
    """
    arr = nib.load(str(path)).get_fdata(caching="unchanged")
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def normalise(image):
    """
    Per-slice z-score normalisation: zero mean, unit standard deviation.

    Adapted from the normImage option of load_data_2D in the assessment
    specification (Appendix B). The 1e-8 term avoids division by zero on
    blank slices.

    Args:
        image (np.ndarray): raw slice intensities.

    Returns:
        np.ndarray with the same shape.
    """
    return (image - image.mean()) / (image.std() + 1e-8)

def center_crop(arr, shape=IMAGE_SHAPE):
    """
    Centre-crop a 2D array to the target shape.

    A small number of HipMRI volumes have a wider field of view
    (e.g. 256 x 144). Cropping equally from both edges keeps the
    central pelvis, where the prostate lies, and avoids interpolating
    label values as resizing would.
    """
    height, width = arr.shape
    target_h, target_w = shape
    if height < target_h or width < target_w:
        raise ValueError(f"Slice {arr.shape} is smaller than target {shape}")
    top = (height - target_h) // 2
    left = (width - target_w) // 2
    return arr[top:top + target_h, left:left + target_w]

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HipMRIDataset(Dataset):
    """
    PyTorch Dataset for one split of the HipMRI 2D slices.

    All slices are loaded into memory once in __init__ (about 1.9 GB for the
    training split), so each epoch avoids re-decompressing thousands of
    .nii.gz files.

    Each item is a tuple (image, mask):
        image: float32 tensor of shape (1, 256, 128), z-score normalised
        mask:  int64 tensor of shape (256, 128), values in 0..5

    Args:
        root (str | Path): folder containing the keras_slices_* subfolders.
        split (str): "train", "validate" or "test".
        augment (bool): apply random left-right flips. Use for training only.
    """

    def __init__(self, root, split, augment=False):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

        self.split = split
        self.augment = augment
        self.pairs = get_pairs(root, split)

        n = len(self.pairs)
        height, width = IMAGE_SHAPE

        images = np.zeros((n, height, width), dtype=np.float32)
        masks = np.zeros((n, height, width), dtype=np.uint8)

        n_cropped = 0

        for i, (img_path, seg_path) in enumerate(tqdm(self.pairs, desc=f"Loading {split}")):
            image = load_nifti_2d(img_path)
            mask = load_nifti_2d(seg_path)

            if image.shape != mask.shape:
                raise ValueError(
                    f"Image/mask size mismatch in {img_path.name}: "
                    f"{image.shape} vs {mask.shape}"
                )
            if image.shape != IMAGE_SHAPE:
                image = center_crop(image)
                mask = center_crop(mask)
                n_cropped += 1
            
            images[i] = normalise(image)
            masks[i] = np.rint(mask)

        if n_cropped:
            print(f"[{split}] centre-cropped {n_cropped} slices to {IMAGE_SHAPE}")

        self.images = torch.from_numpy(images)
        self.masks = torch.from_numpy(masks)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image = self.images[idx].unsqueeze(0)
        mask = self.masks[idx].long()

        if self.augment:
            image, mask = self._random_flip(image, mask)

        return image, mask

    @staticmethod
    def _random_flip(image, mask):
        """
        Flip image and mask together along the 256-pixel axis with probability 0.5.

        In these slices the 256-pixel axis runs between the left and right
        femoral heads, so this is an anatomical left-right mirror, which the
        roughly symmetric pelvis tolerates. The 128-pixel axis (anterior-
        posterior) is never flipped, since that orientation does not occur
        in real scans.
        """
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[0])
        return image, mask


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def get_dataloaders(root, batch_size=16, num_workers=0):
    """
    Build train, validation and test DataLoaders.

    Only the training loader shuffles and augments; validation and test are
    kept in a fixed order without augmentation so evaluation is deterministic.

    Args:
        root (str | Path): folder containing the keras_slices_* subfolders.
        batch_size (int): slices per batch.
        num_workers (int): worker processes for loading. 0 is fine because
            the data is already in memory (and avoids multiprocessing issues
            on Windows).

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: train, validation, test.
    """
    train_set = HipMRIDataset(root, "train", augment=True)
    val_set = HipMRIDataset(root, "validate", augment=False)
    test_set = HipMRIDataset(root, "test", augment=False)


    pin = torch.cuda.is_available()


    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def check_all_shapes(root):
    """
    Read only the NIfTI headers of every file and report any slice that is
    not 256 x 128. Fast, because pixel data is never loaded.
    """
    root = Path(root)
    bad = []
    total = 0
    for split in SPLITS:
        for sub in (f"keras_slices_{split}", f"keras_slices_seg_{split}"):
            for p in (root / sub).glob("*.nii.gz"):
                total += 1
                if tuple(nib.load(str(p)).shape[:2]) != IMAGE_SHAPE:
                    bad.append(p.name)
    print(f"Header check: {total} files, {len(bad)} with unexpected shape")
    for name in bad[:10]:
        print(f"  {name}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Self-test for the HipMRI data pipeline.")
    parser.add_argument("--root", default=r"D:\datasets\HipMRI\keras_slices_data",
                        help="folder containing the keras_slices_* subfolders")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    # load everything and time it
    start = time.time()
    train_loader, val_loader, test_loader = get_dataloaders(args.root, args.batch_size)
    print(f"\nLoaded all splits in {time.time() - start:.1f} s")

    # split sizes
    expected = {"train": 11460, "validate": 660, "test": 540}
    for name, loader in zip(SPLITS, (train_loader, val_loader, test_loader)):
        n = len(loader.dataset)
        status = "OK" if n == expected[name] else "MISMATCH"
        print(f"{name:>8}: {n:>5} slices  [{status}]")

    # one batch: shapes, dtypes, label range
    images, masks = next(iter(train_loader))
    print(f"\nbatch images: shape={tuple(images.shape)}, dtype={images.dtype}")
    print(f"batch masks : shape={tuple(masks.shape)}, dtype={masks.dtype}")
    print(f"labels in batch: {torch.unique(masks).tolist()}")
    assert masks.min() >= 0 and masks.max() < NUM_CLASSES, "label out of range"

    # How often does the prostate appear? (useful for the README)
    train_masks = train_loader.dataset.masks
    has_prostate = (train_masks == PROSTATE_LABEL).flatten(1).any(1).float().mean().item()
    prostate_pixels = (train_masks == PROSTATE_LABEL).float().mean().item()
    print(f"\ntrain slices containing prostate: {has_prostate:.1%}")
    print(f"prostate share of all train pixels: {prostate_pixels:.2%}")

    # header-only shape check across every file
    print()
    check_all_shapes(args.root)