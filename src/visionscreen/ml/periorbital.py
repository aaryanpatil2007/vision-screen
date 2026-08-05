"""Periorbital segmentation on visible-light consumer-camera photographs.

This is the benchmark that actually matches what this project does, and
replacing OpenEDS with it was a correction, not an addition.

OpenEDS — where the earlier work here was aimed — is 640x400 near-infrared
imagery from cameras mounted a couple of centimetres from the cornea inside a
VR headset, cropped to one eye. This system takes visible-light frames of a
whole face from a webcam roughly half a metre away. Those differ in spectrum,
in optics, in framing and in pixel statistics, so a score on one says almost
nothing about the other, and claiming state of the art on OpenEDS for a webcam
tool would be a category error. Two further facts made OpenEDS a dead end
regardless: its challenge test labels were never released, and its evaluation
server is gone, which is why no paper since about 2020 reports the 95.3% figure
at all.

The dataset here is the periorbital segmentation set of Nahass et al.
(*Ophthalmology Science* 2025, DOI 10.1016/j.xops.2025.100757; data at Zenodo
10.5281/zenodo.13916845, CC BY 4.0), built from two face-photograph corpora:

* **CFD** — the Chicago Face Database, 827 studio portraits.
* **CelebAMask-HQ** — 2,015 in-the-wild celebrity photographs, which are far
  closer to webcam conditions: uncontrolled lighting, pose and image quality.

Five annotated structures plus background. The published reference is a
DeepLabV3-ResNet101 (~60 M parameters), and its per-class Dice is what this is
measured against — see `REFERENCE_DICE`.

Class indices are not documented in the archive and were recovered from mask
geometry rather than assumed, since guessing wrong would silently invert the
whole comparison:

    1 brow      two elongated blobs, circularity 0.17, centroid high in frame
    2 sclera    ~6.6 blobs, because it fragments into crescents either side of
                each iris — the signature that identifies it unambiguously
    3 iris      two blobs, circularity 0.66, the only round class
    4 caruncle  two blobs, ~90 px, by far the smallest
    5 lid       two elongated blobs, circularity 0.15

Sclera and caruncle are the classes worth watching. Sclera is the hardest
structure in every eye-segmentation benchmark in the literature — 0.674 IoU in
the OpenEDS 2020 baseline, 0.074 for zero-shot SAM 2 — because its boundary
with the lid is a soft shadow rather than an edge. Caruncle is simply tiny, and
a few pixels of error move its Dice a long way.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path("data/periorbital/periorbital_dataset")

N_CLASSES = 6
CLASS_NAMES = ("background", "brow", "sclera", "iris", "caruncle", "lid")

#: Nahass et al. 2025, DeepLabV3-ResNet101 (~60M params), per-class Dice.
#: The combined set is the harder and more representative of the two, because
#: CelebAMask-HQ contributes uncontrolled in-the-wild photographs.
REFERENCE_DICE = {
    "combined": {"iris": 0.93, "sclera": 0.81, "lid": 0.79, "caruncle": 0.65},
    "cfd": {"iris": 0.96, "sclera": 0.88, "brow": 0.90, "lid": 0.87,
            "caruncle": 0.78},
}
REFERENCE_MODEL = "DeepLabV3-ResNet101 (Nahass et al., Ophthalmol Sci 2025)"
REFERENCE_PARAMS = 60_000_000       # approximate, for the efficiency comparison

SUBSETS = {
    "combined": ("combined_final_data/combined_output_images",
                 "combined_final_data/combined_output_masks"),
    "cfd": ("cfd_final_data/cfd_output_images", "cfd_final_data/cfd_output_masks"),
    "celeb": ("celeb_final_data/celeb_output_images",
              "celeb_final_data/celeb_output_masks"),
}


def pairs(subset: str = "combined", root: Path = ROOT) -> list[tuple[Path, Path]]:
    """Image/mask pairs that both exist — the archive ships more images than masks."""
    img_dir, msk_dir = (root / p for p in SUBSETS[subset])
    if not img_dir.exists():
        return []
    out = []
    for m in sorted(msk_dir.glob("*.png")):
        for ext in (".jpg", ".jpeg", ".png"):
            i = img_dir / (m.stem + ext)
            if i.exists():
                out.append((i, m))
                break
    return out


def split_pairs(items: list, val_frac: float = 0.2, seed: int = 0):
    """Deterministic 80/20 split, matching the reference paper's protocol.

    Split on a shuffled index rather than on filename order: the archive is
    ordered by source corpus, so a positional split would put CFD in train and
    CelebA in test and measure domain transfer instead of segmentation.
    """
    idx = np.arange(len(items))
    np.random.default_rng(seed).shuffle(idx)
    cut = int(len(items) * (1.0 - val_frac))
    return [items[i] for i in idx[:cut]], [items[i] for i in idx[cut:]]


class PeriorbitalDataset(Dataset):
    def __init__(self, items: list[tuple[Path, Path]],
                 size: tuple[int, int] = (128, 320), augment: bool = False,
                 native_labels: bool = False):
        self.items = items
        self.size = size
        self.augment = augment
        self.native_labels = native_labels

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        ip, mp = self.items[i]
        img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        lab = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if img is None or lab is None:
            raise RuntimeError(f"unreadable pair: {ip}")
        if lab.ndim == 3:
            lab = lab[..., 0]

        h, w = self.size
        img_r = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        lab_r = lab if self.native_labels else cv2.resize(
            lab, (w, h), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img_r, lab_r = self._augment(img_r, lab_r)

        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img_r.transpose(2, 0, 1).astype(np.float32) / 255.0)
        y = torch.from_numpy(lab_r.astype(np.int64))
        return x, y

    def _augment(self, img: np.ndarray, lab: np.ndarray):
        rng = np.random
        h, w = lab.shape
        if rng.random() < 0.5:
            # a horizontal flip swaps left and right structures, but the classes
            # here are already merged across sides, so the labels stay valid
            img, lab = img[:, ::-1].copy(), lab[:, ::-1].copy()
        if rng.random() < 0.6:
            ang = rng.uniform(-10, 10)
            sc = rng.uniform(0.92, 1.10)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
            M[0, 2] += rng.uniform(-0.04, 0.04) * w
            M[1, 2] += rng.uniform(-0.04, 0.04) * h
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            lab = cv2.warpAffine(lab, M, (w, h), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if rng.random() < 0.5:                    # exposure and white balance
            img = np.clip(img.astype(np.float32)
                          * rng.uniform(0.75, 1.3, size=(1, 1, 3)), 0, 255).astype(np.uint8)
        if rng.random() < 0.25:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.3))
        if rng.random() < 0.25:
            img = np.clip(img.astype(np.float32)
                          + rng.normal(0, rng.uniform(2, 10), img.shape), 0, 255).astype(np.uint8)
        return img, lab


@torch.no_grad()
def evaluate_dice(model, loader, device: str, native: bool = False,
                  tta: bool = False) -> dict:
    """Per-class Dice, the metric the reference paper reports.

    Dice is accumulated over the whole split rather than averaged per image.
    A per-image mean would be dominated by frames where a small class is barely
    present, and caruncle occupies about 0.3% of pixels — on those frames a
    handful of pixels would swing the average wildly.
    """
    import torch.nn.functional as F

    model = model.to(device).eval()
    inter = np.zeros(N_CLASSES, np.float64)
    total = np.zeros(N_CLASSES, np.float64)
    inter_i = np.zeros(N_CLASSES, np.float64)
    union_i = np.zeros(N_CLASSES, np.float64)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if tta:
            logits = logits + torch.flip(model(torch.flip(x, [-1])), [-1])
        if native and logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear",
                                   align_corners=False)
        pred = logits.argmax(1)
        for c in range(N_CLASSES):
            p, t = pred == c, y == c
            inter[c] += (p & t).sum().item()
            total[c] += p.sum().item() + t.sum().item()
            inter_i[c] += (p & t).sum().item()
            union_i[c] += (p | t).sum().item()

    dice = np.where(total > 0, 2.0 * inter / np.maximum(total, 1), np.nan)
    iou = np.where(union_i > 0, inter_i / np.maximum(union_i, 1), np.nan)
    out = {f"{CLASS_NAMES[c]}_dice": round(float(dice[c]), 4)
           for c in range(N_CLASSES)}
    out.update({f"{CLASS_NAMES[c]}_iou": round(float(iou[c]), 4)
                for c in range(N_CLASSES)})
    out["mean_dice"] = round(float(np.nanmean(dice)), 4)
    out["mIoU"] = round(float(np.nanmean(iou)), 4)
    # the structures the reference paper reports, excluding background
    out["mean_dice_structures"] = round(float(np.nanmean(dice[1:])), 4)
    return out


def compare_to_reference(metrics: dict, subset: str = "combined") -> dict:
    """Per-class comparison against the published DeepLabV3 numbers."""
    ref = REFERENCE_DICE[subset]
    rows = {}
    for name, ref_dice in ref.items():
        ours = metrics.get(f"{name}_dice")
        if ours is None:
            continue
        rows[name] = {"ours": ours, "reference": ref_dice,
                      "delta": round(ours - ref_dice, 4),
                      "beats": ours >= ref_dice}
    rows["_summary"] = {
        "classes_matched_or_beaten": sum(1 for k, v in rows.items()
                                         if not k.startswith("_") and v["beats"]),
        "classes_compared": sum(1 for k in rows if not k.startswith("_")),
    }
    return rows
