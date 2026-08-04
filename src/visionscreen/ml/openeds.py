"""OpenEDS 2019 Semantic Segmentation — the benchmark, its metric, its loader.

12,759 near-infrared eye images from a head-mounted display, 640x400, labelled
into four classes: background, sclera, iris, pupil. Split 8,916 / 2,403 / 1,440
(train / validation / test). Test labels are held out by the challenge, so
validation is the reproducible comparison surface.

Published result to beat: **RITnet, 95.3% mIoU**, 248,900 parameters, <1 MB
(Chaudhary et al., arXiv:1910.00694, winner of the 2019 challenge). The
challenge score was mIoU + min(1/S, 1) with S the model size in MB, so
parameter count is part of the target, not an afterthought.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

N_CLASSES = 4                       # background, sclera, iris, pupil
CLASS_NAMES = ("background", "sclera", "iris", "pupil")
RITNET_MIOU = 0.953
RITNET_PARAMS = 248_900


def _clahe(img: np.ndarray, clip: float = 1.5, grid: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation.

    RITnet reports CLAHE plus gamma correction as worth ~0.2% mIoU; these are
    IR images with strong illumination gradients across the eye, so local
    equalisation matters more than it would on a webcam frame.
    """
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(img)


def _gamma(img: np.ndarray, g: float = 0.8) -> np.ndarray:
    lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(img, lut)


def preprocess(img: np.ndarray) -> np.ndarray:
    return _clahe(_gamma(img))


class OpenEDSDataset(Dataset):
    """Reads directly from the challenge zip; no extraction step."""

    def __init__(self, zip_path: Path, split: str = "train",
                 size: tuple[int, int] = (192, 320), augment: bool = False,
                 limit: int | None = None):
        self.zip_path = Path(zip_path)
        self.split = split
        self.size = size
        self.augment = augment
        self._zf: zipfile.ZipFile | None = None

        with zipfile.ZipFile(self.zip_path) as zf:
            names = zf.namelist()
        want = f"/{split}/"
        imgs = sorted(n for n in names
                      if want in n and "/images/" in n and n.endswith(".png"))
        labs = {Path(n).stem: n for n in names
                if want in n and "/labels/" in n and n.endswith(".npy")}
        self.pairs = [(i, labs[Path(i).stem]) for i in imgs if Path(i).stem in labs]
        if limit:
            self.pairs = self.pairs[:limit]

    def _zip(self) -> zipfile.ZipFile:
        # one handle per worker process, opened lazily so the dataset pickles
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path)
        return self._zf

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_name, lab_name = self.pairs[idx]
        zf = self._zip()
        buf = np.frombuffer(zf.read(img_name), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        import io
        lab = np.load(io.BytesIO(zf.read(lab_name)))

        img = preprocess(img)
        h, w = self.size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        lab = cv2.resize(lab.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img, lab = self._augment(img, lab)

        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        y = torch.from_numpy(lab.astype(np.int64))
        return x, y

    def _augment(self, img: np.ndarray, lab: np.ndarray):
        rng = np.random
        h, w = img.shape
        if rng.random() < 0.5:                       # horizontal flip: eyes are chiral
            img, lab = img[:, ::-1].copy(), lab[:, ::-1].copy()
        if rng.random() < 0.6:                       # small affine
            ang = rng.uniform(-12, 12)
            tx, ty = rng.uniform(-0.05, 0.05) * w, rng.uniform(-0.05, 0.05) * h
            sc = rng.uniform(0.92, 1.08)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
            M[0, 2] += tx; M[1, 2] += ty
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            lab = cv2.warpAffine(lab, M, (w, h), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if rng.random() < 0.4:                       # exposure / gamma jitter
            img = np.clip(img.astype(np.float32) * rng.uniform(0.8, 1.25), 0, 255).astype(np.uint8)
        if rng.random() < 0.3:                       # blur, as in defocused frames
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.4))
        if rng.random() < 0.3:                       # sensor noise
            img = np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(2, 9), img.shape),
                          0, 255).astype(np.uint8)
        return img, lab


@torch.no_grad()
def evaluate_miou(model, loader, device: str) -> dict:
    """Challenge metric: mean IoU across the four classes, computed over the
    whole split rather than averaged per image."""
    model = model.to(device).eval()
    inter = np.zeros(N_CLASSES, np.float64)
    union = np.zeros(N_CLASSES, np.float64)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        for c in range(N_CLASSES):
            p, t = pred == c, y == c
            inter[c] += (p & t).sum().item()
            union[c] += (p | t).sum().item()
    ious = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return {
        **{f"{CLASS_NAMES[c]}_iou": round(float(ious[c]), 4) for c in range(N_CLASSES)},
        "mIoU": round(float(np.nanmean(ious)), 4),
        "pixel_acc": round(correct / max(total, 1), 4),
    }
