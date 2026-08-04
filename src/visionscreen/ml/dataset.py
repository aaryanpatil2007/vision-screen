from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from visionscreen.synth.realistic import render_labeled_eye, sample_params


def _to_tensors(img: np.ndarray, mask: np.ndarray, out_size: tuple[int, int]):
    h, w = out_size
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
    y = torch.from_numpy(mask.astype(np.int64))
    return x, y


class SyntheticEyeDataset(Dataset):
    """Domain-randomized labeled eye crops, generated deterministically per index."""

    def __init__(self, n: int, seed: int = 0, out_size: tuple[int, int] = (64, 96)):
        self.n = n
        self.seed = seed
        self.out_size = out_size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        img, mask, _ = render_labeled_eye(sample_params(rng), rng)
        return _to_tensors(img, mask, self.out_size)


class RealEyeDataset(Dataset):
    """Real eye crops with masks produced by the annotation pipeline.

    Expects `root/images/*.png` and matching `root/masks/*.png` where mask
    pixel values are class indices. Missing directory yields an empty dataset
    so training works before any real data has been fetched.
    """

    def __init__(self, root: Path, out_size: tuple[int, int] = (64, 96)):
        self.out_size = out_size
        img_dir, mask_dir = Path(root) / "images", Path(root) / "masks"
        self.pairs: list[tuple[Path, Path]] = []
        if img_dir.is_dir() and mask_dir.is_dir():
            for ip in sorted(img_dir.glob("*.png")):
                mp = mask_dir / ip.name
                if mp.exists():
                    self.pairs.append((ip, mp))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ip, mp = self.pairs[idx]
        img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        return _to_tensors(img, mask, self.out_size)


def collate_resize(batch):
    imgs = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    return imgs, masks
