"""Learned-perception inference wrapper.

Segments an eye crop into iris / pupil / corneal reflex and reduces the masks
to the geometric quantities the physics modules consume. Absent or broken
checkpoints degrade to `available=False` so callers fall back to the classical
detectors instead of failing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from visionscreen.ml.model import (
    IRIS_CLASS,
    PUPIL_CLASS,
    REFLEX_CLASS,
    EyeSegNet,
)
from visionscreen.ml.train import DEFAULT_CKPT, pick_device

MIN_PUPIL_PIXELS = 12
MIN_REFLEX_PIXELS = 3


@dataclass
class SegResult:
    pupil_center: tuple[float, float] | None
    pupil_radius_px: float
    iris_center: tuple[float, float] | None
    iris_radius_px: float
    reflex_center: tuple[float, float] | None
    confidence: float
    mask: np.ndarray


def _largest_component_centroid(binary: np.ndarray, min_px: int):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8)
    )
    if n < 2:
        return None, 0.0
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[idx, cv2.CC_STAT_AREA]
    if area < min_px:
        return None, 0.0
    cx, cy = centroids[idx]
    radius = float(np.sqrt(area / np.pi))
    return (float(cx), float(cy)), radius


class EyeSegmenter:
    def __init__(self, checkpoint: str | Path | None = None, device: str | None = None,
                 out_size: tuple[int, int] = (64, 96)):
        self.out_size = out_size
        self.device = pick_device(device)
        self.available = False
        self.model: EyeSegNet | None = None
        path = Path(checkpoint) if checkpoint is not None else DEFAULT_CKPT
        if not path.exists():
            return
        try:
            blob = torch.load(path, map_location=self.device, weights_only=False)
            model = EyeSegNet()
            model.load_state_dict(blob["state_dict"])
            model.to(self.device).eval()
            self.model = model
            self.available = True
        except Exception:
            self.available = False

    def segment(self, crop_gray: np.ndarray) -> SegResult | None:
        if not self.available or crop_gray.size == 0:
            return None
        h0, w0 = crop_gray.shape[:2]
        oh, ow = self.out_size
        small = cv2.resize(crop_gray, (ow, oh), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small.astype(np.float32) / 255.0)[None, None].to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred = probs.argmax(0).cpu().numpy().astype(np.uint8)
            conf = float(probs.max(0).values.mean().item())

        # back to original crop scale
        mask = cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_NEAREST)
        sx, sy = w0 / ow, h0 / oh

        pupil_c, pupil_r = _largest_component_centroid(mask == PUPIL_CLASS, MIN_PUPIL_PIXELS)
        iris_bin = (mask == IRIS_CLASS) | (mask == PUPIL_CLASS) | (mask == REFLEX_CLASS)
        iris_c, iris_r = _largest_component_centroid(iris_bin, MIN_PUPIL_PIXELS)
        reflex_c, _ = _largest_component_centroid(mask == REFLEX_CLASS, MIN_REFLEX_PIXELS)

        # the reflex must lie on the cornea; reject stray speculars
        if reflex_c is not None and iris_c is not None and iris_r > 0:
            if np.hypot(reflex_c[0] - iris_c[0], reflex_c[1] - iris_c[1]) > iris_r * 1.3:
                reflex_c = None

        _ = (sx, sy)  # mask already resized; centroids are in crop coords
        return SegResult(
            pupil_center=pupil_c,
            pupil_radius_px=pupil_r,
            iris_center=iris_c,
            iris_radius_px=iris_r,
            reflex_center=reflex_c,
            confidence=conf,
            mask=mask,
        )
