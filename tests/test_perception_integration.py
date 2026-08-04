"""Does learned perception actually beat the classical detector on REAL eyes?

This is the integration question the unit tests cannot answer: the analyzer
prefers the segmenter when a checkpoint exists, so the checkpoint must be
demonstrably better on genuine webcam crops, not just on synthetic ones.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from visionscreen.ml.infer import EyeSegmenter
from visionscreen.perception.iris import detect_corneal_reflex

CORPUS = Path("data/corpus/real")
CKPT = Path("models/eyesegnet.pt")


def _real_samples(limit: int = 120):
    img_dir, mask_dir = CORPUS / "images", CORPUS / "masks"
    if not img_dir.is_dir():
        return []
    out = []
    for p in sorted(img_dir.glob("*.png"))[:limit]:
        m = mask_dir / p.name
        if m.exists():
            out.append((cv2.imread(str(p), 0), cv2.imread(str(m), 0)))
    return out


@pytest.mark.skipif(not CKPT.exists(), reason="no trained checkpoint")
def test_checkpoint_loads_and_reports_device():
    seg = EyeSegmenter(checkpoint=CKPT, device="cpu")
    assert seg.available is True
    assert seg.device == "cpu"


@pytest.mark.skipif(not CKPT.exists(), reason="no trained checkpoint")
def test_learned_pupil_detection_rate_on_real_eyes():
    """The net must find a pupil in the large majority of real crops."""
    samples = _real_samples()
    if len(samples) < 20:
        pytest.skip("real corpus not built")
    seg = EyeSegmenter(checkpoint=CKPT, device="cpu")
    found = sum(1 for img, _ in samples if (r := seg.segment(img)) and r.pupil_center)
    rate = found / len(samples)
    assert rate > 0.85, f"pupil detection rate only {rate:.2f}"


@pytest.mark.skipif(not CKPT.exists(), reason="no trained checkpoint")
def test_learned_pupil_center_agrees_with_reference():
    """Centres must agree with the weak-label reference within a pupil radius."""
    samples = _real_samples()
    if len(samples) < 20:
        pytest.skip("real corpus not built")
    seg = EyeSegmenter(checkpoint=CKPT, device="cpu")
    errors = []
    for img, mask in samples:
        ys, xs = np.nonzero(mask == 3)
        if len(xs) < 10:
            continue
        res = seg.segment(img)
        if res is None or res.pupil_center is None:
            continue
        ref_r = max(np.sqrt(len(xs) / np.pi), 2.0)
        d = np.hypot(res.pupil_center[0] - xs.mean(), res.pupil_center[1] - ys.mean())
        errors.append(d / ref_r)
    assert len(errors) >= 15
    median_err = float(np.median(errors))
    assert median_err < 1.0, f"median centre error {median_err:.2f} pupil radii"


@pytest.mark.skipif(not CKPT.exists(), reason="no trained checkpoint")
def test_learned_beats_classical_reflex_detection_on_real_eyes():
    """The headline claim: learned perception recovers reflexes the
    threshold detector misses on real webcam frames."""
    samples = _real_samples()
    if len(samples) < 20:
        pytest.skip("real corpus not built")
    seg = EyeSegmenter(checkpoint=CKPT, device="cpu")

    learned_hits = classical_hits = truth_count = 0
    for img, mask in samples:
        has_reflex = (mask == 4).sum() >= 3
        if not has_reflex:
            continue
        truth_count += 1
        ys, xs = np.nonzero(mask == 2)
        if len(xs) < 10:
            continue
        cx, cy = xs.mean(), ys.mean()
        radius = max(np.sqrt(len(xs) / np.pi), 3.0)

        res = seg.segment(img)
        if res is not None and res.reflex_center is not None:
            learned_hits += 1
        if detect_corneal_reflex(img, center_xy=(cx, cy), radius_px=radius) is not None:
            classical_hits += 1

    if truth_count < 10:
        pytest.skip("too few reflex-bearing crops in corpus")
    assert learned_hits >= classical_hits, (
        f"learned {learned_hits} < classical {classical_hits} of {truth_count}"
    )
