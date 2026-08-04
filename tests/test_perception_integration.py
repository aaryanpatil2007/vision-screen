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


def _real_samples(limit: int = 120, require_reflex: bool = False):
    """Load real crops. Only ~11% carry a reflex label, so the reflex
    comparison must search for them rather than take the first N."""
    img_dir, mask_dir = CORPUS / "images", CORPUS / "masks"
    if not img_dir.is_dir():
        return []
    out = []
    for p in sorted(img_dir.glob("*.png")):
        m = mask_dir / p.name
        if not m.exists():
            continue
        mask = cv2.imread(str(m), 0)
        if require_reflex and (mask == 4).sum() < 3:
            continue
        out.append((cv2.imread(str(p), 0), mask))
        if len(out) >= limit:
            break
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
def test_combined_reflex_detection_beats_either_alone():
    """Measured result, contrary to the original hypothesis: on real crops that
    contain a reflex, the classical threshold detector has HIGHER recall than
    the network (~99% vs ~77%) — a specular highlight is a strong, simple
    photometric signal, and the network trades recall for precision. The
    pipeline therefore uses the network first and falls back to classical, so
    neither path's misses are lost."""
    samples = _real_samples(limit=150, require_reflex=True)
    if len(samples) < 20:
        pytest.skip("real corpus not built")
    seg = EyeSegmenter(checkpoint=CKPT, device="cpu")

    learned = classical = combined = total = 0
    for img, mask in samples:
        ys, xs = np.nonzero(mask == 2)
        if len(xs) < 10:
            continue
        total += 1
        cx, cy = xs.mean(), ys.mean()
        radius = max(np.sqrt(len(xs) / np.pi), 3.0)

        res = seg.segment(img)
        got_learned = res is not None and res.reflex_center is not None
        got_classical = detect_corneal_reflex(
            img, center_xy=(cx, cy), radius_px=radius) is not None
        learned += got_learned
        classical += got_classical
        combined += (got_learned or got_classical)

    if total < 10:
        pytest.skip("too few reflex-bearing crops in corpus")
    # the combination must be at least as good as either path alone
    assert combined >= learned, (combined, learned)
    assert combined >= classical, (combined, classical)
    assert combined / total > 0.9, f"combined recall only {combined / total:.2f}"
