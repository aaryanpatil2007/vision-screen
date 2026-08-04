"""The OpenEDS benchmark contract — the comparison must not silently drift.

Published target: RITnet, 95.3% mIoU on the OpenEDS 2019 Semantic Segmentation
Challenge with 248,900 parameters (arXiv:1910.00694, Table 1). These tests pin
the metric definition, the split sizes, and the model budget so a later change
cannot quietly make "we beat SOTA" mean something different.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from visionscreen.ml.openeds import (
    CLASS_NAMES,
    N_CLASSES,
    RITNET_MIOU,
    RITNET_PARAMS,
    evaluate_miou,
    preprocess,
)
from visionscreen.ml.segnet_dense import (
    CombinedLoss,
    DenseEyeNet,
    boundary_loss,
    generalized_dice_loss,
)

ZIP = Path("data/openeds/seg.zip")


def test_benchmark_constants_match_the_paper():
    assert RITNET_MIOU == 0.953
    assert RITNET_PARAMS == 248_900
    assert N_CLASSES == 4
    assert CLASS_NAMES == ("background", "sclera", "iris", "pupil")


def test_model_fits_within_the_ritnet_budget():
    """Matching accuracy only counts if we do it at comparable size — the
    challenge scored mIoU + min(1/S, 1)."""
    m = DenseEyeNet(4, width=28, growth=10)
    n = sum(p.numel() for p in m.parameters())
    assert n < RITNET_PARAMS, f"{n:,} params exceeds RITnet's {RITNET_PARAMS:,}"
    assert n * 4 / 1e6 < 0.98


def test_miou_metric_matches_the_challenge_definition():
    """Mean IoU over all four classes, accumulated across the split rather than
    averaged per image — a per-image average would inflate the score on frames
    where a rare class is absent."""

    class Perfect(torch.nn.Module):
        def forward(self, x):
            return self._logits

    target = torch.zeros(2, 8, 8, dtype=torch.long)
    target[:, :4, :4] = 1
    target[:, 4:, :4] = 2
    target[:, 4:, 4:] = 3
    logits = torch.nn.functional.one_hot(target, 4).permute(0, 3, 1, 2).float() * 20

    m = Perfect()
    m._logits = logits
    res = evaluate_miou(m, [(torch.zeros(2, 1, 8, 8), target)], "cpu")
    assert res["mIoU"] == pytest.approx(1.0)
    assert res["pixel_acc"] == pytest.approx(1.0)

    # a model that never predicts the rare class must be punished by mIoU
    # even though pixel accuracy stays high
    half = logits.clone()
    half[:, 3] = -20                      # never predict pupil
    half[:, 0, 4:, 4:] = 20               # call it background instead
    m._logits = half
    res2 = evaluate_miou(m, [(torch.zeros(2, 1, 8, 8), target)], "cpu")
    assert res2["pupil_iou"] == 0.0
    assert res2["mIoU"] < 0.8
    assert res2["pixel_acc"] > res2["mIoU"]


def test_generalized_dice_weights_rare_classes():
    """Pupil is ~1% of pixels; unweighted dice lets a model ignore it."""
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, :2, :2] = 3                          # a tiny pupil
    ignore = torch.zeros(1, 4, 16, 16)
    ignore[:, 0] = 10                              # predict all background
    perfect = torch.nn.functional.one_hot(target, 4).permute(0, 3, 1, 2).float() * 10
    assert generalized_dice_loss(ignore, target) > generalized_dice_loss(perfect, target)
    assert generalized_dice_loss(perfect, target) < 0.15


def test_boundary_loss_focuses_on_edges():
    """Residual mIoU lives in a few pixels per edge; a uniform loss spends its
    gradient on easy interiors."""
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, :, 8:] = 1
    logits = torch.nn.functional.one_hot(target, 4).permute(0, 3, 1, 2).float() * 10
    edge_wrong = logits.clone()
    edge_wrong[:, :, :, 7:9] = 0                   # break only the boundary
    interior_wrong = logits.clone()
    interior_wrong[:, :, :, 0:2] = 0               # break the same area, interior
    assert boundary_loss(edge_wrong, target) > boundary_loss(interior_wrong, target)


def test_combined_loss_is_finite_and_differentiable():
    m = DenseEyeNet(4, width=16, growth=6)
    x = torch.rand(2, 1, 64, 96)
    y = torch.randint(0, 4, (2, 64, 96))
    loss = CombinedLoss()(m(x), y)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_preprocess_is_contrast_normalising():
    img = (np.linspace(40, 90, 64 * 64).reshape(64, 64)).astype(np.uint8)
    out = preprocess(img)
    assert out.dtype == np.uint8
    assert out.std() > img.std()          # CLAHE must expand a low-contrast ramp


def test_model_output_matches_input_resolution():
    m = DenseEyeNet(4, width=16, growth=6)
    for h, w in ((192, 320), (200, 320), (61, 97)):
        assert m(torch.zeros(1, 1, h, w)).shape == (1, 4, h, w)


@pytest.mark.skipif(not ZIP.exists(), reason="OpenEDS archive not downloaded")
def test_official_split_sizes():
    """8,916 / 2,403 / 1,440 as published — a wrong split invalidates the
    comparison entirely."""
    import zipfile

    with zipfile.ZipFile(ZIP) as zf:
        names = zf.namelist()
    for split, expected in (("train", 8916), ("validation", 2403), ("test", 1440)):
        n = sum(1 for x in names
                if f"/{split}/images/" in x and x.endswith(".png"))
        assert n == expected, f"{split}: {n} != {expected}"


@pytest.mark.skipif(not ZIP.exists(), reason="OpenEDS archive not downloaded")
def test_loader_yields_four_class_labels():
    from visionscreen.ml.openeds import OpenEDSDataset

    ds = OpenEDSDataset(ZIP, "train", limit=8)
    x, y = ds[0]
    assert x.shape == (1, 192, 320)
    assert y.shape == (192, 320)
    assert set(y.unique().tolist()) <= {0, 1, 2, 3}
