import numpy as np
import torch

from visionscreen.ml.dataset import SyntheticEyeDataset, collate_resize
from visionscreen.ml.train import evaluate, train_model


def test_dataset_returns_tensors():
    ds = SyntheticEyeDataset(n=4, seed=0, out_size=(64, 64))
    img, mask = ds[0]
    assert img.shape == (1, 64, 64)
    assert mask.shape == (64, 64)
    assert img.dtype == torch.float32
    assert mask.dtype == torch.int64
    assert 0.0 <= img.min() and img.max() <= 1.0


def test_dataset_is_deterministic_per_seed():
    a = SyntheticEyeDataset(n=2, seed=5, out_size=(32, 32))[1][0]
    b = SyntheticEyeDataset(n=2, seed=5, out_size=(32, 32))[1][0]
    assert torch.allclose(a, b)


def test_collate_stacks_batch():
    ds = SyntheticEyeDataset(n=4, seed=1, out_size=(48, 48))
    imgs, masks = collate_resize([ds[i] for i in range(4)])
    assert imgs.shape == (4, 1, 48, 48)
    assert masks.shape == (4, 48, 48)


def test_training_reduces_loss_and_learns():
    """A short real training run must generalize to unseen held-out eyes.

    Kept small enough for CI; the full run (bench_segmentation) reaches
    pupil IoU ≈ 0.9.
    """
    model, history = train_model(
        n_train=320, n_val=48, epochs=4, batch_size=16, out_size=(48, 64),
        seed=0, device="cpu", lr=4e-3, log_every=0,
    )
    assert history["train_loss"][-1] < history["train_loss"][0]
    metrics = evaluate(model, n=48, seed=99, out_size=(48, 64), device="cpu")
    assert metrics["pupil_iou"] > 0.4, metrics
    assert metrics["iris_iou"] > 0.4, metrics
