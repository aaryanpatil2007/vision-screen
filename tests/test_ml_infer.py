import numpy as np
import pytest
import torch

from visionscreen.ml.infer import EyeSegmenter, SegResult
from visionscreen.ml.model import EyeSegNet
from visionscreen.synth.realistic import EyeParams, render_labeled_eye, sample_params


@pytest.fixture(scope="module")
def trained_segmenter(tmp_path_factory):
    """Train a small net once and wrap it — exercises the real load path."""
    from visionscreen.ml.train import train_model

    model, _ = train_model(
        n_train=600, n_val=32, epochs=5, batch_size=16, out_size=(48, 64),
        seed=3, device="cpu", lr=4e-3, log_every=0,
    )
    ckpt = tmp_path_factory.mktemp("m") / "net.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt)
    return EyeSegmenter(checkpoint=ckpt, device="cpu")


def test_missing_checkpoint_disables_gracefully():
    seg = EyeSegmenter(checkpoint="/nonexistent/path.pt", device="cpu")
    assert seg.available is False
    assert seg.segment(np.zeros((40, 60), np.uint8)) is None


def test_segments_pupil_and_iris(trained_segmenter):
    rng = np.random.default_rng(500)
    img, mask, params = render_labeled_eye(sample_params(rng), rng)
    res = trained_segmenter.segment(img)
    assert isinstance(res, SegResult)
    assert res.pupil_center is not None
    # centre within a quarter-iris of truth
    tol = max(6.0, params.iris_radius * 0.5)
    assert abs(res.pupil_center[0] - params.iris_center[0]) < tol
    assert abs(res.pupil_center[1] - params.iris_center[1]) < tol
    assert res.iris_radius_px > 0


def test_finds_reflex_when_present(trained_segmenter):
    rng = np.random.default_rng(9)
    p = sample_params(rng)
    p = EyeParams(**{**p.__dict__, "reflex_intensity": 1.0,
                     "reflex_offset": (0.3, 0.0), "distractor_specular": 2})
    img, _, _ = render_labeled_eye(p, rng)
    res = trained_segmenter.segment(img)
    assert res.reflex_center is not None
    # the reflex must be found on the cornea, not on a distractor specular
    dist = np.hypot(res.reflex_center[0] - p.iris_center[0],
                    res.reflex_center[1] - p.iris_center[1])
    assert dist <= p.iris_radius * 1.2


def test_confidence_reported(trained_segmenter):
    rng = np.random.default_rng(21)
    img, _, _ = render_labeled_eye(sample_params(rng), rng)
    res = trained_segmenter.segment(img)
    assert 0.0 <= res.confidence <= 1.0


def test_training_is_reproducible_from_its_seed():
    """`seed` once reached the dataset but not torch, so weight initialisation
    was unseeded and every call trained a different network. Invisible in an
    averaged metric, very visible on a single borderline case — it made the
    reflex test pass alone and fail under load."""
    from visionscreen.ml.train import train_model

    def first_weights(seed=11):
        m, _ = train_model(n_train=120, n_val=16, epochs=1, batch_size=16,
                           out_size=(48, 64), seed=seed, device="cpu",
                           lr=4e-3, log_every=0)
        return next(iter(m.state_dict().values())).flatten()[:8].tolist()

    a, b = first_weights(), first_weights()
    for x, y in zip(a, b):
        assert x == pytest.approx(y, abs=1e-4), (a, b)
    # guard the other direction: a seed that changes nothing would mean the fix
    # silently pinned every run to one network
    assert first_weights(11) != first_weights(12)
