"""Conjunctival pallor: the physics of the features, and the honesty of the split.

Two things need pinning here. The features must respond to haemoglobin the way
haemoglobin actually behaves — pallor must move the erythema index down, and
exposure changes must *not* move it much, since that is the whole reason for
preferring channel ratios over absolute brightness. And the evaluation must
stay grouped by hospital, because a random split on this dataset would let the
model recognise a site whose prevalence ranges from 48% to 88%.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from visionscreen.ml.anemia import (
    COLLINGS_VALIDATION_SENS,
    COLLINGS_VALIDATION_SPEC,
    FEATURE_NAMES,
    WHO_ANAEMIA_HB,
    conjunctiva_mask,
    extract_features,
    load_index,
)

DATA = Path("data/anemia/cp-anemic")
RESULTS = Path("results/anemia.json")
needs_data = pytest.mark.skipif(not DATA.exists(), reason="CP-AnemiC not downloaded")


def _patch(rgb, size=(80, 200), noise=6, seed=0):
    """A flat conjunctiva-like patch of a given colour, with mild texture."""
    rng = np.random.default_rng(seed)
    img = np.full((*size, 3), 0, np.uint8)
    bgr = (rgb[2], rgb[1], rgb[0])
    base = np.array(bgr, np.float64)
    img = np.clip(base + rng.normal(0, noise, (*size, 3)), 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------- features --

def test_pallor_lowers_the_erythema_index():
    """Haemoglobin absorbs green far more than red, so less of it raises green
    relative to red and drives log10(R/G) down. If this inverted, the model
    would be learning the opposite of the physics."""
    healthy = extract_features(_patch((190, 90, 95)))     # well-perfused, red
    pale = extract_features(_patch((205, 165, 165)))      # pale
    i = FEATURE_NAMES.index("erythema_mean")
    assert pale[i] < healthy[i]


def test_features_are_robust_to_exposure():
    """A channel ratio should largely divide out illumination intensity. This
    is why the features are ratios: the ten hospitals used ten cameras."""
    base = _patch((180, 100, 100))
    dim = np.clip(base.astype(np.float64) * 0.65, 0, 255).astype(np.uint8)
    bright = np.clip(base.astype(np.float64) * 1.35, 0, 255).astype(np.uint8)
    i = FEATURE_NAMES.index("erythema_mean")
    a, b, c = (extract_features(x)[i] for x in (dim, base, bright))
    assert abs(a - b) < 0.05 and abs(c - b) < 0.05

    # ... whereas raw brightness is not robust, which is the point
    v = FEATURE_NAMES.index("value_mean")
    assert abs(extract_features(dim)[v] - extract_features(bright)[v]) > 0.15


def test_feature_vector_is_finite_and_the_right_length():
    f = extract_features(_patch((170, 110, 105)))
    assert f.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(f).all()


def test_mask_rejects_lashes_and_glare():
    """Dark lashes and specular highlights both distort a colour average, in
    opposite directions."""
    img = _patch((180, 105, 105))
    img[:6, :] = 5                       # lash line
    img[40:46, 90:110] = 252             # specular highlight
    mask = conjunctiva_mask(img)
    assert not mask[:4, :].mean() > 0.5
    assert mask.sum() > img.shape[0] * img.shape[1] * 0.4


def test_mask_falls_back_rather_than_returning_nothing():
    """A degenerate crop must not produce an empty mask and a crash."""
    flat = np.full((40, 60, 3), 128, np.uint8)
    assert conjunctiva_mask(flat).sum() > 0


def test_erythema_ordering_is_monotone_across_a_pallor_series():
    """Not just two points: the index must move monotonically as the tissue
    pales, or it is not measuring what it claims to."""
    series = [(180, 80, 85), (185, 110, 112), (195, 140, 142), (205, 175, 176)]
    vals = [extract_features(_patch(c))[FEATURE_NAMES.index("erythema_mean")]
            for c in series]
    assert vals == sorted(vals, reverse=True), vals


# -------------------------------------------------------------------- data --

@needs_data
def test_index_pairs_every_image_with_a_laboratory_value():
    samples = load_index()
    assert len(samples) > 600
    assert all(s.path.exists() for s in samples)
    assert all(2.0 <= s.hb <= 20.0 for s in samples)
    # the label must agree with the WHO threshold it claims to encode
    for s in samples:
        if s.hb < WHO_ANAEMIA_HB - 0.05:
            assert s.anaemic, (s.image_id, s.hb)


@needs_data
def test_multiple_sites_are_present_for_grouped_evaluation():
    """The grouped split is only meaningful with several sites, and site
    prevalence must genuinely differ or there would be nothing to leak."""
    samples = load_index()
    by_site: dict[str, list] = {}
    for s in samples:
        by_site.setdefault(s.hospital, []).append(s)
    assert len(by_site) >= 5
    rates = [np.mean([x.anaemic for x in v]) for v in by_site.values() if len(v) >= 15]
    assert max(rates) - min(rates) > 0.15, "site prevalence barely varies"


# ----------------------------------------------------------------- results --

@pytest.mark.skipif(not RESULTS.exists(), reason="benchmark not run")
def test_reported_result_is_grouped_and_beats_the_reference_fairly():
    """The headline must come from the grouped protocol with nested model
    selection, and must be compared at the reference study's own specificity —
    comparing at whatever threshold each side happened to use compares
    operating points, not tests."""
    import json

    r = json.loads(RESULTS.read_text())
    head = r["headline"]
    assert "leave-one-hospital-out" in head["protocol"]
    assert "nested" in head["protocol"]
    assert head["sensitivity_at_reference_specificity"] >= COLLINGS_VALIDATION_SENS
    assert head["specificity"] >= COLLINGS_VALIDATION_SPEC - 0.02
    # and it must generalise to every held-out site, not just on average
    assert r["nested_selection"]["worst_site_auc"] >= 0.70
