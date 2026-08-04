import numpy as np
import pytest

from visionscreen.synth.photoref import crescent_width_px, render_reflex

PX_PER_M = 8000.0  # ~80 px pupil radius for an 8 mm dilated pupil... (r=0.004m→32px)


def test_width_monotone_in_defocus():
    widths = [
        crescent_width_px(A, pupil_radius_px=32, e_m=0.005, d_m=0.5, px_per_m=PX_PER_M)
        for A in (1.0, 2.0, 4.0)
    ]
    assert widths[0] < widths[1] < widths[2]
    assert all(0 <= w <= 64 for w in widths)


def test_dead_zone_no_crescent():
    # small defocus: e/(d*|A|) > 2r → no visible crescent
    w = crescent_width_px(0.5, pupil_radius_px=32, e_m=0.005, d_m=0.5, px_per_m=PX_PER_M)
    assert w == 0.0
    img, truth = render_reflex(32, S=0.5, px_per_m=PX_PER_M)
    pupil = img[img > 0]
    assert pupil.max() < 140  # uniform dim reflex, nothing bright


def test_bright_area_grows_with_defocus():
    img1, _ = render_reflex(32, S=-1.5, px_per_m=PX_PER_M)
    img2, _ = render_reflex(32, S=-4.0, px_per_m=PX_PER_M)
    assert (img2 > 140).sum() > (img1 > 140).sum() > 0


def test_myopic_and_hyperopic_sides_differ():
    myo, _ = render_reflex(32, S=-3.0, px_per_m=PX_PER_M)
    hyp, _ = render_reflex(32, S=+3.0, px_per_m=PX_PER_M)
    h, w = myo.shape
    left_myo = (myo[:, : w // 2] > 140).sum()
    right_myo = (myo[:, w // 2 :] > 140).sum()
    left_hyp = (hyp[:, : w // 2] > 140).sum()
    right_hyp = (hyp[:, w // 2 :] > 140).sum()
    # crescent flips sides between myopic and hyperopic defocus
    assert (left_myo > right_myo) != (left_hyp > right_hyp)
