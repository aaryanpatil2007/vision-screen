import pytest

from visionscreen.modules.amsler import AMSLER_SQUARES, amsler_geometry, score_amsler
from visionscreen.modules.colorvision import (
    ISHIHARA_STYLE_PLATES,
    classify_color_deficiency,
    score_color_vision,
)


# ---------- Amsler ----------

def test_amsler_geometry_subtends_20_degrees():
    # Standard Amsler: 10 cm grid at 33 cm = 20 deg total (10 deg each side)
    g = amsler_geometry(distance_cm=33.0, px_per_cm=37.8)
    assert g["total_deg"] == pytest.approx(20.0, abs=0.6)
    assert g["square_deg"] == pytest.approx(1.0, abs=0.1)
    assert g["squares"] == AMSLER_SQUARES
    assert g["grid_px"] > 0


def test_amsler_clean_result():
    f = score_amsler(distortions=[], missing=[], eyes_tested=2, valid_fraction=0.9)
    assert f.tier == "measured"
    assert f.metrics["flags"] == []


def test_amsler_metamorphopsia_flagged():
    f = score_amsler(distortions=[{"x": 0.4, "y": 0.55}], missing=[],
                     eyes_tested=2, valid_fraction=0.9)
    assert "distorted lines (metamorphopsia)" in f.metrics["flags"]
    assert "optometrist" in f.summary.lower() or "urgent" in f.summary.lower()


def test_amsler_scotoma_flagged():
    f = score_amsler(distortions=[], missing=[{"x": 0.5, "y": 0.5}],
                     eyes_tested=2, valid_fraction=0.9)
    assert "missing area (possible scotoma)" in f.metrics["flags"]


def test_amsler_one_eye_only_downgrades():
    f = score_amsler(distortions=[], missing=[], eyes_tested=1, valid_fraction=0.9)
    assert f.tier == "weak-signal"


# ---------- Color vision ----------

def test_plates_have_expected_structure():
    assert len(ISHIHARA_STYLE_PLATES) >= 8
    for p in ISHIHARA_STYLE_PLATES:
        assert {"id", "digit", "type"} <= set(p)
        assert p["type"] in ("demo", "protan", "deutan", "general")


def test_normal_vision_scores_clean():
    answers = {p["id"]: p["digit"] for p in ISHIHARA_STYLE_PLATES}
    f = score_color_vision(answers, valid_fraction=0.9)
    assert f.tier == "weak-signal"  # screen color is never "measured"
    assert f.metrics["flags"] == []


def test_red_green_deficiency_detected():
    answers = {
        p["id"]: (None if p["type"] in ("protan", "deutan", "general") else p["digit"])
        for p in ISHIHARA_STYLE_PLATES
    }
    f = score_color_vision(answers, valid_fraction=0.9)
    assert "possible red-green color deficiency" in f.metrics["flags"]


def test_classify_protan_vs_deutan():
    assert classify_color_deficiency(protan_errors=3, deutan_errors=0) == "protan-leaning"
    assert classify_color_deficiency(protan_errors=0, deutan_errors=3) == "deutan-leaning"
    assert classify_color_deficiency(protan_errors=2, deutan_errors=2) == "unclassified"


def test_color_summary_states_calibration_limit():
    answers = {p["id"]: p["digit"] for p in ISHIHARA_STYLE_PLATES}
    f = score_color_vision(answers, valid_fraction=0.9)
    assert "calibrat" in f.summary.lower()
