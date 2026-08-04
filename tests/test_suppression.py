import pytest

from visionscreen.modules.suppression import (
    WORTH_RESPONSES,
    interpret_worth_response,
    score_suppression,
)


def test_response_map_covers_the_classical_five():
    assert set(WORTH_RESPONSES) == {"four", "five_uncrossed", "five_crossed",
                                    "two_red", "three_green"}


def test_fusion_response():
    r = interpret_worth_response("four")
    assert r["fusion"] is True
    assert r["suppression"] is None
    assert r["diplopia"] is None


def test_diplopia_directions():
    assert interpret_worth_response("five_uncrossed")["diplopia"] == "esotropic"
    assert interpret_worth_response("five_crossed")["diplopia"] == "exotropic"


def test_suppression_side_follows_the_filter():
    # red filter over the right eye: seeing only green means the RED-filtered
    # (right) eye is suppressed
    assert interpret_worth_response("three_green")["suppression"] == "right"
    assert interpret_worth_response("two_red")["suppression"] == "left"


def test_fusion_at_both_distances_is_clean():
    f = score_suppression({"near": "four", "far": "four"}, valid_fraction=0.9)
    assert f.metrics["flags"] == []
    assert f.tier == "weak-signal"      # screen dissociation is never 'measured'


def test_suppression_flagged():
    f = score_suppression({"near": "four", "far": "three_green"}, valid_fraction=0.9)
    assert "suppression of one eye" in f.metrics["flags"]
    assert f.metrics["suppressing_eye"] == "right"


def test_distance_only_suppression_is_explained():
    """Central suppression scotomas show at distance, where the target subtends
    a smaller angle, and can be missed at near."""
    f = score_suppression({"near": "four", "far": "two_red"}, valid_fraction=0.9)
    assert "distance" in f.summary.lower()


def test_diplopia_flagged():
    f = score_suppression({"near": "five_crossed", "far": "five_crossed"},
                          valid_fraction=0.9)
    assert "double vision (diplopia)" in f.metrics["flags"]


def test_missing_responses_inconclusive():
    f = score_suppression({}, valid_fraction=0.9)
    assert f.tier == "inconclusive"
    assert f.retakes


def test_unknown_response_rejected():
    with pytest.raises(KeyError):
        interpret_worth_response("seven_purple")
