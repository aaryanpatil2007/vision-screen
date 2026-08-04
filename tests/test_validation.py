import math

import pytest

from visionscreen.validation import (
    ACCEPTANCE,
    Study,
    agreement,
    format_report,
    report,
)


def build(diffs, measure="acuity_logmar", base=0.2):
    s = Study(name="test")
    for i, d in enumerate(diffs):
        s.record_pair(
            subject_id=f"s{i}", measure=measure, eye="OD",
            index_value=base + d, reference_value=base,
            reference_method="ETDRS chart 4 m",
        )
    return s


def test_bias_and_loa_are_bland_altman():
    s = build([0.1, 0.1, -0.1, -0.1])
    a = agreement(s, "acuity_logmar")
    assert a["bias"] == pytest.approx(0.0, abs=1e-9)
    # reported values are rounded to 4 dp, so compare at that resolution
    sd = a["sd_of_differences"]
    assert a["loa95"][0] == pytest.approx(-1.96 * sd, abs=1e-3)
    assert a["loa95"][1] == pytest.approx(+1.96 * sd, abs=1e-3)
    assert a["loa_half_width"] == pytest.approx(1.96 * sd, abs=1e-3)


def test_acceptance_criterion_met_for_a_good_test():
    # tight agreement: bias ~0, differences within a few hundredths
    s = build([0.02, -0.02, 0.03, -0.03, 0.01, -0.01])
    a = agreement(s, "acuity_logmar")
    assert abs(a["bias"]) <= ACCEPTANCE["acuity_logmar"]["max_abs_bias"]
    assert a["meets_acceptance"] is True


def test_acceptance_fails_on_wide_limits_even_with_zero_bias():
    """The point of Bland-Altman: zero bias does not mean agreement."""
    s = build([0.4, -0.4, 0.35, -0.35, 0.5, -0.5])
    a = agreement(s, "acuity_logmar")
    assert a["bias"] == pytest.approx(0.0, abs=0.02)
    assert a["meets_acceptance"] is False
    assert a["loa_half_width"] > 0.20


def test_high_correlation_can_still_fail_agreement():
    """A test offset by a constant correlates perfectly and agrees terribly."""
    s = Study(name="offset")
    for i in range(8):
        ref = 0.1 * i
        s.record_pair(subject_id=f"s{i}", measure="acuity_logmar", eye="OD",
                      index_value=ref + 0.3, reference_value=ref,
                      reference_method="ETDRS")
    a = agreement(s, "acuity_logmar")
    assert a["pearson_r"] == pytest.approx(1.0, abs=1e-6)
    assert a["bias"] == pytest.approx(0.3, abs=1e-9)
    assert a["meets_acceptance"] is False


def test_underpowered_measure_returns_none():
    s = Study(name="tiny")
    s.record_pair(subject_id="s0", measure="acuity_logmar", eye="OD",
                  index_value=0.1, reference_value=0.1, reference_method="x")
    assert agreement(s, "acuity_logmar") is None
    rep = report(s)
    assert "acuity_logmar" in rep["underpowered"]


def test_round_trip_serialisation():
    s = build([0.05, -0.05, 0.0])
    restored = Study.from_json(s.to_json())
    assert len(restored.pairs) == 3
    assert restored.pairs[0].reference_method == "ETDRS chart 4 m"


def test_report_formats_a_table():
    s = build([0.02, -0.02, 0.01, -0.01])
    text = format_report(report(s))
    assert "acuity_logmar" in text
    assert "95% LoA" in text
    assert "correlate at r = 0.95" in text   # the caveat must survive


def test_every_measure_has_an_acceptance_criterion():
    for key, crit in ACCEPTANCE.items():
        assert crit["max_abs_bias"] > 0
        assert crit["max_loa_half_width"] > crit["max_abs_bias"]
