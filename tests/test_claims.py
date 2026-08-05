"""The claims gate, and every string the system can show a person, run through it.

The last test here is the one that matters: it walks the real scorers in every
module, collects the summaries they actually emit, and checks that a wellness
build could render each one. Testing the gate against hand-written examples
only proves the regex works; running the system's own output through it is what
catches the day someone adds a module that says "possible cataract".
"""
from __future__ import annotations

import pytest

from visionscreen.claims import (
    ALL_CLEAR_SENTENCE,
    NOT_A_DEVICE,
    REFERRAL_SENTENCE,
    ClaimsMode,
    ClaimsViolation,
    assert_wellness_safe,
    find_violations,
    is_wellness_safe,
    render,
    summarise_outcome,
    value,
)


# ---------------------------------------------------------------- the gate --

@pytest.mark.parametrize("text", [
    "Your acuity is 20/40 in the right eye.",
    "Estimated refraction -2.25 D sphere.",
    "measured 0.30 logMAR",
    "an 18 PD outward deviation",
    "contrast sensitivity 1.65 log CS",
    "6/12 in the left eye",
])
def test_clinical_values_are_caught(text):
    """FDA's wellness guidance excludes outputs that mimic clinical values;
    even cleared web acuity tests were limited to a binary result."""
    assert not is_wellness_safe(text), text
    assert any("clinical value" in v for v in find_violations(text))


@pytest.mark.parametrize("term", [
    "This suggests myopia.", "possible cataract", "signs of glaucoma",
    "consistent with astigmatism", "you may be nearsighted",
    "features of strabismus", "suggests jaundice",
])
def test_condition_names_are_caught(term):
    assert not is_wellness_safe(term), term
    assert any("names a condition" in v for v in find_violations(term))


@pytest.mark.parametrize("term", [
    "This will diagnose your vision.",
    "we screen for eye disease",
    "get your prescription online",
    "an abnormal result",
    "medical-grade accuracy",
    "as accurate as an in-office exam",
])
def test_medical_function_claims_are_caught(term):
    assert not is_wellness_safe(term), term
    assert any("medical function" in v for v in find_violations(term))


def test_all_violations_are_reported_not_just_the_first():
    """Fixing copy one violation at a time is how the last one survives."""
    bad = "Your 20/40 acuity suggests myopia; this diagnosis is abnormal."
    v = find_violations(bad)
    assert len(v) >= 4
    assert any("clinical value" in x for x in v)
    assert any("names a condition" in x for x in v)
    assert any("medical function" in x for x in v)


def test_the_sanctioned_sentences_pass_their_own_gate():
    """If the fallback copy were unsafe the whole mechanism would be theatre."""
    for sentence in (REFERRAL_SENTENCE, ALL_CLEAR_SENTENCE, NOT_A_DEVICE):
        assert_wellness_safe(sentence)


def test_not_a_device_notice_is_exempt_by_construction():
    """It has to be able to say what it does not do. The gate matches whole
    words in claim-assertion form, so the disclaimer's own negations pass."""
    assert is_wellness_safe(NOT_A_DEVICE)


def test_substring_matches_do_not_produce_false_positives():
    """'screening' inside an ordinary sentence is not 'screening for X'; word
    boundaries matter, as an earlier bug in this project proved when
    'photograph' matched a filter for 'graph'."""
    assert is_wellness_safe("This vision check takes about eight minutes.")
    assert is_wellness_safe("Hold still while the camera adjusts.")
    assert is_wellness_safe("Sit about an arm's length from the screen.")


# ------------------------------------------------------------------ modes --

def test_research_mode_passes_everything_through():
    detailed = "Estimated -2.25 D of myopia; acuity 20/40."
    assert render(detailed, ClaimsMode.RESEARCH) == detailed
    assert value("0.30 logMAR", ClaimsMode.RESEARCH) == "0.30 logMAR"


def test_wellness_mode_substitutes_rather_than_crashes():
    """A consumer build must degrade to something true, not fall over."""
    out = render("Estimated -2.25 D of myopia.", ClaimsMode.WELLNESS)
    assert out == REFERRAL_SENTENCE
    assert_wellness_safe(out)


def test_wellness_mode_keeps_text_that_is_already_safe():
    ok = "Hold the card at arm's length and look at the centre."
    assert render(ok, ClaimsMode.WELLNESS) == ok


def test_wellness_mode_drops_clinical_values_entirely():
    """Omitted, not rounded — a vaguer clinical number is still one."""
    assert value("20/40", ClaimsMode.WELLNESS) is None


def test_an_unsafe_fallback_is_an_error_not_a_silent_pass():
    with pytest.raises(ClaimsViolation):
        render("-2.25 D", ClaimsMode.WELLNESS, fallback="You likely have myopia.")


def test_outcome_summary_differs_by_mode():
    assert "outside" in summarise_outcome(True, ClaimsMode.WELLNESS)
    assert_wellness_safe(summarise_outcome(True, ClaimsMode.WELLNESS))
    assert_wellness_safe(summarise_outcome(False, ClaimsMode.WELLNESS))


# ------------------------------------- the real thing: every module's output --

def _every_finding():
    """Drive the actual scorers and collect what they would show a person."""
    import numpy as np

    from visionscreen.modules import anterior, astigmatic, contrast, photoref
    from visionscreen.modules.refraction import estimate_refraction

    out: list[tuple[str, str]] = []

    def add(where, finding):
        out.append((where, finding.summary))
        for r in getattr(finding, "retakes", []) or []:
            out.append((f"{where} retake", r))

    # anterior segment, both the clear and the flagged branch
    size = (120, 160)
    center, iris_r, pupil_r = (80.0, 60.0), 34.0, 12.0
    yy, xx = np.mgrid[0:size[0], 0:size[1]]
    rr = np.hypot(xx - center[0], yy - center[1])
    pupil = rr <= pupil_r
    sclera = np.zeros(size, bool)
    sclera[40:80, 10:150] = True
    sclera &= rr > iris_r + 1

    def eye(reflex=(190, 70, 60), sclera_rgb=(238, 236, 232)):
        img = np.full((*size, 3), 40, np.uint8)
        img[sclera] = sclera_rgb
        img[(rr <= iris_r) & ~pupil] = (95, 80, 70)
        img[pupil] = reflex
        return img

    good = anterior.measure_red_reflex(eye(), pupil)
    dim = anterior.measure_red_reflex(eye(reflex=(60, 25, 20)), pupil)
    pale = anterior.measure_red_reflex(eye(reflex=(215, 213, 210)), pupil)
    add("red reflex clear", anterior.score_red_reflex(good, good))
    add("red reflex asymmetric", anterior.score_red_reflex(good, dim))
    add("red reflex pale", anterior.score_red_reflex(pale, None))
    add("red reflex missing", anterior.score_red_reflex(None, None))

    add("ptosis clear", anterior.score_ptosis(4.2, 4.4))
    add("ptosis flagged", anterior.score_ptosis(1.4, 4.3))
    add("ptosis gaze", anterior.score_ptosis(1.0, 4.5, gaze_ok=False))

    sc_norm = anterior.measure_sclera_colour(eye(), sclera)
    sc_yellow = anterior.measure_sclera_colour(eye(sclera_rgb=(240, 220, 120)), sclera)
    for tag, m, cal in (("clear", sc_norm, True), ("yellow", sc_yellow, True),
                        ("uncal", sc_yellow, False)):
        add(f"sclera {tag}", anterior.score_sclera(m, m, calibrated=cal))

    for tag, age in (("young", 30), ("old", 70), ("unknown", None)):
        add(f"arcus {tag}", anterior.score_arcus(0.30, 0.30, age=age))
    add("arcus clear", anterior.score_arcus(0.02, 0.02))

    # astigmatic dial
    add("dial none", astigmatic.score_astigmatic_dial([], 1.0, no_preference=True))
    add("dial axis", astigmatic.score_astigmatic_dial([90.0] * 4, 1.0, False))
    add("dial bad", astigmatic.score_astigmatic_dial([], 0.1, False))

    # photorefraction
    add("photoref none", photoref.score_photoref([], 5, 0.9))
    add("photoref ok", photoref.score_photoref(
        [(-2.0, 0.5, 90.0)] * 6, 0, 0.9))
    add("photoref bad", photoref.score_photoref([], 0, 0.1))

    # the refraction estimate's own prose
    for kwargs in (
        dict(photoref_sphere=-2.5, photoref_tier="measured",
             acuity_logmar=0.8, acuity_tier="measured", age=30),
        dict(acuity_logmar=0.8, acuity_tier="measured", age=30),
        dict(),
    ):
        est = estimate_refraction(**kwargs)
        out.append(("refraction summary", est.plain_summary))
        out.append(("refraction script", est.prescription_string))
        for c in est.caveats:
            out.append(("refraction caveat", c))

    return out


def test_every_module_summary_can_be_rendered_in_wellness_mode():
    """The gate is only worth having if it is applied to real output.

    Research mode is what this project runs in, so these strings are expected
    to be rich and clinical; the requirement is that `render` reduces each of
    them to something permissible rather than leaking a claim.
    """
    for where, text in _every_finding():
        rendered = render(text, ClaimsMode.WELLNESS, where=where)
        assert_wellness_safe(rendered, where=where)


def test_research_mode_really_does_carry_clinical_detail():
    """Guard against the gate being satisfied by the system having gone mute:
    at least some real findings must contain the detail research mode is for."""
    texts = [t for _, t in _every_finding()]
    assert any(not is_wellness_safe(t) for t in texts), \
        "no module produced clinical detail — the wellness test would be vacuous"
