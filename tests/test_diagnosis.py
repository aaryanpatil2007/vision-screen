"""The differential engine: does the arithmetic, and does it refuse to overreach.

The tests fall into three groups. The first pins the likelihood-ratio maths
itself. The second walks clinical vignettes through the engine and checks the
right condition comes out on top — a differential that scores everything at its
base rate is useless, and one that promotes the wrong thing is worse. The third
group is the important one: it checks the engine declines to name things the
evidence cannot separate, which is where a system like this earns or loses
trust.
"""
from __future__ import annotations

import pytest

from visionscreen.diagnosis import (
    CATALOG,
    EVIDENCE_RULES,
    MAX_EVIDENCE_LR,
    assess,
    differential,
    prevalence,
)


def _p(scored, name):
    return next(c.probability for c in scored if c.name == name)


def _names(scored):
    return [c.name for c in scored]


# ------------------------------------------------------------------- maths --

def test_no_evidence_leaves_every_condition_at_its_base_rate():
    """The prior must survive an empty session untouched — otherwise the
    arithmetic is inventing information."""
    for age in (10, 35, 70):
        for c in assess({}, age=age):
            assert c.probability == pytest.approx(prevalence(c.name, age), abs=1e-6)
            assert c.moved == pytest.approx(1.0, abs=1e-6)


def test_negative_findings_argue_against_not_merely_fail_to_support():
    """This is the whole reason for likelihood ratios over a red-flag tally:
    a tally can only accumulate, so it can never rule anything out."""
    clear = assess({"alignment_pd": 1.0, "amsler_distortion": False,
                    "colour_plate_errors": 0, "anisocoria_mm": 0.2,
                    "mrd1_min_mm": 4.5}, age=40)
    for name in ("strabismus", "amd", "colour_vision_deficiency", "ptosis"):
        assert _p(clear, name) < prevalence(name, 40), name


def test_age_changes_what_the_same_evidence_means():
    """Identical findings in a 25- and a 75-year-old must not yield identical
    conclusions; the priors differ by orders of magnitude."""
    f = {"contrast_logcs": 1.1, "acuity_logmar_left": 0.4, "acuity_logmar_right": 0.4}
    young = _p(assess(f, age=25), "cataract")
    old = _p(assess(f, age=75), "cataract")
    assert old > 8 * young


def test_evidence_is_capped_so_correlated_findings_cannot_run_away():
    """Findings in one battery are not independent. Without a cap, three views
    of the same underlying signal read as three confirmations."""
    piled = {"acuity_logmar_left": 1.2, "acuity_logmar_right": 1.2,
             "contrast_logcs": 0.6, "colour_plate_errors": 8,
             "alignment_pd": 30.0, "amsler_distortion": True,
             "field_defect": True, "anisocoria_mm": 2.0, "mrd1_min_mm": 0.5}
    for c in assess(piled, age=70):
        assert c.moved <= MAX_EVIDENCE_LR * 1.001, (c.name, c.moved)


def test_every_catalogued_condition_has_rules_and_advice():
    """A condition with no rules can never leave its base rate, and one with no
    advice is a label without a next step."""
    for key, spec in CATALOG.items():
        assert spec.what_it_is and spec.what_to_do and spec.limits, key
        assert spec.urgency in ("routine", "soon", "prompt", "urgent")
    unreachable = set(CATALOG) - set(EVIDENCE_RULES)
    assert not unreachable, f"no evidence can ever move: {unreachable}"


def test_probabilities_stay_in_range():
    extreme = {"acuity_logmar_left": 2.0, "acuity_logmar_right": 2.0,
               "contrast_logcs": 0.1, "field_defect": True,
               "amsler_distortion": True, "colour_plate_errors": 12}
    for c in assess(extreme, age=80, symptoms={"sudden_flashes"}):
        assert 0.0 < c.probability < 1.0


# --------------------------------------------------------------- vignettes --

def test_bilateral_blur_with_a_signed_reading_reads_as_myopia():
    d = differential({"acuity_logmar_left": 0.62, "acuity_logmar_right": 0.58,
                      "refraction_se": -2.5, "refraction_sign_known": True,
                      "refraction_confidence": "indicative"}, age=22)
    assert d[0].name == "myopia"
    assert d[0].probability > 0.8


def test_unilateral_loss_reads_as_amblyopia_not_short_sight():
    """A person seeing 0.60 in one eye and 0.05 in the other has something
    wrong with one eye, not a focusing error in both. Taking the worse eye for
    a bilateral diagnosis makes every one-eyed problem look like myopia."""
    d = differential({"acuity_logmar_left": 0.60, "acuity_logmar_right": 0.05,
                      "alignment_pd": 14.0}, age=30)
    assert d[0].name == "amblyopia"
    assert "myopia" not in _names(d)


def test_misalignment_raises_strabismus():
    d = differential({"alignment_pd": 18.0}, age=8)
    assert "strabismus" in _names(d)
    assert _p(d, "strabismus") > 0.3


def test_older_adult_with_contrast_loss_and_glare_raises_cataract():
    d = differential({"acuity_logmar_left": 0.4, "acuity_logmar_right": 0.42,
                      "contrast_logcs": 1.1}, age=75, symptoms={"glare", "haloes"})
    assert "cataract" in _names(d)
    assert _p(d, "cataract") > 0.5


def test_field_defect_raises_glaucoma_above_its_base_rate():
    d = differential({"field_defect": True, "contrast_logcs": 1.2}, age=65)
    assert _p(d, "glaucoma") > 4 * prevalence("glaucoma", 65)


def test_near_blur_with_clear_distance_reads_as_presbyopia():
    d = differential({"acuity_logmar_left": 0.0, "acuity_logmar_right": 0.02,
                      "near_acuity_logmar": 0.5}, age=52)
    assert d[0].name == "presbyopia"


def test_emergency_symptoms_surface_despite_a_tiny_base_rate():
    """Retinal detachment has a 0.2% prior. It must still appear, because the
    cost of dropping it is not symmetric with the cost of mentioning it."""
    d = differential({}, age=55,
                     symptoms={"sudden_flashes", "curtain_shadow"})
    assert "retinal_detachment_risk" in _names(d)
    urgent = [c for c in d if c.name == "retinal_detachment_risk"][0]
    assert urgent.urgency == "urgent"
    assert urgent.probability > 20 * prevalence("retinal_detachment_risk", 55)


def test_a_clean_session_produces_a_short_differential():
    """Normal results must not generate a page of conditions — a screening
    report that always finds something is one nobody reads twice."""
    clean = {"acuity_logmar_left": 0.0, "acuity_logmar_right": 0.0,
             "alignment_pd": 1.0, "contrast_logcs": 1.8,
             "colour_plate_errors": 0, "amsler_distortion": False,
             "field_defect": False, "anisocoria_mm": 0.2,
             "mrd1_min_mm": 4.5, "arcus_contrast": 0.02,
             "dial_detected": False, "npc_cm": 6.0}
    d = differential(clean, age=28)
    assert len(d) <= 2, [c.name for c in d]


# ------------------------------------------------------- refusing to guess --

def test_myopia_and_hyperopia_are_never_both_reported():
    """They are mutually exclusive. Reporting both reads as the system
    contradicting itself, and invites discounting everything else."""
    for age in (12, 30, 55, 75):
        d = differential({"acuity_logmar_left": 0.45,
                          "acuity_logmar_right": 0.44}, age=age)
        names = _names(d)
        assert not ("myopia" in names and "hyperopia" in names), (age, names)


def test_unsigned_blur_is_reported_without_a_direction():
    """Acuity measures how blurred vision is, not which way. Naming a direction
    from it would be asserting something nothing measured."""
    d = differential({"acuity_logmar_left": 0.45, "acuity_logmar_right": 0.44},
                     age=30)
    top = d[0]
    assert top.name == "refractive_error"
    assert "not determined" in top.plain_name
    assert "short sight" in top.what_it_is and "long sight" in top.what_it_is


def test_a_signed_measurement_is_what_unlocks_the_direction():
    """The merge must lift only when photorefraction supplied a sign — not
    merely because myopia has the higher likelihood ratio and therefore always
    pulls ahead on identical blur."""
    blur = {"acuity_logmar_left": 0.45, "acuity_logmar_right": 0.44}
    assert differential(blur, age=30)[0].name == "refractive_error"
    signed = {**blur, "refraction_se": -1.75, "refraction_sign_known": True,
              "refraction_confidence": "indicative"}
    assert differential(signed, age=30)[0].name == "myopia"


def test_the_red_reflex_can_nudge_but_never_decide():
    """A laptop has no flash. This evidence is capped low in both directions so
    it cannot carry a cataract conclusion on its own."""
    only_reflex = differential({"reflex_asymmetry": 0.6}, age=70)
    assert _p(assess({"reflex_asymmetry": 0.6}, age=70), "cataract") < 0.5
    # and a clear reflex must not be treated as reassurance
    clear = assess({"reflex_asymmetry": 0.0}, age=70)
    assert _p(clear, "cataract") > 0.5 * prevalence("cataract", 70)


def test_evidence_trail_is_inspectable():
    """A wrong answer must be traceable to the link that caused it."""
    scored = assess({"alignment_pd": 20.0, "acuity_logmar_left": 0.5,
                     "acuity_logmar_right": 0.5}, age=30)
    strab = [c for c in scored if c.name == "strabismus"][0]
    assert strab.evidence
    for e in strab.evidence:
        assert e.note and e.lr > 0
        assert e.provenance in ("LIT", "EST")
        assert e.direction in ("supports", "argues against")


def test_missing_tests_contribute_nothing_rather_than_something_wrong():
    """A partial session must degrade to weaker conclusions, not wrong ones."""
    partial = assess({"alignment_pd": 20.0}, age=30)
    full = assess({"alignment_pd": 20.0, "amsler_distortion": False}, age=30)
    assert _p(partial, "amd") == pytest.approx(prevalence("amd", 30), abs=1e-9)
    assert _p(full, "amd") < _p(partial, "amd")


# ---------------------------------------------- the bridge from real findings --

def test_findings_bridge_reads_the_modules_actual_metric_keys():
    """The bridge is keyed to metric names the scorers really emit. If a module
    renames one, this catches it — otherwise the finding silently stops
    reaching the differential and the report quietly gets less useful."""
    from visionscreen.diagnosis import findings_to_context
    from visionscreen.report import Finding

    ctx = findings_to_context([
        Finding("acuity", "", tier="measured",
                metrics={"logmar": 0.55, "eye": "left"}),
        Finding("acuity", "", tier="measured",
                metrics={"logmar": 0.50, "eye": "right"}),
        Finding("contrast", "", tier="measured", metrics={"log_cs": 1.4}),
        Finding("alignment", "", tier="measured", metrics={"deviation_pd": 12.0}),
        Finding("color_vision", "", tier="measured", metrics={"errors_total": 4}),
        Finding("pupillometry", "", tier="measured", metrics={"asymmetry_mm": 1.3}),
        Finding("eyelid position", "", tier="measured",
                metrics={"left_mrd1_mm": 4.2, "right_mrd1_mm": 1.4}),
        Finding("corneal arcus", "", tier="measured",
                metrics={"left_arcus_contrast": 0.3}),
    ])
    assert ctx["acuity_logmar_left"] == 0.55
    assert ctx["acuity_logmar_right"] == 0.50
    assert ctx["contrast_logcs"] == 1.4
    assert ctx["alignment_pd"] == 12.0
    assert ctx["colour_plate_errors"] == 4
    assert ctx["anisocoria_mm"] == 1.3
    assert ctx["mrd1_min_mm"] == 1.4          # the worse lid, not the average
    assert ctx["arcus_contrast"] == 0.3


def test_inconclusive_findings_contribute_nothing():
    """An unusable measurement is not a normal one. Letting it through as a
    negative would manufacture reassurance from a test that never worked."""
    from visionscreen.diagnosis import findings_to_context
    from visionscreen.report import Finding

    ctx = findings_to_context([
        Finding("alignment", "", tier="inconclusive", metrics={"deviation_pd": 30.0}),
        Finding("contrast", "", tier="inconclusive", metrics={"log_cs": 0.4}),
    ])
    assert ctx == {}


def test_unknown_modules_are_skipped_not_guessed_at():
    from visionscreen.diagnosis import findings_to_context
    from visionscreen.report import Finding

    ctx = findings_to_context([
        Finding("some_future_module", "", tier="measured", metrics={"x": 1}),
    ])
    assert ctx == {}


def test_age_and_symptoms_reach_the_differential_end_to_end():
    """Captured in the browser, carried in SessionMeta, used by the engine. If
    any link drops them the engine silently reverts to a 45-year-old's base
    rates and the emergency symptoms never surface at all."""
    from visionscreen.diagnosis import differential_from_findings
    from visionscreen.protocol import SessionMeta
    from visionscreen.report import Finding

    meta = SessionMeta.from_json(SessionMeta(
        session_id="s", px_per_cm=40.0, distance_cm=55.0, fps=30.0,
        age_years=71.0, symptoms=["sudden_flashes", "curtain_shadow"],
    ).to_json())
    assert meta.age_years == 71.0
    assert "curtain_shadow" in meta.symptoms

    findings = [Finding("contrast", "", tier="measured", metrics={"log_cs": 1.1})]
    d = differential_from_findings(findings, age=meta.age_years,
                                   symptoms=set(meta.symptoms))
    assert "retinal_detachment_risk" in [c.name for c in d]

    # and the age must actually be in play. Compared on the unfiltered scores,
    # because at 25 a cataract is correctly too unlikely to survive the
    # differential's own cut and would not appear at all.
    from visionscreen.diagnosis import assess, findings_to_context
    ctx = findings_to_context(findings)
    old = _p(assess(ctx, age=meta.age_years), "cataract")
    young = _p(assess(ctx, age=25), "cataract")
    assert old > 5 * young


def test_old_session_metadata_without_age_still_loads():
    """Metadata written before these fields existed must not break the run."""
    import json

    from visionscreen.protocol import SessionMeta

    legacy = json.dumps({"session_id": "old", "px_per_cm": 40.0,
                         "distance_cm": 50.0, "fps": 30.0, "segments": []})
    meta = SessionMeta.from_json(legacy)
    assert meta.age_years is None
    assert meta.symptoms == []
