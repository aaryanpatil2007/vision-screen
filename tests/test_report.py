from visionscreen.report import DISCLAIMER, Finding, render_html


def test_disclaimer_always_present():
    html = render_html([], session_id="s1")
    assert DISCLAIMER in html
    assert DISCLAIMER == (
        "This is a screening signal only, not a diagnosis. "
        "See an optometrist for a clinical evaluation."
    )


def test_measured_finding_shows_metrics():
    f = Finding(
        module="acuity", summary="Estimated acuity 0.30 logMAR",
        tier="measured", metrics={"logmar": 0.30}, retakes=[],
    )
    html = render_html([f], session_id="s1")
    assert "Estimated acuity 0.30 logMAR" in html and "measured" in html


def test_inconclusive_hides_metrics_shows_retakes():
    f = Finding(
        module="behavioral", summary="Could not assess",
        tier="inconclusive", metrics={"squint_fraction": 0.9},
        retakes=["Lighting too dark — add light."],
    )
    html = render_html([f], session_id="s1")
    assert "squint_fraction" not in html          # never report numbers we don't trust
    assert "Lighting too dark — add light." in html


def test_snellen_conversion():
    from visionscreen.report import snellen_from_logmar
    assert snellen_from_logmar(0.0) == "20/20"
    assert snellen_from_logmar(0.3) == "20/40"
    assert snellen_from_logmar(1.0) == "20/200"


def test_banner_reports_no_flags_when_clean():
    fs = [Finding("acuity", "fine", "measured", {"logmar": 0.0, "flags": []})]
    html = render_html(fs, "s")
    assert "No screening flags raised" in html


def test_banner_escalates_urgent_findings():
    fs = [Finding("amsler", "marks", "measured",
                  {"flags": ["distorted lines (metamorphopsia)"]})]
    html = render_html(fs, "s")
    assert "prompt attention" in html
    assert "checked promptly" in html


def test_snellen_row_added_for_acuity():
    fs = [Finding("Acuity (both eyes)", "x", "measured", {"logmar": 0.3, "trials": 20})]
    html = render_html(fs, "s")
    assert "20/40" in html


def test_inconclusive_still_hides_metrics():
    fs = [Finding("photorefraction", "no", "inconclusive",
                  {"sphere_d": -2.5}, ["retake in the dark"])]
    html = render_html(fs, "s")
    assert "sphere_d" not in html and "-2.5" not in html
    assert "retake in the dark" in html


def test_acuity_finding_includes_scale_visual():
    fs = [Finding("Acuity (both eyes)", "x", "measured", {"logmar": 0.3, "trials": 20})]
    html = render_html(fs, "s")
    assert "<svg" in html and "20/20" in html


def test_contrast_finding_includes_bar():
    fs = [Finding("contrast", "x", "measured", {"log_cs": 1.65, "flags": []})]
    html = render_html(fs, "s")
    assert "<svg" in html and "normal" in html


def test_inconclusive_has_no_visual():
    """No chart on a finding with no result — scoped to the finding itself,
    since the page chrome legitimately carries an SVG logo mark."""
    fs = [Finding("Acuity (left eye)", "x", "inconclusive", {"logmar": 0.3},
                  ["retake"])]
    html = render_html(fs, "s")
    finding = html[html.index("<section class='finding"):]
    assert "<svg" not in finding
    assert "scale" not in finding


def test_module_names_are_human_readable():
    from visionscreen.report import module_label
    assert module_label("color_vision") == "Color vision"
    assert module_label("photorefraction") == "Refraction estimate"
    assert module_label("Acuity (both eyes)") == "Acuity (both eyes)"
    html = render_html([Finding("color_vision", "x", "weak-signal", {"flags": []})], "s")
    assert "color_vision" not in html


def test_metric_keys_never_shown_raw():
    fs = [Finding("viewing distance", "x", "measured",
                  {"median_cm": 48.2, "acuity_bias_logmar": 0.017, "flags": []})]
    html = render_html(fs, "s")
    assert "median_cm" not in html and "acuity_bias_logmar" not in html
    assert "Measured distance (cm)" in html


def test_report_states_miss_rate_numerically():
    """Boilerplate is not honesty — the report must quantify its blind spot."""
    html = render_html([Finding("Acuity (both eyes)", "x", "measured",
                                {"logmar": 0.0})], "s")
    assert "one in three" in html
    assert "glaucoma" in html and "retina" in html


def test_refraction_is_labeled_not_a_prescription():
    fs = [Finding("photorefraction", "estimate", "weak-signal",
                  {"sphere_d": -2.25, "cylinder_d": 0.5})]
    html = render_html(fs, "s")
    assert "not a prescription" in html
    assert "order glasses" in html


def test_report_carries_research_prototype_footer():
    html = render_html([], "s")
    assert "not FDA-cleared" in html
    assert "makes no diagnosis" in html


def test_no_prescription_note_when_no_refraction():
    html = render_html([Finding("contrast", "x", "measured", {"log_cs": 1.8})], "s")
    assert "not a prescription" not in html


# ------------------------------------------- the differential in the report --

def _demo_findings():
    return [
        Finding("acuity", "Distance acuity 0.55 logMAR.", tier="measured",
                metrics={"logmar": 0.55, "eye": "left"}),
        Finding("acuity", "Distance acuity 0.52 logMAR.", tier="measured",
                metrics={"logmar": 0.52, "eye": "right"}),
        Finding("astigmatism", "One meridian sharper.", tier="measured",
                metrics={"flags": ["possible astigmatism"], "axis_deg": 90}),
    ]


def test_banner_does_not_contradict_the_differential_below_it():
    """A person can pass every individual threshold while the combination still
    points somewhere. "No flags raised" directly above "likely: short sight"
    reads as the report arguing with itself."""
    from visionscreen.diagnosis import differential_from_findings
    from visionscreen.report import _summary_banner

    f = _demo_findings()
    conds = differential_from_findings(f, age=24)
    assert conds
    banner = _summary_banner(f, conds)
    assert "No screening flags raised" not in banner
    assert "optometrist" in banner or "prompt attention" in banner


def test_clean_session_still_reports_all_clear():
    from visionscreen.report import _summary_banner

    clean = [Finding("acuity", "Normal.", tier="measured",
                     metrics={"logmar": 0.0, "eye": "left"})]
    assert "No screening flags raised" in _summary_banner(clean, [])


def test_banner_does_not_say_the_same_thing_twice():
    from visionscreen.diagnosis import differential_from_findings
    from visionscreen.report import _summary_banner

    f = _demo_findings()
    banner = _summary_banner(f, differential_from_findings(f, age=24))
    assert banner.lower().count("astigmatism") == 1, banner


def test_headline_leads_with_the_answer():
    """A reader who must assemble the conclusion from eighteen test cards will
    not do it."""
    from visionscreen.diagnosis import differential_from_findings

    f = _demo_findings()
    html = render_html(f, "s1", conditions=differential_from_findings(f, age=24),
                       correction="none")
    assert "The short version" in html
    assert html.index("The short version") < html.index("What might explain")
    assert "20/" in html                       # acuity stated in plain Snellen


def test_headline_states_what_correction_was_worn():
    """Identical numbers mean opposite things with and without lenses. Omitting
    that is not terse, it is misleading."""
    from visionscreen.diagnosis import differential_from_findings

    f = _demo_findings()
    conds = differential_from_findings(f, age=24)
    with_lenses = render_html(f, "s2", conditions=conds, correction="contacts")
    without = render_html(f, "s3", conditions=conds, correction="none")
    assert "wearing contact lenses" in with_lenses
    assert "not what your eyes do unaided" in with_lenses
    assert "no correction" in without
    # and an unrecorded answer must say so rather than quietly assume
    unknown = render_html(f, "s4", conditions=conds)
    assert "did not record whether" in unknown


def test_headline_reports_run_quality_up_front():
    """Whether the run worked matters more than any single number in it."""
    good = [Finding("acuity", "", tier="measured", metrics={"logmar": 0.0, "eye": "left"}),
            Finding("contrast", "", tier="measured", metrics={"log_cs": 1.8})]
    assert "went well" in render_html(good, "s5", conditions=[])

    poor = [Finding("acuity", "", tier="inconclusive", metrics={}),
            Finding("contrast", "", tier="inconclusive", metrics={}),
            Finding("stereo", "", tier="measured", metrics={})]
    assert "Poor run" in render_html(poor, "s6", conditions=[])


def test_rejected_acuity_is_blamed_on_the_run_not_the_eyes():
    """Chance-level answers say nothing about vision, and the report must not
    let a reader think otherwise."""
    f = [Finding("acuity", "", tier="inconclusive",
                 metrics={"trials": 56, "correct": 6,
                          "rejected_reason": "responses no better than chance"})]
    html = render_html(f, "s7", conditions=[], correction="contacts")
    assert "No acuity result" in html
    assert "not a finding about your eyes" in html


def test_differential_renders_with_evidence_and_provenance():
    from visionscreen.diagnosis import differential_from_findings

    f = _demo_findings()
    html = render_html(f, "s8", conditions=differential_from_findings(f, age=24))
    assert "Why this came up" in html
    assert "supports" in html
    assert ">LIT<" in html or ">EST<" in html
    assert "possibilities ranked" in html


def test_urgent_conditions_sort_above_more_likely_routine_ones():
    """A 7% chance of retinal detachment matters more than a 90% chance of
    needing reading glasses."""
    from visionscreen.diagnosis import differential

    conds = differential({"near_acuity_logmar": 0.6, "acuity_logmar_left": 0.0,
                          "acuity_logmar_right": 0.0}, age=55,
                         symptoms={"sudden_flashes", "curtain_shadow"})
    html = render_html([], "s9", conditions=conds)
    u, r = html.find("retinal"), html.find("presbyopia")
    assert u != -1
    if r != -1:
        assert u < r


def test_report_renders_without_a_differential():
    html = render_html(_demo_findings(), "s10")
    assert "Your screening report" in html
    assert "What might explain these results" not in html


def test_refraction_card_shows_its_interval_not_just_a_number():
    from visionscreen.modules.refraction import estimate_refraction

    rx = estimate_refraction(photoref_sphere=-2.0, photoref_cyl=0.75,
                             photoref_tier="measured", acuity_logmar=0.52,
                             acuity_tier="measured", pupil_mm=4.2, age=24)
    html = render_html(_demo_findings(), "s11", conditions=[], refraction=rx)
    assert "not a prescription" in html.lower()
    assert "most likely between" in html
