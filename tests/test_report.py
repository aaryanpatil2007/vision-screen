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
