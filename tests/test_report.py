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
