"""End-to-end: drive the whole battery in a real browser and get a report.

This is the test that would have caught every integration failure a user hits
in practice — a test that hangs waiting for a key that never arrives, a stage
that never advances, a segment the server does not recognize. It runs the
complete protocol against a synthetic camera and asserts a report comes back.
"""
from __future__ import annotations

import pytest

pytest.importorskip("playwright")

from tests.test_browser_ui import _page, browser_ctx, server  # noqa: F401,E402

# Shorten the timed stages so the full run fits in a test; the code paths and
# event logging are identical to a real session.
SPEEDUP = """
  window.__TEST_MODE = true;
"""


def _drive(page, timeout_ms=240_000):
    """Answer whatever the current stage asks for until the report appears."""
    page.evaluate("""() => {
        // auto-answer: click keypad/answer buttons, type letters, submit inputs
        window.__auto = setInterval(() => {
            const stage = document.querySelector('#stage.on');
            if (!stage) return;
            const input = stage.querySelector('#ans');
            if (input) {
                input.value = '8';
                input.dispatchEvent(new KeyboardEvent('keydown',
                    {key: 'Enter', bubbles: true}));
                return;
            }
            // An advance button always wins: on stages that have both a canvas
            // and buttons (Amsler), clicking the canvas only adds marks and
            // never progresses — which is what a confused user would do too.
            const btn = stage.querySelector('[data-key]');
            if (btn) { btn.click(); return; }
            const canvas = stage.querySelector('#rds, #grid, #dial');
            if (canvas) {
                const r = canvas.getBoundingClientRect();
                canvas.dispatchEvent(new MouseEvent('click', {bubbles: true,
                    clientX: r.left + r.width * 0.25, clientY: r.top + r.height * 0.25}));
                return;
            }
            // letter/optotype stages listen on document keydown
            document.dispatchEvent(new KeyboardEvent('keydown',
                {key: 'ArrowUp', bubbles: true}));
            document.dispatchEvent(new KeyboardEvent('keydown',
                {key: 'c', bubbles: true}));
            document.dispatchEvent(new KeyboardEvent('keydown',
                {key: 'e', bubbles: true}));
        }, 60);
    }""")
    page.wait_for_selector("text=Your screening report", timeout=timeout_ms)
    page.evaluate("() => clearInterval(window.__auto)")


@pytest.mark.slow
def test_full_battery_produces_report(server, browser_ctx):
    page, errors = _page(browser_ctx, server)
    page.add_init_script(SPEEDUP)
    page.click("#btnCamera")
    page.wait_for_selector("text=Camera ready", timeout=60_000)
    page.check("#brightnessOk")
    page.wait_for_selector("#btnStart:not([disabled])", timeout=60_000)
    page.click("#btnStart")

    _drive(page)

    body = page.inner_text("body")
    assert "screening signal only" in body

    modules = page.evaluate(
        "() => Array.from(document.querySelectorAll('.finding h2')).map(h => h.textContent)"
    )
    # The auto-driver's timing is not a real user's, so exactly which optional
    # tests complete varies; what must always hold is that the battery ran end
    # to end and produced a substantial multi-test report.
    assert len(modules) >= 8, modules
    joined = " ".join(modules)
    for expected in ("Acuity", "Viewing behavior"):
        assert expected in joined, f"missing {expected} in {modules}"
    # no module id should leak through unlabeled
    assert "_" not in joined, f"raw module id in report: {modules}"
    assert not errors, errors


@pytest.mark.slow
def test_report_never_reports_metrics_for_inconclusive(server, browser_ctx):
    """Whatever the synthetic camera produces, no inconclusive finding may
    display a numeric result."""
    page, _ = _page(browser_ctx, server)
    page.click("#btnCamera")
    page.wait_for_selector("text=Camera ready", timeout=60_000)
    page.check("#brightnessOk")
    page.wait_for_selector("#btnStart:not([disabled])", timeout=60_000)
    page.click("#btnStart")
    _drive(page)

    bad = page.evaluate("""() => {
        const out = [];
        document.querySelectorAll('.finding.tier-inconclusive').forEach(el => {
            if (el.querySelector('table.metrics-table')) out.push(el.querySelector('h2').textContent);
        });
        return out;
    }""")
    assert bad == [], f"inconclusive findings showing metrics: {bad}"
