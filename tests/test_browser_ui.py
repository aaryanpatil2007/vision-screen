"""Real-browser tests of the capture UI.

These drive Chromium with a synthetic camera stream, so they catch the class
of bug unit tests cannot: module import errors, MediaPipe wasm failures,
canvas/DOM mistakes, and stimulus geometry that silently renders nothing.
"""
from __future__ import annotations

import socket
import subprocess
import time

import pytest

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    proc = subprocess.Popen(
        [".venv/bin/uvicorn", "webapp.app:app", "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(80):
        try:
            import urllib.request
            urllib.request.urlopen(url, timeout=1)
            break
        except Exception:
            time.sleep(0.25)
    yield url
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture(scope="module")
def browser_ctx():
    with sync_playwright() as p:
        browser = p.chromium.launch(args=[
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",   # synthetic camera
            "--autoplay-policy=no-user-gesture-required",
        ])
        ctx = browser.new_context(permissions=["camera"], viewport={"width": 1280, "height": 900})
        yield ctx
        browser.close()


# MediaPipe routes its native INFO/WARNING logs through console.error; these
# are not failures and must not mask real ones.
_BENIGN = ("xnnpack", "tensorflow lite", "feedback manager", "gl version",
           "favicon", "inference_feedback", "fiber init")


def _is_benign(msg: str) -> bool:
    m = msg.lower()
    return any(b in m for b in _BENIGN)


def _page(ctx, url):
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on(
        "console",
        lambda m: errors.append(m.text)
        if (m.type == "error" and not _is_benign(m.text))
        else None,
    )
    page.goto(url, wait_until="networkidle")
    return page, errors


def test_page_loads_without_js_errors(server, browser_ctx):
    page, errors = _page(browser_ctx, server)
    assert "VisionScreen" in page.title()
    page.wait_for_timeout(1500)
    assert not errors, errors


def test_modules_export_expected_api(server, browser_ctx):
    page, errors = _page(browser_ctx, server)
    api = page.evaluate("""async () => {
        const s = await import('/static/js/stimuli.js');
        const t = await import('/static/js/tracker.js');
        return { stim: Object.keys(s).sort(), tracker: Object.keys(t).sort() };
    }""")
    assert "letterHeightPx" in api["stim"]
    assert "drawTumblingE" in api["stim"]
    assert "drawAmsler" in api["stim"]
    assert "EyeTracker" in api["tracker"]
    assert not errors, errors


def test_optotype_math_matches_server(server, browser_ctx):
    """Client-side letter sizing must equal the server's physics."""
    from visionscreen.modules.acuity import letter_height_px

    page, _ = _page(browser_ctx, server)
    for logmar, dist, ppc in [(0.0, 50, 37.8), (1.0, 40, 45.0), (-0.3, 60, 30.0)]:
        js = page.evaluate(
            """async ([l, d, p]) => {
                const s = await import('/static/js/stimuli.js');
                return s.letterHeightPx(l, d, p);
            }""",
            [logmar, dist, ppc],
        )
        assert js == pytest.approx(letter_height_px(logmar, dist, ppc), rel=1e-9)


def test_contrast_gray_matches_server(server, browser_ctx):
    from visionscreen.modules.contrast import contrast_to_luminance_pair

    page, _ = _page(browser_ctx, server)
    for log_cs in (0.0, 0.6, 1.2, 1.95):
        js = page.evaluate(
            """async (v) => {
                const s = await import('/static/js/stimuli.js');
                return s.contrastToGray(v);
            }""",
            log_cs,
        )
        assert js == contrast_to_luminance_pair(log_cs)[0]


def test_camera_starts_and_tracker_runs(server, browser_ctx):
    page, errors = _page(browser_ctx, server)
    page.click("#btnCamera")
    page.wait_for_selector("#btnStart:not([disabled])", timeout=60_000)
    page.wait_for_timeout(2500)
    stats = page.evaluate("() => window.__app.tracker.stats")
    assert stats["fps"] > 0, stats
    # the fake device is a synthetic pattern, so no face is expected —
    # what matters is that the pipeline runs and reports honestly
    assert stats["faceOk"] in (True, False)
    assert not errors, errors


def test_calibration_updates_scale(server, browser_ctx):
    page, _ = _page(browser_ctx, server)
    page.fill("#cardSlider", "428")
    page.dispatch_event("#cardSlider", "input")
    scale = page.evaluate("() => window.__app.session.pxPerCm")
    assert scale == pytest.approx(428 / 8.56, rel=1e-6)
    assert page.inner_text("#pxPerCm").startswith("50")


def test_stimuli_render_visible_ink(server, browser_ctx):
    """Every stimulus must actually draw something — catches silent no-ops."""
    page, _ = _page(browser_ctx, server)
    inked = page.evaluate("""async () => {
        const s = await import('/static/js/stimuli.js');
        const out = {};
        const mk = () => { const c = document.createElement('canvas');
                           c.width = c.height = 400; return c; };
        const frac = (c) => {
            const d = c.getContext('2d').getImageData(0,0,400,400).data;
            let n = 0;
            for (let i = 0; i < d.length; i += 4)
                if (Math.abs(d[i]-255) > 12 || Math.abs(d[i+1]-255) > 12 || Math.abs(d[i+2]-255) > 12) n++;
            return n / (400*400);
        };
        let c = mk(); let g = c.getContext('2d');
        g.fillStyle='#fff'; g.fillRect(0,0,400,400);
        s.drawTumblingE(g, 200, 200, 200, 'right'); out.tumblingE = frac(c);

        c = mk(); s.drawAstigmaticDial(c.getContext('2d'), 200, 200, 150, 12); out.dial = frac(c);
        c = mk(); s.drawAmsler(c.getContext('2d'), 200, 200, 320, 20); out.amsler = frac(c);
        c = mk(); s.drawColorPlate(c.getContext('2d'), 200, 200, 180, 8, 'general', s.mulberry32(1));
        out.plate = frac(c);
        return out;
    }""")
    assert 0.02 < inked["tumblingE"] < 0.5, inked
    assert inked["dial"] > 0.02, inked
    assert inked["amsler"] > 0.5, inked      # grid is white-on-black: mostly non-white
    assert inked["plate"] > 0.3, inked
