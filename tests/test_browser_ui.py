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
    page.wait_for_selector("text=Camera ready", timeout=60_000)
    page.check("#brightnessOk")
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


def test_color_plate_is_luminance_isochromatic(server, browser_ctx):
    """The figure must not be readable from brightness alone — the defining
    property of a pseudoisochromatic plate."""
    page, _ = _page(browser_ctx, server)
    stats = page.evaluate("""async () => {
        const s = await import('/static/js/stimuli.js');
        const c = document.createElement('canvas'); c.width = c.height = 400;
        const g = c.getContext('2d');
        s.drawColorPlate(g, 200, 200, 180, 8, 'general', s.mulberry32(3));
        const d = g.getImageData(0, 0, 400, 400).data;
        // classify dots by hue side, compare mean luma of the two populations
        let rSum = 0, rN = 0, gSum = 0, gN = 0;
        for (let i = 0; i < d.length; i += 4) {
            const [r, gg, b] = [d[i], d[i+1], d[i+2]];
            if (Math.abs(r - 242) < 6 && Math.abs(gg - 239) < 6) continue;  // paper
            const luma = 0.299*r + 0.587*gg + 0.114*b;
            if (r > gg) { rSum += luma; rN++; } else { gSum += luma; gN++; }
        }
        return { redLuma: rSum / Math.max(rN,1), greenLuma: gSum / Math.max(gN,1),
                 redN: rN, greenN: gN };
    }""")
    assert stats["redN"] > 500 and stats["greenN"] > 500, stats
    assert abs(stats["redLuma"] - stats["greenLuma"]) < 8, stats


def test_rds_renders_and_catch_trial_has_no_disparity(server, browser_ctx):
    """The stereogram must draw dots, and a zero-disparity catch trial must
    contain no offset between the two half-images."""
    page, _ = _page(browser_ctx, server)
    out = page.evaluate("""async () => {
        const s = await import('/static/js/stimuli.js');
        const mk = () => { const c = document.createElement('canvas');
                           c.width = c.height = 200; return c; };
        const inked = (c) => {
            const d = c.getContext('2d').getImageData(0,0,200,200).data;
            let n = 0;
            for (let i = 0; i < d.length; i += 4) if (d[i] + d[i+1] + d[i+2] > 30) n++;
            return n / (200*200);
        };
        // same seed, disparity vs catch -> catch must be identical red/cyan overlay
        let c1 = mk();
        s.drawRDS(c1.getContext('2d'),
                  {size:200, disparityPx:6, quadrant:1, rng: s.mulberry32(4)});
        let c2 = mk();
        s.drawRDS(c2.getContext('2d'),
                  {size:200, disparityPx:0, quadrant:1, rng: s.mulberry32(4)});
        const d2 = c2.getContext('2d').getImageData(0,0,200,200).data;
        // in a catch trial every dot's red and cyan copies coincide exactly
        let mismatched = 0;
        for (let i = 0; i < d2.length; i += 4) {
            const red = d2[i] > 40, cyan = d2[i+1] > 20;
            if (red !== cyan) mismatched++;
        }
        return { inked1: inked(c1), inked2: inked(c2),
                 catchMismatchFrac: mismatched / (200*200) };
    }""")
    assert out["inked1"] > 0.05, out
    assert out["inked2"] > 0.05, out
    assert out["catchMismatchFrac"] < 0.01, out   # no monocular offset to read


def test_disparity_math_matches_server(server, browser_ctx):
    from visionscreen.modules.stereo import disparity_arcsec

    page, _ = _page(browser_ctx, server)
    for d_mm, D_mm in ((0.25, 600.0), (0.055, 400.0), (0.1814, 500.0)):
        js = page.evaluate(
            """async ([d, D]) => {
                const s = await import('/static/js/stimuli.js');
                return s.disparityArcsec(d, D);
            }""",
            [d_mm, D_mm],
        )
        assert js == pytest.approx(disparity_arcsec(d_mm, D_mm), rel=1e-9)


def test_start_blocked_until_brightness_attested(server, browser_ctx):
    """Acuity measured on a dim screen is systematically wrong; the run must
    not start until the user confirms brightness."""
    page, _ = _page(browser_ctx, server)
    page.click("#btnCamera")
    page.wait_for_selector("text=Camera ready", timeout=60_000)
    assert page.is_disabled("#btnStart")
    assert "brightness" in page.inner_text("#btnStart").lower()
    page.check("#brightnessOk")
    page.wait_for_selector("#btnStart:not([disabled])", timeout=10_000)
    assert page.inner_text("#btnStart").startswith("Begin")


def test_gray_ramp_rendered_for_dark_end_check(server, browser_ctx):
    page, _ = _page(browser_ctx, server)
    n = page.evaluate("() => document.querySelectorAll('#grayRamp div').length")
    assert n == 16


def test_display_limit_math_matches_server(server, browser_ctx):
    """Client and server must agree on what the screen can physically show."""
    from visionscreen.modules.acuity import renderable_floor_logmar
    from visionscreen.modules.contrast import display_ceiling_log_cs

    page, _ = _page(browser_ctx, server)
    for dist, ppc in ((40, 36.14), (50, 55.1), (30, 181.8)):
        js = page.evaluate(
            """async ([d, p]) => {
                const s = await import('/static/js/stimuli.js');
                return s.renderableFloorLogmar(d, p);
            }""",
            [dist, ppc],
        )
        assert js == pytest.approx(renderable_floor_logmar(dist, ppc), rel=1e-9)

    js_ceiling = page.evaluate(
        """async () => {
            const s = await import('/static/js/stimuli.js');
            return s.displayCeilingLogCS();
        }"""
    )
    assert js_ceiling == pytest.approx(display_ceiling_log_cs(), rel=1e-9)


def test_spoken_instructions_available_and_off_by_default(server, browser_ctx):
    """A vision test cannot assume its own instructions are legible."""
    page, _ = _page(browser_ctx, server)
    assert page.is_visible("#speakOn")
    assert page.is_checked("#speakOn") is False
    assert page.evaluate("() => window.__app.speak") is False
    page.check("#speakOn")
    assert page.evaluate("() => window.__app.speak") is True
    # say() must be safe to call regardless of platform speech support
    page.evaluate("() => window.__app.say('test utterance')")


def test_dark_room_optout_controls_the_battery(server, browser_ctx):
    page, _ = _page(browser_ctx, server)
    assert page.is_checked("#darkYes")
    assert page.evaluate("() => window.__app.canDarken") is True
    page.check("#darkNo")
    assert page.evaluate("() => window.__app.canDarken") is False


def test_glasses_guidance_is_keep_them_on(server, browser_ctx):
    """Telling people to remove correction is both clinically wrong for
    screening and leaves many unable to see the instructions."""
    page, _ = _page(browser_ctx, server)
    body = page.inner_text("body").lower()
    assert "keep your glasses or contacts on" in body
    assert "otherwise take them off" not in body


def test_hero_title_describes_what_the_tool_does(server, browser_ctx):
    """The old headline promised a test "measured in the dark". It never was —
    and after the dark-room opt-out only two of eighteen tests need darkness at
    all, so the line was both confusing and untrue."""
    page, _ = _page(browser_ctx, server)
    title = page.inner_text("h1.display").lower()
    assert "dark" not in title
    assert "webcam" in title


def test_no_prototype_badge_in_the_header(server, browser_ctx):
    page, _ = _page(browser_ctx, server)
    assert page.query_selector("header .pill") is None


def test_intake_captures_age_correction_and_symptoms(server, browser_ctx):
    """All three change what the results mean. Age drives every prevalence,
    correction decides whether a number describes the eye or the prescription,
    and the symptoms are findings a camera cannot reach."""
    page, _ = _page(browser_ctx, server)
    page.fill("#age", "18")
    page.check("input[name=correction][value=contacts]")
    page.check("#symptoms input[value=glare]")
    page.evaluate("() => window.__app.session.readIntake()")
    got = page.evaluate("""() => {
      const s = window.__app.session;
      return {age: s.ageYears, corr: s.wearingCorrection, sym: s.symptoms};
    }""")
    assert got["age"] == 18
    assert got["corr"] == "contacts"
    assert "glare" in got["sym"]


def test_implausible_age_is_rejected_rather_than_passed_through(server, browser_ctx):
    """An age of 0 or 900 would skew every base rate in the differential."""
    page, _ = _page(browser_ctx, server)
    for bad in ("0", "900"):
        page.fill("#age", bad)
        page.evaluate("() => window.__app.session.readIntake()")
        assert page.evaluate("() => window.__app.session.ageYears") is None


def test_distance_anchors_cover_the_slider_range(server, browser_ctx):
    """Viewing distance scales every acuity result, and "50 cm" means nothing to
    most people. An anchor that never changes is decoration."""
    page, _ = _page(browser_ctx, server)
    seen = []
    for cm in (30, 40, 50, 60, 75, 95):
        page.eval_on_selector("#distance",
                              f"el => {{ el.value = {cm}; "
                              f"el.dispatchEvent(new Event('input', {{bubbles: true}})); }}")
        text = page.text_content("#distanceAnchor")
        assert text and len(text) > 20, (cm, text)
        seen.append(text)
    assert len(set(seen)) >= 4, f"anchors barely vary: {set(seen)}"


def test_camera_distance_geometry_round_trips(server, browser_ctx):
    """The iris is 11.71 mm across in nearly everyone, so its pixel size ranges
    the face. Converting needs a focal length the browser will not give us, so a
    typical field of view is assumed — good to roughly a fifth, which is useless
    as a measurement and useful for catching a grossly wrong slider."""
    page, _ = _page(browser_ctx, server)
    got = page.evaluate("""() => {
      const d = window.__vsDistance;
      if (!d) return null;
      const w = 1280;
      const focal = (w / 2) / Math.tan((d.ASSUMED_HFOV_DEG * Math.PI / 180) / 2);
      const r = (focal * d.IRIS_MM) / (2 * 500);
      return {round: d.cameraDistanceCm(r, w),
              nearer: d.cameraDistanceCm(r * 2, w),
              none: d.cameraDistanceCm(0, w)};
    }""")
    assert got is not None, "distance helpers not exposed"
    assert 48 <= got["round"] <= 52, got
    assert 23 <= got["nearer"] <= 27, got
    assert got["none"] is None


def test_distance_check_is_silent_until_it_has_samples(server, browser_ctx):
    page, _ = _page(browser_ctx, server)
    assert page.get_attribute("#distanceCheck", "hidden") is not None
