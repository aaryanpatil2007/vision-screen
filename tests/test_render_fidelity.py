"""Does the rendered optotype actually subtend the angle the maths intends?

Every acuity number rests on a chain: logMAR -> arcmin -> cm -> pixels ->
drawn shape. The last link has never been checked. If the drawn E is even 10%
off the intended height, that is a systematic 0.04 logMAR bias on every
measurement — comparable to the whole error budget — and no amount of
repeatability testing would reveal it, because a consistently wrong renderer is
consistently wrong.

These tests rasterise the real stimulus in a real browser and measure the
result in pixels, closing a bias term that would otherwise need a clinician to
notice.
"""
from __future__ import annotations

import pytest

pytest.importorskip("playwright")

from tests.test_browser_ui import _page, browser_ctx, server  # noqa: F401,E402


def _measure_glyph(page, size_px: int) -> dict:
    """Draw a tumbling E of the requested overall size and measure the ink."""
    return page.evaluate(
        """async (size) => {
            const s = await import('/static/js/stimuli.js');
            const c = document.createElement('canvas');
            c.width = c.height = 600;
            const g = c.getContext('2d');
            g.fillStyle = '#fff'; g.fillRect(0, 0, 600, 600);
            s.drawTumblingE(g, 300, 300, size, 'right');
            const d = g.getImageData(0, 0, 600, 600).data;
            let minX = 1e9, maxX = -1, minY = 1e9, maxY = -1;
            const rowInk = new Array(600).fill(0);
            for (let y = 0; y < 600; y++) {
                for (let x = 0; x < 600; x++) {
                    if (d[(y * 600 + x) * 4] < 128) {
                        if (x < minX) minX = x;
                        if (x > maxX) maxX = x;
                        if (y < minY) minY = y;
                        if (y > maxY) maxY = y;
                        rowInk[y]++;
                    }
                }
            }
            // stroke width: the spine is the only ink on a row through a gap
            const gapRows = [];
            for (let y = minY; y <= maxY; y++) gapRows.push(rowInk[y]);
            return {
                height: maxY - minY + 1,
                width: maxX - minX + 1,
                minRowInk: Math.min(...gapRows.filter(v => v > 0)),
                maxRowInk: Math.max(...gapRows),
            };
        }""",
        size_px,
    )


@pytest.mark.parametrize("size", [50, 100, 200, 350])
def test_rendered_optotype_matches_requested_size(server, browser_ctx, size):
    """The drawn E must be the size the physics asked for, within a pixel."""
    page, _ = _page(browser_ctx, server)
    m = _measure_glyph(page, size)
    # a tumbling E is square: 5x5 units
    assert m["height"] == pytest.approx(size, abs=2), m
    assert m["width"] == pytest.approx(size, abs=2), m


@pytest.mark.parametrize("size", [50, 100, 200, 350])
def test_stroke_is_one_fifth_of_height(server, browser_ctx, size):
    """The 5x5 construction is what makes the critical detail 1/5 of the
    optotype — if the stroke drifts, the measured acuity is wrong by the
    same ratio even though the overall size looks right."""
    page, _ = _page(browser_ctx, server)
    m = _measure_glyph(page, size)
    # rows crossing a gap contain only the spine: that run IS the stroke width
    assert m["minRowInk"] == pytest.approx(size / 5, rel=0.10), m


def test_render_bias_across_the_measurement_range(server, browser_ctx):
    """Quantify the systematic size error as a logMAR bias.

    A rendered size ratio r maps to a logMAR bias of log10(r): if every letter
    draws 10% large, every acuity reading is 0.041 logMAR optimistic.
    """
    import math

    page, _ = _page(browser_ctx, server)
    ratios = []
    for size in (20, 40, 80, 160, 320):
        m = _measure_glyph(page, size)
        ratios.append(m["height"] / size)
    biases = [abs(math.log10(r)) for r in ratios]
    worst = max(biases)
    mean = sum(biases) / len(biases)
    # the rendering must contribute far less than the 0.05 logMAR bias budget
    assert worst < 0.02, f"worst render bias {worst:.4f} logMAR, ratios {ratios}"
    assert mean < 0.01, f"mean render bias {mean:.4f} logMAR"


def test_contrast_letter_luminance_is_actually_produced(server, browser_ctx):
    """The contrast ladder is only valid if the drawn grey equals the computed
    code value — a canvas colour-management surprise would bias every log CS."""
    page, _ = _page(browser_ctx, server)
    out = page.evaluate(
        """async () => {
            const s = await import('/static/js/stimuli.js');
            const c = document.createElement('canvas');
            c.width = c.height = 40;
            const g = c.getContext('2d');
            const res = [];
            for (const logCS of [0.3, 0.9, 1.5, 1.95]) {
                const v = s.contrastToGray(logCS);
                g.fillStyle = `rgb(${v},${v},${v})`;
                g.fillRect(0, 0, 40, 40);
                const px = g.getImageData(20, 20, 1, 1).data;
                res.push({ logCS, requested: v, drawn: px[0] });
            }
            return res;
        }"""
    )
    for row in out:
        assert row["drawn"] == row["requested"], row
