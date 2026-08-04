from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Tier = Literal["measured", "weak-signal", "inconclusive"]

DISCLAIMER = (
    "This is a screening signal only, not a diagnosis. "
    "See an optometrist for a clinical evaluation."
)

# Stating the blind spot as a number beats boilerplate. In a self-declared-
# healthy cohort with a median age of 70, a full examination found that 25%
# needed referral and a further 9% needed monitoring — 34% with findings that
# no screen-and-webcam test can see (PMC7798115).
MISS_RATE_NOTE = (
    "What this cannot see: eye pressure, the retina, and the internal structures "
    "an examination uses to find glaucoma, diabetic retinopathy, macular "
    "degeneration and cataract. In one study of adults who considered themselves "
    "healthy (median age 70), about one in three had a finding at a full eye exam "
    "that tests like this one cannot detect. A clear result here is not a "
    "substitute for that exam."
)

NOT_A_PRESCRIPTION = (
    "The refraction figure is a research estimate of focus, not a prescription. "
    "It cannot be used to order glasses or contact lenses."
)

TIER_HELP = {
    "measured": "Enough good data to report a number.",
    "weak-signal": "Partial or noisy data — treat as a hint, not a result.",
    "inconclusive": "Not enough usable data. No result is reported.",
}

# Findings whose flags mean "get this looked at sooner rather than later".
URGENT_FLAGS = {
    "distorted lines (metamorphopsia)",
    "missing area (possible scotoma)",
    "asymmetric pupil response",
    "possible eye misalignment",
}

_METRIC_LABELS = {
    "logmar": "Acuity (logMAR)",
    "snellen": "Snellen equivalent",
    "trials": "Trials",
    "log_cs": "Contrast sensitivity (log CS)",
    "weber_contrast_pct": "Faintest contrast seen (%)",
    "axis_deg": "Axis (degrees)",
    "dark_meridian_deg": "Sharpest meridian (degrees)",
    "response_spread_deg": "Response spread (degrees)",
    "deviation_pd": "Deviation (prism diopters)",
    "asymmetry_mm": "Reflex asymmetry (mm)",
    "conjugacy": "Eye-to-eye tracking correlation",
    "gain_left": "Pursuit gain, left",
    "gain_right": "Pursuit gain, right",
    "saccades_per_s": "Catch-up saccades / s",
    "sphere_d": "Sphere (diopters)",
    "cylinder_d": "Cylinder (diopters)",
    "sphere_std": "Frame-to-frame spread (D)",
    "squint_fraction": "Frames squinting",
    "lean_ratio": "Lean-in ratio",
    "errors_total": "Color plates missed",
    "plates": "Plates shown",
    "distortion_marks": "Areas marked distorted",
    "missing_marks": "Areas marked missing",
    "eyes_tested": "Eyes tested",
    "frames": "Usable frames",
    "responses": "Repeats",
    "axis": "Confusion axis",
    "errors_protan": "Protan-line errors",
    "errors_deutan": "Deutan-line errors",
    "dead_zone_d": "Dead zone (diopters)",
    "median_cm": "Measured distance (cm)",
    "spread_pct": "Distance variation (%)",
    "acuity_bias_logmar": "Acuity bias from distance (logMAR)",
    "measured_distance_cm": "Measured distance (cm)",
    "logmar_uncorrected": "Acuity before distance correction",
    "distance_correction_logmar": "Distance correction (logMAR)",
    "lag_left_s": "Tracking lag, left (s)",
    "min_mm": "Smallest pupil (mm)",
    "baseline_mm": "Resting pupil (mm)",
    "constriction_pct": "Constriction (%)",
    "latency_s": "Response latency (s)",
    "samples": "Samples",
}

# Display names for modules whose internal ids are snake_case.
_MODULE_LABELS = {
    "color_vision": "Color vision",
    "photorefraction": "Refraction estimate",
    "pupillometry": "Pupil response",
    "motility": "Eye movement",
    "amsler": "Central field (Amsler)",
    "contrast": "Contrast sensitivity",
    "behavioral": "Viewing behavior",
    "astigmatism": "Astigmatism",
    "alignment": "Eye alignment",
    "stereo": "Depth perception",
    "suppression": "Binocular fusion",
    "viewing distance": "Viewing distance",
}


def module_label(module: str) -> str:
    if module in _MODULE_LABELS:
        return _MODULE_LABELS[module]
    return module.replace("_", " ")


@dataclass
class Finding:
    module: str
    summary: str
    tier: Tier
    metrics: dict = field(default_factory=dict)
    retakes: list[str] = field(default_factory=list)


def snellen_from_logmar(logmar: float) -> str:
    """logMAR -> 20/x Snellen equivalent (US notation)."""
    return f"20/{round(20 * (10 ** logmar))}"


def _flags(f: Finding) -> list[str]:
    v = f.metrics.get("flags")
    return list(v) if isinstance(v, (list, tuple)) else []


def _metric_rows(f: Finding) -> str:
    rows = []
    for k, v in f.metrics.items():
        if k == "flags" or v is None or isinstance(v, dict):
            continue
        label = _METRIC_LABELS.get(k, k.replace("_", " "))
        rows.append(
            f"<tr><td>{_html.escape(label)}</td>"
            f"<td>{_html.escape(str(v))}</td></tr>"
        )
    if "logmar" in f.metrics and isinstance(f.metrics["logmar"], (int, float)):
        rows.insert(1, f"<tr><td>Snellen equivalent</td>"
                       f"<td>{snellen_from_logmar(f.metrics['logmar'])}</td></tr>")
    return "".join(rows)


def _acuity_scale_svg(logmar: float) -> str:
    """A logMAR scale with the result marked, plus the normal-vision reference."""
    lo, hi = -0.3, 1.3
    pos = max(0.0, min(1.0, (logmar - lo) / (hi - lo)))
    x = 8 + pos * 384
    normal_x = 8 + ((0.0 - lo) / (hi - lo)) * 384
    ticks = ""
    for v, lbl in ((-0.3, "20/10"), (0.0, "20/20"), (0.3, "20/40"),
                   (0.7, "20/100"), (1.0, "20/200"), (1.3, "20/400")):
        tx = 8 + ((v - lo) / (hi - lo)) * 384
        ticks += (f"<line x1='{tx:.1f}' y1='30' x2='{tx:.1f}' y2='36' "
                  f"stroke='currentColor' opacity='.35'/>"
                  f"<text x='{tx:.1f}' y='50' font-size='9' text-anchor='middle' "
                  f"fill='currentColor' opacity='.55'>{lbl}</text>")
    return f"""<svg viewBox="0 0 400 60" width="100%" height="60" class="scale"
      role="img" aria-label="acuity {logmar:.2f} logMAR on a 20/10 to 20/400 scale">
      <defs><linearGradient id="ag" x1="0" x2="1">
        <stop offset="0" stop-color="#38d39f"/><stop offset="0.45" stop-color="#ffb74d"/>
        <stop offset="1" stop-color="#ff6b6b"/></linearGradient></defs>
      <rect x="8" y="18" width="384" height="8" rx="4" fill="url(#ag)" opacity=".55"/>
      {ticks}
      <line x1="{normal_x:.1f}" y1="12" x2="{normal_x:.1f}" y2="32"
            stroke="currentColor" stroke-dasharray="2 2" opacity=".5"/>
      <circle cx="{x:.1f}" cy="22" r="7" fill="#fff" stroke="#0b0f14" stroke-width="2"/>
    </svg>"""


def _bar_svg(value: float, lo: float, hi: float, good_above: float | None,
             label: str) -> str:
    pos = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    x = 8 + pos * 384
    ref = ""
    if good_above is not None:
        gx = 8 + max(0.0, min(1.0, (good_above - lo) / (hi - lo))) * 384
        ref = (f"<line x1='{gx:.1f}' y1='12' x2='{gx:.1f}' y2='32' "
               f"stroke='currentColor' stroke-dasharray='2 2' opacity='.5'/>"
               f"<text x='{gx:.1f}' y='50' font-size='9' text-anchor='middle' "
               f"fill='currentColor' opacity='.55'>normal</text>")
    return f"""<svg viewBox="0 0 400 60" width="100%" height="60" class="scale"
      role="img" aria-label="{_html.escape(label)}">
      <rect x="8" y="18" width="384" height="8" rx="4" fill="currentColor" opacity=".18"/>
      <rect x="8" y="18" width="{max(2, x-8):.1f}" height="8" rx="4"
            fill="currentColor" opacity=".55"/>
      {ref}
      <circle cx="{x:.1f}" cy="22" r="7" fill="#fff" stroke="#0b0f14" stroke-width="2"/>
    </svg>"""


def _visual(f: Finding) -> str:
    """A small chart when one communicates better than a number."""
    if f.tier == "inconclusive":
        return ""
    m = f.metrics
    if f.module.lower().startswith("acuity") and isinstance(m.get("logmar"), (int, float)):
        return _acuity_scale_svg(float(m["logmar"]))
    if f.module == "contrast" and isinstance(m.get("log_cs"), (int, float)):
        return _bar_svg(float(m["log_cs"]), 0.0, 2.25, 1.75, "contrast sensitivity")
    if f.module == "alignment" and isinstance(m.get("deviation_pd"), (int, float)):
        return _bar_svg(float(m["deviation_pd"]), 0.0, 40.0, 10.0, "ocular deviation")
    return ""


def _finding_html(f: Finding) -> str:
    flags = _flags(f)
    urgent = [x for x in flags if x in URGENT_FLAGS]
    parts = [f"<section class='finding tier-{f.tier}'>"]
    parts.append(
        "<div class='finding-head'>"
        f"<h2>{_html.escape(module_label(f.module))}</h2>"
        f"<span class='tier-badge' title='{_html.escape(TIER_HELP[f.tier])}'>{f.tier}</span>"
        "</div>"
    )
    parts.append(f"<p>{_html.escape(f.summary)}</p>")
    if urgent:
        parts.append(
            "<p class='urgent'>This finding is worth getting checked promptly.</p>"
        )
    if f.tier != "inconclusive":
        parts.append(_visual(f))
        rows = _metric_rows(f)
        if rows:
            parts.append(f"<table class='metrics-table'>{rows}</table>")
        # Reporting sphere/cylinder/axis is the exact output that makes a
        # product a prescribing device; say plainly that it is not one.
        if "sphere_d" in f.metrics:
            parts.append(f"<p class='caveat'>{NOT_A_PRESCRIPTION}</p>")
    if f.retakes:
        items = "".join(f"<li>{_html.escape(r)}</li>" for r in f.retakes)
        parts.append(f"<p class='retake-title'>To get a result, retake:</p><ul>{items}</ul>")
    parts.append("</section>")
    return "".join(parts)


def _summary_banner(findings: list[Finding]) -> str:
    all_flags = [x for f in findings for x in _flags(f)]
    urgent = sorted({x for x in all_flags if x in URGENT_FLAGS})
    other = sorted({x for x in all_flags if x not in URGENT_FLAGS})
    n_measured = sum(1 for f in findings if f.tier == "measured")
    n_incon = sum(1 for f in findings if f.tier == "inconclusive")

    if urgent:
        cls, head = "banner urgent", "Findings worth prompt attention"
        body = ", ".join(urgent) + "."
    elif other:
        cls, head = "banner warn", "Some findings to discuss with an optometrist"
        body = ", ".join(other) + "."
    else:
        cls, head = "banner good", "No screening flags raised"
        body = ("Nothing in this battery looked abnormal. Screening tests miss things — "
                "keep your regular eye exams.")
    return (
        f"<div class='{cls}'><h3>{head}</h3><p>{_html.escape(body)}</p>"
        f"<p class='counts'>{n_measured} of {len(findings)} tests produced a measured "
        f"result; {n_incon} were inconclusive.</p></div>"
    )


def _css() -> str:
    path = Path(__file__).resolve().parents[2] / "webapp" / "static" / "css" / "app.css"
    base = path.read_text() if path.exists() else ""
    return base + """
    main.report { max-width: 860px; margin: 0 auto; padding: 28px 22px 90px; }
    .banner { border-radius: 14px; padding: 20px 22px; margin: 0 0 22px; border: 1px solid; }
    .banner h3 { margin: 0 0 6px; font-size: 18px; }
    .banner p { margin: 0; font-size: 14px; line-height: 1.55; }
    .banner .counts { margin-top: 10px; opacity: .75; font-size: 12.5px; }
    .banner.good { background: rgba(56,211,159,.08); border-color: rgba(56,211,159,.35); color: var(--good); }
    .banner.warn { background: rgba(255,183,77,.08); border-color: rgba(255,183,77,.35); color: var(--warn); }
    .banner.urgent { background: rgba(255,107,107,.10); border-color: rgba(255,107,107,.4); color: var(--bad); }
    .finding-head { display:flex; align-items:center; gap:12px; margin-bottom:8px; }
    .finding-head h2 { margin:0; font-size:16.5px; text-transform: capitalize; }
    .finding p { margin: 0 0 4px; font-size: 14px; line-height: 1.6; color: var(--text); }
    .finding .urgent { color: var(--bad); font-weight: 600; margin-top: 8px; }
    .retake-title { color: var(--muted); font-size: 13px; margin-top: 10px; }
    .finding ul { margin: 6px 0 0; padding-left: 18px; color: var(--muted); font-size: 13px; line-height:1.6; }
    .legend { display:flex; gap:18px; flex-wrap:wrap; margin: 26px 0 0; color: var(--muted); font-size:12.5px; }
    .legend b { color: var(--text); }
    .actions { margin-top: 28px; display:flex; gap:12px; }
    .scale { color: var(--muted); margin: 10px 0 2px; display:block; }
    .caveat { color: var(--muted); font-size: 12.5px; margin-top: 10px;
              border-left: 2px solid var(--line); padding-left: 10px; }
    .footnote { color: var(--muted); font-size: 12px; line-height: 1.6;
                margin-top: 26px; border-top: 1px solid var(--line); padding-top: 16px; }
    @media print {
      body { background:#fff; color:#111; }
      .card, .finding { break-inside: avoid; }
      .actions { display:none; }
    }
    """


def render_html(findings: list[Finding], session_id: str) -> str:
    body = "".join(_finding_html(f) for f in findings) or "<p>No results.</p>"
    legend = "".join(
        f"<span><b>{k}</b> — {v}</span>" for k, v in TIER_HELP.items()
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vision Screening Report</title><style>{_css()}</style></head>
<body><div class="chrome">
<header class="topbar">
  <div class="logo"><span class="iris"></span> VisionScreen</div>
  <div class="spacer"></div><span class="pill">report</span>
</header>
<main class="report">
  <h1>Your screening report</h1>
  <div class="disclaimer"><strong>{DISCLAIMER}</strong> {MISS_RATE_NOTE}</div>
  {_summary_banner(findings)}
  {body}
  <div class="legend">{legend}</div>
  <div class="actions">
    <button class="primary" onclick="window.print()">Save as PDF</button>
    <button class="ghost" onclick="location.href='/'">Run another screening</button>
  </div>
  <p class="footnote">
    Research prototype — not FDA-cleared and not a medical device. It issues no
    prescription and makes no diagnosis. Results describe visual function
    measured on this screen at this moment; they are not a substitute for a
    comprehensive eye examination by a licensed eye care professional.
    <br><br>Session {_html.escape(session_id)}
  </p>
</main></div></body></html>"""
