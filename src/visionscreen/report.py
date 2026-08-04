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
    "eyes_tested": "Eyes tested",
    "frames": "Usable frames",
}


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


def _finding_html(f: Finding) -> str:
    flags = _flags(f)
    urgent = [x for x in flags if x in URGENT_FLAGS]
    parts = [f"<section class='finding tier-{f.tier}'>"]
    parts.append(
        "<div class='finding-head'>"
        f"<h2>{_html.escape(f.module)}</h2>"
        f"<span class='tier-badge' title='{_html.escape(TIER_HELP[f.tier])}'>{f.tier}</span>"
        "</div>"
    )
    parts.append(f"<p>{_html.escape(f.summary)}</p>")
    if urgent:
        parts.append(
            "<p class='urgent'>This finding is worth getting checked promptly.</p>"
        )
    if f.tier != "inconclusive":
        rows = _metric_rows(f)
        if rows:
            parts.append(f"<table class='metrics-table'>{rows}</table>")
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
  <div class="disclaimer"><strong>{DISCLAIMER}</strong>
    This battery cannot measure eye pressure, examine the retina, or rule out
    disease. Use it to decide whether to book an exam, never to skip one.</div>
  {_summary_banner(findings)}
  {body}
  <div class="legend">{legend}</div>
  <div class="actions">
    <button class="primary" onclick="window.print()">Save as PDF</button>
    <button class="ghost" onclick="location.href='/'">Run another screening</button>
  </div>
  <p class="hint" style="margin-top:22px">Session {_html.escape(session_id)}</p>
</main></div></body></html>"""
