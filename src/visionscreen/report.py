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
    "acuity below measurable range",
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
    "optotype": "Optotype",
    "threshold_arcsec": "Stereo threshold (arcsec)",
    "display_floor_arcsec": "Finest this screen can show (arcsec)",
    "catch_trials": "Catch trials",
    "near_response": "Near response",
    "far_response": "Distance response",
    "suppressing_eye": "Suppressed eye",
    "anisocoria_mm": "Pupil size difference (mm)",
    "expected_error_pd": "Expected error (PD)",
    "asymmetry_dispersion_mm": "Frame-to-frame spread (mm)",
    "display_ceiling_log_cs": "Faintest this screen can show (log CS)",
    "display_floor_logmar": "Finest letter this screen can draw (logMAR)",
    "logmar_raw_tumbling_e": "Raw threshold (logMAR)",
    "worse_than_logmar": "Worse than (logMAR)",
    "worse_than_snellen": "Worse than (Snellen)",
    "not_attempted": "Not attempted",
    "optotype_correction_logmar": "Chart-scale correction (logMAR)",
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


def _summary_banner(findings: list[Finding], conditions=None) -> str:
    all_flags = [x for f in findings for x in _flags(f)]
    urgent = sorted({x for x in all_flags if x in URGENT_FLAGS})
    other = sorted({x for x in all_flags if x not in URGENT_FLAGS})
    n_measured = sum(1 for f in findings if f.tier == "measured")
    n_incon = sum(1 for f in findings if f.tier == "inconclusive")

    # The banner sits directly above the differential, so it must agree with it.
    # A per-test flag sweep alone does not: a person can pass every individual
    # threshold while the combination still points somewhere. Printing "no flags
    # raised" on top of "likely: short sight" reads as the report arguing with
    # itself, and a reader who spots that discounts the whole page.
    conditions = list(conditions or [])
    pressing = [c for c in conditions if c.urgency in ("urgent", "prompt")]
    notable = [c for c in conditions if c.band in ("likely", "possible")]

    def _merge(flags, conds):
        """Condition names win; drop any raw flag they already cover."""
        names = [c.plain_name for c in conds]
        stems = {n.split("(")[0].strip().lower() for n in names}
        kept = [f for f in flags
                if not any(st and (st in f.lower() or f.lower() in st) for st in stems)]
        return ", ".join(dict.fromkeys(names + kept)) + "."

    if urgent or pressing:
        cls, head = "banner urgent", "Findings worth prompt attention"
        body = _merge(urgent, pressing)
    elif other or notable:
        cls, head = "banner warn", "Some findings to discuss with an optometrist"
        body = _merge(other, notable)
    else:
        cls, head = "banner good", "No screening flags raised"
        body = ("Nothing in this battery looked abnormal. Screening tests miss things — "
                "keep your regular eye exams.")
    return (
        f"<div class='{cls}'><h3>{head}</h3><p>{_html.escape(body)}</p>"
        f"<p class='counts'>{n_measured} of {len(findings)} tests produced a measured "
        f"result; {n_incon} {'was' if n_incon == 1 else 'were'} inconclusive.</p></div>"
    )


URGENCY_LABEL = {
    "urgent": "See someone today",
    "prompt": "Arrange an appointment soon",
    "soon": "Worth getting checked",
    "routine": "Mention at your next eye test",
}

CORRECTION_FRAME = {
    "contacts": ("while wearing contact lenses",
                 "Because you wore contacts, this measured how well your current "
                 "prescription is working — not what your eyes do unaided. Anything "
                 "found here is what your lenses are not already fixing."),
    "glasses": ("while wearing glasses",
                "Because you wore glasses, this measured how well your current "
                "prescription is working — not what your eyes do unaided. Anything "
                "found here is what your lenses are not already fixing."),
    "none": ("without glasses or contacts",
             "You wore no correction, so these results describe your eyes as they are."),
}


def _headline_html(findings, conditions, refraction=None, correction=None) -> str:
    """The answer, before the evidence.

    A reader who must assemble the conclusion themselves from eighteen test
    cards will not do it. This says what the run found, how much of it worked,
    and what to do — in that order — and everything below becomes supporting
    detail rather than something to decode.

    It is also where the correction question earns its place: identical numbers
    mean opposite things depending on whether lenses were worn, and a report
    that omits that is not merely terse, it is misleading.
    """
    measured = [f for f in findings if f.tier == "measured"]
    incon = [f for f in findings if f.tier == "inconclusive"]
    worn, worn_note = CORRECTION_FRAME.get(
        correction or "", ("", "This run did not record whether you were wearing "
                               "glasses or contacts, which changes what the numbers mean."))

    acuity_line = ""
    acu = [f for f in findings if f.module == "acuity"]
    usable = [f for f in acu if f.tier == "measured" and f.metrics.get("logmar") is not None]
    rejected = [f for f in acu if f.tier == "inconclusive"]
    if usable:
        worst = max(usable, key=lambda f: f.metrics["logmar"])
        lm = worst.metrics["logmar"]
        sn = snellen_from_logmar(lm)
        if lm <= 0.1:
            acuity_line = f"Your distance vision measured about <b>{sn}</b> {worn} — a normal result."
        elif lm <= 0.3:
            acuity_line = f"Your distance vision measured about <b>{sn}</b> {worn} — mildly below the 20/20 mark."
        else:
            acuity_line = (f"Your distance vision measured about <b>{sn}</b> {worn}, "
                           "which is well below normal.")
            if correction in ("glasses", "contacts"):
                acuity_line += (" Reading that far below normal <em>with</em> correction "
                                "usually points at the prescription needing an update — or "
                                "at the test not having worked. Check the reliability note "
                                "below before reading anything into it.")
    elif rejected:
        acuity_line = ("<b>No acuity result.</b> The answers given were not distinguishable "
                       "from guessing, so nothing can be read from them — that is a problem "
                       "with the run, not a finding about your eyes.")

    rx_line = ""
    if refraction is not None and refraction.spherical_equivalent is not None:
        lo, hi = refraction.se_interval
        if correction in ("glasses", "contacts"):
            rx_line = (f"A focus estimate was possible, but it describes the <em>leftover</em> "
                       f"error on top of your lenses, not a prescription: roughly {lo:+.2f} "
                       f"to {hi:+.2f} D.")
        else:
            rx_line = (f"Ballpark focus estimate <b>{_html.escape(refraction.prescription_string)}</b>, "
                       f"most likely between {lo:+.2f} and {hi:+.2f} D. That range is wide on "
                       "purpose — no home test is more precise, and it is not a prescription.")

    rank = {"urgent": 0, "prompt": 1, "soon": 2, "routine": 3}
    ordered = sorted(conditions, key=lambda c: (rank.get(c.urgency, 9), -c.probability))
    if ordered and ordered[0].urgency == "urgent":
        cls, action = "act-urgent", f"<b>See someone today.</b> {_html.escape(ordered[0].what_to_do)}"
    elif ordered and ordered[0].urgency == "prompt":
        cls, action = "act-soon", (f"<b>Arrange an eye examination soon.</b> "
                                   f"{_html.escape(ordered[0].what_to_do)}")
    elif ordered:
        names = ", ".join(c.plain_name for c in ordered[:3])
        cls, action = "act-routine", f"<b>Worth an eye test.</b> Most likely: {_html.escape(names)}."
    else:
        cls, action = "act-clear", ("<b>Nothing stood out.</b> Keep to your normal "
                                     "eye-test interval.")

    total = max(len(findings), 1)
    frac = len(measured) / total
    if frac >= 0.75 and not incon:
        rel, rel_cls = "This run went well — nearly every test produced a usable measurement.", "rel-good"
    elif frac >= 0.5:
        rel, rel_cls = (f"Middling run: {len(measured)} of {total} tests gave a usable "
                        f"measurement and {len(incon)} produced none. Treat the rest as "
                        "provisional."), "rel-mixed"
    else:
        rel, rel_cls = (f"Poor run: only {len(measured)} of {total} tests produced a usable "
                        "measurement. There is not enough here to conclude much; retaking it "
                        "is worth more than reading it closely."), "rel-poor"

    body = "".join(f"<p>{x}</p>" for x in (acuity_line, rx_line) if x)
    return f"""
    <section class="headline">
      <h2>The short version</h2>
      <div class="action {cls}">{action}</div>
      {body}
      <p class="worn-note">{_html.escape(worn_note)}</p>
      <p class="reliability {rel_cls}">{_html.escape(rel)}</p>
    </section>"""


def _differential_html(conditions, refraction=None) -> str:
    """The 'what might explain this' section.

    Ordered by urgency first and probability second, because a 7% chance of a
    retinal detachment matters more than a 90% chance of needing reading
    glasses, and a list sorted purely by likelihood would bury it.
    """
    if not conditions and refraction is None:
        return ""
    rank = {"urgent": 0, "prompt": 1, "soon": 2, "routine": 3}
    ordered = sorted(conditions, key=lambda c: (rank.get(c.urgency, 9), -c.probability))

    rx = ""
    if refraction is not None and refraction.spherical_equivalent is not None:
        lo, hi = refraction.se_interval
        rx = f"""
      <div class="rx">
        <div class="rx-label">Ballpark focus estimate — not a prescription</div>
        <div class="rx-value">{_html.escape(refraction.prescription_string)}</div>
        <div class="rx-range">most likely between {lo:+.2f} and {hi:+.2f} D</div>
        <p>{_html.escape(refraction.plain_summary)}</p>
        <ul class="rx-caveats">{''.join(f'<li>{_html.escape(c)}</li>' for c in refraction.caveats)}</ul>
      </div>"""

    rows = []
    for c in ordered:
        ev = "".join(
            f'<li class="{"pro" if e.lr > 1 else "con"}">'
            f'<span class="dir">{e.direction}</span> {_html.escape(e.note)}'
            f'<span class="prov">{e.provenance}</span></li>' for e in c.evidence)
        rows.append(f"""
      <article class="cond {c.urgency}">
        <header>
          <span class="band {c.band.replace(' ', '-')}">{c.band}</span>
          <h3>{_html.escape(c.plain_name)}</h3>
          <span class="urg">{URGENCY_LABEL.get(c.urgency, '')}</span>
        </header>
        <p class="what">{_html.escape(c.what_it_is)}</p>
        <p class="do"><strong>What to do:</strong> {_html.escape(c.what_to_do)}</p>
        <details><summary>Why this came up</summary>
          <ul class="evidence">{ev}</ul>
          <p class="limits"><strong>Limits:</strong> {_html.escape(c.limits)}</p>
        </details>
      </article>""")

    return f"""
    <section class="differential">
      <h2>What might explain these results</h2>
      <p class="lede">These are possibilities ranked from what was measured, not
        conclusions. Each shows the evidence behind it and what that evidence
        cannot settle.</p>
      {rx}{''.join(rows)}
    </section>"""


def _differential_css() -> str:
    return """
    .headline { border: 1px solid var(--line, #2a2a2a); border-radius: 16px; padding: 1.4rem 1.5rem;
                margin: 1.6rem 0 2rem; background: rgba(255,255,255,.025); }
    .headline > h2 { margin: 0 0 .9rem; font-size: 1.35rem; }
    .headline p { margin: .6rem 0; font-size: 1rem; line-height: 1.55; }
    .headline .action { font-size: 1.06rem; line-height: 1.5; padding: .8rem 1rem; border-radius: 10px;
                        border-left: 3px solid var(--line, #2a2a2a); margin-bottom: .9rem; }
    .action.act-urgent  { border-left-color: #ff6b6b; background: rgba(255,107,107,.10); }
    .action.act-soon    { border-left-color: #ffa94d; background: rgba(255,169,77,.10); }
    .action.act-routine { border-left-color: #ffd43b; background: rgba(255,212,59,.07); }
    .action.act-clear   { border-left-color: #8bd4a0; background: rgba(139,212,160,.08); }
    .headline .worn-note { color: var(--muted, #9aa); font-size: .89rem; }
    .headline .reliability { font-size: .89rem; margin-top: .9rem; padding-top: .8rem;
                             border-top: 1px solid var(--line, #2a2a2a); }
    .rel-good { color: #8bd4a0; } .rel-mixed { color: var(--muted, #9aa); } .rel-poor { color: #ffd08a; }
    .differential { margin: 2.4rem 0 2.8rem; }
    .differential > h2 { font-size: 1.6rem; margin: 0 0 .3rem; }
    .differential .lede { color: var(--muted, #9aa); max-width: 62ch; margin: 0 0 1.3rem; font-size: .94rem; }
    .rx { border: 1px solid var(--line, #2a2a2a); border-radius: 14px; padding: 1.15rem 1.3rem;
          margin-bottom: 1.6rem; background: linear-gradient(180deg, rgba(120,190,255,.07), transparent); }
    .rx-label { font-size: .7rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted, #9aa); }
    .rx-value { font-size: 2.1rem; font-variant-numeric: tabular-nums; margin: .2rem 0 .1rem; }
    .rx-range { color: var(--muted, #9aa); font-size: .9rem; margin-bottom: .7rem; }
    .rx p { margin: .5rem 0 0; font-size: .93rem; }
    .rx-caveats { margin: .7rem 0 0; padding-left: 1.1rem; color: var(--muted, #9aa); font-size: .84rem; }
    .cond { border: 1px solid var(--line, #2a2a2a); border-left-width: 3px; border-radius: 12px;
            padding: .95rem 1.15rem; margin: .75rem 0; background: var(--card, rgba(255,255,255,.02)); }
    .cond.urgent { border-left-color: #ff6b6b; }
    .cond.prompt { border-left-color: #ffa94d; }
    .cond.soon   { border-left-color: #ffd43b; }
    .cond header { display: flex; align-items: center; gap: .65rem; flex-wrap: wrap; }
    .cond h3 { margin: 0; font-size: 1.04rem; font-weight: 600; }
    .cond .urg { margin-left: auto; font-size: .75rem; color: var(--muted, #9aa); }
    .band { font-size: .66rem; letter-spacing: .08em; text-transform: uppercase; padding: .18rem .55rem;
            border-radius: 999px; border: 1px solid var(--line, #2a2a2a); }
    .band.likely { background: rgba(255,107,107,.16); }
    .band.possible { background: rgba(255,169,77,.16); }
    .band.less-likely { background: rgba(255,212,59,.11); }
    .cond .what { margin: .6rem 0 .35rem; color: var(--muted, #9aa); font-size: .9rem; }
    .cond .do { margin: .25rem 0; font-size: .9rem; }
    .cond summary { cursor: pointer; font-size: .83rem; color: var(--muted, #9aa); }
    .evidence { margin: .55rem 0; padding-left: 1.15rem; font-size: .85rem; }
    .evidence .dir { font-weight: 600; margin-right: .3rem; }
    .evidence li.pro .dir { color: #ff9a8b; }
    .evidence li.con .dir { color: #8bd4a0; }
    .evidence .prov { font-size: .62rem; opacity: .55; margin-left: .45rem;
                      border: 1px solid var(--line, #2a2a2a); border-radius: 4px; padding: .05rem .3rem; }
    .cond .limits { font-size: .82rem; color: var(--muted, #9aa); margin: .5rem 0 0; }
    @media print { .cond details { display: block; } .cond details summary { display: none; } }
"""


def _css() -> str:
    path = Path(__file__).resolve().parents[2] / "webapp" / "static" / "css" / "app.css"
    base = path.read_text() if path.exists() else ""
    return base + _differential_css() + """
    main.report { max-width: 880px; margin: 0 auto; padding: 2.5rem clamp(1.2rem,5vw,3.5rem) 6rem; }
    main.report h1 { font-size: clamp(2.6rem, 6vw, 4.2rem); line-height: 0.95; margin: 0 0 1.6rem; }

    .banner { padding: 1.5rem 1.7rem; margin: 0 0 2rem; border-left: 2px solid; position: relative; }
    .banner h3 { font-family: var(--display); margin: 0 0 0.4rem; font-size: 1.5rem; }
    .banner p { margin: 0; font-size: 0.95rem; line-height: 1.6; max-width: none; }
    .banner .counts { margin-top: 0.8rem; font-family: var(--mono); font-size: 0.68rem;
                      letter-spacing: 0.06em; opacity: 0.75; }
    .banner.good   { border-color: var(--good); background: linear-gradient(90deg, rgba(127,214,164,.09), transparent 72%); color: var(--good); }
    .banner.warn   { border-color: var(--warn); background: linear-gradient(90deg, rgba(240,195,110,.09), transparent 72%); color: var(--warn); }
    .banner.urgent { border-color: var(--bad);  background: linear-gradient(90deg, rgba(232,131,111,.11), transparent 72%); color: var(--bad); }

    .finding { border-left: 1px solid var(--line); background: var(--ink-050);
               padding: 1.5rem 1.7rem; margin-bottom: 1px; }
    .finding.tier-measured     { border-left: 2px solid var(--good); }
    .finding.tier-weak-signal  { border-left: 2px solid var(--warn); }
    .finding.tier-inconclusive { border-left: 2px solid var(--dim); }
    .finding-head { display:flex; align-items:baseline; gap:0.9rem; margin-bottom:0.5rem; flex-wrap:wrap; }
    .finding-head h2 { margin:0; font-size:1.45rem; text-transform: capitalize; }
    .finding p { margin: 0 0 0.3rem; font-size: 0.95rem; line-height: 1.62; color: var(--paper); max-width: 68ch; }
    .finding .urgent { color: var(--bad); font-weight: 600; margin-top: 0.7rem; }
    .tier-badge { font-family: var(--mono); font-size: 0.58rem; letter-spacing: 0.13em;
                  text-transform: uppercase; color: var(--dim);
                  border: 1px solid var(--line); padding: 0.16rem 0.5rem; }
    .retake-title { color: var(--muted); font-size: 0.85rem; margin-top: 0.9rem; }
    .finding ul { margin: 0.4rem 0 0; padding-left: 1.1rem; color: var(--muted);
                  font-size: 0.85rem; line-height: 1.6; }

    table.metrics-table { width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.85rem; }
    table.metrics-table td { padding: 0.5rem 0; border-bottom: 1px solid var(--line); color: var(--muted); }
    table.metrics-table td:last-child { text-align: right; font-family: var(--mono); color: var(--cyan); }

    .scale { color: var(--muted); margin: 0.9rem 0 0.2rem; display: block; }
    .caveat { color: var(--muted); font-size: 0.82rem; margin-top: 0.9rem;
              border-left: 1px solid var(--line); padding-left: 0.9rem; }
    .disclaimer { border-left: 2px solid var(--amber);
                  background: linear-gradient(90deg, var(--amber-glow), transparent 70%);
                  padding: 1.2rem 1.4rem; font-size: 0.92rem; line-height: 1.6;
                  margin-bottom: 2rem; }
    .disclaimer strong { color: var(--amber); }
    .legend { display:flex; gap:1.6rem; flex-wrap:wrap; margin: 2rem 0 0;
              color: var(--dim); font-size: 0.78rem; }
    .legend b { color: var(--paper); font-family: var(--mono); font-size: 0.68rem;
                letter-spacing: 0.08em; text-transform: uppercase; }
    .actions { margin-top: 2.2rem; display:flex; gap:0.8rem; flex-wrap: wrap; }
    .footnote { color: var(--dim); font-size: 0.78rem; line-height: 1.7;
                margin-top: 2.5rem; border-top: 1px solid var(--line); padding-top: 1.4rem;
                max-width: 72ch; }
    @media print {
      body { background:#fff; color:#111; }
      body::after { display:none; }
      .finding, .card { break-inside: avoid; }
      .actions, .topbar { display:none; }
    }
    """


def render_html(findings: list[Finding], session_id: str,
                conditions=None, refraction=None, correction=None) -> str:
    conditions = list(conditions or [])
    body = "".join(_finding_html(f) for f in findings) or "<p>No results.</p>"
    legend = "".join(
        f"<span><b>{k}</b> — {v}</span>" for k, v in TIER_HELP.items()
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vision Screening Report</title><style>{_css()}</style></head>
<body><div class="chrome">
<header class="topbar">
  <div class="logo">
    <svg class="iris-mark" viewBox="0 0 26 26" aria-hidden="true">
      <circle class="r2" cx="13" cy="13" r="12"/><circle class="r1" cx="13" cy="13" r="8"/>
      <circle class="pupil" cx="13" cy="13" r="3.4"/>
    </svg> VisionScreen</div>
  <div class="spacer"></div><span class="pill">report</span>
</header>
<main class="report">
  <h1>Your screening report</h1>
  <div class="disclaimer"><strong>{DISCLAIMER}</strong> {MISS_RATE_NOTE}</div>
  {_headline_html(findings, conditions, refraction, correction)}\n  {_summary_banner(findings, conditions)}\n  {_differential_html(conditions, refraction)}
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
