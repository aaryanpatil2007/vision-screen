from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from typing import Literal

Tier = Literal["measured", "weak-signal", "inconclusive"]

DISCLAIMER = (
    "This is a screening signal only, not a diagnosis. "
    "See an optometrist for a clinical evaluation."
)


@dataclass
class Finding:
    module: str
    summary: str
    tier: Tier
    metrics: dict = field(default_factory=dict)
    retakes: list[str] = field(default_factory=list)


def _finding_html(f: Finding) -> str:
    parts = [f"<section class='finding tier-{f.tier}'>"]
    parts.append(f"<h2>{_html.escape(f.module)} <em>({f.tier})</em></h2>")
    parts.append(f"<p>{_html.escape(f.summary)}</p>")
    if f.tier != "inconclusive" and f.metrics:
        rows = "".join(
            f"<tr><td>{_html.escape(k)}</td><td>{_html.escape(str(v))}</td></tr>"
            for k, v in f.metrics.items()
        )
        parts.append(f"<table>{rows}</table>")
    if f.retakes:
        items = "".join(f"<li>{_html.escape(r)}</li>" for r in f.retakes)
        parts.append(f"<p>To improve this result, retake:</p><ul>{items}</ul>")
    parts.append("</section>")
    return "".join(parts)


def render_html(findings: list[Finding], session_id: str) -> str:
    body = "".join(_finding_html(f) for f in findings) or "<p>No results.</p>"
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Vision Screening Report</title></head><body>"
        f"<h1>Vision Screening Report — session {_html.escape(session_id)}</h1>"
        f"<p class='disclaimer'><strong>{DISCLAIMER}</strong></p>"
        f"{body}"
        "</body></html>"
    )
