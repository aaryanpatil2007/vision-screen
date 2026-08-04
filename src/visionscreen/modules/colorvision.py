"""Pseudoisochromatic color-vision screening (Ishihara-style).

Honest framing up front: a browser on an uncalibrated display cannot reproduce
the spectral properties of printed Ishihara plates. Published work is
consistent that screen-based plates over-refer and cannot grade severity.
This module therefore NEVER returns a `measured` tier — the ceiling is
`weak-signal`, and the summary states the calibration limit explicitly.

What it can legitimately do is flag a likely red-green deficiency (the common
congenital case, ~8% of males) and hint at the protan/deutan axis by which
confusion lines the errors fall along.
"""
from __future__ import annotations

from visionscreen.report import Finding

# Plates are generated client-side as dot mosaics along protan/deutan
# confusion lines; `digit` is the numeral a color-normal viewer reads.
ISHIHARA_STYLE_PLATES = [
    {"id": "p0", "digit": 12, "type": "demo"},
    {"id": "p1", "digit": 8, "type": "general"},
    {"id": "p2", "digit": 6, "type": "protan"},
    {"id": "p3", "digit": 5, "type": "deutan"},
    {"id": "p4", "digit": 29, "type": "general"},
    {"id": "p5", "digit": 3, "type": "protan"},
    {"id": "p6", "digit": 15, "type": "deutan"},
    {"id": "p7", "digit": 74, "type": "general"},
    {"id": "p8", "digit": 2, "type": "protan"},
    {"id": "p9", "digit": 7, "type": "deutan"},
]

FAIL_THRESHOLD = 2  # missed diagnostic plates that trigger a flag


def classify_color_deficiency(protan_errors: int, deutan_errors: int) -> str:
    if protan_errors >= 2 and protan_errors > deutan_errors:
        return "protan-leaning"
    if deutan_errors >= 2 and deutan_errors > protan_errors:
        return "deutan-leaning"
    return "unclassified"


def score_color_vision(answers: dict, valid_fraction: float) -> Finding:
    """answers: {plate_id: digit_reported_or_None}"""
    graded = [p for p in ISHIHARA_STYLE_PLATES if p["type"] != "demo"]
    responded = [p for p in graded if p["id"] in answers]
    if valid_fraction < 0.4 or len(responded) < 4:
        return Finding(
            module="color_vision",
            summary="Color plates test was not completed.",
            tier="inconclusive",
            retakes=["Complete the color plates test in a well-lit room at full screen brightness."],
        )

    errors = {"protan": 0, "deutan": 0, "general": 0}
    for p in responded:
        if answers.get(p["id"]) != p["digit"]:
            errors[p["type"]] += 1
    total_errors = sum(errors.values())

    flags: list[str] = []
    axis = "unclassified"
    if total_errors >= FAIL_THRESHOLD:
        flags.append("possible red-green color deficiency")
        axis = classify_color_deficiency(errors["protan"], errors["deutan"])

    summary = (
        (f"Missed {total_errors} of {len(responded)} color plates — "
         f"possible red-green color deficiency ({axis}). "
         if flags else
         f"Read {len(responded) - total_errors} of {len(responded)} color plates correctly. ")
        + "Screen-based color testing is not calibrated to printed plates; treat this "
        "as a rough screen only and confirm with Ishihara plates if flagged."
    )
    return Finding(
        module="color_vision",
        summary=summary,
        tier="weak-signal",  # never 'measured' on an uncalibrated display
        metrics={
            "flags": flags,
            "errors_total": total_errors,
            "errors_protan": errors["protan"],
            "errors_deutan": errors["deutan"],
            "axis": axis,
            "plates": len(responded),
        },
    )
