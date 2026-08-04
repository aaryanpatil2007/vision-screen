"""Error budget: what agreement with a clinical chart should be expected.

Repeatability bounds the noise; it says nothing about bias. The usual way to
bound bias without the reference instrument is a budget — enumerate every term
that can shift or scatter the result, quantify each from measurement where
possible, and combine them.

This is a *prediction*, not a validation. It says what the design is capable of
if the components behave as measured, and it identifies which term dominates
so effort goes to the right place. It cannot detect a bias nobody thought of;
only comparison against a clinician does that.

Each term carries its provenance: `measured` values come from a benchmark in
this repo, `literature` from a published figure, `assumed` from reasoning that
has not been checked.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Provenance = Literal["measured", "literature", "assumed"]


@dataclass(frozen=True)
class Term:
    name: str
    bias_logmar: float          # systematic shift (signed)
    sd_logmar: float            # random contribution (1 SD)
    provenance: Provenance
    note: str
    correctable: bool = False   # can be removed by a known adjustment


# --- acuity budget -----------------------------------------------------------
# Ratio errors become logMAR via log10(ratio): a 10% size error is 0.041 logMAR.

def _ratio_to_logmar(ratio_error: float) -> float:
    return abs(math.log10(1.0 + ratio_error))


ACUITY_TERMS: tuple[Term, ...] = (
    Term(
        "optotype rendering", 0.004, 0.004, "measured",
        "tumbling-E drawn size vs requested, measured in-browser across "
        "20-320 px (test_render_fidelity): worst 0.02, mean <0.01 logMAR",
    ),
    Term(
        "optotype equivalence", -0.15, 0.05, "literature",
        "tumbling E reads ~0.15 logMAR conservative vs ETDRS Sloan letters "
        "(Landolt C -0.12 vs ETDRS; tumbling E -0.05 vs Landolt C)",
        correctable=True,
    ),
    Term(
        "screen scale calibration", 0.0, _ratio_to_logmar(0.012), "assumed",
        "credit-card mire, +/-1 mm on an 85.60 mm reference = 1.2% scale error",
    ),
    Term(
        "viewing distance", 0.0, _ratio_to_logmar(0.036), "literature",
        "distance derived from iris diameter; HVID population CV 3.6% "
        "(11.71 +/- 0.42 mm) propagates directly as a scale error",
    ),
    Term(
        "staircase estimation", 0.0, 0.049, "measured",
        "from the measured test-retest CoR of 0.137 logMAR: a single session "
        "carries SD = CoR / (1.96 * sqrt(2))",
    ),
    Term(
        "threshold criterion", 0.0, 0.0, "measured",
        "staircase converges at the guessing-corrected 50% point "
        "(S_up/S_down = 5/3); the previous 0.1/0.2 pair sat at 55.6%",
    ),
)

ACCEPTANCE_BIAS = 0.05
ACCEPTANCE_LOA = 0.20


def combine(terms: tuple[Term, ...], apply_corrections: bool = True) -> dict:
    """Total bias and 95% limits of agreement from independent terms."""
    bias = sum(
        t.bias_logmar for t in terms
        if not (apply_corrections and t.correctable)
    )
    var = sum(t.sd_logmar ** 2 for t in terms)
    sd = math.sqrt(var)
    return {
        "bias": round(bias, 4),
        "sd": round(sd, 4),
        "loa_half_width": round(1.96 * sd, 4),
        "loa95": [round(bias - 1.96 * sd, 4), round(bias + 1.96 * sd, 4)],
        "meets_bias_criterion": abs(bias) <= ACCEPTANCE_BIAS,
        "meets_loa_criterion": 1.96 * sd <= ACCEPTANCE_LOA,
    }


def dominant_terms(terms: tuple[Term, ...], top: int = 3) -> list[tuple[str, float]]:
    """Variance contributions, largest first — where effort should go."""
    total = sum(t.sd_logmar ** 2 for t in terms) or 1.0
    ranked = sorted(
        ((t.name, t.sd_logmar ** 2 / total) for t in terms),
        key=lambda kv: -kv[1],
    )
    return [(n, round(f, 3)) for n, f in ranked[:top] if f > 0]


def acuity_budget() -> dict:
    corrected = combine(ACUITY_TERMS, apply_corrections=True)
    raw = combine(ACUITY_TERMS, apply_corrections=False)
    return {
        "measure": "acuity_logmar",
        "with_known_corrections_applied": corrected,
        "without_corrections": raw,
        "dominant_variance_terms": dominant_terms(ACUITY_TERMS),
        "acceptance": {"max_abs_bias": ACCEPTANCE_BIAS,
                       "max_loa_half_width": ACCEPTANCE_LOA},
        "terms": [
            {
                "name": t.name, "bias": t.bias_logmar, "sd": round(t.sd_logmar, 4),
                "provenance": t.provenance, "correctable": t.correctable,
                "note": t.note,
            }
            for t in ACUITY_TERMS
        ],
        "caveat": (
            "A budget predicts agreement from components that were measured or "
            "cited; it cannot reveal a bias nobody enumerated. It is not a "
            "substitute for comparison against a clinician."
        ),
    }


def format_budget(b: dict) -> str:
    c = b["with_known_corrections_applied"]
    r = b["without_corrections"]
    lines = [
        f"# Predicted agreement — {b['measure']}",
        "",
        "| term | bias | SD | provenance |",
        "|---|---|---|---|",
    ]
    for t in b["terms"]:
        mark = " (correctable)" if t["correctable"] else ""
        lines.append(
            f"| {t['name']}{mark} | {t['bias']:+.3f} | {t['sd']:.3f} | {t['provenance']} |"
        )
    lines += [
        "",
        f"**Uncorrected:** bias {r['bias']:+.3f}, 95% LoA "
        f"{r['loa95'][0]:+.3f} to {r['loa95'][1]:+.3f}",
        f"**With the known optotype correction applied:** bias {c['bias']:+.3f}, "
        f"95% LoA {c['loa95'][0]:+.3f} to {c['loa95'][1]:+.3f}",
        "",
        f"Acceptance: |bias| <= {b['acceptance']['max_abs_bias']}, "
        f"LoA half-width <= {b['acceptance']['max_loa_half_width']}",
        f"  bias criterion: {'met' if c['meets_bias_criterion'] else 'NOT met'}",
        f"  LoA criterion:  {'met' if c['meets_loa_criterion'] else 'NOT met'}",
        "",
        "Dominant variance terms: "
        + ", ".join(f"{n} ({f:.0%})" for n, f in b["dominant_variance_terms"]),
        "",
        b["caveat"],
    ]
    return "\n".join(lines)


def main() -> None:
    import json
    from pathlib import Path

    b = acuity_budget()
    Path("results").mkdir(exist_ok=True)
    Path("results/error_budget.json").write_text(json.dumps(b, indent=2))
    print(format_budget(b))


if __name__ == "__main__":
    main()
