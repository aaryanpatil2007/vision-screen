"""Test-retest repeatability — the validation half that needs no clinician.

Agreement with a clinician requires a clinician. Repeatability does not: run
the same subject twice and measure how far the two answers land apart. It is
half of validity (a test that disagrees with itself cannot agree with anything
else), it is what the published app studies report, and it is directly
comparable to them.

Reference points from the literature:

    DigiVis test-retest        95% LoA  +/-0.12 logMAR
    Peek Acuity test-retest    CoR       0.033 logMAR
    ETDRS chart, normals       95% TRV  +/-0.11 logMAR
    ETDRS chart, 1.00 D blur   95% TRV  +/-0.25 logMAR
    Pelli-Robson (letter)      CoR       0.14-0.21 log CS

The coefficient of repeatability is 1.96 * SD of the differences between two
measurements of the same unchanged subject. Each virtual subject here is run
twice with independent response noise and independent stimulus randomisation —
the same subject, two honest sessions.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import numpy as np

from benchmarks.bench_battery import (
    CS_CEILING,
    simulate_acuity,
    simulate_alignment,
    simulate_contrast,
)
from visionscreen.modules.acuity import score_trials
from visionscreen.modules.alignment import score_alignment
from visionscreen.modules.contrast import score_contrast

REFERENCE = {
    "acuity_logmar": {
        "ours": None,
        "DigiVis test-retest (LoA)": 0.12,
        "Peek Acuity (CoR)": 0.033,
        "ETDRS normals (95% TRV)": 0.11,
        "ETDRS 1.00 D blur (95% TRV)": 0.25,
    },
    "contrast_logcs": {
        "ours": None,
        "Pelli-Robson letter-by-letter (CoR)": 0.17,
        "PeekCS (ICC 0.93 vs chart 0.96)": None,
    },
}


def coefficient_of_repeatability(diffs: list[float]) -> dict:
    n = len(diffs)
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    bias = statistics.fmean(diffs)
    return {
        "n_pairs": n,
        "bias": round(bias, 4),
        "sd_of_differences": round(sd, 4),
        "coefficient_of_repeatability": round(1.96 * sd, 4),
        "loa95": [round(bias - 1.96 * sd, 4), round(bias + 1.96 * sd, 4)],
        "within_0.1": round(sum(abs(d) <= 0.1 for d in diffs) / n, 3) if n else None,
        "within_0.15": round(sum(abs(d) <= 0.15 for d in diffs) / n, 3) if n else None,
    }


def run(n_subjects: int = 300, seed: int = 17) -> dict:
    rng = np.random.default_rng(seed)
    acuity_diffs, contrast_diffs, align_diffs = [], [], []
    acuity_flip = contrast_flip = align_flip = 0
    align_pairs = 0

    for _ in range(n_subjects):
        # a subject's own lapse rate is a property of the person, held fixed
        lapse = float(rng.uniform(0.02, 0.10))

        true_acuity = float(rng.uniform(-0.1, 1.0))
        a1 = score_trials(simulate_acuity(true_acuity, lapse, rng))
        a2 = score_trials(simulate_acuity(true_acuity, lapse, rng))
        v1 = a1.metrics.get("logmar_raw_tumbling_e")
        v2 = a2.metrics.get("logmar_raw_tumbling_e")
        if v1 is not None and v2 is not None:
            acuity_diffs.append(v1 - v2)
            # would the two sessions disagree about referral (>0.3 logMAR)?
            if (v1 > 0.3) != (v2 > 0.3):
                acuity_flip += 1

        true_cs = float(rng.uniform(0.6, 2.0))
        c1 = score_contrast(simulate_contrast(true_cs, lapse, rng, CS_CEILING), 0.9)
        c2 = score_contrast(simulate_contrast(true_cs, lapse, rng, CS_CEILING), 0.9)
        w1, w2 = c1.metrics.get("log_cs"), c2.metrics.get("log_cs")
        if w1 is not None and w2 is not None:
            contrast_diffs.append(w1 - w2)
            if (w1 < 1.5) != (w2 < 1.5):
                contrast_flip += 1

        dev = float(rng.uniform(0, 40))
        s1 = score_alignment(simulate_alignment(dev, rng, n=30), None, 0.9)
        s2 = score_alignment(simulate_alignment(dev, rng, n=30), None, 0.9)
        d1, d2 = s1.metrics.get("deviation_pd"), s2.metrics.get("deviation_pd")
        if d1 is not None and d2 is not None:
            align_diffs.append(d1 - d2)
            align_pairs += 1
            f1 = "possible eye misalignment" in s1.metrics.get("flags", [])
            f2 = "possible eye misalignment" in s2.metrics.get("flags", [])
            if f1 != f2:
                align_flip += 1

    out = {
        "n_subjects": n_subjects,
        "acuity_logmar": coefficient_of_repeatability(acuity_diffs),
        "contrast_logcs": coefficient_of_repeatability(contrast_diffs),
        "alignment_pd": coefficient_of_repeatability(align_diffs),
        "decision_stability": {
            "acuity_referral_flip_rate": round(acuity_flip / max(len(acuity_diffs), 1), 4),
            "contrast_flag_flip_rate": round(contrast_flip / max(len(contrast_diffs), 1), 4),
            "alignment_flag_flip_rate": round(align_flip / max(align_pairs, 1), 4),
        },
        "reference_points": REFERENCE,
    }
    out["reference_points"]["acuity_logmar"]["ours"] = \
        out["acuity_logmar"]["coefficient_of_repeatability"]
    out["reference_points"]["contrast_logcs"]["ours"] = \
        out["contrast_logcs"]["coefficient_of_repeatability"]
    return out


def main() -> None:
    r = run()
    Path("results").mkdir(exist_ok=True)
    Path("results/repeatability.json").write_text(json.dumps(r, indent=2))

    print(f"Test-retest repeatability, {r['n_subjects']} virtual subjects "
          "run twice each\n")
    print("| measure | bias | CoR (1.96 SD) | 95% LoA |")
    print("|---|---|---|---|")
    for key, unit in (("acuity_logmar", "logMAR"), ("contrast_logcs", "log CS"),
                      ("alignment_pd", "PD")):
        m = r[key]
        print(f"| {key} | {m['bias']:+.3f} | {m['coefficient_of_repeatability']:.3f} "
              f"{unit} | {m['loa95'][0]:+.3f} to {m['loa95'][1]:+.3f} |")

    a = r["acuity_logmar"]
    print(f"\nacuity: {a['within_0.1']:.0%} of repeat pairs within 0.10 logMAR, "
          f"{a['within_0.15']:.0%} within 0.15")
    print("\n| comparison (acuity CoR / LoA, logMAR) | value |")
    print("|---|---|")
    for k, v in r["reference_points"]["acuity_logmar"].items():
        if v is not None:
            print(f"| {k} | {v} |")

    d = r["decision_stability"]
    print(f"\nDecision stability across repeat sessions:")
    print(f"  acuity referral (>0.3 logMAR) flips: {d['acuity_referral_flip_rate']:.1%}")
    print(f"  contrast flag flips:                 {d['contrast_flag_flip_rate']:.1%}")
    print(f"  alignment flag flips:                {d['alignment_flag_flip_rate']:.1%}")


if __name__ == "__main__":
    main()
