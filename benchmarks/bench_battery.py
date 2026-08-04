"""End-to-end battery benchmark on simulated patients.

Each virtual patient has a ground-truth condition set; we simulate their
responses (with realistic lapse/guess behaviour) and their eye images, run the
real scoring modules, and measure detection performance. This is the closest
thing to a clinical validation available without recruiting humans, and it is
labeled as such.

Reference points used for interpretation (clinical literature):
  * chart acuity test-retest repeatability is ~0.1-0.2 logMAR, so anything
    below ~0.15 logMAR error is at the noise floor of the reference test itself
  * clinically significant strabismus is >= ~10 prism diopters
  * clinically reduced contrast sensitivity is < 1.5 log CS
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from visionscreen.modules.acuity import Staircase, score_trials
from visionscreen.modules.alignment import AlignmentFrame, score_alignment
from visionscreen.modules.astigmatic import score_astigmatic_dial
from visionscreen.modules.colorvision import ISHIHARA_STYLE_PLATES, score_color_vision
from visionscreen.modules.contrast import display_ceiling_log_cs, score_contrast
from visionscreen.modules.photoref import measure_reflex
from visionscreen.modules.pupillometry import PupilTrace, score_pupillometry
from visionscreen.synth.photoref import render_reflex

DIRS = ["up", "right", "down", "left"]
CHART_REPEATABILITY_LOGMAR = 0.15
CS_CEILING = display_ceiling_log_cs()


# ---------- virtual observers ----------

def simulate_acuity(true_logmar: float, lapse: float, rng) -> list[dict]:
    s = Staircase()
    out = []
    while not s.done:
        level = s.current()
        shown = DIRS[rng.integers(4)]
        sees = level >= true_logmar and rng.random() > lapse
        answered = shown if sees else DIRS[rng.integers(4)]
        out.append({"logmar": round(level, 2), "shown": shown, "answered": answered})
        s.record(answered == shown)
    return out


def simulate_contrast(true_log_cs: float, lapse: float, rng,
                      ceiling: float | None = None) -> list[dict]:
    """Pelli-Robson triplets: 3 letters per level, stop when a triplet fails.
    The ladder is truncated at the display ceiling, exactly as the app does."""
    out, consecutive_fails = [], 0
    for i in range(16):
        log_cs = round(0.15 * i, 2)
        if ceiling is not None and log_cs > ceiling:
            break
        n_correct = 0
        for _ in range(3):
            correct = (log_cs <= true_log_cs) and (rng.random() > lapse)
            out.append({"log_cs": log_cs, "correct": bool(correct)})
            n_correct += int(correct)
        # One bad triplet can be a lapse; two in a row is the real endpoint.
        consecutive_fails = 0 if n_correct >= 2 else consecutive_fails + 1
        if consecutive_fails >= 2:
            break
    return out


def simulate_color(deficient: bool, rng) -> dict:
    ans = {}
    for p in ISHIHARA_STYLE_PLATES:
        if p["type"] == "demo":
            ans[p["id"]] = p["digit"]
        elif deficient and p["type"] in ("protan", "deutan", "general"):
            ans[p["id"]] = None if rng.random() < 0.8 else p["digit"]
        else:
            ans[p["id"]] = p["digit"] if rng.random() > 0.03 else None
    return ans


def simulate_pupil(amplitude_mm: float, rng, fps=60.0, dur=4.0, flash=0.6,
                   baseline: float = 4.2):
    ts, d = [], []
    base = baseline
    for i in range(int(dur * fps)):
        t = i / fps
        v = base
        since = t - (flash + 0.24)
        if since >= 0:
            v = base - amplitude_mm * np.exp(-since / 0.9) * (1 - np.exp(-since / 0.08))
        ts.append(t)
        d.append(v + rng.normal(0, 0.03))
    return PupilTrace(ts=ts, diameter_mm=d, flash_ts=flash)


def simulate_alignment(deviation_pd: float, rng, n=60) -> list[AlignmentFrame]:
    mm = deviation_pd / 18.0
    return [
        AlignmentFrame(
            (float(rng.normal(0, 0.04)), float(rng.normal(0, 0.04))),
            (float(rng.normal(mm, 0.04)), float(rng.normal(0, 0.04))),
        )
        for _ in range(n)
    ]


# ---------- benchmark ----------

def run(n_patients: int = 120, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    acuity_err, contrast_err = [], []
    contrast_censored = 0
    align_tp = align_fp = align_tn = align_fn = 0
    color_tp = color_fp = color_tn = color_fn = 0
    aniso_tp = aniso_fp = aniso_tn = aniso_fn = 0
    astig_axis_err = []
    photoref_err = []

    for _ in range(n_patients):
        lapse = float(rng.uniform(0.02, 0.10))

        # --- acuity ---
        true_acuity = float(rng.uniform(-0.1, 1.0))
        f = score_trials(simulate_acuity(true_acuity, lapse, rng))
        if f.metrics.get("logmar") is not None:
            acuity_err.append(abs(f.metrics["logmar"] - true_acuity))

        # --- contrast ---
        true_cs = float(rng.uniform(0.6, 2.1))
        f = score_contrast(simulate_contrast(true_cs, lapse, rng, CS_CEILING), 0.9)
        if f.metrics.get("log_cs") is not None:
            if "display_ceiling_log_cs" in f.metrics:
                # The subject exceeded what an 8-bit screen can present. That is
                # an instrument RANGE limit, not an estimation error, and
                # averaging it into the error would misattribute the cause.
                contrast_censored += 1
            else:
                contrast_err.append(abs(f.metrics["log_cs"] - true_cs))

        # --- alignment (half the cohort strabismic) ---
        has_strab = bool(rng.random() < 0.5)
        dev = float(rng.uniform(12, 40)) if has_strab else float(rng.uniform(0, 6))
        f = score_alignment(simulate_alignment(dev, rng), None, 0.9)
        flagged = "possible eye misalignment" in f.metrics.get("flags", [])
        if has_strab and flagged: align_tp += 1
        elif has_strab and not flagged: align_fn += 1
        elif not has_strab and flagged: align_fp += 1
        else: align_tn += 1

        # --- color vision ---
        cvd = bool(rng.random() < 0.35)
        f = score_color_vision(simulate_color(cvd, rng), 0.9)
        flagged = "possible red-green color deficiency" in f.metrics.get("flags", [])
        if cvd and flagged: color_tp += 1
        elif cvd and not flagged: color_fn += 1
        elif not cvd and flagged: color_fp += 1
        else: color_tn += 1

        # --- anisocoria (NOT RAPD: a bilateral flash cannot reveal an afferent
        # defect, so we benchmark the resting size difference we can measure) ---
        has_aniso = bool(rng.random() < 0.3)
        base_l = 4.2
        base_r = base_l - (rng.uniform(1.1, 2.0) if has_aniso else rng.uniform(0, 0.6))
        left = simulate_pupil(1.1, rng, baseline=base_l)
        right = simulate_pupil(1.05, rng, baseline=base_r)
        f = score_pupillometry(left, right, 0.9, fps=60.0)
        flagged = "unequal pupil sizes (anisocoria)" in f.metrics.get("flags", [])
        if has_aniso and flagged: aniso_tp += 1
        elif has_aniso and not flagged: aniso_fn += 1
        elif not has_aniso and flagged: aniso_fp += 1
        else: aniso_tn += 1

        # --- astigmatic dial (axis recovery with human response noise) ---
        true_axis = float(rng.uniform(0, 180))
        dark = (true_axis + 90.0) % 180.0
        responses = [(dark + rng.normal(0, 8)) % 180 for _ in range(2)]
        f = score_astigmatic_dial(responses, 0.9, False)
        if f.metrics.get("axis_deg") is not None:
            d = abs(f.metrics["axis_deg"] - true_axis) % 180
            astig_axis_err.append(min(d, 180 - d))

        # --- photorefraction (image-level, real optics path) ---
        S = float(rng.choice([-1, 1]) * rng.uniform(1.6, 4.0))
        img, truth = render_reflex(32, S=S, noise_sigma=8.0, rng=rng,
                                   e_m=0.005, d_m=0.5, px_per_m=8000.0)
        est = measure_reflex(img, truth["center_px"], truth["pupil_radius_px"],
                             e_m=0.005, d_m=0.5, px_per_m=8000.0)
        if est is not None:
            photoref_err.append(abs((est[0] + est[1] / 2) - S))

    def prf(tp, fp, tn, fn):
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        return {"sensitivity": round(sens, 3), "specificity": round(spec, 3),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn}

    return {
        "n_patients": n_patients,
        "acuity": {
            "mean_abs_error_logmar": round(float(np.mean(acuity_err)), 3),
            "p95_abs_error_logmar": round(float(np.percentile(acuity_err, 95)), 3),
            "within_chart_repeatability_pct": round(
                100 * float(np.mean(np.array(acuity_err) <= CHART_REPEATABILITY_LOGMAR)), 1),
        },
        "contrast": {
            "mean_abs_error_log_cs": round(float(np.mean(contrast_err)), 3),
            "p95_abs_error_log_cs": round(float(np.percentile(contrast_err, 95)), 3),
            "n_measurable": len(contrast_err),
            "n_above_display_ceiling": contrast_censored,
            "display_ceiling_log_cs": round(CS_CEILING, 2),
        },
        "alignment_strabismus": prf(align_tp, align_fp, align_tn, align_fn),
        "color_vision": prf(color_tp, color_fp, color_tn, color_fn),
        "anisocoria": prf(aniso_tp, aniso_fp, aniso_tn, aniso_fn),
        "astigmatism_axis": {
            "mean_abs_axis_error_deg": round(float(np.mean(astig_axis_err)), 1),
        },
        "photorefraction": {
            "mean_abs_sphere_equiv_error_d": round(float(np.mean(photoref_err)), 3),
        },
    }


def main() -> None:
    result = run()
    Path("results").mkdir(exist_ok=True)
    Path("results/battery.json").write_text(json.dumps(result, indent=2))
    a = result["acuity"]
    print(f"patients: {result['n_patients']}\n")
    print("| test | metric | value |")
    print("|---|---|---|")
    print(f"| acuity | mean abs error | {a['mean_abs_error_logmar']} logMAR |")
    print(f"| acuity | within chart repeatability (0.15) | {a['within_chart_repeatability_pct']}% |")
    c = result["contrast"]
    print(f"| contrast | mean abs error (in range) | {c['mean_abs_error_log_cs']} log CS |")
    print(f"| contrast | above display ceiling ({c['display_ceiling_log_cs']}) | "
          f"{c['n_above_display_ceiling']}/{result['n_patients']} |")
    for key in ("alignment_strabismus", "color_vision", "anisocoria"):
        r = result[key]
        print(f"| {key} | sensitivity / specificity | {r['sensitivity']} / {r['specificity']} |")
    print(f"| astigmatism | mean axis error | {result['astigmatism_axis']['mean_abs_axis_error_deg']}° |")
    print(f"| photorefraction | mean abs SE error | "
          f"{result['photorefraction']['mean_abs_sphere_equiv_error_d']} D |")


if __name__ == "__main__":
    main()
