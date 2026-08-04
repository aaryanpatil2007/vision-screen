"""Real-patient validation: does the alignment module flag actual strabismus?

Every other benchmark in this repo scores simulated observers or synthetic
images. This one runs the *production* alignment path over photographs of
people who were clinically categorised as having esotropia, exotropia or
hypertropia, with sex/age/lighting/camera all uncontrolled — and over a sample
of ordinary faces as negatives.

**What this does and does not establish.** It is a real-patient test of the
Hirschberg pathway on single still images, so a positive result means the
geometry survives contact with real clinical photographs rather than only with
renders. It is NOT a clinical validation: n is small, the "diagnosis" is
category membership rather than a cover test, the negatives are presumed-normal
rather than examined, and a single photograph is not the guided multi-frame
capture the product actually uses. Confidence intervals are Wilson intervals
and are wide by construction.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

from visionscreen.analyzer import _eye_measurements
from visionscreen.ml.infer import EyeSegmenter
from visionscreen.modules.alignment import AlignmentFrame, score_alignment
from visionscreen.perception.landmarks import LandmarkExtractor

CLINICAL = Path("data/real/clinical/labels.json")
NEGATIVE_POOL = Path("data/real/hf/RafeiKAr__eye_tracking_gazecapture/raw")
RESULTS = Path("results/clinical_validation.json")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — honest at the tiny n this benchmark has."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def measure_image(path: Path, extractor: LandmarkExtractor,
                  segmenter: EyeSegmenter) -> dict | None:
    """Run the production alignment path on one still photograph."""
    frame = cv2.imread(str(path))
    if frame is None:
        return None
    # upscale small crops: the pipeline needs enough pixels across the iris
    h, w = frame.shape[:2]
    if max(h, w) < 900:
        scale = 900 / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_CUBIC)
    face = extractor.extract(frame)
    if not face.ok:
        return {"face": False}

    dec = {}
    for side in ("left", "right"):
        d, _ = _eye_measurements(frame, face.landmarks, side, segmenter)
        if d is not None:
            dec[side] = d
    if len(dec) < 2:
        return {"face": True, "reflex": False}

    finding = score_alignment(
        [AlignmentFrame(dec["left"], dec["right"])], None, valid_fraction=1.0
    )
    return {
        "face": True,
        "reflex": True,
        "flagged": "possible eye misalignment" in finding.metrics.get("flags", []),
        "deviation_pd": finding.metrics.get("deviation_pd"),
        "tier": finding.tier,
    }


def run(n_negatives: int = 60, seed: int = 5) -> dict:
    if not CLINICAL.exists():
        raise SystemExit("run visionscreen.data.clinical_images first")
    labels = json.loads(CLINICAL.read_text())
    positives = [
        it for it in labels["items"]
        if it.get("misalignment_positive") and Path(it["local_path"]).exists()
    ]

    negatives: list[Path] = []
    if NEGATIVE_POOL.is_dir():
        pool = sorted(NEGATIVE_POOL.glob("*.jpg"))
        rng = random.Random(seed)
        negatives = rng.sample(pool, min(n_negatives, len(pool)))

    out = {"positives": [], "negatives": [], "notes": {}}
    segmenter = EyeSegmenter()
    with LandmarkExtractor() as extractor:
        for it in positives:
            r = measure_image(Path(it["local_path"]), extractor, segmenter)
            out["positives"].append({"title": it["title"], "label": it["label"],
                                     **(r or {"face": False})})
        for p in negatives:
            r = measure_image(p, extractor, segmenter)
            out["negatives"].append({"file": p.name, **(r or {"face": False})})

    pos_measurable = [r for r in out["positives"] if r.get("reflex")]
    neg_measurable = [r for r in out["negatives"] if r.get("reflex")]
    tp = sum(1 for r in pos_measurable if r["flagged"])
    fn = len(pos_measurable) - tp
    fp = sum(1 for r in neg_measurable if r["flagged"])
    tn = len(neg_measurable) - fp

    out["summary"] = {
        "positives_total": len(out["positives"]),
        "positives_measurable": len(pos_measurable),
        "negatives_total": len(out["negatives"]),
        "negatives_measurable": len(neg_measurable),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "sensitivity": round(tp / len(pos_measurable), 3) if pos_measurable else None,
        "sensitivity_ci95": [round(x, 3) for x in wilson(tp, len(pos_measurable))],
        "specificity": round(tn / len(neg_measurable), 3) if neg_measurable else None,
        "specificity_ci95": [round(x, 3) for x in wilson(tn, len(neg_measurable))],
        "measurable_rate_positives": round(
            len(pos_measurable) / max(len(out["positives"]), 1), 3),
        "measurable_rate_negatives": round(
            len(neg_measurable) / max(len(out["negatives"]), 1), 3),
    }
    devs = [r["deviation_pd"] for r in pos_measurable if r.get("deviation_pd") is not None]
    if devs:
        out["summary"]["median_deviation_pd_positives"] = round(float(np.median(devs)), 1)
    devs_n = [r["deviation_pd"] for r in neg_measurable if r.get("deviation_pd") is not None]
    if devs_n:
        out["summary"]["median_deviation_pd_negatives"] = round(float(np.median(devs_n)), 1)
    return out


def main() -> None:
    result = run()
    Path("results").mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=2))
    s = result["summary"]
    print("REAL-PATIENT alignment validation (single still photographs)\n")
    print("| metric | value |")
    print("|---|---|")
    print(f"| diagnosed patients | {s['positives_total']} "
          f"({s['positives_measurable']} measurable) |")
    print(f"| presumed-normal faces | {s['negatives_total']} "
          f"({s['negatives_measurable']} measurable) |")
    print(f"| sensitivity | {s['sensitivity']} "
          f"(95% CI {s['sensitivity_ci95'][0]}-{s['sensitivity_ci95'][1]}) |")
    print(f"| specificity | {s['specificity']} "
          f"(95% CI {s['specificity_ci95'][0]}-{s['specificity_ci95'][1]}) |")
    if "median_deviation_pd_positives" in s:
        print(f"| median deviation, patients | {s['median_deviation_pd_positives']} PD |")
    if "median_deviation_pd_negatives" in s:
        print(f"| median deviation, controls | {s['median_deviation_pd_negatives']} PD |")
    print("\nNot a clinical validation: category membership is not a cover test, "
          "controls are presumed-normal, and one still is not a guided capture.")


if __name__ == "__main__":
    main()
