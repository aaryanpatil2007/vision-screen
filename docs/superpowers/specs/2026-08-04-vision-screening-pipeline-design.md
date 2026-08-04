# Vision Screening Pipeline — Design

**Date:** 2026-08-04
**Status:** Approved design, pending implementation plan
**Context:** Learning/research project. Deliverable is a working demo backed by quantified benchmarks. All data acquisition must be autonomous (public datasets, scraped public examples, synthetic generation) — no human-subject collection.

## Goal

A computer-vision screening pipeline that takes a guided webcam/phone video of a person performing on-screen tests and outputs a vision **screening report** — flags for possible refractive error (myopia/hyperopia/astigmatism), eye misalignment (strabismus), and reduced acuity, each with a confidence tier. It is explicitly a screening signal ("see an optometrist"), never a diagnosis.

## Architecture (Approach C: physics spine + learned perception)

The pipeline's structure is physics: clinically established geometric/optical relations (Hirschberg ratio, eccentric-photorefraction equations) perform the final inversion to clinical quantities. The perception steps feeding them (pupil segmentation, corneal-reflex and crescent extraction under real-world lighting) are small learned models trained on synthetic renders plus public datasets, each with a classical CV fallback.

Rationale over alternatives: pure classical CV (Approach A) is brittle on dim webcam frames and teaches less ML; end-to-end learning on synthetic data (Approach B) makes the sim-to-real gap the unmeasurable core risk. C keeps A's explainability and per-module benchmarkability while using ML where it earns its place, and degrades gracefully — classical fallbacks keep the demo alive if a net underperforms.

```
Guided capture app (webcam/phone, on-screen instructions)
        │  one video per test segment + metadata (timestamps, screen state)
        ▼
Perception layer: face/iris landmarks → eye crops → pupil segmentation,
                  corneal-reflex + crescent extraction
        ▼
Physics/scoring layer: Module 1 (behavioral+acuity), Module 2 (alignment),
                       Module 3 (photorefraction)
        ▼
Screening report: per-test findings + confidence tier, screening-only framing
```

Supporting systems: a **synthetic eye renderer** (labeled training/benchmark data) and a **benchmark harness** (one command → results tables).

## Test Modules

Build order: Module 1 → 2 → 3 (increasing difficulty). Each module is independently runnable and benchmarkable; a working demo exists from Module 1 onward.

### Module 1 — Behavioral + acuity
On-screen tumbling-E chart at scaled sizes; the user answers by keypress/gesture while the camera watches for squinting (eye-aspect-ratio), leaning in (face-size change as distance proxy), and head tilt. Output: estimated acuity line + behavioral flags.

### Module 2 — Alignment (strabismus)
Screen shows a moving dot; both eyes' pursuit is tracked. Hirschberg check: screen-flash corneal reflex position relative to pupil center compared across eyes — asymmetry converts to degrees of deviation via the Hirschberg ratio. A guided cover-one-eye step catches latent deviation on the uncovered eye.

### Module 3 — Photorefraction (refractive error)
Dim room; phone flash or bright screen near the camera axis. The red-reflex crescent is segmented; crescent extent inverts to spherical defocus (diopters) via the eccentric-photorefraction equation, and crescent-axis rotation indicates astigmatism axis. Hardest module; the method is clinically proven (commercial photoscreeners), but real-webcam accuracy remains unvalidated until tested on real eyes — the design reports this honestly rather than hiding it.

## Data Strategy

Priority order; every scraped item gets provenance logged (URL, license, label source):

1. **Public datasets** for perception training: MPIIGaze / GazeCapture (gaze), UnityEyes (public synthetic eye renders), CASIA-Iris / UBIRIS (pupil/iris segmentation), CEW (closed/squinting eyes).
2. **Scraped real-world condition examples**: strabismus photo datasets (Kaggle/figshare), red-reflex and photoscreening images from open-access ophthalmology papers, leukocoria sets, clinical figures with crescents at known diopters. Small-N but real — used as a reality-check evaluation set and for fine-tuning.
3. **Synthetic renderer**: dense, perfectly labeled coverage of gaze × alignment × refraction × lighting. Physically motivated 2-D crescent model first; upgrade to ray-traced eye only if benchmarks demand it.

## Error Handling

- **Capture-quality gates** before analysis (face found, eye pixel size, lighting range, protocol step followed, flash visible for Module 3). Failure → "retake this segment," never a garbage estimate.
- **Confidence tiers** per module: measured / weak-signal / inconclusive. The report never fills in an inconclusive module.
- **Classical fallbacks**: when a learned component's uncertainty is high, thresholded pupil detection / template-matched reflex takes over so the demo never hard-fails.

## Benchmarks

One command runs the full evaluation and emits results tables:

- **Perception:** segmentation IoU and landmark error (pixels) on held-out synthetic + public splits, with the scraped real-world set as a separate "reality" column — the sim-to-real gap is itself a reported number.
- **Physics inversions:** Hirschberg angle error (degrees), photorefraction error (diopters) on synthetic ground truth, swept across lighting/gaze/distance to map failure boundaries.
- **End-to-end:** screening-flag accuracy on synthetic "patients" with assigned conditions.

## Deliverables

1. **Demo app** — browser-based capture UI walking through the guided protocol; outputs an HTML screening report with per-test findings, confidence tiers, and screening-only framing.
2. **Research-style writeup** — benchmark tables, methods, error analysis, dataset provenance.

## Stack

Python; MediaPipe for face/iris landmarks; PyTorch for perception nets; OpenCV for classical fallbacks; Python renderer; thin web front-end for the capture protocol.

## Out of Scope

- Any claim of diagnosis; regulatory (FDA/SaMD) work.
- Human-subject data collection.
- Conditions beyond refractive error, alignment, and acuity (no fundus/disease detection).
