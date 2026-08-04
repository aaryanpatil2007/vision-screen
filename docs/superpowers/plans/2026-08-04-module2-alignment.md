# Vision Screening — Module 2 (Alignment) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Plan 2 of 3; builds on foundation-module1 interfaces.

**Goal:** Add strabismus/alignment screening: iris + corneal-reflex perception, Hirschberg deviation estimate, pursuit conjugacy, cover-test refixation detection — with a synthetic eye renderer for tests/benchmarks.

**Tech notes (verified empirically):** FaceLandmarker iris centers: index 468 belongs to the eye with corners 33/133 (`eyes.py` "left"); 473 to corners 362/263 ("right"). Iris ring points: 469–472 and 474–477.

**Clinical constants** (documented in code, cited in writeup): horizontal visible iris diameter HVID ≈ 11.7 mm (px→mm scale); Hirschberg ratio ≈ 18 prism diopters (PD) per mm of reflex decentration; 1 PD = arctan(1/100).

## Global Constraints

Same as Plan 1 (tiers, disclaimer, gates-never-guess). New: alignment flag thresholds — inter-eye reflex-decentration asymmetry ≥ 1.0 mm (≈18 PD) → flag "possible eye misalignment"; ≥ 0.5 mm → weak-signal note; pursuit conjugacy flag when inter-eye gaze-correlation < 0.8 over a valid pursuit segment.

### Task 1: Iris + corneal reflex perception (`perception/iris.py`)
- `iris_center(landmarks, side) -> np.ndarray (2,)` normalized coords (468/473).
- `iris_diameter_px(landmarks, side, frame_w, frame_h) -> float` from iris ring extremes.
- `eye_crop(frame_bgr, landmarks, side, pad=0.5) -> (crop, origin_xy)` around eye corners.
- `detect_corneal_reflex(crop_gray) -> (x, y) | None`: brightest pixel cluster ≥ 96th percentile and ≥ 200 intensity; centroid; None if no such cluster (reflex absent → gates/weak-signal path).
- Tests: synthetic crop with a bright dot (exact centroid recovered ±1 px); all-dark crop → None; real astronaut image — iris centers fall inside eye-corner x-ranges (regression-pins the 468/473 mapping).

### Task 2: Synthetic eye renderer v1 (`synth/eyes2d.py`)
- `render_eye(width_px, iris_diameter_px, reflex_offset_mm=(dx, dy), pupil_ratio=0.4) -> (img_bgr, truth)`: sclera background, iris disk, pupil disk, specular reflex dot at pupil-center + offset·px_per_mm (px_per_mm = iris_diameter_px / 11.7). Truth dict: reflex px position, centers, px_per_mm.
- `render_eye_pair(offset_left_mm, offset_right_mm, ...) -> (img, truth)` side-by-side canvas.
- Tests: reflex lands at truth position; px_per_mm scale correct. (This file is the seed the Plan 3 crescent renderer extends.)

### Task 3: Hirschberg + alignment scoring (`modules/alignment.py`)
- `reflex_decentration_mm(reflex_xy_px, iris_center_px, iris_diameter_px) -> (dx_mm, dy_mm)`.
- `hirschberg_pd(decentration_mm) -> float` (magnitude · 18 PD/mm).
- `score_alignment(per_frame: list[AlignmentFrame], pursuit: PursuitResult | None, valid_fraction: float) -> Finding` where `AlignmentFrame(dec_left_mm, dec_right_mm)`. Uses median decentrations; asymmetry = |median_L − median_R| (vector norm); thresholds per Global Constraints; tiers by valid_fraction as in behavioral.
- Tests: symmetric decentrations → no flag (a shared offset is camera geometry, not strabismus); 1.5 mm asymmetry → flag + PD value ≈ 27; low valid_fraction → inconclusive with retakes.

### Task 4: Pursuit conjugacy (`modules/alignment.py`, same file)
- `pursuit_conjugacy(gaze_left: list[float], gaze_right: list[float], dot_x: list[float]) -> PursuitResult(corr_left, corr_right, conjugacy)`: Pearson correlations eye-vs-dot and eye-vs-eye (conjugacy); needs ≥ 20 samples else None.
- Gaze signal = normalized iris-center x within eye corners: `(iris_x − corner_a_x) / (corner_b_x − corner_a_x)`.
- Tests: perfectly conjugate synthetic sinusoid gaze → conjugacy ≈ 1, no flag; one eye lagging/flat → conjugacy < 0.8 → flag via `score_alignment`.

### Task 5: Analyzer + webapp integration
- `analyzer.py`: for segment `test_id="alignment"`, per gate-passing frame in the segment's time window (frame ts = idx/meta.fps) collect AlignmentFrames + gaze series; dot positions from events kind `"dot"` (payload `{"x": float}`), resampled to frame count by nearest ts. Alignment finding appended to results (inconclusive when segment absent).
- `webapp/static/index.html`: after the acuity trials, a 10 s pursuit segment — dot sweeping sinusoidally, positions logged as `dot` events; then upload includes `alignment` segment. Keep the protocol linear and simple.
- Tests: analyzer returns 3 findings (acuity, behavioral, alignment) on the astronaut video with an alignment segment; webapp /analyze still returns report containing "alignment".

### Task 6: Module 2 benchmark (`benchmarks/bench_module2.py`)
- Sweep reflex asymmetries 0–2.0 mm (step 0.25) × 20 noise seeds on `render_eye_pair`; run reflex detection + decentration + Hirschberg; report mean abs deviation error in PD and detection rate. Assert mean abs error < 3 PD in test; write `results/module2.json` + markdown table via `python -m benchmarks.bench_module2`.

Every task: pytest first (fail), implement, pass, commit. Full suite green at the end.
