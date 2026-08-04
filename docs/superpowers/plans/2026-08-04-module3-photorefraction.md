# Vision Screening — Module 3 (Photorefraction) Implementation Plan

> Plan 3 of 3. REQUIRED SUB-SKILL: superpowers:executing-plans. Builds on Plans 1–2.

**Goal:** Eccentric-photorefraction module: synthetic crescent renderer grounded in the standard optical model, meridian-profile measurement inverting crescent width to diopters (sphere + cylinder axis), analyzer/webapp integration, benchmark, plus best-effort autonomous acquisition of real public reference images with provenance.

**Optical model (Bobier & Braddick eccentric photorefraction):** For camera distance `d` (m), flash eccentricity `e` (m) from the aperture, pupil radius `r` (m), and defocus relative to the camera plane `A = refractive_error − 1/d` (diopters): dark-crescent-free bright reflex extent from the flash-side pupil edge is `w = 2r − e / (d · |A|)`, clipped to [0, 2r]. `w ≤ 0` = dead zone (defocus too small to see). Crescent appears on flash side for myopic defocus (A < 0), opposite for hyperopic — encoded as a side sign. Astigmatism: `A(θ) = S + C·sin²(θ − axis)` per meridian; the renderer draws per-meridian widths, the measurer profiles the pupil at many angles and fits S, C, axis.

## Tasks

### Task 1: Crescent renderer (`synth/photoref.py`)
- `crescent_width_px(A_diopters, pupil_radius_px, e_m=0.02, d_m=0.5, px_per_m) -> float` — the model above, returns bright-region extent in px (0 = dead zone).
- `render_reflex(pupil_radius_px, S, C=0.0, axis_deg=0.0, e_m, d_m, px_per_m, noise_sigma=0) -> (img_gray, truth)`: dark background, pupil disk of dim red-reflex base (60), bright crescent (220): for each pixel in pupil, compute its meridian angle θ from center relative to flash direction (flash below camera → crescent vertical by default; use horizontal flash axis for simplicity: flash offset along +x). Pixel is bright iff its signed distance from the flash-side edge along the flash axis < w(θ_meridian of pixel). Simplification documented: per-pixel meridian uses the pixel's angular position; produces rotated-crescent morphology adequate for measurement development.
- Tests: w monotone in |A| beyond dead zone; dead zone → uniform pupil; larger |S| → larger bright area; C>0 with axis 45° changes measured orientation (asserted via measurer in Task 2 round-trip).

### Task 2: Photorefraction measurement (`modules/photoref.py`)
- `measure_crescent_profile(img_gray, center_px, radius_px, n_angles=36) -> list[(theta_deg, width_px)]`: ray profiles from flash-side edge through center; width = extent of pixels > 140 from the flash-side edge along that chord.
- `invert_width(w_px, pupil_radius_px, e_m, d_m, px_per_m) -> |A|` (inverse of model; None in dead zone).
- `fit_srx(profile, ...) -> (S_abs, C_abs, axis_deg)` — take |A|(θ) over meridians, min = |S|-ish, max−min = cylinder magnitude, argmax angle = axis. Sign of S from crescent side vs flash side (side detected by comparing bright-centroid x to pupil center).
- `score_photoref(per_frame_estimates: list[(S, C, axis)], valid_fraction) -> Finding`: median S/C/axis; tier by valid_fraction and inter-frame consistency (std(S) ≤ 0.75 D for measured); dead-zone-dominated → "within ±X D of screen distance focus" honest wording; always includes "screening estimate" caveat in summary.
- Tests: render→measure round trip: sphere-only |S| ∈ {1, 2, 3, 4} recovered within 0.5 D; cylinder 2 D at 45° recovers axis ±15°; dead zone honest; score tiers.

### Task 3: Analyzer + webapp integration
- Analyzer: segment `test_id="photoref"`: per gate-passing frame — pupil center = iris center, pupil radius = 0.4 × iris radius (dim-light dilation noted), px_per_m from iris scale (HVID); crop, grayscale, measure, collect estimates; e_m and d_m from segment events kind `"photoref_config"` payload (defaults e=0.02, d=distance_cm/100). Gates for this segment use a dim-range brightness check (5–90) instead of the standard range — dark room is REQUIRED here (`check_frame(..., brightness_range=(5.0, 90.0))`).
- Webapp: third protocol phase — instructions ("dim the lights, hold still"), screen flashes a white band near the camera for 5 s while recording; logs `photoref_config` event.
- Tests: analyzer emits 4 findings; photoref inconclusive (with retakes) on the bright astronaut video — honest path; synthetic-video path exercised at module level in Task 2.

### Task 4: Benchmark (`benchmarks/bench_module3.py`)
- Sweep S ∈ −4..−0.75 and +0.75..+4 (0.25 steps beyond dead zone), C ∈ {0, 1, 2}, noise σ ∈ {0, 6}; render → measure → recover. Report mean abs sphere error (assert < 0.5 D clean, < 0.75 D noisy), cylinder detection rate, axis error. Emit `results/module3.json` + table.

### Task 5: Real-world reference data, best-effort (`data/fetch_public.py`)
- Attempt downloads of truly-open items (open-access figures/datasets found via web search at run time); save under `data/real/<source>/` with `provenance.json` (url, license, label, fetched-date passed in). Failures logged, never fatal. This is a curation aid, not a training pipeline — small-N reality-check set per the spec.

Every task: pytest first, implement, pass, full suite green, commit.
