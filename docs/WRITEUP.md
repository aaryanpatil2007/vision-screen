# VisionScreen: Webcam-Based Vision Screening from a Guided Capture Protocol

**Status:** research prototype · **Date:** 2026-08-04

> This is a screening signal only, not a diagnosis. See an optometrist for a
> clinical evaluation. Nothing in this system is a medical device.

## 1. Overview

VisionScreen analyzes a guided webcam/phone recording of a person performing
three on-screen tests and produces a screening report with per-test findings,
each carrying a confidence tier (`measured` / `weak-signal` / `inconclusive`).
The architecture is a **physics spine with learned perception**: clinically
established geometric/optical relations perform the inversion to clinical
quantities, while ML components (MediaPipe FaceLandmarker) handle perception.
Capture-quality gates reject unusable frames and ask for a retake rather than
ever emitting a low-confidence number.

Modules:

1. **Acuity + behavior** — tumbling-E staircase scored from keyed responses;
   squint (eye-aspect-ratio), lean-in (interocular distance drift), and head
   tilt flags from the video.
2. **Alignment (strabismus)** — Hirschberg corneal-reflex asymmetry in prism
   diopters + smooth-pursuit conjugacy while tracking an on-screen dot.
3. **Photorefraction (refractive error)** — eccentric-photorefraction crescent
   analysis of the red reflex in a dim room, inverting crescent geometry to
   defocus in diopters with sphere / cylinder / axis decomposition.

## 2. Methods

### 2.1 Perception

MediaPipe FaceLandmarker (Tasks API, 478 landmarks incl. iris) supplies eye
corners and iris rings. Empirically pinned index mapping (regression-tested):
iris center 468 pairs with corners 33/133; 473 with 362/263. Pixel→mm scale
uses the population-stable horizontal visible iris diameter (HVID ≈ 11.7 mm).
The corneal specular reflex is segmented as the brightest connected component
*restricted to the iris disk* — unmasked detection provably fails (bright
sclera pollutes the centroid; caught by our benchmark, §3.2).

### 2.2 Acuity

Letter size for logMAR L at distance d: the optotype subtends `5·10^L` arcmin,
so height = `2d·tan(2.5·10^L arcmin)`. A 1-down/2-up staircase (start 1.0,
floor −0.3, ceiling 1.3) terminates after 6 reversals — but never before 12
trials, a guard that halved benchmark error by preventing early-lapse reversal
clusters from ending the test near its starting level. Threshold = smallest
level with majority-correct performance over ≥ 2 trials.

### 2.3 Alignment

Hirschberg: reflex decentration (mm, via HVID scale) asymmetry between eyes
converts at ≈ 18 prism diopters per mm (literature range 15–22 PD/mm).
Symmetric decentration is deliberately not flagged (shared offset = camera
geometry, not strabismus). Flags: asymmetry ≥ 1.0 mm (≈ 18 PD); borderline
note ≥ 0.5 mm. Pursuit conjugacy = inter-eye Pearson correlation of
normalized iris positions while tracking a sinusoidal dot; flag < 0.8.

### 2.4 Photorefraction

Eccentric-photorefraction model (Bobier & Braddick, 1985): for defocus
`A = refractive error − 1/d` (relative to the camera plane), flash
eccentricity `e`, camera distance `d`, pupil radius `r`, the bright-reflex
extent from the flash-side pupil edge is `w = 2r − e/(d·|A|)`, clipped to
[0, 2r]; `w = 0` defines the dead zone `|A| < e/(2rd)` (≈ 1.25 D at e = 5 mm,
d = 0.5 m, 2r = 8 mm). Crescent side encodes myopic vs hyperopic sign.
Meridional profiling at 5° steps within ±60° of the flash axis, with an exact
chord-depth-to-width correction, feeds a linearized fit of
`|A|(θ) = S + C·sin²(θ − axis)` (cos 2θ/sin 2θ least squares). For myopic
eyes the fitted parameters are remapped (`S = −(S_fit + C_fit)`, axis − 90°)
— an algebraic identity of the absolute-value profile. Per-frame estimates
are aggregated by median; the tier requires inter-frame σ(S) ≤ 0.75 D.

### 2.5 Quality gates and honesty rules

Per-frame gates: face present, interocular ≥ 60 px, brightness in [30, 225]
(photorefraction segment instead **requires** dim: [5, 90]). Modules never
emit numbers below their evidence tier: metrics are suppressed for
inconclusive findings and every report carries the screening disclaimer.

## 3. Results (synthetic benchmarks, fully reproducible)

Run: `python -m benchmarks.bench_module1` (2, 3). JSON in `results/`.

### 3.1 Module 1 — acuity recovery (50 simulated observers, 5% lapse rate)

| metric | value |
|---|---|
| mean abs error | **0.063 logMAR** |
| max abs error | 0.20 logMAR |

Under one chart line of error on average; the max improved 0.877 → 0.20 after
the minimum-trials staircase guard.

### 3.2 Module 2 — Hirschberg deviation (0–2 mm asymmetry × noise seeds)

| metric | value |
|---|---|
| detection rate | 1.00 |
| mean abs error | **0.60 PD** |
| max abs error | 1.18 PD |

Clinically meaningful strabismus is ≳ 10 PD; measurement error is an order of
magnitude below the signal. This benchmark caught a real bug during
development: unmasked reflex detection read 14.6 PD mean error (sclera
pollution); iris-disk masking brought it to 0.60 PD.

### 3.3 Module 3 — photorefraction (S ∈ ±[1.5, 4] D, C ∈ {0, 1, 2} D)

| condition | detection | mean abs SE error | max | mean axis error |
|---|---|---|---|---|
| clean | 1.00 | **0.063 D** | 0.64 D | 32° |
| noise σ=25 | 1.00 | 0.064 D | 0.67 D | 33° |

Spherical-equivalent recovery is robust to heavy sensor noise. Axis error is
large — expected: a single flash axis observes meridians only within ±60°,
so near-perpendicular axes are weakly constrained (§4).

## 4. Limitations (read before trusting anything)

1. **No real-eye validation.** All accuracy numbers are on synthetic data
   generated from the same optical model the measurement inverts — they
   validate the *pipeline*, not real-world performance. The renderer-measurer
   round trip is not independent evidence. First real test: point the demo at
   a person with a known prescription.
2. **Dead zone.** Refractive errors within ≈ ±1.25 D of the screen distance
   are invisible to this geometry — reported honestly as such.
3. **Axis observability.** One flash axis → weak astigmatism-axis constraint
   (~33° error). Fix: second capture with a rotated flash band.
4. **Webcam reality.** Laptop screens are weak flashes; consumer sensors may
   not resolve the red reflex at all in many rooms. The dim-room gate and
   inconclusive tier handle this honestly, but yield will vary.
5. **Hirschberg κ-angle.** Individual angle-kappa differences add a few PD of
   inter-eye baseline; the symmetric-offset rule absorbs the shared part only.
6. **Population constants.** HVID (11.7 mm) and pupil/iris ratio (0.35) are
   population means; per-subject deviation propagates ~5–10% scale error.

## 5. Real-world reference data

`python -m visionscreen.data.fetch_public` downloads openly licensed clinical
example images (strabismus, esotropia/exotropia, red reflex, leukocoria,
Hirschberg test) from Wikimedia Commons with full provenance
(`data/real/commons/provenance.json`: title, URL, license, artist, search
term). Current snapshot: 11 images. These are a qualitative reality-check
set, not training data.

## 6. Reproducing

```
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest            # 69 tests
.venv/bin/python -m benchmarks.bench_module1
.venv/bin/python -m benchmarks.bench_module2
.venv/bin/python -m benchmarks.bench_module3
.venv/bin/uvicorn webapp.app:app     # live demo at localhost:8000
```

## 7. References

- Bobier, W.R. & Braddick, O.J. (1985). Eccentric photorefraction: optical
  analysis and empirical measures. *Am J Optom Physiol Opt* 62(9).
- Wheeler, M. (1943 tradition) / modern reviews of the Hirschberg test:
  ~15–22 PD per mm of reflex decentration.
- Bailey, I.L. & Lovie, J.E. (1976). New design principles for visual acuity
  letter charts (logMAR). *Am J Optom Physiol Opt* 53(11).
- MediaPipe FaceLandmarker (Google, Tasks API) — 478-landmark face mesh.
