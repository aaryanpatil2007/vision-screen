# What this system can and cannot do, with the numbers

Every claim here is either measured in this repository or cited to a published
source. Where a number is an estimate with nothing behind it, it says so. The
document is organised by how much confidence each capability deserves, which is
not the same order as how impressive each one sounds.

Reproduce any measured figure with the command listed beside it.

---

## 1. Segmentation: visible-light periorbital anatomy

**Benchmark.** Nahass et al., *Ophthalmology Science* 2025
(DOI 10.1016/j.xops.2025.100757), periorbital segmentation dataset
(Zenodo 10.5281/zenodo.13916845, CC BY 4.0): 2,842 annotated face photographs
from the Chicago Face Database (studio portraits) and CelebAMask-HQ
(in-the-wild celebrity photographs), five structures plus background, 80/20
split. Reference model is a DeepLabV3-ResNet101, roughly 60 M parameters.

`python -m benchmarks.bench_periorbital --subset combined --epochs 80`

| class | ours (Dice) | DeepLabV3-ResNet101 | delta |
|---|---|---|---|
| iris | **0.940** | 0.93 | +0.010 |
| sclera | **0.846** | 0.81 | +0.036 |
| lid | **0.867** | 0.79 | +0.077 |
| caruncle | **0.755** | 0.65 | +0.105 |
| brow | 0.877 | — | not compared (combined-set reference omits it) |

Mean Dice across the five structures **0.877**; mIoU 0.788. 2,273 train / 569
validation, 80 epochs, best at epoch 56. Horizontal-flip test-time augmentation
was evaluated and *not* used — it scored 0.8569 against the plain model's
0.8570, so the reported figure is the single forward pass.

### How the comparison was made — and what it is not

**This is a same-dataset, same-protocol comparison, not a shared-test-set one.**
The reference paper's exact train/validation split is not published, so the 80/20
split here is a different draw from the same 2,842 images. Split variance on a
set this size is worth roughly a percentage point per class, which means:

* lid (+0.077), caruncle (+0.105) and sclera (+0.035) are comfortably outside
  that band and are real;
* **iris (+0.010) is inside it and should be read as a tie**, not a win.

There is no held-out leaderboard for this dataset, so no stronger form of the
comparison is currently available. What is directly comparable, and not
subject to split variance at all, is the parameter count.

### Why it wins

Four choices, in descending order of how much they contributed:

* **Average pooling instead of max pooling.** The boundaries that matter here —
  limbus, lid margin — are soft intensity ramps rather than edges. Max-pooling
  keeps the brightest pixel in each window and discards the gradient that
  localises a soft boundary. This is why the gains concentrate on sclera and
  lid, the two classes whose borders are shadows.
* **Generalized Dice loss with inverse-square class-frequency weighting.**
  Caruncle occupies about 0.3% of pixels; under an unweighted loss a model can
  ignore it entirely and still score well. That class shows the largest margin
  (+0.105).
* **Boundary-weighted cross-entropy.** Nearly all residual error sits within a
  few pixels of each edge, and a uniform loss spends its gradient on easy
  interiors instead.
* **Dense blocks.** Each layer sees the concatenation of all previous features,
  so it needs far fewer channels of its own. This is where the ~200x parameter
  reduction comes from, not from a smaller or shallower network.

Parameters: **284,214** against roughly 60,000,000 — **0.47%** of the reference
model. The architecture is a dense encoder–decoder with average pooling instead
of max pooling, because the boundaries that matter here (limbus, lid margin)
are soft intensity ramps and max-pooling discards the gradient that localises
them.

Sclera and caruncle are the two worth noting. Sclera is the hardest class in
every eye-segmentation benchmark in the literature — 0.674 IoU for the OpenEDS
2020 baseline, 0.074 for zero-shot SAM 2 — because its boundary with the lid is
a shadow rather than an edge. Caruncle occupies about 0.3% of pixels, so a
handful of misassigned pixels moves its score a long way.

### Why not OpenEDS/RITnet

This project originally targeted RITnet (95.3% mIoU, OpenEDS 2019) and reached
**94.33% mIoU at 205,720 parameters** — 83% of RITnet's 248,900 — before that
line of work was abandoned as invalid. Three reasons:

1. **Wrong modality.** OpenEDS is 640×400 near-infrared imagery from cameras
   mounted a couple of centimetres from the cornea inside a VR headset, cropped
   to a single eye. This system takes visible-light frames of a whole face from
   a webcam at roughly half a metre. Different spectrum, optics, framing and
   pixel statistics.
2. **The benchmark is closed.** The challenge test labels were never released
   and the evaluation server is gone, which is why no paper since about 2020
   reports the figure at all.
3. **Two widely-repeated numbers on it are wrong.** OpenEDS's frequently-cited
   "98.3% mIoU" is its pixel accuracy; its actual mIoU is 91.4. EyeNet's
   "0.974" is the composite challenge score `(mIoU + min(1/S,1))/2`, not mIoU —
   its mIoU is 94.9.

The 94.33% figure is retained only as an architecture and efficiency datapoint.
It is not a claim about this system's performance on webcam images.

---

## 2. Anaemia from conjunctival pallor

The strongest genuinely-clinical result here, because it is the one task with
real labelled data: laboratory haemoglobin, not clinical impression.

**Dataset.** CP-AnemiC (Mendeley 10.17632/m53vz6b7fx, CC BY 4.0): 710
conjunctival photographs of children aged 6–60 months across ten hospitals in
four regions of Ghana, each with a lab haemoglobin value. A darkly-pigmented
cohort — the population where published screening tools degrade most and are
validated least.

`python -m benchmarks.bench_anemia`

| metric | value |
|---|---|
| AUC (leave-one-hospital-out, nested selection) | **0.855** (95% CI 0.827–0.882) |
| sensitivity at the reference study's specificity | **0.705** |
| specificity | 0.832 |
| haemoglobin MAE | 1.56 g/dL (r = 0.47) |
| worst held-out hospital | AUC 0.790 (range 0.790–1.000) |

**Reference:** Collings et al. 2016, *PLoS One* 11:e0153286 — conjunctival
pallor from digital photographs, **57% sensitivity / 83% specificity** on
validation (93%/78% in training). Compared at matched specificity, this system
is **13.5 points more sensitive**.

### How the comparison was made — and what it is not

**This is a cross-study comparison, not head-to-head.** Collings et al. used
their own dataset, which is not public; the numbers here are measured on
CP-AnemiC. Different populations, different cameras, different prevalence, and
different disease severity distributions. Comparing across datasets is how this
literature routinely compares itself, and it is still weaker than a shared test
set. Treat the 13.5-point margin as indicative, not as a measured difference
between two methods on the same data.

What makes it worth stating at all is that the protocol here is **stricter than
theirs in three specific ways**, so the comparison is not tilted in this
system's favour:

* they validated on a random split; this is leave-one-hospital-out, where every
  test image comes from a site the model has never seen;
* the model family is chosen by an inner loop inside each training fold, not by
  its held-out score;
* the operating point is theirs, not the one that flatters this system — at the
  Youden point the numbers would read 0.783/0.776 instead.

### Why it works

The physics, not the model. The primary feature is the erythema index
log10(R/G): haemoglobin absorbs strongly in green and weakly in red, so the
red-to-green ratio tracks blood content directly. Using a *ratio* rather than
absolute brightness divides out exposure and illuminant intensity, which is what
lets one model work across ten hospitals' cameras. With 710 images across 10
sites, a fine-tuned CNN would have the capacity to memorise sites instead.

Three deliberate choices make that comparison honest:

* **Leave-one-hospital-out.** Site prevalence ranges from 48% to 88%, and each
  hospital means a different camera, lighting and operator. A random split lets
  a model recognise the site, and site predicts the label.
* **Nested model selection.** Choosing the better of two model families by
  their held-out score is test-set selection. The choice is made by an inner
  leave-one-group-out loop inside each training fold.
* **Matched operating point.** Comparing at whatever threshold each side
  happened to use compares operating points, not tests.

### A negative result worth recording

The expectation was that a random split would inflate the score substantially.
It did not: the gap was **0.001 AUC**. Hand-built colour features do not encode
site identity the way learned features do, so the Collings training-to-
validation collapse appears to be a property of learned representations rather
than of the task. The random-split figure is reported alongside the grouped one
precisely so this can be checked rather than assumed.

Features are physical, not learned: the workhorse is the erythema index
log₁₀(R/G), because haemoglobin absorbs strongly in green and weakly in red, so
the ratio tracks blood content while dividing out exposure. With 710 images
across 10 sites, a fine-tuned CNN would memorise sites.

---

## 3. Refractive error: a ballpark, and why it cannot be better

The forward model is published, not invented: Blendowske 2015,
*Optom Vis Sci* 92(6):e121–e125, fitted across seven pooled datasets at
R²(adj) = 0.99.

```
logMAR_unaided = logMAR_best_corrected + log10(1 + b²)
b² = M² + (C/2)²                      (Thibos blur strength)
```

The paper's own worked example — 1 D of error costs 3 lines (0.30 logMAR), not
the "4 lines per dioptre" of folklore — is pinned as a test, so the citation
cannot drift from the code.

### The error budget

| source | magnitude | provenance |
|---|---|---|
| acuity test–retest, clinic conditions | ±0.12–0.15 logMAR | Arditi & Cagenello 1993; Siderov & Tiu 1999 |
| → propagated into blur strength | ≈ ±0.30–0.43 D | derived |
| photorefraction calibration variability | **±40% of the reading** | Bharadwaj et al. 2013, *JOSA A* 30:923 |
| pigmentation bias in that calibration | up to **+64 ± 11%** | Sravani et al. 2015, *Sci Rep* 5:7976 |
| non-cycloplegic accommodation (photoscreeners) | −0.78 D, 95% PI −1.70 to +0.10 | Roque et al. 2026 meta-analysis |
| pupil size, unmodelled | slope varies up to ~3.5× (1→5 mm) | Kamiya et al. 2012 |
| sign of the error from acuity | **unrecoverable** — b² is even in M | Blendowske 2015 |
| subjective refraction's own repeatability | ±0.78 D between clinicians | Bullimore 1998; MacKenzie 2008 |
| best published photoscreener agreement | ±1.3 to 1.5 D | Mirzajani 2013; Kanclerz 2024 |

Three limits are enforced in code rather than described in prose:

1. **The interval is floored at 1.5 D.** No purpose-built photoscreener does
   better than that against cycloplegic refraction, and this has none of their
   hardware — no flash, no infrared, no controlled eccentricity, no fixed
   working distance. A narrower interval would claim an accuracy the equipment
   cannot deliver.
2. **Beyond ~4 D no value is reported.** The pupil luminance gradient is linear
   in refractive state only to about 4 D at a 5 mm pupil, then saturates and
   *reverses* (Wu, Thibos & Candy 2018), so a high myope can produce the same
   gradient as a moderate one.
3. **The direction is never named without a signed measurement.** Acuity
   measures how blurred vision is, not which way. Without photorefraction the
   report says "a focusing error, direction not determined".

### The hyperopia blind spot

Acuity screening detects myopia at 92% sensitivity and hyperopia at **41–54%**
(O'Donoghue et al. 2012, *PLoS ONE* 7:e34441, n = 1,053). Between 11% and 15%
of children with clinically significant hyperopia or astigmatism read 0.20
logMAR or better. A young hyperope focuses through the error, so the test
cannot see it. The engine models this by subtracting available accommodation
(Hofstetter minimum) from positive defocus — which reproduces the blindness
rather than hiding it.

### What is not claimed

There is **no published continuous-dioptre regression from a bare external eye
photograph**. The honest ceiling for imaging with no added optics is binary
myopia classification at AUC ~0.91–0.93 (Yang et al. 2020). Fundus-based
regression reaches MAE 0.56 D (Varadarajan et al. 2018, *IOVS* 59:2861) and
requires a fundus camera; its cylinder head is worthless (R² = 0.05).

---

## 4. The differential engine

Diagnostic likelihood ratios over age-dependent prevalence:
`post-odds = pre-odds × ΠLR`. Chosen over a red-flag tally because a tally can
only accumulate — it has no way to rule anything out — and because prevalence
must stay in the arithmetic: the same evidence means different things at 25 and
at 75.

18 conditions catalogued. Likelihood ratios are marked `LIT` where derived from
published sensitivity/specificity and `EST` where structural. The `EST` values
are the first thing to replace when labelled data arrives.

Published ratios in use:

| finding | condition | sens/spec | source |
|---|---|---|---|
| acuity ≥ 0.20 logMAR | myopia | 0.92 / 0.91 | O'Donoghue 2012 |
| acuity ≥ 0.20 logMAR | hyperopia | 0.47 / 0.865 | O'Donoghue 2012 |
| meridian preference | astigmatism | 0.62 / 0.85 | O'Donoghue 2012 |
| Hirschberg > 8Δ | strabismus | referral threshold | AAPOS 2021 |
| anisocoria ≥ 1.0 mm | anisocoria | ≥0.4 mm common in normals | Lam 1987 |
| MRD1 < 2 mm | ptosis | surgical threshold | standard |

Combined evidence is capped at ×60 in either direction. Findings within one
battery are correlated — reduced acuity and reduced contrast sensitivity share
an underlying cause — and unbounded multiplication would treat three views of
one signal as three confirmations.

---

## 5. What this cannot do

Listed because a screening tool's failure modes matter more than its successes,
and because the most dangerous output is false reassurance.

**Cataract and media opacity — not feasible.** A Brückner/red-reflex test needs
illumination close to coaxial with the lens and bright enough to light the
fundus. A laptop has no flash; the screen is broad, dim, offset from the camera,
and filtered by the webcam's own infrared-cut filter, which rejects exactly the
wavelengths that return most strongly. CRADLE, a purpose-built leukocoria app
using a real camera flash, reported 90% per-child sensitivity from its
developers and **15.4%** in independent prospective validation (Vagge 2019).
LOCS III cataract grading is *defined* on slit-lamp and retro-illumination
views. This module is capped below any actionable tier and states that a clear
result means nothing was visible, not that nothing is there.

**Diabetic retinopathy, glaucoma staging, retinal disease — not feasible.**
These need a view of the retina or a pressure measurement. Note the trap:
Babenko et al. 2022 appears to detect diabetic retinopathy from "external eye
photos", but its images came from table-top fundus cameras in external mode
with a chin rest and circular flash, and the authors state they have "limited
data with which to understand if smartphone or webcam images are sufficient".

**Absolute colour claims — not without a reference.** Webcam auto-white-balance
will turn a jaundiced sclera neutral or a neutral sclera yellow. The system asks
for a sheet of white paper in frame; without it, colour findings are computed
but capped. The scleral yellowness and redness thresholds have **no labelled
corpus behind them** and are held below the actionable tier by an explicit flag
(`SCLERA_COLOUR_CALIBRATED = False`) with a test enforcing it.

**Anything measured at an unknown distance.** Screen size, viewing distance and
pixel density are all self-reported. Purpose-built app studies show why this
matters: of 11 iPhone Snellen apps, optotype size accuracy ranged 4.4–39.9%,
and the best app's limits of agreement against 6-metre Snellen were ±0.35
logMAR — about 3.5 lines (Perera et al. 2015, *Eye* 29:888).

**Base rates.** In a low-prevalence population, false positives dominate even a
good test. GoCheck Kids — FDA-cleared, purpose-built — achieved a positive
predictive value of 50% in primary care, dropping to 26% in infants aged 3–12
months (Law et al. 2020). Roughly one referral in two was unnecessary.

---

## 6. Reproducing everything

```bash
python -m pytest -q                            # 423 tests
python -m benchmarks.bench_periorbital         # segmentation vs DeepLabV3
python -m benchmarks.bench_anemia              # anaemia, grouped protocol
python -m benchmarks.bench_openeds --resume    # archived, wrong modality
```

Datasets download themselves where licensing permits; CP-AnemiC and the
periorbital set are both open and fetched automatically by their benchmark
scripts.
