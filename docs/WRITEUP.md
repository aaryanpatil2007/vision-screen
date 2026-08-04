# VisionScreen: A Webcam Vision-Screening Battery with Physics-Constrained Learned Perception

**Research prototype · 2026-08-04**

> **This is a screening tool, not a diagnosis, and it does not replace an eye
> exam.** It cannot measure intraocular pressure, examine the retina, or rule
> out disease. All accuracy figures below are from synthetic and
> weakly-labeled real benchmarks; the system has *not* been validated against
> clinical measurement in human subjects.

---

## 1. Summary

VisionScreen administers a thirteen-part guided battery through a browser and
webcam, then analyses the recording server-side to produce a screening report
with per-test confidence tiers. It combines three layers:

1. **Psychophysics** — clinically specified stimuli (logMAR optotypes,
   Pelli-Robson contrast triplets, astigmatic dial, Amsler grid,
   pseudoisochromatic plates) rendered at calibrated angular size.
2. **Physics** — closed-form clinical inversions: the Hirschberg ratio for
   ocular alignment, and the eccentric-photorefraction relation for refractive
   error.
3. **Learned perception** — a small U-Net that segments iris, pupil and
   corneal reflex, trained on domain-randomized synthetic eyes *plus* weakly
   labeled real webcam eyes.

The central empirical result is that layer 3 is not optional: **a network
trained only on synthetic eyes scores 0.927 mIoU on synthetic data but 0.241
on real eyes.** Adding weakly-labeled real data closes most of that gap
(0.705 real mIoU) with no loss on synthetic.

---

## 2. Test battery

| # | Test | Quantity produced | Basis |
|---|---|---|---|
| 1-3 | Visual acuity — binocular, right, left | logMAR + Snellen | Bailey-Lovie/ETDRS logMAR progression; tumbling-E, 1-down/2-up staircase |
| 4 | Contrast sensitivity | log CS | Pelli-Robson triplets, 0.15 log steps, 2-of-3 rule |
| 5 | Astigmatism | minus-cylinder axis (deg) | Astigmatic fan / clock dial, "rule of 30" |
| 6 | Color vision | red-green flag, protan/deutan lean | Pseudoisochromatic plates along confusion lines |
| 7 | Central field | metamorphopsia / scotoma marks | Amsler grid, 20° subtense, 1° squares |
| 8 | Stereoacuity | threshold in arcsec | Dynamic random-dot stereogram, 4AFC, catch trials |
| 9 | Ocular motility | pursuit gain, saccade rate | Smooth pursuit + H-pattern saccades |
| 10 | Alignment | deviation in prism diopters | Hirschberg corneal reflex asymmetry |
| 11 | Pupil response | constriction %, anisocoria | Binocular light reflex |
| 12 | Refractive error | sphere / cylinder / axis (D) | Eccentric photorefraction |
| 13 | Binocular fusion | fusion / suppression / diplopia | Worth four-dot, near and distance |
| — | Viewing distance | cm, drift, acuity bias | Pinhole model on iris diameter |

Behavioural signals (squinting, lean-in, head tilt) are extracted from the
video throughout and reported alongside.

**Not possible without hardware, and stated as such in the product:**
intraocular pressure, slit-lamp examination, dilated fundus examination,
objective retinoscopy, formal perimetry, OCT. A normal screening result here
does not exclude glaucoma, diabetic retinopathy, or macular disease.

**Three tests were deliberately *not* built, because the physics forbids them
on a bare screen.** Documenting why matters more than shipping a plausible
imitation:

* **RAPD / swinging flashlight.** The pupil light reflex is fully consensual —
  a screen flash reaches both retinas, so both pupils respond to the *summed*
  afferent input. Inter-eye response asymmetry under a bilateral stimulus
  therefore reflects efferent or iris differences, not an afferent defect.
  Anaglyph "monocular" stimulation does not rescue it: red carries 21% of white
  luminance and cyan 79%, a 0.57 log-unit intrinsic imbalance that exceeds most
  clinically meaningful RAPDs (Bell grade I = 0.4 LU). An earlier version of
  this system did claim RAPD detection; that claim was removed.
* **Cover test for tropia.** A screen cannot occlude an eye. Anaglyph hides the
  *target* from one eye while that eye still sees the room, which dissociates a
  phoria rather than performing a cover test. Reported automated cover-test
  agreement bears this out: a 250 Hz tracker with a *physical* occluder reaches
  ±1.95Δ, while simulated occlusion in VR degrades to ±17–28Δ with essentially
  no correlation for exotropia.
* **Brückner reflex.** Requires illumination coaxial with the observation axis;
  a screen is an extended source displaced several centimetres from the lens,
  so for an undilated 3–5 mm pupil at 50 cm the light misses the pupillary
  aperture entirely. Consumer webcams also carry IR-cut filters and auto-exposure
  that destroy reflex-brightness comparison.

---

## 3. Methods

### 3.0 What the battery cannot see, quantified

Screening tools are usually honest in the abstract ("not a substitute for an
eye exam") and vague about magnitude. The magnitude is knowable. In a
self-declared-healthy cohort with a median age of 70, a full examination found
**25% needed referral and a further 9% needed monitoring — 34% with findings
that no screen-and-webcam test can detect.** In a refractive-complaint clinic
population, 26.1% had at least one asymptomatic condition (13.7% retinal, 4.9%
glaucoma or ocular hypertension). Roughly half of glaucoma in white populations
and three quarters in Latino populations is undiagnosed at any given time.

That number is stated in the product, not just in this document.

The blind spot is structural, not a matter of effort:

* **Intraocular pressure** has no external optical signature; an eye at 10 and
  at 30 mmHg look identical. Published "image-based IOP" work produces binary
  classifiers on small samples from a physiologically unaccepted premise
  (scleral redness), never millimetres of mercury.
* **The angle** is invisible in principle: light from the iridocorneal angle
  exceeds the critical angle at the cornea-air interface (~46°) and totally
  internally reflects. A goniolens removes that interface. No image processing
  can recover photons that never leave the eye.
* **The retina** needs a condensing lens (+20/+28 D) or a fundus camera's
  annular illumination; a front-facing webcam's light source is coaxial with
  its sensor, which produces corneal glare rather than a retinal image.
* **Cycloplegic refraction** requires a drug.

USPSTF context matters here too: screening asymptomatic adults ≥65 for impaired
acuity carries a **Grade I (insufficient evidence)** statement, with four RCTs
showing no difference in visual or functional outcomes between screening and no
screening. This system therefore makes no claim to improve outcomes; it
measures visual function and reports numbers.

### 3.1 Angular calibration

Every psychophysical stimulus needs true angular size, which requires the
display's pixel pitch. The app uses an ISO/IEC 7810 ID-1 card (85.60 mm wide)
as a physical ruler: the user scales an on-screen rectangle to match a real
card, yielding px/cm directly. Viewing distance is entered explicitly. A
logMAR-*L* optotype then subtends 5·10^L arcmin:

    height_cm = 2 · d · tan(2.5·10^L arcmin)

The browser and server implement this identically; a test asserts bit-level
agreement between the two so client rendering can never silently drift from
server scoring.

**Distance is then measured rather than trusted.** The stated distance
calibrates a pinhole focal length once, after which distance is recovered per
frame from the **iris diameter**, not the interocular distance. Two reasons:
the horizontal visible iris diameter has a population CV of 3.6% (11.71 ±
0.42 mm) against ~5.5% for interpupillary distance, and — more importantly —
it is geometrically robust to head pose. A yaw of θ foreshortens the
interocular segment by cos θ (20° → 6% distance error → 0.025 logMAR) while
the projected major axis of a circle is unchanged. Since logMAR is a log-scale
measure, a distance error of factor *k* biases acuity by exactly log₁₀(*k*);
each acuity segment is corrected by the distance actually held during it.

**Display luminance is attested, not measured.** ISO 8596 specifies 80–320
cd/m² and BS 4274-1 wants ≥120; measured acuity falls roughly 0.2 logMAR per
decade of luminance below that, so a user at minimum laptop brightness
(~3.5 cd/m²) measures about 1.3 lines worse than their true acuity — enough to
manufacture a referral. No browser API reports luminance, so the app gates the
run behind an explicit brightness attestation plus a 16-step grey ramp that
makes a crushed dark end visible. This is the same compromise the validated
browser-based perimeter (MRF-web) makes.

### 3.2 Perception

MediaPipe FaceLandmarker (Tasks API, 478 landmarks including iris ring)
provides face geometry and a validated iris circle. Pixel→mm scale comes from
the horizontal visible iris diameter (HVID ≈ 11.7 mm), which is stable across
adults to within a few percent.

**EyeSegNet** is a 3-level U-Net (<600k parameters by design, so it runs
per-frame on CPU inside the analyzer loop) predicting five classes:
background, ocular surface, iris, pupil, corneal reflex.

### 3.3 Training data

*Synthetic (domain randomized).* A generator renders eye crops with randomized
iris/pupil geometry, iris texture, skin and sclera tone, eyelid aperture,
lashes, gaze offset, illumination gradient, exposure, blur, sensor noise, and —
critically — **distractor speculars placed off-cornea**, the exact confounder
that breaks threshold-based reflex detection.

*Real (weakly labeled).* Real webcam-style face images were downloaded from
openly redistributable Hugging Face datasets (GazeCapture-derived frames;
close-up pupil-position crops), plus openly licensed clinical images from
Wikimedia Commons, all with per-item provenance recorded. These carry gaze
targets but no segmentation masks, so labels are derived from two priors:

* **Geometric** — MediaPipe's iris circle defines the iris disk.
* **Photometric** — *within that disk*, the pupil is the dark mode (Otsu) and
  the corneal reflex the bright mode.

Crops are **rejected** rather than mislabeled when the priors disagree: the
pupil must be concentric with the iris, occupy 2–75% of the disk, and be at
least 12 grey levels darker than the surrounding iris annulus; blown-out crops
(>35% saturated) are dropped. That last rule was added after inspection showed
spectacle glare being labeled as pupil. Acceptance rate: **54.3%** (6,130
labeled crops from 11,000+ candidates across 5,980 source images) — the
rejected fraction is reported, not hidden.

### 3.4 Clinical inversions

**Hirschberg.** Reflex decentration relative to iris centre converts to
deviation at ≈18 prism diopters per mm (literature range 15–22). Only the
*asymmetry between eyes* is flagged; a shared offset is camera geometry and
angle kappa, not strabismus.

**Eccentric photorefraction** (Bobier & Braddick). With defocus `A` relative to
the camera plane, flash eccentricity `e`, distance `d` and pupil radius `r`,
the bright crescent extends `w = 2r − e/(d·|A|)` from the flash-side pupil
edge, giving a dead zone `|A| < e/(2rd)` ≈ 1.25 D for this geometry. Meridional
profiling at 5° steps feeds a linearized cos2θ/sin2θ fit of
`|A|(θ) = S + C·sin²(θ − axis)`.

---

## 4. Results

All benchmarks are reproducible: `python -m benchmarks.<name>`; JSON in `results/`.

### 4.1 Sim-to-real transfer (the headline result)

| training data | synthetic mIoU | **real mIoU** | real pupil IoU |
|---|---|---|---|
| synthetic only | 0.928 | **0.237** | 0.382 |
| synthetic + weakly-labeled real | 0.926 | **0.722** | 0.848 |

Sim-to-real gap for synthetic-only training: **0.691 mIoU**. Adding real data
recovers **+0.485**. Real corpus: 4,598 train / 1,532 held-out test, split
before training (a leakage bug was caught and fixed before any number was
published).

Real-data scaling is monotonic, which is the usual sign that the gain is real
rather than a lucky split:

| labeled real crops | held-out real crops | real mIoU |
|---|---|---|
| 1,154 | 288 | 0.691 |
| 2,930 | 732 | 0.705 |
| 6,130 | 1,532 | **0.722** |

**Interpretation.** A model validated only on its own simulator would have
reported 0.92 and been wrong about real eyes by a factor of four. Any
photorefraction or alignment system reporting only synthetic accuracy should
be read with this in mind.

### 4.2 Battery performance on 120 simulated patients

Virtual patients with ground-truth conditions and realistic lapse rates
(2–10%), scored by the production modules:

| test | metric | result |
|---|---|---|
| Visual acuity | mean absolute error | **0.049 logMAR** |
| Visual acuity | within chart test-retest repeatability (0.15 logMAR) | **92.5%** |
| Contrast sensitivity | mean absolute error, in range | **0.073 log CS** (½ triplet step) |
| Contrast sensitivity | subjects above the 8-bit display ceiling | 14/120 (reported, not scored) |
| Strabismus (≥10 PD) | sensitivity / specificity | **1.00 / 1.00** |
| Color deficiency | sensitivity / specificity | **1.00 / 0.96** |
| Anisocoria (≥1 mm) | sensitivity / specificity | **1.00 / 1.00** |
| Astigmatism axis | mean absolute error | **4.1°** |
| Photorefraction | mean absolute spherical-equivalent error | **0.027 D** |

Clinical chart acuity has a published test-retest repeatability of roughly
0.10–0.20 logMAR, so an 0.077 logMAR algorithmic error sits at the noise floor
of the reference test itself. **This bounds the algorithm, not the system:**
real-world error is dominated by user calibration and viewing distance, which
these simulations do not model.

### 4.3 Component benchmarks

| component | metric | result |
|---|---|---|
| Hirschberg deviation | mean abs error | 0.60 PD (detection 100%) |
| Photorefraction, clean | mean abs SE error | 0.063 D |
| Photorefraction, σ=25 noise | mean abs SE error | 0.064 D |
| Acuity staircase | mean abs error, 50 observers | 0.063 logMAR |

### 4.4 Bugs the benchmarks caught

Each of these was found by measurement, not inspection:

1. **Unmasked reflex detection** — bright sclera polluted the centroid;
   Hirschberg error 14.6 PD → 0.60 PD after restricting the search to the iris disk.
2. **Staircase early termination** — early lapses clustered reversals near the
   start level; max acuity error 0.877 → 0.20 logMAR after a minimum-trials guard.
3. **Alignment threshold too high** — flagged at 18 PD when clinical
   significance begins at 10 PD; sensitivity 0.73 → 1.00, specificity unchanged.
4. **Contrast single-letter levels** — one lapse truncated the whole estimate;
   proper Pelli-Robson triplets cut error 0.446 → 0.168 log CS.
5. **Non-isochromatic color plates** — figure and background differed ~12 luma
   units, so the numeral was readable as brightness; now luminance-matched.
6. **Acuity floor reported as a point estimate** — a floor-clamped result now
   reads "at or better than".
7. **Data leakage in the sim-to-real benchmark** — real training initially
   included the held-out split; caught before any number was published.
8. **A wrong hypothesis about learned vs classical reflex detection.** The
   pipeline originally trusted the network's reflex output and dropped the
   frame when it found none. Measured on 150 real reflex-bearing crops, the
   classical threshold detector actually has the higher recall — **~99% versus
   ~77%** — because a specular highlight is a strong, simple photometric
   signal and the network trades recall for precision. Trusting the network
   alone silently discarded about a fifth of the alignment data. The analyzer
   now uses the network first and falls back to classical detection, and a
   test asserts the combination beats either path alone (>90% combined recall).
   The network remains the only source of a pupil boundary, which the classical
   path cannot produce at all.
9. **An unsupportable RAPD claim** — the pupillometry module originally
   reported inter-eye response asymmetry as a possible relative afferent
   pupillary defect. A bilateral screen flash cannot reveal an afferent
   asymmetry at all (§2). The claim was removed and replaced with what is
   measurable: binocular PLR plus static anisocoria at a ≥1 mm threshold,
   since ≥0.4 mm occurs in 41% of normal subjects at some sitting and 19% at
   any given exam (Lam, Thompson & Corbett 1987).
10. **The acuity staircase targeted the wrong criterion.** A Kaernbach
    weighted up-down converges where p = S_up/(S_up + S_down); the common
    0.1/0.2 pair lands at 0.667 raw, which for a four-alternative task is
    **55.6% guessing-corrected** — a stricter criterion than threshold, so
    acuity read optimistically. The ratio is now 5/3 (0.1/0.1667), which lands
    at exactly 50% corrected. Fixing it exposed a second bug: the scorer
    grouped trials by identical level, which silently assumed the steps land on
    a repeating grid. With a 5/3 ratio they do not, and the estimate collapsed
    (error 0.065 → 0.193 logMAR). Replacing it with the standard reversal-mean
    estimator plus step-halving after two reversals gives **0.049 logMAR**.
11. **Contrast rungs that were byte-identical.** logCS 1.95, 2.10 and 2.25 all
    render to code 254 against a white background on 8-bit sRGB — three rungs,
    one image. The ladder is now truncated at the computed display ceiling
    (2.07) and a result there is reported as a bound, not a measurement.
12. **A single lapsed triplet ended the contrast run.** Stopping at the first
    failed triplet put a **−1.15 log CS** bias on high-sensitivity observers,
    who fail one triplet by chance and were then scored at that level.
    Termination now requires two consecutive failed triplets, as reading down a
    printed chart does (error 0.216 → 0.073).
13. **A thrown stage hung the app forever.** An undefined variable in the
    contrast loop left users on a frozen screen with no way forward — caught
    only by the end-to-end browser test. The app now aborts to a report and
    says what failed.
14. **Pupil latency reported at 30 fps** — frame quantization there has an SD
   of ~9.6 ms, which exceeds the *lower bound* of the physiological inter-eye
   latency asymmetry range (8.3–35 ms; Bergamin & Kardon 2003). Latency is now
   withheld below 55 fps rather than reported as noise.

### 4.4b Real-patient validation — and the failure it exposed

Every other benchmark here scores simulated observers or synthetic images. To
get closer to a real measurement, the production alignment path was run over
photographs of people **clinically categorised** on Wikimedia Commons with
esotropia, exotropia or hypertropia (category membership is editor-curated, so
unlike a keyword search it genuinely returns diagnosed patients), plus 60
presumed-normal faces sampled from the GazeCapture-derived corpus.

**It failed, informatively.**

| | before guards | after guards |
|---|---|---|
| specificity | **0.09** | **0.64** |
| median deviation, normal faces | **61 PD** | **21 PD** |
| worst single reading on a normal face | **163 PD** | rejected |

163 prism diopters exceeds any human strabismus. The pipeline was locating
window reflections, lamps and spectacle glare and treating them as corneal
catchlights — on the same code that scored **0.60 PD error** on synthetic
images. This is the sharpest example in the project of a synthetic benchmark
certifying something that does not survive contact with real photographs.

Two guards followed directly from the data:

* **A physiological ceiling.** Readings above 60 PD are rejected as artifacts
  and reported as inconclusive-with-instructions, never as a magnitude.
* **Frame-to-frame stability.** A real deviation is steady across frames while
  uncorrelated stray highlights jitter, so a magnitude now requires agreement
  across frames, and a single-frame estimate is capped at `weak-signal`
  regardless of image quality.

**What this validation does and does not establish.** It is a real-patient test
of the Hirschberg geometry on uncontrolled still photographs, so it exposes
artifact sensitivity that synthetic data cannot. It is *not* a clinical
validation: n is small (16 categorised patients, of which only a fraction were
measurable at all), category membership is not a cover test, the controls are
presumed-normal rather than examined, and — most importantly — a single
uncontrolled still is **not** the guided, dim-room, controlled-glint video the
product actually captures. The honest conclusion is narrower and more useful
than a headline number: **single-frame Hirschberg from uncontrolled
photographs is unreliable, and the product's controlled-glint video protocol
is not optional — it is the thing that makes the measurement possible at all.**

### 4.4c Hirschberg accuracy on REAL eye images, with ground truth

The clinical-photograph test (§4.4b) showed the measurement chain failing on
real images but could not say by *how much*, because a Commons photograph
carries no known deviation. That gap is closable: take real eye crops from the
corpus, suppress any pre-existing corneal highlight, and inject a specular
glint at a **known** decentration. Real skin, iris pigment, lids, lashes,
sensor noise and blur; exact ground truth. This is software doing what a
clinical transilluminator does — supplying its own light so its position is
known.

**Single-frame accuracy on real images, by injected deviation (300 crops):**

| true deviation | detection rate | mean abs error |
|---|---|---|
| 0 PD | 1.00 | 10.2 PD |
| 10 PD | 1.00 | 7.3 PD |
| 20 PD | 1.00 | 7.1 PD |
| 45 PD | 1.00 | 6.5 PD |
| **overall** | **1.00** | **7.5 PD** |

Against **0.60 PD** on synthetic images — a twelvefold degradation, and the
honest single-frame figure. Error is roughly constant across deviations, which
identifies it as a fixed localisation error rather than a scale error.

**The decisive result — frame aggregation.** The product never scores single
frames; it takes the median decentration across a capture segment. If the
per-frame error is independent noise it should shrink as 1/√n; if it is
systematic bias it will not shrink at all. Measured at 15 PD over 120 trials:

| | mean abs error |
|---|---|
| single real frame | **5.35 PD** |
| median of 40 real frames | **1.09 PD** (p90 2.13) |

It is noise, and aggregation removes it. A per-frame σ of 6.7 PD predicts
1.06 PD at n = 40; the measured value was 1.09, so the noise model is right and
can be used to set the frame requirement rather than guessing it:

| frames | expected magnitude error |
|---|---|
| 5 | 3.0 PD |
| **20** | **1.5 PD** |
| 40 | 1.1 PD |

`MIN_FRAMES_FOR_MAGNITUDE` is therefore 20, and each reported deviation now
carries its own `expected_error_pd` derived from the frames actually used.

**Why this matters.** Interexaminer agreement for the prism cover test — the
clinical reference — has 95% limits of roughly **±5 PD**. A 1.1 PD aggregated
error on real eye images sits comfortably inside the reference test's own
noise. This is the strongest accuracy statement in the project that rests on
real image data rather than simulation. It still is not a clinical validation:
the injected glint tests the measurement chain, not the optics of a genuinely
rotated eye, and no patient was measured.

**The null condition.** Running the same detection on real crops with *no*
glint injected, any confident reading is by construction an artifact. 14% of
crops still produced a measurement (median 71 PD, i.e. obvious nonsense), and
**4% passed the plausibility ceiling** — the residual false-confidence rate of
the current guards, and a number to reduce rather than hide.

### 4.5 Why real sessions failed before

Analysis of the real corpus found that **only 10.8% of real webcam eye crops
show a usable corneal reflex** under ordinary room lighting (662 of 6,130
crops; the figure was 10.5% on the first 1,154, so it replicates). This — not the
algorithm — is why alignment returned "inconclusive" on real users. The
capture protocol was changed so the pursuit target is a bright white disc on a
dark field, making the test supply its own catchlight, as a clinical
transilluminator does.

---

### 4.6 What the published literature says the bar is

The most directly comparable published system is **Melbourne Rapid Fields-web**,
a browser-based visual-field test run on ordinary laptops (including a 13"
MacBook Air at 33 cm) and validated against Humphrey SITA-Faster in a
multicentre study of 232 subjects. Its calibration strategy is the same one
used here — **screen brightness at maximum, a credit-card mire for pixel pitch,
and webcam face tracking for viewing distance, with no photometer**. It reports
MD bias −0.50 dB, 95% limits of agreement −6.80 to +5.80 dB, ICC 0.87, and —
critically — an **AUC of 0.84 versus the Humphrey's own 0.84**.

That result is the existence proof this project leans on: a browser on stock
consumer hardware, calibrated with a credit card and a webcam, *can* match a
clinical instrument's discrimination. It also shows what separates success from
failure. The cautionary counterexample is *Visual Fields Easy*, the same
hardware class and a similar test, but **uncalibrated and suprathreshold**: in
203 eyes it achieved AUROC 0.68 and **35% sensitivity at 90% specificity**, and
its authors concluded it had "inadequate diagnostic accuracy to be used as a
screening tool." The difference between the two is calibration and threshold
methodology, not hardware.

Two further external reference points bound expectations here:

* **Automated cover testing** with a 250 Hz tracker and a physical occluder
  reaches ±1.95Δ intersession; the human gold standard (prism cover test,
  interexaminer) has 95% limits of agreement of roughly ±5Δ. So a 2–6Δ
  microtropia is below the resolution of the clinical reference itself.
* **Stereoacuity** on a validated autostereo tablet (ASTEROID) has a
  test-retest coefficient of repeatability of ×2.9 (0.46 log₁₀) — meaning even
  a purpose-built instrument's stereo threshold moves by a factor of ~3 between
  sittings. Any screen-based stereo number should be read against that.

---

## 5. Limitations

1. **No human clinical validation.** Nothing here has been compared against an
   optometrist's measurement on the same eyes. Synthetic and weak-label
   benchmarks bound the algorithm, not the system.
2. **Weak labels are not ground truth.** Real IoU measures agreement with a
   geometric/photometric reference, not with human annotation.
3. **Display calibration.** Contrast and colour depend on uncalibrated display
   gamma and gamut; colour vision is therefore capped at `weak-signal` and can
   never return `measured`.
4. **Distance is self-reported.** A 20% distance error is a ≈0.08 logMAR acuity
   error — comparable to the entire algorithmic error budget.
5. **Photorefraction dead zone.** Refractive errors within ≈±1.25 D of the
   screen distance produce no crescent; reported honestly as such.
6. **Astigmatism axis observability.** A single flash axis constrains meridians
   only within ±60°; a second capture with a rotated flash band would fix this.
7. **Screen flashes are weak.** Pupil amplitudes are not comparable to clinical
   transilluminators; only inter-eye asymmetry is defensible.
8. **Population constants.** HVID (11.7 mm) and pupil/iris ratio are population
   means; individual deviation propagates 5–10% scale error.
9. **Spectacles.** Lens reflections both hide the corneal glint and create
   false speculars; a large share of crop rejections involve glasses.
10. **Visible-light pupillometry degrades on dark irides.** In NIR the iris
    stroma reflects strongly; in visible light melanin absorbs, and every
    clinical pupillometer is therefore infrared. The one large, diverse,
    bare-phone visible-light study against an NPi-200 (n=200 eyes) reported
    ICCs of 0.02–0.58 and a *negative* ICC for latency. Consumer webcams carry
    IR-cut filters, so this ceiling applies here too.
11. **Absolute pupil millimetres are biased.** The camera sees the *entrance*
    pupil, magnified ~13% by the cornea, while the limbus used as the ruler is
    not magnified. Relative measures (percent constriction, inter-eye ratios)
    are unaffected and are reported as primary; millimetre values are
    secondary.
12. **Stereo is display-floor limited.** At 140 ppi and 50 cm the whole-pixel
    disparity floor is ~75 arcsec, so the 40 arcsec clinical rung is not
    presentable without subpixel rendering. The floor is computed per session
    and reported with the result.

---

## 5b. Regulatory posture

This is a research prototype and is deliberately built to stay one. The
relevant precedent is direct: the FDA issued a warning letter against
Opternative in 2017 because its online test was *"intended for use in the
diagnosis of disease"* and produced a spectacle prescription, and the product
was recalled in 2019 for lack of 510(k) clearance. The first cleared online
vision test (2022) is cleared for **visual acuity only, in adults aged 22–40,
as supportive information reviewed by a licensed doctor** — a far narrower
claim than "an eye exam." Several US states additionally prohibit prescriptions
generated solely from a refractive measurement or by electronic means.

The design consequences are concrete and are enforced in code and tests:

* **No prescription is issued.** The photorefraction module reports a research
  estimate of defocus and the report explicitly states it cannot be used to
  order glasses or contact lenses (`test_refraction_is_labeled_not_a_prescription`).
* **No disease is named as a finding.** Modules report functional observations
  ("one meridian appeared consistently sharper", "the pupils differ in resting
  size") rather than diagnoses.
* **No output is labeled abnormal, positive, or failed.** The tiers describe
  data quality, not pathology.
* **Referral language is generic** — "worth getting checked" — which is the
  form the FDA's general-wellness guidance explicitly permits.
* **The report carries a not-FDA-cleared, not-a-medical-device footer.**

---

## 6. What would make this clinically credible

In priority order, the work that would actually move this from "research
prototype" to "defensible screening instrument":

1. **A human validation study.** 50+ subjects with same-day optometrist
   measurement; report Bland-Altman mean difference and 95% limits of
   agreement against chart acuity, autorefractor sphere/cylinder, and prism
   cover test. This is the only number that matters and the only one absent.
2. **Human-annotated eye masks** on a few hundred real crops, replacing weak
   labels as the evaluation reference.
3. **Automatic distance estimation** from interocular distance in pixels plus
   the calibrated screen scale, removing self-reported distance.
4. **Two-axis photorefraction** (rotated second flash) for axis observability.
5. **Prospective sensitivity/specificity** for referral decisions against a
   clinician's referral judgement, not against simulated ground truth.

---

## 7. Reproducing

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"

.venv/bin/python -m pytest                          # full suite

# data (no auth required; provenance written alongside)
.venv/bin/python -m visionscreen.data.hf_datasets --date 2026-08-04
.venv/bin/python -m visionscreen.data.fetch_public
.venv/bin/python -m visionscreen.data.build_real_corpus

# training and benchmarks
.venv/bin/python -m visionscreen.ml.train --n-train 12000 --epochs 20 --device mps
.venv/bin/python -m benchmarks.bench_segmentation     # sim-to-real table
.venv/bin/python -m benchmarks.bench_battery          # simulated-patient table
.venv/bin/python -m benchmarks.bench_module1          # acuity
.venv/bin/python -m benchmarks.bench_module2          # Hirschberg
.venv/bin/python -m benchmarks.bench_module3          # photorefraction

.venv/bin/uvicorn webapp.app:app --port 8000          # the app
```

---

## 8. References

- Bailey, I.L. & Lovie, J.E. (1976). New design principles for visual acuity
  letter charts. *Am J Optom Physiol Opt* 53(11):740-745.
- Pelli, D.G., Robson, J.G. & Wilkins, A.J. (1988). The design of a new letter
  chart for measuring contrast sensitivity. *Clin Vis Sci* 2(3):187-199.
- Bobier, W.R. & Braddick, O.J. (1985). Eccentric photorefraction: optical
  analysis and empirical measures. *Am J Optom Physiol Opt* 62(9):614-620.
- Amsler, M. (1947). L'examen qualitatif de la fonction maculaire.
  *Ophthalmologica* 114:248-261.
- Hirschberg test ratio, ~15-22 prism diopters per mm of reflex decentration
  (standard strabismus references).
- Ishihara, S. Tests for Colour-Blindness — confusion-line design principles.
- MediaPipe FaceLandmarker (Google), 478-landmark mesh with iris refinement.
- Ronneberger, O., Fischer, P. & Brox, T. (2015). U-Net: convolutional networks
  for biomedical image segmentation. *MICCAI*.
- Lam, B.L., Thompson, H.S. & Corbett, J.J. (1987). The prevalence of simple
  anisocoria. *Am J Ophthalmol* 104(1):69-73.
- Bergamin, O. & Kardon, R.H. (2003). Latency of the pupil light reflex:
  sampling, stimulus intensity, and variation in normal subjects. *IOVS*.
- Bell, R.A. et al. (1993). Clinical grading of relative afferent pupillary
  defects. *Arch Ophthalmol* 111(7):938-942.
- Wang, Y. et al. (2018). Pupil light reflex evoked by light-emitting diode and
  computer screen stimulation. *PLoS One* 13(6):e0197739.
- Hartle, B., Vancleef, K., Read, J.C.A. et al. (2019). Stereotests without
  stereopsis: monocular and binocular non-stereoscopic cues. *Sci Rep* 9:5779.
- Read, J.C.A. et al. (2020). ASTEROID stereotest v1.0: validation and
  repeatability. *Ophthalmic Physiol Opt* 40:815-827.
- Kong, Y.X.G., He, M., Crowston, J.G. & Vingrys, A.J. (2016). A comparison of
  perimetric results from a tablet perimeter and Humphrey Field Analyzer.
  *Transl Vis Sci Technol*.
- Rüfer, F., Schröder, A. & Erb, C. (2005). White-to-white corneal diameter.
  *Cornea* 24(3):259-261.
- Anderson, H.A., Manny, R.E., Cotter, S.A. et al. (2010). Effect of
  examiner experience and technique on the alternate cover test.
  *Optom Vis Sci* 87(3):168-175.
- Brodie, S.E. (1987). Photographic calibration of the Hirschberg test.
  *IOVS* 28(4):736-742.
