# Clinical Validation Protocol

**Purpose.** Every accuracy figure in this project is synthetic or weak-label.
This document specifies the study that would replace them with real ones, and
the repo ships the analysis half (`visionscreen.validation`) so the numbers
come out the moment the data exists.

**Status: not yet run. No human subject data has been collected.**

---

## 1. What is being tested

That VisionScreen's outputs agree with a clinician's measurement on the same
eyes on the same day, within the tolerance published tests are held to.

**Primary endpoint** — Bland-Altman bias and 95% limits of agreement for
distance visual acuity, per eye, VisionScreen versus ETDRS chart.

**Secondary endpoints** — contrast sensitivity versus Pelli-Robson; ocular
alignment versus prism cover test; refraction versus autorefractor or
subjective refraction.

**Correlation is not an endpoint.** A test offset by a constant correlates at
r = 1.00 and agrees with nothing. The repo's test suite encodes this
explicitly (`test_high_correlation_can_still_fail_agreement`).

---

## 2. Acceptance criteria

Taken from the systematic reviews of app-based acuity, where the stated bar is
that limits of agreement should compare with conventional chart tests —
below ±0.20 logMAR and ideally approaching ±0.15.

| Measure | Reference | Max \|bias\| | Max LoA half-width |
|---|---|---|---|
| Acuity (logMAR) | ETDRS chart, 4 m | 0.05 | 0.20 |
| Contrast (log CS) | Pelli-Robson, 1 m | 0.10 | 0.30 |
| Alignment (PD) | prism cover test | 2.0 | 6.0 |
| Refraction (D, spherical equivalent) | autorefractor or subjective | 0.25 | 1.00 |

The alignment figure deserves a note: interexaminer agreement for the prism
cover test itself is roughly ±5 PD, so ±6 is close to the reference's own
noise. A tighter criterion would be measuring the reference, not the test.

---

## 3. Sample size

For a limits-of-agreement study, precision on the LoA depends on n:

| n subjects | 95% CI half-width on each limit (approx.) |
|---|---|
| 20 | ±0.62 SD |
| 50 | ±0.39 SD |
| 100 | ±0.27 SD |

**Minimum useful: 50 subjects (100 eyes).** Below ~20 the interval on the
limits is wider than the limits themselves and the study cannot distinguish a
good test from a bad one.

Enrich for abnormality: a cohort of only normal eyes cannot estimate
sensitivity. Target roughly one third with corrected refractive error, one
third with a known ocular condition, one third unremarkable.

---

## 4. Procedure per subject

1. **Consent and record** age, habitual correction, known ocular history.
2. **Clinician first, masked to the app result.** ETDRS at 4 m per eye;
   Pelli-Robson at 1 m with +0.75 add; cover test at near and distance;
   autorefraction.
3. **VisionScreen session**, examiner not coaching: card calibration, stated
   distance, brightness attestation, full battery.
4. **Record both** into a `Study` (below). Do not round or adjust either value.
5. **Order counterbalanced** across subjects so learning effects do not load
   onto one arm.

Critical: the app must not be re-run after seeing the clinical value. One
session per subject, recorded whatever it produces, including inconclusive.

---

## 5. Recording and analysis

```python
from visionscreen.validation import Study, report, format_report

study = Study(name="visionscreen-v1-agreement")
study.record_pair(
    subject_id="S001", measure="acuity_logmar", eye="OD",
    index_value=0.12,          # what VisionScreen reported
    reference_value=0.10,      # what the chart measured
    reference_method="ETDRS chart 4 m, per-letter scoring",
)
# ... one call per paired measurement ...
open("study.json", "w").write(study.to_json())
```

```bash
python -m visionscreen.validation study.json
```

Output is bias, 95% limits of agreement, the confidence interval on the bias,
and a pass/fail against the criteria above.

---

## 6. Reporting rules

- Report **every** enrolled subject, including sessions the app called
  inconclusive. The inconclusive rate is a headline result, not an exclusion.
- Report the **measurable rate** separately from accuracy among the measurable.
  A test that only works on a third of people is not accurate — it is narrow.
- Report per-eye, not per-subject averaged; eyes within a subject are
  correlated and averaging hides it.
- Publish the raw pairs.

---

## 6b. A partial substitute assembled from the literature

`visionscreen.data.pmc_cases` mines open-access case reports for figures that
pair a patient photograph with a clinician's prism cover test measurement in
the caption. Yield is 14 pairs per 150 articles; the ~1,136 matching
open-access articles should give on the order of 100.

This is real paired data — a real patient, a real clinician's measurement —
and it is the closest substitute for the study above that can be assembled
without recruiting anyone. Its limits are severe and must travel with any
number derived from it:

* The caption's deviation may refer to a **different gaze position or time
  point** than the photograph; pre/post-operative figure pairs are common.
* **Camera distance and magnification are unknown**, so the iris-diameter scale
  is the only calibration available.
* Publication selects for **photogenic, larger deviations** — a spectrum bias
  that inflates apparent sensitivity and says nothing about specificity.
* There are **no negative controls**: case reports do not publish photographs
  of normally-aligned eyes with a documented 0 PD cover test.

It can therefore bound agreement on large deviations in selected patients. It
cannot estimate specificity, and it is not a substitute for §4.

**Status: index built, images not retrieved.** Direct figure endpoints return
404/500, and `pmc.ncbi.nlm.nih.gov` serves a reCAPTCHA challenge to scripted
clients. That is a deliberate access control and the right response is to use a
sanctioned route rather than defeat it: the **PMC Open Access package service**
(`oa.fcgi`, bulk FTP mirror) or an institutional subscription. The tool records
the canonical image URL and OA package lookup for every pair so retrieval and
scoring run unchanged wherever that access exists.

## 7. What this study still will not establish

Agreement is not clinical utility. Even a perfectly agreeing screener does not
show that screening improves outcomes — the USPSTF gives screening asymptomatic
adults for impaired acuity an **I statement** (insufficient evidence), with
randomised trials showing no difference in visual or functional outcomes. A
separate, much larger study would be needed for any outcome claim, and none is
made here.

Nor does it touch the structural blind spot: intraocular pressure, the
iridocorneal angle and the retina remain invisible to a camera, so a normal
result cannot exclude glaucoma, diabetic retinopathy or macular disease.
