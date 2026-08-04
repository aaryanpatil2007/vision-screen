# VisionScreen

A webcam-based **vision screening battery** — thirteen guided tests in a browser,
analysed with clinical physics and a purpose-trained segmentation network.

> **Screening tool, not a diagnosis.** It cannot measure eye pressure, examine
> the retina, or rule out disease, and it has not been validated against
> clinical measurement in human subjects. In one study of adults who considered
> themselves healthy (median age 70), about **one in three** had a finding at a
> full eye exam that a test like this cannot detect. Use it to decide whether to
> book an eye exam — never to skip one. Research prototype: not FDA-cleared,
> issues no prescription, makes no diagnosis.

---

## What it measures

| Test | Output |
|---|---|
| Visual acuity — binocular, right eye, left eye | logMAR + Snellen, corrected for measured viewing distance |
| Contrast sensitivity | log CS (Pelli-Robson triplets, 2-of-3 rule) |
| Astigmatism | minus-cylinder axis from the clock dial |
| Color vision | red-green screen, protan/deutan lean |
| Central field | metamorphopsia / scotoma marks (Amsler grid) |
| Depth perception | stereo threshold in arcsec (dynamic random-dot, catch trials) |
| Binocular fusion | fusion / suppression / diplopia (Worth four-dot) |
| Eye movement | smooth-pursuit gain, catch-up saccades |
| Eye alignment | deviation in prism diopters (Hirschberg) |
| Pupil response | constriction %, anisocoria |
| Refraction | sphere / cylinder / axis estimate (eccentric photorefraction) |
| Viewing distance | measured per frame from iris diameter; corrects acuity |
| Viewing behaviour | squinting, lean-in, head tilt |

Two tests need red-cyan 3-D glasses. Everything else runs on a bare webcam.

Every finding carries a confidence tier — **measured**, **weak-signal**, or
**inconclusive** — and the system reports *inconclusive with instructions*
rather than guessing when the data will not support a number.

---

## Quickstart

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/uvicorn webapp.app:app --port 8000     # then open http://localhost:8000
```

The app calibrates your screen against a credit card (ISO/IEC 7810: 85.60 mm),
shows a live eye-tracking overlay while you test, and produces a printable
report. Everything runs locally; the video never leaves the machine.

---

## Key results

**Sim-to-real transfer** — the reason this project trains on real data at all:

| training data | synthetic mIoU | **real mIoU** | real pupil IoU |
|---|---|---|---|
| synthetic only | 0.928 | **0.237** | 0.382 |
| synthetic + weakly-labeled real | 0.926 | **0.722** | 0.848 |

**Battery accuracy** (120 simulated patients with realistic lapse rates):

| test | result |
|---|---|
| Visual acuity | 0.049 logMAR mean error; 93% within chart test-retest repeatability |
| Contrast sensitivity | 0.073 log CS mean error (in display range) |
| Strabismus (≥10 PD) | sensitivity 1.00 / specificity 1.00 |
| Anisocoria (≥1 mm) | sensitivity 1.00 / specificity 1.00 |
| Color deficiency | sensitivity 1.00 / specificity 0.96 |
| Astigmatism axis | 4.1° mean error |
| Photorefraction | 0.027 D mean spherical-equivalent error |

**Alignment accuracy on real eye images** — a controlled glint injected at a
known decentration onto real eye crops gives real appearance with exact ground
truth:

| condition | mean abs error |
|---|---|
| synthetic images | 0.60 PD |
| single real frame | **5.35 PD** |
| median of 40 real frames | **1.09 PD** |

Per-frame error is independent noise, so aggregation removes it — which is why
the capture is video, not a photo. For scale, interexaminer agreement on the
prism cover test is about ±5 PD, so 1.1 PD sits inside the clinical reference's
own noise. The frame requirement is set from this measurement, and every
reported deviation carries its own expected error.

**Real-patient check** — the production alignment path run over photographs of
clinically categorised strabismus patients plus presumed-normal faces:

| | before artifact guards | after |
|---|---|---|
| specificity | 0.09 | **0.64** |
| median deviation on normal faces | 61 PD | **21 PD** |

The first column is the point: on the same code that scored 0.60 PD error on
synthetic images, real uncontrolled photographs produced readings up to 163
prism diopters — beyond any human strabismus — because window and lamp
reflections were being read as corneal catchlights. That failure drove two
guards (a physiological ceiling and a frame-stability requirement) and is the
clearest evidence in the project for why the controlled-glint video protocol
is not optional.

Full methods, limitations, and citations: [`docs/WRITEUP.md`](docs/WRITEUP.md).
Clinical study design and analysis kit: [`docs/VALIDATION_PROTOCOL.md`](docs/VALIDATION_PROTOCOL.md).

---

## Data pipeline

```bash
.venv/bin/python -m visionscreen.data.hf_datasets --date 2026-08-04  # real images
.venv/bin/python -m visionscreen.data.fetch_public                   # clinical examples
.venv/bin/python -m visionscreen.data.build_real_corpus              # weak labels
```

All sources are openly downloadable without credentials, and every item is
logged with its URL, license, and fetch date. Real images carry gaze targets
but no masks, so labels come from geometric (MediaPipe iris circle) and
photometric (dark pupil / bright reflex within that circle) priors, with
disagreeing crops **rejected** rather than mislabeled — a 54% acceptance rate
yielding 6,130 labeled crops from 5,980 source images.

## Training and benchmarks

```bash
.venv/bin/python -m visionscreen.ml.train --n-train 12000 --epochs 20 --device mps
.venv/bin/python -m benchmarks.bench_segmentation   # sim-to-real
.venv/bin/python -m benchmarks.bench_battery        # simulated patients
.venv/bin/python -m benchmarks.bench_module1        # acuity staircase
.venv/bin/python -m benchmarks.bench_module2        # Hirschberg
.venv/bin/python -m benchmarks.bench_module3        # photorefraction
```

## Tests

```bash
.venv/bin/python -m pytest        # unit, integration, and real-browser (Playwright)
```

Browser tests drive Chromium with a synthetic camera and assert that the
client's optotype and contrast maths match the server's bit for bit — so
rendering can never silently drift from scoring.

---

## Layout

```
src/visionscreen/
  protocol.py        session / segment / event schema
  perception/        landmarks, eye geometry, iris + reflex, distance
  ml/                EyeSegNet, datasets, training, inference
  synth/             domain-randomized eyes, photorefraction crescents
  quality/           capture gates -> retake instructions
  modules/           per-test scoring (acuity, contrast, astigmatic, amsler,
                     colorvision, motility, alignment, pupillometry, photoref)
  data/              dataset fetchers, weak labeling, corpus builder
  analyzer.py        video + protocol -> findings
  report.py          tiers, visuals, printable HTML report
webapp/              FastAPI + ES-module front end with live tracking overlay
benchmarks/          accuracy harnesses -> results/*.json
docs/                spec, plans, research writeup
```

## Design docs

- Spec: `docs/superpowers/specs/2026-08-04-vision-screening-pipeline-design.md`
- Plans: `docs/superpowers/plans/`
- Writeup: `docs/WRITEUP.md`
