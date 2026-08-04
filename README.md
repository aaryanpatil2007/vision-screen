# VisionScreen

Webcam-based **vision screening** from a guided browser test: visual acuity,
eye alignment (strabismus), and refractive-error estimation via eccentric
photorefraction — physics-based inversion on top of MediaPipe perception.

> **This is a screening signal only, not a diagnosis. See an optometrist for
> a clinical evaluation.** Accuracy numbers below are synthetic-benchmark
> results; the system has not been validated on real eyes.

## What it does

You sit in front of your webcam and the browser walks you through:

1. **Tumbling-E acuity test** — press arrow keys for letter directions; a
   staircase homes in on your acuity (logMAR) while the camera watches for
   squinting and leaning in.
2. **Dot pursuit** — follow a moving dot; corneal-reflex symmetry (Hirschberg)
   and inter-eye pursuit conjugacy screen for misalignment.
3. **Dim-room flash** — a bright screen band in a dark room produces a red
   reflex crescent; its geometry is inverted to a defocus estimate (diopters).

Output: an HTML report with per-module findings and confidence tiers
(`measured` / `weak-signal` / `inconclusive`). Quality gates ask for retakes
instead of guessing.

## Quickstart

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest                      # 69 tests
.venv/bin/uvicorn webapp.app:app --port 8000    # open http://localhost:8000
```

## Benchmarks (synthetic, reproducible)

| module | headline result | command |
|---|---|---|
| acuity | 0.063 logMAR mean abs error (50 observers) | `python -m benchmarks.bench_module1` |
| alignment | 0.60 PD mean abs error, 100% detection | `python -m benchmarks.bench_module2` |
| photorefraction | 0.063 D mean abs spherical-equivalent error | `python -m benchmarks.bench_module3` |

Full methods, limitations, and citations: [`docs/WRITEUP.md`](docs/WRITEUP.md).

## Real-world reference images

```bash
.venv/bin/python -m visionscreen.data.fetch_public
```

fetches openly licensed clinical examples (strabismus, red reflex, …) from
Wikimedia Commons into `data/real/commons/` with per-item provenance.

## Layout

```
src/visionscreen/
  protocol.py        session/segment/event schema
  perception/        MediaPipe landmarks, eye geometry, iris + reflex
  quality/gates.py   capture-quality gates → retake instructions
  modules/           acuity, behavioral, alignment, photoref scoring
  synth/             schematic eye + photorefraction crescent renderers
  analyzer.py        video + session meta → findings
  data/              provenance-logged public image fetcher
webapp/              FastAPI + vanilla-JS guided protocol UI
benchmarks/          per-module accuracy harnesses → results/*.json
docs/                spec, plans, research writeup
```

## Design docs

- Spec: `docs/superpowers/specs/2026-08-04-vision-screening-pipeline-design.md`
- Plans: `docs/superpowers/plans/` (foundation+module1, alignment, photorefraction)
