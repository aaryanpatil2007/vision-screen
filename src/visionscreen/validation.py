"""Agreement analysis against a clinical reference — the study this needs.

Every accuracy number in this project is synthetic or weak-label. The one
measurement that would change that is a same-day comparison against an
optometrist, reported the way the literature reports it: Bland-Altman bias and
95% limits of agreement, not correlation.

Correlation is the standard mistake here. A test can correlate at r = 0.95 with
the reference and still be two chart lines out on every patient, because
correlation measures covariation, not agreement. The acceptance bar taken from
the systematic reviews of app-based acuity is **bias under 0.05 logMAR with 95%
limits of agreement inside +/-0.15 to 0.20 logMAR** — chart acuity's own
test-retest range.

This module is the analysis half. Collect paired measurements with
`record_pair`, then `agreement()` and `report()` produce the numbers a writeup
needs, including whether the acceptance criterion was met.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Acceptance criteria drawn from published app-vs-chart validation.
ACCEPTANCE = {
    "acuity_logmar": {"max_abs_bias": 0.05, "max_loa_half_width": 0.20},
    "contrast_logcs": {"max_abs_bias": 0.10, "max_loa_half_width": 0.30},
    "alignment_pd": {"max_abs_bias": 2.0, "max_loa_half_width": 6.0},
    "refraction_d": {"max_abs_bias": 0.25, "max_loa_half_width": 1.00},
}


@dataclass
class PairedMeasurement:
    subject_id: str
    measure: str          # key into ACCEPTANCE
    eye: str              # "OD" | "OS" | "OU"
    index_value: float    # what VisionScreen reported
    reference_value: float  # what the clinician measured
    reference_method: str   # e.g. "ETDRS chart 4 m", "prism cover test"
    notes: str = ""


@dataclass
class Study:
    name: str
    pairs: list[PairedMeasurement] = field(default_factory=list)

    def record_pair(self, **kwargs) -> None:
        self.pairs.append(PairedMeasurement(**kwargs))

    def to_json(self) -> str:
        return json.dumps(
            {"name": self.name, "pairs": [asdict(p) for p in self.pairs]}, indent=2
        )

    @classmethod
    def from_json(cls, text: str) -> "Study":
        d = json.loads(text)
        return cls(name=d["name"],
                   pairs=[PairedMeasurement(**p) for p in d["pairs"]])


def agreement(study: Study, measure: str) -> dict | None:
    """Bland-Altman agreement for one measure.

    Returns bias (index - reference), SD of differences, 95% limits of
    agreement, and the confidence interval on the bias. Correlation is
    reported too, but explicitly as a secondary descriptor.
    """
    diffs, means = [], []
    for p in study.pairs:
        if p.measure != measure:
            continue
        diffs.append(p.index_value - p.reference_value)
        means.append((p.index_value + p.reference_value) / 2)
    n = len(diffs)
    if n < 2:
        return None

    bias = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    loa_lower, loa_upper = bias - 1.96 * sd, bias + 1.96 * sd
    se_bias = sd / math.sqrt(n)

    # Pearson r, secondary only
    r = float("nan")
    if n >= 3 and statistics.pstdev(means) > 0 and statistics.pstdev(diffs) >= 0:
        xs = [p.index_value for p in study.pairs if p.measure == measure]
        ys = [p.reference_value for p in study.pairs if p.measure == measure]
        if statistics.pstdev(xs) > 0 and statistics.pstdev(ys) > 0:
            mx, my = statistics.fmean(xs), statistics.fmean(ys)
            num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
            den = math.sqrt(sum((a - mx) ** 2 for a in xs)
                            * sum((b - my) ** 2 for b in ys))
            r = num / den if den else float("nan")

    crit = ACCEPTANCE.get(measure)
    meets = None
    if crit:
        meets = (
            abs(bias) <= crit["max_abs_bias"]
            and max(abs(loa_lower - bias), abs(loa_upper - bias))
            <= crit["max_loa_half_width"]
        )

    return {
        "measure": measure,
        "n": n,
        "bias": round(bias, 4),
        "bias_ci95": [round(bias - 1.96 * se_bias, 4), round(bias + 1.96 * se_bias, 4)],
        "sd_of_differences": round(sd, 4),
        "loa95": [round(loa_lower, 4), round(loa_upper, 4)],
        "loa_half_width": round(1.96 * sd, 4),
        "pearson_r": None if math.isnan(r) else round(r, 4),
        "acceptance": crit,
        "meets_acceptance": meets,
    }


def report(study: Study) -> dict:
    measures = sorted({p.measure for p in study.pairs})
    results = {m: agreement(study, m) for m in measures}
    return {
        "study": study.name,
        "n_subjects": len({p.subject_id for p in study.pairs}),
        "n_pairs": len(study.pairs),
        "measures": {m: r for m, r in results.items() if r is not None},
        "underpowered": [m for m, r in results.items() if r is None],
    }


def format_report(rep: dict) -> str:
    lines = [
        f"# Clinical agreement — {rep['study']}",
        "",
        f"{rep['n_subjects']} subjects, {rep['n_pairs']} paired measurements.",
        "",
        "| measure | n | bias | 95% LoA | meets criterion |",
        "|---|---|---|---|---|",
    ]
    for m, r in rep["measures"].items():
        ok = "yes" if r["meets_acceptance"] else (
            "no" if r["meets_acceptance"] is False else "-")
        lines.append(
            f"| {m} | {r['n']} | {r['bias']:+.3f} | "
            f"{r['loa95'][0]:+.3f} to {r['loa95'][1]:+.3f} | {ok} |"
        )
    if rep["underpowered"]:
        lines += ["", f"Too few pairs to analyse: {', '.join(rep['underpowered'])}."]
    lines += [
        "",
        "Bias is index minus reference. Limits of agreement are bias +/- 1.96 SD "
        "of the differences. Correlation is deliberately not the headline: a test "
        "can correlate at r = 0.95 and still be two chart lines out on every "
        "patient.",
    ]
    return "\n".join(lines)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Analyse a paired clinical study")
    ap.add_argument("study_json", help="path to a study file written by Study.to_json()")
    ap.add_argument("--out", default="results/clinical_agreement.json")
    args = ap.parse_args()

    study = Study.from_json(Path(args.study_json).read_text())
    rep = report(study)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(format_report(rep))


if __name__ == "__main__":
    main()
