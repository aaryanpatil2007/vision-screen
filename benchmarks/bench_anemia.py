"""Anaemia from conjunctival pallor, evaluated two ways to show what leaks.

Runs the same model under two evaluation protocols:

* **leave-one-hospital-out** — every test image comes from a site the model has
  never seen, so it cannot recognise the camera, the lighting or the operator.
  This is the number that means something.
* **stratified random split** — the protocol most papers use, reported here
  purely so the gap between the two is visible rather than assumed.

The gap *is* the result. Collings et al. 2016 measured 93% sensitivity in
training and 57% in validation on this same task; if the two protocols here
disagree by a similar margin, that is confirmation that site identity, not
pallor, is doing much of the work under a random split.

    python -m benchmarks.bench_anemia
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from visionscreen.ml.anemia import (
    COLLINGS_VALIDATION_SENS,
    COLLINGS_VALIDATION_SPEC,
    WHO_ANAEMIA_HB,
    build_matrix,
    load_index,
)

RESULTS = Path("results/anemia.json")


def _classifier(kind: str):
    if kind == "linear":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=0.5, max_iter=2000,
                                                class_weight="balanced"))
    return GradientBoostingClassifier(random_state=0, n_estimators=200,
                                      max_depth=2, learning_rate=0.05)


def _regressor(kind: str):
    if kind == "linear":
        return make_pipeline(StandardScaler(), Ridge(alpha=5.0))
    return GradientBoostingRegressor(random_state=0, n_estimators=250,
                                     max_depth=2, learning_rate=0.05)


def _threshold_at_specificity(y: np.ndarray, prob: np.ndarray,
                              target_spec: float) -> float:
    """The score cut-off that achieves a given specificity.

    Comparing two tests at whatever threshold each happened to use compares
    operating points, not tests. A model with a higher AUC can look worse at
    0.5 simply because it sits further along its own curve. Matching
    specificity to the reference study is what makes the sensitivities
    comparable.
    """
    neg = np.sort(prob[~y])
    if neg.size == 0:
        return 0.5
    idx = min(int(np.ceil(target_spec * neg.size)), neg.size - 1)
    return float(neg[idx])


def _youden_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    """The cut-off maximising sensitivity + specificity - 1."""
    order = np.unique(prob)
    best, best_j = 0.5, -1.0
    for t in order:
        pred = prob >= t
        sens = (pred & y).sum() / max(y.sum(), 1)
        spec = (~pred & ~y).sum() / max((~y).sum(), 1)
        if sens + spec - 1 > best_j:
            best, best_j = float(t), sens + spec - 1
    return best


def _binary_metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = prob >= threshold
    tp = int((pred & y).sum())
    tn = int((~pred & ~y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "sensitivity": round(sens, 4), "specificity": round(spec, 4),
        "ppv": round(tp / max(tp + fp, 1), 4),
        "npv": round(tn / max(tn + fn, 1), 4),
        "balanced_accuracy": round((sens + spec) / 2, 4),
        "auc": round(float(roc_auc_score(y, prob)), 4) if len(set(y.tolist())) > 1 else None,
        "n": int(len(y)), "prevalence": round(float(y.mean()), 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


CANDIDATES = ("linear", "boosted")


def _select_model(X, y, groups) -> str:
    """Choose the model family using only the training fold.

    Running both families on the held-out result and reporting the winner is
    test-set selection: with two candidates it is worth a few tenths of AUC of
    optimism, and it is exactly the practice that makes published screening
    numbers fail to replicate. The choice is therefore made by an inner
    leave-one-group-out loop *inside* each training fold, so the outer estimate
    never sees the decision.
    """
    inner = LeaveOneGroupOut()
    scores = {}
    for kind in CANDIDATES:
        prob = np.zeros(len(y), float)
        ok = True
        for tr, te in inner.split(X, y, groups=groups):
            if len(set(y[tr].tolist())) < 2:
                ok = False
                break
            prob[te] = _classifier(kind).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        scores[kind] = roc_auc_score(y, prob) if ok and len(set(y.tolist())) > 1 else 0.0
    return max(scores, key=scores.get)


def evaluate_nested(X, hb, y, groups) -> dict:
    """Leave-one-hospital-out, with model choice made inside each fold."""
    prob = np.zeros(len(y), float)
    hb_hat = np.zeros(len(y), float)
    chosen: list[str] = []
    per_site = {}
    for train, test in LeaveOneGroupOut().split(X, y, groups=groups):
        kind = _select_model(X[train], y[train], groups[train])
        chosen.append(kind)
        prob[test] = _classifier(kind).fit(X[train], y[train]).predict_proba(X[test])[:, 1]
        hb_hat[test] = _regressor(kind).fit(X[train], hb[train]).predict(X[test])
        if len(set(y[test].tolist())) > 1:
            per_site[groups[test][0]] = {
                "n": int(len(test)), "model": kind,
                "auc": round(float(roc_auc_score(y[test], prob[test])), 4)}
    res = _binary_metrics(y, prob, 0.5)
    res["at_youden"] = _binary_metrics(y, prob, _youden_threshold(y, prob))
    res["at_reference_specificity"] = _binary_metrics(
        y, prob, _threshold_at_specificity(y, prob, COLLINGS_VALIDATION_SPEC))
    res["hb_mae"] = round(float(np.abs(hb_hat - hb).mean()), 3)
    res["hb_r"] = round(float(np.corrcoef(hb_hat, hb)[0, 1]), 4)
    res["models_chosen"] = dict(zip(per_site.keys(), chosen))
    res["per_site"] = per_site
    res["auc_ci95"] = bootstrap_ci(y, prob)
    aucs = [v["auc"] for v in per_site.values()]
    res["per_site_auc_range"] = [min(aucs), max(aucs)] if aucs else None
    res["worst_site_auc"] = min(aucs) if aucs else None
    return res


def bootstrap_ci(y, prob, n_boot: int = 2000, seed: int = 0) -> list[float]:
    """95% CI on AUC. A point estimate from 710 images invites over-reading."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx].tolist())) < 2:
            continue
        vals.append(roc_auc_score(y[idx], prob[idx]))
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def evaluate(X, hb, y, groups, kind: str = "boosted") -> dict:
    """Out-of-fold predictions under both protocols."""
    out: dict = {}

    for name, splitter, split_args in (
        ("leave_one_hospital_out", LeaveOneGroupOut(), {"groups": groups}),
        ("random_5fold", StratifiedKFold(5, shuffle=True, random_state=0), {}),
    ):
        prob = np.zeros(len(y), float)
        hb_hat = np.zeros(len(y), float)
        per_site = {}
        for train, test in splitter.split(X, y, **split_args):
            clf = _classifier(kind).fit(X[train], y[train])
            prob[test] = clf.predict_proba(X[test])[:, 1]
            reg = _regressor(kind).fit(X[train], hb[train])
            hb_hat[test] = reg.predict(X[test])
            if name == "leave_one_hospital_out":
                site = groups[test][0]
                if len(set(y[test].tolist())) > 1:
                    per_site[site] = {
                        "n": int(len(test)),
                        "auc": round(float(roc_auc_score(y[test], prob[test])), 4),
                        "hb_mae": round(float(np.abs(hb_hat[test] - hb[test]).mean()), 3),
                    }

        res = _binary_metrics(y, prob, 0.5)
        # the two comparisons that actually mean something
        res["at_youden"] = _binary_metrics(y, prob, _youden_threshold(y, prob))
        res["at_reference_specificity"] = _binary_metrics(
            y, prob, _threshold_at_specificity(y, prob, COLLINGS_VALIDATION_SPEC))
        res["hb_mae"] = round(float(np.abs(hb_hat - hb).mean()), 3)
        res["hb_rmse"] = round(float(np.sqrt(((hb_hat - hb) ** 2).mean())), 3)
        # correlation of predicted with true haemoglobin
        res["hb_r"] = round(float(np.corrcoef(hb_hat, hb)[0, 1]), 4)
        if per_site:
            res["per_site"] = per_site
            aucs = [v["auc"] for v in per_site.values()]
            res["per_site_auc_range"] = [min(aucs), max(aucs)]
        out[name] = res

    honest = out["leave_one_hospital_out"]
    optimistic = out["random_5fold"]
    out["leakage_gap"] = {
        "auc_drop": round((optimistic["auc"] or 0) - (honest["auc"] or 0), 4),
        "sensitivity_drop": round(optimistic["sensitivity"] - honest["sensitivity"], 4),
        "hb_mae_increase": round(honest["hb_mae"] - optimistic["hb_mae"], 3),
        "note": ("How much of the random-split score came from recognising the "
                 "site rather than the pallor."),
    }
    return out


def main() -> None:
    samples = load_index()
    X, hb, y, groups = build_matrix(samples)
    print(f"{len(y)} images, {len(set(groups.tolist()))} hospitals, "
          f"{y.mean():.1%} anaemic, Hb {hb.min():.1f}-{hb.max():.1f} g/dL",
          flush=True)

    results = {}
    for kind in ("linear", "boosted"):
        results[kind] = evaluate(X, hb, y, groups, kind=kind)
        h = results[kind]["leave_one_hospital_out"]
        r = results[kind]["random_5fold"]
        print(f"\n[{kind}]")
        print(f"  leave-one-hospital-out : AUC {h['auc']:.3f}  "
              f"sens {h['sensitivity']:.3f}  spec {h['specificity']:.3f}  "
              f"Hb MAE {h['hb_mae']:.2f} g/dL")
        print(f"  random 5-fold          : AUC {r['auc']:.3f}  "
              f"sens {r['sensitivity']:.3f}  spec {r['specificity']:.3f}  "
              f"Hb MAE {r['hb_mae']:.2f} g/dL")
        m = h["at_reference_specificity"]; j = h["at_youden"]
        print(f"  at Youden point        : sens {j['sensitivity']:.3f}  "
              f"spec {j['specificity']:.3f}  bal.acc {j['balanced_accuracy']:.3f}")
        print(f"  at reference spec {COLLINGS_VALIDATION_SPEC:.2f}   : "
              f"sens {m['sensitivity']:.3f}  spec {m['specificity']:.3f}")
        print(f"  leakage gap            : AUC {results[kind]['leakage_gap']['auc_drop']:+.3f}")

    # the defensible headline: model chosen inside each fold, not on the result
    nested = evaluate_nested(X, hb, y, groups)
    print(f"\n[nested]  model selected within each training fold")
    ci = nested["auc_ci95"]
    print(f"  leave-one-hospital-out : AUC {nested['auc']:.3f} "
          f"(95% CI {ci[0]:.3f}-{ci[1]:.3f})  Hb MAE {nested['hb_mae']:.2f} g/dL  "
          f"r={nested['hb_r']:.2f}")
    print(f"  worst held-out site    : AUC {nested['worst_site_auc']:.3f}  "
          f"(range {nested['per_site_auc_range'][0]:.3f}-"
          f"{nested['per_site_auc_range'][1]:.3f} across 10 hospitals)")
    nm = nested["at_reference_specificity"]
    print(f"  at reference spec {COLLINGS_VALIDATION_SPEC:.2f}   : "
          f"sens {nm['sensitivity']:.3f}  spec {nm['specificity']:.3f}")

    best = max(results, key=lambda k: results[k]["leave_one_hospital_out"]["auc"] or 0)
    h = results[best]["leave_one_hospital_out"]
    results["nested"] = nested
    payload = {
        "dataset": "CP-AnemiC (Mendeley 10.17632/m53vz6b7fx, CC BY 4.0)",
        "n_images": int(len(y)), "n_hospitals": len(set(groups.tolist())),
        "who_threshold_g_dl": WHO_ANAEMIA_HB,
        "results": results,
        "best_model_by_test_score": best,
        "nested_selection": nested,
        "note_on_selection": (
            "The headline is the nested figure: picking the better of two model "
            "families by their held-out score is test-set selection, so the "
            "choice is made by an inner leave-one-group-out loop instead."),
        "headline": {
            "protocol": "leave-one-hospital-out, nested model selection",
            "auc": nested["auc"],
            "sensitivity_at_reference_specificity":
                nested["at_reference_specificity"]["sensitivity"],
            "specificity": nested["at_reference_specificity"]["specificity"],
            "hb_mae_g_dl": nested["hb_mae"],
            "hb_correlation": nested["hb_r"],
        },
        "reference": {
            "study": "Collings et al. 2016, PLoS One 11:e0153286",
            "validation_sensitivity": COLLINGS_VALIDATION_SENS,
            "validation_specificity": COLLINGS_VALIDATION_SPEC,
        },
        # compared where it is fair: at the reference study's own specificity
        "beats_reference": (
            nested["at_reference_specificity"]["sensitivity"] >= COLLINGS_VALIDATION_SENS
            and nested["at_reference_specificity"]["specificity"]
            >= COLLINGS_VALIDATION_SPEC - 0.02),
    }
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=2))

    print(f"\nreference (Collings 2016 validation): "
          f"sens {COLLINGS_VALIDATION_SENS:.2f} spec {COLLINGS_VALIDATION_SPEC:.2f}")
    m = nested["at_reference_specificity"]
    print(f"ours (nested, leave-one-hospital-out, matched specificity): "
          f"sens {m['sensitivity']:.2f} spec {m['specificity']:.2f}")
    print(f"-> {'matches or beats' if payload['beats_reference'] else 'below'} reference")


if __name__ == "__main__":
    main()
