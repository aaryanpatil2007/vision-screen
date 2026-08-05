"""Anaemia from conjunctival pallor — the one external-eye task with real labels.

Haemoglobin is red. Less of it makes the palpebral conjunctiva — the inner
surface of the lower lid, which is mucous membrane with no melanin and a dense
capillary bed — visibly paler. That last property is what makes this task
different from every other colour measurement in this project: the conjunctiva
carries no pigment of its own, so its colour reports blood rather than skin
tone. Nothing else visible on the eye has that property.

The data is CP-AnemiC (Asare et al., Mendeley `10.17632/m53vz6b7fx`, CC BY
4.0): 710 conjunctiva photographs from children aged 6-60 months across ten
hospitals in four regions of Ghana, each with a laboratory haemoglobin value.
Two things make it unusually good for this purpose — the labels are lab
measurements rather than clinical impressions, and the cohort is darkly
pigmented, which is the population where published screening tools degrade most
and are least often validated.

**The evaluation design is the point of this module.** Site prevalence in this
dataset ranges from 48% to 88%, and each hospital means a different camera,
different lighting, and a different operator. A random train/test split
therefore lets the model recognise the *site* rather than the *pallor*, and
site predicts the label. That is not a hypothetical: Collings et al. 2016
reported 93% sensitivity in training and 57% in validation on this exact task.

So the headline number here is leave-one-hospital-out, and the random-split
number is computed alongside it purely to show the size of the gap. Reporting
the random-split figure alone would be reporting the same illusion again.

Features are hand-built rather than learned, because 710 images across 10 sites
will not support a fine-tuned CNN without memorising sites, and because the
physical signal is known: the quantity of interest is the ratio of red to green
reflectance, which is what haemoglobin absorption actually modulates. Ratios
are also the exposure-robust choice — absolute brightness varies with every
camera and every room, while a channel ratio largely divides that out.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path("data/anemia/cp-anemic")
SHEET = DATA_DIR / "Anemia_Data_Collection_Sheet.xlsx"

#: WHO anaemia threshold for children 6-59 months, g/dL.
WHO_ANAEMIA_HB = 11.0

#: Collings et al. 2016 (PLoS One 11:e0153286) — conjunctival pallor from
#: digital photographs. Training 93% sensitivity / 78% specificity collapsed to
#: **57% / 83%** on validation. The validation pair is the honest benchmark.
COLLINGS_VALIDATION_SENS = 0.57
COLLINGS_VALIDATION_SPEC = 0.83


@dataclass
class Sample:
    image_id: str
    path: Path
    hb: float
    anaemic: bool
    severity: str
    age_months: float
    sex: str
    hospital: str
    region: str


def load_index(data_dir: Path = DATA_DIR) -> list[Sample]:
    """Pair every image with its laboratory haemoglobin and its site."""
    import openpyxl

    wb = openpyxl.load_workbook(data_dir / SHEET.name)
    rows = list(wb.active.iter_rows(values_only=True))[1:]
    out: list[Sample] = []
    for r in rows:
        if not r or not r[0] or not isinstance(r[1], (int, float)):
            continue
        image_id, hb, severity, age, sex, remark, hospital, _city, _mun, region = r[:10]
        anaemic = str(remark).strip().lower() == "anemic"
        folder = "Anemic" if anaemic else "Non-anemic"
        p = data_dir / folder / f"{image_id}.png"
        if not p.exists():
            continue
        out.append(Sample(
            image_id=str(image_id), path=p, hb=float(hb), anaemic=anaemic,
            severity=str(severity), age_months=float(age or 0), sex=str(sex),
            hospital=str(hospital).strip(), region=str(region).strip()))
    return out


# ------------------------------------------------------------------ features --

FEATURE_NAMES = (
    "erythema_mean", "erythema_p25", "erythema_p75", "erythema_sd",
    "rg_ratio_mean", "rg_ratio_p25", "rb_ratio_mean",
    "lab_a_mean", "lab_a_p25", "lab_a_p75", "lab_b_mean",
    "hsv_h_mean", "hsv_s_mean", "hsv_s_p25", "hsv_s_p75",
    "redness_mean", "redness_p25", "redness_sd",
    "value_mean", "chroma_mean",
)


def conjunctiva_mask(bgr: np.ndarray) -> np.ndarray:
    """Keep the fleshy interior of the crop, drop lashes, skin edge and glare.

    The images are already cropped to the everted lower lid, but they still
    include dark lash pixels and specular highlights from the photographer's
    light. Both distort a colour average badly and in opposite directions, so
    each is trimmed by percentile rather than by a fixed threshold — the images
    come from ten different cameras and no fixed cut-point survives that.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lo, hi = np.percentile(gray, [12, 97])
    mask = (gray > lo) & (gray < hi)
    # drop near-white specular pixels, which carry the illuminant not the tissue
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask &= hsv[..., 1] > 25
    if mask.sum() < 200:                       # fall back rather than return junk
        mask = np.ones(gray.shape, bool)
    return mask


def extract_features(bgr: np.ndarray) -> np.ndarray:
    """Colour statistics chosen for what haemoglobin actually does to light.

    The erythema index, log10(R/G), is the workhorse: haemoglobin absorbs
    strongly in green and weakly in red, so the red-to-green ratio tracks blood
    content and largely divides out exposure and illuminant intensity. The
    remaining features add the same information in other colour spaces plus
    distribution shape, since pallor is often patchy rather than uniform.
    """
    mask = conjunctiva_mask(bgr)
    px = bgr[mask].astype(np.float64) + 1.0
    b, g, r = px[:, 0], px[:, 1], px[:, 2]

    erythema = np.log10(r / g)
    rg = r / g
    rb = r / b
    redness = (r - g) / (r + g)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    a_ch = lab[..., 1][mask] - 128.0
    b_ch = lab[..., 2][mask] - 128.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    h_ch = hsv[..., 0][mask] * 2.0                 # OpenCV packs hue into 0-179
    s_ch = hsv[..., 1][mask] / 255.0
    v_ch = hsv[..., 2][mask] / 255.0

    def p(x, q):
        return float(np.percentile(x, q))

    return np.array([
        float(erythema.mean()), p(erythema, 25), p(erythema, 75), float(erythema.std()),
        float(rg.mean()), p(rg, 25), float(rb.mean()),
        float(a_ch.mean()), p(a_ch, 25), p(a_ch, 75), float(b_ch.mean()),
        float(h_ch.mean()), float(s_ch.mean()), p(s_ch, 25), p(s_ch, 75),
        float(redness.mean()), p(redness, 25), float(redness.std()),
        float(v_ch.mean()), float(np.hypot(a_ch, b_ch).mean()),
    ], dtype=np.float64)


def build_matrix(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """-> (X, hb, anaemic, group) with group = hospital, for grouped splits."""
    X, hb, y, groups = [], [], [], []
    for s in samples:
        img = cv2.imread(str(s.path))
        if img is None:
            continue
        X.append(extract_features(img))
        hb.append(s.hb)
        y.append(s.anaemic)
        groups.append(s.hospital)
    return (np.vstack(X), np.array(hb, float),
            np.array(y, bool), np.array(groups, object))
