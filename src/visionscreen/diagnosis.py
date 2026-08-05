"""From measurements to named conditions: a differential, with its reasoning shown.

Every other module answers a question about one measurement. This one does the
thing a clinician does at the end of an exam — looks at all of it together and
asks what would explain it.

The method is diagnostic likelihood ratios, not a scoring heuristic. Each
condition starts at its population prevalence, expressed as odds; each piece of
evidence multiplies those odds by its likelihood ratio; the result converts
back to a probability:

    post-odds = pre-odds * LR_1 * LR_2 * ... * LR_n
    LR+ = sensitivity / (1 - specificity)
    LR- = (1 - sensitivity) / specificity

Three reasons this framework and not a weighted sum of red flags:

* **Absent evidence counts.** A negative finding multiplies by LR-, which is
  below 1 and therefore actively argues against the condition. A red-flag
  tally can only ever accumulate, so it has no way to rule anything out.
* **Prevalence stays in the arithmetic.** The same evidence means different
  things in a 30-year-old and a 70-year-old, because the priors differ by
  orders of magnitude. Cataract at 25 needs far stronger evidence than cataract
  at 75 to reach the same probability, and this falls out of the maths instead
  of being special-cased.
* **The reasoning is inspectable.** Every condition reports the exact evidence
  that moved it and by how much, so a wrong answer can be traced to the link
  that caused it rather than to "the model said so".

Independence is the standing assumption and it is only approximately true —
reduced acuity and reduced contrast sensitivity are correlated, so multiplying
both LRs double-counts some of the same underlying signal. `MAX_EVIDENCE_LR`
caps how far any one condition can be moved by correlated evidence, which is a
blunt instrument but an honest one; the alternative is a full covariance model
that there is no data to fit.

Provenance of the numbers is marked on every entry: `LIT` where the sensitivity
and specificity come from a published study, `EST` where they are structural
estimates chosen to encode a clinical relationship whose direction is certain
but whose magnitude is not. `EST` values are the ones to replace first when
real labelled data arrives.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Literal

Provenance = Literal["LIT", "EST"]
Urgency = Literal["routine", "soon", "prompt", "urgent"]

#: No single chain of correlated evidence may move a condition by more than
#: this factor, in either direction. Findings within a battery are not
#: independent, and unbounded multiplication would turn three views of one
#: signal into three separate confirmations.
MAX_EVIDENCE_LR = 60.0


@dataclass(frozen=True)
class Evidence:
    """One observation's bearing on one condition."""

    key: str                       # what was observed
    lr: float                      # likelihood ratio applied
    note: str                      # why it bears on this condition
    provenance: Provenance = "EST"

    @property
    def direction(self) -> str:
        return "supports" if self.lr > 1.0 else "argues against"


@dataclass
class ConditionAssessment:
    name: str
    plain_name: str
    probability: float
    prior: float
    urgency: Urgency
    evidence: list[Evidence] = field(default_factory=list)
    what_it_is: str = ""
    what_to_do: str = ""
    limits: str = ""

    @property
    def moved(self) -> float:
        """How far the evidence shifted this from its base rate, as an odds ratio."""
        return _odds(self.probability) / max(_odds(self.prior), 1e-9)

    @property
    def band(self) -> str:
        if self.probability >= 0.60:
            return "likely"
        if self.probability >= 0.25:
            return "possible"
        if self.probability >= 0.08:
            return "less likely"
        return "unlikely"


def _odds(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return p / (1.0 - p)


def _prob(o: float) -> float:
    return o / (1.0 + o)


# --------------------------------------------------------------- prevalence --

def prevalence(condition: str, age: float | None) -> float:
    """Base rate before any measurement, by age where age drives it.

    Age matters enormously here and the curves are steep: cataract is a
    minority finding at 50 and near-universal at 80; presbyopia is essentially
    absent at 35 and universal at 50. Using a flat adult prevalence would make
    the arithmetic wrong in both directions at once.
    """
    a = 45.0 if age is None else age
    table: dict[str, Callable[[float], float]] = {
        # Refractive. Myopia prevalence is rising and varies hugely by region;
        # this is a broad Western adult figure.
        "myopia": lambda x: 0.30 if x >= 18 else 0.05 + 0.02 * max(x - 6, 0),
        "hyperopia": lambda x: 0.10 if x < 40 else 0.25,
        "astigmatism": lambda x: 0.30,
        # Presbyopia is the most predictable condition in medicine.
        "presbyopia": lambda x: 0.02 if x < 38 else min(0.98, 0.05 + 0.09 * (x - 38)),
        # Amblyopia is set in childhood and does not resolve untreated.
        "amblyopia": lambda x: 0.03,
        "strabismus": lambda x: 0.04,
        # Cataract: rare before 50, then climbs steeply.
        # *Visually significant* cataract, not any lens change. Nearly every
        # eye over 70 has some opacity on slit-lamp, but prevalence of the
        # degree that affects vision is roughly 5% at 60, 20% at 70 and 40%
        # at 80. Using the "any opacity" curve would put a 75-year-old at a
        # 60% prior and let two weak findings carry it past 95%.
        "cataract": lambda x: (0.005 if x < 50 else
                               min(0.60, 0.012 * math.exp(0.098 * (x - 50)))),
        # Glaucoma: strongly age-dependent, and asymptomatic until late.
        "glaucoma": lambda x: (0.002 if x < 40 else
                               min(0.12, 0.005 * math.exp(0.075 * (x - 40)))),
        "amd": lambda x: (0.001 if x < 50 else
                          min(0.30, 0.002 * math.exp(0.115 * (x - 50)))),
        # Inherited, sex-linked, and essentially constant with age.
        "colour_vision_deficiency": lambda x: 0.045,
        "convergence_insufficiency": lambda x: 0.07,
        "dry_eye": lambda x: 0.05 + 0.004 * max(x - 30, 0),
        "ptosis": lambda x: 0.01 + 0.003 * max(x - 50, 0),
        "anisocoria": lambda x: 0.20,        # physiological anisocoria is common
        "corneal_arcus": lambda x: (0.02 if x < 45 else
                                    min(0.85, 0.05 * math.exp(0.075 * (x - 45)))),
        "media_opacity_other": lambda x: 0.01,
        "optic_neuropathy": lambda x: 0.003,
        "retinal_detachment_risk": lambda x: 0.002,
    }
    fn = table.get(condition)
    return float(min(max(fn(a) if fn else 0.02, 1e-4), 0.99))


# ------------------------------------------------------------------- catalog --

@dataclass(frozen=True)
class ConditionSpec:
    key: str
    plain_name: str
    urgency: Urgency
    what_it_is: str
    what_to_do: str
    limits: str


CATALOG: dict[str, ConditionSpec] = {
    "myopia": ConditionSpec(
        "myopia", "short sight (myopia)", "routine",
        "Distant things are blurred because the eye focuses light in front of "
        "the retina rather than on it.",
        "An optometrist can confirm it and prescribe lenses. It is not an "
        "illness and it is fully correctable.",
        "This estimates how much blur is present, not the exact lens power."),
    "hyperopia": ConditionSpec(
        "hyperopia", "long sight (hyperopia)", "routine",
        "The eye focuses light behind the retina. Younger eyes can often "
        "compensate by focusing harder, which is why it can hide.",
        "Worth an eye test, especially if close work causes tired eyes or "
        "headaches.",
        "This is the hardest refractive error to detect without dilating "
        "drops, and this screening will miss a real share of cases."),
    "astigmatism": ConditionSpec(
        "astigmatism", "astigmatism", "routine",
        "The eye's surface is shaped more like a rugby ball than a football, "
        "so one direction focuses differently from another.",
        "Correctable with lenses; an optometrist can measure the axis exactly.",
        "The axis found here is approximate."),
    "presbyopia": ConditionSpec(
        "presbyopia", "age-related focusing change (presbyopia)", "routine",
        "The lens inside the eye stiffens with age, so near objects become "
        "hard to focus on. It happens to everyone.",
        "Reading glasses. An optometrist will match the strength to your arms.",
        "Assessed from near vision and age together."),
    "amblyopia": ConditionSpec(
        "amblyopia", "lazy eye (amblyopia)", "soon",
        "One eye sees worse than the other because the brain learned to favour "
        "one during childhood, even though the eye itself is healthy.",
        "Worth an eye examination. Treatment is far more effective the earlier "
        "it starts, so this is not one to leave.",
        "A difference between the eyes has several possible causes; this "
        "cannot tell which."),
    "strabismus": ConditionSpec(
        "strabismus", "eye misalignment (strabismus)", "soon",
        "The two eyes do not point at the same place, so they send the brain "
        "conflicting views.",
        "Worth an examination. In an adult, a misalignment that is new or "
        "came on suddenly should be looked at quickly.",
        "Measured from where a light reflects on each eye; head tilt and "
        "camera angle both affect it."),
    "cataract": ConditionSpec(
        "cataract", "clouding of the lens (cataract)", "soon",
        "The lens inside the eye becomes cloudy, scattering light. It is very "
        "common with age and develops slowly.",
        "An eye examination will confirm it. It is treatable with routine "
        "surgery when it starts to interfere with daily life.",
        "A webcam sees the reflex far less well than an examiner's light, so "
        "this can only raise the question."),
    "glaucoma": ConditionSpec(
        "glaucoma", "glaucoma", "prompt",
        "Damage to the optic nerve, usually from pressure inside the eye. It "
        "takes peripheral vision first and silently — most people do not "
        "notice until a lot is gone.",
        "Worth arranging an eye examination rather than waiting for symptoms. "
        "Sight already lost does not come back, but treatment protects what "
        "remains.",
        "This cannot measure eye pressure or see the optic nerve, which are "
        "the two things the diagnosis actually rests on."),
    "amd": ConditionSpec(
        "amd", "age-related macular degeneration", "prompt",
        "The central part of the retina deteriorates, blurring or distorting "
        "the middle of vision while the edges stay clear.",
        "Straight lines newly appearing bent or a new blur in the centre of "
        "vision should be examined promptly.",
        "Based on a grid test that depends entirely on what you reported "
        "seeing."),
    "colour_vision_deficiency": ConditionSpec(
        "colour_vision_deficiency", "colour vision deficiency", "routine",
        "Certain colours are hard to tell apart, usually reds and greens. It "
        "is almost always inherited and lifelong.",
        "Nothing needs treating. Worth knowing about for some jobs.",
        "A screen cannot show colours accurately enough to classify the type "
        "reliably."),
    "convergence_insufficiency": ConditionSpec(
        "convergence_insufficiency", "difficulty turning the eyes inward", "routine",
        "The eyes struggle to turn inward together for close work, which "
        "causes tired eyes, headaches or doubled text when reading.",
        "Worth mentioning to an optometrist; it responds well to eye exercises.",
        "Estimated from how the eyes track a target approaching the camera."),
    "dry_eye": ConditionSpec(
        "dry_eye", "dry eye", "routine",
        "The tear film breaks up too quickly, leaving the surface exposed. It "
        "causes grittiness, burning, and blurring that clears when you blink.",
        "Usually managed with drops. Worth mentioning if it is persistent.",
        "Inferred from blink behaviour, which many things affect."),
    "ptosis": ConditionSpec(
        "ptosis", "drooping upper eyelid", "soon",
        "The upper lid sits lower than it should, sometimes far enough to "
        "block part of the view.",
        "Worth an examination. A droop that is new, rapid, or comes with "
        "double vision should be seen quickly.",
        "Measured from lid position relative to the pupil."),
    "anisocoria": ConditionSpec(
        "anisocoria", "unequal pupil sizes", "soon",
        "One pupil is larger than the other. A small difference is normal and "
        "present in about a fifth of people.",
        "A long-standing small difference is usually nothing. A new one, "
        "especially with a drooping lid or double vision, should be seen "
        "the same day.",
        "Pupil size is measured from video and is sensitive to lighting."),
    "corneal_arcus": ConditionSpec(
        "corneal_arcus", "pale ring at the edge of the iris", "routine",
        "A deposit of fat at the edge of the cornea. Extremely common with "
        "age and harmless to vision.",
        "Under about 45 it is worth a cholesterol blood test. Later in life it "
        "needs nothing.",
        "Detected from image contrast at the iris edge."),
    "optic_neuropathy": ConditionSpec(
        "optic_neuropathy", "optic nerve problem", "urgent",
        "Damage to the nerve carrying signals from eye to brain, which can "
        "affect colour, contrast and the pupil's response to light.",
        "Reduced vision in one eye with a sluggish pupil warrants prompt "
        "medical attention.",
        "This screening cannot see the optic nerve; it can only notice that "
        "several measures point the same way."),
    "media_opacity_other": ConditionSpec(
        "media_opacity_other", "something blocking light inside the eye", "prompt",
        "Light is being scattered or blocked somewhere between the front of "
        "the eye and the retina.",
        "Worth an examination to find out where and why.",
        "The reflex test locates nothing; it only notices that light did not "
        "come back as expected."),
    "retinal_detachment_risk": ConditionSpec(
        "retinal_detachment_risk", "retinal warning signs", "urgent",
        "The retina can pull away from the back of the eye. It announces "
        "itself with sudden flashes, a shower of new floaters, or a shadow "
        "moving across vision.",
        "These specific symptoms are an emergency — same-day assessment, not "
        "an appointment in a few weeks.",
        "Based entirely on symptoms reported, not on anything measured."),
}


# ------------------------------------------------------------------- engine --

def assess(findings: dict, *, age: float | None = None,
           symptoms: set[str] | None = None) -> list[ConditionAssessment]:
    """Score every condition in the catalog against the findings.

    `findings` is a flat dictionary of what the battery measured — see
    `EVIDENCE_RULES` for the keys consulted. Missing keys simply contribute no
    evidence, so a partial session degrades to weaker conclusions rather than
    to wrong ones.
    """
    symptoms = symptoms or set()
    ctx = {**findings, "age": age, "symptoms": symptoms}

    out: list[ConditionAssessment] = []
    for key, spec in CATALOG.items():
        prior = prevalence(key, age)
        odds = _odds(prior)
        evidence: list[Evidence] = []

        for rule in EVIDENCE_RULES.get(key, []):
            got = rule(ctx)
            if got is None:
                continue
            evidence.append(got)

        # cap the combined swing: findings in one battery are correlated, and
        # unbounded multiplication treats three views of one signal as three
        # independent confirmations
        combined = 1.0
        for e in evidence:
            combined *= e.lr
        combined = max(1.0 / MAX_EVIDENCE_LR, min(MAX_EVIDENCE_LR, combined))

        prob = _prob(odds * combined)
        out.append(ConditionAssessment(
            name=key, plain_name=spec.plain_name, probability=prob,
            prior=prior, urgency=spec.urgency, evidence=evidence,
            what_it_is=spec.what_it_is, what_to_do=spec.what_to_do,
            limits=spec.limits))

    out.sort(key=lambda c: (c.probability, c.moved), reverse=True)
    return out


#: Pairs that cannot both be true. When the evidence cannot separate them --
#: which is the normal case, since acuity measures how blurred vision is and
#: not which direction -- reporting both is worse than reporting neither: it
#: reads as the system contradicting itself, and it is the kind of output that
#: makes a reader discount everything else on the page.
EXCLUSIVE_PAIRS = (("myopia", "hyperopia"),)

MERGED_SPEC = ConditionSpec(
    "refractive_error", "a focusing error (direction not determined)", "routine",
    "Vision is blurred in a way that lenses would correct, but this screening "
    "could not tell whether the eye focuses light in front of the retina "
    "(short sight) or behind it (long sight).",
    "An optometrist settles this in about a minute and can prescribe lenses.",
    "Telling the two apart needs a measurement this cannot make reliably — "
    "blur looks the same in both directions.")


def _collapse_exclusive(scored: list[ConditionAssessment],
                        sign_known: bool = False) -> list[ConditionAssessment]:
    """Merge mutually exclusive conditions the evidence cannot separate.

    Separation requires a signal that can actually see the difference, and for
    myopia versus hyperopia that means a signed measurement — photorefraction.
    Odds alone are not enough: acuity has a much higher likelihood ratio for
    myopia than for hyperopia (10.2 against 3.5, O'Donoghue 2012), so identical
    blur in both eyes will always push myopia further ahead. Reading that gap
    as "the evidence distinguished them" would let an asymmetry in the *test's*
    sensitivity masquerade as a finding about the *eye*, and the report would
    confidently name a direction nothing in the session measured.
    """
    by_name = {c.name: c for c in scored}
    out = list(scored)
    for a_name, b_name in EXCLUSIVE_PAIRS:
        a, b = by_name.get(a_name), by_name.get(b_name)
        if not a or not b:
            continue
        hi, lo = (a, b) if a.probability >= b.probability else (b, a)
        # only a signed measurement can tell these apart
        decisive = sign_known and _odds(hi.probability) >= 3.0 * _odds(lo.probability)
        out = [c for c in out if c.name not in (a_name, b_name)]
        if decisive:
            out.append(hi)
        else:
            merged = ConditionAssessment(
                name=MERGED_SPEC.key, plain_name=MERGED_SPEC.plain_name,
                probability=_prob(_odds(hi.probability) + _odds(lo.probability)),
                prior=max(hi.prior, lo.prior), urgency=MERGED_SPEC.urgency,
                evidence=list(hi.evidence),
                what_it_is=MERGED_SPEC.what_it_is,
                what_to_do=MERGED_SPEC.what_to_do, limits=MERGED_SPEC.limits)
            out.append(merged)
    out.sort(key=lambda c: (c.probability, c.moved), reverse=True)
    return out


def differential(findings: dict, *, age: float | None = None,
                 symptoms: set[str] | None = None,
                 min_probability: float = 0.08) -> list[ConditionAssessment]:
    """The subset worth showing: conditions the evidence actually raised.

    A condition that merely sits at its base rate is not a finding — listing it
    would pad the report with everything in the catalog and bury the two things
    that matter. Anything urgent is kept at a lower bar, because the cost of
    dropping it is not symmetric with the cost of mentioning it.
    """
    scored = _collapse_exclusive(
        assess(findings, age=age, symptoms=symptoms),
        sign_known=bool(findings.get("refraction_sign_known")))
    keep = []
    for c in scored:
        raised = c.moved > 1.2
        if c.urgency in ("urgent", "prompt"):
            if c.probability >= min_probability / 2 and raised:
                keep.append(c)
        elif c.probability >= min_probability and raised:
            keep.append(c)
    return keep


# ---------------------------------------------------------------- the rules --
#
# Each rule reads the findings dictionary and returns an Evidence, or None when
# the relevant test was not done. Likelihood ratios are derived from published
# sensitivity/specificity where those exist, and marked EST otherwise.
#
# The single richest source here is O'Donoghue et al. 2012 (PLoS ONE 7:e34441,
# n=1,053), which measured how well uncorrected acuity detects each refractive
# error against cycloplegic autorefraction. Its numbers encode the asymmetry
# that matters most in this battery: acuity finds myopia well and hyperopia
# badly, because a young hyperope focuses through the error.

def _lr(sens: float, spec: float) -> tuple[float, float]:
    """(LR+, LR-) from a sensitivity/specificity pair."""
    return (sens / max(1 - spec, 1e-3), (1 - sens) / max(spec, 1e-3))


# O'Donoghue 2012, 0.20 logMAR referral cut-off, cycloplegic reference
MYOPIA_LR_POS, MYOPIA_LR_NEG = _lr(0.92, 0.91)        # LIT
HYPEROPIA_LR_POS, HYPEROPIA_LR_NEG = _lr(0.47, 0.865)  # LIT (41-54% / 84-89%)
ASTIG_LR_POS, ASTIG_LR_NEG = _lr(0.62, 0.85)          # LIT (50-74% / 85%)

REDUCED_ACUITY_LOGMAR = 0.20      # the cut-off those figures were measured at


def _get(ctx: dict, *keys):
    for k in keys:
        v = ctx.get(k)
        if v is not None:
            return v
    return None


def _worse_acuity(ctx) -> float | None:
    vals = [ctx.get("acuity_logmar_left"), ctx.get("acuity_logmar_right")]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else ctx.get("acuity_logmar")


def _better_acuity(ctx) -> float | None:
    """The good eye, which is what bears on a *bilateral* diagnosis.

    Uncorrected refractive error is usually close to symmetric, so the
    right question for myopia is "are both eyes blurred?" — and the better
    eye answers it. Taking the worse eye instead makes every one-eyed
    problem look like short sight: a person seeing 0.60 logMAR in one eye
    and 0.05 in the other has something wrong with one eye, not a focusing
    error in both. Anisometropia is the genuine exception, and it is
    covered by the interocular-difference rule pointing at amblyopia,
    which is the finding that actually needs acting on.
    """
    vals = [ctx.get("acuity_logmar_left"), ctx.get("acuity_logmar_right")]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else ctx.get("acuity_logmar")


def _rule_reduced_acuity(condition: str, lr_pos: float, lr_neg: float,
                         provenance: Provenance = "LIT", eye: str = "better"):
    def rule(ctx):
        a = _better_acuity(ctx) if eye == "better" else _worse_acuity(ctx)
        if a is None:
            return None
        label = "both eyes" if eye == "better" else "worse eye"
        if a >= REDUCED_ACUITY_LOGMAR:
            return Evidence(f"reduced distance acuity ({label})", lr_pos,
                            f"uncorrected acuity {a:.2f} logMAR in {label}, at "
                            f"or past the {REDUCED_ACUITY_LOGMAR:.2f} referral "
                            "cut-off", provenance)
        return Evidence(f"normal distance acuity ({label})", lr_neg,
                        f"uncorrected acuity {a:.2f} logMAR in {label}",
                        provenance)
    return rule


def _rule_refraction_sign(condition: str, want_negative: bool):
    """The signed estimate, when photorefraction supplied a direction."""
    def rule(ctx):
        se = _get(ctx, "refraction_se", "refraction_se_worse")
        if se is None or not ctx.get("refraction_sign_known"):
            return None
        conf = ctx.get("refraction_confidence", "broad")
        strength = {"indicative": 6.0, "broad": 2.5}.get(conf, 1.2)
        matches = (se <= -0.75) if want_negative else (se >= 0.75)
        if matches:
            return Evidence("signed refraction estimate", strength,
                            f"estimated {se:+.2f} D spherical equivalent", "EST")
        if abs(se) < 0.5:
            return Evidence("refraction near neutral", 1.0 / strength,
                            f"estimated {se:+.2f} D, close to neutral", "EST")
        return Evidence("refraction points the other way", 1.0 / strength,
                        f"estimated {se:+.2f} D", "EST")
    return rule


def _rule_cylinder(ctx):
    cyl = _get(ctx, "refraction_cylinder")
    dial = ctx.get("dial_detected")
    if cyl is None and dial is None:
        return None
    if dial is True or (cyl is not None and cyl >= 0.75):
        return Evidence("meridian preference on the dial", ASTIG_LR_POS,
                        "one meridian appeared consistently sharper than "
                        "the other", "LIT")
    if dial is False:
        return Evidence("no meridian preference", ASTIG_LR_NEG,
                        "all spokes of the dial looked equally sharp", "LIT")
    return None


def _rule_interocular_difference(ctx):
    """Two lines or more between the eyes is the classic amblyopia signal."""
    l, r = ctx.get("acuity_logmar_left"), ctx.get("acuity_logmar_right")
    if l is None or r is None:
        return None
    diff = abs(l - r)
    if diff >= 0.20:
        return Evidence("difference between the eyes", 8.0,
                        f"{diff:.2f} logMAR between eyes — two lines or more",
                        "EST")
    return Evidence("eyes match", 0.35,
                    f"only {diff:.2f} logMAR between the eyes", "EST")


def _rule_alignment(ctx):
    """AAPOS 2021 refers manifest strabismus above 8 prism dioptres."""
    pd = ctx.get("alignment_pd")
    if pd is None:
        return None
    if pd >= 8.0:
        return Evidence("eyes not aligned", 12.0,
                        f"corneal reflections offset by about {pd:.0f} prism "
                        "dioptres; the referral threshold is 8", "LIT")
    return Evidence("eyes aligned", 0.25,
                    f"offset about {pd:.0f} prism dioptres, within normal", "LIT")


def _rule_contrast(ctx):
    """Contrast loss out of proportion to acuity points at the media or the
    optic nerve rather than at a focusing error."""
    cs = ctx.get("contrast_logcs")
    if cs is None:
        return None
    if cs < 1.35:
        return Evidence("reduced contrast sensitivity", 4.0,
                        f"{cs:.2f} log contrast sensitivity, below the normal "
                        "range of about 1.65 and up", "EST")
    return Evidence("normal contrast sensitivity", 0.45,
                    f"{cs:.2f} log contrast sensitivity", "EST")


def _rule_colour(ctx):
    n = ctx.get("colour_plate_errors")
    if n is None:
        return None
    if n >= 3:
        return Evidence("colour plate errors", 20.0,
                        f"{n} plates misread", "EST")
    return Evidence("colour plates read correctly", 0.15,
                    f"only {n} plate(s) misread", "EST")


def _rule_amsler(ctx):
    d = ctx.get("amsler_distortion")
    if d is None:
        return None
    if d:
        return Evidence("distortion on the grid", 15.0,
                        "straight lines on the grid were reported as bent, "
                        "wavy or missing", "EST")
    return Evidence("grid looked normal", 0.4,
                    "the grid lines looked straight and complete", "EST")


def _rule_field_defect(ctx):
    d = ctx.get("field_defect")
    if d is None:
        return None
    if d:
        return Evidence("missed targets in peripheral vision", 10.0,
                        "targets away from the centre were missed repeatedly",
                        "EST")
    return Evidence("peripheral vision intact", 0.5,
                    "peripheral targets were seen", "EST")


def _rule_anisocoria(ctx):
    """Lam 1987: a difference of 0.4 mm or more is present in about a fifth of
    normal people, so only a larger difference carries information."""
    mm = ctx.get("anisocoria_mm")
    if mm is None:
        return None
    if mm >= 1.0:
        return Evidence("unequal pupils", 9.0,
                        f"{mm:.1f} mm difference between pupils", "LIT")
    return Evidence("pupils equal", 0.3, f"{mm:.1f} mm difference", "LIT")


def _rule_mrd1(ctx):
    v = ctx.get("mrd1_min_mm")
    if v is None:
        return None
    if v < 2.0:
        return Evidence("low upper lid", 14.0,
                        f"lid sits {v:.1f} mm above the pupil centre; below "
                        "2 mm is the usual threshold", "LIT")
    return Evidence("lid height normal", 0.2, f"{v:.1f} mm above pupil centre",
                    "LIT")


def _rule_arcus(ctx):
    v = ctx.get("arcus_contrast")
    if v is None:
        return None
    if v > 0.18:
        return Evidence("pale ring at the iris edge", 8.0,
                        "a lighter annulus was measured at the corneal "
                        "periphery", "EST")
    return Evidence("no ring at the iris edge", 0.3, "no annulus measured", "EST")


def _rule_reflex(ctx):
    """Deliberately weak. A webcam has no flash, and the one purpose-built app
    that tried this scored 15% sensitivity in independent validation, so the
    evidence is capped low in both directions -- it can nudge, never decide."""
    a = ctx.get("reflex_asymmetry")
    if a is None:
        return None
    if a > 0.35:
        return Evidence("uneven reflex between the eyes", 2.0,
                        f"one eye returned {a:.0%} less light than the other",
                        "EST")
    return Evidence("reflex looked even", 0.9,
                    "no difference between the eyes was visible, though this "
                    "check cannot rule anything out", "EST")


def _rule_near_acuity(ctx):
    """Presbyopia: near blurred while distance is not."""
    near = ctx.get("near_acuity_logmar")
    dist = _better_acuity(ctx)
    if near is None:
        return None
    if near >= 0.30 and (dist is None or dist < 0.20):
        return Evidence("near vision blurred, distance clear", 12.0,
                        f"near acuity {near:.2f} logMAR with distance vision "
                        "unaffected", "EST")
    if near < 0.20:
        return Evidence("near vision clear", 0.15,
                        f"near acuity {near:.2f} logMAR", "EST")
    return None


def _rule_npc(ctx):
    """Near point of convergence beyond about 10 cm is the standard cut-off."""
    cm = ctx.get("npc_cm")
    if cm is None:
        return None
    if cm > 10.0:
        return Evidence("eyes lose convergence early", 10.0,
                        f"the eyes stopped converging at {cm:.0f} cm; under "
                        "10 cm is normal", "EST")
    return Evidence("convergence normal", 0.25, f"converged to {cm:.0f} cm",
                    "EST")


def _rule_blink(ctx):
    r = ctx.get("blink_rate_per_min")
    if r is None:
        return None
    if r > 22:
        return Evidence("frequent blinking", 3.0,
                        f"{r:.0f} blinks per minute, above the usual 10-20",
                        "EST")
    return Evidence("blink rate normal", 0.6, f"{r:.0f} blinks per minute", "EST")


def _rule_sclera_yellow(ctx):
    v = ctx.get("sclera_yellowness")
    if v is None or not ctx.get("sclera_colour_calibrated"):
        return None
    if v > 18.0:
        return Evidence("yellow tinge to the sclera", 6.0,
                        "the whites measured yellower than usual against a "
                        "white reference", "EST")
    return Evidence("sclera colour normal", 0.5, "no yellow tinge measured", "EST")


def _rule_symptom(condition: str, *terms: str, lr: float = 12.0):
    def rule(ctx):
        s = ctx.get("symptoms") or set()
        hit = [t for t in terms if t in s]
        if hit:
            return Evidence("reported symptom", lr,
                            "reported: " + ", ".join(t.replace("_", " ")
                                                     for t in hit), "EST")
        return None
    return rule


def _rule_anemia_probability(ctx):
    p = ctx.get("anemia_probability")
    if p is None:
        return None
    if p >= 0.5:
        return Evidence("pale conjunctiva", 3.0,
                        f"conjunctival colour scored {p:.0%} toward the pale "
                        "end", "LIT")
    return Evidence("conjunctiva well coloured", 0.6,
                    f"conjunctival colour scored {p:.0%} toward the pale end",
                    "LIT")


EVIDENCE_RULES: dict[str, list] = {
    "myopia": [
        _rule_reduced_acuity("myopia", MYOPIA_LR_POS, MYOPIA_LR_NEG),
        _rule_refraction_sign("myopia", want_negative=True),
    ],
    "hyperopia": [
        _rule_reduced_acuity("hyperopia", HYPEROPIA_LR_POS, HYPEROPIA_LR_NEG),
        _rule_refraction_sign("hyperopia", want_negative=False),
    ],
    "astigmatism": [_rule_cylinder],
    "presbyopia": [_rule_near_acuity],
    "amblyopia": [_rule_interocular_difference, _rule_alignment],
    "strabismus": [_rule_alignment,
                   _rule_symptom("strabismus", "double_vision", lr=6.0)],
    "cataract": [_rule_contrast, _rule_reflex,
                 _rule_symptom("cataract", "glare", "haloes", lr=4.0)],
    "glaucoma": [_rule_field_defect, _rule_contrast],
    "amd": [_rule_amsler,
            _rule_symptom("amd", "central_blur", "distorted_lines", lr=8.0)],
    "colour_vision_deficiency": [_rule_colour],
    "convergence_insufficiency": [
        _rule_npc, _rule_symptom("convergence_insufficiency",
                                 "eye_strain_reading", "double_vision", lr=4.0)],
    "dry_eye": [_rule_blink,
                _rule_symptom("dry_eye", "grittiness", "burning", lr=8.0)],
    "ptosis": [_rule_mrd1],
    "anisocoria": [_rule_anisocoria],
    "corneal_arcus": [_rule_arcus],
    "optic_neuropathy": [_rule_contrast, _rule_colour,
                         _rule_interocular_difference],
    "media_opacity_other": [_rule_reflex, _rule_contrast],
    "retinal_detachment_risk": [
        _rule_symptom("retinal_detachment_risk", "sudden_flashes",
                      "new_floaters", "curtain_shadow", lr=40.0)],
}


# ------------------------------------------------------- bridge from findings --

def findings_to_context(findings, *, eye_key: str | None = None) -> dict:
    """Flatten a list of `Finding` objects into the keys the rules read.

    Deliberately conservative in two ways. A finding whose tier is
    "inconclusive" contributes nothing at all — an unusable measurement is not
    a normal one, and letting it through as a negative would generate false
    reassurance from a test that never worked. And an unrecognised module is
    skipped rather than guessed at, so adding a module cannot silently change
    an existing conclusion until it is wired in here on purpose.
    """
    ctx: dict = {}

    def m(f, *keys):
        for k in keys:
            v = f.metrics.get(k)
            if v is not None:
                return v
        return None

    for f in findings:
        if f.tier == "inconclusive":
            continue
        flags = [str(x).lower() for x in (f.metrics.get("flags") or [])]
        mod = f.module

        if mod == "acuity":
            v = m(f, "logmar")
            if v is not None:
                ctx.setdefault("acuity_logmar", v)
                side = str(m(f, "eye") or "").lower()
                if "left" in side:
                    ctx["acuity_logmar_left"] = v
                elif "right" in side:
                    ctx["acuity_logmar_right"] = v
            near = m(f, "near_logmar")
            if near is not None:
                ctx["near_acuity_logmar"] = near

        elif mod == "contrast":
            v = m(f, "log_cs")
            if v is not None:
                ctx["contrast_logcs"] = v

        elif mod == "astigmatism":
            # "no meridian preference" is informative; a failed test is not,
            # and the tier filter above has already removed the latter
            ctx["dial_detected"] = any("astigmat" in x for x in flags)
            ax = m(f, "axis_deg")
            if ax is not None:
                ctx["dial_axis"] = ax

        elif mod == "alignment":
            v = m(f, "deviation_pd")
            if v is not None:
                ctx["alignment_pd"] = v

        elif mod == "color_vision":
            v = m(f, "errors_total")
            if v is not None:
                ctx["colour_plate_errors"] = v

        elif mod == "amsler":
            marks = m(f, "distortion_marks")
            missing = m(f, "missing_marks")
            if marks is not None or missing is not None:
                ctx["amsler_distortion"] = bool((marks or 0) + (missing or 0))

        elif mod == "pupillometry":
            v = m(f, "asymmetry_mm")
            if v is not None:
                ctx["anisocoria_mm"] = v

        elif mod == "photorefraction":
            se = m(f, "sphere_d", "sphere")
            if se is not None:
                ctx["refraction_se"] = se
                ctx["refraction_sign_known"] = f.tier == "measured"
                ctx["refraction_confidence"] = (
                    "indicative" if f.tier == "measured" else "broad")
            cyl = m(f, "cylinder_d", "cylinder")
            if cyl is not None:
                ctx["refraction_cylinder"] = abs(cyl)

        elif mod == "eyelid position":
            vals = [m(f, "left_mrd1_mm"), m(f, "right_mrd1_mm")]
            vals = [v for v in vals if v is not None]
            if vals:
                ctx["mrd1_min_mm"] = min(vals)

        elif mod == "corneal arcus":
            vals = [m(f, "left_arcus_contrast"), m(f, "right_arcus_contrast")]
            vals = [v for v in vals if v is not None]
            if vals:
                ctx["arcus_contrast"] = max(vals)

        elif mod == "red reflex":
            v = m(f, "reflex_asymmetry")
            if v is not None:
                ctx["reflex_asymmetry"] = v

        elif mod == "sclera appearance":
            vals = [m(f, "left_yellowness"), m(f, "right_yellowness")]
            vals = [v for v in vals if v is not None]
            if vals and "uncalibrated" not in flags:
                ctx["sclera_yellowness"] = max(vals)
                ctx["sclera_colour_calibrated"] = True

    return ctx


def differential_from_findings(findings, *, age: float | None = None,
                               symptoms: set[str] | None = None,
                               extra: dict | None = None):
    """Convenience path: findings in, ranked conditions out."""
    ctx = findings_to_context(findings)
    if extra:
        ctx.update(extra)
    return differential(ctx, age=age, symptoms=symptoms)
