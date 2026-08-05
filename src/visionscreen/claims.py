"""What this system is allowed to say, enforced rather than promised.

The regulatory position for a consumer-facing tool that interprets images of
the eye is narrower than it first appears, and the narrowness is the reason
this module exists as code instead of a paragraph in a README.

Three findings drive the design, all from primary sources:

* **"Screening" is not a safer word than "diagnosis."** FD&C 201(h) reaches
  the "diagnosis of disease *or other conditions*", and FDA's General Wellness
  guidance lists screening alongside diagnosis among the intended uses that
  disqualify a product from the wellness category. The device classification
  for retinal software (21 CFR 886.1100) uses the phrase "diagnostic
  screening" for itself.

* **Emitting a clinical value is itself a regulated act.** The wellness
  guidance excludes products whose outputs "mimic those used clinically unless
  validated". When Visibly finally obtained clearance (K220090, 2022) for a
  web acuity test, the cleared labelling permitted only a binary output —
  consistent or not consistent with normal vision — not a Snellen fraction.

* **A disclaimer does not change intended use.** Opternative's online
  refraction drew an FDA warning letter in 2017 and a Class 2 recall in 2019
  ("Lack of 510K clearance"). What made it a device was what it produced, not
  what it said about itself.

So the honest engineering is a switch, not a sentence. `ClaimsMode.RESEARCH`
renders everything — dioptres, logMAR, condition names — because that is what
a benchmark report and a developer need, and what this project is.
`ClaimsMode.WELLNESS` is what a build shown to a member of the public would
have to use: no clinical values, no named conditions, no "abnormal", and a
single referral sentence modelled on the guidance's own notification
carve-out.

`assert_wellness_safe` is the enforcement. The test suite runs every
user-facing string this system can emit through it, so a mode violation is a
failing test rather than a discovery made later by someone else.

None of this constitutes legal advice, and clearing the checks here is not the
same as being cleared to distribute anything.
"""
from __future__ import annotations

import re
from enum import Enum


class ClaimsMode(str, Enum):
    #: Full detail. Benchmarks, validation reports, developer output.
    RESEARCH = "research"
    #: What a consumer build would be limited to.
    WELLNESS = "wellness"


# Condition names may not appear in wellness output at all. Naming the thing
# is what converts a general observation into a claim about a disease.
CONDITION_TERMS = (
    "myopia", "myopic", "hyperopia", "hyperopic", "astigmatism", "astigmatic",
    "presbyopia", "cataract", "glaucoma", "amblyopia", "strabismus",
    "retinoblastoma", "leukocoria", "ptosis", "pterygium", "arcus", "icterus",
    "jaundice", "anaemia", "anemia", "conjunctivitis", "keratoconus",
    "macular degeneration", "diabetic retinopathy", "colour blindness",
    "color blindness", "nearsighted", "farsighted", "short sight", "long sight",
    "anisocoria", "nystagmus", "esotropia", "exotropia",
)

# Verbs and framings that assert a medical function.
CLAIM_TERMS = (
    "diagnose", "diagnosis", "diagnostic", "screen for", "screening for",
    "prescription", "prescribe", "abnormal", "pathological", "medical grade",
    "medical-grade", "clinical grade", "clinical-grade", "clinically accurate",
    "replaces an eye exam", "replace an eye exam", "eye exam replacement",
    "as accurate as",
)

# Numeric formats that mimic a clinical measurement.
CLINICAL_VALUE_PATTERNS = (
    re.compile(r"\b20\s*/\s*\d{2,3}\b"),           # Snellen
    re.compile(r"\b6\s*/\s*\d{1,2}\b"),            # metric Snellen
    re.compile(r"[-+]?\d+(?:\.\d+)?\s*(?:D|dioptre|diopter)s?\b", re.I),
    re.compile(r"\blogmar\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:PD|prism dioptres?|prism diopters?)\b", re.I),
    re.compile(r"\blog\s*CS\b", re.I),
)

#: Modelled on the General Wellness guidance's notification carve-out: no named
#: condition, no "abnormal", no threshold, no ongoing monitoring.
REFERRAL_SENTENCE = (
    "Some of your results fell outside the range this tool is designed for. "
    "An evaluation by an eye care professional may be helpful."
)

ALL_CLEAR_SENTENCE = (
    "Your results fell within the range this tool is designed for. Regular "
    "check-ups with an eye care professional are still worthwhile."
)

NOT_A_DEVICE = (
    "This is not a medical device. It does not diagnose or screen for any "
    "disease or eye condition, and it does not replace a comprehensive eye "
    "examination."
)


class ClaimsViolation(AssertionError):
    """Raised when text intended for a consumer build carries a clinical claim."""


# Negation cues. "This does not diagnose disease" is the disclaimer the
# guidance effectively requires, and it necessarily contains the word
# "diagnose"; a gate that rejected it would be unusable.
NEGATION_CUES = (
    "not", "no", "never", "doesn't", "does not", "do not", "don't", "cannot",
    "can't", "won't", "will not", "isn't", "is not", "aren't", "without",
    "neither", "nor", "unable to", "rather than", "instead of",
)
NEGATION_WINDOW_WORDS = 5


def _is_negated(text_low: str, start: int) -> bool:
    """True if a negation cue governs the term beginning at `start`.

    Looks only at the few words immediately before the term, and stops at
    sentence boundaries so a negation in the previous sentence cannot launder
    an assertion in this one.
    """
    before = text_low[:start]
    # a clause or sentence break resets the scope of any earlier negation
    for boundary in (".", ";", " but ", " however ", " although "):
        idx = before.rfind(boundary)
        if idx != -1:
            before = before[idx + len(boundary):]
    words = before.split()[-NEGATION_WINDOW_WORDS:]
    window = " ".join(words)
    return any(re.search(rf"(?:^|\s){re.escape(cue)}(?:\s|$)", window)
               for cue in NEGATION_CUES)


def find_violations(text: str) -> list[str]:
    """Every reason `text` would be unsafe in wellness mode.

    Returns all of them rather than the first, because fixing copy one
    violation at a time is how the last one gets missed.

    Negation exempts a *function* claim but deliberately not a *condition*
    name. "This does not diagnose disease" is a permitted disclaimer; "you do
    not have cataracts" is still a statement about a named condition, and
    ruling a disease out is as much a diagnostic act as ruling it in.
    """
    low = text.lower()
    found: list[str] = []
    for term in CONDITION_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", low):
            found.append(f"names a condition: {term!r}")
    for term in CLAIM_TERMS:
        for m in re.finditer(rf"\b{re.escape(term)}", low):
            if not _is_negated(low, m.start()):
                found.append(f"asserts a medical function: {term!r}")
            break
    for pattern in CLINICAL_VALUE_PATTERNS:
        m = pattern.search(text)
        if m:
            found.append(f"emits a clinical value: {m.group(0)!r}")
    return found


def is_wellness_safe(text: str) -> bool:
    return not find_violations(text)


def assert_wellness_safe(text: str, where: str = "") -> None:
    bad = find_violations(text)
    if bad:
        loc = f" in {where}" if where else ""
        raise ClaimsViolation(
            f"text{loc} is not permissible in wellness mode: "
            + "; ".join(bad) + f"\n  text: {text!r}")


def render(text: str, mode: ClaimsMode, *, fallback: str = REFERRAL_SENTENCE,
           where: str = "") -> str:
    """Return `text` in research mode; in wellness mode, `text` only if it is
    already safe, otherwise the neutral fallback.

    Substituting rather than raising is deliberate: a consumer build must
    degrade to something true and useful, not crash, when a module produces
    detail it is not allowed to show.
    """
    if mode is ClaimsMode.RESEARCH:
        return text
    if is_wellness_safe(text):
        return text
    if not is_wellness_safe(fallback):
        raise ClaimsViolation(
            f"the fallback{' for ' + where if where else ''} is itself unsafe: "
            + "; ".join(find_violations(fallback)))
    return fallback


def value(text: str, mode: ClaimsMode) -> str | None:
    """A clinical number, or nothing at all in wellness mode.

    Used for report table cells: research builds show the measurement, consumer
    builds omit the row entirely rather than showing a rounded or vaguer number,
    which would still be a clinical value.
    """
    return text if mode is ClaimsMode.RESEARCH else None


def summarise_outcome(any_flags: bool, mode: ClaimsMode) -> str:
    """The one sentence a consumer build is permitted to conclude with."""
    if mode is ClaimsMode.RESEARCH:
        return ("One or more measures fell outside the expected range."
                if any_flags else "All measures fell within the expected range.")
    return REFERRAL_SENTENCE if any_flags else ALL_CLEAR_SENTENCE
