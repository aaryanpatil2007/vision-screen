"""Worth four-dot test: binocular fusion, suppression, and diplopia.

The classical target is a diamond of four lights — one red at the top, two
green at the sides, one white at the bottom — viewed through a red filter over
the right eye and green over the left. The white dot is visible to both eyes,
so what the patient reports separates the possibilities cleanly:

* four lights          -> fusion
* five, reds to the right of the greens -> uncrossed diplopia (esotropic)
* five, reds to the left                -> crossed diplopia (exotropic)
* two red only         -> the green-filtered (left) eye is suppressed
* three green only     -> the red-filtered (right) eye is suppressed

**Why both distances.** At 6 m the target subtends a small angle that falls
inside a central suppression scotoma, so suppression shows up. At 33 cm the
same target subtends a much larger angle, reaching peripheral retina outside
the scotoma, and the same patient may fuse. A near-only test therefore misses
suppression; the direction of that effect is standard, so both are asked.

**Screen validity.** A tablet implementation of this test agreed closely with
the physical flashlight on suppression-zone size (ICC ~0.97), so the format
transfers. The dissociation, however, depends on anaglyph filter quality: good
gel filters leak ~13.5% into the red eye, and inkjet-printed filters leak 92%,
at which point the test measures nothing. Results are therefore capped at
`weak-signal`.
"""
from __future__ import annotations

from visionscreen.report import Finding

# Red filter is worn over the RIGHT eye by convention.
WORTH_RESPONSES = {
    "four": {"fusion": True, "suppression": None, "diplopia": None},
    "five_uncrossed": {"fusion": False, "suppression": None, "diplopia": "esotropic"},
    "five_crossed": {"fusion": False, "suppression": None, "diplopia": "exotropic"},
    # only the two RED lights: the red filter passes them, so the eye behind
    # the GREEN filter (left) is contributing nothing
    "two_red": {"fusion": False, "suppression": "left", "diplopia": None},
    "three_green": {"fusion": False, "suppression": "right", "diplopia": None},
}


def interpret_worth_response(response: str) -> dict:
    return WORTH_RESPONSES[response]


def score_suppression(responses: dict[str, str], valid_fraction: float) -> Finding:
    """responses: {"near": <response>, "far": <response>}"""
    known = {k: v for k, v in responses.items() if v in WORTH_RESPONSES}
    if valid_fraction < 0.4 or not known:
        return Finding(
            module="suppression",
            summary="Binocular fusion test was not completed.",
            tier="inconclusive",
            retakes=["Repeat the four-dot test wearing the red-cyan glasses, "
                     "red lens over your right eye."],
        )

    interp = {k: interpret_worth_response(v) for k, v in known.items()}
    flags: list[str] = []
    suppressing = next((i["suppression"] for i in interp.values() if i["suppression"]), None)
    diplopia = next((i["diplopia"] for i in interp.values() if i["diplopia"]), None)

    if suppressing:
        flags.append("suppression of one eye")
    if diplopia:
        flags.append("double vision (diplopia)")

    near_fused = interp.get("near", {}).get("fusion")
    far_fused = interp.get("far", {}).get("fusion")

    if suppressing:
        side = "right" if suppressing == "right" else "left"
        summary = (
            f"One eye's image was not seen — the {side} eye appears to be suppressed. "
            "Suppression is how the brain avoids double vision when the eyes are "
            "misaligned, and it is the mechanism behind amblyopia."
        )
        if near_fused and far_fused is False:
            summary += (
                " It appeared at distance but not at near, which is the usual pattern: "
                "the distance target is smaller and falls inside the suppressed area."
            )
    elif diplopia:
        summary = (
            f"The lights were seen doubled ({diplopia} pattern), meaning the two eyes "
            "were not pointing at the same place."
        )
    else:
        summary = "Both eyes contributed and the images fused normally."

    return Finding(
        module="suppression",
        summary=summary + " Screen-based dissociation depends on the filter quality "
                          "of the glasses; treat as a rough screen.",
        tier="weak-signal",
        metrics={
            "flags": flags,
            "suppressing_eye": suppressing,
            "diplopia": diplopia,
            "near_response": known.get("near"),
            "far_response": known.get("far"),
        },
    )
