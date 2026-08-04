import math

import pytest

from visionscreen.modules.acuity import (
    MAX_TRIALS, Staircase, letter_height_px, score_trials,
)


def test_letter_height_physics():
    # logMAR 0.0 at 50 cm: 5 arcmin → height = 2*50*tan(2.5') ≈ 0.0727 cm
    px = letter_height_px(0.0, distance_cm=50.0, px_per_cm=37.8)
    assert px == pytest.approx(2 * 50 * math.tan(math.radians(5 / 60 / 2)) * 37.8, rel=1e-3)
    # one logMAR unit = 10x the size
    assert letter_height_px(1.0, 50.0, 37.8) == pytest.approx(px * 10, rel=1e-3)


def simulate(true_logmar: float) -> Staircase:
    s = Staircase()
    while not s.done:
        s.record(s.current() >= true_logmar)  # ideal observer: correct iff letter big enough
    return s


def test_staircase_converges_to_true_threshold():
    s = simulate(0.4)
    assert s.threshold() == pytest.approx(0.4, abs=0.15)


def test_staircase_always_terminates_at_the_cap():
    """Whatever the responses, the run must end at the trial cap."""
    s = Staircase()
    for _ in range(MAX_TRIALS):
        if s.done:
            break
        s.record(False)
    assert s.done


def make_trials(n: int, logmar: float = 0.3) -> list[dict]:
    return [{"logmar": logmar, "shown": "up", "answered": "up"} for _ in range(n)]


def test_floor_reported_as_at_or_better():
    trials = make_trials(20, logmar=-0.3)  # perfect run down at the floor
    f = score_trials(trials)
    assert "at or better than" in f.summary
    assert f.metrics["logmar"] == -0.3


def test_score_trials_tiers():
    assert score_trials(make_trials(20)).tier == "measured"
    assert score_trials(make_trials(10)).tier == "weak-signal"
    f = score_trials(make_trials(3))
    assert f.tier == "inconclusive"
    assert f.metrics == {} or "logmar" not in f.metrics


def test_staircase_targets_guessing_corrected_threshold():
    """Kaernbach weighted up-down converges at p = S_up/(S_up+S_down). With a
    4-alternative task the 0.1/0.2 pair lands at 55.6% *corrected*, a criterion
    stricter than threshold that reports acuity optimistically."""
    from visionscreen.modules.acuity import (
        STEP_DOWN, STEP_UP, convergence_probability, corrected_convergence,
    )
    assert convergence_probability() == pytest.approx(0.625, abs=1e-6)
    assert corrected_convergence() == pytest.approx(0.50, abs=1e-3)
    # the old 0.2 step would have been biased
    assert corrected_convergence(step_up=0.2) == pytest.approx(0.556, abs=1e-3)
    assert STEP_UP / STEP_DOWN == pytest.approx(5 / 3, abs=1e-9)


def test_display_floor_matches_pixel_geometry():
    """A 1080p 24-inch laptop at 40 cm cannot render logMAR 0.0."""
    from visionscreen.modules.acuity import letter_height_px, renderable_floor_logmar

    px_per_cm = 1 / 0.02767   # 24" 1080p, 0.2767 mm pitch
    floor = renderable_floor_logmar(distance_cm=40, px_per_cm=px_per_cm)
    assert floor > 0.0, f"floor {floor:.2f} should be coarser than logMAR 0"
    # at the floor exactly, the stroke is one pixel
    assert letter_height_px(floor, 40, px_per_cm) == pytest.approx(5.0, abs=0.05)
    # a 460 ppi phone at 30 cm does much better but STILL cannot reach -0.3:
    # that needs ~38 cm on the same panel
    phone = renderable_floor_logmar(30, 1 / 0.0055)
    assert phone < 0.0
    assert phone == pytest.approx(-0.20, abs=0.03)
    assert renderable_floor_logmar(40, 1 / 0.0055) < -0.3


def test_display_limited_acuity_reported_as_a_bound():
    trials = make_trials(20, logmar=-0.1)
    f = score_trials(trials, display_floor=-0.12)
    assert "at or better than" in f.summary.lower()
    assert "screen can draw" in f.summary
    assert f.metrics["display_floor_logmar"] == -0.12


def test_etdrs_equivalent_reported():
    """Tumbling E reads ~0.15 logMAR conservative vs an ETDRS letter chart."""
    f = score_trials(make_trials(20, logmar=0.30))
    assert f.metrics["logmar"] == 0.3
    assert f.metrics["etdrs_equivalent_logmar"] == pytest.approx(0.15, abs=0.01)


def test_step_halving_preserves_convergence_criterion():
    """Halving both steps must not move the threshold criterion."""
    from visionscreen.modules.acuity import (
        STEP_DOWN, STEP_UP, corrected_convergence,
    )
    full = corrected_convergence(STEP_UP, STEP_DOWN)
    halved = corrected_convergence(STEP_UP * 0.5, STEP_DOWN * 0.5)
    assert full == pytest.approx(halved, abs=1e-12)


def test_reversal_scoring_survives_non_grid_levels():
    """With a 5/3 step ratio, levels do not repeat — the estimator must not
    depend on grouping trials by identical level (the bug this replaced)."""
    trials, level, last = [], 1.0, None
    for i in range(20):
        correct = level >= 0.4
        trials.append({"logmar": round(level, 4), "shown": "up",
                       "answered": "up" if correct else "down"})
        if last is not None and correct != last:
            pass
        last = correct
        level += (-0.1 if correct else 0.1 * 5 / 3)
    f = score_trials(trials)
    assert f.metrics["logmar"] == pytest.approx(0.4, abs=0.12)


def test_trial_budget_set_for_repeatability():
    """The budget is chosen from a measured repeatability sweep, not taste:
    30 trials gave CoR 0.212 logMAR, 60 gives 0.137 — inside the 0.15
    clinical bar and near the ETDRS chart's own 0.11."""
    from visionscreen.modules.acuity import MAX_REVERSALS, MAX_TRIALS, MIN_TRIALS
    assert MAX_TRIALS >= 50
    assert MAX_REVERSALS >= 12
    assert MIN_TRIALS >= 20
    assert MIN_TRIALS < MAX_TRIALS


def test_staircase_uses_the_full_budget_when_needed():
    """A noisy observer must be allowed to run to the trial cap."""
    s = Staircase()
    n = 0
    import random
    rng = random.Random(0)
    while not s.done and n < 200:
        s.record(rng.random() < 0.6)
        n += 1
    assert n >= 20, "terminated before the minimum trial count"
    assert n <= MAX_TRIALS
