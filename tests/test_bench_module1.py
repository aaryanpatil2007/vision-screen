import numpy as np

from benchmarks.bench_module1 import run_benchmark, simulate_observer


def test_simulated_observer_produces_trials():
    rng = np.random.default_rng(0)
    trials = simulate_observer(true_logmar=0.4, lapse=0.05, rng=rng)
    assert len(trials) >= 8
    assert {"logmar", "shown", "answered"} <= set(trials[0])


def test_benchmark_recovers_acuity():
    result = run_benchmark(n_observers=50, seed=7)
    assert result["n"] == 50
    assert result["mean_abs_error"] < 0.15  # logMAR — the headline Module 1 number
