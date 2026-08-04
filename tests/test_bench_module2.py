from benchmarks.bench_module2 import run_benchmark


def test_benchmark_recovers_deviation():
    result = run_benchmark(seeds=5)
    assert result["detection_rate"] > 0.95
    assert result["mean_abs_error_pd"] < 3.0  # headline Module 2 number
