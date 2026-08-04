from benchmarks.bench_module3 import run_benchmark


def test_benchmark_recovers_refraction():
    result = run_benchmark()
    assert result["clean"]["mean_abs_sphere_error_d"] < 0.5
    assert result["noisy"]["mean_abs_sphere_error_d"] < 0.75
    assert result["clean"]["detection_rate"] > 0.95
