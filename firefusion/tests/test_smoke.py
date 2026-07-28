from pathlib import Path

import pandas as pd

from firefusion.benchmark import run_benchmark
from firefusion.example import generate_example
from firefusion.transfer import run_transfer


def test_cpu_end_to_end(tmp_path: Path):
    data = generate_example(tmp_path / "data")
    config = Path(__file__).parents[1] / "configs" / "smoke.json"
    benchmark = run_benchmark(config, data, tmp_path / "benchmark")
    transfer = run_transfer(config, data, tmp_path / "transfer")
    benchmark_metrics = pd.read_csv(benchmark / "split_metrics.csv")
    transfer_metrics = pd.read_csv(transfer / "split_metrics.csv")
    assert {"RMSE", "R2", "AUC"} <= set(benchmark_metrics.metric)
    assert set(transfer_metrics.metric) == {"RMSE", "R2"}
    assert "transfer/bayesian_two_expert" in set(transfer_metrics.method)
