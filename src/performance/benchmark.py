"""Latency, memory, throughput, and scalability benchmarks."""

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any
import numpy as np
import pandas as pd
import psutil
import os

from src.dataset.loader import TARGETS, load_dataset
from src.dataset.split import official_split
from src.surrogate.hybrid import HybridPhysicsMLModel
from src.surrogate.train import create_model


@dataclass
class BenchmarkResult:
    name: str
    kind: str
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_ops_s: float
    memory_mb: float
    model_size_mb: float
    n_features: int
    n_targets: int
    config: dict[str, Any] = field(default_factory=dict)


def run_benchmark_suite(
    data_path: str | Path = "data/turbojet_complete_dataset.csv",
    output_dir: str | Path = "results/benchmarks",
) -> list[BenchmarkResult]:
    """Run latency/throughput/memory benchmarks for all model variants."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_dataset(data_path)
    train, _ = official_split(frame, seed=42)
    process = psutil.Process(os.getpid())

    variants = [
        ("hist_gradient_boosting", {}),
        ("extra_trees", {}),
        ("random_forest", {"n_estimators": 200}),
        ("stacking", {}),
        ("hybrid", {}),
    ]

    results: list[BenchmarkResult] = []

    for kind, kwargs in variants:
        start_mem = process.memory_info().rss / (1024 * 1024)
        start = perf_counter()

        if kind == "hybrid":
            model = HybridPhysicsMLModel.train(train, ml_kind="hist_gradient_boosting")
        else:
            model = create_model(kind, n_estimators=400, scale_targets=True).fit(train)

        fit_time_s = perf_counter() - start
        fit_mem = process.memory_info().rss / (1024 * 1024) - start_mem

        # Model size
        import tempfile
        import joblib

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            tmp_path = f.name
            joblib.dump(model, tmp_path)
        model_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        os.unlink(tmp_path)

        # Latency benchmark
        batch_sizes = [1, 10, 100]
        best_latencies = []

        for bs in batch_sizes:
            batch = train.iloc[:bs]
            num_warmup = 20
            num_runs = 200
            for _ in range(num_warmup):
                _ = model.predict(batch)
            timings = []
            for _ in range(num_runs):
                t0 = perf_counter()
                _ = model.predict(batch)
                timings.append((perf_counter() - t0) * 1000 / bs)
            best_latencies.extend(timings)

        latencies = np.sort(best_latencies)

        # Throughput: samples per second on batch=100
        batch100 = train.iloc[:100]
        t0 = perf_counter()
        for _ in range(50):
            _ = model.predict(batch100)
        total_s = perf_counter() - t0
        throughput = (50 * 100) / total_s

        n_features = len(train.columns) - len(TARGETS)
        n_targets = len(TARGETS)

        br = BenchmarkResult(
            name=f"{kind}",
            kind=kind,
            mean_latency_ms=float(np.mean(latencies)),
            p50_latency_ms=float(np.median(latencies)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            throughput_ops_s=float(throughput),
            memory_mb=float(max(fit_mem, 0.1)),
            model_size_mb=float(model_size_mb),
            n_features=n_features,
            n_targets=n_targets,
            config={"fit_time_s": round(fit_time_s, 2), **kwargs},
        )
        results.append(br)

    # Save
    summary = pd.DataFrame([vars(r) for r in results])
    summary.to_csv(output_dir / "benchmark_summary.csv", index=False)
    _generate_report(results, output_dir / "benchmark_report.md")
    return results


def _generate_report(results: list[BenchmarkResult], path: Path) -> None:
    lines = [
        "# Performance Benchmark Report",
        "",
        "## Latency & Throughput",
        "",
        "| Model | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput (ops/s) |",
        "|-------|-----------|----------|----------|----------|--------------------|",
    ]
    for r in sorted(results, key=lambda x: x.mean_latency_ms):
        lines.append(
            f"| {r.kind} | {r.mean_latency_ms:.3f} | {r.p50_latency_ms:.3f} | "
            f"{r.p95_latency_ms:.3f} | {r.p99_latency_ms:.3f} | {r.throughput_ops_s:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Resource Usage",
            "",
            "| Model | Memory (MB) | Model Size (MB) |",
            "|-------|-------------|-----------------|",
        ]
    )
    for r in sorted(results, key=lambda x: x.memory_mb):
        lines.append(f"| {r.kind} | {r.memory_mb:.1f} | {r.model_size_mb:.1f} |")
    lines.append("")
    path.write_text("\n".join(lines))
