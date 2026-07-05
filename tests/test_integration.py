"""Integration tests for hybrid model, validation, and benchmark pipelines."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.dataset.loader import FEATURES, TARGETS
from src.performance.benchmark import run_benchmark_suite
from src.surrogate.hybrid import HybridPhysicsMLModel
from src.validation.benchmark import run_validation_suite


def _demo_frame(engines: int = 3, cycles: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for engine in range(1, engines + 1):
        rate = rng.uniform(0.0015, 0.004)
        for cycle in range(1, cycles + 1):
            altitude = rng.uniform(0, 10_000)
            mach = rng.uniform(0.1, 0.9)
            tamb = 288.15 - 0.0065 * altitude
            pamb = 101325 * (tamb / 288.15) ** 5.256
            rpm = rng.uniform(70_000, 98_000)
            fuel = rng.uniform(0.5, 1.2)
            health = np.clip(1 - rate * cycle + rng.normal(0, 0.004), 0.2, 1)
            p2 = pamb * (1 + 0.2 * mach**2) ** 3.5
            t2 = tamb * (1 + 0.2 * mach**2)
            p3 = p2 * (1 + 10 * (rpm / 100000) ** 2) * health
            t3 = t2 * (p3 / p2) ** (0.286 / max(0.7 * health, 0.4))
            t4 = min(1750, t3 + fuel * 43000000 / (26 * 1150)) - 250
            p4 = max(pamb * 1.05, p3 * 0.3)
            thrust = max(500, 20000 * health * (rpm / 100000) - altitude * 0.3)
            rows.append([
                engine, cycle, altitude, mach, tamb, pamb, rpm, fuel,
                p2, t2, p3, t3, p4, t4,
                health * 0.99, health * 0.995, health * 0.98, health * 0.987,
                thrust, fuel / thrust,
            ])
    return pd.DataFrame(rows, columns=FEATURES + TARGETS)


def test_hybrid_model_train_and_predict():
    frame = _demo_frame(engines=3, cycles=20)
    model = HybridPhysicsMLModel.train(frame, ml_kind="hist_gradient_boosting", seed=42)
    preds = model.predict(frame)
    assert list(preds.columns) == TARGETS
    assert len(preds) == len(frame)
    assert preds["Thrust"].min() >= 0
    assert preds["OverallHealth"].between(0, 1).all()
    assert preds["CompressorHealth"].between(0, 1).all()


def test_hybrid_model_uncertainty():
    frame = _demo_frame(engines=2, cycles=10)
    model = HybridPhysicsMLModel.train(frame, ml_kind="hist_gradient_boosting", seed=42)
    point, lower, upper, confidence = model.predict_with_uncertainty(frame)
    assert point.shape == (len(frame), len(TARGETS))
    assert (lower.values <= point.values + 1e-10).all()
    assert (point.values <= upper.values + 1e-10).all()
    assert 0 <= confidence <= 1


def test_hybrid_model_save_load(tmp_path: Path):
    frame = _demo_frame()
    model = HybridPhysicsMLModel.train(frame, seed=42)
    path = tmp_path / "hybrid.joblib"
    model.save(str(path))
    loaded = HybridPhysicsMLModel.load(str(path))
    preds_orig = model.predict(frame)
    preds_loaded = loaded.predict(frame)
    pd.testing.assert_frame_equal(preds_orig, preds_loaded)


@pytest.mark.slow
def test_validation_suite_runs():
    frame = _demo_frame()
    path = Path("results/test_validation_data.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    out_dir = Path("results/test_validation")
    results = run_validation_suite(data_path=str(path), output_dir=str(out_dir))
    assert len(results) > 0
    for r in results:
        assert r.rmse >= 0
        assert -1 <= r.r2 <= 1
        assert r.inference_time_ms > 0


@pytest.mark.slow
def test_benchmark_suite_runs():
    frame = _demo_frame()
    path = Path("results/test_benchmark_data.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    out_dir = Path("results/test_benchmark")
    results = run_benchmark_suite(data_path=str(path), output_dir=str(out_dir))
    assert len(results) > 0
    for r in results:
        assert r.mean_latency_ms > 0
        assert r.throughput_ops_s > 0


def test_explain_prediction():
    from src.explainability.shap_explainer import explain_prediction
    import numpy as np
    frame = _demo_frame()
    model = HybridPhysicsMLModel.train(frame, seed=42)
    raw = frame[model.ml_model.feature_names].iloc[:3]
    prepped = model.ml_model._prepare(raw)
    pipeline = model.ml_model.pipeline

    def predict_fn(x: pd.DataFrame) -> np.ndarray:
        return np.asarray(pipeline.predict(x))

    explanation = explain_prediction(
        predict_fn,
        prepped,
        feature_names=model.ml_model.pipeline_feature_names,
    )
    assert "method" in explanation
    assert len(explanation["global_importance"]) > 0
