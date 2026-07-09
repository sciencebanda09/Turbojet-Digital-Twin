"""End-to-end training workflow."""

from pathlib import Path
from typing import Any
import json
from src.dataset.loader import TARGETS, load_dataset
from src.dataset.split import grouped_split, official_split
from src.metrics.regression import regression_metrics
from src.surrogate.hybrid import HybridPhysicsMLModel
from src.surrogate.train import create_model

_STRATEGIES = {
    "official": official_split,
    "grouped": grouped_split,
}


def train_from_csv(
    data_path: str | Path,
    output_path: str | Path,
    kind: str = "extra_trees",
    seed: int = 42,
    strategy: str = "official",
    n_estimators: int = 300,
) -> dict[str, float]:
    """Validate, split, fit, evaluate, and persist a surrogate.

    ``strategy="official"`` (default) holds out a fraction of each engine's
    cycles, matching the officially distributed train.csv/test.csv split —
    use this to get metrics comparable to how submissions are graded.
    ``strategy="grouped"`` holds out entire engines instead, a harder
    generalization stress test not used by the official evaluation.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unsupported split strategy: {strategy}")
    frame = load_dataset(data_path)
    train, test = _STRATEGIES[strategy](frame, seed=seed)
    # Hold out half of test for conformal calibration to avoid data leak
    calibration = test.iloc[: len(test) // 2]
    heldout = test.iloc[len(test) // 2 :]
    if kind == "hybrid":
        model: Any = HybridPhysicsMLModel.train(train, ml_kind="hist_gradient_boosting", seed=seed)
        prediction = model.predict(heldout)
    else:
        model = create_model(kind, seed=seed, n_estimators=n_estimators).fit(train)
        prediction = model.predict(heldout)
    metrics = regression_metrics(heldout[TARGETS].to_numpy(), prediction.to_numpy())
    if kind == "hybrid":
        model.save(output_path)
    else:
        model.calibrate(calibration)
        model.save(output_path)
    metrics_path = Path(output_path).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
