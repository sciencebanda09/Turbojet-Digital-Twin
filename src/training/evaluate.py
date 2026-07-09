"""Saved-model evaluation."""

from pathlib import Path
from src.dataset.loader import TARGETS, load_dataset
from src.metrics.regression import regression_metrics
from src.surrogate.hybrid import HybridPhysicsMLModel
from src.surrogate.model import SurrogateModel


def evaluate_model(model_path: str | Path, data_path: str | Path) -> dict[str, float]:
    """Evaluate a serialized surrogate (SurrogateModel or HybridPhysicsMLModel) against a labeled CSV."""
    frame = load_dataset(data_path)
    try:
        model = SurrogateModel.load(model_path)
    except TypeError:
        model = HybridPhysicsMLModel.load(model_path)
    prediction = model.predict(frame)
    return regression_metrics(frame[TARGETS].to_numpy(), prediction.to_numpy())
