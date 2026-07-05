"""Serializable multi-output surrogate model with target scaling and stacking support."""

from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.dataset.features import engineer_all_features
from src.uncertainty.conformal import ConformalRegressor


class SurrogateModel:
    """Column-aware wrapper around a fitted scikit-learn pipeline with target scaling.

    ``feature_names`` remains the raw sensor schema. Internally expands each row
    with ``engineer_all_features``. If ``target_scalers`` are provided, targets are
    scaled before fitting and inverse-scaled after prediction.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        feature_names: list[str],
        target_names: list[str],
        pipeline_feature_names: list[str] | None = None,
        calibrator: ConformalRegressor | None = None,
        target_scalers: dict[str, StandardScaler] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.feature_names = feature_names
        self.target_names = target_names
        self.pipeline_feature_names = pipeline_feature_names or feature_names
        self.calibrator = calibrator
        self.target_scalers = target_scalers or {}

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        return engineer_all_features(frame[self.feature_names])

    def _scale_targets(self, values: np.ndarray) -> np.ndarray:
        if not self.target_scalers:
            return values
        out = values.copy()
        for i, name in enumerate(self.target_names):
            scaler = self.target_scalers.get(name)
            if scaler is not None:
                out[:, i] = scaler.transform(out[:, i : i + 1]).ravel()
        return out

    def _unscale_targets(self, values: np.ndarray) -> np.ndarray:
        if not self.target_scalers:
            return values
        out = values.copy()
        for i, name in enumerate(self.target_names):
            scaler = self.target_scalers.get(name)
            if scaler is not None:
                out[:, i] = scaler.inverse_transform(out[:, i : i + 1]).ravel()
        return out

    def _postprocess(self, values: pd.DataFrame) -> pd.DataFrame:
        out = values.copy()
        for column in out.columns:
            if column.endswith("Health"):
                out[column] = out[column].clip(0.0, 1.0)
            elif column in {"Thrust", "TSFC"}:
                out[column] = out[column].clip(lower=0.0)
        return out

    def fit(self, frame: pd.DataFrame) -> "SurrogateModel":
        targets = frame[self.target_names].to_numpy()
        if self.target_scalers:
            for i, name in enumerate(self.target_names):
                scaler = self.target_scalers.get(name)
                if scaler is not None:
                    scaler.fit(targets[:, i: i + 1])
        scaled = self._scale_targets(targets)
        self.pipeline.fit(self._prepare(frame), scaled)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        values = np.asarray(self.pipeline.predict(self._prepare(frame)))
        values = self._unscale_targets(values)
        prediction = pd.DataFrame(values, columns=self.target_names, index=frame.index)
        return self._postprocess(prediction)

    def calibrate(self, frame: pd.DataFrame, coverage: float = 0.9) -> "SurrogateModel":
        prediction = self.predict(frame)
        self.calibrator = ConformalRegressor(coverage).fit(
            frame[self.target_names].to_numpy(), prediction.to_numpy()
        )
        return self

    def predict_with_uncertainty(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
        prediction = self.predict(frame)
        if self.calibrator is None:
            lower = prediction.copy()
            upper = prediction.copy()
            return prediction, lower, upper, 0.0
        lower_values, upper_values = self.calibrator.predict_interval(prediction.to_numpy())
        lower = self._postprocess(
            pd.DataFrame(lower_values, columns=self.target_names, index=frame.index)
        )
        upper = self._postprocess(
            pd.DataFrame(upper_values, columns=self.target_names, index=frame.index)
        )
        return prediction, lower, upper, self.calibrator.coverage

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        joblib.dump(self, temporary)
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "SurrogateModel":
        model: Any = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("artifact is not a SurrogateModel")
        if not hasattr(model, "pipeline_feature_names"):
            model.pipeline_feature_names = model.feature_names
        if not hasattr(model, "calibrator"):
            model.calibrator = None
        if not hasattr(model, "target_scalers"):
            model.target_scalers = {}
        return model
