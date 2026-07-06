"""Hybrid physics + ML surrogate: ML models the residual from the physics simulation.

Instead of predicting targets directly, the ML model learns the *error* of the
physics model::

    actual = physics_prediction + ML_prediction(residual)

This is a powerful digital twin approach because:
1. The physics handles condition-dependent variation
2. The ML only needs to model the degradation signal (much simpler)
3. The combined prediction is physically grounded and data-corrected
4. The residual magnitude itself is a diagnostic (model mismatch -> novel degradation)

TSFC is NOT modelled as an independent hybrid target. Because TSFC = FuelFlow / Thrust
the residual is inherently nonlinear (proportional to 1/Thrust). Instead, TSFC is
derived *post-hoc* from the hybrid Thrust prediction and the known fuel flow.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.dataset.loader import SENSOR_FEATURES, TARGETS
from src.physics.cycle_model import BraytonCycle, CycleInput
from src.surrogate.model import SurrogateModel

# Targets that the ML residual model should learn.
# TSFC is excluded because it is derived post-hoc as fuel_flow / thrust.
HYBRID_ML_TARGETS = [t for t in TARGETS if t != "TSFC"]


class HybridPhysicsMLModel:
    """Physics-based prediction + ML residual correction.

    The ML model is a ``SurrogateModel`` trained to predict the *residual*
    (actual - physics_prediction). At inference, the combined prediction is::

        hybrid_prediction = physics_prediction + ml_residual_prediction

    TSFC is always computed as ``fuel_flow / thrust`` after the hybrid thrust
    prediction has been assembled, because modeling TSFC residuals directly
    introduces a nonlinearity that degrades hybrid performance.
    """

    def __init__(
        self,
        ml_model: SurrogateModel,
        physics: BraytonCycle | None = None,
    ) -> None:
        self.ml_model = ml_model
        self.physics = physics or BraytonCycle()

    @property
    def feature_names(self) -> list[str]:
        return self.ml_model.feature_names

    @property
    def pipeline_feature_names(self) -> list[str]:
        return self.ml_model.pipeline_feature_names

    @property
    def pipeline(self) -> "Pipeline":
        return self.ml_model.pipeline

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.ml_model._prepare(frame)

    @classmethod
    def train(
        cls,
        frame: pd.DataFrame,
        ml_kind: str = "hist_gradient_boosting",
        n_estimators: int = 300,
        seed: int = 42,
    ) -> "HybridPhysicsMLModel":
        """Train a hybrid model by computing physics residuals first.

        For each row, evaluates the physics model at the same flight condition
        (assuming fully healthy components), then computes::

            residual = actual - physics_prediction

        The ML residual model is trained for HYBRID_ML_TARGETS only (TSFC is
        excluded and derived post-hoc from thrust).
        """
        physics = BraytonCycle()
        physics_preds = _batch_physics_predict(physics, frame)
        residual_frame = frame.copy()
        for target in HYBRID_ML_TARGETS:
            if target == "OverallHealth":
                continue
            actual = frame[target].values.astype(float)
            if target in physics_preds:
                physics_val = physics_preds[target].values.astype(float)
                residual_frame[target] = actual - physics_val
            else:
                residual_frame[target] = actual

        ml_model = SurrogateModel.__new__(SurrogateModel)
        from sklearn.pipeline import Pipeline
        from src.dataset.preprocess import build_preprocessor
        from src.surrogate.train import PIPELINE_FEATURES, _base_estimator
        from sklearn.multioutput import MultiOutputRegressor
        estimator = MultiOutputRegressor(_base_estimator(ml_kind, seed, n_estimators))
        pipeline = Pipeline(
            [("preprocess", build_preprocessor(PIPELINE_FEATURES)), ("model", estimator)]
        )
        # clip_predictions=False: this model predicts residuals
        # (actual - physics_baseline), which are legitimately negative for
        # "*Health" columns. The default SurrogateModel postprocessing clips
        # any "*Health" column to [0, 1], which would zero out every
        # negative residual and destroy the learned degradation signal.
        ml_model.__init__(
            pipeline, SENSOR_FEATURES, HYBRID_ML_TARGETS, PIPELINE_FEATURES,
            target_scalers={}, clip_predictions=False,
        )
        ml_model.fit(residual_frame)
        return cls(ml_model, physics)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return hybrid prediction = physics + ML residual (TSFC derived post-hoc)."""
        physics_preds = _batch_physics_predict(self.physics, frame)
        ml_residuals = self.ml_model.predict(frame)
        combined = physics_preds.copy()
        for target in HYBRID_ML_TARGETS:
            if target == "OverallHealth":
                combined[target] = _overall_from_components(combined, ml_residuals)
            else:
                combined[target] = combined[target].values + ml_residuals[target].values
        combined["TSFC"] = _derive_tsfc(frame, combined)
        return _clip_physical(combined)

    def predict_with_uncertainty(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
        """Return (point, lower, upper, confidence) using ML residual uncertainty."""
        physics_preds = _batch_physics_predict(self.physics, frame)
        _, lower_res, upper_res, confidence = self.ml_model.predict_with_uncertainty(frame)
        point = physics_preds.copy()
        lower = physics_preds.copy()
        upper = physics_preds.copy()
        for target in HYBRID_ML_TARGETS:
            if target == "OverallHealth":
                continue
            point[target] = point[target].values + self.ml_model.predict(frame)[target].values
            lower[target] = lower[target].values + lower_res[target].values
            upper[target] = upper[target].values + upper_res[target].values
        point["OverallHealth"] = _overall_from_components(point, point)
        lower["OverallHealth"] = _overall_from_components(lower, lower)
        upper["OverallHealth"] = _overall_from_components(upper, upper)
        point["TSFC"] = _derive_tsfc(frame, point)
        lower["TSFC"] = _derive_tsfc(frame, upper)
        upper["TSFC"] = _derive_tsfc(frame, lower)
        return _clip_physical(point), _clip_physical(lower), _clip_physical(upper), confidence

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        joblib.dump({"ml_model": self.ml_model}, temporary)
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "HybridPhysicsMLModel":
        data: dict = joblib.load(path)
        return cls(data["ml_model"])


def _batch_physics_predict(physics: BraytonCycle, frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the physics model for every row assuming a *fully healthy* engine.

    Component health defaults to 1.0 regardless of any ``CompressorHealth`` etc.
    columns in *frame* — the function computes the healthy-engine baseline so
    that the ML model learns the *residual* (actual − healthy physics), which is
    the pure degradation signal.
    """
    results = {t: np.zeros(len(frame)) for t in TARGETS}
    for i, (_, row) in enumerate(frame.iterrows()):
        cin = CycleInput(
            altitude_m=float(row.get("Altitude", 0)),
            mach=float(row.get("Mach", 0)),
            ambient_temperature_k=float(row.get("Tamb", 288.15)),
            ambient_pressure_pa=float(row.get("Pamb", 101325)),
            rpm=float(row.get("RPM", 80000)),
            fuel_flow_kg_s=float(row.get("FuelFlow", 0.5)),
            compressor_health=1.0,
            combustor_health=1.0,
            turbine_health=1.0,
        )
        state = physics.evaluate(cin)
        results["CompressorHealth"][i] = 1.0
        results["CombustorHealth"][i] = 1.0
        results["TurbineHealth"][i] = 1.0
        results["Thrust"][i] = state.thrust_n
        results["TSFC"][i] = state.tsfc_kg_n_s
    results["OverallHealth"] = np.ones(len(frame))
    return pd.DataFrame(results, index=frame.index)


def _clip_physical(values: pd.DataFrame) -> pd.DataFrame:
    """Clip the FINAL combined (physics + residual) prediction to physical bounds.

    This must only be applied to the combined output, never to the raw ML
    residual — residuals are legitimately negative for "*Health" columns
    (see HybridPhysicsMLModel.train docstring and SurrogateModel.clip_predictions).
    """
    out = values.copy()
    for column in out.columns:
        if column.endswith("Health"):
            out[column] = out[column].clip(0.0, 1.0)
        elif column in {"Thrust", "TSFC"}:
            out[column] = out[column].clip(lower=0.0)
    return out


def _derive_tsfc(frame: pd.DataFrame, thrust_preds: pd.DataFrame) -> pd.Series:
    """Derive TSFC = FuelFlow / Thrust from known fuel flow and predicted thrust.

    TSFC is not modelled as an independent hybrid residual target (see module
    docstring): because TSFC = FuelFlow / Thrust, an additive residual on TSFC
    is nonlinear in Thrust and degrades badly under the physics+ML residual
    scheme. Instead it is always derived post-hoc from the (known) FuelFlow
    input and the model's own Thrust prediction, guarding against
    division-by-zero / negative thrust with a small epsilon floor.
    """
    fuel_flow = frame["FuelFlow"].values.astype(float)
    thrust = thrust_preds["Thrust"].values.astype(float)
    safe_thrust = np.maximum(thrust, 1e-6)
    return pd.Series(fuel_flow / safe_thrust, index=frame.index)


def _overall_from_components(
    comp_preds: pd.DataFrame, res_preds: pd.DataFrame
) -> pd.Series:
    """Compute overall health from component health predictions."""
    from src.health.overall import overall_health
    c = comp_preds["CompressorHealth"].values if "CompressorHealth" in comp_preds else res_preds.get("CompressorHealth", pd.Series(1.0))
    co = comp_preds["CombustorHealth"].values if "CombustorHealth" in comp_preds else res_preds.get("CombustorHealth", pd.Series(1.0))
    t = comp_preds["TurbineHealth"].values if "TurbineHealth" in comp_preds else res_preds.get("TurbineHealth", pd.Series(1.0))
    return pd.Series([overall_health(float(c[i]), float(co[i]), float(t[i])) for i in range(len(c))], index=comp_preds.index)
