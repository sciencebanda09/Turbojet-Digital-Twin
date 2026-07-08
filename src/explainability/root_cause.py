"""Root cause analysis for health and RUL predictions. Ranks input contributions
to health deltas using physics sensitivity or SHAP values."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import shap

    _HAS_SHAP = True
except ImportError:  # pragma: no cover
    shap = None
    _HAS_SHAP = False


_SENSITIVITY = {
    "FuelFlow": (
        0.6,
        "Fuel flow increased -> turbine inlet temperature rose -> thermal stress increased",
    ),
    "RPM": (0.4, "RPM increased -> compressor pressure ratio rose -> mechanical loading increased"),
    "Tamb": (0.15, "Ambient temperature increased -> compressor work increased -> margin reduced"),
    "Pamb": (0.1, "Ambient pressure changed -> station pressures shifted -> cycle margin shifted"),
    "compressor_efficiency": (1.0, "Compressor efficiency decreased -> pressure ratio degraded"),
    "turbine_efficiency": (1.0, "Turbine efficiency decreased -> expansion work degraded"),
}


@dataclass(frozen=True)
class ContributingFactor:
    """One ranked cause behind an observed prediction change."""

    factor: str
    contribution: float
    explanation: str


@dataclass(frozen=True)
class RootCauseReport:
    """Ranked explanation for a health/RUL change."""

    summary: str
    factors: list[ContributingFactor]
    causal_chain: list[str]


def _relative_change(before: float | None, after: float | None) -> float:
    """Return the signed relative change of after vs before, guarding zero."""
    if before is None or after is None:
        return 0.0
    denom = abs(before) if abs(before) > 1e-9 else 1.0
    return (after - before) / denom


def analyze_scenario(
    baseline_inputs: dict[str, float],
    adjusted_inputs: dict[str, float],
    health_delta: float,
) -> RootCauseReport:
    """Explain a what-if health change via physics-sensitivity ranking."""
    scored: list[ContributingFactor] = []
    for key, (weight, narrative) in _SENSITIVITY.items():
        if key not in baseline_inputs or key not in adjusted_inputs:
            continue
        rel_change = _relative_change(baseline_inputs[key], adjusted_inputs[key])
        if abs(rel_change) < 1e-6:
            continue
        contribution = weight * rel_change
        scored.append(ContributingFactor(key, round(contribution, 4), narrative))

    scored.sort(key=lambda item: abs(item.contribution), reverse=True)
    direction = (
        "decreased" if health_delta < 0 else "increased" if health_delta > 0 else "did not change"
    )
    summary = f"Overall health {direction} by {abs(health_delta):.3f}."
    if scored:
        summary += (
            f" Primary driver: {scored[0].factor} ({scored[0].explanation.split(' -> ')[0]})."
        )
    chain = [f.explanation for f in scored[:3]]
    if chain:
        chain.append("-> RUL and failure probability updated accordingly")
    logger.info("root cause analysis: %d factors ranked", len(scored))
    return RootCauseReport(summary, scored, chain)


def analyze_faults(fault_summary: list[dict[str, Any]], health_delta: float) -> RootCauseReport:
    """Explain a health change driven by active faults."""
    narratives = {
        "compressor_fouling": "Compressor fouling -> pressure ratio degraded -> compressor health decreased",
        "turbine_erosion": "Turbine erosion -> expansion efficiency degraded -> turbine health decreased",
        "fuel_nozzle_blockage": "Fuel nozzle blockage -> fuel delivery restricted -> combustion energy decreased",
        "bearing_wear": "Bearing wear -> spool friction increased -> compressor and turbine health decreased",
        "sensor_drift": "Sensor drift -> measurement bias grew over time -> estimated health distorted",
        "sensor_bias": "Sensor bias -> fixed measurement offset -> estimated health distorted",
    }
    scored = sorted(
        (
            ContributingFactor(
                item["fault_type"],
                round(float(item.get("severity", 0.0)), 4),
                narratives.get(item["fault_type"], "Fault active -> health impacted"),
            )
            for item in fault_summary
        ),
        key=lambda item: item.contribution,
        reverse=True,
    )
    direction = (
        "decreased" if health_delta < 0 else "increased" if health_delta > 0 else "did not change"
    )
    summary = f"Overall health {direction} by {abs(health_delta):.3f} due to {len(scored)} active fault(s)."
    chain = [f.explanation for f in scored[:3]]
    if chain:
        chain.append("-> RUL decreased -> maintenance urgency increased")
    return RootCauseReport(summary, scored, chain)


def shap_feature_importance(model: Any, frame: Any) -> list[ContributingFactor] | None:
    """Rank ML-model feature contributions with SHAP, if available."""
    if not _HAS_SHAP:
        logger.info("shap not installed; skipping SHAP-based root cause ranking")
        return None
    pipeline = getattr(model, "pipeline", None)
    estimator = pipeline.steps[-1][1] if hasattr(pipeline, "steps") else pipeline
    try:
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(frame)
    except Exception as error:  # pragma: no cover - depends on optional estimator support
        logger.warning("SHAP explanation failed: %s", error)
        return None
    row = values[0] if hasattr(values, "__len__") and len(values) else values
    factors = [
        ContributingFactor(name, round(float(val), 4), f"{name} contributed via learned model")
        for name, val in zip(model.feature_names, row, strict=False)
    ]
    factors.sort(key=lambda item: abs(item.contribution), reverse=True)
    return factors
