"""What-if scenario simulation engine.

Lets a user perturb operating conditions (fuel flow, RPM, ambient temperature and
pressure, compressor/turbine efficiency, sensor noise) around a baseline
observation and immediately see how health, RUL, failure probability, thrust,
TSFC, and prediction confidence change.

The simulator is stateless per call: it does not mutate any :class:`DigitalTwin`
history. It reuses the existing physics (:mod:`src.physics.cycle_model`), health
(:mod:`src.health.overall`), RUL (:mod:`src.prediction.rul`), and risk
(:mod:`src.prediction.failure_probability`) modules directly so results stay
consistent with the rest of the platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np

from src.health.overall import overall_health
from src.physics.cycle_model import BraytonCycle, CycleInput
from src.prediction.failure_probability import failure_probability
from src.prediction.rul import estimate_rul

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScenarioAdjustment:
    """User-controllable deltas applied on top of a baseline observation.

    Each field is an absolute override when not ``None``; ``None`` means "keep
    baseline value". Efficiency fields are multiplicative health factors in
    ``[0, 1]`` applied on top of baseline (assumed healthy, i.e. 1.0) component
    health. ``sensor_noise_std`` is a fractional standard deviation (e.g. ``0.02``
    == 2% of signal) used only to compute prediction confidence, not to corrupt
    the deterministic physics evaluation.
    """

    fuel_flow_kg_s: float | None = None
    rpm: float | None = None
    ambient_temperature_k: float | None = None
    ambient_pressure_pa: float | None = None
    compressor_efficiency: float | None = None
    turbine_efficiency: float | None = None
    sensor_noise_std: float = 0.0

    def __post_init__(self) -> None:
        for name in ("compressor_efficiency", "turbine_efficiency"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.sensor_noise_std < 0.0:
            raise ValueError("sensor_noise_std must be nonnegative")


@dataclass(frozen=True)
class ScenarioSnapshot:
    """Predicted outputs for one operating point (baseline or adjusted)."""

    compressor_health: float
    combustor_health: float
    turbine_health: float
    overall_health: float
    remaining_useful_life_cycles: float
    failure_probability: float
    thrust_n: float
    tsfc_kg_n_s: float
    confidence: float


@dataclass(frozen=True)
class ScenarioComparison:
    """Before vs after result of a what-if simulation."""

    baseline: ScenarioSnapshot
    adjusted: ScenarioSnapshot
    delta: dict[str, float] = field(default_factory=dict)


class ScenarioSimulator:
    """Runs before/after what-if comparisons against the physics-based twin."""

    def __init__(
        self,
        physics: BraytonCycle | None = None,
        degradation_threshold: float = 0.3,
        max_rul_cycles: float = 5_000.0,
    ):
        """Create a simulator.

        Args:
            physics: Cycle model to evaluate; a fresh :class:`BraytonCycle` is
                created if omitted.
            degradation_threshold: Health value treated as functional failure,
                consistent with :func:`src.prediction.rul.estimate_rul`.
            max_rul_cycles: Cap applied to the single-shot RUL extrapolation.
                Because this simulator has no degradation-trend history, a
                near-1.0 health with zero observed slope would otherwise
                extrapolate to an unrealistically large RUL; this cap keeps
                the comparison meaningful.
        """
        self.physics = physics or BraytonCycle()
        self.degradation_threshold = degradation_threshold
        self.max_rul_cycles = max_rul_cycles

    def _snapshot(self, cycle_input: CycleInput, sensor_noise_std: float) -> ScenarioSnapshot:
        """Evaluate one operating point end-to-end into a comparable snapshot."""
        try:
            state = self.physics.evaluate(cycle_input)
        except ValueError as error:
            raise ValueError(f"invalid scenario operating point: {error}") from error

        health = overall_health(
            cycle_input.compressor_health, cycle_input.combustor_health, cycle_input.turbine_health
        )
        # Single-shot RUL: extrapolate a synthetic two-point trend from a
        # healthy baseline (cycle 0, health=1.0) to the current health at
        # cycle 1, matching estimate_rul's linear-trend contract without
        # requiring twin history.
        rul = estimate_rul(
            np.array([0.0, 1.0]), np.array([1.0, health]), threshold=self.degradation_threshold
        )
        remaining_cycles = min(rul.remaining_cycles, self.max_rul_cycles)
        probability = failure_probability(health, remaining_cycles)
        confidence = float(np.clip(1.0 - sensor_noise_std, 0.0, 1.0))
        return ScenarioSnapshot(
            compressor_health=cycle_input.compressor_health,
            combustor_health=cycle_input.combustor_health,
            turbine_health=cycle_input.turbine_health,
            overall_health=health,
            remaining_useful_life_cycles=remaining_cycles,
            failure_probability=probability,
            thrust_n=state.thrust_n,
            tsfc_kg_n_s=state.tsfc_kg_n_s,
            confidence=confidence,
        )

    def run(
        self, baseline_observation: dict[str, float], adjustment: ScenarioAdjustment
    ) -> ScenarioComparison:
        """Compare baseline vs adjusted operating conditions.

        Args:
            baseline_observation: Dict with at least ``Altitude``, ``Mach``,
                ``Tamb``, ``Pamb``, ``RPM``, ``FuelFlow`` (matching the API
                ``Observation`` schema).
            adjustment: Overrides to apply on top of the baseline.

        Returns:
            A :class:`ScenarioComparison` with baseline, adjusted, and delta.
        """
        base_input = CycleInput(
            altitude_m=baseline_observation.get("Altitude", 0.0),
            mach=baseline_observation.get("Mach", 0.0),
            ambient_temperature_k=baseline_observation["Tamb"],
            ambient_pressure_pa=baseline_observation["Pamb"],
            rpm=baseline_observation["RPM"],
            fuel_flow_kg_s=baseline_observation["FuelFlow"],
        )
        baseline_snapshot = self._snapshot(base_input, sensor_noise_std=0.0)

        adjusted_input = CycleInput(
            altitude_m=base_input.altitude_m,
            mach=base_input.mach,
            ambient_temperature_k=adjustment.ambient_temperature_k
            if adjustment.ambient_temperature_k is not None
            else base_input.ambient_temperature_k,
            ambient_pressure_pa=adjustment.ambient_pressure_pa
            if adjustment.ambient_pressure_pa is not None
            else base_input.ambient_pressure_pa,
            rpm=adjustment.rpm if adjustment.rpm is not None else base_input.rpm,
            fuel_flow_kg_s=adjustment.fuel_flow_kg_s
            if adjustment.fuel_flow_kg_s is not None
            else base_input.fuel_flow_kg_s,
            mass_flow_kg_s=base_input.mass_flow_kg_s,
            compressor_health=adjustment.compressor_efficiency
            if adjustment.compressor_efficiency is not None
            else base_input.compressor_health,
            combustor_health=base_input.combustor_health,
            turbine_health=adjustment.turbine_efficiency
            if adjustment.turbine_efficiency is not None
            else base_input.turbine_health,
        )
        adjusted_snapshot = self._snapshot(
            adjusted_input, sensor_noise_std=adjustment.sensor_noise_std
        )

        delta = {
            "overall_health": adjusted_snapshot.overall_health - baseline_snapshot.overall_health,
            "remaining_useful_life_cycles": adjusted_snapshot.remaining_useful_life_cycles
            - baseline_snapshot.remaining_useful_life_cycles,
            "failure_probability": adjusted_snapshot.failure_probability
            - baseline_snapshot.failure_probability,
            "thrust_n": adjusted_snapshot.thrust_n - baseline_snapshot.thrust_n,
            "tsfc_kg_n_s": adjusted_snapshot.tsfc_kg_n_s - baseline_snapshot.tsfc_kg_n_s,
        }
        logger.info("scenario simulated: delta_health=%.4f", delta["overall_health"])
        return ScenarioComparison(baseline_snapshot, adjusted_snapshot, delta)
