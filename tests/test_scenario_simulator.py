"""Unit tests for src.simulation.what_if."""

import pytest

from src.simulation.what_if import ScenarioAdjustment, ScenarioSimulator


def _baseline() -> dict:
    return {
        "Altitude": 0.0,
        "Mach": 0.0,
        "Tamb": 288.15,
        "Pamb": 101_325.0,
        "RPM": 80_000.0,
        "FuelFlow": 1.0,
    }


def test_no_adjustment_yields_zero_delta() -> None:
    comparison = ScenarioSimulator().run(_baseline(), ScenarioAdjustment())
    assert comparison.delta["overall_health"] == pytest.approx(0.0, abs=1e-9)
    assert comparison.delta["thrust_n"] == pytest.approx(0.0, abs=1e-6)


def test_lower_compressor_efficiency_reduces_health() -> None:
    adjustment = ScenarioAdjustment(compressor_efficiency=0.5)
    comparison = ScenarioSimulator().run(_baseline(), adjustment)
    assert comparison.adjusted.overall_health < comparison.baseline.overall_health
    assert comparison.delta["overall_health"] < 0


def test_higher_fuel_flow_changes_thrust() -> None:
    adjustment = ScenarioAdjustment(fuel_flow_kg_s=2.0)
    comparison = ScenarioSimulator().run(_baseline(), adjustment)
    assert comparison.adjusted.thrust_n != comparison.baseline.thrust_n


def test_sensor_noise_reduces_confidence() -> None:
    adjustment = ScenarioAdjustment(sensor_noise_std=0.3)
    comparison = ScenarioSimulator().run(_baseline(), adjustment)
    assert comparison.adjusted.confidence == pytest.approx(0.7)
    assert comparison.baseline.confidence == pytest.approx(1.0)


def test_invalid_efficiency_raises() -> None:
    with pytest.raises(ValueError):
        ScenarioAdjustment(compressor_efficiency=1.5)


def test_negative_noise_raises() -> None:
    with pytest.raises(ValueError):
        ScenarioAdjustment(sensor_noise_std=-0.1)


def test_missing_required_field_raises() -> None:
    incomplete = {"Tamb": 288.15, "Pamb": 101_325.0, "RPM": 80_000.0}
    with pytest.raises(KeyError):
        ScenarioSimulator().run(incomplete, ScenarioAdjustment())
