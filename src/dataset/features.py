"""Physics-informed feature engineering: ratios, deltas, and physics residuals.

Residual features (ResP2–ResT4) express measured station values as fractional
deviations from a healthy-engine prediction at the same flight condition.
This removes operating-condition variance that would otherwise dominate the
degradation signal the model needs to learn."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.physics.cycle_model import BraytonCycle, CycleInput

RESIDUAL_COLUMNS = ["ResP2", "ResT2", "ResP3", "ResT3", "ResP4", "ResT4"]

_CYCLE = BraytonCycle()


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add stable ratios and thermodynamic deltas without mutating input."""
    out = frame.copy()
    eps = np.finfo(float).eps
    out["CompressorPR"] = out["P3"] / out["P2"].clip(lower=eps)
    out["TurbinePR"] = out["P4"] / out["P3"].clip(lower=eps)
    out["CompressorDeltaT"] = out["T3"] - out["T2"]
    out["TurbineDeltaT"] = out["T4"] - out["T3"]
    out["FuelPerRPM"] = out["FuelFlow"] / out["RPM"].clip(lower=eps)
    out["CorrectedRPM"] = out["RPM"] / np.sqrt((out["T2"] / 288.15).clip(lower=eps))
    out["TempRatioComp"] = out["T3"] / out["T2"].clip(lower=eps)
    out["TempRatioTurb"] = out["T4"] / out["T3"].clip(lower=eps)
    out["OverallPR"] = out["P3"] / out["Pamb"].clip(lower=eps)
    out["BurnerTempRise"] = out["T3"] - out["T2"]
    out["FlowSquared"] = out["FuelFlow"] ** 2
    out["RPMSquared"] = (out["RPM"] / 100_000.0) ** 2
    out["FuelFlowRPM"] = out["FuelFlow"] * (out["RPM"] / 100_000.0)
    out["CorrectedFuelFlow"] = out["FuelFlow"] / np.sqrt((out["Tamb"] / 288.15).clip(lower=eps))
    return out


def _healthy_station_state(row: pd.Series) -> tuple[float, float, float, float, float, float]:
    """Evaluate the Brayton cycle at this row's flight condition, health = 1.0."""
    cycle_input = CycleInput(
        altitude_m=float(row["Altitude"]),
        mach=float(row["Mach"]),
        ambient_temperature_k=float(row["Tamb"]),
        ambient_pressure_pa=float(row["Pamb"]),
        rpm=float(row["RPM"]),
        fuel_flow_kg_s=float(row["FuelFlow"]),
    )
    try:
        state = _CYCLE.evaluate(cycle_input)
    except ValueError:
        return (np.nan,) * 6
    return state.p2, state.t2, state.p3, state.t3, state.p4, state.t4


def healthy_reference_residuals(frame: pd.DataFrame) -> pd.DataFrame:
    """Fractional deviation of measured stations from a healthy-engine prediction."""
    eps = np.finfo(float).eps
    healthy = frame.apply(_healthy_station_state, axis=1, result_type="expand")
    healthy.columns = ["hP2", "hT2", "hP3", "hT3", "hP4", "hT4"]
    out = pd.DataFrame(index=frame.index)
    out["ResP2"] = (frame["P2"] - healthy["hP2"]) / healthy["hP2"].abs().clip(lower=eps)
    out["ResT2"] = (frame["T2"] - healthy["hT2"]) / healthy["hT2"].abs().clip(lower=eps)
    out["ResP3"] = (frame["P3"] - healthy["hP3"]) / healthy["hP3"].abs().clip(lower=eps)
    out["ResT3"] = (frame["T3"] - healthy["hT3"]) / healthy["hT3"].abs().clip(lower=eps)
    out["ResP4"] = (frame["P4"] - healthy["hP4"]) / healthy["hP4"].abs().clip(lower=eps)
    out["ResT4"] = (frame["T4"] - healthy["hT4"]) / healthy["hT4"].abs().clip(lower=eps)
    return out[RESIDUAL_COLUMNS]


def engineer_all_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Full feature set used by the surrogate: ratios/deltas + physics residuals."""
    out = engineer_features(frame)
    residuals = healthy_reference_residuals(frame)
    for column in RESIDUAL_COLUMNS:
        out[column] = residuals[column]
    return out
