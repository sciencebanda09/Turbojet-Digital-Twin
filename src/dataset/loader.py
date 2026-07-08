"""Canonical turbojet dataset loader."""

from pathlib import Path
import pandas as pd

# Identifier columns for grouping/tracking only — NEVER as model input features.
IDENTIFIER_COLUMNS = ["EngineID", "Cycle"]

# Physical sensor readings only — these are the legitimate model input features.
SENSOR_FEATURES = [
    "Altitude",
    "Mach",
    "Tamb",
    "Pamb",
    "RPM",
    "FuelFlow",
    "P2",
    "T2",
    "P3",
    "T3",
    "P4",
    "T4",
]

# Legacy full feature list (includes identifiers).
# WARNING: Do NOT use FEATURES for model training/inference — use SENSOR_FEATURES instead.
FEATURES = IDENTIFIER_COLUMNS + SENSOR_FEATURES
TARGETS = [
    "CompressorHealth",
    "CombustorHealth",
    "TurbineHealth",
    "OverallHealth",
    "Thrust",
    "TSFC",
]

# Raw exports (e.g. Kaggle-style dumps) ship unit-suffixed column names.
# Map every known variant back to the canonical schema above.
_COLUMN_ALIASES = {
    "Altitude_m": "Altitude",
    "Tamb_K": "Tamb",
    "Pamb_Pa": "Pamb",
    "RPM_rev_min": "RPM",
    "FuelFlow_kg_s": "FuelFlow",
    "P2_Pa": "P2",
    "T2_K": "T2",
    "P3_Pa": "P3",
    "T3_K": "T3",
    "P4_Pa": "P4",
    "T4_K": "T4",
    "Thrust_N": "Thrust",
    "TSFC_g_N_s": "TSFC",
}


def load_dataset(path: str | Path, require_targets: bool = True) -> pd.DataFrame:
    """Read a CSV and enforce the official numeric schema."""
    frame = pd.read_csv(path)
    tsfc_in_grams = "TSFC" not in frame.columns and "TSFC_g_N_s" in frame.columns
    frame = frame.rename(columns={k: v for k, v in _COLUMN_ALIASES.items() if k in frame.columns})
    required = FEATURES + (TARGETS if require_targets else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    frame = frame.copy()
    frame[required] = frame[required].apply(pd.to_numeric, errors="raise")
    if tsfc_in_grams and "TSFC" in frame.columns:
        frame["TSFC"] = frame["TSFC"] / 1000.0
    if frame[required].isna().any().any():
        raise ValueError("Dataset contains missing required values")
    return frame


# Default location of the real official challenge dataset, used by
# sample_real_dataset() below. Overridable for testing against a different
# copy (e.g. a repo checked out somewhere other than the project root).
DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "turbojet_complete_dataset.csv"
)


def sample_real_dataset(
    n_engines: int | None = None,
    n_cycles: int | None = None,
    seed: int = 42,
    path: str | Path = DEFAULT_DATASET_PATH,
) -> pd.DataFrame:
    """Deterministic slice of the official dataset for tests and demos."""
    frame = load_dataset(path)
    all_engine_ids = sorted(frame["EngineID"].unique())
    if n_engines is not None and n_engines < len(all_engine_ids):
        rng = pd.Series(all_engine_ids).sample(n=n_engines, random_state=seed)
        selected_ids = sorted(rng.tolist())
        frame = frame[frame["EngineID"].isin(selected_ids)]
    if n_cycles is not None:
        frame = (
            frame.sort_values(["EngineID", "Cycle"])
            .groupby("EngineID", group_keys=False)
            .head(n_cycles)
        )
    return frame.reset_index(drop=True)
