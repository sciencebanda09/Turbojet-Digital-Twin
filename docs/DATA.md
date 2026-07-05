# Dataset Contract

## Schema

One row represents one engine cycle. All quantities use SI units: metres, kelvin, pascals, revolutions/minute, kg/s, newtons, kg/(N·s). Health values are dimensionless on `[0, 1]`.

### Input Columns (8 required)

| Column | Unit | Description |
|---|---|---|
| `EngineID` | — | Engine identifier (integer) |
| `Cycle` | — | Cycle number for this engine |
| `Altitude` | m | Flight altitude |
| `Mach` | — | Flight Mach number |
| `Tamb` | K | Ambient temperature |
| `Pamb` | Pa | Ambient pressure |
| `RPM` | rev/min | Engine shaft speed |
| `FuelFlow` | kg/s | Fuel mass flow rate |

### Station Measurement Columns (6 required)

| Column | Unit | Description |
|---|---|---|
| `P2` | Pa | Compressor inlet pressure |
| `T2` | K | Compressor inlet temperature |
| `P3` | Pa | Compressor exit / combustor inlet pressure |
| `T3` | K | Compressor exit / combustor inlet temperature |
| `P4` | Pa | Turbine exit pressure |
| `T4` | K | Turbine exit temperature |

### Target Columns (6, required for training only)

| Column | Unit | Description |
|---|---|---|
| `CompressorHealth` | — | Compressor health state `[0, 1]` |
| `CombustorHealth` | — | Combustor health state `[0, 1]` |
| `TurbineHealth` | — | Turbine health state `[0, 1]` |
| `OverallHealth` | — | Fused health indicator `[0, 1]` |
| `Thrust` | N | Engine thrust output |
| `TSFC` | kg/(N·s) | Thrust-specific fuel consumption |

## Engineered Features

The feature engineering step (`src/dataset/features.py`) adds 20 derived features to the 8 raw inputs, producing **34 total features** used by the surrogate pipeline.

### Physics-Residual Columns (6)

Each is the fractional deviation of a measured station value from the healthy-engine prediction at the *same flight condition*, computed by evaluating the Brayton-cycle physics model with component health = 1.0:

| Column | Formula |
|---|---|
| `ResP2` | `(P2 − hP2) / hP2` |
| `ResT2` | `(T2 − hT2) / hT2` |
| `ResP3` | `(P3 − hP3) / hP3` |
| `ResT3` | `(T3 − hT3) / hT3` |
| `ResP4` | `(P4 − hP4) / hP4` |
| `ResT4` | `(T4 − hT4) / hT4` |

### Thermodynamic Ratio Columns (10)

| Column | Formula |
|---|---|
| `CompressorPR` | `P3 / P2` |
| `TurbinePR` | `P4 / P3` |
| `CompressorDeltaT` | `T3 − T2` |
| `TurbineDeltaT` | `T4 − T3` |
| `FuelPerRPM` | `FuelFlow / RPM` |
| `CorrectedRPM` | `RPM / √(T2 / 288.15)` |
| `TempRatioComp` | `T3 / T2` |
| `TempRatioTurb` | `T4 / T3` |
| `OverallPR` | `P3 / Pamb` |
| `BurnerTempRise` | `T3 − T2` |
| `FlowSquared` | `FuelFlow²` |
| `RPMSquared` | `(RPM / 100000)²` |
| `FuelFlowRPM` | `FuelFlow × (RPM / 100000)` |
| `CorrectedFuelFlow` | `FuelFlow / √(Tamb / 288.15)` |

> **Note:** `CompressorDeltaT` and `BurnerTempRise` are identical in the current implementation (both = `T3 − T2`). This is by design — the former is a compressor diagnostic, the latter a combustor diagnostic, and the model will assign near-zero weight to the redundant column if it provides no additional information.

## Split Strategies

Two strategies are available in `src/dataset/split.py`:

| Strategy | Description | Use Case |
|---|---|---|
| `official_split` | Holds out a fraction of each engine's own cycles | Matches the official train.csv/test.csv distribution — use for graded metrics |
| `grouped_split` | Holds out entire engines | Harder generalization stress test — use for research / ablation studies |

Both accept `seed` for reproducibility and `test_size` (default 0.2).

## Data Source

The dataset is expected at `data/turbojet_complete_dataset.csv` by default. The `pipeline.py demo` command generates a synthetic dataset with 5 engines × 60 cycles that matches this schema.
