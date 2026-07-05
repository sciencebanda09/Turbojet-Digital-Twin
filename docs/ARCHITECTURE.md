# Architecture

## Data Flow

```
Sensor Record (EngineID, Cycle, Altitude, Mach, Tamb, Pamb, RPM, FuelFlow, P2–T4)
    │
    ├─► Schema / Range Validation
    │
    ├─► [Feature Engineering]  ──►  34 total features
    │    8 raw sensors
    │   + 6 physics-normalized residuals (ResP2–ResT4)
    │   + 10 engineered ratios/deltas (CompressorPR, TempRatioComp, etc.)
    │
    ├─► [Brayton-Cycle Physics Model]
    │    Variable specific heats: cp_air(T), cp_gas(T, far)
    │    ISA standard atmosphere: isa_temperature(), isa_pressure()
    │    4th-order component efficiency maps
    │    Spool energy balance
    │
    ├─► [Learned Surrogate / Hybrid Physics+ML]
    │    Model kinds: HGBT, ExtraTrees, RandomForest, GradientBoosting,
    │    Stacking, XGBoost, MLP, Hybrid (physics + ML residual)
    │    Target scaling per output (StandardScaler)
    │    Uncertainty modes: conformal · quantile · ensemble
    │
    └─► [Bayesian State Estimator]
         EKF or UKF (selectable)
         State: [health, degradation_rate]
         Observation: surrogate prediction
         Prior: constant degradation (1e-4/cycle)
         │
         └─► DigitalTwin Facade
              │
              ├─► RUL (configurable failure/warning thresholds)
              ├─► Failure Probability (data-calibrated logistic regression)
              ├─► Health Trajectories
              ├─► Fleet Ranking
              │
              ├─► FastAPI (8 endpoints)
              ├─► Streamlit Dashboard (18 pages)
              ├─► Report Generator (Markdown)
              └─► Model Export (joblib)
```

## Module Dependencies

```
src/physics         ←  src/surrogate/hybrid (physics baseline for residual)
src/dataset         →  src/surrogate (feature preparation)
src/surrogate       →  src/uncertainty (conformal / quantile)
src/surrogate       →  src/explainability (SHAP explainer)
src/digital_twin    →  src/estimation (EKF/UKF)
src/digital_twin    →  src/failure (calibrator)
src/digital_twin    →  src/rul (config)
src/simulation      →  src/physics (what-if scenarios)
src/validation      →  src/surrogate + src/surrogate/hybrid
src/performance     →  src/surrogate + src/surrogate/hybrid
pipeline.py         →  all modules (CLI orchestration)
```

## Key Design Decisions

1. **Hybrid Physics + ML** (`src/surrogate/hybrid.py`) — rather than predicting targets directly, the ML model learns the residual: `prediction = physics + ml_residual(error)`. The physics handles condition-dependent variation; ML only models the degradation signal. Residual magnitude itself is a diagnostic (model mismatch → novel degradation pattern).

2. **Target Scaling** — `StandardScaler` per target because Thrust (0–90 kN) and Health (0–1) have vastly different scales. Without it, Thrust dominates aggregate RMSE and R² becomes misleading. Per-target metrics are reported alongside aggregate.

3. **Three Uncertainty Modes** — conformal (fast, marginal coverage), quantile (medium, conditional coverage), ensemble (slow, approximate). Users choose the trade-off between computational cost and interval quality. SurrogateModel stores a calibrator, quantile model, and supports bootstrapped ensemble.

4. **Data-Calibrated Failure Probability** — logistic regression fitted on degradation trajectories from training data, replacing heuristic thresholds. Integrated into DigitalTwin for online queries.

5. **Configurable RUL Thresholds** — `RULConfig` dataclass with `failure_threshold=0.3`, `warning_threshold=0.7` replaces hardcoded constants.

6. **Per-Target Model Architecture** — separate handling for health targets (clipped [0,1]) vs. performance targets (Thrust/TSFC clipped ≥ 0). Target scaling, per-target R² metrics, and separate evaluation.

7. **Stateful API** — each engine has an in-memory DigitalTwin with Kalman state. REST `/update` and batch `/batch` both advance the same estimator. No database dependency.

8. **Feature Engineering** — 20 derived features: physics-normalized station residuals (`ResP2`–`ResT4`) which remove operating-condition variance, plus thermodynamic ratios that capture component performance independent of scale.

## CLI Commands

| Command | Entry Point | Description |
|---|---|---|
| `train` | `pipeline.py` | Train one model variant |
| `tune` | `pipeline.py` | Grid-search hyperparameters |
| `evaluate` | `pipeline.py` | Evaluate saved model |
| `predict` | `pipeline.py` | Batch inference |
| `experiment` | `pipeline.py` | Logged experiment run |
| `ablation` | `pipeline.py` | Cross-model ablation |
| `report` | `pipeline.py` | Markdown report generation |
| `validation` | `pipeline.py` | Cross-model validation suite |
| `benchmark` | `pipeline.py` | Latency/throughput benchmarks |
| `orchestrate` | `pipeline.py` | End-to-end: train all → validate → benchmark |
| `demo` | `pipeline.py` | Generate + train demo data |

## New Modules (added during development)

| Module | File | Purpose |
|---|---|---|
| Hybrid Physics+ML | `src/surrogate/hybrid.py` | ML predicts residual from physics model |
| SHAP Explainability | `src/explainability/shap_explainer.py` | SHAG with fallback to permutation importance |
| Validation Suite | `src/validation/benchmark.py` | Cross-model validation with Markdown report |
| Performance Benchmark | `src/performance/benchmark.py` | Latency, throughput, memory, model size |
| Quantile Uncertainty | `src/uncertainty/quantile.py` | Quantile regression intervals |
| Adaptive Conformal | `src/uncertainty/adaptive_conformal.py` | Locally-weighted conformal prediction |
| Experiment Runner | `src/research/experiment.py` | Logged experiments with config + metrics |
| Research Report | `src/report/research.py` | Auto-generated Markdown research reports |
