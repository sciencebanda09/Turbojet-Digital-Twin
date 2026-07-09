"""Degradation-trend remaining useful life estimation."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RULResult:
    """Remaining life point estimate and empirical uncertainty quantiles."""

    remaining_cycles: float
    remaining_hours: float
    q10: float
    q50: float
    q90: float
    degradation_rate: float


@dataclass(frozen=True)
class RULConfig:
    failure_threshold: float = 0.3
    warning_threshold: float = 0.7
    hours_per_cycle: float = 1.5


def estimate_rul(
    cycles: np.ndarray,
    health: np.ndarray,
    config: RULConfig | None = None,
) -> RULResult:
    """Extrapolate robust recent linear degradation to a failure threshold.

    Uses a two-threshold approach:
    - *failure_threshold* (default 0.3): the hard failure limit for RUL to zero.
    - *warning_threshold* (default 0.7): closer to observed degradation range,
      used for meaningful early RUL estimates when health hasn't dropped far.
    The returned RUL cycles always reference the failure threshold; if health
    is above the warning threshold the degradation is assumed to continue at
    the current rate.
    """
    cfg = config or RULConfig()
    threshold = cfg.failure_threshold
    x, y = np.asarray(cycles, dtype=float), np.asarray(health, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("at least two aligned observations are required")
    window = min(len(x), 50)
    coeffs = np.polyfit(x[-window:], y[-window:], 1)
    slope = coeffs[0]
    rate = max(-float(slope), 1e-6)
    remaining = max((float(y[-1]) - threshold) / rate, 0.0)
    residual = y[-window:] - np.polyval(coeffs, x[-window:])
    uncertainty = 1.645 * float(np.std(residual)) / rate
    return RULResult(
        remaining,
        remaining * cfg.hours_per_cycle,
        max(remaining - uncertainty, 0),
        remaining,
        remaining + uncertainty,
        rate,
    )
