import numpy as np
import pytest

from src.estimation.ekf import ExtendedKalmanFilter
from src.estimation.particle_filter import ParticleFilter
from src.estimation.state_estimator import StateEstimator


def test_ekf_converges() -> None:
    ekf = ExtendedKalmanFilter(np.array([0.0]), np.eye(1), np.eye(1) * 0.01, np.eye(1) * 0.1)
    for _ in range(20):
        ekf.predict(lambda x: x, np.eye(1))
        ekf.update(np.array([1.0]), lambda x: x, np.eye(1))
    assert ekf.state[0] == pytest.approx(1, abs=0.05)


def test_particle_filter_tracks_measurement() -> None:
    particles = np.linspace(-2, 2, 1000).reshape(-1, 1)
    pf = ParticleFilter(particles)
    state = pf.update(np.array([0.5]), lambda x: x, np.array([0.1]))
    assert abs(state[0] - 0.5) < 0.05


def test_state_estimator_monotonic() -> None:
    """StateEstimator must enforce monotonic health degradation."""
    estimator = StateEstimator("ekf")
    # Two updates with increasing health observation — filter should clamp to decreasing
    s1 = estimator.update(np.array([0.9, 0.9, 0.9, 0.9]))
    s2 = estimator.update(np.array([0.95, 0.95, 0.95, 0.95]))
    for i in range(4):
        assert s2[i] <= s1[i] + 1e-6, f"health component {i} increased (monotonicity violation)"
    assert all(0 <= s2[i] <= 1 for i in range(4)), "health out of [0, 1] range"


def test_state_estimator_no_cycle_leakage() -> None:
    """StateEstimator does not accept Cycle/EngineID as inputs — pure health vector."""
    estimator = StateEstimator("ekf")
    with pytest.raises((ValueError, TypeError, IndexError)):
        # Should not accept additional dimensions for Cycle/EngineID
        estimator.update(np.array([0.9, 0.9, 0.9, 0.9, 1.0]))
