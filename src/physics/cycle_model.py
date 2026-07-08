"""Single-spool turbojet Brayton-cycle model with variable gas properties.

Station labels (P2/T2, P3/T3, P4/T4) match the dataset schema:
P2/T2 = compressor exit, P3/T3 = combustor exit/turbine inlet,
P4/T4 = turbine exit."""

from dataclasses import dataclass
import math
from .constants import (
    LOWER_HEATING_VALUE,
    cp_air,
    cp_gas,
    gamma_from_cp,
)
from .component_maps import (
    combustor_efficiency,
    compressor_efficiency,
    compressor_pressure_ratio,
    turbine_efficiency,
)
from .thermodynamics import speed_of_sound, total_pressure, total_temperature


@dataclass(frozen=True)
class CycleInput:
    """Cycle boundary conditions in SI units."""

    altitude_m: float
    mach: float
    ambient_temperature_k: float
    ambient_pressure_pa: float
    rpm: float
    fuel_flow_kg_s: float
    mass_flow_kg_s: float = 25.0
    compressor_health: float = 1.0
    combustor_health: float = 1.0
    turbine_health: float = 1.0


@dataclass(frozen=True)
class CycleState:
    """Reconstructed station state and performance.

    p2/t2 = compressor exit, p3/t3 = combustor exit (turbine inlet),
    p4/t4 = turbine exit — see module docstring for the station convention.
    """

    p2: float
    t2: float
    p3: float
    t3: float
    p4: float
    t4: float
    thrust_n: float
    tsfc_kg_n_s: float
    thermal_efficiency: float
    compressor_work_w: float
    turbine_work_w: float
    energy_residual_w: float


class BraytonCycle:
    """Zero-dimensional turbojet cycle with variable specific heats and component maps."""

    def __init__(
        self,
        max_temperature_k: float = 1900.0,
        design_pr: float = 10.0,
        design_mass_flow: float = 55.0,
        design_rpm: float = 100_000.0,
        thrust_k1: float = 0.102924362,
        thrust_k2: float = 22945.3568,
        thrust_k3: float = 55.3911686,
        thrust_c: float = 15523.6852,
    ) -> None:
        self.max_temperature_k = max_temperature_k
        self.design_pr = design_pr
        self.design_mass_flow = design_mass_flow
        self.design_rpm = design_rpm
        self._thrust_k1 = thrust_k1
        self._thrust_k2 = thrust_k2
        self._thrust_k3 = thrust_k3
        self._thrust_c = thrust_c

    def _compute_mass_flow(self, rpm: float, pamb: float, tamb: float) -> float:
        """Compressor inlet mass flow scaled by corrected-speed relationships."""
        p0 = 101325.0
        t0 = 288.15
        delta = pamb / p0
        theta = tamb / t0
        speed_frac = max(0.1, rpm / self.design_rpm)
        return self.design_mass_flow * speed_frac * delta / max(theta, 0.1)

    def evaluate(self, value: CycleInput) -> CycleState:
        """Reconstruct cycle stations. Returns station pressures, temperatures, thrust, and efficiency."""
        if not (
            0 <= value.mach <= 3
            and value.ambient_temperature_k > 0
            and value.ambient_pressure_pa > 0
            and value.fuel_flow_kg_s >= 0
        ):
            raise ValueError("nonphysical cycle input")
        tamb = value.ambient_temperature_k
        pamb = value.ambient_pressure_pa
        mach = value.mach

        gamma_air = gamma_from_cp(cp_air(tamb))
        t1 = total_temperature(tamb, mach, gamma_air)
        p1 = 0.98 * total_pressure(pamb, mach, gamma_air)

        speed_fraction = max(0.2, min(1.15, value.rpm / self.design_rpm))
        pr = compressor_pressure_ratio(speed_fraction, value.compressor_health, self.design_pr)
        eta_c = compressor_efficiency(speed_fraction, value.compressor_health)
        p2 = p1 * pr

        gamma_c = gamma_from_cp(cp_air(0.5 * (t1 + t1 * pr ** ((gamma_air - 1) / gamma_air))))
        t2s = t1 * pr ** ((gamma_c - 1) / gamma_c)
        t2 = t1 + (t2s - t1) / max(eta_c, 0.01)

        air_flow = self._compute_mass_flow(value.rpm, pamb, tamb)
        cp_comp = cp_air(0.5 * (t1 + t2))
        compressor_work_w = air_flow * cp_comp * (t2 - t1)

        far = value.fuel_flow_kg_s / max(air_flow, 1e-9)
        eta_burn = combustor_efficiency(speed_fraction, value.combustor_health)
        fuel_energy = value.fuel_flow_kg_s * LOWER_HEATING_VALUE * eta_burn
        turbine_flow = air_flow + value.fuel_flow_kg_s
        cp_burn = cp_gas(t2, far)
        t3 = t2 + fuel_energy / (turbine_flow * cp_burn)
        if t3 > self.max_temperature_k:
            t3 = self.max_temperature_k
        p3 = p2 * (0.96 - 0.03 * (1.0 - value.combustor_health))

        eta_t = turbine_efficiency(speed_fraction, value.turbine_health)
        cp_turb_avg = cp_gas(0.5 * (t3 + max(t3 - 400.0, 400.0)), far)
        gamma_t = gamma_from_cp(cp_turb_avg)
        t4 = t3 - compressor_work_w / (turbine_flow * max(cp_turb_avg, 1.0))
        cp_turb_avg = cp_gas(0.5 * (t3 + t4), far)
        t4 = t3 - compressor_work_w / (turbine_flow * max(cp_turb_avg, 1.0))
        cp_turb_avg = cp_gas(0.5 * (t3 + t4), far)
        gamma_t = gamma_from_cp(cp_turb_avg)

        t4s = t3 - (t3 - t4) / max(eta_t, 0.01)
        p4_factor = max(0.05, t4s / max(t3, 1.0))
        p4 = p3 * p4_factor ** (gamma_t / (gamma_t - 1))

        # Calibrated momentum-thrust form: Thrust = k1*RPM*(P4/Pamb) + k2*FuelFlow - k3*V_inf + c.
        # Replaces the isentropic-nozzle equation which did not match this dataset's Thrust.
        flight_velocity = mach * speed_of_sound(tamb)
        pr_nozzle = max(p4, 1.0) / max(pamb, 1.0)
        thrust = max(
            0.0,
            self._thrust_k1 * value.rpm * pr_nozzle
            + self._thrust_k2 * value.fuel_flow_kg_s
            - self._thrust_k3 * flight_velocity
            + self._thrust_c,
        )
        tsfc = value.fuel_flow_kg_s / max(thrust, 1e-9)
        turbine_work = turbine_flow * cp_turb_avg * (t3 - t4)
        exit_gamma = gamma_from_cp(cp_gas(t4, far))
        pressure_ratio_nozzle_isentropic = max(pamb, 1.0) / max(p4, pamb, 1.0)
        exit_velocity_shape = math.sqrt(
            max(
                0.0,
                2
                * cp_turb_avg
                * t4
                * (1.0 - pressure_ratio_nozzle_isentropic ** ((exit_gamma - 1) / exit_gamma)),
            )
        )
        jet_power = 0.5 * turbine_flow * max(exit_velocity_shape**2 - flight_velocity**2, 0)
        efficiency = jet_power / max(fuel_energy, 1e-9)
        return CycleState(
            p2,
            t2,
            p3,
            t3,
            p4,
            t4,
            thrust,
            tsfc,
            min(max(efficiency, 0.0), 1.0),
            compressor_work_w,
            turbine_work,
            turbine_work - compressor_work_w,
        )
