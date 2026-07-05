"""Physically consistent single-spool turbojet Brayton-cycle model with variable gas properties."""

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
    """Reconstructed station state and performance."""

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
    """Zero-dimensional turbojet cycle with variable specific heats and realistic component maps."""

    def __init__(self, max_temperature_k: float = 1900.0, design_pr: float = 10.0) -> None:
        self.max_temperature_k = max_temperature_k
        self.design_pr = design_pr

    def evaluate(self, value: CycleInput) -> CycleState:
        """Reconstruct cycle stations and reject nonphysical boundary conditions."""
        if not (
            0 <= value.mach <= 3
            and value.ambient_temperature_k > 0
            and value.ambient_pressure_pa > 0
            and value.mass_flow_kg_s > 0
            and value.fuel_flow_kg_s >= 0
        ):
            raise ValueError("nonphysical cycle input")
        tamb = value.ambient_temperature_k
        pamb = value.ambient_pressure_pa
        mach = value.mach

        gamma_air = gamma_from_cp(cp_air(tamb))
        t2 = total_temperature(tamb, mach, gamma_air)
        p2 = 0.98 * total_pressure(pamb, mach, gamma_air)

        speed_fraction = max(0.2, min(1.15, value.rpm / 100_000.0))
        pr = compressor_pressure_ratio(speed_fraction, value.compressor_health, self.design_pr)
        eta_c = compressor_efficiency(speed_fraction, value.compressor_health)
        p3 = p2 * pr

        gamma_c = gamma_from_cp(cp_air(0.5 * (t2 + t2 * pr ** ((gamma_air - 1) / gamma_air))))
        t3s = t2 * pr ** ((gamma_c - 1) / gamma_c)
        t3 = t2 + (t3s - t2) / max(eta_c, 0.01)

        air_flow = value.mass_flow_kg_s
        far = value.fuel_flow_kg_s / max(air_flow, 1e-9)
        eta_burn = combustor_efficiency(speed_fraction, value.combustor_health)
        fuel_energy = value.fuel_flow_kg_s * LOWER_HEATING_VALUE * eta_burn
        turbine_flow = air_flow + value.fuel_flow_kg_s
        cp_burn = cp_gas(t3, far)
        t_turbine_in = t3 + fuel_energy / (turbine_flow * cp_burn)
        if t_turbine_in > self.max_temperature_k:
            t_turbine_in = self.max_temperature_k

        # Compressor work
        cp_comp = cp_air(0.5 * (t2 + t3))
        compressor_work_w = air_flow * cp_comp * (t3 - t2)

        # Turbine: spool power balance determines T4
        eta_t = turbine_efficiency(speed_fraction, value.turbine_health)
        cp_turb_avg = cp_gas(0.5 * (t_turbine_in + 0.5 * (t_turbine_in + 500.0)), far)
        gamma_t = gamma_from_cp(cp_turb_avg)
        # turbine_flow * cp_turb * (T4_in - T4) * eta_t = compressor_work
        t4 = t_turbine_in - compressor_work_w / (turbine_flow * max(eta_t, 0.01) * max(cp_turb_avg, 1.0))
        # Converge on gas Cp at actual T4
        cp_turb_avg = cp_gas(0.5 * (t_turbine_in + t4), far)
        t4 = t_turbine_in - compressor_work_w / (turbine_flow * max(eta_t, 0.01) * max(cp_turb_avg, 1.0))
        # Final gas Cp at actual conditions
        cp_turb_avg = cp_gas(0.5 * (t_turbine_in + t4), far)
        # Turbine exit pressure from expansion ratio and efficiency
        p4_factor = max(0.05, 1.0 - (t_turbine_in - t4) / max(t_turbine_in, 1.0))
        p4 = 0.95 * p3 * p4_factor ** (gamma_t / (gamma_t - 1))
        exit_temp = t4
        exit_gamma = gamma_from_cp(cp_gas(exit_temp, far))
        pressure_ratio_nozzle = max(pamb, 1.0) / max(p4, pamb, 1.0)
        exit_velocity = math.sqrt(
            max(
                0.0,
                2 * cp_turb_avg * t4 * (1.0 - pressure_ratio_nozzle ** ((exit_gamma - 1) / exit_gamma)),
            )
        )
        flight_velocity = mach * speed_of_sound(tamb)
        thrust = max(0.0, turbine_flow * exit_velocity - air_flow * flight_velocity)
        tsfc = value.fuel_flow_kg_s / max(thrust, 1e-9)
        turbine_work = turbine_flow * cp_turb_avg * (t_turbine_in - t4)
        jet_power = 0.5 * turbine_flow * max(exit_velocity**2 - flight_velocity**2, 0)
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
