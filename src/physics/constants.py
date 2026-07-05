"""SI thermodynamic constants and engineering defaults."""

import math

# ---- Reference values ----
CP_AIR = 1005.0
CP_GAS = 1150.0
GAMMA_AIR = 1.4
GAMMA_GAS = 1.33
GAS_CONSTANT_AIR = 287.05
LOWER_HEATING_VALUE = 43_000_000.0
SEA_LEVEL_PRESSURE = 101_325.0
SEA_LEVEL_TEMPERATURE = 288.15

# ---- Variable specific heat (temperature-dependent) ----


def cp_air(temperature_k: float) -> float:
    """Temperature-dependent specific heat of air (J/kg·K)."""
    t = max(temperature_k, 200.0)
    return 1002.5 + 0.027 * (t - 250.0) - 3.5e-5 * (t - 250.0) ** 2


def cp_gas(temperature_k: float, far: float = 0.02) -> float:
    """Temperature- and fuel-air-ratio dependent specific heat of combustion gas."""
    t = max(temperature_k, 400.0)
    base = 1050.0 + 0.12 * (t - 500.0) - 2.0e-5 * (t - 500.0) ** 2
    return base + 400.0 * far


def gamma_from_cp(cp: float, gas_constant: float = GAS_CONSTANT_AIR) -> float:
    """Ratio of specific heats from Cp and R."""
    return cp / (cp - gas_constant)


# ---- Standard atmosphere (ISA) ----

_LAPSE_RATE = 0.0065  # K/m
_TROPOPAUSE_ALT = 11_000.0  # m
_G0 = 9.80665  # m/s^2
_M = 0.0289644  # kg/mol molar mass of air
_R_STAR = 8.314462618  # J/(mol·K) universal gas constant


def isa_temperature(altitude_m: float) -> float:
    """International Standard Atmosphere temperature at altitude (K)."""
    if altitude_m <= _TROPOPAUSE_ALT:
        return SEA_LEVEL_TEMPERATURE - _LAPSE_RATE * altitude_m
    return 216.65  # isothermal stratosphere


def isa_pressure(altitude_m: float) -> float:
    """International Standard Atmosphere static pressure at altitude (Pa)."""
    if altitude_m <= _TROPOPAUSE_ALT:
        t = SEA_LEVEL_TEMPERATURE - _LAPSE_RATE * altitude_m
        exponent = _G0 * _M / (_R_STAR * _LAPSE_RATE)
        return SEA_LEVEL_PRESSURE * (t / SEA_LEVEL_TEMPERATURE) ** exponent
    tropo_p = isa_pressure(_TROPOPAUSE_ALT)
    delta_h = altitude_m - _TROPOPAUSE_ALT
    return tropo_p * math.exp(-_G0 * _M * delta_h / (_R_STAR * 216.65))
