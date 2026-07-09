# Chapter 3: Equations

[← Chapter 2: Theory](Theory.md) · [Chapter 4: Architecture →](ARCHITECTURE.md) · [References](../research/references.bib)

---

## 1 ISA Atmosphere

Source: [NOAA/NASA/USAF 1976], [Mattingly 2002 §3.2]

Temperature (K):

```
T(h) = T_0 - L · h                  h ≤ 11 000 m
T(h) = 216.65                       h > 11 000 m

T_0 = 288.15 K,  L = 0.0065 K/m
```

Pressure (Pa):

```
P(h) = P_0 · (T(h) / T_0)^(gM / RL)          h ≤ 11 000 m
P(h) = P_tropo · exp(-gMΔh / RT_tropo)      h > 11 000 m

P_0 = 101 325 Pa,  g = 9.80665 m/s²
M = 0.0289644 kg/mol,  R = 8.31446 J/(mol·K)
```

## 2 Inlet (Ram Compression)

Source: [Mattingly 2002 §4.3], [Rolls-Royce 1996 §3]

Total temperature and pressure at flight Mach M:

```
T_t = T_s · (1 + ½(γ - 1)M²)
P_t = P_s · (1 + ½(γ - 1)M²)^(γ/(γ-1))

γ_air = cp_air / (cp_air - R_air)
```

Inlet recovery factor: `P_1 = 0.98 · P_t`.

## 3 Compressor

Source: [Walsh & Fletcher 2004 §3.4], [Mattingly 2002 §5.6]

Pressure ratio and temperature rise:

```
PR = f_comp(N/N_design, health_comp)

T_2s = T_1 · PR^((γ_c - 1)/γ_c)
T_2 = T_1 + (T_2s - T_1) / η_comp

η_comp(N/N_des, h_c) = [0.87 - 0.30(s-0.88)² - 0.10(s-0.88)⁴] · (0.85 + 0.15·h_c)
PR_comp(N/N_des, h_c) = 1 + (PR_design - 1) · q(s) · h_c
  q(s) = (1 + 8.5s²·⁵ - 2.5s⁵) / 7,  s = N/N_design
```

Compressor work:
```
W_c = m_dot_air · cp_air · (T_2 - T_1)
```

## 4 Combustor

Source: [Walsh & Fletcher 2004 §4.3], [Mattingly 2002 §5.10]

Combustion temperature rise from fuel energy:

```
η_burn = f_comb(N/N_design, health_comb)
E_fuel = m_dot_fuel · LHV · η_burn
T_3 = T_2 + E_fuel / ((m_dot_air + m_dot_fuel) · cp_gas(T_2, FAR))

FAR = m_dot_fuel / m_dot_air
LHV = 43 MJ/kg

IF T_3 > T_max:  T_3 = T_max
P_3 = P_2 · (0.96 - 0.03 · (1 - health_comb))
```

## 5 Turbine

Source: [Walsh & Fletcher 2004 §5.4], [Mattingly 2002 §5.12]

Expansion work matching compressor demand:

```
η_turb = f_turb(N/N_design, health_turb)
η_turb(s, h_t) = [0.90 - 0.25(s-0.90)² - 0.08(s-0.90)⁴] · (0.85 + 0.15·h_t)

First estimate T_4 from energy balance:
T_4_est = T_3 - W_c / ((m_dot_air + m_dot_fuel) · cp_gas(T_3, FAR))

Iterate once with mean cp:
cp_avg = cp_gas(½(T_3 + T_4_est), FAR)
T_4 = T_3 - W_c / ((m_dot_air + m_dot_fuel) · cp_avg)

Turbine pressure ratio from isentropic relation:
T_4s = T_3 - (T_3 - T_4) / η_turb
P_4 = P_3 · (T_4s / T_3)^(γ_t / (γ_t - 1))
```

## 6 Calibrated Thrust

Source: [Walsh & Fletcher 2004 §6.2], data-calibrated

```
Thrust = k₁ · RPM · (P_4 / P_amb) + k₂ · m_dot_fuel - k₃ · V_flight + C
```

Where `k₁, k₂, k₃, C` are data-calibrated coefficients. TSFC:

```
TSFC = m_dot_fuel / Thrust
```

## 7 Health Fusion

Source: safety-conservative geometric mean design

```
OverallHealth = exp(0.35·ln(H_comp) + 0.25·ln(H_comb) + 0.40·ln(H_turb))
```

Each component health is clipped to `[1e-8, 1]` before the log. The geometric mean ensures a single failed subsystem drives overall health to zero.

## 8 Extended Kalman Filter

Source: [Thrun 2005 §3.3], [Papoulis 2002 §8]

State vector: `x = [H_comp, H_comb, H_turb, H_overall]ᵀ`

**Predict:**
```
x⁻ = clip(x⁺_prev - δ, 0, 1),    δ = 1e-4
P⁻ = F · P⁺_prev · Fᵀ + Q
F = I,  Q = I · 1e-5
```

**Update:**
```
y = z - x⁻                    (innovation)
S = H · P⁻ · Hᵀ + R           (innovation covariance)
K = P⁻ · Hᵀ · S⁻¹            (Kalman gain)
x⁺ = x⁻ + K · y              (state update)
P⁺ = (I - K·H) · P⁻ · (I - K·H)ᵀ + K·R·Kᵀ    (Joseph form)

H = I,  R = I · 0.01
```

**Monotonicity clamp:**
```
FOR EACH i WHERE x⁺_i > x_prev_i:
    x⁺_i = x_prev_i
    P⁺[i,:] = P⁺[:,i] = 0
    P⁺[i,i] = Q[i,i]
```

## 9 Feature Engineering

### 9.1 Thermodynamic Ratios and Deltas

```
CompressorPR   = P_3 / P_2
TurbinePR      = P_4 / P_3
CompressorΔT   = T_3 - T_2
TurbineΔT      = T_4 - T_3
TempRatioComp  = T_3 / T_2
TempRatioTurb  = T_4 / T_3
OverallPR      = P_3 / P_amb
BurnerTempRise = T_3 - T_2
```

### 9.2 Normalised Flow Terms

```
CorrectedRPM      = RPM / √(T_2 / 288.15)
CorrectedFuelFlow = m_dot_fuel / √(T_amb / 288.15)
FuelPerRPM        = m_dot_fuel / RPM
```

### 9.3 Quadratic and Interaction Terms

```
FlowSquared   = m_dot_fuel²
RPMSquared    = (RPM / 100 000)²
FuelFlowRPM   = m_dot_fuel · (RPM / 100 000)
```

### 9.4 Physics Residuals

Source: [Walsh & Fletcher 2004 §12.3], residual learning framework

For each station variable S ∈ {P2, T2, P3, T3, P4, T4}:

```
Res_S = (S_measured - S_healthy) / S_healthy
```

Where `S_healthy` is the Brayton-cycle prediction at the same flight condition with fully healthy components. These residuals are the key degradation signal: a healthy engine has residuals ≈ 0, while degradation causes systematic deviations.

## 10 Remaining Useful Life

Source: [Box & Jenkins 1970], [Maciejowski 2002 §4]

Windowed linear degradation trend:

```
window = min(n_cycles, 50)
X = cycles[-window:]
Y = health[-window:]
slope, intercept = polyfit(X, Y, 1)

degradation_rate = max(-slope, 1e-6)          [cycles⁻¹]
RUL = max((Y[-1] - θ_fail) / rate, 0)         [cycles]

θ_fail = 0.3
```

Uncertainty (90% confidence):

```
σ_res = std(Y - (slope·X + intercept))
RUL_90 = 1.645 · σ_res / rate
```

## 11 Conformal Prediction

Source: [Shafer & Vovk 2008], [Angelopoulos & Bates 2023]

Split conformal calibration on held-out set:

```
R_i = |y_i - ŷ_i|           (absolute residuals on calibration)
q̂ = Q(R, ⌈(n+1)·α⌉/n)      (calibrated quantile, α = 0.9)
Lower = ŷ - q̂
Upper = ŷ + q̂
```

Marginal coverage guarantee: `P(y ∈ [ŷ - q̂, ŷ + q̂]) ≥ α` asymptotically.

## 12 Failure Probability

Logistic risk model with horizon:

```
health_term     = max(θ_health - H, 0) · a
horizon_term    = max(H - remaining, 0) / H · b
score           = clamp(health_term + horizon_term, -40, 40)
P(failure)      = 1 / (1 + exp(-score))

θ_health = 0.3,  H = horizon_cycles (default 25)
```

Default coefficients (fallback, before calibration): `a = 12, b = 5`.

Calibration fits `a, b` via logistic regression on historical trajectories:

```
y_t = 1 if min(health[t : t+H]) < θ_cal else 0
θ_cal = 0.7
features: [max(θ_cal - health_t, 0), max(H - (N - t), 0) / H]
```

## 13 Engineered Features — Complete List

| Feature | Formula | Type |
|---------|---------|------|
| Altitude | — | raw |
| Mach | — | raw |
| Tamb | — | raw |
| Pamb | — | raw |
| RPM | — | raw |
| FuelFlow | — | raw |
| P2–T4 | — | raw (6) |
| CompressorPR | P3/P2 | ratio |
| TurbinePR | P4/P3 | ratio |
| CompressorDeltaT | T3 − T2 | delta |
| TurbineDeltaT | T4 − T3 | delta |
| FuelPerRPM | FuelFlow / RPM | ratio |
| CorrectedRPM | RPM / √(T2 / 288.15) | normalised |
| TempRatioComp | T3 / T2 | ratio |
| TempRatioTurb | T4 / T3 | ratio |
| OverallPR | P3 / Pamb | ratio |
| BurnerTempRise | T3 − T2 | delta |
| FlowSquared | FuelFlow² | quadratic |
| RPMSquared | (RPM / 10⁵)² | quadratic |
| FuelFlowRPM | FuelFlow · RPM / 10⁵ | interaction |
| CorrectedFuelFlow | FuelFlow / √(Tamb / 288.15) | normalised |
| ResP2–ResT4 | (S − S_healthy) / S_healthy | residual (6) |

---

[← Chapter 2: Theory](Theory.md) · [Chapter 4: Architecture →](ARCHITECTURE.md) · [References](../research/references.bib)
