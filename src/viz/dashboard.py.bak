"""Professional Streamlit operations dashboard."""

from pathlib import Path

import pandas as pd
import streamlit as st
from src.dataset.loader import _COLUMN_ALIASES
from src.digital_twin.engine import DigitalTwin
from src.digital_twin.fleet import rank_fleet
from src.explainability.root_cause import analyze_faults, analyze_scenario
from src.faults.injection import FaultInjector, FaultSpec, FaultType
from src.maintenance.decision_engine import MaintenanceDecisionEngine
from src.simulation.what_if import ScenarioAdjustment, ScenarioSimulator
from src.viz.engine_animation import engine_schematic
from src.viz.plots import health_gauge, trend

st.set_page_config(page_title="Turbojet Digital Twin", page_icon="✈", layout="wide")
st.title("Four-Stage Turbojet Digital Twin")
page = st.sidebar.radio(
    "Workspace",
    [
        "Overview",
        "Engine Health",
        "Performance",
        "RUL",
        "Maintenance",
        "What-If Simulator",
        "Fault Injection",
        "Root Cause Analysis",
        "Maintenance Options",
        "Fleet",
        "Model Explainability",
        "Upload & Inference",
        "Settings",
    ],
)
uploaded = st.sidebar.file_uploader("Sensor dataset", type="csv")
if uploaded is None:
    st.info("Upload an official-schema CSV to begin inference.")
else:
    try:
        data = pd.read_csv(uploaded)
        data = data.rename(columns={k: v for k, v in _COLUMN_ALIASES.items() if k in data.columns})
        model_path = st.sidebar.text_input("Model artifact", "models/best_model.joblib")

        outputs = []
        for engine_id, group in data.groupby("EngineID", sort=False):
            twin = DigitalTwin(str(engine_id))
            if Path(model_path).exists():
                twin.load_model(model_path)
            result = twin.batch_predict(group)
            result["EngineID"] = engine_id
            result["Cycle"] = group["Cycle"].reset_index(drop=True)
            outputs.append(result)
        output = pd.concat(outputs, ignore_index=True)
        output = output.loc[:, ~output.columns.duplicated()]

        latest = output.iloc[-1]
        latest_per_engine = output.sort_values("Cycle").groupby("EngineID", as_index=False).tail(1)

        if page == "Overview":
            a, b, c, d = st.columns(4)
            a.metric("Health", f"{latest['OverallHealth']:.1%}")
            b.metric("Thrust", f"{latest['Thrust']:.0f} N")
            c.metric("RUL", f"{latest['RULCycles']:.0f} cycles")
            d.metric("Risk", str(latest["RiskLevel"]).upper())
            left, right = st.columns(2)
            with left:
                st.plotly_chart(health_gauge(float(latest["OverallHealth"])), width="stretch")
            with right:
                schematic_health = {
                    "CompressorHealth": float(latest["CompressorHealth"]),
                    "CombustorHealth": float(latest["CombustorHealth"]),
                    "TurbineHealth": float(latest["TurbineHealth"]),
                }
                st.plotly_chart(engine_schematic(schematic_health), width="stretch")

        elif page in {"Engine Health", "Performance", "RUL"}:
            columns = (
                ["CompressorHealth", "CombustorHealth", "TurbineHealth", "OverallHealth"]
                if page == "Engine Health"
                else (
                    ["Thrust", "TSFC"]
                    if page == "Performance"
                    else ["RULCycles", "FailureProbability"]
                )
            )
            single_engine = output[output["EngineID"] == latest["EngineID"]].reset_index(drop=True)
            st.plotly_chart(trend(single_engine, columns), width="stretch")

        elif page == "Maintenance":
            st.subheader(str(latest["Maintenance"]))
            st.write(f"Risk level: {latest['RiskLevel']}")

        elif page == "What-If Simulator":
            st.subheader("Scenario simulator")
            baseline_row = group.iloc[-1].to_dict()
            c1, c2, c3 = st.columns(3)
            fuel_flow = c1.slider(
                "Fuel flow (kg/s)", 0.0, 5.0, float(baseline_row.get("FuelFlow", 1.0)), 0.01
            )
            rpm = c2.slider("RPM", 10_000.0, 120_000.0, float(baseline_row.get("RPM", 80_000.0)))
            tamb = c3.slider(
                "Ambient temperature (K)", 200.0, 350.0, float(baseline_row.get("Tamb", 288.0))
            )
            c4, c5, c6 = st.columns(3)
            pamb = c4.slider(
                "Ambient pressure (Pa)", 40_000.0, 105_000.0, float(baseline_row.get("Pamb", 101_325.0))
            )
            compressor_eff = c5.slider("Compressor efficiency", 0.3, 1.0, 1.0)
            turbine_eff = c6.slider("Turbine efficiency", 0.3, 1.0, 1.0)
            noise = st.slider("Sensor noise std (fraction)", 0.0, 0.2, 0.0, 0.01)

            adjustment = ScenarioAdjustment(
                fuel_flow_kg_s=fuel_flow,
                rpm=rpm,
                ambient_temperature_k=tamb,
                ambient_pressure_pa=pamb,
                compressor_efficiency=compressor_eff,
                turbine_efficiency=turbine_eff,
                sensor_noise_std=noise,
            )
            try:
                comparison = ScenarioSimulator().run(baseline_row, adjustment)
                st.session_state["scenario_comparison"] = comparison
                st.session_state["scenario_inputs"] = (baseline_row, adjustment)
                before, after = st.columns(2)
                for label, snapshot, col in (
                    ("Before", comparison.baseline, before),
                    ("After", comparison.adjusted, after),
                ):
                    with col:
                        st.markdown(f"**{label}**")
                        st.metric("Health", f"{snapshot.overall_health:.1%}")
                        st.metric("RUL (cycles)", f"{snapshot.remaining_useful_life_cycles:.0f}")
                        st.metric("Failure probability", f"{snapshot.failure_probability:.1%}")
                        st.metric("Thrust (N)", f"{snapshot.thrust_n:.0f}")
                        st.metric("TSFC", f"{snapshot.tsfc_kg_n_s:.5f}")
                        st.metric("Confidence", f"{snapshot.confidence:.1%}")
                st.write("Delta:", comparison.delta)
            except ValueError as error:
                st.error(str(error))

        elif page == "Fault Injection":
            st.subheader("Fault injection controls")
            selected_faults = st.multiselect(
                "Active faults", [ft.value for ft in FaultType], default=[]
            )
            specs: list[FaultSpec] = []
            for fault_name in selected_faults:
                fault_type = FaultType(fault_name)
                severity = st.slider(f"{fault_name} severity", 0.0, 1.0, 0.5, key=f"sev_{fault_name}")
                target = None
                if fault_type in {FaultType.SENSOR_DRIFT, FaultType.SENSOR_BIAS}:
                    target = st.selectbox(
                        f"{fault_name} target sensor",
                        ["T2", "T3", "T4", "P2", "P3", "P4"],
                        key=f"target_{fault_name}",
                    )
                specs.append(FaultSpec(fault_type, severity, target))
            injector = FaultInjector(specs)
            twin_with_faults = DigitalTwin(str(latest["EngineID"]))
            if Path(model_path).exists():
                twin_with_faults.load_model(model_path)
            twin_with_faults.fault_injector = injector
            faulted = twin_with_faults.batch_predict(group)
            st.write("Predictions with active faults applied:")
            st.dataframe(faulted, width="stretch")

        elif page == "Root Cause Analysis":
            st.subheader("Root cause analysis")
            comparison = st.session_state.get("scenario_comparison")
            if comparison is None:
                st.info("Run the What-If Simulator first to generate a comparison to explain.")
            else:
                baseline_row, adjustment = st.session_state["scenario_inputs"]
                baseline_inputs = {
                    "FuelFlow": baseline_row.get("FuelFlow", 0.0),
                    "RPM": baseline_row.get("RPM", 0.0),
                    "Tamb": baseline_row.get("Tamb", 0.0),
                    "Pamb": baseline_row.get("Pamb", 0.0),
                    "compressor_efficiency": 1.0,
                    "turbine_efficiency": 1.0,
                }
                adjusted_inputs = {
                    "FuelFlow": adjustment.fuel_flow_kg_s,
                    "RPM": adjustment.rpm,
                    "Tamb": adjustment.ambient_temperature_k,
                    "Pamb": adjustment.ambient_pressure_pa,
                    "compressor_efficiency": adjustment.compressor_efficiency or 1.0,
                    "turbine_efficiency": adjustment.turbine_efficiency or 1.0,
                }
                report = analyze_scenario(
                    baseline_inputs, adjusted_inputs, comparison.delta["overall_health"]
                )
                st.write(report.summary)
                for factor in report.factors:
                    st.write(f"- **{factor.factor}** ({factor.contribution:+.3f}): {factor.explanation}")
                st.write("Causal chain:")
                st.write(" ".join(report.causal_chain))

        elif page == "Maintenance Options":
            st.subheader("Ranked maintenance options")
            options = MaintenanceDecisionEngine().generate_options(
                float(latest["OverallHealth"]),
                float(latest["RULCycles"]),
                float(latest["FailureProbability"]),
            )
            options_df = pd.DataFrame([vars(option) for option in options])
            st.dataframe(options_df, width="stretch")
            st.success(f"Top recommendation: {options[0].action}")
            st.caption(options[0].rationale)

        elif page == "Fleet":
            fleet_input = latest_per_engine.drop(columns=["engine_id"], errors="ignore").rename(columns={"EngineID": "engine_id"})
            ranked = rank_fleet(fleet_input)
            st.dataframe(ranked, width="stretch")

        elif page == "Model Explainability":
            twin_for_importance = DigitalTwin()
            if Path(model_path).exists():
                twin_for_importance.load_model(model_path)
            model = twin_for_importance.model
            if model is None:
                st.warning("No model loaded — cannot compute feature importances.")
            else:
                pipeline = model.pipeline
                estimator = pipeline.steps[-1][1] if hasattr(pipeline, "steps") else pipeline
                if hasattr(estimator, "feature_importances_"):
                    importances = pd.DataFrame(
                        {
                            "Feature": model.feature_names,
                            "Importance": estimator.feature_importances_,
                        }
                    ).sort_values("Importance", ascending=False)
                    st.bar_chart(importances.set_index("Feature"))
                else:
                    st.info("Loaded model does not expose feature importances.")

        else:
            st.dataframe(output, width="stretch")

        st.download_button("Export predictions", output.to_csv(index=False), "predictions.csv")
    except (ValueError, KeyError) as error:
        st.error(str(error))

