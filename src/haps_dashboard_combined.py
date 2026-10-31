"""
HAPS Link-Quality Prediction Dashboard — Combined validation edition
Team RVTS

Includes:
1) ITU-model/link-budget sanity checks
2) Random Forest training and held-out trajectory evaluation
3) Scenario comparisons across distance and rain-rate settings
4) Automatic trend checks and downloadable result tables

Keep this file beside haps_link_quality_itu_complete.py.
Run from D:\\aiori:
    py -m streamlit run haps_dashboard_combined.py

Important: the data are simulated. These checks verify consistency and expected
trends, not real-world HAPS performance. P.528 is an aeronautical model whose
applicability to HAPS should be justified. P.838 supplies specific attenuation;
this script uses the project's assumed effective rain path length.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from haps_link_quality_itu_complete import (
        calculate_link, HAPS_ALTITUDE_M, FREQUENCY_MHZ, TX_POWER_DBM,
        TX_ANTENNA_GAIN_DBI, RX_ANTENNA_GAIN_DBI, BANDWIDTH_HZ,
        NOISE_FIGURE_DB, RAIN_PATH_LENGTH_KM, OUTPUT_DIR,
    )
except Exception as exc:
    st.set_page_config(page_title="HAPS Validation Dashboard", layout="wide")
    st.error("Cannot import haps_link_quality_itu_complete.py. Keep both files in D:\\aiori and check the required packages.")
    st.exception(exc)
    st.stop()

st.set_page_config(page_title="HAPS Link-Quality Validation", page_icon="📡", layout="wide")
st.title("📡 HAPS Link-Quality Prediction — Validation Dashboard")
st.caption("Team RVTS | P.525-5 + P.528-5 + P.838-3 | Simulation and sanity checks")

with st.sidebar:
    st.header("Interactive link settings")
    distance_km = st.slider("Horizontal ground distance (km)", 0.1, 50.0, 10.0, 0.5)
    rain_rate = st.slider("Rain rate (mm/h)", 0.0, 100.0, 0.0, 1.0)
    st.divider()
    st.subheader("Fixed assumptions")
    st.write(f"HAPS altitude: {HAPS_ALTITUDE_M / 1000:.1f} km")
    st.write(f"Frequency: {FREQUENCY_MHZ / 1000:.2f} GHz")
    st.write(f"Transmit power: {TX_POWER_DBM:.1f} dBm")
    st.write(f"TX/RX antenna gains: {TX_ANTENNA_GAIN_DBI:.1f}/{RX_ANTENNA_GAIN_DBI:.1f} dBi")
    st.write(f"Bandwidth: {BANDWIDTH_HZ / 1e6:.1f} MHz")
    st.write(f"Noise figure: {NOISE_FIGURE_DB:.1f} dB")
    st.write(f"Assumed effective rain path: {RAIN_PATH_LENGTH_KM:.1f} km")

@st.cache_data(show_spinner=False)
def link_at(distance, rain):
    return calculate_link(float(distance), float(rain))

try:
    current = link_at(distance_km, rain_rate)
except Exception as exc:
    st.error("A propagation calculation failed for the selected settings.")
    st.exception(exc)
    st.stop()

st.header("1. Current link estimate")
a, b, c, d = st.columns(4)
a.metric("Received power", f"{current['received_power_dbm']:.2f} dBm")
b.metric("SNR", f"{current['snr_db']:.2f} dB")
c.metric("P.528 basic loss", f"{current['p528_basic_loss_db']:.2f} dB")
d.metric("Estimated rain loss", f"{current['rain_attenuation_db']:.3f} dB")

with st.expander("View detailed link budget"):
    detail = pd.DataFrame([
        ("Horizontal distance", f"{current['horizontal_distance_km']:.2f} km"),
        ("Slant distance", f"{current['slant_distance_km']:.2f} km"),
        ("P.525 free-space loss", f"{current['p525_fspl_db']:.2f} dB"),
        ("P.528 basic transmission loss", f"{current['p528_basic_loss_db']:.2f} dB"),
        ("P.838 specific rain attenuation", f"{current['p838_specific_rain_attenuation_db_per_km']:.5f} dB/km"),
        ("Assumed effective rain path", f"{current['assumed_rain_path_length_km']:.2f} km"),
        ("Estimated rain attenuation", f"{current['rain_attenuation_db']:.4f} dB"),
        ("Total loss used in link budget", f"{current['total_loss_using_p528_plus_rain_db']:.2f} dB"),
        ("Noise power", f"{current['noise_power_dbm']:.2f} dBm"),
    ], columns=["Quantity", "Value"])
    st.dataframe(detail, hide_index=True, width="stretch")

st.header("2. Automated propagation sanity checks")
st.caption("These checks look for expected trends; they do not certify that the selected propagation models are appropriate for every HAPS deployment.")
try:
    near = link_at(2.0, 0.0)
    far = link_at(40.0, 0.0)
    dry = link_at(distance_km, 0.0)
    wet = link_at(distance_km, max(10.0, rain_rate))
    checks = [
        ("All key outputs are finite", all(np.isfinite(current[k]) for k in [
            "p525_fspl_db", "p528_basic_loss_db", "p838_specific_rain_attenuation_db_per_km",
            "rain_attenuation_db", "received_power_dbm", "snr_db"
        ]), "Loss, received power and SNR are finite numbers."),
        ("Distance trend: received power", far["received_power_dbm"] < near["received_power_dbm"], "A longer link should generally have lower received power."),
        ("Distance trend: SNR", far["snr_db"] < near["snr_db"], "With fixed bandwidth/noise, lower received power should reduce SNR."),
        ("Rain trend: attenuation", wet["rain_attenuation_db"] >= dry["rain_attenuation_db"], "Higher rain rate should not reduce the calculated rain attenuation."),
        ("Link-budget identity", abs(current["snr_db"] - (current["received_power_dbm"] - current["noise_power_dbm"])) < 1e-6, "SNR should equal received power minus noise power in dB."),
        ("Rain-loss identity", abs(current["rain_attenuation_db"] - current["p838_specific_rain_attenuation_db_per_km"] * RAIN_PATH_LENGTH_KM) < 1e-5, "Total rain loss uses specific attenuation multiplied by the assumed path length."),
    ]
    checks_df = pd.DataFrame([
        {"Check": name, "Status": "PASS" if passed else "REVIEW", "Meaning": explanation}
        for name, passed, explanation in checks
    ])
    passed_count = sum(item[1] for item in checks)
    st.metric("Sanity checks passed", f"{passed_count}/{len(checks)}")
    st.dataframe(checks_df, hide_index=True, width="stretch")
    st.download_button("Download sanity-check results (CSV)", checks_df.to_csv(index=False).encode("utf-8"),
                       "haps_sanity_checks.csv", "text/csv")
except Exception as exc:
    st.warning(f"Could not complete all sanity checks: {exc}")

st.header("3. Scenario comparison")
st.caption("Compare clear-sky and rain cases at the selected distance. Rain attenuation uses the stated effective-path assumption.")
scenario_rows = []
for label, rate in [("Clear sky", 0.0), ("Light rain", 5.0), ("Moderate rain", 25.0), ("Heavy rain", 50.0)]:
    try:
        r = link_at(distance_km, rate)
        scenario_rows.append({
            "Scenario": label,
            "Rain rate (mm/h)": rate,
            "P.838 specific attenuation (dB/km)": r["p838_specific_rain_attenuation_db_per_km"],
            "Estimated rain loss (dB)": r["rain_attenuation_db"],
            "Received power (dBm)": r["received_power_dbm"],
            "SNR (dB)": r["snr_db"],
        })
    except Exception as exc:
        st.warning(f"Scenario '{label}' failed: {exc}")
scenario_df = pd.DataFrame(scenario_rows)
if not scenario_df.empty:
    st.dataframe(scenario_df, hide_index=True, width="stretch")
    st.download_button("Download scenario comparison (CSV)", scenario_df.to_csv(index=False).encode("utf-8"),
                       "haps_scenario_comparison.csv", "text/csv")

st.header("4. Link quality versus distance")
try:
    distances = np.linspace(0.1, 50.0, 40)
    curve_df = pd.DataFrame([link_at(round(float(x), 3), rain_rate) for x in distances])
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(curve_df["horizontal_distance_km"], curve_df["received_power_dbm"])
        ax.set_xlabel("Horizontal distance (km)")
        ax.set_ylabel("Received power (dBm)")
        ax.set_title("Received power versus distance")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(curve_df["horizontal_distance_km"], curve_df["snr_db"])
        ax.set_xlabel("Horizontal distance (km)")
        ax.set_ylabel("SNR (dB)")
        ax.set_title("SNR versus distance")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    st.download_button("Download distance-curve data (CSV)", curve_df.to_csv(index=False).encode("utf-8"),
                       "haps_distance_curve.csv", "text/csv")
except Exception as exc:
    st.warning(f"Could not draw the full distance curve: {exc}")

st.header("5. Random Forest training and held-out evaluation")
dataset_path = Path(OUTPUT_DIR) / "haps_itu_synthetic_dataset.csv"
if not dataset_path.exists():
    st.warning("Synthetic dataset not found. Run `py haps_link_quality_itu_complete.py` once from D:\\aiori, then refresh this dashboard.")
else:
    try:
        data = pd.read_csv(dataset_path)
        features = [
            "horizontal_distance_km", "slant_distance_km", "frequency_mhz",
            "rain_rate_mm_h", "p525_fspl_db", "p528_basic_loss_db",
            "p838_specific_rain_attenuation_db_per_km", "rain_attenuation_db",
        ]
        target = "snr_db"
        required = set(features + [target, "trajectory_id", "scenario"])
        if not required.issubset(data.columns):
            st.error("The dataset does not contain all expected columns. Run haps_link_quality_itu_complete.py again.")
        elif data.empty:
            st.error("The dataset is empty.")
        else:
            missing = int(data[list(required)].isna().sum().sum())
            st.write(f"Dataset rows: **{len(data):,}** | Trajectories: **{data['trajectory_id'].nunique():,}** | Missing values in required fields: **{missing}**")
            if missing:
                st.warning("Missing values exist in required fields; inspect the dataset before relying on the metrics.")
            X = data[features]
            y = data[target]
            groups = data["trajectory_id"]
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
            train_idx, test_idx = next(splitter.split(X, y, groups=groups))
            model = RandomForestRegressor(n_estimators=150, random_state=42, min_samples_leaf=2, n_jobs=-1)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[test_idx])
            mae = mean_absolute_error(y.iloc[test_idx], pred)
            rmse = np.sqrt(mean_squared_error(y.iloc[test_idx], pred))
            r2 = r2_score(y.iloc[test_idx], pred)
            m1, m2, m3 = st.columns(3)
            m1.metric("Held-out MAE", f"{mae:.4f} dB")
            m2.metric("Held-out RMSE", f"{rmse:.4f} dB")
            m3.metric("Held-out R²", f"{r2:.4f}")
            st.caption("The split holds out entire trajectory IDs. These are tests against simulated labels, not measured field data.")
            live = pd.DataFrame([{
                "horizontal_distance_km": current["horizontal_distance_km"],
                "slant_distance_km": current["slant_distance_km"],
                "frequency_mhz": current["frequency_mhz"],
                "rain_rate_mm_h": current["rain_rate_mm_h"],
                "p525_fspl_db": current["p525_fspl_db"],
                "p528_basic_loss_db": current["p528_basic_loss_db"],
                "p838_specific_rain_attenuation_db_per_km": current["p838_specific_rain_attenuation_db_per_km"],
                "rain_attenuation_db": current["rain_attenuation_db"],
            }])[features]
            st.metric("Random Forest predicted SNR for current settings", f"{float(model.predict(live)[0]):.2f} dB")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            actual = y.iloc[test_idx].to_numpy()
            ax.scatter(actual, pred, alpha=0.4)
            low = min(float(actual.min()), float(pred.min()))
            high = max(float(actual.max()), float(pred.max()))
            ax.plot([low, high], [low, high], linestyle="--")
            ax.set_xlabel("Simulated SNR (dB)")
            ax.set_ylabel("Predicted SNR (dB)")
            ax.set_title("Held-out trajectories: actual versus predicted")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
            importance = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
            with st.expander("Feature importance"):
                st.dataframe(importance, hide_index=True, width="stretch")
            st.download_button("Download held-out predictions (CSV)",
                pd.DataFrame({"trajectory_id": groups.iloc[test_idx].to_numpy(),
                              "scenario": data["scenario"].iloc[test_idx].to_numpy(),
                              "actual_snr_db": actual, "predicted_snr_db": pred}).to_csv(index=False).encode("utf-8"),
                "haps_rf_heldout_predictions.csv", "text/csv")
            st.info("Interpretation: the target SNR is calculated by the same simulation/link-budget pipeline as the features. The Random Forest is learning simulated relationships; this is not independent validation against real measurements.")
    except Exception as exc:
        st.error("Could not evaluate the Random Forest model.")
        st.exception(exc)

st.divider()
st.warning(
    "Model limitations: P.528 is an aeronautical propagation model; its applicability to HAPS must be justified. "
    "P.838 provides specific rain attenuation (dB/km), while this project multiplies it by an assumed effective rain-path length; "
    "it does not implement a complete rain-path availability model. Synthetic metrics do not establish real-world performance."
)
st.caption("Combined validation dashboard | Team RVTS")
