"""
HAPS Link-Quality Prediction Using ITU-R Propagation Models
Team RVTS

Models:
- ITU-R P.525-5: free-space basic transmission loss (implemented as the published
  free-space equation).
- ITU-R P.528-5: aeronautical air-to-ground basic transmission loss, through
  the Py528 implementation of the ITU/NTIA reference algorithm.
- ITU-R P.838-3: rain specific attenuation, through ITU-Rpy (itur).

Important modelling notes:
- P.528-5 was developed for aeronautical mobile/radionavigation services. Its
  use for a HAPS link is a modelling choice that should be justified with your
  mentor; it is not automatically a HAPS-specific validation.
- P.838-3 calculates specific rain attenuation (dB/km), not the complete
  end-to-end rain attenuation by itself. This script multiplies it by an
  explicitly assumed effective rain-path length. For a defensible rain-link
  budget, estimate that path length from the link geometry and a suitable
  rain-height model, e.g. the relevant ITU-R rain-height/path-length method.
- Synthetic data and ML metrics are simulation consistency checks, not
  real-world validation.
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# OPTIONAL DEPENDENCIES: install using the command provided after this code.
# ---------------------------------------------------------------------------
try:
    from Py528 import P528
except ImportError as exc:
    raise SystemExit(
        "\nMissing P.528 implementation.\n"
        "Install it in the same Python environment with:\n"
        "  py -m pip install git+https://github.com/eeveetza/Py528.git\n"
        "Then run this script again.\n"
        f"Original error: {exc}"
    )

try:
    import itur
    from itur.models import itu838
except ImportError as exc:
    raise SystemExit(
        "\nMissing ITU-Rpy package for P.838-3.\n"
        "Install it with:\n"
        "  py -m pip install itur\n"
        "Then run this script again.\n"
        f"Original error: {exc}"
    )

# ---------------------------------------------------------------------------
# SYSTEM PARAMETERS (illustrative starting assumptions; update to your design)
# ---------------------------------------------------------------------------
HAPS_ALTITUDE_M = 20_000.0       # HAPS height above ground/mean sea level (m)
GROUND_RX_HEIGHT_M = 1.5         # receiver antenna height (m)
FREQUENCY_MHZ = 2000.0           # 2 GHz
TX_POWER_DBM = 30.0
TX_ANTENNA_GAIN_DBI = 10.0
RX_ANTENNA_GAIN_DBI = 3.0
BANDWIDTH_HZ = 10e6
NOISE_FIGURE_DB = 5.0
POLARIZATION = 1                 # P.528: 0=horizontal, 1=vertical
TIME_PERCENTAGE = 50.0           # P.528 time percentage, valid 1..99
POLARIZATION_TILT_DEG = 90.0     # P.838 tilt: 0=horizontal, 90=vertical
RAIN_PATH_LENGTH_KM = 2.0        # Explicit assumed effective rain-path length
CLEAR_RAIN_RATE_MM_H = 0.0
RAIN_RATE_MIN_MM_H = 5.0
RAIN_RATE_MAX_MM_H = 50.0

# Dataset parameters
N_TRAJECTORIES = 60
POINTS_PER_TRAJECTORY = 40
MAX_HORIZONTAL_DISTANCE_KM = 50.0
RANDOM_SEED = 42

OUTPUT_DIR = Path(__file__).resolve().parent / "haps_project_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NOISE_POWER_DBM = -174.0 + 10.0 * np.log10(BANDWIDTH_HZ) + NOISE_FIGURE_DB


def p525_free_space_loss_db(distance_km: float, frequency_mhz: float) -> float:
    """ITU-R P.525 free-space basic transmission loss, dB."""
    if distance_km <= 0 or frequency_mhz <= 0:
        raise ValueError("Distance and frequency must be positive.")
    return 32.45 + 20.0 * np.log10(distance_km) + 20.0 * np.log10(frequency_mhz)


def p528_basic_loss_db(distance_km: float,
                       low_terminal_height_m: float = GROUND_RX_HEIGHT_M,
                       high_terminal_height_m: float = HAPS_ALTITUDE_M,
                       frequency_mhz: float = FREQUENCY_MHZ,
                       polarization: int = POLARIZATION,
                       time_percentage: float = TIME_PERCENTAGE) -> float:
    """
    ITU-R P.528-5 basic transmission loss in dB.

    Py528 function returns a result structure with A_db as basic transmission
    loss. Inputs are great-circle path distance (km), terminal heights (m),
    frequency (MHz), polarization (0 horizontal / 1 vertical), and time %.
    """
    if distance_km <= 0:
        # P.528 expects a positive terminal-to-terminal path distance.
        distance_km = 0.01
    if not (100.0 <= frequency_mhz <= 30000.0):
        raise ValueError("P.528-5 frequency must be between 100 and 30000 MHz.")
    if not (1.0 <= time_percentage <= 99.0):
        raise ValueError("P.528-5 time percentage must be between 1 and 99.")
    result = P528.bt_loss(
        float(distance_km),
        float(low_terminal_height_m),
        float(high_terminal_height_m),
        float(frequency_mhz),
        int(polarization),
        float(time_percentage),
    )
    loss = float(result.A_db)
    if not np.isfinite(loss):
        raise RuntimeError(f"P.528 returned a non-finite loss at {distance_km} km.")
    return loss


def p838_specific_rain_attenuation_db_per_km(
    rain_rate_mm_h: float,
    frequency_mhz: float = FREQUENCY_MHZ,
    elevation_deg: float = 90.0,
    polarization_tilt_deg: float = POLARIZATION_TILT_DEG,
) -> float:
    """
    ITU-R P.838-3 specific attenuation gamma_R (dB/km).

    ITU-Rpy evaluates the frequency/polarization-dependent k and alpha
    coefficients and applies gamma_R = k * R**alpha.
    """
    if rain_rate_mm_h <= 0:
        return 0.0
    gamma = itu838.rain_specific_attenuation(
        float(rain_rate_mm_h),
        float(frequency_mhz / 1000.0),  # GHz
        float(elevation_deg),
        float(polarization_tilt_deg),
    )
    # ITU-Rpy versions may return an Astropy Quantity or a numeric array.
    if hasattr(gamma, "value"):
        gamma = gamma.value
    return float(np.asarray(gamma).squeeze())


def calculate_link(distance_km: float, rain_rate_mm_h: float) -> dict:
    """Calculate P.525, P.528, P.838 rain attenuation and link budget."""
    slant_distance_km = float(np.hypot(distance_km, HAPS_ALTITUDE_M / 1000.0))

    p525_loss = p525_free_space_loss_db(slant_distance_km, FREQUENCY_MHZ)

    # P.528 uses the path distance between terminals and terminal heights.
    # Horizontal distance is used as the specified path-distance input here.
    p528_loss = p528_basic_loss_db(
        distance_km=max(distance_km, 0.01),
        low_terminal_height_m=GROUND_RX_HEIGHT_M,
        high_terminal_height_m=HAPS_ALTITUDE_M,
        frequency_mhz=FREQUENCY_MHZ,
        polarization=POLARIZATION,
        time_percentage=TIME_PERCENTAGE,
    )

    # P.838 gives specific attenuation. Total rain loss below uses an
    # explicit assumed effective path length, not a full P.618/P.839 path model.
    elevation_deg = np.degrees(
        np.arctan2(HAPS_ALTITUDE_M / 1000.0, max(distance_km, 0.01))
    )
    gamma_r = p838_specific_rain_attenuation_db_per_km(
        rain_rate_mm_h,
        FREQUENCY_MHZ,
        elevation_deg=elevation_deg,
        polarization_tilt_deg=POLARIZATION_TILT_DEG,
    )
    rain_loss_db = gamma_r * RAIN_PATH_LENGTH_KM

    # P.528 basic transmission loss is used as the primary loss model.
    total_loss_db = p528_loss + rain_loss_db
    received_power_dbm = (
        TX_POWER_DBM + TX_ANTENNA_GAIN_DBI + RX_ANTENNA_GAIN_DBI - total_loss_db
    )
    snr_db = received_power_dbm - NOISE_POWER_DBM

    return {
        "horizontal_distance_km": float(distance_km),
        "slant_distance_km": slant_distance_km,
        "frequency_mhz": FREQUENCY_MHZ,
        "rain_rate_mm_h": float(rain_rate_mm_h),
        "p525_fspl_db": p525_loss,
        "p528_basic_loss_db": p528_loss,
        "p838_specific_rain_attenuation_db_per_km": gamma_r,
        "assumed_rain_path_length_km": RAIN_PATH_LENGTH_KM,
        "rain_attenuation_db": rain_loss_db,
        "total_loss_using_p528_plus_rain_db": total_loss_db,
        "received_power_dbm": received_power_dbm,
        "noise_power_dbm": NOISE_POWER_DBM,
        "snr_db": snr_db,
    }


def create_dataset() -> pd.DataFrame:
    """Create two synthetic scenarios, grouped by trajectory for safe splitting."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for trajectory_id in range(N_TRAJECTORIES):
        # Different trajectories sample different ground distances.
        start = rng.uniform(0.1, 2.0)
        end = rng.uniform(20.0, MAX_HORIZONTAL_DISTANCE_KM)
        distances = np.linspace(start, end, POINTS_PER_TRAJECTORY)

        for scenario in ("clear_sky", "rain"):
            for point_idx, distance_km in enumerate(distances):
                if scenario == "clear_sky":
                    rain_rate = CLEAR_RAIN_RATE_MM_H
                else:
                    rain_rate = float(rng.uniform(RAIN_RATE_MIN_MM_H, RAIN_RATE_MAX_MM_H))

                metrics = calculate_link(float(distance_km), rain_rate)
                rows.append({
                    "trajectory_id": trajectory_id,
                    "point_index": point_idx,
                    "scenario": scenario,
                    **metrics,
                })

    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 72)
    print("HAPS LINK-QUALITY PREDICTION - TEAM RVTS")
    print("ITU-R P.525-5 + P.528-5 + P.838-3")
    print("=" * 72)

    sample = calculate_link(distance_km=10.0, rain_rate_mm_h=0.0)
    print("\nSINGLE-LINK CHECK (10 km horizontal distance, clear sky)")
    print(f"HAPS altitude                         : {HAPS_ALTITUDE_M / 1000:.2f} km")
    print(f"Horizontal distance                   : {sample['horizontal_distance_km']:.2f} km")
    print(f"Slant distance                        : {sample['slant_distance_km']:.2f} km")
    print(f"Frequency                             : {FREQUENCY_MHZ / 1000:.2f} GHz")
    print(f"ITU-R P.525 free-space loss           : {sample['p525_fspl_db']:.2f} dB")
    print(f"ITU-R P.528-5 basic transmission loss : {sample['p528_basic_loss_db']:.2f} dB")
    print(f"ITU-R P.838-3 specific rain loss      : {sample['p838_specific_rain_attenuation_db_per_km']:.4f} dB/km")
    print(f"Received signal power                 : {sample['received_power_dbm']:.2f} dBm")
    print(f"Noise power                           : {sample['noise_power_dbm']:.2f} dBm")
    print(f"SNR                                   : {sample['snr_db']:.2f} dB")

    rain_sample = calculate_link(distance_km=10.0, rain_rate_mm_h=25.0)
    print("\nRAIN SCENARIO EXAMPLE (25 mm/h)")
    print(f"P.838 specific attenuation            : {rain_sample['p838_specific_rain_attenuation_db_per_km']:.4f} dB/km")
    print(f"Assumed effective rain path            : {RAIN_PATH_LENGTH_KM:.2f} km")
    print(f"Estimated rain attenuation             : {rain_sample['rain_attenuation_db']:.4f} dB")
    print("Note: rain path length is an explicit assumption, not computed by P.838.")

    print("\nGENERATING SYNTHETIC DATASET...")
    df = create_dataset()
    dataset_path = OUTPUT_DIR / "haps_itu_synthetic_dataset.csv"
    df.to_csv(dataset_path, index=False)

    # Random Forest predicts SNR. Split by trajectory IDs, so points from one
    # trajectory cannot appear in both train and test sets.
    feature_columns = [
        "horizontal_distance_km",
        "slant_distance_km",
        "frequency_mhz",
        "rain_rate_mm_h",
        "p525_fspl_db",
        "p528_basic_loss_db",
        "p838_specific_rain_attenuation_db_per_km",
        "rain_attenuation_db",
    ]
    target_column = "snr_db"
    X = df[feature_columns]
    y = df[target_column]
    groups = df["trajectory_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    model = RandomForestRegressor(
        n_estimators=250,
        random_state=RANDOM_SEED,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    predictions = model.predict(X.iloc[test_idx])

    mae = mean_absolute_error(y.iloc[test_idx], predictions)
    rmse = np.sqrt(mean_squared_error(y.iloc[test_idx], predictions))
    r2 = r2_score(y.iloc[test_idx], predictions)

    pred_df = df.iloc[test_idx][
        ["trajectory_id", "point_index", "scenario", "horizontal_distance_km",
         "rain_rate_mm_h", "snr_db"]
    ].copy()
    pred_df["predicted_snr_db"] = predictions
    pred_df["absolute_error_db"] = np.abs(pred_df["snr_db"] - pred_df["predicted_snr_db"])
    pred_path = OUTPUT_DIR / "haps_itu_held_out_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(OUTPUT_DIR / "random_forest_feature_importance.csv", index=False)

    print("\nDATASET")
    print(f"Rows generated                         : {len(df)}")
    print(f"Trajectories                           : {df['trajectory_id'].nunique()}")
    print(f"Scenarios                              : {', '.join(sorted(df['scenario'].unique()))}")
    print(f"Dataset saved                          : {dataset_path}")

    train_groups = set(groups.iloc[train_idx].unique())
    test_groups = set(groups.iloc[test_idx].unique())
    print("\nMODEL EVALUATION (synthetic held-out trajectories)")
    print(f"Training rows                          : {len(train_idx)}")
    print(f"Testing rows                           : {len(test_idx)}")
    print(f"Training trajectories                  : {len(train_groups)}")
    print(f"Held-out test trajectories             : {len(test_groups)}")
    print(f"MAE                                    : {mae:.4f} dB")
    print(f"RMSE                                   : {rmse:.4f} dB")
    print(f"R-squared                              : {r2:.4f}")
    print(f"Predictions saved                      : {pred_path}")
    print(f"Feature importance saved               : {OUTPUT_DIR / 'random_forest_feature_importance.csv'}")

    # Plot 1: model loss curves
    distances = np.linspace(0.1, MAX_HORIZONTAL_DISTANCE_KM, 150)
    clear_rows = [calculate_link(float(d), 0.0) for d in distances]
    rain_rows = [calculate_link(float(d), 25.0) for d in distances]
    plt.figure(figsize=(10, 6))
    plt.plot(distances, [r["received_power_dbm"] for r in clear_rows], label="Clear sky")
    plt.plot(distances, [r["received_power_dbm"] for r in rain_rows], label="Rain: 25 mm/h")
    plt.xlabel("Horizontal ground distance (km)")
    plt.ylabel("Received power (dBm)")
    plt.title("HAPS Link Budget Using P.528-5 and P.838-3")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "haps_itu_received_power.png", dpi=160)
    plt.close()

    # Plot 2: SNR vs distance
    plt.figure(figsize=(10, 6))
    plt.plot(distances, [r["snr_db"] for r in clear_rows], label="Clear sky")
    plt.plot(distances, [r["snr_db"] for r in rain_rows], label="Rain: 25 mm/h")
    plt.xlabel("Horizontal ground distance (km)")
    plt.ylabel("SNR (dB)")
    plt.title("SNR vs Distance with ITU-R Propagation Models")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "haps_itu_snr_vs_distance.png", dpi=160)
    plt.close()

    # Plot 3: held-out prediction performance
    plt.figure(figsize=(7, 7))
    plt.scatter(y.iloc[test_idx], predictions, alpha=0.45)
    low = min(float(y.iloc[test_idx].min()), float(predictions.min()))
    high = max(float(y.iloc[test_idx].max()), float(predictions.max()))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Simulated SNR (dB)")
    plt.ylabel("Random Forest predicted SNR (dB)")
    plt.title("Held-out Trajectory Prediction")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "haps_itu_prediction_evaluation.png", dpi=160)
    plt.close()

    # Plot 4: feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance["feature"][::-1], importance["importance"][::-1])
    plt.xlabel("Random Forest feature importance")
    plt.title("Feature Importance for SNR Prediction")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "haps_itu_feature_importance.png", dpi=160)
    plt.close()

    print("\nPLOTS CREATED")
    for filename in (
        "haps_itu_received_power.png",
        "haps_itu_snr_vs_distance.png",
        "haps_itu_prediction_evaluation.png",
        "haps_itu_feature_importance.png",
    ):
        print(f"  {OUTPUT_DIR / filename}")

    print("\nIMPORTANT INTERPRETATION")
    print("- P.525-5 is used for the free-space reference loss.")
    print("- P.528-5 is used for basic transmission loss; check its suitability for your HAPS scenario.")
    print("- P.838-3 computes rain specific attenuation; total rain loss uses the assumed 2 km effective path.")
    print("- This script does not implement the full P.618/P.839 rain-path availability method.")
    print("- The dataset is synthetic; MAE/RMSE/R-squared are not real-world validation.")
    print("\nDone.")


if __name__ == "__main__":
    main()
