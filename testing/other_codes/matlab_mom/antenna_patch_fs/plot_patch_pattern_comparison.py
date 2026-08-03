"""Plot normalised MATLAB-MoM and gprMax-KSIR patch antenna patterns."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
PLOT_FLOOR_DB = -40.0
S11_FFT_ZERO_PADDING = 8


def _read_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")


def _mirror_asymmetry(angle, values, limit=90):
    """Return RMS and maximum dB difference between positive/negative angles."""

    positive_angles = np.arange(2.0, limit + 0.1, 2.0)
    positive = np.asarray([values[angle == item][0] for item in positive_angles])
    negative = np.asarray([values[angle == -item][0] for item in positive_angles])
    valid = (positive > PLOT_FLOOR_DB) & (negative > PLOT_FLOOR_DB)
    difference = positive[valid] - negative[valid]
    return {
        "rms_difference_db": float(np.sqrt(np.mean(difference**2))),
        "maximum_absolute_difference_db": float(np.max(np.abs(difference))),
    }


def _complex_s11(data):
    """Reconstruct complex S11 from a structured CSV array."""

    return np.asarray(data["s11_real"] + 1j * data["s11_imag"])


def _minimum_s11(frequency, magnitude_db):
    """Return a three-point parabolic estimate around the sampled minimum."""

    index = int(np.argmin(magnitude_db))
    result = {
        "frequency_hz": float(frequency[index]),
        "magnitude_db": float(magnitude_db[index]),
        "sampled_frequency_hz": float(frequency[index]),
        "sampled_magnitude_db": float(magnitude_db[index]),
    }
    if index == 0 or index == frequency.size - 1:
        return result

    frequency_scale = np.mean(np.diff(frequency))
    local_frequency = (frequency[index - 1 : index + 2] - frequency[index]) / frequency_scale
    coefficients = np.polyfit(local_frequency, magnitude_db[index - 1 : index + 2], 2)
    if coefficients[0] <= 0:
        return result

    vertex = -coefficients[1] / (2 * coefficients[0])
    if not -1 <= vertex <= 1:
        return result
    result["frequency_hz"] = float(frequency[index] + vertex * frequency_scale)
    result["magnitude_db"] = float(np.polyval(coefficients, vertex))
    return result


def _threshold_bandwidth(frequency, magnitude_db, threshold_db=-10.0):
    """Linearly interpolate the first two crossings of an S11 threshold."""

    crossings = []
    offset = magnitude_db - threshold_db
    for index in range(offset.size - 1):
        if offset[index] * offset[index + 1] <= 0 and offset[index] != offset[index + 1]:
            fraction = -offset[index] / (offset[index + 1] - offset[index])
            crossings.append(
                frequency[index] + fraction * (frequency[index + 1] - frequency[index])
            )
    if len(crossings) < 2:
        return None
    return {
        "threshold_db": threshold_db,
        "lower_frequency_hz": float(crossings[0]),
        "upper_frequency_hz": float(crossings[1]),
        "bandwidth_hz": float(crossings[1] - crossings[0]),
    }


def main():
    matlab = _read_csv(RESULTS_DIR / "patch_antenna_matlab_pattern.csv")
    gprmax = _read_csv(RESULTS_DIR / "patch_antenna_gprmax_pattern.csv")
    single_pattern = _read_csv(RESULTS_DIR / "patch_antenna_gprmax_single_feed_pattern.csv")
    series_pattern = _read_csv(RESULTS_DIR / "patch_antenna_gprmax_series_feed_pattern.csv")
    fine_z_pattern = _read_csv(RESULTS_DIR / "patch_antenna_gprmax_single_feed_fine_z_pattern.csv")
    fine_xyz_pattern = _read_csv(
        RESULTS_DIR / "patch_antenna_gprmax_single_feed_fine_xyz_pattern.csv"
    )
    if not np.array_equal(matlab["angle_deg"], gprmax["angle_deg"]):
        raise ValueError("MATLAB and gprMax angle grids do not match")
    if not np.array_equal(gprmax["angle_deg"], single_pattern["angle_deg"]):
        raise ValueError("The two gprMax feed pattern grids do not match")
    if not np.array_equal(gprmax["angle_deg"], series_pattern["angle_deg"]):
        raise ValueError("The series and distributed gprMax pattern grids do not match")
    if not np.array_equal(gprmax["angle_deg"], fine_z_pattern["angle_deg"]):
        raise ValueError("The fine-z and standard gprMax pattern grids do not match")
    if not np.array_equal(gprmax["angle_deg"], fine_xyz_pattern["angle_deg"]):
        raise ValueError("The fine-xyz and standard gprMax pattern grids do not match")

    matlab_peak = max(
        np.max(matlab["xz_directivity_dbi"]),
        np.max(matlab["yz_directivity_dbi"]),
    )
    matlab_xz = matlab["xz_directivity_dbi"] - matlab_peak
    matlab_yz = matlab["yz_directivity_dbi"] - matlab_peak
    gprmax_xz = gprmax["xz_co_normalized_db"]
    gprmax_yz = gprmax["yz_co_normalized_db"]
    single_xz = single_pattern["xz_co_normalized_db"]
    single_yz = single_pattern["yz_co_normalized_db"]
    series_xz = series_pattern["xz_co_normalized_db"]
    series_yz = series_pattern["yz_co_normalized_db"]
    fine_z_xz = fine_z_pattern["xz_co_normalized_db"]
    fine_z_yz = fine_z_pattern["yz_co_normalized_db"]
    fine_xyz_xz = fine_xyz_pattern["xz_co_normalized_db"]
    fine_xyz_yz = fine_xyz_pattern["yz_co_normalized_db"]

    angle = matlab["angle_deg"]
    angle_rad = np.deg2rad(angle)
    cuts = (
        ("E-plane (x-z)", matlab_xz, gprmax_xz),
        ("H-plane (y-z)", matlab_yz, gprmax_yz),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), subplot_kw={"projection": "polar"})
    for ax, (title, matlab_cut, gprmax_cut) in zip(axes, cuts):
        ax.plot(
            angle_rad,
            np.maximum(matlab_cut, PLOT_FLOOR_DB),
            color="#d95f02",
            linewidth=2,
            label="MATLAB Antenna Toolbox (MoM)",
        )
        ax.plot(
            angle_rad,
            np.maximum(gprmax_cut, PLOT_FLOOR_DB),
            color="#1b6ca8",
            linewidth=2,
            linestyle="--",
            label="gprMax FDTD + KSIR",
        )
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(PLOT_FLOOR_DB, 0)
        ax.set_rticks((-40, -30, -20, -10, 0))
        ax.set_rlabel_position(135)
        ax.grid(alpha=0.45)
        ax.set_title(title, pad=18)
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.08, -0.19), ncol=2, frameon=False)
    fig.suptitle("Rectangular patch at 2.37 GHz — normalised co-polar field pattern")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    polar_path = RESULTS_DIR / "patch_pattern_comparison.png"
    fig.savefig(polar_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)
    for ax, (title, matlab_cut, gprmax_cut) in zip(axes, cuts):
        ax.plot(angle, matlab_cut, color="#d95f02", linewidth=2, label="MATLAB MoM")
        ax.plot(
            angle,
            gprmax_cut,
            color="#1b6ca8",
            linewidth=2,
            linestyle="--",
            label="gprMax KSIR",
        )
        ax.set_ylim(PLOT_FLOOR_DB, 2)
        ax.set_ylabel("Normalised pattern (dB)")
        ax.set_title(title)
        ax.grid(alpha=0.35)
    axes[0].legend()
    axes[-1].set_xlabel("Signed angle from +z (degrees)")
    axes[-1].set_xlim(-180, 180)
    fig.tight_layout()
    cartesian_path = RESULTS_DIR / "patch_pattern_comparison_cartesian.png"
    fig.savefig(cartesian_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    upper = np.abs(angle) <= 90
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (title, matlab_cut, gprmax_cut) in zip(axes, cuts):
        ax.plot(
            angle[upper],
            matlab_cut[upper],
            color="#d95f02",
            linewidth=2,
            label="MATLAB MoM",
        )
        ax.plot(
            angle[upper],
            gprmax_cut[upper],
            color="#1b6ca8",
            linewidth=2,
            linestyle="--",
            label="gprMax KSIR",
        )
        ax.set_xlim(-90, 90)
        ax.set_ylim(-17, 1)
        ax.set_xlabel("Signed angle from +z (degrees)")
        ax.set_title(title)
        ax.grid(alpha=0.35)
    axes[0].set_ylabel("Normalised pattern (dB)")
    axes[0].legend()
    fig.suptitle("Radiating upper hemisphere")
    fig.tight_layout()
    upper_path = RESULTS_DIR / "patch_pattern_comparison_upper.png"
    fig.savefig(upper_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    mesh_cuts = (
        ("E-plane (x-z)", matlab_xz, single_xz, fine_z_xz, fine_xyz_xz),
        ("H-plane (y-z)", matlab_yz, single_yz, fine_z_yz, fine_xyz_yz),
    )
    for ax, (title, matlab_cut, standard_cut, fine_z_cut, fine_xyz_cut) in zip(axes, mesh_cuts):
        ax.plot(
            angle[upper],
            matlab_cut[upper],
            color="#d95f02",
            linewidth=2,
            label="MATLAB MoM",
        )
        ax.plot(
            angle[upper],
            standard_cut[upper],
            color="#2b8c4b",
            linewidth=1.8,
            linestyle="-.",
            label="gprMax standard",
        )
        ax.plot(
            angle[upper],
            fine_z_cut[upper],
            color="#222222",
            linewidth=1.8,
            linestyle=(0, (5, 2)),
            label="gprMax fine z",
        )
        ax.plot(
            angle[upper],
            fine_xyz_cut[upper],
            color="#0096a6",
            linewidth=2,
            linestyle=(0, (3, 1, 1, 1)),
            label="gprMax fine xyz",
        )
        ax.set_xlim(-90, 90)
        ax.set_ylim(-17, 1)
        ax.set_xlabel("Signed angle from +z (degrees)")
        ax.set_title(title)
        ax.grid(alpha=0.35)
    axes[0].set_ylabel("Normalised pattern (dB)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Patch-pattern mesh convergence at 2.37 GHz")
    fig.tight_layout()
    mesh_path = RESULTS_DIR / "patch_pattern_mesh_convergence.png"
    fig.savefig(mesh_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    metrics = {}
    for key, matlab_cut, gprmax_cut in (
        ("xz", matlab_xz, gprmax_xz),
        ("yz", matlab_yz, gprmax_yz),
    ):
        difference = np.maximum(gprmax_cut, PLOT_FLOOR_DB) - np.maximum(matlab_cut, PLOT_FLOOR_DB)
        upper = np.abs(angle) <= 90
        metrics[key] = {
            "clipped_full_cut_rms_difference_db": float(np.sqrt(np.mean(difference**2))),
            "clipped_upper_hemisphere_rms_difference_db": float(
                np.sqrt(np.mean(difference[upper] ** 2))
            ),
            "maximum_absolute_clipped_difference_db": float(np.max(np.abs(difference))),
            "matlab_upper_mirror_asymmetry": _mirror_asymmetry(angle, matlab_cut),
            "gprmax_upper_mirror_asymmetry": _mirror_asymmetry(angle, gprmax_cut),
        }
    metrics["normalisation"] = "one global co-polar peak per solver"
    metrics["plot_floor_db"] = PLOT_FLOOR_DB
    upper = np.abs(angle) <= 90
    metrics["gprmax_feed_pattern_difference"] = {}
    for feed_name, feed_xz, feed_yz in (
        ("single_vs_distributed", single_xz, single_yz),
        ("series_vs_distributed", series_xz, series_yz),
    ):
        metrics["gprmax_feed_pattern_difference"][feed_name] = {}
        for key, distributed_cut, candidate_cut in (
            ("xz", gprmax_xz, feed_xz),
            ("yz", gprmax_yz, feed_yz),
        ):
            valid = upper & (distributed_cut > PLOT_FLOOR_DB) & (candidate_cut > PLOT_FLOOR_DB)
            difference = candidate_cut[valid] - distributed_cut[valid]
            metrics["gprmax_feed_pattern_difference"][feed_name][key] = {
                "upper_hemisphere_rms_db": float(np.sqrt(np.mean(difference**2))),
                "upper_hemisphere_maximum_absolute_db": float(np.max(np.abs(difference))),
            }
    metrics["gprmax_mesh_pattern_difference"] = {}
    for comparison, reference_xz, reference_yz, candidate_xz, candidate_yz in (
        ("fine_z_vs_standard", single_xz, single_yz, fine_z_xz, fine_z_yz),
        ("fine_xyz_vs_standard", single_xz, single_yz, fine_xyz_xz, fine_xyz_yz),
        ("fine_xyz_vs_fine_z", fine_z_xz, fine_z_yz, fine_xyz_xz, fine_xyz_yz),
    ):
        metrics["gprmax_mesh_pattern_difference"][comparison] = {}
        for key, reference_cut, candidate_cut in (
            ("xz", reference_xz, candidate_xz),
            ("yz", reference_yz, candidate_yz),
        ):
            valid = upper & (reference_cut > PLOT_FLOOR_DB) & (candidate_cut > PLOT_FLOOR_DB)
            difference = candidate_cut[valid] - reference_cut[valid]
            metrics["gprmax_mesh_pattern_difference"][comparison][key] = {
                "upper_hemisphere_rms_db": float(np.sqrt(np.mean(difference**2))),
                "upper_hemisphere_maximum_absolute_db": float(np.max(np.abs(difference))),
            }
    (RESULTS_DIR / "patch_pattern_comparison_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    combined_path = RESULTS_DIR / "patch_pattern_comparison.csv"
    with combined_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            (
                "angle_deg",
                "matlab_xz_normalized_db",
                "gprmax_xz_normalized_db",
                "matlab_yz_normalized_db",
                "gprmax_yz_normalized_db",
            )
        )
        writer.writerows(zip(angle, matlab_xz, gprmax_xz, matlab_yz, gprmax_yz))

    matlab_s11_data = _read_csv(RESULTS_DIR / "patch_antenna_matlab_s11.csv")
    gprmax_s11_data = _read_csv(RESULTS_DIR / "patch_antenna_gprmax_s11.csv")
    single_s11_data = _read_csv(RESULTS_DIR / "patch_antenna_gprmax_single_feed_s11.csv")
    series_s11_data = _read_csv(RESULTS_DIR / "patch_antenna_gprmax_series_feed_s11.csv")
    fine_z_s11_data = _read_csv(RESULTS_DIR / "patch_antenna_gprmax_single_feed_fine_z_s11.csv")
    fine_xyz_s11_data = _read_csv(RESULTS_DIR / "patch_antenna_gprmax_single_feed_fine_xyz_s11.csv")
    trim_1_s11_data = _read_csv(
        RESULTS_DIR / "patch_antenna_gprmax_single_feed_patch_trim_1_s11.csv"
    )
    trim_2_s11_data = _read_csv(
        RESULTS_DIR / "patch_antenna_gprmax_single_feed_patch_trim_2_s11.csv"
    )
    board_trim_1_s11_data = _read_csv(
        RESULTS_DIR / "patch_antenna_gprmax_single_feed_board_trim_1_s11.csv"
    )
    board_trim_2_s11_data = _read_csv(
        RESULTS_DIR / "patch_antenna_gprmax_single_feed_board_trim_2_s11.csv"
    )
    frequency = np.atleast_1d(gprmax_s11_data["frequency_hz"])
    matlab_frequency = np.atleast_1d(matlab_s11_data["frequency_hz"])
    single_frequency = np.atleast_1d(single_s11_data["frequency_hz"])
    series_frequency = np.atleast_1d(series_s11_data["frequency_hz"])
    fine_z_frequency = np.atleast_1d(fine_z_s11_data["frequency_hz"])
    fine_xyz_frequency = np.atleast_1d(fine_xyz_s11_data["frequency_hz"])
    trim_1_frequency = np.atleast_1d(trim_1_s11_data["frequency_hz"])
    trim_2_frequency = np.atleast_1d(trim_2_s11_data["frequency_hz"])
    board_trim_1_frequency = np.atleast_1d(board_trim_1_s11_data["frequency_hz"])
    board_trim_2_frequency = np.atleast_1d(board_trim_2_s11_data["frequency_hz"])
    for name, candidate in (
        ("MATLAB", matlab_frequency),
        ("single-feed gprMax", single_frequency),
        ("series-feed gprMax", series_frequency),
        ("39 mm gprMax", trim_1_frequency),
        ("38 mm gprMax", trim_2_frequency),
        ("79 by 59 mm board gprMax", board_trim_1_frequency),
        ("78 by 58 mm board gprMax", board_trim_2_frequency),
    ):
        if frequency.shape != candidate.shape or not np.allclose(
            frequency, candidate, rtol=1e-12, atol=1e-3
        ):
            raise ValueError(f"{name} and distributed-feed gprMax S11 grids do not match")

    matlab_s11 = _complex_s11(matlab_s11_data)
    gprmax_s11 = _complex_s11(gprmax_s11_data)
    single_s11 = _complex_s11(single_s11_data)
    series_s11 = _complex_s11(series_s11_data)
    fine_z_s11 = _complex_s11(fine_z_s11_data)
    fine_xyz_s11 = _complex_s11(fine_xyz_s11_data)
    trim_1_s11 = _complex_s11(trim_1_s11_data)
    trim_2_s11 = _complex_s11(trim_2_s11_data)
    board_trim_1_s11 = _complex_s11(board_trim_1_s11_data)
    board_trim_2_s11 = _complex_s11(board_trim_2_s11_data)
    matlab_s11_db = 20 * np.log10(np.maximum(np.abs(matlab_s11), np.finfo(float).tiny))
    gprmax_s11_db = 20 * np.log10(np.maximum(np.abs(gprmax_s11), np.finfo(float).tiny))
    single_s11_db = 20 * np.log10(np.maximum(np.abs(single_s11), np.finfo(float).tiny))
    series_s11_db = 20 * np.log10(np.maximum(np.abs(series_s11), np.finfo(float).tiny))
    fine_z_s11_db = 20 * np.log10(np.maximum(np.abs(fine_z_s11), np.finfo(float).tiny))
    fine_xyz_s11_db = 20 * np.log10(np.maximum(np.abs(fine_xyz_s11), np.finfo(float).tiny))
    trim_1_s11_db = 20 * np.log10(np.maximum(np.abs(trim_1_s11), np.finfo(float).tiny))
    trim_2_s11_db = 20 * np.log10(np.maximum(np.abs(trim_2_s11), np.finfo(float).tiny))
    board_trim_1_s11_db = 20 * np.log10(np.maximum(np.abs(board_trim_1_s11), np.finfo(float).tiny))
    board_trim_2_s11_db = 20 * np.log10(np.maximum(np.abs(board_trim_2_s11), np.finfo(float).tiny))
    matlab_phase = np.rad2deg(np.unwrap(np.angle(matlab_s11)))
    gprmax_phase = np.rad2deg(np.unwrap(np.angle(gprmax_s11)))
    single_phase = np.rad2deg(np.unwrap(np.angle(single_s11)))
    series_phase = np.rad2deg(np.unwrap(np.angle(series_s11)))
    fine_z_phase = np.rad2deg(np.unwrap(np.angle(fine_z_s11)))
    fine_xyz_phase = np.rad2deg(np.unwrap(np.angle(fine_xyz_s11)))

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True)
    axes[0].plot(
        frequency / 1e9,
        matlab_s11_db,
        color="#d95f02",
        linewidth=2,
        label="MATLAB Antenna Toolbox (MoM)",
    )
    axes[0].plot(
        frequency / 1e9,
        gprmax_s11_db,
        color="#1b6ca8",
        linewidth=2,
        linestyle="--",
        label="gprMax FDTD, nine-edge feed",
    )
    axes[0].plot(
        frequency / 1e9,
        single_s11_db,
        color="#2b8c4b",
        linewidth=2,
        linestyle="-.",
        label="gprMax FDTD, single-edge feed",
    )
    axes[0].plot(
        frequency / 1e9,
        series_s11_db,
        color="#7651a6",
        linewidth=2,
        linestyle=":",
        label="gprMax FDTD, three-cell series feed",
    )
    axes[0].plot(
        fine_z_frequency / 1e9,
        fine_z_s11_db,
        color="#222222",
        linewidth=1.8,
        linestyle=(0, (5, 2)),
        label="gprMax FDTD, single feed, six z cells",
    )
    axes[0].plot(
        fine_xyz_frequency / 1e9,
        fine_xyz_s11_db,
        color="#0096a6",
        linewidth=2,
        linestyle=(0, (3, 1, 1, 1)),
        label="gprMax FDTD, fully refined mesh",
    )
    axes[0].axhline(-10, color="0.45", linewidth=1, linestyle=":")
    axes[0].set_ylabel(r"$|S_{11}|$ (dB)")
    axes[0].set_title(r"Rectangular patch — 50 $\Omega$ port comparison")
    axes[0].grid(alpha=0.35)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].plot(frequency / 1e9, matlab_phase, color="#d95f02", linewidth=2)
    axes[1].plot(
        frequency / 1e9,
        gprmax_phase,
        color="#1b6ca8",
        linewidth=2,
        linestyle="--",
    )
    axes[1].plot(
        frequency / 1e9,
        single_phase,
        color="#2b8c4b",
        linewidth=2,
        linestyle="-.",
    )
    axes[1].plot(
        frequency / 1e9,
        series_phase,
        color="#7651a6",
        linewidth=2,
        linestyle=":",
    )
    axes[1].plot(
        fine_z_frequency / 1e9,
        fine_z_phase,
        color="#222222",
        linewidth=1.8,
        linestyle=(0, (5, 2)),
    )
    axes[1].plot(
        fine_xyz_frequency / 1e9,
        fine_xyz_phase,
        color="#0096a6",
        linewidth=2,
        linestyle=(0, (3, 1, 1, 1)),
    )
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].set_ylabel(r"Unwrapped $S_{11}$ phase (degrees)")
    axes[1].grid(alpha=0.35)
    fig.tight_layout()
    s11_plot_path = RESULTS_DIR / "patch_s11_comparison.png"
    fig.savefig(s11_plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    length_cases = (
        ("MATLAB MoM, 40 mm", matlab_frequency, matlab_s11_db, "#d95f02", "-"),
        ("gprMax, 40 mm", single_frequency, single_s11_db, "#1b6ca8", "--"),
        ("gprMax, 39 mm", trim_1_frequency, trim_1_s11_db, "#2b8c4b", "-."),
        ("gprMax, 38 mm", trim_2_frequency, trim_2_s11_db, "#7651a6", ":"),
    )
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for label, case_frequency, case_s11_db, color, linestyle in length_cases:
        minimum = _minimum_s11(case_frequency, case_s11_db)
        ax.plot(
            case_frequency / 1e9,
            case_s11_db,
            color=color,
            linewidth=2.2,
            linestyle=linestyle,
            label=f"{label} ({minimum['frequency_hz'] / 1e9:.4f} GHz)",
        )
        ax.plot(
            minimum["frequency_hz"] / 1e9,
            minimum["magnitude_db"],
            marker="o",
            markersize=5,
            color=color,
        )
    ax.axhline(-10, color="0.45", linewidth=1, linestyle=(0, (2, 2)))
    ax.set_xlim(2.28, 2.54)
    ax.set_ylim(-20, 0)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(r"$|S_{11}|$ (dB)")
    ax.set_title("Patch-length sensitivity on the standard gprMax mesh")
    ax.grid(alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    length_s11_plot_path = RESULTS_DIR / "patch_s11_length_trim_comparison.png"
    fig.savefig(length_s11_plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    board_cases = (
        ("MATLAB MoM, 80 x 60 mm", matlab_frequency, matlab_s11_db, "#d95f02", "-"),
        ("gprMax, 80 x 60 mm", single_frequency, single_s11_db, "#1b6ca8", "--"),
        (
            "gprMax, 79 x 59 mm",
            board_trim_1_frequency,
            board_trim_1_s11_db,
            "#2b8c4b",
            "-.",
        ),
        (
            "gprMax, 78 x 58 mm",
            board_trim_2_frequency,
            board_trim_2_s11_db,
            "#7651a6",
            ":",
        ),
    )
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for label, case_frequency, case_s11_db, color, linestyle in board_cases:
        minimum = _minimum_s11(case_frequency, case_s11_db)
        ax.plot(
            case_frequency / 1e9,
            case_s11_db,
            color=color,
            linewidth=2.2,
            linestyle=linestyle,
            label=f"{label} ({minimum['frequency_hz'] / 1e9:.4f} GHz)",
        )
    ax.axhline(-10, color="0.45", linewidth=1, linestyle=(0, (2, 2)))
    ax.set_xlim(2.28, 2.50)
    ax.set_ylim(-20, 0)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(r"$|S_{11}|$ (dB)")
    ax.set_title("Ground-plane and dielectric-footprint sensitivity")
    ax.grid(alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    board_s11_plot_path = RESULTS_DIR / "patch_s11_board_trim_comparison.png"
    fig.savefig(board_s11_plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    s11_difference = gprmax_s11_db - matlab_s11_db
    single_s11_difference = single_s11_db - matlab_s11_db
    series_s11_difference = series_s11_db - matlab_s11_db
    fine_z_s11_db_common = np.interp(frequency, fine_z_frequency, fine_z_s11_db)
    fine_z_s11_difference = fine_z_s11_db_common - matlab_s11_db
    fine_xyz_s11_db_common = np.interp(frequency, fine_xyz_frequency, fine_xyz_s11_db)
    fine_xyz_s11_difference = fine_xyz_s11_db_common - matlab_s11_db
    gprmax_minimum = _minimum_s11(frequency, gprmax_s11_db)
    single_minimum = _minimum_s11(frequency, single_s11_db)
    series_minimum = _minimum_s11(frequency, series_s11_db)
    fine_z_minimum = _minimum_s11(fine_z_frequency, fine_z_s11_db)
    fine_xyz_minimum = _minimum_s11(fine_xyz_frequency, fine_xyz_s11_db)
    trim_1_minimum = _minimum_s11(trim_1_frequency, trim_1_s11_db)
    trim_2_minimum = _minimum_s11(trim_2_frequency, trim_2_s11_db)
    board_trim_1_minimum = _minimum_s11(board_trim_1_frequency, board_trim_1_s11_db)
    board_trim_2_minimum = _minimum_s11(board_trim_2_frequency, board_trim_2_s11_db)
    matlab_minimum = _minimum_s11(frequency, matlab_s11_db)
    metrics["s11"] = {
        "reference_impedance_ohm": 50.0,
        "frequency_spacing_hz": float(np.mean(np.diff(frequency))),
        "fft_zero_padding_factor": S11_FFT_ZERO_PADDING,
        "independent_frequency_resolution_hz": float(
            S11_FFT_ZERO_PADDING * np.mean(np.diff(frequency))
        ),
        "gprmax_distributed_minimum": gprmax_minimum,
        "gprmax_single_minimum": single_minimum,
        "gprmax_series_minimum": series_minimum,
        "gprmax_single_fine_z_minimum": fine_z_minimum,
        "gprmax_single_fine_xyz_minimum": fine_xyz_minimum,
        "gprmax_single_39_mm_minimum": trim_1_minimum,
        "gprmax_single_38_mm_minimum": trim_2_minimum,
        "gprmax_single_79_by_59_mm_board_minimum": board_trim_1_minimum,
        "gprmax_single_78_by_58_mm_board_minimum": board_trim_2_minimum,
        "matlab_minimum": matlab_minimum,
        "distributed_minimum_frequency_offset_hz": float(
            matlab_minimum["frequency_hz"] - gprmax_minimum["frequency_hz"]
        ),
        "single_minimum_frequency_offset_hz": float(
            matlab_minimum["frequency_hz"] - single_minimum["frequency_hz"]
        ),
        "series_minimum_frequency_offset_hz": float(
            matlab_minimum["frequency_hz"] - series_minimum["frequency_hz"]
        ),
        "single_fine_z_minimum_frequency_offset_hz": float(
            matlab_minimum["frequency_hz"] - fine_z_minimum["frequency_hz"]
        ),
        "single_fine_xyz_minimum_frequency_offset_hz": float(
            matlab_minimum["frequency_hz"] - fine_xyz_minimum["frequency_hz"]
        ),
        "gprmax_distributed_minus_10_db_bandwidth": _threshold_bandwidth(frequency, gprmax_s11_db),
        "gprmax_single_minus_10_db_bandwidth": _threshold_bandwidth(frequency, single_s11_db),
        "gprmax_series_minus_10_db_bandwidth": _threshold_bandwidth(frequency, series_s11_db),
        "gprmax_single_fine_z_minus_10_db_bandwidth": _threshold_bandwidth(
            fine_z_frequency, fine_z_s11_db
        ),
        "gprmax_single_fine_xyz_minus_10_db_bandwidth": _threshold_bandwidth(
            fine_xyz_frequency, fine_xyz_s11_db
        ),
        "matlab_minus_10_db_bandwidth": _threshold_bandwidth(frequency, matlab_s11_db),
        "distributed_magnitude_rms_difference_db": float(np.sqrt(np.mean(s11_difference**2))),
        "single_magnitude_rms_difference_db": float(np.sqrt(np.mean(single_s11_difference**2))),
        "series_magnitude_rms_difference_db": float(np.sqrt(np.mean(series_s11_difference**2))),
        "single_fine_z_magnitude_rms_difference_db": float(
            np.sqrt(np.mean(fine_z_s11_difference**2))
        ),
        "single_fine_xyz_magnitude_rms_difference_db": float(
            np.sqrt(np.mean(fine_xyz_s11_difference**2))
        ),
        "distributed_magnitude_maximum_absolute_difference_db": float(
            np.max(np.abs(s11_difference))
        ),
        "single_magnitude_maximum_absolute_difference_db": float(
            np.max(np.abs(single_s11_difference))
        ),
        "series_magnitude_maximum_absolute_difference_db": float(
            np.max(np.abs(series_s11_difference))
        ),
        "single_fine_z_magnitude_maximum_absolute_difference_db": float(
            np.max(np.abs(fine_z_s11_difference))
        ),
        "single_fine_xyz_magnitude_maximum_absolute_difference_db": float(
            np.max(np.abs(fine_xyz_s11_difference))
        ),
        "gprmax_distributed_incident_spectrum_minimum_relative_db": float(
            np.min(gprmax_s11_data["incident_relative_db"])
        ),
        "gprmax_distributed_feed_edge_rms_nonuniformity_maximum": float(
            np.max(gprmax_s11_data["edge_voltage_rms_nonuniformity"])
        ),
        "gprmax_single_feed_edge_rms_nonuniformity_maximum": float(
            np.max(single_s11_data["edge_voltage_rms_nonuniformity"])
        ),
        "gprmax_series_feed_edge_rms_nonuniformity_maximum": float(
            np.max(series_s11_data["edge_voltage_rms_nonuniformity"])
        ),
        "gprmax_single_fine_z_edge_rms_nonuniformity_maximum": float(
            np.max(fine_z_s11_data["edge_voltage_rms_nonuniformity"])
        ),
        "gprmax_single_fine_xyz_edge_rms_nonuniformity_maximum": float(
            np.max(fine_xyz_s11_data["edge_voltage_rms_nonuniformity"])
        ),
    }
    (RESULTS_DIR / "patch_pattern_comparison_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Saved comparison plot to {polar_path}")
    print(f"Saved Cartesian comparison to {cartesian_path}")
    print(f"Saved upper-hemisphere comparison to {upper_path}")
    print(f"Saved mesh-convergence comparison to {mesh_path}")
    print(f"Saved comparison data to {combined_path}")
    print(f"Saved S11 comparison to {s11_plot_path}")
    print(f"Saved patch-length S11 comparison to {length_s11_plot_path}")
    print(f"Saved board-footprint S11 comparison to {board_s11_plot_path}")


if __name__ == "__main__":
    main()
