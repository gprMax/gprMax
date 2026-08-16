"""Compare CST FEM, CST FIT, and gprMax directivity cuts at 2.45 GHz."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
CST_FEM_PATH = ROOT / "patch_ff_fem_cst.txt"
CST_FIT_PATH = ROOT / "patch_ff_fit_cst.txt"
GPRMAX_PATH = ROOT / "patch_recentered_closed_ntff.h5"
PLOT_PATH = ROOT / "patch_farfield_comparison_2p45ghz.png"
POLAR_PLOT_PATH = ROOT / "patch_farfield_polar_comparison_2p45ghz.png"
CSV_PATH = ROOT / "patch_farfield_comparison_2p45ghz.csv"
SUMMARY_PATH = ROOT / "patch_farfield_comparison_2p45ghz.json"

FREQUENCY_HZ = 2.45e9
PHI_CUTS_DEG = (0.0, 90.0)
ALL_PHI_CUTS_DEG = (0.0, 90.0, 180.0, 270.0)
POLAR_PLANES = (
    (0.0, 180.0, r"$\phi=0^\circ/180^\circ$ (x-z plane)", ("+z", "+x", "−z", "−x")),
    (90.0, 270.0, r"$\phi=90^\circ/270^\circ$ (y-z plane)", ("+z", "+y", "−z", "−y")),
)
COMPARISON_FLOOR_DBI = -40.0


def load_cst(path: Path) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Return theta and total directivity for each requested CST phi cut."""

    values = np.loadtxt(path, skiprows=2)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"Unexpected CST export shape {values.shape} in {path}")

    cuts = {}
    for phi in ALL_PHI_CUTS_DEG:
        selected = np.isclose(values[:, 1], phi, rtol=0, atol=1e-6)
        theta = values[selected, 0]
        directivity = values[selected, 2]
        order = np.argsort(theta)
        cuts[phi] = (theta[order], directivity[order])
    return cuts


def load_gprmax() -> tuple[float, dict[float, tuple[np.ndarray, np.ndarray]]]:
    """Return the nearest stored frequency and requested gprMax phi cuts."""

    band_path = "ntff/patch_surface/frequency/patch_farfield_band"
    with h5py.File(GPRMAX_PATH, "r") as output:
        band = output[band_path]
        frequencies = np.asarray(band["frequencies"], dtype=np.float64)
        frequency_index = int(np.argmin(np.abs(frequencies - FREQUENCY_HZ)))
        frequency = float(frequencies[frequency_index])
        # Frequencies are stored as float32, so 2.45 GHz is represented 128 Hz low.
        if not np.isclose(frequency, FREQUENCY_HZ, rtol=0, atol=1e3):
            raise ValueError(
                f"Nearest gprMax frequency is {frequency:g} Hz, not {FREQUENCY_HZ:g} Hz"
            )

        far_field = band["far_field/full_sphere"]
        theta_all = np.asarray(far_field["theta"], dtype=np.float64)
        phi_all = np.asarray(far_field["phi"], dtype=np.float64)
        directivity_all = np.asarray(
            far_field["fields/directivity_dbi"][frequency_index], dtype=np.float64
        )
        cuts = {}
        for phi in ALL_PHI_CUTS_DEG:
            selected = np.isclose(phi_all, phi, rtol=0, atol=1e-6)
            theta = theta_all[selected]
            directivity = directivity_all[selected]
            order = np.argsort(theta)
            cuts[phi] = (theta[order], directivity[order])
    return frequency, cuts


def cut_metrics(
    theta: np.ndarray, reference: np.ndarray, gprmax: np.ndarray
) -> dict[str, float]:
    """Calculate peak, endpoint, and floor-clipped error metrics."""

    clipped_difference = np.maximum(gprmax, COMPARISON_FLOOR_DBI) - np.maximum(
        reference, COMPARISON_FLOOR_DBI
    )
    front = theta <= 90
    reference_peak = int(np.argmax(reference))
    gprmax_peak = int(np.argmax(gprmax))
    return {
        "reference_peak_dbi": float(reference[reference_peak]),
        "reference_peak_theta_deg": float(theta[reference_peak]),
        "gprmax_peak_dbi": float(gprmax[gprmax_peak]),
        "gprmax_peak_theta_deg": float(theta[gprmax_peak]),
        "peak_difference_db": float(gprmax[gprmax_peak] - reference[reference_peak]),
        "boresight_difference_db": float(gprmax[0] - reference[0]),
        "back_direction_difference_db": float(gprmax[-1] - reference[-1]),
        "mae_floor_clipped_db": float(np.mean(np.abs(clipped_difference))),
        "rmse_floor_clipped_db": float(np.sqrt(np.mean(clipped_difference**2))),
        "front_hemisphere_mae_floor_clipped_db": float(
            np.mean(np.abs(clipped_difference[front]))
        ),
        "front_hemisphere_rmse_floor_clipped_db": float(
            np.sqrt(np.mean(clipped_difference[front] ** 2))
        ),
    }


def write_csv(
    theta: np.ndarray,
    cst_fem_cuts: dict[float, tuple[np.ndarray, np.ndarray]],
    cst_fit_cuts: dict[float, tuple[np.ndarray, np.ndarray]],
    gprmax_cuts: dict[float, tuple[np.ndarray, np.ndarray]],
) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = ["theta_deg"]
        for phi in ALL_PHI_CUTS_DEG:
            label = f"phi{phi:g}"
            header.extend(
                (
                    f"cst_fem_{label}_directivity_dbi",
                    f"cst_fit_{label}_directivity_dbi",
                    f"gprmax_{label}_directivity_dbi",
                    f"gprmax_minus_cst_fem_{label}_db",
                    f"gprmax_minus_cst_fit_{label}_db",
                )
            )
        writer.writerow(header)
        for index, angle in enumerate(theta):
            row = [angle]
            for phi in ALL_PHI_CUTS_DEG:
                cst_fem = cst_fem_cuts[phi][1][index]
                cst_fit = cst_fit_cuts[phi][1][index]
                gprmax = gprmax_cuts[phi][1][index]
                row.extend(
                    (
                        cst_fem,
                        cst_fit,
                        gprmax,
                        gprmax - cst_fem,
                        gprmax - cst_fit,
                    )
                )
            writer.writerow(row)


def main_beam_metrics(
    angles_deg: np.ndarray, directivity: np.ndarray, angle_key: str
) -> dict[str, float]:
    """Return the main-beam direction and directivity for JSON output."""

    peak = int(np.argmax(directivity))
    angle_deg = float(angles_deg[peak])
    if np.isclose(angle_deg, 0.0, rtol=0, atol=1e-9):
        angle_deg = 0.0
    return {
        angle_key: round(angle_deg, 6),
        "directivity_dbi": round(float(directivity[peak]), 6),
    }


def plot_cuts(
    theta: np.ndarray,
    cst_fem_cuts: dict[float, tuple[np.ndarray, np.ndarray]],
    cst_fit_cuts: dict[float, tuple[np.ndarray, np.ndarray]],
    gprmax_cuts: dict[float, tuple[np.ndarray, np.ndarray]],
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11, 7),
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": (3, 1)},
        constrained_layout=True,
    )
    for column, phi in enumerate(PHI_CUTS_DEG):
        cst_fem = cst_fem_cuts[phi][1]
        cst_fit = cst_fit_cuts[phi][1]
        gprmax = gprmax_cuts[phi][1]
        clipped_cst_fem = np.maximum(cst_fem, COMPARISON_FLOOR_DBI)
        clipped_cst_fit = np.maximum(cst_fit, COMPARISON_FLOOR_DBI)
        clipped_gprmax = np.maximum(gprmax, COMPARISON_FLOOR_DBI)

        pattern_axis = axes[0, column]
        difference_axis = axes[1, column]
        pattern_axis.plot(
            theta,
            clipped_cst_fem,
            color="#cc79a7",
            linewidth=2.4,
            linestyle=(0, (2, 2)),
            zorder=6,
            label="CST FEM (frequency-domain, adaptive mesh refinement)",
        )
        pattern_axis.plot(
            theta,
            clipped_gprmax,
            color="#0072b2",
            linewidth=2.4,
            linestyle="-",
            zorder=3,
            label="gprMax closed NTFF",
        )
        pattern_axis.plot(
            theta,
            clipped_cst_fit,
            color="#e69f00",
            linewidth=2.4,
            linestyle=(0, (5, 2)),
            zorder=5,
            label="CST FIT (time-domain)",
        )
        pattern_axis.set_title(rf"$\phi={phi:g}^\circ$")
        pattern_axis.set_ylim(COMPARISON_FLOOR_DBI, 8)
        pattern_axis.set_ylabel("Directivity (dBi)")
        pattern_axis.grid(True, alpha=0.3)
        pattern_axis.legend(loc="lower left")

        difference_axis.axhline(0, color="0.35", linewidth=1)
        difference_axis.plot(
            theta,
            clipped_gprmax - clipped_cst_fem,
            color="#cc79a7",
            linewidth=2.4,
            linestyle=(0, (2, 2)),
            zorder=6,
            label="gprMax − CST FEM (adaptive mesh refinement)",
        )
        difference_axis.plot(
            theta,
            clipped_gprmax - clipped_cst_fit,
            color="#e69f00",
            linewidth=2.4,
            linestyle=(0, (5, 2)),
            zorder=5,
            label="gprMax − CST FIT",
        )
        difference_axis.set_xlim(0, 180)
        difference_axis.set_xticks(np.arange(0, 181, 30))
        difference_axis.set_xlabel(r"$\theta$ (degrees)")
        difference_axis.set_ylabel("Difference\n(dB)")
        difference_axis.grid(True, alpha=0.3)
        difference_axis.legend(loc="lower left", fontsize=8)

    figure.suptitle(
        "Patch antenna directivity cuts at 2.45 GHz\n"
        f"Patterns and differences clipped at {COMPARISON_FLOOR_DBI:g} dBi"
    )
    figure.savefig(PLOT_PATH, dpi=200)
    plt.close(figure)


def _full_plane(
    positive_phi: float,
    negative_phi: float,
    cuts: dict[float, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Join opposing spherical-azimuth cuts into signed plane angles."""

    positive_theta, positive_values = cuts[positive_phi]
    negative_theta, negative_values = cuts[negative_phi]
    np.testing.assert_allclose(positive_theta, negative_theta, rtol=0, atol=1e-6)
    angles_deg = np.concatenate((-negative_theta[::-1], positive_theta[1:]))
    values = np.concatenate((negative_values[::-1], positive_values[1:]))
    return np.deg2rad(angles_deg), values


def plot_polar_planes(
    cst_fem_cuts: dict[float, tuple[np.ndarray, np.ndarray]],
    cst_fit_cuts: dict[float, tuple[np.ndarray, np.ndarray]],
    gprmax_cuts: dict[float, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Plot complete x-z and y-z directivity planes on polar axes."""

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6.2),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    radial_ticks_dbi = np.asarray([-40, -30, -20, -10, 0, 5], dtype=float)
    for axis, (positive_phi, negative_phi, title, direction_labels) in zip(
        axes, POLAR_PLANES
    ):
        cst_fem_angle, cst_fem = _full_plane(
            positive_phi, negative_phi, cst_fem_cuts
        )
        cst_fit_angle, cst_fit = _full_plane(
            positive_phi, negative_phi, cst_fit_cuts
        )
        gprmax_angle, gprmax = _full_plane(positive_phi, negative_phi, gprmax_cuts)
        np.testing.assert_allclose(cst_fem_angle, cst_fit_angle, rtol=0, atol=1e-12)
        np.testing.assert_allclose(cst_fem_angle, gprmax_angle, rtol=0, atol=1e-12)

        axis.plot(
            cst_fem_angle,
            np.maximum(cst_fem, COMPARISON_FLOOR_DBI) - COMPARISON_FLOOR_DBI,
            color="#cc79a7",
            linewidth=2.4,
            linestyle=(0, (2, 2)),
            zorder=6,
            label="CST FEM (frequency-domain, adaptive mesh refinement)",
        )
        axis.plot(
            gprmax_angle,
            np.maximum(gprmax, COMPARISON_FLOOR_DBI) - COMPARISON_FLOOR_DBI,
            color="#0072b2",
            linewidth=2.4,
            linestyle="-",
            zorder=3,
            label="gprMax closed NTFF",
        )
        axis.plot(
            cst_fit_angle,
            np.maximum(cst_fit, COMPARISON_FLOOR_DBI) - COMPARISON_FLOOR_DBI,
            color="#e69f00",
            linewidth=2.4,
            linestyle=(0, (5, 2)),
            zorder=5,
            label="CST FIT (time-domain)",
        )
        axis.set_theta_zero_location("N")
        axis.set_theta_direction(-1)
        axis.set_thetagrids((0, 90, 180, 270), labels=direction_labels)
        axis.set_ylim(0, 48)
        axis.set_yticks(radial_ticks_dbi - COMPARISON_FLOOR_DBI)
        axis.set_yticklabels([f"{value:g} dBi" for value in radial_ticks_dbi])
        axis.set_rlabel_position(35)
        axis.set_title(title, pad=20)
        axis.grid(True, alpha=0.35)
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.19))

    figure.suptitle(
        "Patch antenna full-plane directivity at 2.45 GHz\n"
        f"Radial floor {COMPARISON_FLOOR_DBI:g} dBi"
    )
    figure.savefig(POLAR_PLOT_PATH, dpi=200)
    plt.close(figure)


def main() -> int:
    cst_fem_cuts = load_cst(CST_FEM_PATH)
    cst_fit_cuts = load_cst(CST_FIT_PATH)
    stored_frequency, gprmax_cuts = load_gprmax()
    reference_theta = cst_fem_cuts[PHI_CUTS_DEG[0]][0]

    summary = {
        "requested_frequency_hz": FREQUENCY_HZ,
        "gprmax_stored_frequency_hz": stored_frequency,
        "comparison_floor_dbi": COMPARISON_FLOOR_DBI,
        "main_beams": {"phi_cuts": {}, "full_planes": {}},
        "cuts": {},
    }
    for phi in PHI_CUTS_DEG:
        cst_fem_theta, cst_fem = cst_fem_cuts[phi]
        cst_fit_theta, cst_fit = cst_fit_cuts[phi]
        gprmax_theta, gprmax = gprmax_cuts[phi]
        np.testing.assert_allclose(cst_fem_theta, reference_theta, rtol=0, atol=1e-6)
        np.testing.assert_allclose(cst_fit_theta, reference_theta, rtol=0, atol=1e-6)
        np.testing.assert_allclose(gprmax_theta, reference_theta, rtol=0, atol=1e-6)
        summary["cuts"][f"phi_{phi:g}_deg"] = {
            "gprmax_vs_cst_fem": cut_metrics(reference_theta, cst_fem, gprmax),
            "gprmax_vs_cst_fit": cut_metrics(reference_theta, cst_fit, gprmax),
        }
        summary["main_beams"]["phi_cuts"][f"phi_{phi:g}_deg"] = {
            "cst_fem_frequency_domain_adaptive_mesh_refinement": main_beam_metrics(
                reference_theta, cst_fem, "theta_deg"
            ),
            "gprmax_closed_ntff": main_beam_metrics(
                reference_theta, gprmax, "theta_deg"
            ),
            "cst_fit_time_domain": main_beam_metrics(
                reference_theta, cst_fit, "theta_deg"
            ),
        }

    solver_cuts = {
        "cst_fem_frequency_domain_adaptive_mesh_refinement": cst_fem_cuts,
        "gprmax_closed_ntff": gprmax_cuts,
        "cst_fit_time_domain": cst_fit_cuts,
    }
    for positive_phi, negative_phi, _title, _direction_labels in POLAR_PLANES:
        plane_key = f"phi_{positive_phi:g}_{negative_phi:g}_deg"
        summary["main_beams"]["full_planes"][plane_key] = {}
        for solver_name, cuts in solver_cuts.items():
            plane_angles, directivity = _full_plane(
                positive_phi, negative_phi, cuts
            )
            summary["main_beams"]["full_planes"][plane_key][solver_name] = (
                main_beam_metrics(
                    np.rad2deg(plane_angles),
                    directivity,
                    "signed_plane_angle_deg",
                )
            )

    write_csv(reference_theta, cst_fem_cuts, cst_fit_cuts, gprmax_cuts)
    plot_cuts(reference_theta, cst_fem_cuts, cst_fit_cuts, gprmax_cuts)
    plot_polar_planes(cst_fem_cuts, cst_fit_cuts, gprmax_cuts)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {PLOT_PATH}")
    print(f"Wrote {POLAR_PLOT_PATH}")
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
