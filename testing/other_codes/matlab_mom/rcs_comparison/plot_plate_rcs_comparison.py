"""Plot the PEC-plate RCS comparisons and write quantitative metrics."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.special import j1

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
WAVELENGTH = 3.25e-2
SQUARE_LENGTH = 10.16e-2
SQUARE_WIDTH = 10.16e-2
CIRCLE_RADIUS = 10.16e-2
PLOT_FLOOR_DB = -60.0


def _read_csv(path):
    """Read a named-column CSV as a one-dimensional array."""

    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))


def _read_matlab_reference(target, mesh):
    """Read a retained MATLAB curve when its convenience CSV is absent."""

    path = RESULTS_DIR / "plate_rcs_matlab.mat"
    if not path.exists():
        return None
    contents = loadmat(path, simplify_cells=True)
    references = contents.get("matlab_reference", {})
    columns = np.atleast_1d(contents.get("reference_columns", ())).tolist()
    key = f"{target}_{mesh}"
    if key not in references or not columns:
        return None
    data = np.atleast_2d(np.asarray(references[key], dtype=np.float64))
    if data.shape[1] != len(columns):
        raise ValueError(f"Invalid retained MATLAB reference matrix for {key}")
    names = ",".join(str(value) for value in columns)
    return np.rec.fromarrays(data.T, names=names)


def _square_rcs(elevation):
    """Return analytical physical-optics square-plate RCS in square metres."""

    elevation = np.deg2rad(np.asarray(elevation))
    argument = 2 * np.pi * SQUARE_LENGTH * np.cos(elevation) / WAVELENGTH
    aperture = SQUARE_LENGTH * SQUARE_WIDTH
    return (
        4
        * np.pi
        * aperture**2
        / WAVELENGTH**2
        * np.sin(elevation) ** 2
        * np.sinc(argument / np.pi) ** 2
    )


def _circle_rcs(elevation):
    """Return analytical physical-optics circular-plate RCS in square metres."""

    elevation = np.deg2rad(np.asarray(elevation))
    argument = 4 * np.pi * CIRCLE_RADIUS * np.cos(elevation) / WAVELENGTH
    airy = np.ones_like(argument)
    nonzero = np.abs(argument) > np.finfo(float).eps
    airy[nonzero] = 2 * j1(argument[nonzero]) / argument[nonzero]
    aperture = np.pi * CIRCLE_RADIUS**2
    return 4 * np.pi * aperture**2 / WAVELENGTH**2 * np.sin(elevation) ** 2 * airy**2


def _dbsm(rcs):
    """Convert square metres to dBsm without producing infinities."""

    return 10 * np.log10(np.maximum(rcs, np.finfo(float).tiny))


def _error_metrics(candidate, reference):
    """Return direct dB error statistics for two full-wave curves."""

    difference = np.asarray(candidate) - np.asarray(reference)
    return {
        "rms_difference_db": float(np.sqrt(np.mean(difference**2))),
        "mean_difference_db": float(np.mean(difference)),
        "maximum_absolute_difference_db": float(np.max(np.abs(difference))),
    }


def _load_case(target, mesh):
    """Load one matching gprMax/MATLAB result pair when it exists."""

    gprmax_path = RESULTS_DIR / f"{target}_{mesh}_gprmax.csv"
    matlab_path = RESULTS_DIR / f"{target}_{mesh}_matlab.csv"
    if not gprmax_path.exists():
        return None
    gprmax = _read_csv(gprmax_path)
    matlab = (
        _read_csv(matlab_path) if matlab_path.exists() else _read_matlab_reference(target, mesh)
    )
    if matlab is None:
        return None
    if not np.allclose(gprmax["actual_elevation_deg"], matlab["elevation_deg"], atol=1e-9):
        raise ValueError(f"Angle grids differ for {target}/{mesh}")
    return gprmax, matlab


def _plot_target(target):
    """Plot all available meshes for one target and return metrics."""

    cases = {
        mesh: values
        for mesh in ("coarse", "fine")
        if (values := _load_case(target, mesh)) is not None
    }
    if not cases:
        return None, None
    primary_mesh = "coarse" if "coarse" in cases else next(iter(cases))
    primary_gprmax, primary_matlab = cases[primary_mesh]
    elevation = primary_gprmax["actual_elevation_deg"]
    analytical_function = _square_rcs if target == "square" else _circle_rcs
    smooth_elevation = np.linspace(0.05, 90.0, 1800)
    smooth_analytical_db = _dbsm(analytical_function(smooth_elevation))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.5, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": (2.2, 1)},
    )
    axes[0].plot(
        smooth_elevation,
        np.maximum(smooth_analytical_db, PLOT_FLOOR_DB),
        color="0.25",
        linewidth=1.5,
        linestyle=":",
        label="Analytical PO",
    )
    axes[0].plot(
        elevation,
        np.maximum(primary_matlab["matlab_po_rcs_dbsm"], PLOT_FLOOR_DB),
        color="#d95f02",
        linewidth=1.7,
        marker="x",
        label="MATLAB PO",
    )
    axes[0].plot(
        elevation,
        np.maximum(primary_matlab["matlab_mom_rcs_dbsm"], PLOT_FLOOR_DB),
        color="#7b3294",
        linewidth=2,
        marker="s",
        label="MATLAB MoM",
    )
    colours = {"coarse": "#1b6ca8", "fine": "#2ca02c"}
    markers = {"coarse": "o", "fine": "D"}
    metrics = {}
    for mesh, (gprmax, matlab) in cases.items():
        mesh_elevation = gprmax["actual_elevation_deg"]
        gprmax_db = gprmax["gprmax_rcs_dbsm"]
        linestyle = "-" if mesh_elevation.size == elevation.size else "None"
        axes[0].plot(
            mesh_elevation,
            np.maximum(gprmax_db, PLOT_FLOOR_DB),
            color=colours[mesh],
            linewidth=2,
            linestyle=linestyle,
            marker=markers[mesh],
            label=f"gprMax FDTD ({mesh})",
        )
        difference = gprmax_db - matlab["matlab_mom_rcs_dbsm"]
        axes[1].plot(
            mesh_elevation,
            difference,
            color=colours[mesh],
            linewidth=2,
            linestyle=linestyle,
            marker=markers[mesh],
            label=mesh,
        )
        broadside_index = int(np.argmax(mesh_elevation))
        metrics[mesh] = {
            "number_of_angles": int(mesh_elevation.size),
            "gprmax_vs_matlab_mom": _error_metrics(gprmax_db, matlab["matlab_mom_rcs_dbsm"]),
            "broadside_sample": {
                "elevation_deg": float(mesh_elevation[broadside_index]),
                "gprmax_rcs_dbsm": float(gprmax_db[broadside_index]),
                "matlab_mom_rcs_dbsm": float(matlab["matlab_mom_rcs_dbsm"][broadside_index]),
                "matlab_po_rcs_dbsm": float(matlab["matlab_po_rcs_dbsm"][broadside_index]),
                "analytical_po_rcs_dbsm": float(matlab["analytical_po_rcs_dbsm"][broadside_index]),
            },
        }

    clipped_po = np.maximum(primary_matlab["matlab_po_rcs_dbsm"], PLOT_FLOOR_DB)
    clipped_analytic = np.maximum(primary_matlab["analytical_po_rcs_dbsm"], PLOT_FLOOR_DB)
    metrics["matlab_po_vs_analytical_po_clipped"] = _error_metrics(clipped_po, clipped_analytic)
    if "coarse" in cases and "fine" in cases:
        coarse = cases["coarse"][0]
        fine = cases["fine"][0]
        common, coarse_index, fine_index = np.intersect1d(
            np.round(coarse["actual_elevation_deg"], 9),
            np.round(fine["actual_elevation_deg"], 9),
            return_indices=True,
        )
        metrics["gprmax_coarse_vs_fine"] = {
            "number_of_common_angles": int(common.size),
            **_error_metrics(
                coarse["gprmax_rcs_dbsm"][coarse_index],
                fine["gprmax_rcs_dbsm"][fine_index],
            ),
        }

    axes[0].set_ylabel("Monostatic HH RCS (dBsm)")
    axes[0].set_ylim(PLOT_FLOOR_DB, 15)
    target_title = "Square" if target == "square" else "Circular"
    axes[0].set_title(f"{target_title} PEC plate at {299792458 / WAVELENGTH / 1e9:.4f} GHz")
    axes[0].legend(ncol=2)
    axes[0].grid(alpha=0.35)
    axes[1].axhline(0, color="0.35", linewidth=1, linestyle=":")
    axes[1].set_xlabel("Elevation from plate plane (degrees)")
    axes[1].set_ylabel("gprMax − MATLAB MoM (dB)")
    axes[1].grid(alpha=0.35)
    axes[1].legend()
    fig.tight_layout()
    output = RESULTS_DIR / f"{target}_plate_rcs_comparison.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output, metrics


def main():
    """Generate target plots and the combined metrics JSON file."""

    all_metrics = {
        "frequency_hz": 299792458 / WAVELENGTH,
        "wavelength_m": WAVELENGTH,
        "angle_definition": "elevation from plate plane; monostatic HH",
        "plot_floor_dbsm": PLOT_FLOOR_DB,
        "targets": {},
    }
    for target in ("square", "circle"):
        output, metrics = _plot_target(target)
        if output is not None:
            print(f"Saved {target}-plate RCS comparison to {output}")
            all_metrics["targets"][target] = metrics
    metrics_path = RESULTS_DIR / "plate_rcs_comparison_metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved plate RCS metrics to {metrics_path}")


if __name__ == "__main__":
    main()
