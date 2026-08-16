"""Compare gprMax and CST horn-antenna principal-plane far fields."""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
GPRMAX_OUTPUT_PATH = HERE / "horn_antenna.h5"
CST_PRINCIPAL_PLANE_PATH = HERE / "horn_farfield_principal_planes_cst.csv"
FAR_FIELD_PLOT_PATH = HERE / "horn_antenna_farfield_polar_comparison.png"
NTFF_ROOT = "ntff/horn_surface/frequency/horn_ntff"
FREQUENCIES_GHZ = (8, 9, 10, 11, 12)
DIRECTIVITY_FLOOR_DBI = -40.0
DIRECTIVITY_CEILING_DBI = 25.0

SOLVER_STYLES = {
    "gprMax": {
        "color": "#0072b2",
        "linestyle": "-",
        "linewidth": 2.4,
        "zorder": 3,
    },
    "CST FIT": {
        "color": "#e69f00",
        "linestyle": (0, (5, 2)),
        "linewidth": 2.4,
        "zorder": 5,
    },
    "CST FEM (adaptive mesh refinement)": {
        "color": "#cc79a7",
        "linestyle": (0, (2, 2)),
        "linewidth": 2.4,
        "zorder": 6,
    },
}
CST_SOLVER_CODES = {
    "CST FIT": "fit",
    "CST FEM (adaptive mesh refinement)": "fem",
}
CST_PLANES = ("xz", "yz")
CST_CUT_COLUMNS = (
    "solver",
    "frequency_ghz",
    "plane",
    "angle_deg",
    "directivity_dbi",
)


def signed_principal_plane_cut(
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    directivity_dbi: np.ndarray,
    positive_phi_deg: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a closed signed-angle cut for a pair of opposite azimuths."""

    negative_phi_deg = (positive_phi_deg + 180.0) % 360.0
    positive = np.isclose(phi_deg, positive_phi_deg, rtol=0, atol=1e-5)
    negative = (
        np.isclose(phi_deg, negative_phi_deg, rtol=0, atol=1e-5)
        & (theta_deg > 0.0)
        & (theta_deg < 180.0)
    )
    if not np.any(positive) or not np.any(negative):
        return None

    angle_deg = np.concatenate((-theta_deg[negative], theta_deg[positive]))
    if directivity_dbi.ndim == 1:
        cut = np.concatenate((directivity_dbi[negative], directivity_dbi[positive]))
    else:
        cut = np.concatenate(
            (directivity_dbi[:, negative], directivity_dbi[:, positive]), axis=1
        )
    order = np.argsort(angle_deg)
    angle_deg = angle_deg[order]
    cut = cut[order] if cut.ndim == 1 else cut[:, order]

    # Close the polar trace at -180 degrees using the equivalent +180 pole.
    if angle_deg[0] > -180.0:
        pole = cut[-1] if cut.ndim == 1 else cut[:, -1:]
        angle_deg = np.concatenate(([-180.0], angle_deg))
        cut = (
            np.concatenate(([pole], cut))
            if cut.ndim == 1
            else np.concatenate((pole, cut), axis=1)
        )
    return angle_deg, cut


def read_gprmax_far_fields() -> tuple[
    np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]
]:
    with h5py.File(GPRMAX_OUTPUT_PATH, "r") as output:
        ntff = output[NTFF_ROOT]
        frequencies_hz = np.asarray(ntff["frequencies"], dtype=np.float64)
        cuts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for plane, group_name, positive_phi_deg in (
            ("xz", "xz_plane", 0.0),
            ("yz", "yz_plane", 90.0),
        ):
            group = ntff[f"far_field/{group_name}"]
            cut = signed_principal_plane_cut(
                np.asarray(group["theta"], dtype=np.float64),
                np.asarray(group["phi"], dtype=np.float64),
                np.asarray(group["fields/directivity_dbi"], dtype=np.float64),
                positive_phi_deg,
            )
            if cut is None:
                raise ValueError(f"gprMax output is missing the {plane} plane")
            cuts[plane] = cut
    return frequencies_hz, cuts


@lru_cache(maxsize=None)
def read_cst_far_field_table(path: Path) -> np.ndarray:
    """Load each full-sphere CST export once while plotting both plane cuts."""

    values = np.loadtxt(path, skiprows=2)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"Unexpected CST far-field shape {values.shape} in {path}")
    return values


def read_cst_far_field(path: Path, plane: str) -> tuple[np.ndarray, np.ndarray] | None:
    values = read_cst_far_field_table(path)
    positive_phi_deg = {"xz": 0.0, "yz": 90.0}[plane]
    return signed_principal_plane_cut(
        values[:, 0], values[:, 1], values[:, 2], positive_phi_deg
    )


def validate_cst_cut(
    key: tuple[str, int, str], angle_deg: np.ndarray, directivity_dbi: np.ndarray
) -> None:
    """Reject incomplete or malformed retained principal-plane cuts."""

    if angle_deg.ndim != 1 or directivity_dbi.ndim != 1:
        raise ValueError(f"CST cut {key} must contain one-dimensional arrays")
    if angle_deg.shape != directivity_dbi.shape:
        raise ValueError(
            f"CST cut {key} has {angle_deg.size} angles and "
            f"{directivity_dbi.size} directivity values"
        )
    if not np.all(np.isfinite(angle_deg)) or not np.all(np.isfinite(directivity_dbi)):
        raise ValueError(f"CST cut {key} contains non-finite values")
    expected_angles = np.arange(-180.0, 181.0, dtype=np.float64)
    if not np.array_equal(angle_deg, expected_angles):
        raise ValueError(
            f"CST cut {key} must contain one-degree samples from -180 to 180"
        )


def expected_cst_cut_keys() -> set[tuple[str, int, str]]:
    return {
        (solver_code, frequency_ghz, plane)
        for solver_code in CST_SOLVER_CODES.values()
        for frequency_ghz in FREQUENCIES_GHZ
        for plane in CST_PLANES
    }


def extract_cst_principal_plane_cuts() -> dict[
    tuple[str, int, str], tuple[np.ndarray, np.ndarray]
]:
    """Extract the retained cuts from the optional full-sphere CST exports."""

    cuts: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]] = {}
    for solver_code in CST_SOLVER_CODES.values():
        for frequency_ghz in FREQUENCIES_GHZ:
            raw_path = HERE / f"horn_ff_{solver_code}_cst_{frequency_ghz}ghz.txt"
            if not raw_path.is_file():
                raise FileNotFoundError(
                    f"Missing raw CST export required for refresh/audit: {raw_path}"
                )
            for plane in CST_PLANES:
                key = (solver_code, frequency_ghz, plane)
                cut = read_cst_far_field(raw_path, plane)
                if cut is None:
                    raise ValueError(f"Raw CST export {raw_path} is missing {plane}")
                validate_cst_cut(key, *cut)
                cuts[key] = cut
    return cuts


def write_cst_principal_plane_cuts(
    cuts: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]],
    path: Path = CST_PRINCIPAL_PLANE_PATH,
) -> None:
    """Write a deterministic, lossless text representation of the compact cuts."""

    if set(cuts) != expected_cst_cut_keys():
        raise ValueError("Cannot write an incomplete set of CST principal-plane cuts")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(CST_CUT_COLUMNS)
        for solver_code in CST_SOLVER_CODES.values():
            for frequency_ghz in FREQUENCIES_GHZ:
                for plane in CST_PLANES:
                    key = (solver_code, frequency_ghz, plane)
                    angle_deg, directivity_dbi = cuts[key]
                    validate_cst_cut(key, angle_deg, directivity_dbi)
                    writer.writerows(
                        (
                            solver_code,
                            frequency_ghz,
                            plane,
                            format(float(angle), ".17g"),
                            format(float(directivity), ".17g"),
                        )
                        for angle, directivity in zip(angle_deg, directivity_dbi)
                    )


def read_cst_principal_plane_cuts(
    path: Path = CST_PRINCIPAL_PLANE_PATH,
) -> dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]]:
    """Load and validate the compact principal-plane data retained in Git."""

    rows: dict[tuple[str, int, str], list[tuple[float, float]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CST_CUT_COLUMNS:
            raise ValueError(
                f"Expected columns {CST_CUT_COLUMNS} in {path}, got {reader.fieldnames}"
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                key = (
                    row["solver"],
                    int(row["frequency_ghz"]),
                    row["plane"],
                )
                sample = (float(row["angle_deg"]), float(row["directivity_dbi"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid CST cut data at {path}:{row_number}") from exc
            rows.setdefault(key, []).append(sample)

    if set(rows) != expected_cst_cut_keys():
        missing = sorted(expected_cst_cut_keys() - set(rows))
        unexpected = sorted(set(rows) - expected_cst_cut_keys())
        raise ValueError(
            f"Incomplete CST cut table {path}; missing={missing}, unexpected={unexpected}"
        )

    cuts: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]] = {}
    for key, samples in rows.items():
        values = np.asarray(samples, dtype=np.float64)
        order = np.argsort(values[:, 0])
        angle_deg = values[order, 0]
        directivity_dbi = values[order, 1]
        validate_cst_cut(key, angle_deg, directivity_dbi)
        cuts[key] = angle_deg, directivity_dbi
    return cuts


def audit_cst_principal_plane_cuts() -> None:
    """Check that retained cuts exactly reproduce the optional raw exports."""

    retained = read_cst_principal_plane_cuts()
    extracted = extract_cst_principal_plane_cuts()
    for key in sorted(expected_cst_cut_keys()):
        retained_angle, retained_directivity = retained[key]
        raw_angle, raw_directivity = extracted[key]
        if not np.array_equal(retained_angle, raw_angle):
            raise ValueError(f"Retained CST angles differ from raw export for {key}")
        if not np.array_equal(retained_directivity, raw_directivity):
            difference = float(np.max(np.abs(retained_directivity - raw_directivity)))
            raise ValueError(
                f"Retained CST directivity differs from raw export for {key}; "
                f"maximum absolute difference={difference:.17g} dB"
            )


def configure_polar_axis(axis: plt.Axes, plane: str, frequency_ghz: int) -> None:
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    plane_axis = "x" if plane == "xz" else "y"
    axis.set_thetagrids(
        np.arange(0, 360, 45),
        (
            "+z",
            "45°",
            f"+{plane_axis}",
            "135°",
            "−z",
            "−135°",
            f"−{plane_axis}",
            "−45°",
        ),
    )
    axis.set_rlim(0.0, DIRECTIVITY_CEILING_DBI - DIRECTIVITY_FLOOR_DBI)
    directivity_ticks = np.arange(-30.0, 21.0, 10.0)
    axis.set_rticks(directivity_ticks - DIRECTIVITY_FLOOR_DBI)
    axis.set_yticklabels([f"{value:.0f}" for value in directivity_ticks])
    axis.set_rlabel_position(135)
    axis.grid(True, alpha=0.3)
    axis.set_title(f"{frequency_ghz} GHz — {plane} plane", pad=16)


def shifted_polar_radius(directivity_dbi: np.ndarray) -> np.ndarray:
    return (
        np.clip(
            directivity_dbi, DIRECTIVITY_FLOOR_DBI, DIRECTIVITY_CEILING_DBI
        )
        - DIRECTIVITY_FLOOR_DBI
    )


def plot_far_field_comparison() -> None:
    gprmax_frequencies_hz, gprmax_cuts = read_gprmax_far_fields()
    cst_cuts = read_cst_principal_plane_cuts()
    figure, axes = plt.subplots(
        len(FREQUENCIES_GHZ),
        2,
        figsize=(12.0, 25.0),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )

    for row, frequency_ghz in enumerate(FREQUENCIES_GHZ):
        frequency_index = int(
            np.argmin(np.abs(gprmax_frequencies_hz - frequency_ghz * 1e9))
        )
        if not np.isclose(
            gprmax_frequencies_hz[frequency_index],
            frequency_ghz * 1e9,
            rtol=0,
            atol=2e3,
        ):
            raise ValueError(f"No gprMax far field at {frequency_ghz} GHz")

        for column, plane in enumerate(("xz", "yz")):
            axis = axes[row, column]
            configure_polar_axis(axis, plane, frequency_ghz)

            angle_deg, directivity_dbi = gprmax_cuts[plane]
            axis.plot(
                np.deg2rad(angle_deg),
                shifted_polar_radius(directivity_dbi[frequency_index]),
                label="gprMax",
                **SOLVER_STYLES["gprMax"],
            )

            for label, solver_code in CST_SOLVER_CODES.items():
                cst_angle_deg, cst_directivity_dbi = cst_cuts[
                    solver_code, frequency_ghz, plane
                ]
                style = dict(SOLVER_STYLES[label])
                if label == "CST FIT":
                    style.update(
                        marker="o",
                        markersize=2.8,
                        markerfacecolor="white",
                        markeredgewidth=0.8,
                        markevery=18,
                    )
                axis.plot(
                    np.deg2rad(cst_angle_deg),
                    shifted_polar_radius(cst_directivity_dbi),
                    label=label,
                    **style,
                )

    # Collect solver labels from a yz subplot, where all three exports exist.
    handles, labels = axes[0, 1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncols=3)
    figure.savefig(FAR_FIELD_PLOT_PATH, dpi=180)
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the retained gprMax/CST horn comparisons, or refresh/audit the "
            "compact CST principal-plane data from optional raw exports."
        )
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--refresh-cst-cuts",
        action="store_true",
        help=(
            "extract and overwrite the retained compact CST principal-plane CSV "
            "from all ten local full-sphere CST text exports, then exit"
        ),
    )
    actions.add_argument(
        "--audit-cst-cuts",
        action="store_true",
        help=(
            "compare the retained CST principal-plane CSV exactly with all ten "
            "local full-sphere CST text exports, then exit"
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.refresh_cst_cuts:
        write_cst_principal_plane_cuts(extract_cst_principal_plane_cuts())
        print(f"Wrote {CST_PRINCIPAL_PLANE_PATH}")
        return
    if arguments.audit_cst_cuts:
        audit_cst_principal_plane_cuts()
        print(f"Retained CST cuts exactly match the raw exports: {CST_PRINCIPAL_PLANE_PATH}")
        return

    plot_far_field_comparison()
    print(f"Wrote {FAR_FIELD_PLOT_PATH}")


if __name__ == "__main__":
    main()
