"""Validate layered NTFF against Engheta's interfacial-dipole patterns.

An infinitesimal electric-current element is placed exactly on the planar
interface between free space and a lossless dielectric with refractive index
``n``.  The FDTD Love-current result is compared with the independent
asymptotic Poynting-pattern expressions of Engheta, Pappas, and Elachi.

The vertical dipole is azimuthally symmetric.  For the horizontal x-directed
dipole, the phi=0 and phi=90 degree principal planes independently exercise
TM (Etheta) and TE (Ephi) propagation.

Reference
---------
N. Engheta, C. H. Pappas, and C. Elachi, "Radiation patterns of
interfacial dipole antennas," Radio Science, 17(6), 1557--1566, 1982,
doi:10.1029/RS017i006p01557.  The equations used below are (3D.2)--(3D.4)
and (4D.2)--(4D.4) in Engheta's 1982 Caltech thesis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import physical_constants

import gprMax


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "engheta_interfacial_dipoles"

DL = 1.0e-3
DOMAIN = 0.120
INTERFACE = DOMAIN / 2
SOURCE = (DOMAIN / 2, DOMAIN / 2, INTERFACE)
ORIGIN = SOURCE
FREQUENCY = 2.0e9
TIME_WINDOW = 5.0e-9
THETA = np.concatenate((np.arange(0.0, 90.0), np.arange(91.0, 181.0)))
ETA0 = physical_constants["characteristic impedance of vacuum"][0]


def build_scene(refractive_index: float, polarisation: str, dl: float = DL) -> gprMax.Scene:
    """Build one of the lossless interfacial-dipole configurations."""

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name=(f"Engheta interfacial {polarisation}-dipole, " f"n={refractive_index:g}")))
    scene.add(gprMax.Domain(p1=(DOMAIN,) * 3))
    scene.add(gprMax.Discretisation(p1=(dl,) * 3))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW))
    scene.add(
        gprMax.Material(
            er=refractive_index**2,
            se=0,
            mr=1,
            sm=0,
            id="lower",
        )
    )
    scene.add(
        gprMax.Box(
            p1=(0, 0, 0),
            p2=(DOMAIN, DOMAIN, INTERFACE),
            material_id="lower",
        )
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=FREQUENCY, id="pulse"))
    scene.add(
        gprMax.HertzianDipole(
            p1=SOURCE,
            polarisation=polarisation,
            waveform_id="pulse",
        )
    )
    scene.add(
        gprMax.NTFFSurface(
            p1=(0.025,) * 3,
            p2=(DOMAIN - 0.025,) * 3,
            id="surface",
            origin=ORIGIN,
        )
    )
    scene.add(
        gprMax.NTFFLayeredBackground(
            id="half_space",
            axis="z",
            materials=("free_space", "lower"),
            interfaces=(INTERFACE,),
        )
    )
    scene.add(
        gprMax.NTFFLayeredFrequencyTransform(
            surface_id="surface",
            id="spectrum",
            background_id="half_space",
            frequencies=(FREQUENCY,),
            window="rectangular",
            save_surface_dft=False,
        )
    )

    if polarisation == "z":
        scene.add(
            gprMax.NTFFFarField(
                theta=THETA,
                phi=np.zeros_like(THETA),
                transform_id="spectrum",
                id="vertical",
                outputs=("Etheta", "Ephi"),
            )
        )
    else:
        for output_id, phi in (("horizontal_e", 0.0), ("horizontal_h", 90.0)):
            scene.add(
                gprMax.NTFFFarField(
                    theta=THETA,
                    phi=np.full_like(THETA, phi),
                    transform_id="spectrum",
                    id=output_id,
                    outputs=("Etheta", "Ephi"),
                )
            )
    return scene


def _vertical_power(theta: np.ndarray, n: float) -> np.ndarray:
    """Engheta thesis Eqs. (3D.2)--(3D.4), without common constants."""

    theta = np.asarray(theta, dtype=float)
    power = np.zeros_like(theta)
    sine = np.sin(theta)
    cosine = np.cos(theta)
    theta_c = np.arcsin(1 / n)
    upper = theta <= np.pi / 2
    lower_propagating = theta >= np.pi - theta_c
    lower_lateral = (~upper) & (~lower_propagating)

    root_upper = np.sqrt(np.maximum(n**2 - sine**2, 0))
    power[upper] = n**4 * sine[upper] ** 2 * cosine[upper] ** 2 / (n**2 * cosine[upper] + root_upper[upper]) ** 2

    root_lower = np.sqrt(np.maximum(1 - n**2 * sine**2, 0))
    power[lower_propagating] = (
        n**5
        * sine[lower_propagating] ** 2
        * cosine[lower_propagating] ** 2
        / (n * root_lower[lower_propagating] - cosine[lower_propagating]) ** 2
    )
    power[lower_lateral] = (
        n**5
        * sine[lower_lateral] ** 2
        * cosine[lower_lateral] ** 2
        / (n**2 * (n**2 * sine[lower_lateral] ** 2 - 1) + cosine[lower_lateral] ** 2)
    )
    return power


def _horizontal_power(theta: np.ndarray, phi: float, n: float) -> np.ndarray:
    """Engheta thesis Eqs. (4D.2)--(4D.4), without common constants."""

    theta = np.asarray(theta, dtype=float)
    power = np.zeros_like(theta)
    sine = np.sin(theta)
    cosine = np.cos(theta)
    cos_phi2 = np.cos(phi) ** 2
    sin_phi2 = np.sin(phi) ** 2
    theta_c = np.arcsin(1 / n)
    upper = theta <= np.pi / 2
    lower_propagating = theta >= np.pi - theta_c
    lower_lateral = (~upper) & (~lower_propagating)

    root_upper = np.sqrt(np.maximum(n**2 - sine**2, 0))
    amplitude = cosine**2 / (cosine + root_upper) - sine**2 * cosine * (cosine - root_upper) / (
        n**2 * cosine + root_upper
    )
    power[upper] = (
        amplitude[upper] ** 2 * cos_phi2 + cosine[upper] ** 2 * sin_phi2 / (cosine[upper] + root_upper[upper]) ** 2
    )

    root_lower = np.sqrt(np.maximum(1 - n**2 * sine**2, 0))
    amplitude = sine**2 * cosine * (root_lower + n * cosine) / (n * root_lower - cosine) - cosine**2 / (
        root_lower - n * cosine
    )
    power[lower_propagating] = n**3 * (
        amplitude[lower_propagating] ** 2 * cos_phi2
        + cosine[lower_propagating] ** 2
        * sin_phi2
        / (root_lower[lower_propagating] - n * cosine[lower_propagating]) ** 2
    )

    denominator = n**2 * (n**2 * sine**2 - 1) + cosine**2
    power[lower_lateral] = n**3 * (
        (
            (n**2 - 1) * sine[lower_lateral] ** 4 * cosine[lower_lateral] ** 2 * cos_phi2
            - 2 * cos_phi2 * sine[lower_lateral] ** 2 * cosine[lower_lateral] ** 4
        )
        / denominator[lower_lateral]
        + (cosine[lower_lateral] ** 4 * cos_phi2 + sin_phi2 * cosine[lower_lateral] ** 2) / (n**2 - 1)
    )
    return power


def _read_power(output: h5py.File, output_id: str, n: float) -> np.ndarray:
    fields = output[f"ntff/surface/frequency/spectrum/far_field/{output_id}/fields"]
    etheta = np.asarray(fields["Etheta"])[0]
    ephi = np.asarray(fields["Ephi"])[0]
    inverse_relative_impedance = np.where(THETA < 90, 1.0, n)
    return inverse_relative_impedance * (np.abs(etheta) ** 2 + np.abs(ephi) ** 2)


def _normalise_pair(actual: np.ndarray, analytical: np.ndarray):
    if max(float(np.max(actual)), float(np.max(analytical))) == 0:
        raise ValueError("cannot normalise a zero radiation pattern")
    actual = actual / np.max(actual)
    analytical = analytical / np.max(analytical)
    return actual, analytical


def compare_case(output_path: Path, n: float, polarisation: str):
    theta_radians = np.deg2rad(THETA)
    with h5py.File(output_path, "r") as output:
        if polarisation == "z":
            actual = {"vertical": _read_power(output, "vertical", n)}
            analytical = {"vertical": _vertical_power(theta_radians, n)}
        else:
            actual = {
                "horizontal_e": _read_power(output, "horizontal_e", n),
                "horizontal_h": _read_power(output, "horizontal_h", n),
            }
            analytical = {
                "horizontal_e": _horizontal_power(theta_radians, 0.0, n),
                "horizontal_h": _horizontal_power(theta_radians, np.pi / 2, n),
            }

    metrics = {}
    normalised = {}
    theta_c = np.rad2deg(np.arcsin(1 / n))
    regular = (np.abs(THETA - 90) >= 3) & (np.abs(THETA - (180 - theta_c)) >= 3)
    for output_id in actual:
        fdtd, theory = _normalise_pair(actual[output_id], analytical[output_id])
        error = fdtd - theory
        normalised[output_id] = (fdtd, theory, error)
        metrics[output_id] = {
            "rms_error_normalised_power": float(np.sqrt(np.mean(error[regular] ** 2))),
            "maximum_error_normalised_power": float(np.max(np.abs(error[regular]))),
            "nearest_interface_fdtd_normalised_power": float(max(fdtd[THETA == 89][0], fdtd[THETA == 91][0])),
        }
    return metrics, normalised


def _plot(results, output_directory: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(10.0, 10.5), sharex=True, sharey=True)
    rows = (
        ("vertical", "vertical dipole"),
        ("horizontal_e", r"horizontal, $\phi=0^\circ$"),
        ("horizontal_h", r"horizontal, $\phi=90^\circ$"),
    )
    for column, n in enumerate((2.0, 4.0)):
        for row, (output_id, title) in enumerate(rows):
            axis = axes[row, column]
            fdtd, theory, _ = results[n][output_id]
            axis.plot(THETA, theory, "k-", label="Engheta analytical")
            axis.plot(
                THETA,
                fdtd,
                "ko",
                markerfacecolor="white",
                markersize=3,
                markevery=4,
                label="gprMax FDTD",
            )
            axis.axvline(90, color="0.65", linewidth=0.8)
            axis.axvline(
                180 - np.rad2deg(np.arcsin(1 / n)),
                color="0.65",
                linewidth=0.8,
                linestyle=":",
            )
            axis.set_title(f"{title}, n={n:g}")
            axis.grid(True, alpha=0.2)
            if row == 2:
                axis.set_xlabel(r"$\theta$ (degrees)")
            if column == 0:
                axis.set_ylabel("normalised radial power")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Interfacial electric-dipole radiation patterns")
    fig.tight_layout()
    fig.savefig(output_directory / "engheta_interfacial_dipoles.png", dpi=240)
    plt.close(fig)


def _full_plane(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror a symmetric 0--180 degree principal-plane cut to 360 degrees."""

    angles = np.concatenate((THETA, 360 - THETA[-2::-1]))
    return np.deg2rad(angles), np.concatenate((values, values[-2::-1]))


def _power_db(values: np.ndarray, floor: float = -40.0) -> np.ndarray:
    """Return normalised power in dB with a finite plotting floor."""

    values = np.asarray(values, dtype=float)
    values = values / np.max(values)
    return np.maximum(10 * np.log10(np.maximum(values, np.finfo(float).tiny)), floor)


def _plot_polar(results, output_directory: Path) -> None:
    """Plot complete principal-plane patterns using azimuthal symmetry."""

    fig, axes = plt.subplots(3, 2, figsize=(10.0, 12.0), subplot_kw={"projection": "polar"})
    rows = (
        ("vertical", "vertical dipole"),
        ("horizontal_e", r"horizontal, E plane ($\phi=0/180^\circ$)"),
        ("horizontal_h", r"horizontal, H plane ($\phi=90/270^\circ$)"),
    )
    for column, n in enumerate((2.0, 4.0)):
        for row, (output_id, title) in enumerate(rows):
            axis = axes[row, column]
            fdtd, theory, _ = results[n][output_id]
            angles, theory_full = _full_plane(theory)
            _, fdtd_full = _full_plane(fdtd)
            axis.plot(angles, _power_db(theory_full), "k-", linewidth=1.2, label="Engheta analytical")
            axis.plot(
                angles,
                _power_db(fdtd_full),
                "ko",
                markerfacecolor="white",
                markersize=2.5,
                markevery=8,
                label="gprMax FDTD",
            )
            axis.set_theta_zero_location("N")
            axis.set_theta_direction(-1)
            axis.set_rlim(-40, 0)
            axis.set_rticks((-40, -30, -20, -10, 0))
            axis.set_rlabel_position(135)
            axis.grid(True, alpha=0.3)
            axis.set_title(f"{title}, n={n:g}", pad=16)
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(-0.25, 1.18), fontsize=8)
    fig.suptitle("Interfacial electric-dipole radiation patterns (normalised power, dB)", y=0.995)
    fig.tight_layout()
    fig.savefig(output_directory / "engheta_interfacial_dipoles_polar.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", type=int, help="CUDA device index; omit for double-precision CPU")
    parser.add_argument("--dl", type=float, default=DL, help="uniform spatial step (m)")
    parser.add_argument("--no-run", action="store_true", help="analyse existing outputs")
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)

    metrics = {}
    plot_results = {}
    for n in (2.0, 4.0):
        metrics[n] = {}
        plot_results[n] = {}
        for polarisation in ("z", "x"):
            output_path = args.output_directory / f"n{n:g}_{polarisation}.h5"
            if not args.no_run:
                options = {"cpu_precision": "double"}
                if args.gpu is not None:
                    options = {"gpu": [args.gpu], "gpu_precision": "double"}
                gprMax.run(
                    scenes=[build_scene(n, polarisation, args.dl)],
                    n=1,
                    outputfile=output_path.with_suffix(""),
                    hide_progress_bars=True,
                    **options,
                )
            case_metrics, case_results = compare_case(output_path, n, polarisation)
            metrics[n].update(case_metrics)
            plot_results[n].update(case_results)

    _plot(plot_results, args.output_directory)
    _plot_polar(plot_results, args.output_directory)
    (args.output_directory / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
