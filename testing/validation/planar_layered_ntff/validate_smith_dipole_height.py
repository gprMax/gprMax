"""Validate finite-height half-space patterns following Smith (1984).

The example is an x-directed infinitesimal electric dipole in air above a
lossless dielectric half-space with relative permittivity nine.  The source
heights reproduce the canonical values h/lambda0 = 0.1, 0.2, and 0.35 used
to demonstrate the directive effect of the interface.

The reference is the asymptotic plane-wave-spectrum result of Smith.  For a
source at height h, the interfacial-dipole far field is multiplied by the
upper-medium propagation factor.  In the lower-medium lateral-wave sector
this gives the power attenuation

    exp[-2 k0 h sqrt(n^2 sin^2(theta) - 1)].

References
----------
G. S. Smith, "Directive properties of antennas for transmission into a
material half-space," IEEE TAP, 32(3), 232--246, 1984,
doi:10.1109/TAP.1984.1143307.

The selected epsilon_r=9 height sequence is also reproduced as Fig. 3.14 in
X. L. Travassos, M. F. Pantoja, and N. Ida, Ground Penetrating Radar:
Improving Sensing and Imaging Through Numerical Modeling, IET, 2021.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import speed_of_light

import gprMax

from .validate_engheta_interfacial_dipoles import _horizontal_power


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "smith_dipole_height"

DL = 1.5e-3
DOMAIN = 0.162
INTERFACE = 0.060
CENTRE = DOMAIN / 2
FREQUENCY = 2.0e9
WAVELENGTH = speed_of_light / FREQUENCY
REFRACTIVE_INDEX = 3.0
HEIGHT_RATIOS = (0.1, 0.2, 0.35)
TIME_WINDOW = 5.0e-9
THETA = np.concatenate((np.arange(0.0, 90.0), np.arange(91.0, 181.0)))


def build_scene(height_ratio: float) -> gprMax.Scene:
    """Build the horizontal-dipole half-space configuration."""

    source = (CENTRE, CENTRE, INTERFACE + height_ratio * WAVELENGTH)
    scene = gprMax.Scene()
    scene.add(gprMax.Title(name=f"Smith horizontal dipole, h/lambda0={height_ratio:g}"))
    scene.add(gprMax.Domain(p1=(DOMAIN,) * 3))
    scene.add(gprMax.Discretisation(p1=(DL,) * 3))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW))
    scene.add(gprMax.Material(er=9, se=0, mr=1, sm=0, id="lower"))
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
            p1=source,
            polarisation="x",
            waveform_id="pulse",
        )
    )
    scene.add(
        gprMax.NTFFSurface(
            p1=(0.024,) * 3,
            p2=(0.138,) * 3,
            id="surface",
            origin=(CENTRE, CENTRE, INTERFACE),
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
    for output_id, phi in (("e_plane", 0.0), ("h_plane", 90.0)):
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


def _smith_power(theta: np.ndarray, phi: float, height: float) -> np.ndarray:
    """Return the Smith asymptotic full-space power pattern."""

    interfacial = _horizontal_power(theta, phi, REFRACTIVE_INDEX)
    power = np.array(interfacial, copy=True)
    upper = theta < np.pi / 2
    cosine = np.cos(theta[upper])
    sine = np.sin(theta[upper])
    transmitted_cosine = np.sqrt(1 - (sine / REFRACTIVE_INDEX) ** 2)
    phase = 2 * np.pi * height * cosine / WAVELENGTH
    incident = np.exp(-1j * phase)
    reflected = np.exp(1j * phase)
    if np.isclose(np.mod(phi, np.pi), 0):
        reflection = (REFRACTIVE_INDEX * cosine - transmitted_cosine) / (
            REFRACTIVE_INDEX * cosine + transmitted_cosine
        )
        amplitude = 0.5 * cosine * (incident - reflection * reflected)
    else:
        root = REFRACTIVE_INDEX * transmitted_cosine
        reflection = (cosine - root) / (cosine + root)
        amplitude = 0.5 * (incident + reflection * reflected)
    power[upper] = np.abs(amplitude) ** 2

    lower = theta > np.pi / 2
    transverse = REFRACTIVE_INDEX * np.sin(theta)
    evanescent_decay = np.sqrt(np.maximum(transverse**2 - 1, 0))
    height_factor = np.exp(-2 * (2 * np.pi / WAVELENGTH) * height * evanescent_decay)
    power[lower] = interfacial[lower] * height_factor[lower]
    return power


def _read_power(output: h5py.File, output_id: str) -> np.ndarray:
    fields = output[f"ntff/surface/frequency/spectrum/far_field/{output_id}/fields"]
    etheta = np.asarray(fields["Etheta"])[0]
    ephi = np.asarray(fields["Ephi"])[0]
    inverse_relative_impedance = np.where(THETA < 90, 1.0, REFRACTIVE_INDEX)
    return inverse_relative_impedance * (np.abs(etheta) ** 2 + np.abs(ephi) ** 2)


def compare(output_path: Path):
    with h5py.File(output_path, "r") as output:
        source_height = float(output["srcs/src1"].attrs["Position"][2] - INTERFACE)
        actual = {
            "e_plane": _read_power(output, "e_plane"),
            "h_plane": _read_power(output, "h_plane"),
        }

    theta_radians = np.deg2rad(THETA)
    analytical = {
        "e_plane": _smith_power(theta_radians, 0.0, source_height),
        "h_plane": _smith_power(theta_radians, np.pi / 2, source_height),
    }
    critical = 180 - np.rad2deg(np.arcsin(1 / REFRACTIVE_INDEX))
    regular = (np.abs(THETA - critical) >= 3) & (THETA >= 94)
    lower = THETA > 90
    metrics = {"actual_height_m": source_height}
    curves = {}
    for output_id in actual:
        fdtd_lower = actual[output_id] / np.max(actual[output_id][lower])
        theory_lower = analytical[output_id] / np.max(analytical[output_id][lower])
        error = fdtd_lower - theory_lower
        fdtd = actual[output_id] / np.max(actual[output_id])
        theory = analytical[output_id] / np.max(analytical[output_id])
        curves[output_id] = (fdtd, theory)
        metrics[output_id] = {
            "rms_error_normalised_power": float(np.sqrt(np.mean(error[regular] ** 2))),
            "maximum_error_normalised_power": float(np.max(np.abs(error[regular]))),
        }
    return metrics, curves


def _plot(results, output_directory: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), sharex=True, sharey=True)
    for column, height_ratio in enumerate(HEIGHT_RATIOS):
        for row, (output_id, title) in enumerate(
            (("e_plane", r"E plane, $E_\theta$"), ("h_plane", r"H plane, $E_\phi$"))
        ):
            axis = axes[row, column]
            fdtd, theory = results[height_ratio][output_id]
            axis.plot(THETA, theory, "k-", label="Smith analytical")
            axis.plot(
                THETA,
                fdtd,
                "ko",
                markerfacecolor="white",
                markersize=3,
                markevery=3,
                label="gprMax FDTD",
            )
            axis.axvline(
                180 - np.rad2deg(np.arcsin(1 / REFRACTIVE_INDEX)),
                color="0.65",
                linewidth=0.8,
                linestyle=":",
            )
            axis.set_title(f"{title}, h/$\\lambda_0$={height_ratio:g}")
            axis.grid(True, alpha=0.2)
            if row == 1:
                axis.set_xlabel(r"$\theta$ (degrees)")
            if column == 0:
                axis.set_ylabel("normalised radial power")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(r"Horizontal electric dipole above an $\epsilon_r=9$ half-space")
    fig.tight_layout()
    fig.savefig(output_directory / "smith_dipole_height.png", dpi=240)
    plt.close(fig)


def _full_plane(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror a symmetric 0--180 degree principal-plane cut to 360 degrees."""

    angles = np.concatenate((THETA, 360 - THETA[-2::-1]))
    return np.deg2rad(angles), np.concatenate((values, values[-2::-1]))


def _power_db(values: np.ndarray, floor: float = -40.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values / np.max(values)
    return np.maximum(10 * np.log10(np.maximum(values, np.finfo(float).tiny)), floor)


def _plot_polar(results, output_directory: Path) -> None:
    """Plot complete E- and H-plane patterns in polar form."""

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.0), subplot_kw={"projection": "polar"})
    for column, height_ratio in enumerate(HEIGHT_RATIOS):
        for row, (output_id, title) in enumerate(
            (("e_plane", "E plane"), ("h_plane", "H plane"))
        ):
            axis = axes[row, column]
            fdtd, theory = results[height_ratio][output_id]
            angles, theory_full = _full_plane(theory)
            _, fdtd_full = _full_plane(fdtd)
            axis.plot(angles, _power_db(theory_full), "k-", linewidth=1.2, label="Smith analytical")
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
            axis.set_title(f"{title}, h/$\\lambda_0$={height_ratio:g}", pad=16)
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(-0.25, 1.18), fontsize=8)
    fig.suptitle(
        r"Horizontal electric dipole above an $\epsilon_r=9$ half-space (normalised power, dB)",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(output_directory / "smith_dipole_height_polar.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", type=int, help="CUDA device index; omit for double-precision CPU")
    parser.add_argument("--no-run", action="store_true", help="analyse existing outputs")
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)

    metrics = {}
    plot_results = {}
    for height_ratio in HEIGHT_RATIOS:
        height_token = f"{height_ratio:g}".replace(".", "p")
        output_path = args.output_directory / f"h{height_token}.h5"
        if not args.no_run:
            options = {"cpu_precision": "double"}
            if args.gpu is not None:
                options = {"gpu": [args.gpu], "gpu_precision": "double"}
            gprMax.run(
                scenes=[build_scene(height_ratio)],
                n=1,
                outputfile=output_path.with_suffix(""),
                hide_progress_bars=True,
                **options,
            )
        case_metrics, curves = compare(output_path)
        metrics[height_ratio] = case_metrics
        plot_results[height_ratio] = curves

    _plot(plot_results, args.output_directory)
    _plot_polar(plot_results, args.output_directory)
    (args.output_directory / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
