"""Validate layered NTFF against a point-current Green-function solution.

The production FDTD run collects Love currents on a closed Huygens surface
that crosses both planar interfaces.  The reference bypasses that surface:
it applies the independently tested Capoglu transmission-line Green dyadic
directly to the exact discrete current moment consumed by the Hertzian
dipole.  This separates source/surface collection errors from arbitrary
waveform scaling and phase conventions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0

import gprMax

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "layered_point_dipole"
DL = 0.001
DOMAIN = 0.120
ORIGIN = np.asarray((0.060, 0.060, 0.070))
SOURCE = np.asarray((0.060, 0.060, 0.060))
INTERFACES = np.asarray((0.070, 0.050))
MATERIAL_IDS = ("free_space", "film", "lower")
ER = np.asarray((1.0, 3.2, 2.1))
MR = np.asarray((1.0, 1.15, 1.0))
SIGMA = np.asarray((0.0, 0.03, 0.0))
FREQUENCIES = np.linspace(1.0e9, 3.0e9, 9)
THETA = np.concatenate((np.arange(5.0, 90.0, 5.0), np.arange(95.0, 180.0, 5.0)))
PHI = np.full(THETA.shape, 35.0)


def build_scene(dl: float = DL) -> gprMax.Scene:
    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="Planar-layered NTFF point-dipole validation"))
    scene.add(gprMax.Domain(p1=(DOMAIN,) * 3))
    scene.add(gprMax.Discretisation(p1=(dl,) * 3))
    scene.add(gprMax.TimeWindow(time=5e-9))
    scene.add(gprMax.Material(er=ER[1], se=SIGMA[1], mr=MR[1], sm=0, id="film"))
    scene.add(gprMax.Material(er=ER[2], se=SIGMA[2], mr=MR[2], sm=0, id="lower"))
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(DOMAIN, DOMAIN, INTERFACES[1]), material_id="lower"))
    scene.add(
        gprMax.Box(
            p1=(0, 0, INTERFACES[1]),
            p2=(DOMAIN, DOMAIN, INTERFACES[0]),
            material_id="film",
        )
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=2e9, id="pulse"))
    scene.add(gprMax.HertzianDipole(p1=tuple(SOURCE), polarisation="z", waveform_id="pulse"))
    scene.add(
        gprMax.NTFFSurface(
            p1=(0.028, 0.028, 0.028),
            p2=(0.092, 0.092, 0.092),
            id="surface",
            origin=tuple(ORIGIN),
        )
    )
    scene.add(
        gprMax.NTFFLayeredBackground(
            id="three_layer",
            axis="z",
            materials=MATERIAL_IDS,
            interfaces=tuple(INTERFACES),
        )
    )
    scene.add(
        gprMax.NTFFLayeredFrequencyTransform(
            surface_id="surface",
            id="spectrum",
            background_id="three_layer",
            frequencies=FREQUENCIES,
            window="rectangular",
            save_surface_dft=False,
        )
    )
    scene.add(
        gprMax.NTFFFarField(
            theta=THETA,
            phi=PHI,
            transform_id="spectrum",
            id="angular",
            outputs=("Etheta", "Ephi"),
        )
    )
    return scene


def _source_moment(output: h5py.File) -> np.ndarray:
    excitation = output["srcs/src1/excitation"]
    samples = np.asarray(excitation["samples"])
    dt = float(excitation.attrs["SampleInterval"])
    offset = float(excitation.attrs["TimeSampleOffset"])
    time = offset + dt * np.arange(samples.size)
    spectrum = dt * np.sum(
        samples[np.newaxis, :] * np.exp(-1j * 2 * np.pi * FREQUENCIES[:, None] * time),
        axis=1,
    )
    return spectrum * float(excitation.attrs["SpatialScale"])


def _reference(output: h5py.File):
    """Closed-form three-layer result from Eqs. (33) and (36) of [CAP2012]."""

    moment = _source_moment(output)
    source_position = np.asarray(output["srcs/src1"].attrs["Position"])
    omega = 2 * np.pi * FREQUENCIES[:, None]
    eps_absolute = ER[np.newaxis, :] + SIGMA[np.newaxis, :] / (1j * omega * epsilon_0)
    mu_absolute = np.broadcast_to(MR, eps_absolute.shape).astype(np.complex128)
    theta = np.deg2rad(THETA)
    phi = np.deg2rad(PHI)
    etheta = np.empty((FREQUENCIES.size, THETA.size), dtype=np.complex128)
    ephi = np.empty_like(etheta)
    source_depth = float(source_position[2] - INTERFACES[0])
    film_thickness = float(INTERFACES[0] - INTERFACES[1])

    def outgoing_sqrt(value):
        root = np.sqrt(value + 0j)
        if root.imag > 0 or (abs(root.imag) < 1e-14 and root.real < 0):
            root = -root
        return root

    def voltage_response(line_impedance, beta, upper):
        # Eqs. (18)--(30) imply this interface convention. The numerator
        # printed in Eq. (38) has the opposite sign and is inconsistent with
        # both those recursions and the homogeneous-medium limit.
        gamma10 = (line_impedance[0] - line_impedance[1]) / (line_impedance[1] + line_impedance[0])
        gamma12 = (line_impedance[2] - line_impedance[1]) / (line_impedance[1] + line_impedance[2])
        denominator = 1 - gamma10 * gamma12 * np.exp(-2j * beta[1] * film_thickness)
        if upper:
            coefficient = (1 + gamma10) / denominator
            down = np.exp(1j * beta[1] * source_depth)
            up = gamma12 * np.exp(-1j * beta[1] * (2 * film_thickness + source_depth))
            return coefficient * (down - up)
        coefficient = -(
            (1 + gamma12) * np.exp(-1j * (beta[1] - beta[2]) * film_thickness) / denominator
        )
        up = np.exp(-1j * beta[1] * source_depth)
        down = gamma10 * np.exp(1j * beta[1] * source_depth)
        return coefficient * (up - down)

    for frequency_number, frequency in enumerate(FREQUENCIES):
        for direction_number, (theta_value, phi_value) in enumerate(zip(theta, phi)):
            upper = np.cos(theta_value) > 0
            exterior = 0 if upper else 2
            eps_observation = float(np.real(eps_absolute[frequency_number, exterior]))
            mu_observation = float(np.real(mu_absolute[frequency_number, exterior]))
            eps = eps_absolute[frequency_number] / eps_observation
            mu = mu_absolute[frequency_number] / mu_observation
            sin_theta = float(np.sin(theta_value))
            q = np.asarray([outgoing_sqrt(value) for value in eps * mu - sin_theta**2])
            k = (
                2
                * np.pi
                * frequency
                * np.sqrt(epsilon_0 * eps_observation * 4e-7 * np.pi * mu_observation)
            )
            beta = k * q
            vv_e = voltage_response(q / eps, beta, upper)
            factor = 1j * 2 * np.pi * frequency * (4e-7 * np.pi) * mu_observation / (4 * np.pi)
            # The leading sign in the layered dyadic (8) is positive for
            # upper-half-space observation and negative for the lower half.
            factor *= 1 if upper else -1
            etheta[frequency_number, direction_number] = (
                factor
                * vv_e
                * (eps_observation / eps_absolute[frequency_number, 1])
                * np.sin(theta_value)
                * moment[frequency_number]
            )
            ephi[frequency_number, direction_number] = 0
    return etheta, ephi


def compare(output_path: Path) -> dict[str, float | bool]:
    with h5py.File(output_path, "r") as output:
        spatial_step = float(np.asarray(output.attrs["dx_dy_dz"])[0])
        fields = output["ntff/surface/frequency/spectrum/far_field/angular/fields"]
        actual_theta = np.asarray(fields["Etheta"])
        actual_phi = np.asarray(fields["Ephi"])
        expected_theta, expected_phi = _reference(output)

    actual = np.stack((actual_theta, actual_phi), axis=-1)
    expected = np.stack((expected_theta, expected_phi), axis=-1)
    scale = np.max(np.linalg.norm(expected, axis=-1))
    error = np.linalg.norm(actual - expected, axis=-1) / scale
    active = np.linalg.norm(expected, axis=-1) > 1e-3 * scale
    relative_active = (
        np.linalg.norm(actual - expected, axis=-1)[active]
        / np.linalg.norm(expected, axis=-1)[active]
    )
    resolution_ratio = spatial_step / DL
    maximum_limit = 0.03 * resolution_ratio
    rms_limit = 0.012 * resolution_ratio
    relative_limit = 0.08 * resolution_ratio
    metrics = {
        "spatial_step_m": spatial_step,
        "maximum_error_normalised_to_peak": float(np.max(error)),
        "rms_error_normalised_to_peak": float(np.sqrt(np.mean(error**2))),
        "maximum_relative_error_above_minus_60_db": float(np.max(relative_active)),
        "maximum_error_limit": maximum_limit,
        "rms_error_limit": rms_limit,
        "maximum_relative_error_limit": relative_limit,
    }
    metrics["passed"] = bool(
        metrics["maximum_error_normalised_to_peak"] <= maximum_limit
        and metrics["rms_error_normalised_to_peak"] <= rms_limit
        and metrics["maximum_relative_error_above_minus_60_db"] <= relative_limit
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    central = FREQUENCIES.size // 2
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.5), sharex=True)
    for component, avalues, evalues, marker, linestyle in (
        (r"$E_\theta$", actual_theta, expected_theta, "o", "-"),
        (r"$E_\phi$", actual_phi, expected_phi, "s", "--"),
    ):
        reference_scale = np.max(np.abs(expected[central]))
        axes[0].plot(
            THETA,
            20 * np.log10(np.maximum(np.abs(evalues[central]) / reference_scale, 1e-8)),
            color="k",
            linestyle=linestyle,
            label=f"analytical {component}",
        )
        axes[0].plot(
            THETA,
            20 * np.log10(np.maximum(np.abs(avalues[central]) / reference_scale, 1e-8)),
            marker=marker,
            linestyle="none",
            markerfacecolor="none",
            color="k",
            markevery=2,
            label=f"FDTD {component}",
        )
    axes[0].set_ylabel("normalised magnitude (dB)")
    axes[0].set_ylim(-70, 5)
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(THETA, 100 * error[central], "ko-", markerfacecolor="none")
    axes[1].set_xlabel(r"$\theta$ (degrees), $\phi=35^\circ$")
    axes[1].set_ylabel("vector error / peak (%)")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle(f"Three-layer point dipole at {FREQUENCIES[central] / 1e9:g} GHz")
    fig.tight_layout()
    fig.savefig(output_path.with_name("layered_point_dipole.png"), dpi=220)
    plt.close(fig)

    table = np.column_stack(
        (
            np.repeat(FREQUENCIES, THETA.size),
            np.tile(THETA, FREQUENCIES.size),
            actual_theta.real.ravel(),
            actual_theta.imag.ravel(),
            expected_theta.real.ravel(),
            expected_theta.imag.ravel(),
            actual_phi.real.ravel(),
            actual_phi.imag.ravel(),
            expected_phi.real.ravel(),
            expected_phi.imag.ravel(),
            error.ravel(),
        )
    )
    np.savetxt(
        output_path.with_name("layered_point_dipole.csv"),
        table,
        delimiter=",",
        header=(
            "frequency_hz,theta_deg,Etheta_fdtd_real,Etheta_fdtd_imag,"
            "Etheta_analytical_real,Etheta_analytical_imag,Ephi_fdtd_real,"
            "Ephi_fdtd_imag,Ephi_analytical_real,Ephi_analytical_imag,"
            "error_peak_normalised"
        ),
        comments="",
    )
    output_path.with_name("summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT.with_suffix(".h5"))
    parser.add_argument("--gpu", type=int, help="CUDA device index; omit for double-precision CPU")
    parser.add_argument("--dl", type=float, default=DL, help="uniform spatial step (m)")
    parser.add_argument("--no-run", action="store_true", help="analyse an existing output")
    args = parser.parse_args()
    if not args.no_run:
        options = {"cpu_precision": "double"}
        if args.gpu is not None:
            options = {"gpu": [args.gpu], "gpu_precision": "double"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        gprMax.run(
            scenes=[build_scene(args.dl)],
            n=1,
            outputfile=args.output.with_suffix(""),
            hide_progress_bars=True,
            **options,
        )
    metrics = compare(args.output)
    print(json.dumps(metrics, indent=2))
    if not metrics["passed"]:
        raise SystemExit("planar-layered NTFF analytical validation failed")


if __name__ == "__main__":
    main()
