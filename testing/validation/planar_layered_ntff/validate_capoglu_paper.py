"""Reproduce the eight-layer validation in Figure 2 of Çapoğlu et al.

The published example uses nine differently oriented Hertzian current
elements.  Unlike the simpler vertical-dipole regression in this directory,
it exercises both TE and TM propagation and therefore produces non-zero
``Etheta`` and ``Ephi`` fields.  The FDTD Love-current result is compared with
the same point currents radiating analytically in the unperturbed stack.

Reference
---------
I. R. Çapoğlu, A. Taflove, and V. Backman, "A frequency-domain
near-field-to-far-field transform for planar layered media," IEEE TAP,
60(4), 1878--1885, 2012, doi:10.1109/TAP.2012.2186253.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import epsilon_0, mu_0, speed_of_light

import gprMax
from gprMax.ntff.equivalent_currents import EquivalentCurrentPhasors
from gprMax.ntff.evaluator import project_cartesian_to_spherical, spherical_directions
from gprMax.ntff.layered import LayeredMedium, evaluate_layered_currents


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "capoglu_figure2"

DL = 1.9e-3
DOMAIN = 0.152
ORIGIN = np.asarray((DOMAIN / 2,) * 3)
TIME_WINDOW = 4.0e-9
PULSE_FREQUENCY = 6.0e9
PULSE_TAU = 0.242e-9
PULSE_DELAY = 6 * PULSE_TAU

# Six finite layers, each ten cells thick, occupy the non-PML z extent.
# The two exterior media continue into the z-directed PMLs.
INTERFACES_RELATIVE = np.arange(3, -4, -1, dtype=float) * 10 * DL
INTERFACES = ORIGIN[2] + INTERFACES_RELATIVE
ER = np.linspace(1.3, 1.5, 8)
MR = np.linspace(1.1, 1.3, 8)
SIGMA = np.concatenate(([0.0], np.linspace(0.1, 0.35, 6), [0.0]))
MATERIAL_IDS = tuple(f"layer{number}" for number in range(8))

SPECTRAL_FREQUENCIES = np.linspace(4.0e9, 8.0e9, 100)
FREQUENCIES = np.unique(np.concatenate((SPECTRAL_FREQUENCIES, [PULSE_FREQUENCY])))

SOURCE_SPECIFICATION = (
    ("x", (19.95, 19.0, -47.5)),
    ("x", (19.95, 19.0, 0.0)),
    ("x", (19.95, 19.0, 47.5)),
    ("y", (0.0, 0.95, -47.5)),
    ("y", (0.0, 0.95, 0.0)),
    ("y", (0.0, 0.95, 47.5)),
    ("z", (-19.0, -19.0, -46.55)),
    ("z", (-19.0, -19.0, 0.95)),
    ("z", (-19.0, -19.0, 48.45)),
)
SOURCE_POSITIONS = np.asarray([ORIGIN + 1e-3 * np.asarray(position) for _, position in SOURCE_SPECIFICATION])
SOURCE_AXES = np.asarray(["xyz".index(axis) for axis, _ in SOURCE_SPECIFICATION])

SPECTRAL_THETA = np.asarray([45.0])
SPECTRAL_PHI = np.asarray([45.0])

# Figure 2(b) uses theta from 0 to 2 pi.  gprMax's public spherical API uses
# canonical theta in [0, pi], so map the second half of the same physical
# vertical plane to phi + pi.
THETA_PLOT = np.linspace(0.0, 360.0, 240)
THETA_CANONICAL = np.where(THETA_PLOT <= 180.0, THETA_PLOT, 360.0 - THETA_PLOT)
THETA_PHI = np.where(THETA_PLOT <= 180.0, 45.0, 225.0)

PHI_PLOT = np.linspace(0.0, 360.0, 120)
PHI_THETA = np.full(PHI_PLOT.shape, 135.0)


def _paper_waveform(time: float) -> float:
    shifted = time - PULSE_DELAY
    return float(np.sin(2 * np.pi * PULSE_FREQUENCY * shifted) * np.exp(-(shifted**2) / (2 * PULSE_TAU**2)))


def build_scene() -> gprMax.Scene:
    """Build the physical and sampling configuration of the paper example."""

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name="Çapoğlu et al. eight-layer NTFF validation"))
    scene.add(gprMax.Domain(p1=(DOMAIN,) * 3))
    scene.add(gprMax.Discretisation(p1=(DL,) * 3))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW))

    for number, (er, mr, sigma) in enumerate(zip(ER, MR, SIGMA)):
        scene.add(
            gprMax.Material(
                er=float(er),
                se=float(sigma),
                mr=float(mr),
                sm=0,
                id=MATERIAL_IDS[number],
            )
        )

    # Paint from the lower exterior upward.  Each later box overwrites the
    # cells above its interface and thereby creates one plane-stratified stack.
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(DOMAIN,) * 3, material_id=MATERIAL_IDS[-1]))
    for material_number in range(6, -1, -1):
        scene.add(
            gprMax.Box(
                p1=(0, 0, float(INTERFACES[material_number])),
                p2=(DOMAIN,) * 3,
                material_id=MATERIAL_IDS[material_number],
            )
        )

    scene.add(
        gprMax.Waveform(
            wave_type="user",
            user_func=_paper_waveform,
            id="paper_pulse",
        )
    )
    for (axis, _), position in zip(SOURCE_SPECIFICATION, SOURCE_POSITIONS):
        scene.add(gprMax.HertzianDipole(p1=tuple(position), polarisation=axis, waveform_id="paper_pulse"))

    # The paper locates its surface three cells inside the ten-cell CPML.
    surface_offset = 13 * DL
    scene.add(
        gprMax.NTFFSurface(
            p1=(surface_offset,) * 3,
            p2=(DOMAIN - surface_offset,) * 3,
            id="surface",
            origin=tuple(ORIGIN),
        )
    )
    scene.add(
        gprMax.NTFFLayeredBackground(
            id="eight_layer",
            axis="z",
            materials=MATERIAL_IDS,
            interfaces=tuple(INTERFACES),
        )
    )
    scene.add(
        gprMax.NTFFLayeredFrequencyTransform(
            surface_id="surface",
            id="spectrum",
            background_id="eight_layer",
            frequencies=FREQUENCIES,
            window="rectangular",
            save_surface_dft=False,
        )
    )
    scene.add(
        gprMax.NTFFFarField(
            theta=SPECTRAL_THETA,
            phi=SPECTRAL_PHI,
            transform_id="spectrum",
            id="spectral",
            outputs=("Etheta", "Ephi"),
        )
    )
    scene.add(
        gprMax.NTFFFarField(
            theta=THETA_CANONICAL,
            phi=THETA_PHI,
            transform_id="spectrum",
            id="theta_cut",
            outputs=("Etheta", "Ephi"),
        )
    )
    scene.add(
        gprMax.NTFFFarField(
            theta=PHI_THETA,
            phi=PHI_PLOT,
            transform_id="spectrum",
            id="phi_cut",
            outputs=("Etheta", "Ephi"),
        )
    )
    return scene


def _source_moments(output: h5py.File) -> np.ndarray:
    """Return exact discrete current moments, including Yee time placement."""

    moments = np.zeros((FREQUENCIES.size, len(SOURCE_SPECIFICATION), 3), dtype=np.complex128)
    for source_number, axis in enumerate(SOURCE_AXES, start=1):
        excitation = output[f"srcs/src{source_number}/excitation"]
        samples = np.asarray(excitation["samples"])
        dt = float(excitation.attrs["SampleInterval"])
        offset = float(excitation.attrs["TimeSampleOffset"])
        time = offset + dt * np.arange(samples.size)
        spectrum = dt * np.sum(
            samples[np.newaxis, :] * np.exp(-1j * 2 * np.pi * FREQUENCIES[:, None] * time),
            axis=1,
        )
        moments[:, source_number - 1, axis] = spectrum * float(excitation.attrs["SpatialScale"])
    return moments


def _point_dipole_reference(output: h5py.File, theta, phi) -> np.ndarray:
    """Evaluate the exact nine point currents in the unperturbed stack."""

    omega = 2 * np.pi * FREQUENCIES[:, None]
    eps = ER[np.newaxis, :] + SIGMA[np.newaxis, :] / (1j * omega * epsilon_0)
    mu = np.broadcast_to(MR, eps.shape).astype(np.complex128)
    medium = LayeredMedium(
        axis="z",
        interfaces=INTERFACES,
        material_ids=MATERIAL_IDS,
        relative_permittivity=eps,
        relative_permeability=mu,
    )
    moments = _source_moments(output)
    currents = EquivalentCurrentPhasors(
        positions=SOURCE_POSITIONS,
        normals=np.zeros_like(SOURCE_POSITIONS),
        area_weights=np.ones(SOURCE_POSITIONS.shape[0]),
        electric_current=moments,
        magnetic_current=np.zeros_like(moments),
    )
    directions = spherical_directions(theta, phi, degrees=True)
    reference = evaluate_layered_currents(
        currents,
        FREQUENCIES,
        directions,
        medium,
        origin=ORIGIN,
        nthreads=1,
    )
    return project_cartesian_to_spherical(reference.electric, theta, phi, degrees=True)[..., 1:]


def _read_fields(output: h5py.File, output_id: str) -> np.ndarray:
    fields = output[f"ntff/surface/frequency/spectrum/far_field/{output_id}/fields"]
    return np.stack((np.asarray(fields["Etheta"]), np.asarray(fields["Ephi"])), axis=-1)


def _normalisation(output: h5py.File) -> np.ndarray:
    moments = _source_moments(output)[:, 0, 0]
    return 1j * 2 * np.pi * FREQUENCIES * mu_0 * moments / (4 * np.pi)


def _rms_by_curve(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    difference = actual - expected
    return np.sqrt(np.mean(np.abs(difference) ** 2, axis=0))


def _paper_curve_rms(actual: np.ndarray, expected: np.ndarray) -> list[float]:
    """RMS errors of the real/imaginary curves, relative to each curve peak."""

    errors = []
    for component in range(2):
        for projection in (np.real, np.imag):
            actual_curve = projection(actual[:, component])
            expected_curve = projection(expected[:, component])
            scale = np.max(np.abs(expected_curve))
            errors.append(float(np.sqrt(np.mean((actual_curve - expected_curve) ** 2)) / scale))
    return errors


def compare(output_path: Path) -> dict[str, float | bool]:
    """Compare all three panels in the paper and write plots and metrics."""

    with h5py.File(output_path, "r") as output:
        actual_spectral = _read_fields(output, "spectral")
        actual_theta = _read_fields(output, "theta_cut")
        actual_phi = _read_fields(output, "phi_cut")
        expected_spectral = _point_dipole_reference(output, SPECTRAL_THETA, SPECTRAL_PHI)
        expected_theta = _point_dipole_reference(output, THETA_CANONICAL, THETA_PHI)
        expected_phi = _point_dipole_reference(output, PHI_THETA, PHI_PLOT)
        normalisation = _normalisation(output)

    spectral_indices = np.searchsorted(FREQUENCIES, SPECTRAL_FREQUENCIES)
    centre = int(np.flatnonzero(np.isclose(FREQUENCIES, PULSE_FREQUENCY, rtol=0, atol=1))[0])

    # Equation (1) of the paper applies 1/(2 pi) to the FDTD field DFT,
    # whereas the closed waveform transform printed in (41), and hence E0 in
    # (43), is the conventional unscaled transform.  Retain that published
    # plotting convention here.  It changes only the common ordinate scale,
    # not the FDTD/theory error.
    paper_field_scale = 1 / (2 * np.pi)
    actual_spectral_n = paper_field_scale * actual_spectral[spectral_indices, 0] / normalisation[spectral_indices, None]
    expected_spectral_n = (
        paper_field_scale * expected_spectral[spectral_indices, 0] / normalisation[spectral_indices, None]
    )
    actual_theta_n = paper_field_scale * actual_theta[centre] / normalisation[centre]
    expected_theta_n = paper_field_scale * expected_theta[centre] / normalisation[centre]
    actual_phi_n = paper_field_scale * actual_phi[centre] / normalisation[centre]
    expected_phi_n = paper_field_scale * expected_phi[centre] / normalisation[centre]

    cases = (
        ("spectral", actual_spectral_n, expected_spectral_n),
        ("theta_cut", actual_theta_n, expected_theta_n),
        ("phi_cut", actual_phi_n, expected_phi_n),
    )
    curve_rms = np.concatenate([_rms_by_curve(actual, expected).ravel() for _, actual, expected in cases])
    paper_curve_rms = np.asarray(
        [value for _, actual, expected in cases for value in _paper_curve_rms(actual, expected)]
    )
    all_actual = np.concatenate([actual.reshape(-1, 2) for _, actual, _ in cases])
    all_expected = np.concatenate([expected.reshape(-1, 2) for _, _, expected in cases])
    peak = float(np.max(np.linalg.norm(all_expected, axis=-1)))
    vector_error = np.linalg.norm(all_actual - all_expected, axis=-1) / peak
    metrics = {
        "maximum_component_rms_normalised_to_paper_E0": float(np.max(curve_rms)),
        "maximum_real_or_imag_curve_rms_relative_to_curve_peak": float(np.max(paper_curve_rms)),
        "maximum_vector_error_normalised_to_peak": float(np.max(vector_error)),
        "rms_vector_error_normalised_to_peak": float(np.sqrt(np.mean(vector_error**2))),
        "paper_reported_maximum_rms": 0.01,
        "published_field_dft_scale": paper_field_scale,
    }
    # The exact definition used for the paper's quoted aggregate RMS is not
    # stated.  Retain the 3% implementation threshold used by the independent
    # point-dipole regression until convergence is measured for this case.
    metrics["passed"] = bool(metrics["maximum_real_or_imag_curve_rms_relative_to_curve_peak"] <= 0.01)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.6))
    abscissae = (
        (2 * np.pi * SPECTRAL_FREQUENCIES / speed_of_light, r"$k_0$ (m$^{-1}$)"),
        (THETA_PLOT / 180.0, r"$\theta/\pi$"),
        (PHI_PLOT / 180.0, r"$\phi/\pi$"),
    )
    plotted_cases = (
        (actual_spectral_n, expected_spectral_n),
        (actual_theta_n, expected_theta_n),
        (actual_phi_n, expected_phi_n),
    )
    component_labels = (r"$E_\theta/E_0$", r"$E_\phi/E_0$")
    for column, ((xvalues, xlabel), (actual, expected)) in enumerate(zip(abscissae, plotted_cases)):
        for component in range(2):
            axis = axes[component, column]
            axis.plot(xvalues, actual[:, component].real, "k-", label="real (FDTD)")
            axis.plot(xvalues, actual[:, component].imag, "k--", label="imag. (FDTD)")
            axis.plot(
                xvalues,
                expected[:, component].real,
                "k.",
                markersize=3,
                label="real (point-current theory)",
            )
            axis.plot(
                xvalues,
                expected[:, component].imag,
                linestyle="none",
                marker="x",
                color="k",
                markersize=3,
                label="imag. (point-current theory)",
            )
            axis.set_ylabel(component_labels[component])
            axis.grid(True, alpha=0.2)
            if component == 1:
                axis.set_xlabel(xlabel)
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[0, 0].set_title(r"$\theta=\phi=45^\circ$")
    axes[0, 1].set_title(r"$\phi=45^\circ$, 6 GHz")
    axes[0, 2].set_title(r"$\theta=135^\circ$, 6 GHz")
    fig.suptitle("Çapoğlu et al. Figure 2: nine dipoles in an eight-layer medium")
    fig.tight_layout()
    fig.savefig(output_path.with_name("capoglu_figure2.png"), dpi=220)
    plt.close(fig)

    output_path.with_name("capoglu_figure2_summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT.with_suffix(".h5"))
    parser.add_argument("--gpu", type=int, help="CUDA device index; omit for double-precision CPU")
    parser.add_argument("--no-run", action="store_true", help="analyse an existing output")
    args = parser.parse_args()
    if not args.no_run:
        options = {"cpu_precision": "double"}
        if args.gpu is not None:
            options = {"gpu": [args.gpu], "gpu_precision": "double"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        gprMax.run(
            scenes=[build_scene()],
            n=1,
            outputfile=args.output.with_suffix(""),
            hide_progress_bars=True,
            **options,
        )
    metrics = compare(args.output)
    print(json.dumps(metrics, indent=2))
    if not metrics["passed"]:
        raise SystemExit("Çapoğlu et al. multilayer validation failed")


if __name__ == "__main__":
    main()
