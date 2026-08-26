"""Validate layered NTFF dipole patterns above bare and coated PEC planes.

The bare-PEC cases are exact image-theory tests for tangential and normal
electric and magnetic Hertzian dipoles.  The coated-PEC cases use an
independent plane-wave spectrum far-zone calculation: each spectral TE/TM
component sees the input impedance of a short-circuited dielectric slab.
No production gprMax layered-medium propagation helper is used by the
analytical reference.

The comparison covers both principal-plane field components and the
full-hemisphere maximum directivity.  Fields are normalised once per source
and frequency, rather than independently normalising each cut, so the
relative E- and H-plane levels remain part of the test.

References
----------
J. A. Stratton, Electromagnetic Theory, McGraw-Hill, 1941, sections 8.5--8.7.

J. Tang and W. Hong, "The electromagnetic field produced by a horizontal
electric dipole over a dielectric coated perfect conductor," Progress In
Electromagnetics Research, vol. 36, pp. 139--152, 2002,
doi:10.2528/PIER02011801.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.constants import c

import gprMax

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "grounded_dipoles"

DL = 1.5e-3
DOMAIN = (0.120, 0.120, 0.108)
CENTRE = (0.060, 0.060)
GROUND = 0.018
COATING_TOP = 0.024
SOURCE_ANCHOR = (CENTRE[0], CENTRE[1], 0.0465)
SURFACE_P1 = (0.024, 0.024, GROUND)
SURFACE_P2 = (0.096, 0.096, 0.084)
ORIGIN = (CENTRE[0], CENTRE[1], GROUND)
ER_COATING = 4.0
FREQUENCIES = np.asarray((1.5e9, 2.0e9, 2.5e9))
SOURCE_FREQUENCY = 2.0e9
TIME_WINDOW = 4.0e-9
THETA = np.arange(0.0, 90.0, 1.0)
ACCEPTANCE_LIMITS = {
    "vector_field_maximum_error_peak_normalised": 0.04,
    "power_maximum_error_peak_normalised": 0.055,
    "maximum_directivity_relative_error": 0.04,
}


@dataclass(frozen=True)
class Case:
    name: str
    source_kind: str
    polarisation: str
    coated: bool


CASES = (
    Case("electric_tangential_bare", "electric", "x", False),
    Case("electric_normal_bare", "electric", "z", False),
    Case("magnetic_tangential_bare", "magnetic", "x", False),
    Case("magnetic_normal_bare", "magnetic", "z", False),
    Case("electric_tangential_coated", "electric", "x", True),
    Case("electric_normal_coated", "electric", "z", True),
)


def _physical_source_position(case: Case) -> np.ndarray:
    """Return the physical centre of the staggered source component."""

    position = np.asarray(SOURCE_ANCHOR, dtype=float).copy()
    position["xyz".index(case.polarisation)] += 0.5 * DL
    return position


def build_scene(case: Case) -> gprMax.Scene:
    """Build one grounded dipole model."""

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name=case.name.replace("_", " ")))
    scene.add(gprMax.Domain(p1=DOMAIN))
    scene.add(gprMax.Discretisation(p1=(DL,) * 3))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW))
    if case.coated:
        scene.add(gprMax.Material(er=ER_COATING, se=0, mr=1, sm=0, id="coating"))
        scene.add(
            gprMax.Box(
                p1=(0, 0, GROUND),
                p2=(DOMAIN[0], DOMAIN[1], COATING_TOP),
                material_id="coating",
            )
        )
    scene.add(
        gprMax.Box(
            p1=(0, 0, 0),
            p2=(DOMAIN[0], DOMAIN[1], GROUND),
            material_id="pec",
        )
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=SOURCE_FREQUENCY, id="pulse"))
    source_class = gprMax.HertzianDipole if case.source_kind == "electric" else gprMax.MagneticDipole
    scene.add(
        source_class(
            p1=SOURCE_ANCHOR,
            polarisation=case.polarisation,
            waveform_id="pulse",
        )
    )
    scene.add(
        gprMax.NTFFSurface(
            p1=SURFACE_P1,
            p2=SURFACE_P2,
            id="surface",
            origin=ORIGIN,
            omit_faces=("z0",),
        )
    )
    if case.coated:
        materials = ("free_space", "coating", "pec")
        interfaces = (COATING_TOP, GROUND)
    else:
        materials = ("free_space", "pec")
        interfaces = (GROUND,)
    scene.add(
        gprMax.NTFFLayeredBackground(
            id="grounded",
            axis="z",
            materials=materials,
            interfaces=interfaces,
        )
    )
    scene.add(
        gprMax.NTFFLayeredFrequencyTransform(
            surface_id="surface",
            id="spectrum",
            background_id="grounded",
            frequencies=FREQUENCIES,
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
                outputs=("Etheta", "Ephi", "directivity"),
            )
        )
    return scene


def _grounded_reflection(
    theta: np.ndarray,
    frequency: float,
    coated: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return independent tangential-E TM and TE reflection coefficients.

    The time convention is exp(+j*omega*t).  For the coated case the slab is
    a lossless transmission line short-circuited by the PEC, hence

        Z_in = +j Z_1 tan(k_z1 d),  R = (Z_in - Z_0)/(Z_in + Z_0).
    """

    theta = np.asarray(theta, dtype=float)
    if not coated:
        minus_one = -np.ones(theta.shape, dtype=np.complex128)
        return minus_one, minus_one.copy()

    sine_squared = np.sin(theta) ** 2
    cosine = np.cos(theta)
    q = np.sqrt(ER_COATING - sine_squared + 0j)
    # Normalised impedances for tangential E (common eta_0 factor omitted).
    z0_tm = cosine
    z0_te = 1 / cosine
    z1_tm = q / ER_COATING
    z1_te = 1 / q
    phase = (2 * np.pi * frequency / c) * q * (COATING_TOP - GROUND)
    zin_tm = 1j * z1_tm * np.tan(phase)
    zin_te = 1j * z1_te * np.tan(phase)
    return (zin_tm - z0_tm) / (zin_tm + z0_tm), (zin_te - z0_te) / (zin_te + z0_te)


def analytical_fields(
    case: Case,
    theta: np.ndarray,
    phi: np.ndarray,
    frequency: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return range-normalised far fields up to one common source factor."""

    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    theta, phi = np.broadcast_arrays(theta, phi)
    sine, cosine = np.sin(theta), np.cos(theta)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)

    upward = np.stack((sine * cos_phi, sine * sin_phi, cosine), axis=-1)
    downward = np.stack((sine * cos_phi, sine * sin_phi, -cosine), axis=-1)
    e_theta_up = np.stack((cosine * cos_phi, cosine * sin_phi, -sine), axis=-1)
    # This incident TM basis has the same tangential direction as e_theta_up.
    e_theta_down = np.stack((cosine * cos_phi, cosine * sin_phi, sine), axis=-1)
    e_phi = np.stack((-sin_phi, cos_phi, np.zeros_like(phi)), axis=-1)

    moment = np.zeros(3)
    moment["xyz".index(case.polarisation)] = 1.0
    if case.source_kind == "electric":
        direct = moment - upward * np.sum(upward * moment, axis=-1)[..., None]
        incident = moment - downward * np.sum(downward * moment, axis=-1)[..., None]
    else:
        direct = np.cross(upward, moment)
        incident = np.cross(downward, moment)

    incident_tm = np.sum(incident * e_theta_down, axis=-1)
    incident_te = np.sum(incident * e_phi, axis=-1)
    reflection_tm, reflection_te = _grounded_reflection(theta, frequency, case.coated)
    reflected = (
        reflection_tm[..., None] * incident_tm[..., None] * e_theta_up
        + reflection_te[..., None] * incident_te[..., None] * e_phi
    )

    source_position = _physical_source_position(case)
    reflection_plane = COATING_TOP if case.coated else GROUND
    transverse_phase = (source_position[0] - ORIGIN[0]) * sine * cos_phi + (
        source_position[1] - ORIGIN[1]
    ) * sine * sin_phi
    wavenumber = 2 * np.pi * frequency / c
    direct_phase = wavenumber * (transverse_phase + (source_position[2] - ORIGIN[2]) * cosine)
    reflected_phase = wavenumber * (transverse_phase + (2 * reflection_plane - source_position[2] - ORIGIN[2]) * cosine)
    total = direct * np.exp(1j * direct_phase)[..., None] + reflected * np.exp(1j * reflected_phase)[..., None]
    return np.sum(total * e_theta_up, axis=-1), np.sum(total * e_phi, axis=-1)


def _analytical_directivity(case: Case, frequency: float) -> float:
    """Integrate exact upper-hemisphere intensity independently."""

    cosine, weights = leggauss(180)
    cosine = 0.5 * (cosine + 1)
    weights = 0.5 * weights
    theta = np.arccos(cosine)
    phi = 2 * np.pi * np.arange(360) / 360
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    etheta, ephi = analytical_fields(case, theta_grid, phi_grid, frequency)
    intensity = np.abs(etheta) ** 2 + np.abs(ephi) ** 2
    radiated = np.sum(intensity * weights[:, None]) * (2 * np.pi / phi.size)
    maximum = float(np.max(intensity))
    return 4 * np.pi * maximum / float(radiated)


def _read_case(path: Path):
    fields = {}
    maximum_directivity = None
    with h5py.File(path, "r") as output:
        base = output["ntff/surface/frequency/spectrum/far_field"]
        for output_id in ("e_plane", "h_plane"):
            group = base[output_id]
            fields[output_id] = {
                "Etheta": np.asarray(group["fields/Etheta"]),
                "Ephi": np.asarray(group["fields/Ephi"]),
                "directivity": np.asarray(group["fields/directivity"]),
            }
            candidate = np.asarray(group["maximum_directivity"])
            if maximum_directivity is None:
                maximum_directivity = candidate
            else:
                # The retained maximum includes the internal quadrature plus
                # the requested directions in this output group.  Take the
                # union of the two principal-plane refinements.
                maximum_directivity = np.maximum(maximum_directivity, candidate)
    return fields, maximum_directivity


def compare_case(case: Case, path: Path):
    fields, fdtd_dmax = _read_case(path)
    retained = {}
    metrics = {"frequencies": {}}
    theta = np.deg2rad(THETA)
    for frequency_index, frequency in enumerate(FREQUENCIES):
        fdtd_parts = []
        exact_parts = []
        for output_id, phi_degrees in (("e_plane", 0.0), ("h_plane", 90.0)):
            actual_theta = fields[output_id]["Etheta"][frequency_index]
            actual_phi = fields[output_id]["Ephi"][frequency_index]
            exact_theta, exact_phi = analytical_fields(
                case,
                theta,
                np.full_like(theta, np.deg2rad(phi_degrees)),
                float(frequency),
            )
            fdtd_parts.append(np.stack((actual_theta, actual_phi), axis=-1))
            exact_parts.append(np.stack((exact_theta, exact_phi), axis=-1))
        actual = np.concatenate(fdtd_parts, axis=0)
        exact = np.concatenate(exact_parts, axis=0)
        # One complex least-squares scale removes only the arbitrary source
        # spectrum and range-normalisation constant, not cut-specific levels.
        scale = np.vdot(exact, actual) / np.vdot(exact, exact)
        fitted = scale * exact
        peak = float(np.max(np.linalg.norm(fitted, axis=-1)))
        difference = actual - fitted
        vector_error = np.linalg.norm(difference, axis=-1) / peak
        actual_power = np.sum(np.abs(actual) ** 2, axis=-1)
        exact_power = np.sum(np.abs(fitted) ** 2, axis=-1)
        power_scale = max(float(np.max(exact_power)), np.finfo(float).tiny)
        power_error = (actual_power - exact_power) / power_scale
        exact_dmax = _analytical_directivity(case, float(frequency))
        key = f"{frequency / 1e9:g}_GHz"
        metrics["frequencies"][key] = {
            "vector_field_rms_error_peak_normalised": float(np.sqrt(np.mean(vector_error**2))),
            "vector_field_maximum_error_peak_normalised": float(np.max(vector_error)),
            "power_rms_error_peak_normalised": float(np.sqrt(np.mean(power_error**2))),
            "power_maximum_error_peak_normalised": float(np.max(np.abs(power_error))),
            "fdtd_maximum_directivity": float(fdtd_dmax[frequency_index]),
            "analytical_maximum_directivity": exact_dmax,
            "maximum_directivity_relative_error": float(abs(fdtd_dmax[frequency_index] - exact_dmax) / exact_dmax),
        }
        retained[key] = {
            "actual": actual,
            "exact": fitted,
            "actual_power": actual_power,
            "exact_power": exact_power,
        }
    return metrics, retained


def _write_csv(case: Case, retained, output_directory: Path) -> None:
    for frequency_key, values in retained.items():
        path = output_directory / f"{case.name}_{frequency_key}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("plane", "theta_degrees", "fdtd_power", "analytical_power"))
            for index, plane in enumerate(("e_plane", "h_plane")):
                start = index * THETA.size
                for offset, angle in enumerate(THETA):
                    item = start + offset
                    writer.writerow((plane, angle, values["actual_power"][item], values["exact_power"][item]))


def _power_db(values: np.ndarray) -> np.ndarray:
    normalised = values / max(float(np.max(values)), np.finfo(float).tiny)
    return np.maximum(10 * np.log10(np.maximum(normalised, np.finfo(float).tiny)), -40)


def _plot(results, output_directory: Path) -> None:
    selected_frequency = "2_GHz"
    fig, axes = plt.subplots(
        len(CASES),
        2,
        figsize=(9.5, 3.5 * len(CASES)),
        subplot_kw={"projection": "polar"},
    )
    for row, case in enumerate(CASES):
        values = results[case.name][selected_frequency]
        for column, (plane, title) in enumerate((("e_plane", "E plane"), ("h_plane", "H plane"))):
            axis = axes[row, column]
            start = column * THETA.size
            stop = start + THETA.size
            analytical = _power_db(values["exact_power"])
            fdtd = _power_db(values["actual_power"])
            axis.plot(np.deg2rad(THETA), analytical[start:stop], "k-", lw=1.4, label="analytical")
            axis.plot(
                np.deg2rad(THETA),
                fdtd[start:stop],
                "ko",
                markerfacecolor="white",
                markersize=2.6,
                markevery=3,
                label="gprMax FDTD",
            )
            axis.set_theta_zero_location("N")
            axis.set_theta_direction(-1)
            axis.set_thetamin(0)
            axis.set_thetamax(90)
            axis.set_rlim(-40, 0)
            axis.set_rticks((-40, -30, -20, -10, 0))
            axis.grid(True, alpha=0.3)
            axis.set_title(f"{case.name.replace('_', ' ')}, {title}", pad=12)
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(-0.2, 1.16), fontsize=8)
    fig.suptitle("Dipoles above PEC-terminated planar backgrounds, 2 GHz", y=0.998)
    fig.tight_layout()
    fig.savefig(output_directory / "grounded_dipole_patterns.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(CASES))
    width = 0.23
    styles = (("white", ""), ("0.7", "///"), ("0.2", "xx"))
    for index, frequency in enumerate(FREQUENCIES):
        key = f"{frequency / 1e9:g}_GHz"
        errors = [
            100 * results[case.name + "_metrics"]["frequencies"][key]["maximum_directivity_relative_error"]
            for case in CASES
        ]
        facecolor, hatch = styles[index]
        axis.bar(
            x + (index - 1) * width,
            errors,
            width,
            facecolor=facecolor,
            edgecolor="black",
            hatch=hatch,
            label=f"{frequency / 1e9:g} GHz",
        )
    axis.set_xticks(x, [case.name.replace("_", "\n") for case in CASES])
    axis.set_ylabel("maximum-directivity relative error [%]")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_directory / "grounded_dipole_directivity_error.png", dpi=220)
    plt.close(fig)


def _acceptance(summary) -> dict:
    checks = {}
    for case_name, case_metrics in summary.items():
        for frequency, metrics in case_metrics["frequencies"].items():
            for metric, maximum in ACCEPTANCE_LIMITS.items():
                key = f"{case_name}:{frequency}:{metric}"
                checks[key] = {
                    "value": metrics[metric],
                    "maximum": maximum,
                    "passed": metrics[metric] <= maximum,
                }
    return {"passed": all(item["passed"] for item in checks.values()), "checks": checks}


def main() -> None:
    global DL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", type=int, help="CUDA device index; omit for double-precision CPU")
    parser.add_argument("--no-run", action="store_true", help="analyse existing HDF5 outputs")
    parser.add_argument("--case", choices=[case.name for case in CASES], action="append")
    parser.add_argument("--dl", type=float, default=DL, help="isotropic spatial step in metres")
    args = parser.parse_args()
    if not np.isfinite(args.dl) or args.dl <= 0:
        parser.error("--dl must be finite and positive")
    DL = args.dl
    args.output_directory.mkdir(parents=True, exist_ok=True)
    selected = tuple(case for case in CASES if args.case is None or case.name in args.case)

    summary = {}
    plot_results = {}
    for case in selected:
        output_path = args.output_directory / f"{case.name}.h5"
        if not args.no_run:
            options = {"cpu_precision": "double"}
            if args.gpu is not None:
                options = {"gpu": [args.gpu], "gpu_precision": "double"}
            gprMax.run(
                scenes=[build_scene(case)],
                n=1,
                outputfile=output_path.with_suffix(""),
                hide_progress_bars=True,
                **options,
            )
        metrics, retained = compare_case(case, output_path)
        summary[case.name] = metrics
        plot_results[case.name] = retained
        plot_results[case.name + "_metrics"] = metrics
        _write_csv(case, retained, args.output_directory)

    if len(selected) == len(CASES):
        _plot(plot_results, args.output_directory)
    result = {
        "model": {"dl_metres": DL},
        "cases": summary,
        "acceptance": _acceptance(summary),
    }
    (args.output_directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["acceptance"]["passed"]:
        raise SystemExit("grounded-dipole analytical validation failed")


if __name__ == "__main__":
    main()
