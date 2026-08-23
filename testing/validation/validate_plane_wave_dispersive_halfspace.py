"""Validate axial DPW reflection from dielectric and dispersive half-spaces.

The free-space run supplies the incident field at a receiver inside the TFSF
region. For each material half-space, the reflected field is obtained from

    E_reflected(t) = E_total(t) - E_incident(t).

The FFT ratio is de-embedded from the receiver to the material interface and
compared with the normal-incidence Fresnel coefficient. The Fourier transform
uses the engineering convention exp(-j*omega*t), as implemented by
``numpy.fft.rfft``.

Example::

    python -m testing.validation.validate_plane_wave_dispersive_halfspace
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from time import perf_counter

import h5py
import matplotlib
import numpy as np
from scipy.constants import c, epsilon_0, mu_0

import gprMax

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

DL = 0.5e-3
DOMAIN = (0.16, 0.02, float("inf"))
TIME_WINDOW = 40e-9
PML_CELLS = 12
TFSF_P1 = (0.0125, 0.007, float("inf"))
TFSF_P2 = (0.1475, 0.013, float("inf"))
INTERFACE_X = 0.09
RECEIVER = (0.0725, 0.010, float("inf"))
SOURCE_FREQUENCY = 3e9
FREQUENCY_MIN = 0.25e9
FREQUENCY_MAX = 8.0e9
INCIDENT_SPECTRUM_THRESHOLD = 0.001

ACCEPTANCE_LIMITS = {
    "magnitude_rmse": 0.005,
    "complex_relative_l2_error": 0.01,
    "phase_rmse_degrees": 0.1,
}


CASES = {
    "dielectric": {
        "label": r"Dielectric, $\epsilon_r=4$ (smoothed)",
        "kind": "dielectric",
        "er_inf": 4.0,
        "averaging": "y",
        "interface_offset_cells": 0.0,
    },
    "dielectric_unsmoothed": {
        "label": r"Dielectric, $\epsilon_r=4$ (unsmoothed)",
        "kind": "dielectric",
        "er_inf": 4.0,
        "averaging": "n",
        "interface_offset_cells": -0.5,
    },
    "debye_1pole": {
        "label": "Debye, 1 pole",
        "kind": "debye",
        "er_inf": 2.5,
        "delta_er": (4.5,),
        "tau": (80e-12,),
        "interface_offset_cells": -0.5,
    },
    "debye_3pole": {
        "label": "Debye, 3 poles",
        "kind": "debye",
        "er_inf": 2.2,
        "delta_er": (3.0, 2.0, 1.0),
        "tau": (20e-12, 100e-12, 500e-12),
        "interface_offset_cells": -0.5,
    },
    "lorentz_2pole": {
        "label": "Lorentz, 2 poles",
        "kind": "lorentz",
        "er_inf": 3.0,
        "delta_er": (1.5, 0.8),
        "pole_frequency": (1.2e9, 3.2e9),
        "damping": (0.4e9, 0.8e9),
        "interface_offset_cells": -0.5,
    },
    "drude_2pole": {
        "label": "Drude, 2 poles",
        "kind": "drude",
        "er_inf": 9.0,
        "pole_frequency": (0.5e9, 1.0e9),
        "damping": (0.6e9, 1.0e9),
        "interface_offset_cells": -0.5,
    },
}


def _add_material(scene, case_name):
    """Add one half-space material and its requested pole model."""

    params = CASES[case_name]
    material_id = f"halfspace_{case_name}"
    scene.add(gprMax.Material(er=params["er_inf"], se=0, mr=1, sm=0, id=material_id))

    if params["kind"] == "debye":
        scene.add(
            gprMax.AddDebyeDispersion(
                poles=len(params["delta_er"]),
                er_delta=params["delta_er"],
                tau=params["tau"],
                material_ids=[material_id],
            )
        )
    elif params["kind"] == "lorentz":
        scene.add(
            gprMax.AddLorentzDispersion(
                poles=len(params["delta_er"]),
                er_delta=params["delta_er"],
                omega=params["pole_frequency"],
                delta=params["damping"],
                material_ids=[material_id],
            )
        )
    elif params["kind"] == "drude":
        scene.add(
            gprMax.AddDrudeDispersion(
                poles=len(params["pole_frequency"]),
                omega=params["pole_frequency"],
                alpha=params["damping"],
                material_ids=[material_id],
            )
        )

    box_kwargs = {}
    if "averaging" in params:
        box_kwargs["averaging"] = params["averaging"]
    scene.add(
        gprMax.Box(
            p1=(INTERFACE_X, 0, float("inf")),
            p2=DOMAIN,
            material_id=material_id,
            **box_kwargs,
        )
    )


def build_scene(case_name, threads):
    """Build the reference or material half-space scene."""

    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    scene.add(gprMax.Domain(p1=DOMAIN))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW))
    scene.add(gprMax.OMPThreads(n=threads))
    # There is no PML on the invariant z axis of this 2D TM model.
    scene.add(gprMax.PMLThickness(thickness=(PML_CELLS, PML_CELLS, 0, PML_CELLS, PML_CELLS, 0)))

    if case_name != "free_space":
        _add_material(scene, case_name)

    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=SOURCE_FREQUENCY, id="plane_pulse"))
    scene.add(
        gprMax.DiscretePlaneWaveAxial(
            p1=TFSF_P1,
            p2=TFSF_P2,
            axis="x",
            psi=90,
            waveform_id="plane_pulse",
        )
    )
    scene.add(gprMax.Rx(p1=RECEIVER, id="reflection_probe", outputs=["Ez"]))
    return scene


def run_case(case_name, cache_dir, threads, precision, reuse, gpu):
    """Run one model and return its receiver trace and time step."""

    outputfile = cache_dir / case_name
    h5_path = outputfile.with_suffix(".h5")
    runtime = 0.0
    if not (reuse and h5_path.exists()):
        start = perf_counter()
        solver_options = (
            {"cpu_precision": precision}
            if gpu is None
            else {"gpu": [gpu], "gpu_precision": precision}
        )
        gprMax.run(
            scenes=[build_scene(case_name, threads)],
            n=1,
            outputfile=outputfile,
            hide_progress_bars=True,
            log_level=logging.WARNING,
            **solver_options,
        )
        runtime = perf_counter() - start

    with h5py.File(h5_path, "r") as output:
        trace = np.asarray(output["rxs/rx1/Ez"])
        dt = float(output.attrs["dt"])
    return trace, dt, runtime


def complex_permittivity(case_name, frequencies):
    """Return the continuous-time analytical relative permittivity.

    Pole frequencies supplied to gprMax are in hertz and are converted to
    angular frequencies here, consistently with ``calculate_update_coeffsE``.
    Damping parameters are rates in inverse seconds.
    """

    params = CASES[case_name]
    omega = 2 * np.pi * np.asarray(frequencies)
    er = np.full(omega.shape, params["er_inf"], dtype=np.complex128)

    if params["kind"] == "debye":
        for delta_er, tau in zip(params["delta_er"], params["tau"]):
            er += delta_er / (1 + 1j * omega * tau)
    elif params["kind"] == "lorentz":
        for delta_er, frequency, damping in zip(
            params["delta_er"], params["pole_frequency"], params["damping"]
        ):
            omega_p = 2 * np.pi * frequency
            er += delta_er * omega_p**2 / (omega_p**2 + 2j * omega * damping - omega**2)
    elif params["kind"] == "drude":
        for frequency, damping in zip(params["pole_frequency"], params["damping"]):
            omega_p = 2 * np.pi * frequency
            er -= omega_p**2 / (omega**2 - 1j * omega * damping)

    return er


def fresnel_reflection(er):
    """Electric-field reflection coefficient from free space at normal incidence."""

    impedance_1 = np.sqrt(mu_0 / epsilon_0)
    impedance_2 = impedance_1 / np.sqrt(er)
    return (impedance_2 - impedance_1) / (impedance_2 + impedance_1)


def free_space_numerical_wavenumber(frequencies, dt):
    """Return the axial Yee-grid wavenumber in the source-side free space."""

    courant = c * dt / DL
    argument = np.sin(np.pi * np.asarray(frequencies) * dt) / courant
    if np.any(np.abs(argument) > 1 + 1e-12):
        raise ValueError("Selected frequencies are outside the propagating Yee band")
    return (2 / DL) * np.arcsin(np.clip(argument, -1, 1))


def analyse_case(case_name, incident, total, dt):
    """Calculate simulated and analytical interface reflection coefficients."""

    reflected = total - incident
    frequencies = np.fft.rfftfreq(incident.size, dt)
    incident_fft = np.fft.rfft(incident)
    reflected_fft = np.fft.rfft(reflected)

    band = (frequencies >= FREQUENCY_MIN) & (frequencies <= FREQUENCY_MAX)
    spectrum_ok = np.abs(incident_fft) >= INCIDENT_SPECTRUM_THRESHOLD * np.max(np.abs(incident_fft))
    selected = band & spectrum_ok
    frequencies = frequencies[selected]

    gamma_receiver = reflected_fft[selected] / incident_fft[selected]
    # A rigid (unsmoothed) tangential E assignment begins at index xs, while
    # the corresponding H transition lies half a Yee cell earlier. Its
    # discrete reflection plane is therefore xs - dx/2. Dielectric smoothing
    # centres the transition at the requested geometric interface. Dispersive
    # materials are deliberately non-averagable and use the former case.
    effective_interface_x = INTERFACE_X + CASES[case_name]["interface_offset_cells"] * DL
    distance = effective_interface_x - RECEIVER[0]
    numerical_wavenumber = free_space_numerical_wavenumber(frequencies, dt)
    gamma_fdtd = gamma_receiver * np.exp(1j * 2 * numerical_wavenumber * distance)
    er = complex_permittivity(case_name, frequencies)
    gamma_analytic = fresnel_reflection(er)

    magnitude_error = np.abs(gamma_fdtd) - np.abs(gamma_analytic)
    phase_error = np.angle(gamma_fdtd / gamma_analytic, deg=True)
    return {
        "frequencies": frequencies,
        "incident_fft": incident_fft[selected],
        "reflected": reflected,
        "gamma_receiver": gamma_receiver,
        "gamma_fdtd": gamma_fdtd,
        "gamma_analytic": gamma_analytic,
        "er": er,
        "effective_interface_x": effective_interface_x,
        "receiver_to_interface_distance": distance,
        "magnitude_rmse": float(np.sqrt(np.mean(magnitude_error**2))),
        "magnitude_max_error": float(np.max(np.abs(magnitude_error))),
        "complex_relative_l2_error": float(
            np.linalg.norm(gamma_fdtd - gamma_analytic) / np.linalg.norm(gamma_analytic)
        ),
        "phase_rmse_degrees": float(np.sqrt(np.mean(phase_error**2))),
    }


def _save_csv(path, result):
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            (
                "frequency_hz",
                "fdtd_magnitude",
                "fdtd_phase_degrees",
                "analytic_magnitude",
                "analytic_phase_degrees",
                "epsilon_real",
                "epsilon_imaginary",
            )
        )
        for index, frequency in enumerate(result["frequencies"]):
            writer.writerow(
                (
                    frequency,
                    abs(result["gamma_fdtd"][index]),
                    np.angle(result["gamma_fdtd"][index], deg=True),
                    abs(result["gamma_analytic"][index]),
                    np.angle(result["gamma_analytic"][index], deg=True),
                    result["er"][index].real,
                    result["er"][index].imag,
                )
            )


def _plot_comparison(results, output_dir):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    axes = axes.ravel()
    for axis, (case_name, result) in zip(axes, results.items()):
        frequency_ghz = result["frequencies"] / 1e9
        axis.plot(
            frequency_ghz,
            np.abs(result["gamma_analytic"]),
            "k-",
            linewidth=2,
            label="Analytical",
        )
        axis.plot(
            frequency_ghz,
            np.abs(result["gamma_fdtd"]),
            color="tab:red",
            linestyle="--",
            linewidth=1.5,
            label="gprMax FDTD",
        )
        axis.set_title(CASES[case_name]["label"])
        axis.set_ylabel(r"$|\Gamma|$")
        axis.grid(True, alpha=0.3)
        axis.set_ylim(bottom=0)
    for axis in axes[len(results) :]:
        axis.set_visible(False)
    for axis in axes[-2:]:
        if axis.get_visible():
            axis.set_xlabel("Frequency [GHz]")
    axes[0].legend(loc="best")
    fig.suptitle("Normal-incidence half-space reflection coefficient")
    fig.tight_layout()
    fig.savefig(output_dir / "reflection_magnitude.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    axes = axes.ravel()
    for axis, (case_name, result) in zip(axes, results.items()):
        frequency_ghz = result["frequencies"] / 1e9
        analytic_phase = np.unwrap(np.angle(result["gamma_analytic"])) * 180 / np.pi
        fdtd_phase = np.unwrap(np.angle(result["gamma_fdtd"])) * 180 / np.pi
        offset = 360 * np.round(np.median((analytic_phase - fdtd_phase) / 360))
        axis.plot(frequency_ghz, analytic_phase, "k-", linewidth=2, label="Analytical")
        axis.plot(
            frequency_ghz,
            fdtd_phase + offset,
            color="tab:blue",
            linestyle="--",
            linewidth=1.5,
            label="gprMax FDTD",
        )
        axis.set_title(CASES[case_name]["label"])
        axis.set_ylabel(r"Phase of $\Gamma$ [deg]")
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        # Avoid visually magnifying round-off/windowing ripple for cases whose
        # analytical phase is constant. The dielectric residual is only about
        # 1e-5 degrees, but Matplotlib's automatic limits otherwise make it
        # occupy the full height of the panel and look physically significant.
        phase_values = np.concatenate((analytic_phase, fdtd_phase + offset))
        if np.ptp(phase_values) < 0.2:
            phase_centre = np.mean(phase_values)
            axis.set_ylim(phase_centre - 0.1, phase_centre + 0.1)
        axis.grid(True, alpha=0.3)
    for axis in axes[len(results) :]:
        axis.set_visible(False)
    for axis in axes[-2:]:
        if axis.get_visible():
            axis.set_xlabel("Frequency [GHz]")
    axes[0].legend(loc="best")
    fig.suptitle("De-embedded normal-incidence reflection phase")
    fig.tight_layout()
    fig.savefig(output_dir / "reflection_phase.png", dpi=180)
    plt.close(fig)


def _plot_permittivity(results, output_dir):
    fig, (real_axis, imaginary_axis) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for case_name, result in results.items():
        frequency_ghz = result["frequencies"] / 1e9
        real_axis.plot(frequency_ghz, result["er"].real, label=CASES[case_name]["label"])
        imaginary_axis.plot(frequency_ghz, result["er"].imag)
    real_axis.set_ylabel(r"$\Re\{\epsilon_r\}$")
    imaginary_axis.set_ylabel(r"$\Im\{\epsilon_r\}$")
    imaginary_axis.set_xlabel("Frequency [GHz]")
    real_axis.legend(loc="best", fontsize="small")
    for axis in (real_axis, imaginary_axis):
        axis.grid(True, alpha=0.3)
    fig.suptitle("Analytical material permittivity")
    fig.tight_layout()
    fig.savefig(output_dir / "material_permittivity.png", dpi=180)
    plt.close(fig)


def _plot_time_traces(incident, totals, results, dt, output_dir):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    axes = axes.ravel()
    time_ns = np.arange(incident.size) * dt * 1e9
    time_limit = min(8.0, time_ns[-1])
    time_mask = time_ns <= time_limit
    for axis, (case_name, total) in zip(axes, totals.items()):
        axis.plot(time_ns[time_mask], incident[time_mask], "k-", label="Incident")
        axis.plot(time_ns[time_mask], total[time_mask], color="tab:blue", label="Total")
        axis.plot(
            time_ns[time_mask],
            results[case_name]["reflected"][time_mask],
            color="tab:red",
            label="Total - incident",
        )
        axis.set_title(CASES[case_name]["label"])
        axis.set_ylabel(r"$E_z$ [V/m]")
        axis.grid(True, alpha=0.3)
    for axis in axes[len(totals) :]:
        axis.set_visible(False)
    for axis in axes[-2:]:
        if axis.get_visible():
            axis.set_xlabel("Time [ns]")
    axes[0].legend(loc="best")
    fig.suptitle("Receiver fields inside the TFSF region")
    fig.tight_layout()
    fig.savefig(output_dir / "time_domain_fields.png", dpi=180)
    plt.close(fig)


def _write_report(output_dir, summary):
    """Write a compact human-readable report alongside the raw results."""

    lines = [
        "# Axial plane-wave dispersive half-space validation",
        "",
        "A free-space receiver trace was used as the incident field. For each ",
        "half-space, the reflected field was calculated as `total - incident`, ",
        "transformed with the engineering FFT convention, and de-embedded to ",
        "the appropriate discrete Yee reflection plane. Axial free-space ",
        "propagation was removed using the Yee numerical wavenumber rather ",
        "than the continuous-space wavenumber.",
        "",
        "The analytical comparison is the normal-incidence electric-field Fresnel coefficient",
        "",
        "$$\\Gamma = \\frac{\\eta_2-\\eta_0}{\\eta_2+\\eta_0}, \\qquad "
        "\\eta_2=\\eta_0/\\sqrt{\\epsilon_r(\\omega)}.$$",
        "",
        "With the $\\exp(-j\\omega t)$ FFT convention, the material models are",
        "",
        "$$\\epsilon_{r,\\mathrm{Debye}}=\\epsilon_{r,\\infty}+"
        "\\sum_p\\frac{\\Delta\\epsilon_{r,p}}{1+j\\omega\\tau_p},$$",
        "",
        "$$\\epsilon_{r,\\mathrm{Lorentz}}=\\epsilon_{r,\\infty}+"
        "\\sum_p\\frac{\\Delta\\epsilon_{r,p}\\Omega_p^2}"
        "{\\Omega_p^2-\\omega^2+2j\\delta_p\\omega},$$",
        "",
        "$$\\epsilon_{r,\\mathrm{Drude}}=\\epsilon_{r,\\infty}-"
        "\\sum_p\\frac{\\Omega_p^2}{\\omega^2-j\\gamma_p\\omega}.$$",
        "",
        "This follows the inclusive recursive-convolution formulation in "
        "[Giannakis and Giannopoulos (2014)]"
        "(https://doi.org/10.1109/TAP.2014.2308549).",
        "",
        "## Results",
        "",
        f"Overall validation status: **{'PASS' if summary['acceptance']['passed'] else 'FAIL'}**.",
        "",
        "| Material | Magnitude RMSE | Maximum magnitude error | Phase RMSE |",
        "|---|---:|---:|---:|",
    ]
    for name, metrics in summary["cases"].items():
        lines.append(
            f"| {CASES[name]['label']} | {metrics['magnitude_rmse']:.4g} | "
            f"{metrics['magnitude_max_error']:.4g} | "
            f"{metrics['phase_rmse_degrees']:.3g} deg |"
        )
    lines.extend(
        [
            "",
            "Both magnitude and phase are compared throughout 0.25--8 GHz. The ",
            "smoothed and unsmoothed dielectric cases explicitly verify the half-cell ",
            "change in the discrete reflection plane. The complex Lorentz pole update ",
            "uses the full real part `Re(a*T)` required by the recursive-convolution ",
            "formulation, rather than the incorrect product `Re(a)*Re(T)`.",
            "",
            "## Plots",
            "",
            "- [Reflection magnitude](reflection_magnitude.png)",
            "- [Reflection phase](reflection_phase.png)",
            "- [Time-domain fields](time_domain_fields.png)",
            "- [Material permittivity](material_permittivity.png)",
            "",
            "Per-frequency complex data are stored in the six `*_reflection.csv` files. ",
            "Simulation HDF5 files are reusable cache data and are not retained as ",
            "validation evidence.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(line.rstrip() for line in lines) + "\n")


def _acceptance_summary(results):
    """Apply common tolerances to every analytical material comparison."""

    checks = {}
    for case_name, result in results.items():
        for metric, maximum in ACCEPTANCE_LIMITS.items():
            key = f"{case_name}_{metric}"
            checks[key] = {
                "value": result[metric],
                "maximum": maximum,
                "passed": result[metric] <= maximum,
            }
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks}


def run_validation(
    output_dir,
    threads=4,
    precision="double",
    reuse=False,
    gpu=None,
    cache_dir=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir is not None else output_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    previous_runtimes = {}
    if reuse and summary_path.exists():
        with summary_path.open() as infile:
            previous_runtimes = json.load(infile).get("runtime_seconds", {})

    incident, dt, reference_runtime = run_case(
        "free_space", cache_dir, threads, precision, reuse, gpu
    )
    totals = {}
    results = {}
    runtimes = {
        "free_space": reference_runtime
        if reference_runtime
        else previous_runtimes.get("free_space", 0.0)
    }
    for case_name in CASES:
        total, case_dt, runtime = run_case(case_name, cache_dir, threads, precision, reuse, gpu)
        if not np.isclose(case_dt, dt, rtol=1e-12, atol=0):
            raise RuntimeError(f"Time-step mismatch for {case_name}: {case_dt} != {dt}")
        totals[case_name] = total
        runtimes[case_name] = runtime if runtime else previous_runtimes.get(case_name, 0.0)
        results[case_name] = analyse_case(case_name, incident, total, dt)
        _save_csv(output_dir / f"{case_name}_reflection.csv", results[case_name])

    _plot_comparison(results, output_dir)
    _plot_permittivity(results, output_dir)
    _plot_time_traces(incident, totals, results, dt, output_dir)

    summary = {
        "model": {
            "dl_metres": DL,
            "time_window_seconds": TIME_WINDOW,
            "dt_seconds": dt,
            "interface_x_metres": INTERFACE_X,
            "receiver_x_metres": RECEIVER[0],
            "nominal_receiver_to_interface_metres": INTERFACE_X - RECEIVER[0],
            "source_frequency_hz": SOURCE_FREQUENCY,
            "frequency_band_hz": [FREQUENCY_MIN, FREQUENCY_MAX],
            "precision": precision,
            "threads": threads,
            "backend": "cpu" if gpu is None else f"cuda:{gpu}",
        },
        "runtime_seconds": runtimes,
        "cases": {
            name: {
                "magnitude_rmse": result["magnitude_rmse"],
                "magnitude_max_error": result["magnitude_max_error"],
                "complex_relative_l2_error": result["complex_relative_l2_error"],
                "phase_rmse_degrees": result["phase_rmse_degrees"],
                "effective_interface_x_metres": result["effective_interface_x"],
                "receiver_to_interface_metres": result["receiver_to_interface_distance"],
                "frequency_samples": int(result["frequencies"].size),
            }
            for name, result in results.items()
        },
        "acceptance": _acceptance_summary(results),
    }
    with summary_path.open("w") as outfile:
        json.dump(summary, outfile, indent=2)
    _write_report(output_dir, summary)

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("plane_wave_halfspace_results"),
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--precision", choices=("single", "double"), default="double")
    parser.add_argument("--gpu", type=int, help="CUDA device index; omit for CPU")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="directory for reusable HDF5 runs (default: OUTPUT_DIR/_cache)",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing HDF5 simulations and regenerate analysis/plots.",
    )
    args = parser.parse_args()
    summary = run_validation(
        args.output_dir,
        threads=args.threads,
        precision=args.precision,
        reuse=args.reuse,
        gpu=args.gpu,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(summary, indent=2))
    if not summary["acceptance"]["passed"]:
        raise SystemExit("Dispersive half-space analytical validation failed")


if __name__ == "__main__":
    main()
