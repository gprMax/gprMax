"""Validate DPW reflection from water and Puerto Rico clay half-spaces.

Each material uses a separate spatial/temporal scale and Ricker excitation so
that the useful incident spectrum spans its physical Debye relaxation region.
The reflected field is obtained from a free-space reference at the same Yee
receiver and de-embedded to the discrete half-space reflection plane using the
axial Yee numerical wavenumber and the engineering FFT convention.

Example::

    python -m testing.validation.validate_plane_wave_realistic_materials
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
from gprMax.materials import calculate_water_properties

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

PML_CELLS = 12
# The clay excitation must cover both its 51 MHz and 1.45 GHz poles. Double
# precision makes a 0.01% incident-spectrum cutoff safe for the low-frequency
# side while still avoiding division by an insignificant FFT component.
INCIDENT_SPECTRUM_THRESHOLD = 0.0001

ACCEPTANCE_LIMITS = {
    "magnitude_rmse": 0.01,
    "complex_relative_l2_error": 0.02,
    "phase_rmse_degrees": 0.5,
}


water_er_inf, water_er_static, water_tau, water_conductivity = calculate_water_properties(T=25, S=0)


CASES = {
    "puerto_rico_clay_10pct": {
        "label": "Puerto Rico clay loam, 10% moisture",
        "short_label": "Puerto Rico clay",
        "dl": 0.5e-3,
        "domain": (0.16, 0.02, float("inf")),
        "time_window": 120e-9,
        "tfsf_p1": (0.0125, 0.007, float("inf")),
        "tfsf_p2": (0.1475, 0.013, float("inf")),
        "interface_x": 0.09,
        "receiver": (0.0725, 0.010, float("inf")),
        "source_frequency": 0.75e9,
        "frequency_min": 0.020e9,
        "frequency_max": 2.2e9,
        "er_inf": 5.706,
        "conductivity": 3.022e-3,
        "delta_er": (2.219, 0.958),
        "tau": (3.100e-9, 0.110e-9),
        "citation": "https://doi.org/10.2528/PIER04061002",
    },
    "fresh_water_25c": {
        "label": "Fresh water, 25 degC",
        "short_label": "Fresh water",
        # The fine mesh resolves the shortest lossy-water phase wavelength
        # with more than twice as many cells as the original validation.
        "dl": 0.05e-3,
        "domain": (0.016, 0.002, float("inf")),
        "time_window": 3e-9,
        "tfsf_p1": (0.00125, 0.0007, float("inf")),
        "tfsf_p2": (0.01475, 0.0013, float("inf")),
        "interface_x": 0.009,
        "receiver": (0.00725, 0.0010, float("inf")),
        "source_frequency": 30e9,
        "frequency_min": 1e9,
        "frequency_max": 80e9,
        "er_inf": water_er_inf,
        "conductivity": water_conductivity,
        "delta_er": (water_er_static - water_er_inf,),
        "tau": (water_tau,),
        "citation": "https://doi.org/10.1109/JOE.1977.1145319",
    },
}


def build_scene(case_name, include_halfspace, threads):
    """Build a free-space reference or material half-space scene."""

    params = CASES[case_name]
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(params["dl"],) * 3))
    scene.add(gprMax.Domain(p1=params["domain"]))
    scene.add(gprMax.TimeWindow(time=params["time_window"]))
    scene.add(gprMax.OMPThreads(n=threads))
    scene.add(gprMax.PMLThickness(thickness=(PML_CELLS, PML_CELLS, 0, PML_CELLS, PML_CELLS, 0)))

    if include_halfspace:
        material_id = f"halfspace_{case_name}"
        scene.add(
            gprMax.Material(
                er=params["er_inf"],
                se=params["conductivity"],
                mr=1,
                sm=0,
                id=material_id,
            )
        )
        scene.add(
            gprMax.AddDebyeDispersion(
                poles=len(params["delta_er"]),
                er_delta=params["delta_er"],
                tau=params["tau"],
                material_ids=[material_id],
            )
        )
        scene.add(
            gprMax.Box(
                p1=(params["interface_x"], 0, float("inf")),
                p2=params["domain"],
                material_id=material_id,
            )
        )

    scene.add(
        gprMax.Waveform(
            wave_type="ricker",
            amp=1,
            freq=params["source_frequency"],
            id="plane_pulse",
        )
    )
    scene.add(
        gprMax.DiscretePlaneWaveAxial(
            p1=params["tfsf_p1"],
            p2=params["tfsf_p2"],
            axis="x",
            psi=90,
            waveform_id="plane_pulse",
        )
    )
    scene.add(gprMax.Rx(p1=params["receiver"], id="reflection_probe", outputs=["Ez"]))
    return scene


def run_model(
    case_name,
    include_halfspace,
    cache_dir,
    threads,
    precision,
    reuse,
    gpu,
):
    """Run one model and return its HDF5 receiver trace and time step."""

    suffix = "halfspace" if include_halfspace else "free_space"
    outputfile = cache_dir / f"{case_name}_{suffix}"
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
            scenes=[build_scene(case_name, include_halfspace, threads)],
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
    """Return continuous-time Debye permittivity including conductivity."""

    params = CASES[case_name]
    omega = 2 * np.pi * np.asarray(frequencies)
    er = np.full(omega.shape, params["er_inf"], dtype=np.complex128)
    er += params["conductivity"] / (1j * omega * epsilon_0)
    for delta_er, tau in zip(params["delta_er"], params["tau"]):
        er += delta_er / (1 + 1j * omega * tau)
    return er


def fresnel_reflection(er):
    """Electric-field reflection coefficient from free space."""

    impedance_1 = np.sqrt(mu_0 / epsilon_0)
    impedance_2 = impedance_1 / np.sqrt(er)
    return (impedance_2 - impedance_1) / (impedance_2 + impedance_1)


def numerical_wavenumber(frequencies, dt, dl):
    """Return the axial free-space Yee wavenumber."""

    courant = c * dt / dl
    argument = np.sin(np.pi * np.asarray(frequencies) * dt) / courant
    if np.any(np.abs(argument) > 1 + 1e-12):
        raise ValueError("Selected frequencies are outside the propagating Yee band")
    return (2 / dl) * np.arcsin(np.clip(argument, -1, 1))


def analyse_case(case_name, incident, total, dt):
    """Calculate simulated and analytical interface reflection coefficients."""

    params = CASES[case_name]
    reflected = total - incident
    frequencies = np.fft.rfftfreq(incident.size, dt)
    incident_fft = np.fft.rfft(incident)
    reflected_fft = np.fft.rfft(reflected)
    selected = (
        (frequencies >= params["frequency_min"])
        & (frequencies <= params["frequency_max"])
        & (np.abs(incident_fft) >= INCIDENT_SPECTRUM_THRESHOLD * np.max(np.abs(incident_fft)))
    )
    frequencies = frequencies[selected]
    if frequencies.size == 0:
        raise RuntimeError(f"No usable FFT samples for {case_name}")

    gamma_receiver = reflected_fft[selected] / incident_fft[selected]
    # Debye materials are rigid/non-averagable. Their tangential-E material
    # assignment starts at xs while the H transition lies half a cell earlier.
    effective_interface_x = params["interface_x"] - 0.5 * params["dl"]
    distance = effective_interface_x - params["receiver"][0]
    k_numerical = numerical_wavenumber(frequencies, dt, params["dl"])
    gamma_fdtd = gamma_receiver * np.exp(1j * 2 * k_numerical * distance)
    er = complex_permittivity(case_name, frequencies)
    gamma_analytic = fresnel_reflection(er)

    magnitude_error = np.abs(gamma_fdtd) - np.abs(gamma_analytic)
    phase_error = np.angle(gamma_fdtd / gamma_analytic, deg=True)
    return {
        "frequencies": frequencies,
        "incident": incident,
        "total": total,
        "reflected": reflected,
        "gamma_fdtd": gamma_fdtd,
        "gamma_analytic": gamma_analytic,
        "er": er,
        "effective_interface_x": effective_interface_x,
        "receiver_to_interface_distance": distance,
        "magnitude_error": magnitude_error,
        "phase_error": phase_error,
        "magnitude_rmse": float(np.sqrt(np.mean(magnitude_error**2))),
        "magnitude_max_error": float(np.max(np.abs(magnitude_error))),
        "phase_rmse_degrees": float(np.sqrt(np.mean(phase_error**2))),
        "phase_max_error_degrees": float(np.max(np.abs(phase_error))),
        "complex_relative_l2_error": float(
            np.linalg.norm(gamma_fdtd - gamma_analytic) / np.linalg.norm(gamma_analytic)
        ),
    }


def save_csv(path, result):
    """Write per-frequency complex comparison data."""

    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            (
                "frequency_hz",
                "fdtd_magnitude",
                "fdtd_phase_degrees",
                "analytic_magnitude",
                "analytic_phase_degrees",
                "magnitude_error",
                "phase_error_degrees",
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
                    result["magnitude_error"][index],
                    result["phase_error"][index],
                    result["er"][index].real,
                    result["er"][index].imag,
                )
            )


def plot_results(results, output_dir):
    """Create reflection, error, permittivity, and time-domain plots."""

    fig, axes = plt.subplots(len(results), 2, figsize=(12, 8), squeeze=False)
    for row, (case_name, result) in enumerate(results.items()):
        params = CASES[case_name]
        frequency_ghz = result["frequencies"] / 1e9
        magnitude_axis, phase_axis = axes[row]
        magnitude_axis.plot(
            frequency_ghz,
            np.abs(result["gamma_analytic"]),
            "k-",
            linewidth=2,
            label="Analytical",
        )
        magnitude_axis.plot(
            frequency_ghz,
            np.abs(result["gamma_fdtd"]),
            color="tab:red",
            linestyle="--",
            linewidth=1.5,
            label="gprMax FDTD",
        )
        analytic_phase = np.unwrap(np.angle(result["gamma_analytic"])) * 180 / np.pi
        fdtd_phase = np.unwrap(np.angle(result["gamma_fdtd"])) * 180 / np.pi
        offset = 360 * np.round(np.median((analytic_phase - fdtd_phase) / 360))
        phase_axis.plot(frequency_ghz, analytic_phase, "k-", linewidth=2, label="Analytical")
        phase_axis.plot(
            frequency_ghz,
            fdtd_phase + offset,
            color="tab:blue",
            linestyle="--",
            linewidth=1.5,
            label="gprMax FDTD",
        )
        magnitude_axis.set_title(f"{params['short_label']}: magnitude")
        phase_axis.set_title(f"{params['short_label']}: phase")
        magnitude_axis.set_ylabel(r"$|\Gamma|$")
        phase_axis.set_ylabel(r"Phase of $\Gamma$ [deg]")
        for axis in (magnitude_axis, phase_axis):
            axis.set_xlabel("Frequency [GHz]")
            axis.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best")
    axes[0, 1].legend(loc="best")
    fig.suptitle("Normal-incidence reflection from realistic dispersive materials")
    fig.tight_layout()
    fig.savefig(output_dir / "reflection_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(results), 2, figsize=(12, 8), squeeze=False)
    for row, (case_name, result) in enumerate(results.items()):
        params = CASES[case_name]
        frequency_ghz = result["frequencies"] / 1e9
        axes[row, 0].plot(frequency_ghz, result["magnitude_error"])
        axes[row, 1].plot(frequency_ghz, result["phase_error"])
        axes[row, 0].set_title(f"{params['short_label']}: magnitude residual")
        axes[row, 1].set_title(f"{params['short_label']}: phase residual")
        axes[row, 0].set_ylabel(r"$|\Gamma_{FDTD}|-|\Gamma_{analytic}|$")
        axes[row, 1].set_ylabel("Phase error [deg]")
        for axis in axes[row]:
            axis.set_xlabel("Frequency [GHz]")
            axis.axhline(0, color="black", linewidth=0.8)
            axis.grid(True, alpha=0.3)
    fig.suptitle("gprMax reflection-coefficient residuals")
    fig.tight_layout()
    fig.savefig(output_dir / "reflection_error.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(results), 2, figsize=(12, 8), squeeze=False)
    for row, (case_name, result) in enumerate(results.items()):
        params = CASES[case_name]
        frequency_ghz = result["frequencies"] / 1e9
        axes[row, 0].plot(frequency_ghz, result["er"].real)
        axes[row, 1].plot(frequency_ghz, -result["er"].imag)
        axes[row, 0].set_title(f"{params['short_label']}: real permittivity")
        axes[row, 1].set_title(f"{params['short_label']}: dielectric loss")
        axes[row, 0].set_ylabel(r"$\Re\{\epsilon_r\}$")
        axes[row, 1].set_ylabel(r"$-\Im\{\epsilon_r\}$")
        for axis in axes[row]:
            axis.set_xlabel("Frequency [GHz]")
            axis.grid(True, alpha=0.3)
    fig.suptitle("Analytical dispersive material properties")
    fig.tight_layout()
    fig.savefig(output_dir / "material_permittivity.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(results), 1, figsize=(10, 7), squeeze=False)
    for row, (case_name, result) in enumerate(results.items()):
        params = CASES[case_name]
        axis = axes[row, 0]
        dt = result["dt"]
        time = np.arange(result["incident"].size) * dt * 1e9
        signal = np.maximum.reduce(
            (
                np.abs(result["incident"]),
                np.abs(result["total"]),
                np.abs(result["reflected"]),
            )
        )
        active = np.flatnonzero(signal > 1e-4 * np.max(signal))
        if active.size:
            start = max(0, active[0] - 100)
            stop = min(time.size, active[-1] + 101)
        else:
            start, stop = 0, time.size
        axis.plot(time[start:stop], result["incident"][start:stop], label="Incident")
        axis.plot(time[start:stop], result["total"][start:stop], label="Total")
        axis.plot(time[start:stop], result["reflected"][start:stop], label="Reflected")
        axis.set_title(params["short_label"])
        axis.set_xlabel("Time [ns]")
        axis.set_ylabel(r"$E_z$ [V/m]")
        axis.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best")
    fig.suptitle("Receiver fields inside the TFSF region")
    fig.tight_layout()
    fig.savefig(output_dir / "time_domain_fields.png", dpi=180)
    plt.close(fig)


def write_report(output_dir, summary):
    """Write the material definitions and numerical metrics."""

    lines = [
        "# Water and Puerto Rico clay DPW half-space validation",
        "",
        "The two materials use separate model scales because their Debye relaxation ",
        "frequencies differ by more than two orders of magnitude. In both cases, ",
        "the free-space reference is subtracted at the same receiver, and the ",
        "reflection coefficient is de-embedded with the axial Yee wavenumber to ",
        "the half-cell-shifted reflection plane of the non-averagable material.",
        "",
        "The engineering-convention material response is",
        "",
        "$$\\epsilon_r(\\omega)=\\epsilon_{r,\\infty}"
        "+\\frac{\\sigma}{j\\omega\\epsilon_0}"
        "+\\sum_p\\frac{\\Delta\\epsilon_{r,p}}{1+j\\omega\\tau_p}.$$",
        "",
        "## Material definitions and frequency bands",
        "",
    ]
    for case_name, params in CASES.items():
        relaxations = [1 / (2 * np.pi * tau) / 1e9 for tau in params["tau"]]
        lines.extend(
            [
                f"### {params['label']}",
                "",
                f"- $\\epsilon_{{r,\\infty}}={params['er_inf']:.8g}$",
                f"- $\\Delta\\epsilon_r={params['delta_er']}$",
                f"- $\\tau={params['tau']}$ s",
                f"- $\\sigma={params['conductivity']:.8g}$ S/m",
                "- relaxation frequencies: "
                f"{', '.join(f'{value:.5g}' for value in relaxations)} GHz",
                "- validation band: "
                f"{params['frequency_min'] / 1e9:g}--"
                f"{params['frequency_max'] / 1e9:g} GHz",
                f"- [parameter source]({params['citation']})",
                "",
            ]
        )
    lines.extend(
        [
            "## Results",
            "",
            f"Overall validation status: **{'PASS' if summary['acceptance']['passed'] else 'FAIL'}**.",
            "",
            "| Material | Magnitude RMSE | Maximum magnitude error | "
            "Phase RMSE | Maximum phase error |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for case_name, metrics in summary["cases"].items():
        lines.append(
            f"| {CASES[case_name]['label']} | {metrics['magnitude_rmse']:.4g} | "
            f"{metrics['magnitude_max_error']:.4g} | "
            f"{metrics['phase_rmse_degrees']:.4g} deg | "
            f"{metrics['phase_max_error_degrees']:.4g} deg |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- [Reflection magnitude and phase](reflection_comparison.png)",
            "- [Magnitude and phase residuals](reflection_error.png)",
            "- [Material permittivity](material_permittivity.png)",
            "- [Time-domain fields](time_domain_fields.png)",
            "",
            "Each material also has a CSV file containing the complex comparison data. ",
            "Simulation HDF5 files are reusable cache data and are not retained as ",
            "validation evidence.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(line.rstrip() for line in lines) + "\n")


def _acceptance_summary(results):
    """Apply common tolerances to both realistic-material comparisons."""

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
    """Run both material/reference pairs and generate all outputs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir is not None else output_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    previous_runtimes = {}
    if reuse and summary_path.exists():
        with summary_path.open() as infile:
            previous_runtimes = json.load(infile).get("runtime_seconds", {})
    results = {}
    runtimes = {}
    for case_name in CASES:
        incident, reference_dt, reference_runtime = run_model(
            case_name, False, cache_dir, threads, precision, reuse, gpu
        )
        total, material_dt, material_runtime = run_model(
            case_name, True, cache_dir, threads, precision, reuse, gpu
        )
        if not np.isclose(reference_dt, material_dt, rtol=1e-12, atol=0):
            raise RuntimeError(
                f"Time-step mismatch for {case_name}: " f"{reference_dt} != {material_dt}"
            )
        result = analyse_case(case_name, incident, total, reference_dt)
        result["dt"] = reference_dt
        results[case_name] = result
        runtimes[case_name] = {
            "free_space": reference_runtime
            if reference_runtime
            else previous_runtimes.get(case_name, {}).get("free_space", 0.0),
            "halfspace": material_runtime
            if material_runtime
            else previous_runtimes.get(case_name, {}).get("halfspace", 0.0),
        }
        save_csv(output_dir / f"{case_name}_reflection.csv", result)

    plot_results(results, output_dir)
    summary = {
        "precision": precision,
        "threads": threads,
        "backend": "cpu" if gpu is None else f"cuda:{gpu}",
        "runtime_seconds": runtimes,
        "cases": {
            case_name: {
                "dl_metres": CASES[case_name]["dl"],
                "dt_seconds": result["dt"],
                "time_window_seconds": CASES[case_name]["time_window"],
                "frequency_band_hz": [
                    CASES[case_name]["frequency_min"],
                    CASES[case_name]["frequency_max"],
                ],
                "evaluated_frequency_range_hz": [
                    float(result["frequencies"][0]),
                    float(result["frequencies"][-1]),
                ],
                "frequency_samples": int(result["frequencies"].size),
                "effective_interface_x_metres": result["effective_interface_x"],
                "receiver_to_interface_metres": result["receiver_to_interface_distance"],
                "magnitude_rmse": result["magnitude_rmse"],
                "magnitude_max_error": result["magnitude_max_error"],
                "phase_rmse_degrees": result["phase_rmse_degrees"],
                "phase_max_error_degrees": result["phase_max_error_degrees"],
                "complex_relative_l2_error": result["complex_relative_l2_error"],
            }
            for case_name, result in results.items()
        },
        "acceptance": _acceptance_summary(results),
    }
    with summary_path.open("w") as outfile:
        json.dump(summary, outfile, indent=2)
    write_report(output_dir, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("plane_wave_realistic_material_results"),
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
        raise SystemExit("Realistic-material half-space analytical validation failed")


if __name__ == "__main__":
    main()
