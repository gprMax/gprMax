"""Validate Hertzian-dipole far and near fields against closed-form results.

The far-field case exercises both KSIR and conventional equivalent-current
transforms. The near-field case compares one ``Ez`` component from a direct
Yee receiver and a KSIR reconstruction with the analytical time-domain field.

Examples::

    python -m testing.validation.validate_hertzian_dipole
    python -m testing.validation.validate_hertzian_dipole --gpu 0
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

import gprMax
from testing.analytical_solutions import hertzian_dipole_fs

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

DL = 1e-3
DOMAIN_SIZE = 0.1
SOURCE = (0.05, 0.05, 0.05)
FAR_FREQUENCY = 3e9
FAR_TIME_WINDOW = 2e-9
NEAR_TIME_WINDOW = 3e-9
NEAR_RECEIVER = (0.07, 0.07, 0.07)
THETA = np.arange(0.0, 181.0, 2.0)
PHI = np.arange(0.0, 361.0, 2.0)

ACCEPTANCE_LIMITS = {
    "far_field_directivity_rms": 0.001,
    "far_field_maximum_directivity_error": 0.001,
    "near_field_relative_l2": 0.001,
}


def _solver_options(gpu, precision):
    if gpu is None:
        return {"cpu_precision": precision}
    return {"gpu": [gpu], "gpu_precision": precision}


def _run(scene, outputfile, gpu, precision):
    start = perf_counter()
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=outputfile,
        hide_progress_bars=True,
        log_level=logging.WARNING,
        **_solver_options(gpu, precision),
    )
    return perf_counter() - start


def build_far_field_scene(threads=4):
    """Build simultaneous KSIR and equivalent-current transforms."""

    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(DL,) * 3))
    scene.add(gprMax.Domain(p1=(DOMAIN_SIZE,) * 3))
    scene.add(gprMax.TimeWindow(time=FAR_TIME_WINDOW))
    scene.add(gprMax.OMPThreads(n=threads))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=FAR_FREQUENCY, id="pulse"))
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=SOURCE, waveform_id="pulse"))
    scene.add(
        gprMax.NTFFSurface(
            p1=(0.032,) * 3,
            p2=(0.068,) * 3,
            id="dipole_surface",
            origin=SOURCE,
        )
    )

    ksir_transform = gprMax.KSIRFrequencyTransform(
        "dipole_surface",
        "ksir_spectrum",
        (FAR_FREQUENCY,),
        save_surface_dft=False,
    )
    ntff_transform = gprMax.NTFFFrequencyTransform(
        "dipole_surface",
        "ntff_spectrum",
        (FAR_FREQUENCY,),
        save_surface_dft=False,
    )
    scene.add(ksir_transform)
    scene.add(ntff_transform)

    outputs = ("Etheta", "radiation_intensity", "directivity", "directivity_dbi")
    requests = {}
    for method, transform_id, request_type in (
        ("ksir", "ksir_spectrum", gprMax.KSIRFarField),
        ("equivalent_current", "ntff_spectrum", gprMax.NTFFFarField),
    ):
        requests[f"{method}_e"] = request_type(
            theta=THETA,
            phi=np.zeros(THETA.shape),
            transform_id=transform_id,
            id=f"{method}_e_plane",
            outputs=outputs,
        )
        requests[f"{method}_h"] = request_type(
            theta=np.full(PHI.shape, 90.0),
            phi=PHI,
            transform_id=transform_id,
            id=f"{method}_h_plane",
            outputs=outputs,
        )
        scene.add(requests[f"{method}_e"])
        scene.add(requests[f"{method}_h"])
    return scene, requests


def build_near_field_scene(threads=4):
    """Build a direct receiver and collocated KSIR near-field point."""

    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(DL,) * 3))
    scene.add(gprMax.Domain(p1=(DOMAIN_SIZE,) * 3))
    scene.add(gprMax.TimeWindow(time=NEAR_TIME_WINDOW))
    scene.add(gprMax.OMPThreads(n=threads))
    scene.add(
        gprMax.Waveform(
            wave_type="gaussianprime",
            amp=1,
            freq=1e9,
            id="pulse",
        )
    )
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=SOURCE, waveform_id="pulse"))
    scene.add(gprMax.Rx(p1=NEAR_RECEIVER, id="direct", outputs=["Ez"]))
    scene.add(
        gprMax.NTFFSurface(
            p1=(0.04,) * 3,
            p2=(0.06,) * 3,
            id="near_surface",
            origin=SOURCE,
        )
    )
    # Source and Rx command coordinates identify the lower ends of their Ez
    # Yee edges. Both edge centres have the same dz/2 shift, so the analytical
    # source-to-receiver vector remains NEAR_RECEIVER - SOURCE. KSIR accepts
    # an absolute physical observation point and therefore needs the shift.
    ksir_point = (
        NEAR_RECEIVER[0],
        NEAR_RECEIVER[1],
        NEAR_RECEIVER[2] + 0.5 * DL,
    )
    receiver = gprMax.KSIRTimeRx(
        position=(ksir_point,),
        surface_id="near_surface",
        id="near_ez",
        outputs=("Ez",),
        time_origin="simulation",
    )
    scene.add(receiver)
    return scene, receiver


def _pattern_metrics(actual, analytical):
    difference = actual - analytical
    return {
        "rms_error": float(np.sqrt(np.mean(difference**2))),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
    }


def _waveform_metrics(actual, analytical):
    peak = float(np.max(np.abs(analytical)))
    significant = np.abs(analytical) >= 0.01 * peak
    difference = actual - analytical
    return {
        "relative_l2_error_significant": float(
            np.linalg.norm(difference[significant]) / np.linalg.norm(analytical[significant])
        ),
        "relative_l2_error_full": float(np.linalg.norm(difference) / np.linalg.norm(analytical)),
        "correlation_significant": float(
            np.corrcoef(actual[significant], analytical[significant])[0, 1]
        ),
        "peak_relative_error": float(abs(np.max(np.abs(actual)) - peak) / peak),
    }


def analyse_far_field(requests):
    """Return analytical comparisons for both transform formulations."""

    analytical_field = np.abs(np.sin(np.deg2rad(THETA)))
    analytical_directivity = 1.5 * analytical_field**2
    analytical_h_directivity = np.full(PHI.shape, 1.5)
    results = {}
    for method in ("ksir", "equivalent_current"):
        e_result = requests[f"{method}_e"].result
        h_result = requests[f"{method}_h"].result
        e_field = np.abs(e_result.fields["Etheta"][0])
        e_field /= np.max(e_field)
        e_directivity = np.asarray(e_result.fields["directivity"][0])
        h_directivity = np.asarray(h_result.fields["directivity"][0])
        radiation = e_result.radiation_metrics
        results[method] = {
            "e_field": e_field,
            "e_directivity": e_directivity,
            "h_directivity": h_directivity,
            "field_metrics": _pattern_metrics(e_field, analytical_field),
            "directivity_metrics": _pattern_metrics(e_directivity, analytical_directivity),
            "h_plane_metrics": _pattern_metrics(h_directivity, analytical_h_directivity),
            "maximum_directivity": float(radiation.maximum_directivity[0]),
            "maximum_directivity_dbi": float(radiation.maximum_directivity_dbi[0]),
            "radiated_power": float(radiation.radiated_power[0]),
        }
    return {
        "theta_degrees": THETA,
        "phi_degrees": PHI,
        "analytical_field": analytical_field,
        "analytical_directivity": analytical_directivity,
        "analytical_h_directivity": analytical_h_directivity,
        "methods": results,
    }


def analyse_near_field(h5_path, receiver):
    """Compare direct and reconstructed Ez with the analytical waveform."""

    with h5py.File(h5_path, "r") as output:
        direct = np.asarray(output["rxs/rx1/Ez"], dtype=np.float64)
        dt = float(output.attrs["dt"])
        iterations = int(output.attrs["Iterations"])
    analytical = hertzian_dipole_fs(
        iterations,
        dt,
        (DL,) * 3,
        tuple(np.asarray(NEAR_RECEIVER) - np.asarray(SOURCE)),
    )[:, 2]
    reconstructed = np.asarray(receiver.result.fields["Ez"][0, :iterations])
    return {
        "time_seconds": np.arange(iterations) * dt,
        "analytical_ez": analytical,
        "direct_ez": direct,
        "ksir_ez": reconstructed,
        "direct_metrics": _waveform_metrics(direct, analytical),
        "ksir_metrics": _waveform_metrics(reconstructed, analytical),
        "dt_seconds": dt,
        "distance_metres": float(np.linalg.norm(np.asarray(NEAR_RECEIVER) - np.asarray(SOURCE))),
    }


def _save_far_csv(path, result):
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            (
                "theta_degrees",
                "analytical_field_normalized",
                "analytical_directivity",
                "ksir_field_normalized",
                "ksir_directivity",
                "equivalent_current_field_normalized",
                "equivalent_current_directivity",
            )
        )
        for index, theta in enumerate(result["theta_degrees"]):
            writer.writerow(
                (
                    theta,
                    result["analytical_field"][index],
                    result["analytical_directivity"][index],
                    result["methods"]["ksir"]["e_field"][index],
                    result["methods"]["ksir"]["e_directivity"][index],
                    result["methods"]["equivalent_current"]["e_field"][index],
                    result["methods"]["equivalent_current"]["e_directivity"][index],
                )
            )


def _save_near_csv(path, result):
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(("time_seconds", "analytical_ez", "direct_fdtd_ez", "ksir_ez"))
        writer.writerows(
            zip(
                result["time_seconds"],
                result["analytical_ez"],
                result["direct_ez"],
                result["ksir_ez"],
            )
        )


def _plot_far(path, result):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].plot(THETA, result["analytical_field"], "k-", label="Analytical")
    axes[1].plot(THETA, result["analytical_directivity"], "k-", label="Analytical")
    axes[2].plot(PHI, result["analytical_h_directivity"], "k-", label="Analytical")
    colours = {"ksir": "tab:blue", "equivalent_current": "tab:orange"}
    labels = {"ksir": "KSIR", "equivalent_current": "Equivalent currents"}
    for method, values in result["methods"].items():
        axes[0].plot(THETA, values["e_field"], "--", color=colours[method], label=labels[method])
        axes[1].plot(
            THETA,
            values["e_directivity"],
            "--",
            color=colours[method],
            label=labels[method],
        )
        axes[2].plot(
            PHI,
            values["h_directivity"],
            "--",
            color=colours[method],
            label=labels[method],
        )
    axes[0].set(xlabel=r"$\theta$ [deg]", ylabel="Normalised |Etheta|", title="E-plane field")
    axes[1].set(
        xlabel=r"$\theta$ [deg]", ylabel="Directivity", title="E-plane directivity / ideal gain"
    )
    axes[2].set(
        xlabel=r"$\phi$ [deg]", ylabel="Directivity", title="H-plane directivity / ideal gain"
    )
    for axis in axes:
        axis.grid(True, alpha=0.3)
    axes[0].legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_near(path, result):
    time_ns = result["time_seconds"] * 1e9
    analytical_peak = np.max(np.abs(result["analytical_ez"]))
    direct_error_percent = 100 * (result["direct_ez"] - result["analytical_ez"]) / analytical_peak
    ksir_error_percent = 100 * (result["ksir_ez"] - result["analytical_ez"]) / analytical_peak
    figure, (field_axis, error_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (2.2, 1)},
    )
    field_axis.plot(time_ns, result["analytical_ez"], "k-", linewidth=2, label="Analytical")
    field_axis.plot(time_ns, result["direct_ez"], "--", label="Direct FDTD receiver")
    field_axis.plot(time_ns, result["ksir_ez"], ":", linewidth=2, label="KSIR")
    error_axis.plot(time_ns, direct_error_percent, label="Direct - analytical")
    error_axis.plot(time_ns, ksir_error_percent, label="KSIR - analytical")
    field_axis.set_ylabel(r"$E_z$ [V/m]")
    error_axis.set(xlabel="Time [ns]", ylabel="Error [% of analytical peak]")
    for axis in (field_axis, error_axis):
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _compact_far_summary(result):
    return {
        "analytical_maximum_directivity": 1.5,
        "analytical_maximum_directivity_dbi": float(10 * np.log10(1.5)),
        "methods": {
            method: {
                key: value
                for key, value in values.items()
                if key not in ("e_field", "e_directivity", "h_directivity")
            }
            for method, values in result["methods"].items()
        },
    }


def _compact_near_summary(result):
    return {
        key: value
        for key, value in result.items()
        if key not in ("time_seconds", "analytical_ez", "direct_ez", "ksir_ez")
    }


def _acceptance_summary(far_result, near_result):
    """Apply deliberately conservative analytical-validation tolerances."""

    checks = {}
    for method, values in far_result["methods"].items():
        checks[f"{method}_directivity_rms"] = {
            "value": values["directivity_metrics"]["rms_error"],
            "maximum": ACCEPTANCE_LIMITS["far_field_directivity_rms"],
        }
        checks[f"{method}_maximum_directivity_error"] = {
            "value": abs(values["maximum_directivity"] - 1.5),
            "maximum": ACCEPTANCE_LIMITS["far_field_maximum_directivity_error"],
        }
    for method in ("direct", "ksir"):
        checks[f"{method}_near_field_relative_l2"] = {
            "value": near_result[f"{method}_metrics"]["relative_l2_error_significant"],
            "maximum": ACCEPTANCE_LIMITS["near_field_relative_l2"],
        }
    for check in checks.values():
        check["passed"] = check["value"] <= check["maximum"]
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks}


def _write_report(path, summary):
    analytical_peak_dbi = 10 * np.log10(1.5)
    lines = [
        "# Hertzian-dipole analytical validation",
        "",
        "For a z-directed Hertzian dipole, the far-zone field is proportional ",
        "to `sin(theta)`, the radiation intensity to `sin(theta)^2`, and",
        "",
        "`D(theta) = 1.5 sin(theta)^2`.",
        "",
        f"The analytical peak directivity is 1.5 ({analytical_peak_dbi:.6f} dBi). ",
        "For an ideal lossless dipole, gain equals directivity. gprMax does not ",
        "report port-normalised gain for `HertzianDipole` because this impressed ",
        "current source has no reference impedance or accepted-port power; the ",
        "plot therefore labels the analytical lossless-gain identity explicitly.",
        "",
        "Both KSIR and conventional equivalent-current transforms are compared ",
        "with the closed-form E- and H-plane patterns. The near-field figure ",
        "compares one Ez component from a direct Yee receiver and KSIR with the ",
        "complete analytical dipole field, including reactive, induction, and ",
        "radiation terms.",
        "",
        "The source and receiver command coordinates identify the lower ends ",
        "of z-directed Yee edges. Their physical Ez sample centres are each ",
        "shifted by dz/2, so the analytical relative vector is unchanged. KSIR ",
        "uses absolute Cartesian observation coordinates; its point is therefore ",
        "placed at the receiver coordinate plus `(0, 0, dz/2)`.",
        "",
        "## Results",
        "",
        f"Overall validation status: **{'PASS' if summary['acceptance']['passed'] else 'FAIL'}**.",
        "",
    ]
    for method, values in summary["far_field"]["methods"].items():
        lines.extend(
            [
                f"- {method}: peak directivity {values['maximum_directivity']:.6f} ",
                f"  ({values['maximum_directivity_dbi']:.6f} dBi); E-plane ",
                f"  directivity RMS error {values['directivity_metrics']['rms_error']:.4g}.",
            ]
        )
    lines.extend(
        [
            f"- Direct near-field Ez relative L2 error: "
            f"{summary['near_field']['direct_metrics']['relative_l2_error_significant']:.4g}.",
            f"- KSIR near-field Ez relative L2 error: "
            f"{summary['near_field']['ksir_metrics']['relative_l2_error_significant']:.4g}.",
            "",
            "## Outputs",
            "",
            "- [Far-field patterns and metrics](hertzian_dipole_far_field.png)",
            "- [Near-field waveform](hertzian_dipole_near_field.png)",
            "- `hertzian_dipole_far_field.csv`",
            "- `hertzian_dipole_near_field.csv`",
            "",
        ]
    )
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def run_validation(output_dir, cache_dir, threads=4, precision="double", gpu=None):
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    far_output = cache_dir / "hertzian_dipole_far"
    far_scene, requests = build_far_field_scene(threads)
    far_runtime = _run(far_scene, far_output, gpu, precision)
    far_result = analyse_far_field(requests)

    near_output = cache_dir / "hertzian_dipole_near"
    near_scene, near_receiver = build_near_field_scene(threads)
    near_runtime = _run(near_scene, near_output, gpu, precision)
    near_result = analyse_near_field(near_output.with_suffix(".h5"), near_receiver)

    _save_far_csv(output_dir / "hertzian_dipole_far_field.csv", far_result)
    _save_near_csv(output_dir / "hertzian_dipole_near_field.csv", near_result)
    _plot_far(output_dir / "hertzian_dipole_far_field.png", far_result)
    _plot_near(output_dir / "hertzian_dipole_near_field.png", near_result)

    summary = {
        "configuration": {
            "dl_metres": DL,
            "far_frequency_hz": FAR_FREQUENCY,
            "source_command_position_metres": SOURCE,
            "source_ez_centre_metres": (
                SOURCE[0],
                SOURCE[1],
                round(SOURCE[2] + 0.5 * DL, 15),
            ),
            "receiver_command_position_metres": NEAR_RECEIVER,
            "receiver_ez_centre_metres": (
                NEAR_RECEIVER[0],
                NEAR_RECEIVER[1],
                round(NEAR_RECEIVER[2] + 0.5 * DL, 15),
            ),
            "analytical_relative_position_metres": tuple(
                round(receiver - source, 15) for receiver, source in zip(NEAR_RECEIVER, SOURCE)
            ),
            "backend": "cpu" if gpu is None else f"cuda:{gpu}",
            "precision": precision,
            "threads": threads,
        },
        "runtime_seconds": {"far_field": far_runtime, "near_field": near_runtime},
        "far_field": _compact_far_summary(far_result),
        "near_field": _compact_near_summary(near_result),
        "acceptance": _acceptance_summary(far_result, near_result),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "report.md", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("hertzian_dipole_results"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).with_name("hertzian_dipole_results") / "_cache",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--precision", choices=("single", "double"), default="double")
    parser.add_argument("--gpu", type=int, help="CUDA device index; omit for CPU")
    args = parser.parse_args()
    summary = run_validation(
        args.output_dir,
        args.cache_dir,
        threads=args.threads,
        precision=args.precision,
        gpu=args.gpu,
    )
    print(json.dumps(summary, indent=2))
    if not summary["acceptance"]["passed"]:
        raise SystemExit("Hertzian-dipole analytical validation failed")


if __name__ == "__main__":
    main()
