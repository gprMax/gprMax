"""Check voltage-port consistency and precision through the CUDA/HDF5 path."""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gprMax

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
SINGLE_BASE = RESULTS_DIR / "rx_port_gpu_invariance_single"
DOUBLE_BASE = RESULTS_DIR / "rx_port_gpu_invariance_double"
DL = 2e-3
DOMAIN = (0.220, 0.220, 0.220)
TIME_WINDOW = 16e-9
FREQUENCY_BAND = (0.55e9, 1.35e9)


@dataclass(frozen=True)
class InvarianceCase:
    """One source configuration for the same rotated wire antenna."""

    name: str
    resistance: float = 50.0
    amplitude: float = 1.0
    polarisation: str = "z"


SINGLE_CASES = (
    InvarianceCase("baseline"),
    InvarianceCase("reference_25_ohm", resistance=25.0),
    InvarianceCase("reference_75_ohm", resistance=75.0),
    InvarianceCase("amplitude_0p25", amplitude=0.25),
    InvarianceCase("amplitude_2", amplitude=2.0),
    InvarianceCase("orientation_x", polarisation="x"),
    InvarianceCase("orientation_y", polarisation="y"),
)


def _axis_point(axis, value):
    """Return a point on the selected axis through the cubic-domain centre."""

    point = [DOMAIN[0] / 2, DOMAIN[1] / 2, DOMAIN[2] / 2]
    point["xyz".index(axis)] = value
    return tuple(point)


def build_scene(case):
    """Build one rotated 150 mm, one-edge-feed thin-wire dipole."""

    scene = gprMax.Scene()
    scene.add(gprMax.Title(name=f"Voltage-port GPU invariance: {case.name}"))
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    scene.add(gprMax.Domain(p1=DOMAIN))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW))
    scene.add(gprMax.PMLThickness(thickness=12))
    scene.add(gprMax.OMPThreads(1))
    scene.add(
        gprMax.Waveform(
            wave_type="gaussian",
            amp=case.amplitude,
            freq=1e9,
            id="pulse",
        )
    )
    feed = _axis_point(case.polarisation, 0.108)
    scene.add(
        gprMax.Edge(
            p1=_axis_point(case.polarisation, 0.034),
            p2=feed,
            material_id="pec",
        )
    )
    scene.add(
        gprMax.Edge(
            p1=_axis_point(case.polarisation, 0.110),
            p2=_axis_point(case.polarisation, 0.184),
            material_id="pec",
        )
    )
    scene.add(
        gprMax.VoltageSource(
            p1=feed,
            polarisation=case.polarisation,
            resistance=case.resistance,
            waveform_id="pulse",
            id="feed",
        )
    )
    return scene


def run_models(gpu):
    """Run single-precision cases together, then a double baseline."""

    gprMax.run(
        scenes=[build_scene(case) for case in SINGLE_CASES],
        n=len(SINGLE_CASES),
        outputfile=SINGLE_BASE,
        gpu=[gpu],
        gpu_precision="single",
        hide_progress_bars=True,
        log_level=logging.INFO,
    )
    gprMax.run(
        scenes=[build_scene(InvarianceCase("baseline_double"))],
        outputfile=DOUBLE_BASE,
        gpu=[gpu],
        gpu_precision="double",
        hide_progress_bars=True,
        log_level=logging.INFO,
    )


def _single_path(index):
    return SINGLE_BASE.parent / f"{SINGLE_BASE.name}{index + 1}.h5"


def read_port(path):
    """Read corrected port quantities and validity masks from HDF5."""

    with h5py.File(path, "r") as output:
        port = output["ports/feed"]
        return {
            "frequency": np.asarray(port["frequency"], dtype=np.float64),
            "s11": np.asarray(port["S11"], dtype=np.complex128),
            "zin": np.asarray(port["Zin"], dtype=np.complex128),
            "valid": (
                np.asarray(port["valid_S11"], dtype=bool)
                & np.asarray(port["valid_Zin"], dtype=bool)
            ),
            "tail_relative_db": float(port.attrs["TailRelativeLevelDB"]),
        }


def _error_metrics(values, reference):
    difference = values - reference
    return {
        "maximum_absolute": float(np.max(np.abs(difference))),
        "rms_absolute": float(np.sqrt(np.mean(np.abs(difference) ** 2))),
        "rms_relative": float(
            np.sqrt(np.mean(np.abs(difference) ** 2))
            / max(np.sqrt(np.mean(np.abs(reference) ** 2)), np.finfo(float).tiny)
        ),
    }


def analyse():
    """Calculate invariant-network, rotation, precision, and passivity errors."""

    single = {case.name: read_port(_single_path(index)) for index, case in enumerate(SINGLE_CASES)}
    double = read_port(DOUBLE_BASE.with_suffix(".h5"))
    baseline = single["baseline"]
    for result in (*single.values(), double):
        np.testing.assert_allclose(result["frequency"], baseline["frequency"], rtol=2e-6, atol=1)

    valid = (baseline["frequency"] >= FREQUENCY_BAND[0]) & (
        baseline["frequency"] <= FREQUENCY_BAND[1]
    )
    for result in (*single.values(), double):
        valid &= result["valid"]
    if np.count_nonzero(valid) < 5:
        raise RuntimeError("Too few mutually valid frequency bins for invariance checks")

    frequency = baseline["frequency"][valid]
    baseline_zin = baseline["zin"][valid]
    metrics = {
        "frequency_points": int(frequency.size),
        "frequency_range_hz": [float(frequency[0]), float(frequency[-1])],
        "tail_relative_level_db": {
            name: result["tail_relative_db"] for name, result in single.items()
        },
        "reference_impedance": {},
        "source_amplitude": {},
        "orientation": {},
        "gpu_precision": {},
        "passivity": {},
    }

    for case in SINGLE_CASES[1:3]:
        result = single[case.name]
        expected_s11 = (baseline_zin - case.resistance) / (baseline_zin + case.resistance)
        metrics["reference_impedance"][case.name] = {
            "zin_error_ohm": _error_metrics(result["zin"][valid], baseline_zin),
            "s11_transformation_error": _error_metrics(result["s11"][valid], expected_s11),
        }

    for case in SINGLE_CASES[3:5]:
        result = single[case.name]
        metrics["source_amplitude"][case.name] = {
            "zin_error_ohm": _error_metrics(result["zin"][valid], baseline_zin),
            "s11_error": _error_metrics(result["s11"][valid], baseline["s11"][valid]),
        }

    for case in SINGLE_CASES[5:]:
        result = single[case.name]
        metrics["orientation"][case.name] = {
            "zin_error_ohm": _error_metrics(result["zin"][valid], baseline_zin),
            "s11_error": _error_metrics(result["s11"][valid], baseline["s11"][valid]),
        }

    metrics["gpu_precision"] = {
        "zin_error_ohm": _error_metrics(double["zin"][valid], baseline_zin),
        "s11_error": _error_metrics(double["s11"][valid], baseline["s11"][valid]),
    }
    for name, result in {**single, "baseline_double": double}.items():
        metrics["passivity"][name] = {
            "maximum_abs_s11": float(np.max(np.abs(result["s11"][valid]))),
            "minimum_real_zin_ohm": float(np.min(result["zin"][valid].real)),
        }
    return frequency, valid, single, double, metrics


def create_plot(frequency, valid, single, double):
    """Plot the impedance and differences used by the invariant checks."""

    frequency_ghz = frequency / 1e9
    baseline = single["baseline"]
    baseline_zin = baseline["zin"][valid]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for name in ("baseline", "reference_25_ohm", "reference_75_ohm"):
        axes[0, 0].plot(frequency_ghz, single[name]["zin"][valid].real, label=name)
        axes[0, 1].plot(frequency_ghz, single[name]["zin"][valid].imag, label=name)
    axes[0, 0].set_ylabel(r"Re($Z_\mathrm{in}$) [$\Omega$]")
    axes[0, 1].set_ylabel(r"Im($Z_\mathrm{in}$) [$\Omega$]")

    comparison_names = (
        "amplitude_0p25",
        "amplitude_2",
        "orientation_x",
        "orientation_y",
    )
    for name in comparison_names:
        axes[1, 0].semilogy(
            frequency_ghz,
            np.maximum(np.abs(single[name]["zin"][valid] - baseline_zin), 1e-15),
            label=name,
        )
    axes[1, 0].semilogy(
        frequency_ghz,
        np.maximum(np.abs(double["zin"][valid] - baseline_zin), 1e-15),
        label="double precision",
    )
    axes[1, 0].set_ylabel(r"$|\Delta Z_\mathrm{in}|$ [$\Omega$]")

    for case in SINGLE_CASES[1:3]:
        expected = (baseline_zin - case.resistance) / (baseline_zin + case.resistance)
        axes[1, 1].semilogy(
            frequency_ghz,
            np.maximum(np.abs(single[case.name]["s11"][valid] - expected), 1e-15),
            label=case.name,
        )
    axes[1, 1].set_ylabel(r"$|S_{11}-S_{11}(Z_\mathrm{in},Z_0)|$")

    for axis in axes.flat:
        axis.set_xlabel("Frequency [GHz]")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Voltage-port CUDA invariance and precision checks")
    fig.tight_layout()
    output = RESULTS_DIR / "rx_port_gpu_invariance.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="analyse existing HDF5 outputs without rerunning gprMax",
    )
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.postprocess_only:
        run_models(args.gpu)
    frequency, valid, single, double, metrics = analyse()
    metrics_path = RESULTS_DIR / "rx_port_gpu_invariance_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    plot_path = create_plot(frequency, valid, single, double)
    print(f"Saved invariance plot: {plot_path}")
    print(f"Saved invariance metrics: {metrics_path}")


if __name__ == "__main__":
    main()
