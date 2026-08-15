"""Compare impulse-synthesised pulses with separate direct FDTD runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np

import gprMax
from toolboxes.ImpulseResponse import (
    load_source_sampling,
    sample_builtin_waveform,
    synthesise_output,
)
from toolboxes.SFCW.processing import load_receiver, load_source

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WAVEFORMS = (
    ("ricker", 500e6, "ricker_500MHz"),
    ("gaussian", 350e6, "gaussian_350MHz"),
    ("gaussiandotnorm", 450e6, "gaussiandotnorm_450MHz"),
)


def build_scene(waveform_type, frequency):
    """Build the same scattering problem with a selected source waveform."""

    inf = float("inf")
    scene = gprMax.Scene()
    scene.add(gprMax.Title(name=f"Waveform synthesis validation: {waveform_type}"))
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(0.005, 0.005, 0.005)))
    scene.add(gprMax.Domain(p1=(0.2, 0.2, inf)))
    scene.add(gprMax.TimeWindow(time=30e-9))
    scene.add(gprMax.OMPThreads(n=2))
    scene.add(gprMax.Waveform(wave_type=waveform_type, amp=1, freq=frequency, id="source"))
    scene.add(
        gprMax.HertzianDipole(
            polarisation="z",
            p1=(0.08, 0.14, inf),
            waveform_id="source",
        )
    )
    scene.add(gprMax.Rx(p1=(0.12, 0.14, inf), id="response", outputs=["Ez"]))
    scene.add(
        gprMax.Cylinder(
            p1=(0.10, 0.07, 0),
            p2=(0.10, 0.07, inf),
            r=0.02,
            material_id="pec",
        )
    )
    return scene


def run_model(output_stem, waveform_type, frequency, precision, reuse):
    output = Path(output_stem).with_suffix(".h5")
    if reuse and output.exists():
        return output, 0.0
    start = perf_counter()
    gprMax.run(
        scenes=[build_scene(waveform_type, frequency)],
        n=1,
        outputfile=output_stem,
        hide_progress_bars=True,
        log_level=30,
        cpu_precision=precision,
    )
    return output, perf_counter() - start


def _plot(output, comparisons):
    figure, axes = plt.subplots(
        len(comparisons),
        3,
        figsize=(14, 9),
        constrained_layout=True,
    )
    for row, comparison in enumerate(comparisons):
        target = comparison["target"]
        direct_source = comparison["direct_source"]
        direct_receiver = comparison["direct_receiver"]
        synthesised = comparison["synthesised"]
        source_time = direct_source.times * 1e9
        receiver_time = direct_receiver.times * 1e9
        peak = np.max(np.abs(direct_receiver.samples))

        axes[row, 0].plot(source_time, direct_source.samples, "k-", label="Direct FDTD input")
        axes[row, 0].plot(source_time, target.samples, "--", label="Toolbox samples")
        axes[row, 1].plot(receiver_time, direct_receiver.samples, "k-", label="Direct FDTD")
        axes[row, 1].plot(receiver_time, synthesised, "--", label="Impulse synthesis")
        axes[row, 2].plot(
            receiver_time,
            np.abs(synthesised - direct_receiver.samples) / peak,
            "k-",
        )
        axes[row, 0].set_ylabel(comparison["id"])
        axes[row, 2].set_yscale("log")
        axes[row, 2].set_ylim(1e-17, 1e-11)
        for column in range(3):
            axes[row, column].grid(True, alpha=0.3)
    axes[0, 0].set_title("Source waveform")
    axes[0, 1].set_title("Receiver response")
    axes[0, 2].set_title("Absolute error / direct peak")
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (ns)")
    axes[0, 0].legend()
    axes[0, 1].legend()
    figure.suptitle("Impulse-response synthesis versus independent direct FDTD runs")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run_validation(output_dir, precision="double", reuse=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "_cache"
    cache.mkdir(exist_ok=True)
    impulse_file, impulse_runtime = run_model(cache / "impulse", "impulse", 1.0, precision, reuse)
    impulse_source = load_source_sampling(impulse_file)
    comparisons = []
    metrics = {
        "method": "causal discrete convolution of a stored one-sample FDTD impulse response",
        "precision": precision,
        "impulse_runtime_seconds": impulse_runtime,
        "waveforms": {},
    }

    for waveform_type, frequency, identifier in WAVEFORMS:
        direct_file, runtime = run_model(
            cache / identifier,
            waveform_type,
            frequency,
            precision,
            reuse,
        )
        target = sample_builtin_waveform(
            impulse_source,
            waveform_type,
            1.0,
            frequency,
            identifier,
        )
        result = synthesise_output(impulse_file, target)
        direct_source = load_source(direct_file)
        direct_receiver = load_receiver(direct_file)
        synthesised = result.receivers[0].samples
        peak = float(np.max(np.abs(direct_receiver.samples)))
        source_peak = float(np.max(np.abs(direct_source.samples)))
        maximum_relative_error = float(np.max(np.abs(synthesised - direct_receiver.samples)) / peak)
        source_relative_error = float(
            np.max(np.abs(target.samples - direct_source.samples)) / source_peak
        )
        relative_l2_error = float(
            np.linalg.norm(synthesised - direct_receiver.samples)
            / np.linalg.norm(direct_receiver.samples)
        )
        metrics["waveforms"][identifier] = {
            "type": waveform_type,
            "frequency_hz": frequency,
            "direct_runtime_seconds": runtime,
            "source_maximum_relative_error": source_relative_error,
            "receiver_maximum_relative_error": maximum_relative_error,
            "receiver_relative_l2_error": relative_l2_error,
        }
        comparisons.append(
            {
                "id": identifier,
                "target": target,
                "direct_source": direct_source,
                "direct_receiver": direct_receiver,
                "synthesised": synthesised,
            }
        )

    with (output_dir / "metrics.json").open("w") as stream:
        json.dump(metrics, stream, indent=2)
    _plot(output_dir / "waveform_synthesis_validation.png", comparisons)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results"),
    )
    parser.add_argument("--precision", choices=("single", "double"), default="double")
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_validation(args.output_dir, args.precision, args.reuse), indent=2))


if __name__ == "__main__":
    main()
