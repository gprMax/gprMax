"""Run a paper-inspired analytical FMCW multilayer validation.

The paper used a compact broadband FDTD excitation, an empty-model
subtraction, source deconvolution, a Blackman sweep taper, and a RIMFAX-like
150--1200 MHz band. This canonical validation uses the same processing
sequence for a normal-incidence 2D plane wave, permitting a direct complex
comparison with the exact multilayer reflection coefficient. It does not
reproduce the paper's 3D bistatic antenna geometry or instrument response.

Example::

    python -m testing.validation.fmcw.validate_paper_multilayer --gpu 0
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import h5py
import matplotlib
import numpy as np
from scipy.constants import c

import gprMax
from toolboxes.FMCW import (
    Chirp,
    process_incident_referenced_channel,
    reconstruct_fast_time,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

DL = 1e-3
DOMAIN = (0.30, 0.030, float("inf"))
TIME_WINDOW = 60e-9
PML_CELLS = 12
TFSF_P1 = (0.015, 0.013, float("inf"))
TFSF_P2 = (0.285, 0.017, float("inf"))
RECEIVER = (0.075, 0.015, float("inf"))
INTERFACE = 0.120
LAYERS = ((4.1, 0.035), (2.7, 0.040))
SUBSTRATE_ER = 6.0
CHIRP = Chirp(150e6, 1.2e9, 100e-3, 1024)


def build_scene(target: bool, threads: int):
    """Build the empty reference or lossless multilayer model."""

    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    scene.add(gprMax.Domain(p1=DOMAIN))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW))
    scene.add(gprMax.OMPThreads(n=threads))
    scene.add(gprMax.PMLThickness(thickness=(PML_CELLS, PML_CELLS, 0, PML_CELLS, PML_CELLS, 0)))

    if target:
        media = (("layer1", LAYERS[0][0]), ("layer2", LAYERS[1][0]), ("substrate", SUBSTRATE_ER))
        for material_id, relative_permittivity in media:
            scene.add(
                gprMax.Material(
                    er=relative_permittivity,
                    se=0,
                    mr=1,
                    sm=0,
                    id=material_id,
                )
            )
        first_end = INTERFACE + LAYERS[0][1]
        second_end = first_end + LAYERS[1][1]
        scene.add(gprMax.Box(p1=(INTERFACE, 0, float("inf")), p2=DOMAIN, material_id="layer1"))
        scene.add(gprMax.Box(p1=(first_end, 0, float("inf")), p2=DOMAIN, material_id="layer2"))
        scene.add(gprMax.Box(p1=(second_end, 0, float("inf")), p2=DOMAIN, material_id="substrate"))

    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=675e6, id="pulse"))
    scene.add(
        gprMax.DiscretePlaneWaveAxial(
            p1=TFSF_P1,
            p2=TFSF_P2,
            axis="x",
            psi=90,
            waveform_id="pulse",
        )
    )
    scene.add(gprMax.Rx(p1=RECEIVER, id="probe", outputs=["Ez"]))
    return scene


def run_model(target, output_base, threads, precision, gpu, reuse):
    h5_path = output_base.with_suffix(".h5")
    if reuse and h5_path.exists():
        return h5_path, 0.0
    options = {"cpu_precision": precision} if gpu is None else {"gpu": [gpu], "gpu_precision": precision}
    started = perf_counter()
    gprMax.run(
        scenes=[build_scene(target, threads)],
        n=1,
        outputfile=output_base,
        hide_progress_bars=True,
        log_level=logging.WARNING,
        **options,
    )
    return h5_path, perf_counter() - started


def _multilayer_reflection(frequency):
    """Exact normal-incidence reflection at the first interface."""

    frequency = np.asarray(frequency, dtype=np.float64)
    er = (1.0, LAYERS[0][0], LAYERS[1][0], SUBSTRATE_ER)
    impedance = [1 / np.sqrt(value) for value in er]
    interface = [
        (impedance[index + 1] - impedance[index]) / (impedance[index + 1] + impedance[index]) for index in range(3)
    ]
    reflection = np.full(frequency.shape, interface[-1], dtype=np.complex128)
    for index in (1, 0):
        wavenumber = 2 * np.pi * frequency * np.sqrt(er[index + 1]) / c
        propagation = np.exp(-2j * wavenumber * LAYERS[index][1])
        reflection = (interface[index] + reflection * propagation) / (1 + interface[index] * reflection * propagation)
    distance = INTERFACE - RECEIVER[0]
    return reflection * np.exp(-2j * 2 * np.pi * frequency * distance / c)


def _first_trace(values):
    array = np.asarray(values)
    return array if array.ndim == 1 else array[:, 0]


def _plot(output, channel, analytic_channel, fast_fdtd, fast_analytic):
    target_trace = _first_trace(channel.target.receiver.samples)
    background_trace = _first_trace(channel.background.receiver.samples)
    dt = channel.target.receiver.dt
    time = np.arange(target_trace.size) * dt * 1e9
    display_time = time <= 8

    frequency = CHIRP.frequency / 1e6
    measured = _first_trace(channel.response)
    incident = _first_trace(channel.background.response)
    gamma_fdtd = measured / incident
    gamma_analytic = _first_trace(analytic_channel.response)
    envelope_fdtd = np.abs(_first_trace(fast_fdtd.complex_envelope))
    envelope_analytic = np.abs(_first_trace(fast_analytic.complex_envelope))
    delay_ns = fast_fdtd.delay * 1e9
    delay_mask = delay_ns <= 12

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(time[display_time], target_trace[display_time], label="Multilayer")
    axes[0, 0].plot(time[display_time], background_trace[display_time], "--", label="Empty model")
    axes[0, 0].set(title="Raw FDTD receiver fields", xlabel="Time (ns)", ylabel=r"$E_z$ (V/m)")
    axes[0, 0].legend()

    axes[0, 1].plot(frequency, np.abs(gamma_analytic), "k-", linewidth=2, label="Analytical")
    axes[0, 1].plot(frequency, np.abs(gamma_fdtd), "--", label="gprMax FMCW")
    axes[0, 1].set(title="Multilayer reflection", xlabel="Frequency (MHz)", ylabel=r"$|\Gamma|$")
    axes[0, 1].legend()

    axes[1, 0].plot(delay_ns[delay_mask], envelope_analytic[delay_mask], "k-", linewidth=2, label="Analytical")
    axes[1, 0].plot(delay_ns[delay_mask], envelope_fdtd[delay_mask], "--", label="gprMax FMCW")
    axes[1, 0].set(title="Blackman-windowed fast time", xlabel="Delay (ns)", ylabel="Envelope")
    axes[1, 0].legend()

    difference = np.abs(gamma_fdtd - gamma_analytic)
    axes[1, 1].plot(frequency, difference)
    axes[1, 1].set(
        title="Complex reflection error", xlabel="Frequency (MHz)", ylabel=r"$|\Gamma_{FDTD}-\Gamma_{exact}|$"
    )
    for axis in axes.ravel():
        axis.grid(True, alpha=0.3)
    figure.suptitle("Paper-inspired FMCW validation: analytical lossless multilayer")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run_validation(output_dir, threads=4, precision="double", gpu=None, reuse=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "_cache"
    cache.mkdir(exist_ok=True)
    target_file, target_runtime = run_model(True, cache / "multilayer", threads, precision, gpu, reuse)
    background_file, background_runtime = run_model(False, cache / "empty", threads, precision, gpu, reuse)
    channel = process_incident_referenced_channel(
        target_file,
        background_file,
        CHIRP,
        receiver_path="/rxs/rx1",
        component="Ez",
        tail_taper_fraction=0.05,
    )
    gamma_fdtd = _first_trace(channel.response)
    gamma_analytic = _multilayer_reflection(CHIRP.frequency)
    fdtd_channel = replace(channel, response=gamma_fdtd)
    analytic_channel = replace(channel, response=gamma_analytic)
    fast_fdtd = reconstruct_fast_time(fdtd_channel, window="blackman")
    fast_analytic = reconstruct_fast_time(analytic_channel, window="blackman")

    complex_l2 = float(np.linalg.norm(gamma_fdtd - gamma_analytic) / np.linalg.norm(gamma_analytic))
    magnitude_rmse = float(np.sqrt(np.mean((np.abs(gamma_fdtd) - np.abs(gamma_analytic)) ** 2)))
    envelope_l2 = float(
        np.linalg.norm(fast_fdtd.complex_envelope - fast_analytic.complex_envelope)
        / np.linalg.norm(fast_analytic.complex_envelope)
    )
    metrics = {
        "method": "Paper-inspired Eide et al. 2022 analytical multilayer validation",
        "frequency_band_hz": [CHIRP.f_start, CHIRP.f_stop],
        "frequency_samples": CHIRP.samples,
        "sweep_time_seconds": CHIRP.sweep_time,
        "grid_spacing_metres": DL,
        "precision": precision,
        "backend": "cpu" if gpu is None else f"cuda:{gpu}",
        "runtime_seconds": {"target": target_runtime, "background": background_runtime},
        "complex_relative_l2_error": complex_l2,
        "magnitude_rmse": magnitude_rmse,
        "fast_time_complex_relative_l2_error": envelope_l2,
    }
    with (output_dir / "metrics.json").open("w") as outfile:
        json.dump(metrics, outfile, indent=2)
    np.savez(
        output_dir / "comparison.npz",
        frequency=CHIRP.frequency,
        gamma_fdtd=gamma_fdtd,
        gamma_analytic=gamma_analytic,
        delay=fast_fdtd.delay,
        fast_fdtd=fast_fdtd.complex_envelope,
        fast_analytic=fast_analytic.complex_envelope,
    )
    _plot(
        output_dir / "fmcw_paper_multilayer_validation.png",
        channel,
        analytic_channel,
        fast_fdtd,
        fast_analytic,
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results"),
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--precision", choices=("single", "double"), default="double")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    metrics = run_validation(
        args.output_dir,
        threads=args.threads,
        precision=args.precision,
        gpu=args.gpu,
        reuse=args.reuse,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
