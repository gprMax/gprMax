"""Validate broadband reflection from a lossless PEC-backed dielectric slab.

An axial discrete plane wave is sampled inside its TFSF region.  A free-space
run supplies the incident field and a second run contains the dielectric slab
and PEC backing.  Their difference is the reflected field.  The FFT ratio is
de-embedded to the front face using the axial Yee-grid wavenumber and compared
with the exact input impedance of a short-circuited transmission-line layer.

At normal incidence TE and TM are degenerate.  The complementary oblique TE
and TM checks are supplied by ``validate_grounded_dipoles.py``, whose E- and
H-plane cuts exercise the two coefficients separately.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.constants import c

import gprMax

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "grounded_slab_reflection"
DL = 0.25e-3
DOMAIN = (0.16, 0.02, float("inf"))
PML_CELLS = 12
TFSF_P1 = (0.0125, 0.006, float("inf"))
TFSF_P2 = (0.1475, 0.014, float("inf"))
RECEIVER = (0.0725, 0.010, float("inf"))
INTERFACE = 0.090
GROUND = 0.102
ER_SLAB = 4.0
TIME_WINDOW = 20e-9
SOURCE_FREQUENCY = 3e9
FREQUENCY_MIN = 0.4e9
FREQUENCY_MAX = 7.0e9
SPECTRUM_THRESHOLD = 1e-3
ACCEPTANCE_LIMITS = {
    "magnitude_maximum_error": 1e-6,
    "phase_maximum_error_degrees": 0.1,
    "complex_relative_l2_error": 1e-3,
}


def build_scene(grounded: bool) -> gprMax.Scene:
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(DL,) * 3))
    scene.add(gprMax.Domain(p1=DOMAIN))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW))
    scene.add(gprMax.PMLThickness(thickness=(PML_CELLS, PML_CELLS, 0, PML_CELLS, PML_CELLS, 0)))
    if grounded:
        scene.add(gprMax.Material(er=ER_SLAB, se=0, mr=1, sm=0, id="slab"))
        scene.add(
            gprMax.Box(
                p1=(INTERFACE, 0, float("inf")),
                p2=(GROUND, DOMAIN[1], float("inf")),
                material_id="slab",
            )
        )
        scene.add(
            gprMax.Box(
                p1=(GROUND, 0, float("inf")),
                p2=DOMAIN,
                material_id="pec",
            )
        )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=SOURCE_FREQUENCY, id="pulse"))
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


def _run(path: Path, grounded: bool, no_run: bool, gpu: int | None):
    if not no_run:
        options = {"cpu_precision": "double"}
        if gpu is not None:
            options = {"gpu": [gpu], "gpu_precision": "double"}
        gprMax.run(
            scenes=[build_scene(grounded)],
            n=1,
            outputfile=path.with_suffix(""),
            hide_progress_bars=True,
            **options,
        )
    with h5py.File(path, "r") as output:
        return np.asarray(output["rxs/rx1/Ez"]), float(output.attrs["dt"])


def _axial_numerical_wavenumber(frequencies: np.ndarray, dt: float) -> np.ndarray:
    argument = np.sin(np.pi * frequencies * dt) * DL / (c * dt)
    if np.any(np.abs(argument) > 1 + 1e-12):
        raise ValueError("selected frequencies are outside the propagating Yee band")
    return 2 * np.arcsin(np.clip(argument, -1, 1)) / DL


def _analytical_reflection(frequencies: np.ndarray) -> np.ndarray:
    wavenumber = 2 * np.pi * frequencies / c
    impedance = 1 / np.sqrt(ER_SLAB)
    phase = wavenumber * np.sqrt(ER_SLAB) * (GROUND - INTERFACE)
    input_impedance = 1j * impedance * np.tan(phase)
    return (input_impedance - 1) / (input_impedance + 1)


def analyse(incident: np.ndarray, total: np.ndarray, dt: float):
    reflected = total - incident
    frequencies = np.fft.rfftfreq(incident.size, dt)
    incident_fft = np.fft.rfft(incident)
    reflected_fft = np.fft.rfft(reflected)
    selected = (
        (frequencies >= FREQUENCY_MIN)
        & (frequencies <= FREQUENCY_MAX)
        & (np.abs(incident_fft) >= SPECTRUM_THRESHOLD * np.max(np.abs(incident_fft)))
    )
    frequencies = frequencies[selected]
    receiver_ratio = reflected_fft[selected] / incident_fft[selected]
    distance = INTERFACE - RECEIVER[0]
    numerical_wavenumber = _axial_numerical_wavenumber(frequencies, dt)
    fdtd = receiver_ratio * np.exp(2j * numerical_wavenumber * distance)
    analytical = _analytical_reflection(frequencies)
    phase_error = np.angle(fdtd / analytical, deg=True)
    magnitude_error = np.abs(fdtd) - np.abs(analytical)
    return {
        "frequencies": frequencies,
        "fdtd": fdtd,
        "analytical": analytical,
        "reflected": reflected,
        "magnitude_rmse": float(np.sqrt(np.mean(magnitude_error**2))),
        "magnitude_maximum_error": float(np.max(np.abs(magnitude_error))),
        "phase_rmse_degrees": float(np.sqrt(np.mean(phase_error**2))),
        "phase_maximum_error_degrees": float(np.max(np.abs(phase_error))),
        "complex_relative_l2_error": float(np.linalg.norm(fdtd - analytical) / np.linalg.norm(analytical)),
    }


def _write_outputs(result, incident, total, dt, output_directory: Path) -> None:
    with (output_directory / "grounded_slab_reflection.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "frequency_hz",
                "fdtd_magnitude",
                "fdtd_phase_degrees",
                "analytical_magnitude",
                "analytical_phase_degrees",
            )
        )
        for frequency, fdtd, analytical in zip(result["frequencies"], result["fdtd"], result["analytical"]):
            writer.writerow(
                (
                    frequency,
                    abs(fdtd),
                    np.angle(fdtd, deg=True),
                    abs(analytical),
                    np.angle(analytical, deg=True),
                )
            )

    frequency_ghz = result["frequencies"] / 1e9
    fig, (magnitude_axis, phase_axis) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    residual_scale = 1e8
    magnitude_axis.plot(frequency_ghz, np.zeros_like(frequency_ghz), "k-", label="analytical")
    magnitude_axis.plot(
        frequency_ghz,
        residual_scale * (np.abs(result["fdtd"]) - 1),
        "ko",
        markerfacecolor="white",
        markersize=3,
        label="gprMax FDTD",
    )
    analytical_phase = np.unwrap(np.angle(result["analytical"])) * 180 / np.pi
    fdtd_phase = np.unwrap(np.angle(result["fdtd"])) * 180 / np.pi
    phase_axis.plot(frequency_ghz, analytical_phase, "k-")
    phase_axis.plot(frequency_ghz, fdtd_phase, "ko", markerfacecolor="white", markersize=3)
    magnitude_axis.set_ylabel(r"$(|\Gamma|-1)\times 10^8$")
    phase_axis.set_ylabel(r"phase of $\Gamma$ [degrees]")
    phase_axis.set_xlabel("frequency [GHz]")
    magnitude_axis.legend()
    for axis in (magnitude_axis, phase_axis):
        axis.grid(True, alpha=0.3)
    fig.suptitle(r"Normal-incidence reflection from a $\epsilon_r=4$, 12 mm PEC-backed slab")
    fig.tight_layout()
    fig.savefig(output_directory / "grounded_slab_reflection.png", dpi=220)
    plt.close(fig)

    times = np.arange(incident.size) * dt * 1e9
    shown = times <= 4
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(times[shown], incident[shown], "k-", label="incident")
    axis.plot(times[shown], total[shown], "k--", label="total")
    axis.plot(times[shown], result["reflected"][shown], color="0.5", label="total - incident")
    axis.set_xlabel("time [ns]")
    axis.set_ylabel(r"$E_z$ [V/m]")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_directory / "grounded_slab_time_traces.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)

    incident, dt = _run(args.output_directory / "free_space.h5", False, args.no_run, args.gpu)
    total, total_dt = _run(args.output_directory / "grounded_slab.h5", True, args.no_run, args.gpu)
    if not np.isclose(dt, total_dt, rtol=1e-12, atol=0):
        raise RuntimeError("reference and grounded-slab time steps differ")
    result = analyse(incident, total, dt)
    _write_outputs(result, incident, total, dt, args.output_directory)
    metrics = {
        key: result[key]
        for key in (
            "magnitude_rmse",
            "magnitude_maximum_error",
            "phase_rmse_degrees",
            "phase_maximum_error_degrees",
            "complex_relative_l2_error",
        )
    }
    checks = {
        key: {
            "value": metrics[key],
            "maximum": maximum,
            "passed": metrics[key] <= maximum,
        }
        for key, maximum in ACCEPTANCE_LIMITS.items()
    }
    summary = {
        "model": {
            "dl_metres": DL,
            "slab_relative_permittivity": ER_SLAB,
            "slab_thickness_metres": GROUND - INTERFACE,
            "frequency_band_hz": [FREQUENCY_MIN, FREQUENCY_MAX],
            "frequency_samples": int(result["frequencies"].size),
        },
        "metrics": metrics,
        "acceptance": {"passed": all(item["passed"] for item in checks.values()), "checks": checks},
    }
    (args.output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["acceptance"]["passed"]:
        raise SystemExit("grounded-slab reflection validation failed")


if __name__ == "__main__":
    main()
