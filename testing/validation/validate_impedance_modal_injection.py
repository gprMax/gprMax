"""Validate direct FDFD modal injection into a surface-impedance guide.

The scene is a long, uniform rectangular guide whose four opaque walls use a
constant scalar surface impedance.  One active eigenmode port launches TE10;
two passive ports measure its subsequent propagation.  The primary metric is
the source-plane mismatch ``b1 / a1``.  The attenuation inferred from the two
passive planes is compared with first-order conductor-loss perturbation theory.

The impedance is deliberately 5 ohms rather than a good-metal value.  It is
small compared with the wave impedance, so perturbation theory remains useful,
but produces enough loss to measure over a compact FDTD propagation distance.
The common-metal preset path has separate unit and FDFD-alpha coverage.

The guide ends before the PML because impedance boundaries may not intersect a
PML.  Both wall ends are far enough from every measured plane that even a
signal travelling at the largest in-band TE10 group velocity cannot make a
round trip before the time window closes.  Consequently this is a source and
propagation benchmark, not a termination benchmark.

Example::

    python -m testing.validation.validate_impedance_modal_injection --threads 4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
from scipy.constants import c, epsilon_0, mu_0

import gprMax


DL = 0.5e-3
DOMAIN = (0.150, 0.012, 0.010)
PML_CELLS = 3
TIME_WINDOW = 0.35e-9
GUIDE_WIDTH = 0.006
GUIDE_HEIGHT = 0.004
GUIDE_LOWER = (0.003, 0.003)
GUIDE_UPPER = (0.009, 0.007)
WALL_OUTER_LOWER = (0.002, 0.002)
WALL_OUTER_UPPER = (0.010, 0.008)
WALL_X = (0.002, 0.148)
SOURCE_X = 0.060
NEAR_MONITOR_X = 0.075
FAR_MONITOR_X = 0.095
MONITOR_SPACING = FAR_MONITOR_X - NEAR_MONITOR_X
FMIN = 45e9
FMAX = 65e9
DFT_POINTS = 11
ANCHORS = (30e9, 45e9, 55e9, 65e9, 81e9)
SURFACE_RESISTANCE = 5.0
MODEL_ID = "modal_wall"
MAX_SOURCE_REFLECTION_DB = -20.0
MAX_ALPHA_RELATIVE_L2_ERROR = 0.12
ETA0 = np.sqrt(mu_0 / epsilon_0)


def te10_cutoff() -> float:
    """Return the continuum TE10 cutoff frequency for the benchmark guide."""

    return c / (2 * GUIDE_WIDTH)


def te10_group_velocity(frequency_hz) -> np.ndarray:
    """Return the lossless continuum TE10 group velocity."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    cutoff = te10_cutoff()
    if np.any(frequency <= cutoff):
        raise ValueError("TE10 group velocity requires frequencies above cutoff")
    return c * np.sqrt(1 - np.square(cutoff / frequency))


def perturbation_alpha(frequency_hz, surface_resistance=SURFACE_RESISTANCE) -> np.ndarray:
    """Return first-order TE10 conductor attenuation in nepers per metre.

    For guide width ``a`` and height ``b``, the transverse-resonance/energy
    perturbation result is

    ``alpha = Rs/eta0 * [k/(beta*b) + 2*kc**2/(k*beta*a)]``.
    """

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    resistance = float(surface_resistance)
    if not np.isfinite(resistance) or resistance < 0:
        raise ValueError("surface resistance must be finite and non-negative")
    k = 2 * np.pi * frequency / c
    kc = np.pi / GUIDE_WIDTH
    if np.any(k <= kc):
        raise ValueError("TE10 attenuation requires frequencies above cutoff")
    beta = np.sqrt(np.square(k) - kc**2)
    return resistance / ETA0 * (
        k / (beta * GUIDE_HEIGHT)
        + 2 * kc**2 / (k * beta * GUIDE_WIDTH)
    )


def earliest_wall_end_round_trip() -> float:
    """Return the earliest in-band wall-end return to any measured plane."""

    distances = (
        SOURCE_X - WALL_X[0],
        WALL_X[1] - SOURCE_X,
        NEAR_MONITOR_X - WALL_X[0],
        WALL_X[1] - NEAR_MONITOR_X,
        FAR_MONITOR_X - WALL_X[0],
        WALL_X[1] - FAR_MONITOR_X,
    )
    return 2 * min(distances) / float(te10_group_velocity(FMAX))


def _wall_boxes() -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
    """Return four boxes whose union is a rectangular guide wall."""

    x0, x1 = WALL_X
    y0, z0 = WALL_OUTER_LOWER
    y1, z1 = WALL_OUTER_UPPER
    yi0, zi0 = GUIDE_LOWER
    yi1, zi1 = GUIDE_UPPER
    return (
        ((x0, y0, z0), (x1, yi0, z1)),
        ((x0, yi1, z0), (x1, y1, z1)),
        ((x0, yi0, z0), (x1, yi1, zi0)),
        ((x0, yi0, zi1), (x1, yi1, z1)),
    )


def build_scene(threads: int = 1) -> gprMax.Scene:
    """Return the causally isolated three-plane TE10 benchmark scene."""

    if earliest_wall_end_round_trip() <= TIME_WINDOW:
        raise RuntimeError("benchmark geometry no longer isolates wall-end returns")
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=DOMAIN))
    scene.add(gprMax.Discretisation(p1=(DL, DL, DL)))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW))
    scene.add(gprMax.PMLThickness(thickness=PML_CELLS))
    scene.add(gprMax.OMPThreads(n=threads))
    scene.add(
        gprMax.SurfaceImpedance(
            id=MODEL_ID,
            resistance=SURFACE_RESISTANCE,
            fit_fmin_hz=FMIN,
            fit_fmax_hz=FMAX,
        )
    )
    for lower, upper in _wall_boxes():
        scene.add(gprMax.ImpedanceBox(lower, upper, MODEL_ID))

    scene.add(
        gprMax.EigenmodeBand(
            id="impedance_te10",
            fmin=FMIN,
            fmax=FMAX,
            points=DFT_POINTS,
        )
    )
    for port, x, direction in (
        (1, SOURCE_X, "+"),
        (2, NEAR_MONITOR_X, "-"),
        (3, FAR_MONITOR_X, "-"),
    ):
        scene.add(
            gprMax.EigenmodePort(
                port=port,
                p1=(x, GUIDE_LOWER[0], GUIDE_LOWER[1]),
                p2=(x, GUIDE_UPPER[0], GUIDE_UPPER[1]),
                direction=direction,
                modes=(1,),
                anchors=ANCHORS,
                plot_fields=False,
            )
        )
    scene.add(
        gprMax.EigenmodeExcitation(
            port=1,
            mode=1,
            waveform="auto",
            plot_waveform=False,
        )
    )
    return scene


def magnitude_db(values) -> np.ndarray:
    """Return a finite-safe voltage-wave magnitude in decibels."""

    magnitude = np.maximum(np.abs(values), np.finfo(np.float64).tiny)
    return 20 * np.log10(magnitude)


def analyse_modal_coefficients(
    frequency_hz,
    source_incident,
    source_outgoing,
    near_outgoing,
    far_outgoing,
) -> dict[str, np.ndarray | float]:
    """Calculate injection mismatch and two-plane attenuation metrics."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    a1 = np.asarray(source_incident, dtype=np.complex128)
    b1 = np.asarray(source_outgoing, dtype=np.complex128)
    b2 = np.asarray(near_outgoing, dtype=np.complex128)
    b3 = np.asarray(far_outgoing, dtype=np.complex128)
    if not (frequency.ndim == a1.ndim == b1.ndim == b2.ndim == b3.ndim == 1):
        raise ValueError("modal coefficient inputs must be one-dimensional")
    if not (frequency.shape == a1.shape == b1.shape == b2.shape == b3.shape):
        raise ValueError("modal coefficient inputs must have identical shapes")
    if np.any(~np.isfinite(frequency)) or np.any(frequency <= te10_cutoff()):
        raise ValueError("all modal frequencies must be finite and above TE10 cutoff")
    if np.any(~np.isfinite(a1)) or np.any(~np.isfinite(b1)):
        raise ValueError("source modal coefficients must be finite")
    if np.any(~np.isfinite(b2)) or np.any(~np.isfinite(b3)):
        raise ValueError("passive-port modal coefficients must be finite")
    if np.any(np.abs(a1) == 0) or np.any(np.abs(b2) == 0):
        raise ValueError("incident and near-monitor modal coefficients must be non-zero")

    reflection = b1 / a1
    propagation_ratio = b3 / b2
    measured_alpha = -np.log(np.abs(propagation_ratio)) / MONITOR_SPACING
    analytical_alpha = perturbation_alpha(frequency)
    alpha_error = measured_alpha - analytical_alpha
    return {
        "frequency_hz": frequency,
        "source_reflection": reflection,
        "source_reflection_db": magnitude_db(reflection),
        "propagation_ratio": propagation_ratio,
        "measured_alpha_per_m": measured_alpha,
        "analytical_alpha_per_m": analytical_alpha,
        "alpha_error_per_m": alpha_error,
        "maximum_source_reflection_db": float(np.max(magnitude_db(reflection))),
        "rms_source_reflection": float(np.sqrt(np.mean(np.abs(reflection) ** 2))),
        "alpha_relative_l2_error": float(
            np.linalg.norm(alpha_error) / np.linalg.norm(analytical_alpha)
        ),
    }


def _read_port(path: Path, port: int):
    with h5py.File(path, "r") as data:
        group = data[f"eigenmode_ports/port{port}"]
        frequency = np.asarray(group["frequency"], dtype=np.float64)
        incident = np.asarray(group["incident"])[0]
        outgoing = np.asarray(group["outgoing"])[0]
        valid_name = "power_wave_valid" if "power_wave_valid" in group else "valid"
        valid = np.asarray(group[valid_name], dtype=bool)[0]
    return frequency, incident, outgoing, valid


def analyse_output(path: Path) -> dict[str, np.ndarray | float]:
    """Read one gprMax HDF5 output and calculate benchmark metrics."""

    ports = [_read_port(path, port) for port in (1, 2, 3)]
    frequency = ports[0][0]
    for other_frequency, _, _, _ in ports[1:]:
        np.testing.assert_array_equal(other_frequency, frequency)
    valid = np.logical_and.reduce([item[3] for item in ports])
    valid &= (frequency >= FMIN) & (frequency <= FMAX)
    if not np.all(valid):
        invalid = frequency[~valid]
        raise RuntimeError(
            "benchmark has invalid TE10 power-wave coefficients at "
            + ", ".join(f"{value:g}" for value in invalid)
            + " Hz"
        )
    return analyse_modal_coefficients(
        frequency,
        ports[0][1],
        ports[0][2],
        ports[1][2],
        ports[2][2],
    )


def _write_csv(path: Path, result: dict) -> None:
    table = np.column_stack(
        (
            result["frequency_hz"],
            np.real(result["source_reflection"]),
            np.imag(result["source_reflection"]),
            result["source_reflection_db"],
            np.real(result["propagation_ratio"]),
            np.imag(result["propagation_ratio"]),
            result["measured_alpha_per_m"],
            result["analytical_alpha_per_m"],
            result["alpha_error_per_m"],
        )
    )
    np.savetxt(
        path,
        table,
        delimiter=",",
        header=(
            "frequency_hz,S11_real,S11_imag,S11_magnitude_db,"
            "two_plane_ratio_real,two_plane_ratio_imag,measured_alpha_per_m,"
            "analytical_alpha_per_m,alpha_error_per_m"
        ),
        comments="",
    )


def run_validation(
    output_dir: Path,
    *,
    threads: int = 1,
    reuse: bool = False,
) -> dict:
    """Run the FDTD benchmark and write CSV plus machine-readable metrics."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "_cache"
    cache.mkdir(exist_ok=True)
    stem = cache / "impedance_modal_injection"
    hdf5_path = stem.with_suffix(".h5")
    started = perf_counter()
    if not (reuse and hdf5_path.is_file()):
        gprMax.run(
            scenes=[build_scene(threads)],
            outputfile=stem,
            cpu_precision="double",
            hide_progress_bars=True,
            log_level=logging.WARNING,
        )
    result = analyse_output(hdf5_path)
    _write_csv(output_dir / "impedance_modal_injection.csv", result)

    checks = {
        "source_plane_reflection": {
            "value_db": result["maximum_source_reflection_db"],
            "maximum_db": MAX_SOURCE_REFLECTION_DB,
            "passed": result["maximum_source_reflection_db"] < MAX_SOURCE_REFLECTION_DB,
        },
        "attenuation_perturbation": {
            "relative_l2_error": result["alpha_relative_l2_error"],
            "maximum_relative_l2_error": MAX_ALPHA_RELATIVE_L2_ERROR,
            "passed": result["alpha_relative_l2_error"] < MAX_ALPHA_RELATIVE_L2_ERROR,
        },
    }
    summary = {
        "model": "rectangular_TE10_constant_surface_resistance",
        "surface_resistance_ohm": SURFACE_RESISTANCE,
        "frequency_band_hz": [FMIN, FMAX],
        "guide_width_m": GUIDE_WIDTH,
        "guide_height_m": GUIDE_HEIGHT,
        "monitor_spacing_m": MONITOR_SPACING,
        "time_window_s": TIME_WINDOW,
        "earliest_wall_end_round_trip_s": earliest_wall_end_round_trip(),
        "runtime_seconds": perf_counter() - started,
        "metrics": {
            "maximum_source_reflection_db": result["maximum_source_reflection_db"],
            "rms_source_reflection": result["rms_source_reflection"],
            "alpha_relative_l2_error": result["alpha_relative_l2_error"],
        },
        "acceptance": {
            "passed": all(item["passed"] for item in checks.values()),
            "checks": checks,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("impedance_modal_injection_results"),
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    summary = run_validation(args.output_dir, threads=args.threads, reuse=args.reuse)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["acceptance"]["passed"]:
        raise SystemExit("surface-impedance modal-injection validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
