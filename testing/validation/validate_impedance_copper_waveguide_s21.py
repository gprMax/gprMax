"""Validate complex TE10 transmission through a copper-impedance guide.

An active eigenmode port launches TE10 into a rectangular waveguide whose four
walls use the built-in copper surface-impedance preset.  Two downstream
passive ports define a raw propagation factor for comparison with physical
copper/Yee theory.  Because finite-record ripple can dominate a few tenths of
a decibel, that raw comparison is diagnostic rather than a release gate.  The
quantitative gates instead compare attenuation from the exact FDFD ``neff``
anchors with first-order copper perturbation theory and require low driven-port
reflection after the modal field is injected into FDTD.

The finite impedance walls stop outside the PML.  The time window ends before
the earliest possible reflection from either wall end can return to any port,
so the measured ratio is the one-way propagation S21 and does not depend on a
guide termination model.

Example::

    python -m testing.validation.validate_impedance_copper_waveguide_s21 --threads 4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, epsilon_0, mu_0

import gprMax
from gprMax.surface_impedance_presets import (
    get_metal_surface_preset,
    good_conductor_surface_impedance,
)


DL = 0.1e-3
DOMAIN = (0.160, 0.0028, 0.002)
PML_CELLS = 3
TIME_WINDOW = 0.42e-9
GUIDE_WIDTH = 0.0016
GUIDE_HEIGHT = 0.0008
GUIDE_LOWER = (0.0006, 0.0006)
GUIDE_UPPER = (0.0022, 0.0014)
WALL_OUTER_LOWER = (0.0004, 0.0004)
WALL_OUTER_UPPER = (0.0024, 0.0016)
WALL_X = (4 * DL, 1596 * DL)
SOURCE_X = 0.0675
PORT1_X = 0.0825
PORT2_X = 0.1025
REFERENCE_PLANE_SPACING = PORT2_X - PORT1_X
FMIN = 130e9
FMAX = 150e9
DFT_POINTS = 3
EXCITATION_FMIN = 110e9
EXCITATION_FMAX = 150e9
EXCITATION_POINTS = 2
VALIDATION_FREQUENCIES = tuple(np.linspace(FMIN, FMAX, DFT_POINTS))
# Solving at the actual DFT bins avoids folding broadband modal-interpolation
# error into the copper propagation comparison.
ANCHORS = (
    90e9,
    EXCITATION_FMIN,
    120e9,
    *VALIDATION_FREQUENCIES,
    170e9,
)
MODEL_ID = "copper_wall"
METAL_PRESET = "copper"
FIT_FMIN = 80e9
# The FDFD boundary evaluates the trapezoidal ADE at bilinear-warped anchor
# frequencies, so the declared fit band deliberately extends past 170 GHz.
FIT_FMAX = 180e9
MAX_ATTENUATION_RELATIVE_L2_ERROR = 0.01
MAX_SOURCE_REFLECTION_DB = -20.0
ETA0 = np.sqrt(mu_0 / epsilon_0)


def te10_cutoff() -> float:
    """Return the continuum TE10 cutoff frequency."""

    return c / (2 * GUIDE_WIDTH)


def next_rectangular_mode_cutoff() -> float:
    """Return the lower of the TE20 and TE01 cutoff frequencies."""

    return min(c / GUIDE_WIDTH, c / (2 * GUIDE_HEIGHT))


def te10_group_velocity(frequency_hz) -> np.ndarray:
    """Return the lossless continuum TE10 group velocity."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    cutoff = te10_cutoff()
    if np.any(frequency <= cutoff):
        raise ValueError("TE10 group velocity requires frequencies above cutoff")
    return c * np.sqrt(1 - np.square(cutoff / frequency))


def earliest_wall_end_round_trip() -> float:
    """Return the strict earliest wall-end reflection reaching any port.

    A contaminating path must first travel from the active source plane to a
    wall end, then return from that end to the source or a passive reference
    plane.  Dividing by ``c`` rather than an in-band group velocity makes this
    a conservative causal-front bound, including tiny out-of-band precursors.
    """

    destinations = (SOURCE_X, PORT1_X, PORT2_X)
    paths = (
        *((SOURCE_X - WALL_X[0]) + (position - WALL_X[0]) for position in destinations),
        *((WALL_X[1] - SOURCE_X) + (WALL_X[1] - position) for position in destinations),
    )
    return min(paths) / c


def copper_surface_impedance(frequency_hz) -> np.ndarray:
    """Return the independent bulk-copper good-conductor impedance."""

    copper = get_metal_surface_preset(METAL_PRESET)
    return good_conductor_surface_impedance(
        frequency_hz,
        copper.conductivity_s_per_m,
    )


def perturbation_coefficient(frequency_hz) -> np.ndarray:
    """Return the TE10 wall-impedance propagation coefficient in 1/(Ohm m).

    For guide width ``a``, height ``b``, free-space wavenumber ``k``, and
    ``kc = pi/a``, first-order transverse-resonance/energy perturbation gives

    ``q = 1/eta0 * [k/(beta*b) + 2*kc**2/(k*beta*a)]``.

    A wall impedance ``Zs = Rs + j Xs`` then contributes attenuation
    ``alpha = q Rs`` and phase constant correction ``delta_beta = q Xs``.
    """

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    k = 2 * np.pi * frequency / c
    kc = np.pi / GUIDE_WIDTH
    if np.any(k <= kc):
        raise ValueError("TE10 perturbation theory requires frequencies above cutoff")
    beta = np.sqrt(np.square(k) - kc**2)
    return (
        k / (beta * GUIDE_HEIGHT)
        + 2 * kc**2 / (k * beta * GUIDE_WIDTH)
    ) / ETA0


def continuum_beta(frequency_hz) -> np.ndarray:
    """Return the continuum lossless TE10 phase constant."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    k = 2 * np.pi * frequency / c
    kc = np.pi / GUIDE_WIDTH
    if np.any(k <= kc):
        raise ValueError("TE10 propagation requires frequencies above cutoff")
    return np.sqrt(np.square(k) - kc**2)


def yee_beta(
    frequency_hz,
    *,
    dl: float = DL,
    dt: float | None = None,
) -> np.ndarray:
    """Return the exact lossless TE10 phase constant of the cubic Yee grid."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    spacing = float(dl)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Yee spacing must be finite and positive")
    # Use the timestep recorded by the executed model when supplied.  The
    # Courant value is only a convenient default for analytical/unit calls.
    timestep = spacing / (c * np.sqrt(3)) if dt is None else float(dt)
    if not np.isfinite(timestep) or timestep <= 0:
        raise ValueError("Yee timestep must be finite and positive")
    temporal = np.square(
        np.sin(np.pi * frequency * timestep) / (c * timestep)
    )
    transverse = np.square(np.sin(np.pi * spacing / (2 * GUIDE_WIDTH)) / spacing)
    longitudinal = temporal - transverse
    if np.any(longitudinal <= 0):
        raise ValueError("TE10 is below the discrete Yee cutoff")
    argument = spacing * np.sqrt(longitudinal)
    if np.any(argument >= 1):
        raise ValueError("TE10 longitudinal wavenumber is outside the Yee passband")
    return 2 * np.arcsin(argument) / spacing


def theoretical_s21(
    frequency_hz,
    *,
    numerical_dispersion: bool = True,
    dt: float | None = None,
    propagation_length: float = REFERENCE_PLANE_SPACING,
) -> np.ndarray:
    """Return first-order complex copper TE10 transmission between the ports."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    impedance = copper_surface_impedance(frequency)
    coefficient = perturbation_coefficient(frequency)
    alpha = coefficient * impedance.real
    beta0 = (
        yee_beta(frequency, dt=dt)
        if numerical_dispersion
        else continuum_beta(frequency)
    )
    beta = beta0 + coefficient * impedance.imag
    return np.exp(-(alpha + 1j * beta) * float(propagation_length))


def theoretical_phase_rad(
    frequency_hz,
    *,
    numerical_dispersion: bool = True,
    dt: float | None = None,
    propagation_length: float = REFERENCE_PLANE_SPACING,
) -> np.ndarray:
    """Return the unwrapped phase used by :func:`theoretical_s21`."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    impedance = copper_surface_impedance(frequency)
    beta0 = (
        yee_beta(frequency, dt=dt)
        if numerical_dispersion
        else continuum_beta(frequency)
    )
    beta = beta0 + perturbation_coefficient(frequency) * impedance.imag
    return -beta * float(propagation_length)


def _wall_boxes() -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
    """Return the four opaque boxes forming the rectangular guide walls."""

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
    """Return the causally isolated copper-guide scene."""

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
            preset=METAL_PRESET,
            fit_fmin_hz=FIT_FMIN,
            fit_fmax_hz=FIT_FMAX,
        )
    )
    for lower, upper in _wall_boxes():
        scene.add(gprMax.ImpedanceBox(lower, upper, MODEL_ID))

    scene.add(
        gprMax.EigenmodeBand(
            id="copper_te10",
            fmin=EXCITATION_FMIN,
            fmax=EXCITATION_FMAX,
            points=EXCITATION_POINTS,
            frequencies=VALIDATION_FREQUENCIES,
            transition=20e9,
        )
    )
    for port, x, direction in (
        (1, SOURCE_X, "+"),
        (2, PORT1_X, "-"),
        (3, PORT2_X, "-"),
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
    """Return finite-safe voltage-wave magnitude in decibels."""

    return 20 * np.log10(np.maximum(np.abs(values), np.finfo(np.float64).tiny))


def wrapped_phase_error_deg(measured, reference) -> np.ndarray:
    """Return absolute principal-value phase error in degrees."""

    return np.abs(np.rad2deg(np.angle(np.asarray(measured) / np.asarray(reference))))


def analyse_s21(
    frequency_hz,
    near_outgoing,
    far_outgoing,
    *,
    dt: float | None = None,
    propagation_length: float = REFERENCE_PLANE_SPACING,
) -> dict:
    """Compare measured two-plane S21 with copper perturbation theory."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    near = np.asarray(near_outgoing, dtype=np.complex128)
    far = np.asarray(far_outgoing, dtype=np.complex128)
    if not (frequency.ndim == near.ndim == far.ndim == 1):
        raise ValueError("S21 inputs must be one-dimensional")
    if not (frequency.shape == near.shape == far.shape):
        raise ValueError("S21 inputs must have identical shapes")
    if np.any(~np.isfinite(frequency)) or np.any(frequency <= te10_cutoff()):
        raise ValueError("S21 frequencies must be finite and above cutoff")
    if np.any(~np.isfinite(near)) or np.any(~np.isfinite(far)):
        raise ValueError("passive-port modal coefficients must be finite")
    if np.any(np.abs(near) == 0):
        raise ValueError("near-port outgoing coefficients must be non-zero")

    measured = far / near
    length = float(propagation_length)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("S21 propagation length must be finite and positive")
    analytical = theoretical_s21(frequency, dt=dt, propagation_length=length)
    continuum = theoretical_s21(
        frequency,
        numerical_dispersion=False,
        propagation_length=length,
    )
    magnitude_error = magnitude_db(measured) - magnitude_db(analytical)
    phase_error = wrapped_phase_error_deg(measured, analytical)
    complex_error = measured - analytical
    measured_alpha = -np.log(np.abs(measured)) / length
    analytical_alpha = -np.log(np.abs(analytical)) / length
    alpha_error = measured_alpha - analytical_alpha
    return {
        "frequency_hz": frequency,
        "measured_s21": measured,
        "analytical_s21": analytical,
        "continuum_s21": continuum,
        "magnitude_error_db": magnitude_error,
        "phase_error_deg": phase_error,
        "complex_error": complex_error,
        "measured_alpha_per_m": measured_alpha,
        "analytical_alpha_per_m": analytical_alpha,
        "alpha_error_per_m": alpha_error,
        "maximum_magnitude_error_db": float(np.max(np.abs(magnitude_error))),
        "maximum_phase_error_deg": float(np.max(phase_error)),
        "complex_relative_l2_error": float(
            np.linalg.norm(complex_error) / np.linalg.norm(analytical)
        ),
        "attenuation_relative_l2_error": float(
            np.linalg.norm(alpha_error) / np.linalg.norm(analytical_alpha)
        ),
        "continuum_phase_discretisation_deg": wrapped_phase_error_deg(
            analytical, continuum
        ),
        "dt_s": dt,
        "propagation_length_m": length,
        "analytical_phase_rad": theoretical_phase_rad(
            frequency,
            dt=dt,
            propagation_length=length,
        ),
        "continuum_phase_rad": theoretical_phase_rad(
            frequency,
            numerical_dispersion=False,
            propagation_length=length,
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
        s_parameter = np.asarray(group["S"])[0] if "S" in group else None
        s_valid_name = (
            "power_wave_valid_S"
            if "power_wave_valid_S" in group
            else "valid_S"
        )
        s_valid = (
            np.asarray(group[s_valid_name], dtype=bool)[0]
            if s_parameter is not None
            else None
        )
        anchors = np.asarray(
            group.attrs["CandidateAnchorFrequencies"], dtype=np.float64
        )
        anchor_neff = np.asarray(group["anchor_complex_neff"])[:, 0]
        anchor_valid = np.asarray(group["anchor_mode_valid"], dtype=bool)[:, 0]
        anchor_reference_valid = np.asarray(
            group["anchor_mode_reference_valid"], dtype=bool
        )[:, 0]
        plane_index = int(group.attrs["PlaneIndex"])
    return {
        "frequency": frequency,
        "incident": incident,
        "outgoing": outgoing,
        "valid": valid,
        "s_parameter": s_parameter,
        "s_valid": s_valid,
        "anchors": anchors,
        "anchor_neff": anchor_neff,
        "anchor_valid": anchor_valid,
        "anchor_reference_valid": anchor_reference_valid,
        "plane_index": plane_index,
    }


def _exact_anchor_neff(port: dict, frequency_hz) -> np.ndarray:
    """Return valid mode-1 neff rows at the requested exact anchors."""

    values = []
    for frequency in np.asarray(frequency_hz, dtype=np.float64):
        tolerance = 32 * np.finfo(float).eps * frequency
        matches = np.flatnonzero(
            np.isclose(port["anchors"], frequency, rtol=0, atol=tolerance)
        )
        if matches.size != 1:
            raise RuntimeError(
                f"expected one FDFD anchor at {frequency:g} Hz, found {matches.size}"
            )
        index = int(matches[0])
        if not (
            port["anchor_valid"][index]
            and port["anchor_reference_valid"][index]
        ):
            raise RuntimeError(f"FDFD anchor at {frequency:g} Hz is invalid")
        values.append(port["anchor_neff"][index])
    return np.asarray(values, dtype=np.complex128)


def analyse_output(copper_path: Path) -> dict:
    """Analyse copper propagation, FDFD attenuation, and source mismatch."""

    source, near, far = [_read_port(copper_path, port) for port in (1, 2, 3)]
    with h5py.File(copper_path, "r") as copper_data:
        dt = float(copper_data.attrs["dt"])
        spacing = np.asarray(copper_data.attrs["dx_dy_dz"], dtype=np.float64)
    if not np.allclose(spacing, DL, rtol=0, atol=32 * np.finfo(float).eps):
        raise RuntimeError(
            f"benchmark expected a cubic {DL:g} m grid, output records {spacing}"
        )
    np.testing.assert_array_equal(source["frequency"], near["frequency"])
    np.testing.assert_array_equal(near["frequency"], far["frequency"])
    frequency = near["frequency"]
    valid = source["valid"] & near["valid"] & far["valid"]
    comparison = (frequency >= FMIN) & (frequency <= FMAX)
    if np.count_nonzero(comparison) != DFT_POINTS:
        raise RuntimeError("benchmark output does not contain every comparison bin")
    if not np.all(valid[comparison]):
        invalid = frequency[comparison & ~valid]
        raise RuntimeError(
            "benchmark has invalid TE10 power-wave coefficients at "
            + ", ".join(f"{value:g}" for value in invalid)
            + " Hz"
        )
    length = abs(far["plane_index"] - near["plane_index"]) * spacing[0]
    if not np.isclose(
        length,
        REFERENCE_PLANE_SPACING,
        rtol=0,
        atol=32 * np.finfo(float).eps * max(1.0, length),
    ):
        raise RuntimeError(
            f"HDF5 reference-plane spacing {length:g} m does not match model "
            f"spacing {REFERENCE_PLANE_SPACING:g} m"
        )
    if np.any(np.abs(near["outgoing"][comparison]) == 0):
        raise RuntimeError("copper near-port outgoing coefficient is zero")
    result = analyse_s21(
        frequency[comparison],
        near["outgoing"][comparison],
        far["outgoing"][comparison],
        dt=dt,
        propagation_length=length,
    )
    comparison_frequency = frequency[comparison]
    copper_neff = _exact_anchor_neff(source, comparison_frequency)
    k0 = 2 * np.pi * comparison_frequency / c
    fdfd_alpha = -k0 * np.imag(copper_neff)
    physical_alpha = perturbation_coefficient(comparison_frequency) * np.real(
        copper_surface_impedance(comparison_frequency)
    )
    fdfd_alpha_error = fdfd_alpha - physical_alpha

    if source["s_parameter"] is None or source["s_valid"] is None:
        raise RuntimeError("driven copper port does not contain HDF5 S11")
    if not np.all(source["s_valid"][comparison]):
        raise RuntimeError("driven copper port has invalid HDF5 S11")
    source_reflection = source["s_parameter"][comparison]
    result.update(
        {
            "copper_neff": copper_neff,
            "fdfd_alpha_per_m": fdfd_alpha,
            "physical_alpha_per_m": physical_alpha,
            "fdfd_alpha_error_per_m": fdfd_alpha_error,
            "fdfd_alpha_relative_l2_error": float(
                np.linalg.norm(fdfd_alpha_error) / np.linalg.norm(physical_alpha)
            ),
            "fdfd_insertion_loss_db": 8.685889638 * fdfd_alpha * length,
            "physical_insertion_loss_db": 8.685889638 * physical_alpha * length,
            "source_reflection": source_reflection,
            "source_reflection_db": magnitude_db(source_reflection),
            "maximum_source_reflection_db": float(
                np.max(magnitude_db(source_reflection))
            ),
        }
    )
    return result


def _write_csv(path: Path, result: dict) -> None:
    copper = result["measured_s21"]
    analytical = result["analytical_s21"]
    continuum = result["continuum_s21"]
    copper_phase = result["analytical_phase_rad"] + np.angle(copper / analytical)
    table = np.column_stack(
        (
            result["frequency_hz"],
            copper.real,
            copper.imag,
            magnitude_db(copper),
            np.rad2deg(copper_phase),
            analytical.real,
            analytical.imag,
            magnitude_db(analytical),
            np.rad2deg(result["analytical_phase_rad"]),
            continuum.real,
            continuum.imag,
            magnitude_db(continuum),
            np.rad2deg(result["continuum_phase_rad"]),
            result["copper_neff"].real,
            result["copper_neff"].imag,
            result["source_reflection"].real,
            result["source_reflection"].imag,
            result["source_reflection_db"],
            result["fdfd_alpha_per_m"],
            result["physical_alpha_per_m"],
            result["fdfd_alpha_error_per_m"],
            result["magnitude_error_db"],
            result["phase_error_deg"],
        )
    )
    np.savetxt(
        path,
        table,
        delimiter=",",
        header=(
            "frequency_hz,gprmax_copper_T_real,gprmax_copper_T_imag,"
            "gprmax_copper_T_magnitude_db,gprmax_copper_T_matched_phase_deg,"
            "theory_yee_T_real,theory_yee_T_imag,theory_yee_T_magnitude_db,"
            "theory_yee_T_phase_deg,theory_continuum_T_real,"
            "theory_continuum_T_imag,theory_continuum_T_magnitude_db,"
            "theory_continuum_T_phase_deg,copper_neff_real,copper_neff_imag,"
            "source_S11_real,source_S11_imag,source_S11_magnitude_db,"
            "fdfd_alpha_per_m,physical_alpha_per_m,fdfd_alpha_error_per_m,"
            "raw_magnitude_error_db,raw_phase_error_deg"
        ),
        comments="",
    )


def _write_plot(path: Path, result: dict) -> None:
    """Write raw propagation and independent FDFD attenuation comparisons."""

    frequency_ghz = result["frequency_hz"] * 1e-9
    copper = result["measured_s21"]
    analytical = result["analytical_s21"]
    continuum = result["continuum_s21"]
    copper_phase = result["analytical_phase_rad"] + np.angle(copper / analytical)

    figure, axes = plt.subplots(3, 1, figsize=(8.8, 8.8), sharex=True)
    axes[0].plot(frequency_ghz, magnitude_db(analytical), "k-", label="Theory + Yee")
    axes[0].plot(frequency_ghz, magnitude_db(continuum), ":", color="0.5", label="Continuum theory")
    axes[0].plot(frequency_ghz, magnitude_db(copper), "o--", label="gprMax")
    axes[0].set_ylabel("Raw propagation (dB)")
    axes[0].set_title("Copper rectangular-guide TE10 propagation")
    axes[0].legend()

    axes[1].plot(
        frequency_ghz,
        np.rad2deg(result["analytical_phase_rad"]),
        "k-",
        label="Theory + Yee",
    )
    axes[1].plot(
        frequency_ghz,
        np.rad2deg(result["continuum_phase_rad"]),
        ":",
        color="0.5",
        label="Continuum theory",
    )
    axes[1].plot(frequency_ghz, np.rad2deg(copper_phase), "o--", label="gprMax")
    axes[1].set_ylabel("Matched phase (degrees)")
    axes[1].legend()

    axes[2].plot(
        frequency_ghz,
        result["physical_alpha_per_m"],
        "k-",
        label="Perturbation theory",
    )
    axes[2].plot(
        frequency_ghz,
        result["fdfd_alpha_per_m"],
        "o--",
        label="FDFD anchor neff",
    )
    axes[2].set_xlabel("Frequency (GHz)")
    axes[2].set_ylabel(r"$\alpha$ (Np/m)")
    axes[2].legend()
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_validation(
    output_dir: Path,
    *,
    threads: int = 1,
    reuse: bool = False,
) -> dict:
    """Run the copper-guide benchmark and write CSV plus JSON acceptance."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "_cache"
    cache.mkdir(exist_ok=True)
    copper_stem = cache / "impedance_copper_waveguide_s21_copper"
    copper_path = copper_stem.with_suffix(".h5")
    started = perf_counter()
    if not (reuse and copper_path.is_file()):
        gprMax.run(
            scenes=[build_scene(threads)],
            outputfile=copper_stem,
            cpu_precision="double",
            hide_progress_bars=True,
            log_level=logging.WARNING,
        )
    result = analyse_output(copper_path)
    _write_csv(output_dir / "impedance_copper_waveguide_s21.csv", result)
    _write_plot(output_dir / "impedance_copper_waveguide_s21.png", result)

    checks = {
        "source_reflection": {
            "maximum_db": result["maximum_source_reflection_db"],
            "maximum_allowed_db": MAX_SOURCE_REFLECTION_DB,
            "passed": result["maximum_source_reflection_db"] < MAX_SOURCE_REFLECTION_DB,
        },
        "fdfd_physical_attenuation": {
            "relative_l2_error": result["fdfd_alpha_relative_l2_error"],
            "maximum_relative_l2_error": MAX_ATTENUATION_RELATIVE_L2_ERROR,
            "passed": (
                result["fdfd_alpha_relative_l2_error"]
                < MAX_ATTENUATION_RELATIVE_L2_ERROR
            ),
        },
    }
    copper = get_metal_surface_preset(METAL_PRESET)
    summary = {
        "model": "rectangular_TE10_copper_surface_impedance",
        "s21_definition": (
            "raw diagnostic T = outgoing(port3) / outgoing(port2) between "
            "passive electric reference planes"
        ),
        "metal_preset": METAL_PRESET,
        "conductivity_s_per_m": copper.conductivity_s_per_m,
        "reference_temperature_k": copper.reference_temperature_k,
        "frequency_band_hz": [FMIN, FMAX],
        "excitation_band_hz": [EXCITATION_FMIN, EXCITATION_FMAX],
        "guide_width_m": GUIDE_WIDTH,
        "guide_height_m": GUIDE_HEIGHT,
        "next_mode_cutoff_hz": next_rectangular_mode_cutoff(),
        "reference_plane_spacing_m": result["propagation_length_m"],
        "time_window_s": TIME_WINDOW,
        "fdtd_dt_s": result["dt_s"],
        "earliest_wall_end_round_trip_s": earliest_wall_end_round_trip(),
        "runtime_seconds": perf_counter() - started,
        "metrics": {
            "maximum_source_reflection_db": result[
                "maximum_source_reflection_db"
            ],
            "fdfd_physical_alpha_relative_l2_error": result[
                "fdfd_alpha_relative_l2_error"
            ],
            "physical_insertion_loss_db_min": float(
                np.min(result["physical_insertion_loss_db"])
            ),
            "physical_insertion_loss_db_max": float(
                np.max(result["physical_insertion_loss_db"])
            ),
            "fdfd_insertion_loss_db_min": float(
                np.min(result["fdfd_insertion_loss_db"])
            ),
            "fdfd_insertion_loss_db_max": float(
                np.max(result["fdfd_insertion_loss_db"])
            ),
            "maximum_raw_physical_magnitude_error_db": result[
                "maximum_magnitude_error_db"
            ],
            "maximum_raw_physical_phase_error_deg": result[
                "maximum_phase_error_deg"
            ],
            "maximum_continuum_phase_discretisation_deg": float(
                np.max(result["continuum_phase_discretisation_deg"])
            ),
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
        default=Path(__file__).with_name("impedance_copper_waveguide_s21_results"),
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    summary = run_validation(args.output_dir, threads=args.threads, reuse=args.reuse)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["acceptance"]["passed"]:
        raise SystemExit("copper-guide S21 validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
