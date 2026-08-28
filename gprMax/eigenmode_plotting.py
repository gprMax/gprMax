# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

"""Headless plotting helpers for eigenmode-port diagnostics."""

import math

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from gprMax.fdfd_eigenmode_solver.fdfd_1d_mode_solver import FDFD_1D_mode_solver
from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver


def _frequency_label(frequency):
    frequency = float(frequency)
    for scale, unit in ((1e12, "THz"), (1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if abs(frequency) >= scale:
            return f"{frequency / scale:.6g} {unit}"
    return f"{frequency:.6g} Hz"


def _complex_label(value):
    value = complex(value)
    tolerance = 1e-12 * max(1.0, abs(value))
    if abs(value.imag) <= tolerance:
        return f"{value.real:.6g}"
    return f"{value.real:.6g}{value.imag:+.6g}j"


def _display_phase(electric, magnetic):
    """Choose one phase rotation that makes the combined E/H plot mostly real."""
    normalized = []
    for fields in (electric, magnetic):
        peak = max((float(np.max(np.abs(field))) for field in fields), default=0.0)
        if peak > 0:
            normalized.extend(np.ravel(field / peak) for field in fields)
    if not normalized:
        return 1.0 + 0.0j

    values = np.concatenate(normalized)
    second_moment = np.sum(values * values)
    phase = np.exp(-0.5j * np.angle(second_moment)) if abs(second_moment) > 1e-300 else 1.0 + 0.0j
    rotated = values * phase
    dominant = rotated[np.argmax(np.abs(rotated))]
    if np.real(dominant) < 0:
        phase = -phase
    return phase


def _subsample_steps(shape, maximum_arrows=25):
    return tuple(max(1, int(math.ceil(length / maximum_arrows))) for length in shape)


def _source_dft(samples, dt, frequencies):
    """Evaluate the sampled waveform at the exact port DFT frequencies."""
    samples = np.asarray(samples, dtype=np.float64)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if frequencies.size == 0:
        return np.empty(0, dtype=np.complex128)
    times = np.arange(samples.size, dtype=np.float64) * float(dt)
    result = np.empty(frequencies.size, dtype=np.complex128)
    maximum_phase_elements = 2_000_000
    frequency_block = max(
        1,
        min(frequencies.size, maximum_phase_elements // min(samples.size, maximum_phase_elements)),
    )
    for start in range(0, frequencies.size, frequency_block):
        stop = min(start + frequency_block, frequencies.size)
        accumulator = np.zeros(stop - start, dtype=np.complex128)
        time_block = max(1, maximum_phase_elements // (stop - start))
        for time_start in range(0, samples.size, time_block):
            time_stop = min(time_start + time_block, samples.size)
            phase = np.exp(
                -2j
                * np.pi
                * frequencies[start:stop, None]
                * times[None, time_start:time_stop]
            )
            accumulator += phase @ samples[time_start:time_stop]
        result[start:stop] = float(dt) * accumulator
    return result


def _axis_scale(maximum, scales):
    for scale, unit in scales:
        if maximum >= scale:
            return scale, unit
    return scales[-1]


def plot_eigenmode_excitation(
    *,
    samples,
    dt,
    dft_frequencies,
    band_start,
    band_stop,
    port_index,
    waveform_id,
    spectral_threshold,
    output_path,
):
    """Plot the one eigenmode excitation waveform and its sampled spectrum."""
    samples = np.asarray(samples, dtype=np.float64)
    dft_frequencies = np.asarray(dft_frequencies, dtype=np.float64)
    if samples.ndim != 1 or samples.size < 2:
        raise ValueError("Eigenmode waveform plotting requires at least two samples.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("Eigenmode waveform plotting requires finite samples.")
    if dft_frequencies.ndim != 1 or dft_frequencies.size < 1:
        raise ValueError("Eigenmode waveform plotting requires at least one DFT frequency.")

    dt = float(dt)
    padded_count = 1 << int(np.ceil(np.log2(max(2, 2 * samples.size))))
    spectrum_frequencies = np.fft.rfftfreq(padded_count, d=dt)
    spectrum = dt * np.fft.rfft(samples, n=padded_count)
    source_dft = _source_dft(samples, dt, dft_frequencies)
    spectrum_magnitude = np.abs(spectrum)
    source_magnitude = np.abs(source_dft)
    peak = max(
        float(np.max(spectrum_magnitude, initial=0.0)),
        float(np.max(source_magnitude, initial=0.0)),
    )
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("Eigenmode excitation waveform has no finite spectral energy.")

    floor_db = -100.0
    spectrum_db = np.maximum(20 * np.log10(np.maximum(spectrum_magnitude / peak, 1e-300)), floor_db)
    source_db = np.maximum(20 * np.log10(np.maximum(source_magnitude / peak, 1e-300)), floor_db)

    fig = Figure(figsize=(13, 4.8), constrained_layout=True)
    FigureCanvasAgg(fig)
    time_axis, spectrum_axis = fig.subplots(1, 2)
    fig.suptitle(f"Eigenmode excitation on Port {port_index}: {waveform_id}")

    times = np.arange(samples.size, dtype=np.float64) * dt
    time_scale, time_unit = _axis_scale(
        float(times[-1]),
        ((1.0, "s"), (1e-3, "ms"), (1e-6, "us"), (1e-9, "ns"), (1e-12, "ps")),
    )
    time_axis.plot(times / time_scale, samples, color="tab:blue", linewidth=1.1)
    time_axis.set_title("Injected sampled waveform")
    time_axis.set_xlabel(f"simulation time ({time_unit})")
    time_axis.set_ylabel("amplitude")
    time_axis.grid(True, alpha=0.25)

    frequency_scale, frequency_unit = _axis_scale(
        max(float(band_stop), float(np.max(dft_frequencies))),
        ((1e12, "THz"), (1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz"), (1.0, "Hz")),
    )
    scaled_spectrum_frequency = spectrum_frequencies / frequency_scale
    spectrum_axis.plot(
        scaled_spectrum_frequency,
        spectrum_db,
        color="tab:blue",
        linewidth=1.2,
        label="zero-padded sampled spectrum",
    )
    spectrum_axis.scatter(
        dft_frequencies / frequency_scale,
        source_db,
        s=18,
        color="tab:red",
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
        label="source DFT bins",
    )

    if band_stop > band_start:
        spectrum_axis.axvspan(
            band_start / frequency_scale,
            band_stop / frequency_scale,
            color="tab:green",
            alpha=0.14,
            label="port band",
        )
        spectrum_axis.axvline(band_start / frequency_scale, color="tab:green", linewidth=0.9)
        spectrum_axis.axvline(band_stop / frequency_scale, color="tab:green", linewidth=0.9)
    else:
        spectrum_axis.axvline(
            band_start / frequency_scale,
            color="tab:green",
            linewidth=1.2,
            label="port frequency",
        )

    significant = spectrum_magnitude >= max(float(spectral_threshold), 1e-6) * peak
    significant_indices = np.flatnonzero(significant)
    if significant_indices.size:
        visible_low = min(float(band_start), float(spectrum_frequencies[significant_indices[0]]))
        visible_high = max(float(band_stop), float(spectrum_frequencies[significant_indices[-1]]))
    else:
        visible_low = float(band_start)
        visible_high = float(band_stop)
    visible_span = max(
        visible_high - visible_low,
        0.2 * max(abs(visible_high), abs(visible_low), spectrum_frequencies[1]),
    )
    margin = 0.08 * visible_span
    spectrum_axis.set_xlim(
        max(0.0, visible_low - margin) / frequency_scale,
        min(float(spectrum_frequencies[-1]), visible_high + margin) / frequency_scale,
    )
    spectrum_axis.set_ylim(floor_db, 3.0)
    spectrum_axis.set_title("Waveform DFT magnitude")
    spectrum_axis.set_xlabel(f"frequency ({frequency_unit})")
    spectrum_axis.set_ylabel("relative magnitude (dB)")
    spectrum_axis.grid(True, alpha=0.25)
    spectrum_axis.legend(fontsize=8, loc="best")

    fig.savefig(output_path, dpi=200)
    return output_path


def _plot_2d_vector(ax, solver, u_field, v_field, phase, field_label):
    u = solver._field_to_cells(u_field, "u" if field_label == "E" else "hu") * phase
    v = solver._field_to_cells(v_field, "v" if field_label == "E" else "hv") * phase
    magnitude = np.sqrt(np.abs(u) ** 2 + np.abs(v) ** 2)
    peak = float(np.max(magnitude)) if magnitude.size else 0.0
    relative_magnitude = magnitude / peak if peak > 0 else magnitude
    u_real = np.real(u) / peak if peak > 0 else np.real(u)
    v_real = np.real(v) / peak if peak > 0 else np.real(v)

    du_mm = float(solver.du) * 1e3
    dv_mm = float(solver.dv) * 1e3
    extent = (0.0, solver.Nu * du_mm, 0.0, solver.Nv * dv_mm)
    image = ax.imshow(
        relative_magnitude.T,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )

    step_u, step_v = _subsample_steps(relative_magnitude.shape)
    u_coordinate = (np.arange(solver.Nu) + 0.5) * du_mm
    v_coordinate = (np.arange(solver.Nv) + 0.5) * dv_mm
    grid_u, grid_v = np.meshgrid(u_coordinate, v_coordinate, indexing="ij")
    selection = np.s_[::step_u, ::step_v]
    arrow_length = 0.8 * min(step_u * du_mm, step_v * dv_mm)
    ax.quiver(
        grid_u[selection],
        grid_v[selection],
        u_real[selection] * arrow_length,
        v_real[selection] * arrow_length,
        color="white",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        pivot="mid",
        width=0.004,
    )
    ax.set_xlabel("local u (mm)")
    ax.set_ylabel("local v (mm)")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(f"relative |{field_label}$_t$|")


def _one_dimensional_fields(solver, mode):
    if solver.polarization == "TM":
        return (
            (solver.Ea[:, mode], "a", "node"),
            (solver.Ht[:, mode], "t", "node"),
        )
    return (
        (solver.Et[:, mode], "t", "cell"),
        (solver.Ha[:, mode], "a", "cell"),
    )


def _plot_1d_vector(ax, field, component, stagger, phase, field_label):
    profile = np.asarray(field) * phase
    peak = float(np.max(np.abs(profile))) if profile.size else 0.0
    relative_magnitude = np.abs(profile) / peak if peak > 0 else np.abs(profile)
    real_profile = np.real(profile) / peak if peak > 0 else np.real(profile)
    coordinate = np.arange(profile.size, dtype=np.float64)
    if stagger == "cell":
        coordinate += 0.5

    step = max(1, int(math.ceil(profile.size / 40)))
    selection = np.s_[::step]
    u_vector = real_profile if component == "t" else np.zeros_like(real_profile)
    v_vector = real_profile if component == "a" else np.zeros_like(real_profile)
    arrows = ax.quiver(
        coordinate[selection],
        np.zeros_like(coordinate[selection]),
        u_vector[selection],
        v_vector[selection],
        relative_magnitude[selection],
        cmap="magma",
        clim=(0.0, 1.0),
        angles="uv",
        scale_units="height",
        scale=1.8,
        pivot="mid",
        width=0.008,
    )
    ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
    ax.set_xlim(-0.5, max(0.5, profile.size + 0.5))
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel(f"local t index ({stagger}-sampled)")
    ax.set_ylabel("local a direction (schematic)")
    ax.grid(True, alpha=0.2)
    colorbar = ax.figure.colorbar(arrows, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(f"relative |{field_label}$_t$|")


def plot_eigenmode_port_fields(solvers, frequencies, mode_index, port_index, output_path):
    """Write one two-column tangential-vector figure for a port mode.

    Each solver/frequency pair occupies one row. Electric and magnetic fields
    share a phase rotation within each row but use independent magnitude
    normalization in their respective panels.
    """
    solvers = tuple(solvers)
    frequencies = tuple(float(frequency) for frequency in frequencies)
    if not solvers:
        raise ValueError("At least one solved eigenmode anchor is required for plotting.")
    if len(solvers) != len(frequencies):
        raise ValueError("The eigenmode solver and anchor-frequency counts must match.")
    if any(type(solver) is not type(solvers[0]) for solver in solvers):
        raise ValueError("All plotted eigenmode anchors must use the same solver type.")

    mode = int(mode_index) - 1
    if mode < 0 or any(mode >= solver.num_modes for solver in solvers):
        raise ValueError(f"Mode index {mode_index} is outside the solved range.")

    fig = Figure(figsize=(12, max(3.8, 3.8 * len(solvers))), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(solvers), 2, squeeze=False)
    fig.suptitle(f"Port {port_index}, Mode {mode_index}: tangential modal vector fields")

    for row, (solver, frequency) in enumerate(zip(solvers, frequencies)):
        if isinstance(solver, FDFD_2D_mode_solver):
            electric = (
                solver._field_to_cells(solver.Eu[:, :, mode], "u"),
                solver._field_to_cells(solver.Ev[:, :, mode], "v"),
            )
            magnetic = (
                solver._field_to_cells(solver.Hu[:, :, mode], "hu"),
                solver._field_to_cells(solver.Hv[:, :, mode], "hv"),
            )
            phase = _display_phase(electric, magnetic)
            _plot_2d_vector(
                axes[row, 0], solver, solver.Eu[:, :, mode], solver.Ev[:, :, mode], phase, "E"
            )
            _plot_2d_vector(
                axes[row, 1], solver, solver.Hu[:, :, mode], solver.Hv[:, :, mode], phase, "H"
            )
        elif isinstance(solver, FDFD_1D_mode_solver):
            electric, magnetic = _one_dimensional_fields(solver, mode)
            phase = _display_phase((electric[0],), (magnetic[0],))
            _plot_1d_vector(axes[row, 0], *electric, phase, "E")
            _plot_1d_vector(axes[row, 1], *magnetic, phase, "H")
        else:
            raise TypeError(f"Unsupported eigenmode solver type {type(solver).__name__}.")

        anchor = _frequency_label(frequency)
        neff = _complex_label(solver.complex_neff[mode])
        axes[row, 0].set_title(f"Tangential E vector | {anchor} | n_eff={neff}")
        axes[row, 1].set_title(f"Tangential H vector | {anchor} | n_eff={neff}")

    fig.savefig(output_path, dpi=200)
    return output_path
