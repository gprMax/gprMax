# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax.

"""Headless diagnostics for fitted microwave surface impedances."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from gprMax.surface_impedance_presets import good_conductor_surface_impedance


def surface_impedance_fit_plot_path(output_stem: Path, model_id: str) -> Path:
    """Return a deterministic, filesystem-safe fit-plot path."""

    output_stem = Path(output_stem)
    model_id = str(model_id)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id).strip("._") or "model"
    # Windows paths are case-insensitive, so even two already-safe IDs such
    # as ``wall`` and ``WALL`` need a case-sensitive identity suffix.
    suffix = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:8]
    safe_id = f"{safe_id}_{suffix}"
    return output_stem.with_name(f"{output_stem.name}_surface_impedance_{safe_id}_fit.png")


def plot_good_conductor_surface_impedance_fit(
    *,
    model,
    conductivity_s_per_m: float,
    target_label: str,
    output_path: Path,
    requested_order: str | int,
    selected_pole_count: int,
    fit_tolerance: float,
    sample_count: int = 1601,
) -> Path:
    """Plot the intended good-conductor impedance and its fitted realization."""

    output_path = Path(output_path)
    frequencies = np.geomspace(
        float(model.fit_fmin_hz),
        float(model.fit_fmax_hz),
        int(sample_count),
    )
    intended = good_conductor_surface_impedance(
        frequencies,
        conductivity_s_per_m,
    )
    fitted = model.impedance(frequencies)
    relative_error = np.abs(fitted - intended) / np.abs(intended)

    figure = Figure(figsize=(11.2, 8.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    frequency_ghz = frequencies * 1e-9

    response_panels = (
        (axes[0, 0], intended.real, fitted.real, "Real impedance", "Re(Z) (Ohm)"),
        (axes[0, 1], intended.imag, fitted.imag, "Imaginary impedance", "Im(Z) (Ohm)"),
        (axes[1, 0], np.abs(intended), np.abs(fitted), "Magnitude", "|Z| (Ohm)"),
    )
    for axis, target_values, fit_values, title, ylabel in response_panels:
        axis.plot(frequency_ghz, target_values, color="black", linewidth=2.2, label="Intended")
        axis.plot(
            frequency_ghz,
            fit_values,
            color="#d95f02",
            linestyle="--",
            linewidth=1.8,
            label="Fitted",
        )
        axis.set_xscale("log")
        axis.set_title(title)
        axis.set_xlabel("Frequency (GHz)")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.28)
        axis.legend()

    error_axis = axes[1, 1]
    error_axis.semilogy(
        frequency_ghz,
        np.maximum(100 * relative_error, np.finfo(np.float64).tiny),
        color="#1b9e77",
        linewidth=2,
    )
    error_axis.axhline(
        100 * float(fit_tolerance),
        color="#7570b3",
        linestyle=":",
        linewidth=1.7,
        label="Requested tolerance",
    )
    error_axis.set_title("Complex relative error")
    error_axis.set_xlabel("Frequency (GHz)")
    error_axis.set_ylabel("Error (%)")
    error_axis.grid(True, which="both", alpha=0.28)
    error_axis.legend()

    maximum_error = float(relative_error.max())
    rms_error = float(np.sqrt(np.mean(relative_error**2)))
    figure.suptitle(
        f"Surface impedance {model.ID!r}: {target_label}\n"
        f"requested order {requested_order}, selected runtime poles "
        f"{selected_pole_count}; maximum error {100 * maximum_error:.4g}%, "
        f"RMS error {100 * rms_error:.4g}%"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    figure.clear()
    return output_path
