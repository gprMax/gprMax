# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

"""Plotting helpers for processed FMCW results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .processing import ChannelResponse, DerampedSweep, FastTimeResponse


def _first_trace(values):
    array = np.asarray(values)
    return array if array.ndim == 1 else array[:, 0]


def plot_fmcw_result(
    channel: ChannelResponse,
    fast_time: FastTimeResponse,
    deramped: DerampedSweep | None = None,
    *,
    output: str | Path | None = None,
    show: bool = False,
):
    """Plot the first trace of the channel, fast-time output, and optional I/Q."""

    rows = 3 if deramped is not None else 2
    figure, axes = plt.subplots(rows, 1, figsize=(10, 3.2 * rows), constrained_layout=True)
    response = _first_trace(channel.response)
    axes[0].plot(channel.chirp.frequency / 1e6, 20 * np.log10(np.maximum(np.abs(response), 1e-300)))
    axes[0].set(xlabel="Frequency (MHz)", ylabel="Magnitude (dB)", title="FMCW channel")
    axes[0].grid(True, alpha=0.3)

    envelope = np.abs(_first_trace(fast_time.complex_envelope))
    coordinate = fast_time.delay * 1e9
    label = "Delay (ns)"
    if fast_time.range is not None:
        coordinate = fast_time.range
        label = "Two-way range (m)"
    axes[1].plot(coordinate, envelope)
    axes[1].set(xlabel=label, ylabel="Envelope", title="Processed fast-time response")
    axes[1].grid(True, alpha=0.3)
    if fast_time.range is None:
        recorded_duration = channel.target.receiver.dt * channel.target.receiver.samples.shape[0]
        axes[1].set_xlim(0, min(fast_time.delay[-1], recorded_duration) * 1e9)

    if deramped is not None:
        signal = _first_trace(deramped.complex_signal)
        axes[2].plot(deramped.slow_time * 1e3, signal.real, label="I")
        axes[2].plot(deramped.slow_time * 1e3, signal.imag, label="Q")
        axes[2].set(
            xlabel="Time within sweep (ms)",
            ylabel="Amplitude",
            title="Ideal deramped stretch-receiver samples",
        )
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    if output is not None:
        figure.savefig(output, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return figure
