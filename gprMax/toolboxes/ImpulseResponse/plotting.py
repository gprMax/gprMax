# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

"""Plotting helpers for impulse-response waveform synthesis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_synthesis_results(results, *, receiver_index: int = 0, output=None, show=False):
    """Plot target source samples and one synthesised receiver component."""

    values = tuple(results)
    if not values:
        raise ValueError("at least one synthesis result is required")
    if receiver_index < 0 or receiver_index >= len(values[0].receivers):
        raise ValueError("receiver_index is outside the available receiver range")

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for result in values:
        waveform = result.waveform
        source_time = waveform.source_times * 1e9
        receiver = result.receivers[receiver_index]
        receiver_time = receiver.input.times * 1e9
        samples = receiver.samples
        if samples.ndim == 2:
            samples = samples[:, 0]
        axes[0].plot(source_time, waveform.samples, label=waveform.id)
        axes[1].plot(receiver_time, samples, label=waveform.id)
    axes[0].set(xlabel="Time (ns)", ylabel="Driving waveform", title="Synthesised sources")
    axes[1].set(
        xlabel="Time (ns)",
        ylabel=values[0].receivers[receiver_index].input.quantity,
        title=values[0].receivers[receiver_index].input.path,
    )
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    if output is not None:
        figure.savefig(Path(output), dpi=180)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return figure
