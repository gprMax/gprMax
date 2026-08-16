# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

"""Plotting helpers for processed stepped-frequency data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .processing import FrequencyResponse, TimeResponse


def plot_sfcw_result(
    frequency_response: FrequencyResponse,
    time_response: TimeResponse | None = None,
    *,
    output: str | Path | None = None,
    show: bool = False,
):
    """Plot complex frequency data and an optional reconstructed response."""

    rows = 2 if time_response is not None else 1
    fig, axes = plt.subplots(rows, 2, figsize=(13, 4.5 * rows), squeeze=False)
    frequency_ghz = frequency_response.frequency / 1e9
    response = frequency_response.response

    if response.ndim == 1:
        axes[0, 0].plot(frequency_ghz, np.abs(response), color="black")
        axes[0, 0].set(xlabel="Frequency [GHz]", ylabel="Magnitude", title="SFCW response")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(frequency_ghz, np.unwrap(np.angle(response)), color="black")
        axes[0, 1].set(xlabel="Frequency [GHz]", ylabel="Phase [rad]", title="Unwrapped phase")
        axes[0, 1].grid(True, alpha=0.3)
    else:
        magnitude = axes[0, 0].imshow(
            np.abs(response),
            extent=[0, response.shape[1] - 1, frequency_ghz[-1], frequency_ghz[0]],
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
        )
        axes[0, 0].set(xlabel="Trace number", ylabel="Frequency [GHz]", title="SFCW magnitude")
        fig.colorbar(magnitude, ax=axes[0, 0])
        phase = axes[0, 1].imshow(
            np.angle(response),
            extent=[0, response.shape[1] - 1, frequency_ghz[-1], frequency_ghz[0]],
            aspect="auto",
            interpolation="nearest",
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
        )
        axes[0, 1].set(xlabel="Trace number", ylabel="Frequency [GHz]", title="SFCW phase")
        fig.colorbar(phase, ax=axes[0, 1])

    if time_response is not None:
        time_ns = time_response.time * 1e9
        if time_response.real_bandpass.ndim == 1:
            axes[1, 0].plot(time_ns, time_response.real_bandpass, color="black")
            axes[1, 0].set(
                xlabel="Time [ns]",
                ylabel="Amplitude",
                title=f"Reconstructed response ({time_response.window} window)",
            )
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 1].plot(
                time_ns,
                np.abs(time_response.complex_envelope),
                color="black",
            )
            axes[1, 1].set(
                xlabel="Time [ns]",
                ylabel="Envelope magnitude",
                title="Complex baseband envelope",
            )
            axes[1, 1].grid(True, alpha=0.3)
        else:
            limit = float(np.max(np.abs(time_response.real_bandpass), initial=0.0)) or 1.0
            image = axes[1, 0].imshow(
                time_response.real_bandpass,
                extent=[
                    0,
                    time_response.real_bandpass.shape[1] - 1,
                    time_ns[-1],
                    time_ns[0],
                ],
                aspect="auto",
                interpolation="nearest",
                cmap="seismic",
                vmin=-limit,
                vmax=limit,
            )
            axes[1, 0].set(
                xlabel="Trace number",
                ylabel="Time [ns]",
                title=f"SFCW B-scan ({time_response.window} window)",
            )
            fig.colorbar(image, ax=axes[1, 0])
            envelope = axes[1, 1].imshow(
                np.abs(time_response.complex_envelope),
                extent=[
                    0,
                    time_response.complex_envelope.shape[1] - 1,
                    time_ns[-1],
                    time_ns[0],
                ],
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
            )
            axes[1, 1].set(xlabel="Trace number", ylabel="Time [ns]", title="Complex envelope")
            fig.colorbar(envelope, ax=axes[1, 1])

    fig.tight_layout()
    if output is not None:
        fig.savefig(Path(output), dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig
