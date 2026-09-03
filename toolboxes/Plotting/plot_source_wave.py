# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gprMax.utilities.utilities import fft_power, handle_plot_output, round_value
from gprMax.waveforms import Waveform

logger = logging.getLogger(__name__)


def check_timewindow(timewindow, dt):
    """Checks and sets time window and number of iterations.

    Args:
        timewindow: float of time window.
        dt: flost of time discretisation.

    Returns:
        timewindow: float of time window.
        iterations: int of number of interations.
    """

    # Time window could be a string, float or int, so convert to string then check
    timewindow = str(timewindow)

    try:
        timewindow = int(timewindow)
        if timewindow < 1:
            raise ValueError("Number of iterations must be greater than zero")
        iterations = timewindow
        timewindow = (timewindow - 1) * dt

    except ValueError:
        timewindow = float(timewindow)
        if timewindow > 0:
            iterations = round_value((timewindow / dt)) + 1
        else:
            raise ValueError("Time window must have a value greater than zero")

    return timewindow, iterations


def mpl_plot(w, timewindow, dt, iterations, fft=False, show=True):
    """Plots waveform and prints useful information about its properties.

    Args:
        w: Waveform class instance.
        timewindow: float of time window.
        dt: float of time discretisation.
        iterations: int of number of iterations.
        fft: boolean flag to plot FFT.
        show: boolean flag to display the plot interactively; if False, or
            if the current matplotlib backend is not interactive, the plot
            is saved to file instead.

    Returns:
        plt: matplotlib plot object.
    """

    time = np.linspace(0, (iterations - 1) * dt, num=iterations)
    waveform = np.zeros(len(time))
    timeiter = np.nditer(time, flags=["c_index"])

    while not timeiter.finished:
        waveform[timeiter.index] = w.calculate_value(timeiter[0], dt)
        timeiter.iternext()

    logger.info("Waveform characteristics...")
    logger.info(f"Type: {w.type}")
    logger.info(f"Maximum (absolute) amplitude: {np.max(np.abs(waveform)):g}")

    if w.freq and w.type != "gaussian" and w.type != "impulse":
        logger.info(f"Centre frequency: {w.freq:g} Hz")

    if w.type in [
        "gaussian",
        "gaussiandot",
        "gaussiandotnorm",
        "gaussianprime",
        "gaussiandoubleprime",
    ]:
        delay = 1 / w.freq
        logger.info(f"Time to centre of pulse: {delay:g} s")
    elif w.type in ["gaussiandotdot", "gaussiandotdotnorm", "ricker"]:
        delay = np.sqrt(2) / w.freq
        logger.info(f"Time to centre of pulse: {delay:g} s")

    logger.info(f"Time window: {timewindow:g} s ({iterations} iterations)")
    logger.info(f"Time step: {dt:g} s")

    if fft:
        # FFT
        freqs, power = fft_power(waveform, dt)

        # Set plotting range to 4 times frequency at max power of waveform or
        # 4 times the centre frequency
        positive = np.flatnonzero(freqs >= 0)
        finite = positive[np.isfinite(power[positive])]
        freqmaxpower = finite[np.argmax(power[finite])] if finite.size else 0
        upper_frequency = 4 * max(freqs[freqmaxpower], w.freq or 0)
        above = np.flatnonzero(freqs > upper_frequency)
        pltrange = above[0] if above.size else max(1, len(freqs) // 2)
        pltrange = np.s_[0:pltrange]

        fig, (ax1, ax2) = plt.subplots(
            nrows=1, ncols=2, num=w.type, figsize=(20, 10), facecolor="w", edgecolor="w"
        )

        # Plot waveform
        ax1.plot(time, waveform, "r", lw=2)
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Amplitude")

        # Plot frequency spectra
        markerline, stemlines, baseline = ax2.stem(freqs[pltrange], power[pltrange], "-.")
        plt.setp(baseline, "linewidth", 0)
        plt.setp(stemlines, "color", "r")
        plt.setp(markerline, "markerfacecolor", "r", "markeredgecolor", "r")
        ax2.plot(freqs[pltrange], power[pltrange], "r", lw=2)
        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_ylabel("Power [dB]")

    else:
        fig, ax1 = plt.subplots(num=w.type, figsize=(10, 10), facecolor="w", edgecolor="w")

        # Plot waveform
        ax1.plot(time, waveform, "r", lw=2)
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Amplitude")

    # Turn on grid
    [ax.grid(which="both", axis="both", linestyle="-.") for ax in fig.axes]

    savefile = Path.cwd() / w.type
    handle_plot_output(plt, fig, str(savefile), show=show)

    return plt


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    plottable_waveforms = [wave_type for wave_type in Waveform.types if wave_type != "user"]
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Plot built-in waveforms that can be used for sources.",
        usage="cd gprMax; python -m toolboxes.Plotting.plot_source_wave type amp freq timewindow dt",
    )
    parser.add_argument("type", help="type of waveform", choices=plottable_waveforms)
    parser.add_argument("amp", type=float, help="amplitude of waveform")
    parser.add_argument("freq", type=float, help="centre frequency of waveform")
    parser.add_argument("timewindow", help="time window to view waveform")
    parser.add_argument("dt", type=float, help="time step to view waveform")
    parser.add_argument("-fft", action="store_true", default=False, help="plot FFT of waveform")
    parser.add_argument(
        "-save",
        action="store_true",
        default=False,
        help="save plot directly to file, i.e. do not display",
    )
    args = parser.parse_args()

    # Check waveform parameters
    if args.type.lower() not in plottable_waveforms:
        raise ValueError(f"The waveform must have one of: {', '.join(plottable_waveforms)}")
    if args.freq <= 0 and args.type != "impulse":
        raise ValueError("The waveform requires an excitation frequency greater than zero")

    # Create waveform instance
    w = Waveform()
    w.type = args.type
    w.amp = args.amp
    w.freq = args.freq

    timewindow, iterations = check_timewindow(args.timewindow, args.dt)
    mpl_plot(w, timewindow, args.dt, iterations, fft=args.fft, show=not args.save)
