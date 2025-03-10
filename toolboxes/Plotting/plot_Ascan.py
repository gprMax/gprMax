# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, and John Hartley
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

import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from gprMax.receivers import Rx
from gprMax.utilities.utilities import fft_power

logger = logging.getLogger(__name__)


def mpl_plot(filename, outputs=Rx.defaultoutputs, fft=False, save=False):
    """Plots electric and magnetic fields and currents from all receiver points
        in the given output file. Each receiver point is plotted in a new figure
        window.

    Args:
        filename: string of filename (including path) of output file.
        outputs: list of field/current components to plot.
        fft: boolean flag to plot FFT.
        save: boolean flag to save plot to file.

    Returns:
        plt: matplotlib plot object.
    """

    file = Path(filename)

    # Open output file and read iterations
    f = h5py.File(file, "r")

    # Paths to grid(s) to traverse for outputs
    paths = ["/"]

    # Check if any subgrids and add path(s)
    is_subgrids = "/subgrids" in f
    if is_subgrids:
        paths = paths + ["/subgrids/" + path + "/" for path in f["/subgrids"].keys()]

    # Get number of receivers in grid(s)
    nrxs = []
    for path in paths:
        if f[path].attrs["nrx"] > 0:
            nrxs.append(f[path].attrs["nrx"])
        else:
            paths.remove(path)

    # Check there are any receivers
    if not paths:
        logger.exception(f"No receivers found in {file}")
        raise ValueError

    # Loop through all grids
    for path in paths:
        iterations = f[path].attrs["Iterations"]
        nrx = f[path].attrs["nrx"]
        dt = f[path].attrs["dt"]
        time = np.linspace(0, (iterations - 1) * dt, num=iterations)

        # Check for single output component when doing a FFT
        if fft and not len(outputs) == 1:
            logger.exception(
                "A single output must be specified when using " + "the -fft option"
            )
            raise ValueError

        # New plot for each receiver
        for rx in range(1, nrx + 1):
            rxpath = path + "rxs/rx" + str(rx) + "/"
            availableoutputs = list(f[rxpath].keys())

            # If only a single output is required, create one subplot
            if len(outputs) == 1:
                # Check for polarity of output and if requested output is in file
                if outputs[0][-1] == "-":
                    polarity = -1
                    outputtext = "-" + outputs[0][0:-1]
                    output = outputs[0][0:-1]
                else:
                    polarity = 1
                    outputtext = outputs[0]
                    output = outputs[0]

                if output not in availableoutputs:
                    logger.exception(
                        f"{output} output requested to plot, but "
                        + f"the available output for receiver 1 is "
                        + f"{', '.join(availableoutputs)}"
                    )
                    raise ValueError

                outputdata = f[rxpath + output][:] * polarity

                # Plotting if FFT required
                if fft:
                    # FFT
                    freqs, power = fft_power(outputdata, dt)
                    freqmaxpower = np.where(np.isclose(power, 0))[0][0]

                    # Set plotting range to -60dB from maximum power or 4 times
                    # frequency at maximum power
                    try:
                        pltrange = (
                            np.where(power[freqmaxpower:] < -60)[0][0]
                            + freqmaxpower
                            + 1
                        )
                    except:
                        pltrange = freqmaxpower * 4

                    pltrange = np.s_[0:pltrange]

                    # Plot time history of output component
                    fig, (ax1, ax2) = plt.subplots(
                        nrows=1,
                        ncols=2,
                        num=rxpath + " - " + f[rxpath].attrs["Name"],
                        figsize=(20, 10),
                        facecolor="w",
                        edgecolor="w",
                    )
                    line1 = ax1.plot(time, outputdata, "r", lw=2, label=outputtext)
                    ax1.set_xlabel("Time [s]")
                    ax1.set_ylabel(outputtext + " field strength [V/m]")
                    ax1.set_xlim([0, np.amax(time)])
                    ax1.grid(which="both", axis="both", linestyle="-.")

                    # Plot frequency spectra
                    markerline, stemlines, baseline = ax2.stem(
                        freqs[pltrange], power[pltrange], "-."
                    )
                    plt.setp(baseline, "linewidth", 0)
                    plt.setp(stemlines, "color", "r")
                    plt.setp(markerline, "markerfacecolor", "r", "markeredgecolor", "r")
                    line2 = ax2.plot(freqs[pltrange], power[pltrange], "r", lw=2)
                    ax2.set_xlabel("Frequency [Hz]")
                    ax2.set_ylabel("Power [dB]")
                    ax2.grid(which="both", axis="both", linestyle="-.")

                    # Change colours and labels for magnetic field components
                    # or currents
                    if "H" in outputs[0]:
                        plt.setp(line1, color="g")
                        plt.setp(line2, color="g")
                        plt.setp(ax1, ylabel=outputtext + " field strength [A/m]")
                        plt.setp(stemlines, "color", "g")
                        plt.setp(
                            markerline, "markerfacecolor", "g", "markeredgecolor", "g"
                        )
                    elif "I" in outputs[0]:
                        plt.setp(line1, color="b")
                        plt.setp(line2, color="b")
                        plt.setp(ax1, ylabel=outputtext + " current [A]")
                        plt.setp(stemlines, "color", "b")
                        plt.setp(
                            markerline, "markerfacecolor", "b", "markeredgecolor", "b"
                        )

                    plt.show()

                # Plotting if no FFT required
                else:
                    fig, ax = plt.subplots(
                        subplot_kw=dict(
                            xlabel="Time [s]",
                            ylabel=outputtext + " field strength [V/m]",
                        ),
                        num=rxpath + " - " + f[rxpath].attrs["Name"],
                        figsize=(20, 10),
                        facecolor="w",
                        edgecolor="w",
                    )
                    line = ax.plot(time, outputdata, "r", lw=2, label=outputtext)
                    ax.set_xlim([0, np.amax(time)])
                    # ax.set_ylim([-15, 20])
                    ax.grid(which="both", axis="both", linestyle="-.")

                    if "H" in output:
                        plt.setp(line, color="g")
                        plt.setp(ax, ylabel=outputtext + ", field strength [A/m]")
                    elif "I" in output:
                        plt.setp(line, color="b")
                        plt.setp(ax, ylabel=outputtext + ", current [A]")

            # If multiple outputs required, create all nine subplots and
            # populate only the specified ones
            else:
                fig, ax = plt.subplots(
                    subplot_kw=dict(xlabel="Time [s]"),
                    num=rxpath + " - " + f[rxpath].attrs["Name"],
                    figsize=(20, 10),
                    facecolor="w",
                    edgecolor="w",
                )
                if len(outputs) == 9:
                    gs = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.3)
                else:
                    gs = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.3)

                for output in outputs:
                    # Check for polarity of output and if requested output
                    # is in file
                    if output[-1] == "m":
                        polarity = -1
                        outputtext = "-" + output[0:-1]
                        output = output[0:-1]
                    else:
                        polarity = 1
                        outputtext = output

                    # Check if requested output is in file
                    if output not in availableoutputs:
                        logger.exception(
                            f"Output(s) requested to plot: "
                            + f"{', '.join(outputs)}, but available output(s) "
                            + f"for receiver {rx} in the file: "
                            + f"{', '.join(availableoutputs)}"
                        )
                        raise ValueError

                    outputdata = f[rxpath + output][:] * polarity

                    if output == "Ex":
                        ax = plt.subplot(gs[0, 0])
                        ax.plot(time, outputdata, "r", lw=2, label=outputtext)
                        ax.set_ylabel(outputtext + ", field strength [V/m]")
                    # ax.set_ylim([-15, 20])
                    elif output == "Ey":
                        ax = plt.subplot(gs[1, 0])
                        ax.plot(time, outputdata, "r", lw=2, label=outputtext)
                        ax.set_ylabel(outputtext + ", field strength [V/m]")
                    # ax.set_ylim([-15, 20])
                    elif output == "Ez":
                        ax = plt.subplot(gs[2, 0])
                        ax.plot(time, outputdata, "r", lw=2, label=outputtext)
                        ax.set_ylabel(outputtext + ", field strength [V/m]")
                    # ax.set_ylim([-15, 20])
                    elif output == "Hx":
                        ax = plt.subplot(gs[0, 1])
                        ax.plot(time, outputdata, "g", lw=2, label=outputtext)
                        ax.set_ylabel(outputtext + ", field strength [A/m]")
                    # ax.set_ylim([-0.03, 0.03])
                    elif output == "Hy":
                        ax = plt.subplot(gs[1, 1])
                        ax.plot(time, outputdata, "g", lw=2, label=outputtext)
                        ax.set_ylabel(outputtext + ", field strength [A/m]")
                    # ax.set_ylim([-0.03, 0.03])
                    elif output == "Hz":
                        ax = plt.subplot(gs[2, 1])
                        ax.plot(time, outputdata, "g", lw=2, label=outputtext)
                        ax.set_ylabel(outputtext + ", field strength [A/m]")
                    # ax.set_ylim([-0.03, 0.03])
                    elif output == "Ix":
                        ax = plt.subplot(gs[0, 2])
                        ax.plot(time, outputdata, "b", lw=2, label=outputtext)
                        ax.set_ylabel(outputtext + ", current [A]")
                    elif output == "Iy":
                        ax = plt.subplot(gs[1, 2])
                        ax.plot(time, outputdata, "b", lw=2, label=outputtext)
                        ax.set_ylabel(outputtext + ", current [A]")
                    elif output == "Iz":
                        ax = plt.subplot(gs[2, 2])
                        ax.plot(time, outputdata, "b", lw=2, label=outputtext)
                        ax.set_ylabel(outputtext + ", current [A]")
                for ax in fig.axes:
                    ax.set_xlim([0, np.amax(time)])
                    ax.grid(which="both", axis="both", linestyle="-.")

    f.close()

    if save:
        # Save a PDF of the figure
        fig.savefig(
            filename[:-3] + ".pdf",
            dpi=None,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )
        # Save a PNG of the figure
        # fig.savefig(filename[:-3] + '.png', dpi=150, format='png',
        #             bbox_inches='tight', pad_inches=0.1)

    return plt


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Plots electric and magnetic fields and "
        + "currents from all receiver points in the given output file. "
        + "Each receiver point is plotted in a new figure window.",
        usage="cd gprMax; python -m toolboxes.Plotting.plot_Ascan outputfile",
    )
    parser.add_argument("outputfile", help="name of output file including path")
    parser.add_argument(
        "--outputs",
        help="outputs to be plotted",
        default=Rx.defaultoutputs,
        choices=[
            "Ex",
            "Ey",
            "Ez",
            "Hx",
            "Hy",
            "Hz",
            "Ix",
            "Iy",
            "Iz",
            "Ex-",
            "Ey-",
            "Ez-",
            "Hx-",
            "Hy-",
            "Hz-",
            "Ix-",
            "Iy-",
            "Iz-",
        ],
        nargs="+",
    )
    parser.add_argument(
        "-fft",
        action="store_true",
        default=False,
        help="plot FFT (single output must be specified)",
    )
    parser.add_argument(
        "-save",
        action="store_true",
        default=False,
        help="save plot directly to file, i.e. do not display",
    )
    args = parser.parse_args()

    plthandle = mpl_plot(args.outputfile, args.outputs, fft=args.fft, save=args.save)

    plthandle.show()
