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

import argparse
import warnings
from pathlib import Path

import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from gprMax.receivers import Rx
from gprMax.toolboxes.Utilities.receiver_identity import natural_key, receiver_recorded
from gprMax.toolboxes.Utilities.trace_time import read_time_history
from gprMax.utilities.utilities import fft_power, handle_plot_output


def fft_plot_range(freqs, power, floor_db=-60):
    """Return a positive-frequency slice suitable for plotting an FFT."""
    positive = np.flatnonzero(freqs >= 0)
    if positive.size == 0:
        return np.s_[0:1]

    finite = positive[np.isfinite(power[positive])]
    if finite.size == 0:
        return np.s_[0 : positive[-1] + 1]

    peak = finite[np.argmax(power[finite])]
    below = np.flatnonzero(power[peak:] < floor_db)
    if below.size:
        stop = peak + below[0] + 1
    elif peak == 0:
        stop = positive[-1] + 1
    else:
        stop = max(peak * 4, peak + 1)
    stop = min(max(stop, 1), positive[-1] + 1)
    return np.s_[0:stop]


def mpl_plot(filename, outputs=Rx.defaultoutputs, fft=False, show=True, output_dir=None):
    """Plots electric and magnetic fields and currents from all receiver points
        in the given output file. Each receiver point is plotted in a new figure
        window.

    Args:
        filename: string of filename (including path) of output file.
        outputs: list of field/current components to plot.
        fft: boolean flag to plot FFT.
        show: boolean flag to display each plot interactively; if False, or
            if the current matplotlib backend is not interactive, each plot
            is saved to file instead.
        output_dir: Path or str of output directory for saved plots. Defaults
            to None (saving in the same directory as ``filename``).

    Returns:
        plt: matplotlib plot object.
    """
    if output_dir is not None:
        if not str(output_dir).strip():
            raise ValueError("Output directory cannot be empty")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(filename, "r") as output:
        return _mpl_plot_file(Path(filename), outputs, fft, show, output, output_dir=output_dir)


def _mpl_plot_file(file, outputs, fft, show, f, output_dir=None):
    """Plot while the public entry point owns the file's lifetime."""

    # Paths to grid(s) to traverse for outputs
    paths = ["/"]

    # Check if any subgrids and add path(s)
    is_subgrids = isinstance(f.get("/subgrids"), h5py.Group)
    if is_subgrids:
        paths = paths + [
            "/subgrids/" + path + "/"
            for path in f["/subgrids"].keys()
            if isinstance(f["/subgrids"][path], h5py.Group)
        ]

    # Get number of receivers in grid(s)
    paths = [
        path
        for path in paths
        if f.get(path) is not None
        and f[path].attrs.get("nrx", 0) > 0
        and isinstance(f.get(path + "rxs"), h5py.Group)
        and len(f[path + "rxs"]) > 0
    ]

    # Check there are any receivers
    if not paths:
        f.close()
        raise ValueError(f"No receivers found in {file}")

    # Loop through all grids
    recorded_receivers = False
    for path in paths:
        # Check for single output component when doing a FFT
        if fft and not len(outputs) == 1:
            f.close()
            raise ValueError("A single output must be specified when using the -fft option")

        # New plot for each receiver
        for receiver_key in sorted(f[path + "rxs"], key=natural_key):
            rx = receiver_key.removeprefix("rx")
            rxpath = path + "rxs/" + receiver_key + "/"
            if not receiver_recorded(f[rxpath]):
                warnings.warn(
                    f"{rxpath} was not recorded (record=false); skipping it",
                    stacklevel=2,
                )
                continue
            recorded_receivers = True
            availableoutputs = list(f[rxpath].keys())
            name = f[rxpath].attrs.get("Name", "")
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            figname = f"{rxpath} - {name}" if name else rxpath

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
                    f.close()
                    raise ValueError(
                        f"{output} output requested to plot, but "
                        + f"the available output for receiver {rx} is "
                        + f"{', '.join(availableoutputs)}"
                    )

                history = read_time_history(f[rxpath + output], allow_matrix=not fft)
                outputdata = history.samples * polarity
                time, dt = history.time, history.dt

                # Plotting if FFT required
                if fft:
                    # FFT
                    freqs, power = fft_power(outputdata, dt)
                    pltrange = fft_plot_range(freqs, power)

                    # Plot time history of output component
                    fig, (ax1, ax2) = plt.subplots(
                        nrows=1,
                        ncols=2,
                        num=figname,
                        figsize=(20, 10),
                        facecolor="w",
                        edgecolor="w",
                    )
                    line1 = ax1.plot(time, outputdata, "r", lw=2, label=outputtext)
                    ax1.set_xlabel("Time [s]")
                    ax1.set_ylabel(outputtext + " field strength [V/m]")
                    ax1.set_xlim([time[0], time[-1]])
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
                        plt.setp(markerline, "markerfacecolor", "g", "markeredgecolor", "g")
                    elif "I" in outputs[0]:
                        plt.setp(line1, color="b")
                        plt.setp(line2, color="b")
                        plt.setp(ax1, ylabel=outputtext + " current [A]")
                        plt.setp(stemlines, "color", "b")
                        plt.setp(markerline, "markerfacecolor", "b", "markeredgecolor", "b")

                # Plotting if no FFT required
                else:
                    fig, ax = plt.subplots(
                        subplot_kw=dict(
                            xlabel="Time [s]",
                            ylabel=outputtext + " field strength [V/m]",
                        ),
                        num=figname,
                        figsize=(20, 10),
                        facecolor="w",
                        edgecolor="w",
                    )
                    line = ax.plot(time, outputdata, "r", lw=2, label=outputtext)
                    ax.set_xlim([time[0], time[-1]])
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
                plt_cols = 3 if any(output.startswith("I") for output in outputs) else 2

                fig, axs = plt.subplots(
                    subplot_kw=dict(xlabel="Time [s]"),
                    num=figname,
                    figsize=(20, 10),
                    nrows=3,
                    ncols=plt_cols,
                    facecolor="w",
                    edgecolor="w",
                )

                for output in outputs:
                    # Check for polarity of output and if requested output
                    # is in file
                    if output[-1] == "-":
                        polarity = -1
                        outputtext = "-" + output[0:-1]
                        output = output[0:-1]
                    else:
                        polarity = 1
                        outputtext = output

                    # Check if requested output is in file
                    if output not in availableoutputs:
                        f.close()
                        raise ValueError(
                            f"Output(s) requested to plot: "
                            + f"{', '.join(outputs)}, but available output(s) "
                            + f"for receiver {rx} in the file: "
                            + f"{', '.join(availableoutputs)}"
                        )

                    history = read_time_history(f[rxpath + output], allow_matrix=True)
                    outputdata = history.samples * polarity
                    time = history.time

                    if output == "Ex":
                        axs[0, 0].plot(time, outputdata, "r", lw=2, label=outputtext)
                        axs[0, 0].set_ylabel(outputtext + ", field strength [V/m]")
                    elif output == "Ey":
                        axs[1, 0].plot(time, outputdata, "r", lw=2, label=outputtext)
                        axs[1, 0].set_ylabel(outputtext + ", field strength [V/m]")
                    elif output == "Ez":
                        axs[2, 0].plot(time, outputdata, "r", lw=2, label=outputtext)
                        axs[2, 0].set_ylabel(outputtext + ", field strength [V/m]")
                    elif output == "Hx":
                        axs[0, 1].plot(time, outputdata, "g", lw=2, label=outputtext)
                        axs[0, 1].set_ylabel(outputtext + ", field strength [A/m]")
                    elif output == "Hy":
                        axs[1, 1].plot(time, outputdata, "g", lw=2, label=outputtext)
                        axs[1, 1].set_ylabel(outputtext + ", field strength [A/m]")
                    elif output == "Hz":
                        axs[2, 1].plot(time, outputdata, "g", lw=2, label=outputtext)
                        axs[2, 1].set_ylabel(outputtext + ", field strength [A/m]")
                    elif output == "Ix":
                        axs[0, 2].plot(time, outputdata, "b", lw=2, label=outputtext)
                        axs[0, 2].set_ylabel(outputtext + ", current [A]")
                    elif output == "Iy":
                        axs[1, 2].plot(time, outputdata, "b", lw=2, label=outputtext)
                        axs[1, 2].set_ylabel(outputtext + ", current [A]")
                    elif output == "Iz":
                        axs[2, 2].plot(time, outputdata, "b", lw=2, label=outputtext)
                        axs[2, 2].set_ylabel(outputtext + ", current [A]")
                for ax in fig.axes:
                    if ax.lines:
                        ax.set_xlim(
                            [
                                min(line.get_xdata()[0] for line in ax.lines),
                                max(line.get_xdata()[-1] for line in ax.lines),
                            ]
                        )
                    ax.grid(which="both", axis="both", linestyle="-.")

            # Show or save this receiver's figure now, rather than after the
            # loop over all receivers/paths - otherwise only the last
            # receiver's figure would ever be shown/saved.
            suffix = "_" + rxpath.strip("/").replace("/", "_")
            save_base = (output_dir / file.stem) if output_dir is not None else file
            handle_plot_output(plt, fig, str(save_base), suffix=suffix, show=show)

    if not recorded_receivers:
        f.close()
        raise ValueError(
            f"No recorded receivers found in {file}; every receiver has Recorded=false"
        )

    f.close()

    return plt


def main(argv=None):
    """Plot A-scan from command-line arguments.

    Args:
        argv: argument list excluding the program name. Defaults to
            ``sys.argv[1:]`` when None, matching ``argparse`` behaviour.

    Returns:
        plt: matplotlib plot object from :func:`mpl_plot`.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Plots electric and magnetic fields and "
        + "currents from all receiver points in the given output file. "
        + "Each receiver point is plotted in a new figure window.",
        usage="python -m gprMax.toolboxes.Plotting.plot_Ascan outputfile",
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
    parser.add_argument(
        "-o",
        "--output-dir",
        "--output",
        default=None,
        dest="output_dir",
        help="output directory for saved plots",
    )
    args = parser.parse_args(argv)

    if args.output_dir is not None and not str(args.output_dir).strip():
        raise ValueError("Output directory cannot be empty")

    show = not (args.save or args.output_dir is not None)
    return mpl_plot(
        args.outputfile,
        args.outputs,
        fft=args.fft,
        show=show,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
