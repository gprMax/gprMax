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
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from gprMax.utilities.utilities import handle_plot_output

from ..Utilities.outputfiles_merge import get_output_data


def gather_receiver_outputs(filename, rxcomponent):
    """Gather one component from all receivers without duplicating rx1."""
    with h5py.File(filename, "r") as output:
        nrx = int(output.attrs["nrx"])

    if nrx == 0:
        raise ValueError(f"No receivers found in {filename}")

    traces = []
    dt = None
    for rx in range(1, nrx + 1):
        outputdata, dt = get_output_data(filename, rx, rxcomponent)
        traces.append(np.asarray(outputdata))

    return np.column_stack(traces), dt


def mpl_plot(
    filename,
    outputdata,
    dt,
    rxnumber,
    rxcomponent,
    show=True,
    trace_group=None,
):
    """Creates a plot of the B-scan.

    Args:
        filename: string of filename (including path) of output file.
        outputdata: array of A-scans, i.e. B-scan data.
        dt: float of temporal resolution of the model.
        rxnumber: int of receiver output number.
        rxcomponent: string of receiver output field/current component.
        show: boolean flag to display the plot interactively; if False, or
            if the current matplotlib backend is not interactive, the plot
            is saved to file instead.
        trace_group: optional HDF5 group for a terminal-voltage B-scan.

    Returns:
        plt: matplotlib plot object.
    """

    file = Path(filename)

    trace_id = str(trace_group).strip("/") if trace_group else f"rx{rxnumber}"
    safe_trace_id = trace_id.replace("/", "_")
    fig = plt.figure(
        num=f"{file.stem} - {trace_id}",
        figsize=(20, 10),
        facecolor="w",
        edgecolor="w",
    )
    colour_limit = np.amax(np.abs(outputdata))
    if colour_limit == 0:
        colour_limit = 1

    plt.imshow(
        outputdata,
        extent=[0, outputdata.shape[1], outputdata.shape[0] * dt, 0],
        interpolation="nearest",
        aspect="auto",
        cmap="seismic",
        vmin=-colour_limit,
        vmax=colour_limit,
    )
    plt.xlabel("Trace number")
    plt.ylabel("Time [s]")

    # Grid properties
    ax = fig.gca()
    ax.grid(which="both", axis="both", linestyle="-.")

    cb = plt.colorbar()
    if "E" in rxcomponent:
        cb.set_label("Field strength [V/m]")
    elif "H" in rxcomponent:
        cb.set_label("Field strength [A/m]")
    elif "I" in rxcomponent:
        cb.set_label("Current [A]")
    elif rxcomponent.startswith("V"):
        cb.set_label("Voltage [V]")

    suffix = f"_{safe_trace_id}"
    handle_plot_output(plt, fig, str(file), suffix=suffix, show=show)

    return plt


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Plots a B-scan image.",
        usage="cd gprMax; python -m toolboxes.Plotting.plot_Bscan outputfile output",
    )
    parser.add_argument("outputfile", help="name of output file including path")
    parser.add_argument(
        "rx_component",
        help="name of output component to be plotted",
        choices=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Ix", "Iy", "Iz", "Vtotal"],
    )
    parser.add_argument(
        "--trace-group",
        default=None,
        help=("time-domain terminal-voltage group in a merged file, e.g. " "ports/receive, tls/tl1, or frills/frill1"),
    )
    parser.add_argument(
        "-gather",
        action="store_true",
        default=False,
        help="gather together all receiver outputs in file",
    )
    parser.add_argument(
        "-save",
        action="store_true",
        default=False,
        help="save plot directly to file, i.e. do not display",
    )
    args = parser.parse_args()

    if args.trace_group is not None:
        if args.gather:
            parser.error("--trace-group and -gather cannot be used together")
        if args.rx_component != "Vtotal":
            parser.error("--trace-group requires the Vtotal component")
        outputdata, dt = get_output_data(
            args.outputfile,
            1,
            args.rx_component,
            trace_group=args.trace_group,
        )
        mpl_plot(
            args.outputfile,
            outputdata,
            dt,
            1,
            args.rx_component,
            show=not args.save,
            trace_group=args.trace_group,
        )
    elif args.rx_component == "Vtotal":
        parser.error("Vtotal requires --trace-group")
    elif args.gather:
        rxsgather, dt = gather_receiver_outputs(args.outputfile, args.rx_component)
        with h5py.File(args.outputfile, "r") as f:
            nrx = int(f.attrs["nrx"])
        mpl_plot(args.outputfile, rxsgather, dt, nrx, args.rx_component, show=not args.save)
    else:
        with h5py.File(args.outputfile, "r") as f:
            nrx = int(f.attrs["nrx"])
        if nrx == 0:
            raise ValueError(f"No receivers found in {args.outputfile}")
        for rx in range(1, nrx + 1):
            outputdata, dt = get_output_data(args.outputfile, rx, args.rx_component)
            mpl_plot(args.outputfile, outputdata, dt, rx, args.rx_component, show=not args.save)
