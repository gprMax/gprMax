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
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from gprMax.utilities.utilities import handle_plot_output

from ..Utilities.outputfiles_merge import get_output_data
from ..Utilities.receiver_identity import natural_key
from ..Utilities.trace_time import receiver_time_offset


def gather_receiver_outputs(filename, rxcomponent, *, return_time_offset=False):
    """Gather one component from all receivers without duplicating rx1."""
    with h5py.File(filename, "r") as output:
        nrx = int(output.attrs["nrx"])
        receivers = sorted(output.get("rxs", {}), key=natural_key)

    if nrx == 0:
        raise ValueError(f"No receivers found in {filename}")

    traces = []
    dt = None
    offset = None
    for key in receivers:
        rx = int(key.removeprefix("rx"))
        outputdata, candidate_dt, candidate_offset = get_output_data(
            filename, rx, rxcomponent, return_time_offset=True
        )
        if dt is not None and (
            not np.isclose(dt, candidate_dt, rtol=1e-12, atol=0.0)
            or not np.isclose(offset, candidate_offset, rtol=1e-12, atol=1e-30)
            or outputdata.shape[0] != traces[0].shape[0]
        ):
            raise ValueError("Gathered receivers have inconsistent sample times")
        dt, offset = candidate_dt, candidate_offset
        traces.append(np.asarray(outputdata))

    if return_time_offset:
        return np.column_stack(traces), dt, offset
    return np.column_stack(traces), dt


def mpl_plot(
    filename,
    outputdata,
    dt,
    rxnumber,
    rxcomponent,
    show=True,
    trace_group=None,
    time_offset=None,
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
        time_offset: physical sample-zero time from the loader. Without it,
            use the legacy receiver component convention (voltage defaults to zero).

    Returns:
        plt: matplotlib plot object.
    """

    file = Path(filename)
    outputdata = np.asarray(outputdata)
    if (
        outputdata.ndim != 2
        or 0 in outputdata.shape
        or not np.all(np.isfinite(outputdata))
        or np.iscomplexobj(outputdata)
    ):
        raise ValueError("A B-scan must be a nonempty finite real time-by-trace matrix")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("B-scan sample interval must be finite and positive")
    time_offset = receiver_time_offset(rxcomponent, dt) if time_offset is None else float(time_offset)
    if not np.isfinite(time_offset):
        raise ValueError("B-scan time offset must be finite")

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
        extent=[0, outputdata.shape[1], time_offset + (outputdata.shape[0] - 0.5) * dt, time_offset - 0.5 * dt],
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
        usage="python -m gprMax.toolboxes.Plotting.plot_Bscan outputfile output",
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
        help="trace group (e.g. ports/receive) or receiver selector (e.g. name:surface)",
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
        if args.rx_component != "Vtotal" and not args.trace_group.startswith(("name:", "study:", "build:", "rxs/")):
            parser.error("--trace-group requires the Vtotal component")
        outputdata, dt, offset = get_output_data(
            args.outputfile,
            1,
            args.rx_component,
            trace_group=args.trace_group,
            return_time_offset=True,
        )
        mpl_plot(
            args.outputfile,
            outputdata,
            dt,
            1,
            args.rx_component,
            show=not args.save,
            trace_group=args.trace_group,
            time_offset=offset,
        )
    elif args.rx_component == "Vtotal":
        parser.error("Vtotal requires --trace-group")
    elif args.gather:
        rxsgather, dt, offset = gather_receiver_outputs(args.outputfile, args.rx_component, return_time_offset=True)
        with h5py.File(args.outputfile, "r") as f:
            nrx = int(f.attrs["nrx"])
        mpl_plot(args.outputfile, rxsgather, dt, nrx, args.rx_component, show=not args.save, time_offset=offset)
    else:
        with h5py.File(args.outputfile, "r") as f:
            nrx = int(f.attrs["nrx"])
            receivers = sorted(f.get("rxs", {}), key=natural_key)
        if nrx == 0:
            raise ValueError(f"No receivers found in {args.outputfile}")
        for key in receivers:
            rx = int(key.removeprefix("rx"))
            outputdata, dt, offset = get_output_data(args.outputfile, rx, args.rx_component, return_time_offset=True)
            mpl_plot(args.outputfile, outputdata, dt, rx, args.rx_component, show=not args.save, time_offset=offset)
