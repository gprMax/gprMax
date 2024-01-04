# Copyright (C) 2015-2024: The University of Edinburgh, United Kingdom
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
import matplotlib.pyplot as plt
import numpy as np

from ..Utilities.outputfiles_merge import get_output_data

logger = logging.getLogger(__name__)


def mpl_plot(filename, outputdata, dt, rxnumber, rxcomponent, save=False):
    """Creates a plot of the B-scan.

    Args:
        filename: string of filename (including path) of output file.
        outputdata: array of A-scans, i.e. B-scan data.
        dt: float of temporal resolution of the model.
        rxnumber: int of receiver output number.
        rxcomponent: string of receiver output field/current component.
        save: boolean flag to save plot to file.

    Returns:
        plt: matplotlib plot object.
    """

    file = Path(filename)

    fig = plt.figure(num=file.stem + " - rx" + str(rxnumber), figsize=(20, 10), facecolor="w", edgecolor="w")
    plt.imshow(
        outputdata,
        extent=[0, outputdata.shape[1], outputdata.shape[0] * dt, 0],
        interpolation="nearest",
        aspect="auto",
        cmap="seismic",
        vmin=-np.amax(np.abs(outputdata)),
        vmax=np.amax(np.abs(outputdata)),
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

    if save:
        # Save a PDF of the figure
        fig.savefig(filename[:-3] + ".pdf", dpi=None, format="pdf", bbox_inches="tight", pad_inches=0.1)
        # Save a PNG of the figure
        # fig.savefig(filename[:-3] + '.png', dpi=150, format='png',
        #             bbox_inches='tight', pad_inches=0.1)

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
        choices=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Ix", "Iy", "Iz"],
    )
    parser.add_argument(
        "-gather", action="store_true", default=False, help="gather together all receiver outputs in file"
    )
    parser.add_argument(
        "-save", action="store_true", default=False, help="save plot directly to file, i.e. do not display"
    )
    args = parser.parse_args()

    # Open output file and read number of outputs (receivers)
    f = h5py.File(args.outputfile, "r")
    nrx = f.attrs["nrx"]
    f.close()

    # Check there are any receivers
    if nrx == 0:
        logger.exception(f"No receivers found in {args.outputfile}")
        raise ValueError

    for rx in range(1, nrx + 1):
        outputdata, dt = get_output_data(args.outputfile, rx, args.rx_component)
        if args.gather:
            if rx == 1:
                rxsgather = outputdata
            rxsgather = np.column_stack((rxsgather, outputdata))
        else:
            plthandle = mpl_plot(args.outputfile, outputdata, dt, rx, args.rx_component, save=args.save)

    # Plot all receivers from single output file together if required
    if args.gather:
        plthandle = mpl_plot(args.outputfile, rxsgather, dt, rx, args.rx_component, save=args.save)

    plthandle.show()
