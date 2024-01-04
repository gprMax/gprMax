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
import glob
import logging
import os

import h5py
import numpy as np

from gprMax._version import __version__
from gprMax.utilities.utilities import natural_keys

logger = logging.getLogger(__name__)


def get_output_data(filename, rxnumber, rxcomponent):
    """Gets B-scan output data from a model.

    Args:
        filename: string of tilename (including path) of output file.
        rxnumber: int of receiver output number.
        rxcomponent: string of receiver output field/current component.

    Returns:
        outputdata: array of A-scans, i.e. B-scan data.
        dt: float of temporal resolution of the model.
    """

    # Open output file and read some attributes
    with h5py.File(filename, "r") as f:
        nrx = f.attrs["nrx"]
        dt = f.attrs["dt"]

        # Check there are any receivers
        if nrx == 0:
            logger.exception(f"No receivers found in {filename}")
            raise ValueError

        path = "/rxs/rx" + str(rxnumber) + "/"
        availableoutputs = list(f[path].keys())

        # Check if requested output is in file
        if rxcomponent not in availableoutputs:
            logger.exception(
                f"{rxcomponent} output requested to plot, but the "
                + f"available output for receiver 1 is "
                + f"{', '.join(availableoutputs)}"
            )
            raise ValueError

        outputdata = f[path + "/" + rxcomponent]
        outputdata = np.array(outputdata)

    return outputdata, dt


def merge_files(outputfiles, removefiles=False):
    """Merges traces (A-scans) from multiple output files into one new file,
        then optionally removes the series of output files.

    Args:
        outputfiles: list of output files to be merged.
        removefiles: boolean flag to remove individual output files after merge.
    """

    merged_outputfile = os.path.commonprefix(outputfiles) + "_merged.h5"

    # Combined output file
    fout = h5py.File(merged_outputfile, "w")

    for i, outputfile in enumerate(outputfiles):
        fin = h5py.File(outputfile, "r")
        nrx = fin.attrs["nrx"]

        # Write properties for merged file on first iteration
        if i == 0:
            fout.attrs["gprMax"] = __version__
            fout.attrs["Iterations"] = fin.attrs["Iterations"]
            fout.attrs["nx_ny_nz"] = fin.attrs["nx_ny_nz"]
            fout.attrs["dx_dy_dz"] = fin.attrs["dx_dy_dz"]
            fout.attrs["dt"] = fin.attrs["dt"]
            fout.attrs["nsrc"] = fin.attrs["nsrc"]
            fout.attrs["nrx"] = fin.attrs["nrx"]
            fout.attrs["srcsteps"] = fin.attrs["srcsteps"]
            fout.attrs["rxsteps"] = fin.attrs["rxsteps"]

            for rx in range(1, nrx + 1):
                path = "/rxs/rx" + str(rx)
                grp = fout.create_group(path)
                availableoutputs = list(fin[path].keys())
                for output in availableoutputs:
                    grp.create_dataset(
                        output, (fout.attrs["Iterations"], len(outputfiles)), dtype=fin[path + "/" + output].dtype
                    )

        # For all receivers
        for rx in range(1, nrx + 1):
            path = "/rxs/rx" + str(rx) + "/"
            availableoutputs = list(fin[path].keys())
            # For all receiver outputs
            for output in availableoutputs:
                fout[path + "/" + output][:, i] = fin[path + "/" + output][:]

        fin.close()
    fout.close()

    if removefiles:
        for outputfile in outputfiles:
            os.remove(outputfile)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Merges traces (A-scans) from multiple "
        + "output files into one new file, then "
        + "optionally removes the series of output files.",
        usage="cd gprMax; python -m tools.outputfiles_merge basefilename",
    )
    parser.add_argument("basefilename", help="base name of output file series including path")
    parser.add_argument(
        "--remove-files", action="store_true", default=False, help="flag to remove individual output files after merge"
    )
    args = parser.parse_args()

    files = glob.glob(args.basefilename + "*.h5")
    outputfiles = [filename for filename in files if "_merged" not in filename]
    outputfiles.sort(key=natural_keys)
    merge_files(outputfiles, removefiles=args.remove_files)
