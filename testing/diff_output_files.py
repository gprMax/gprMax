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

import logging
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def diff_output_files(filename1, filename2):
    """Calculates differences between two output files.

    Args:
        filename1: string of filename (including path) of output file 1.
        filename2: string of filename (including path) of output file 2.

    Returns:
        time: numpy array containing time.
        datadiffs: numpy array containing power (dB) of differences.
    """

    file1 = h5py.File(Path(filename1), "r")
    file2 = h5py.File(Path(filename2), "r")
    # Path to receivers in files
    path = "rxs/rx1/"

    # Get available field output component names
    outputs1 = list(file1[path].keys())
    outputs2 = list(file2[path].keys())
    if outputs1 != outputs2:
        logger.exception("Field output components are not the same in each file")
        raise ValueError

    # Check that type of float used to store fields matches
    floattype1 = file1[path + outputs1[0]].dtype
    floattype2 = file2[path + outputs2[0]].dtype
    if floattype1 != floattype2:
        logger.warning(
            f"Type of floating point number in test model ({file1[path + outputs1[0]].dtype}) "
            f"does not match type in reference solution ({file2[path + outputs2[0]].dtype})\n"
        )

    # Arrays for storing time
    time1 = np.zeros((file1.attrs["Iterations"]), dtype=floattype1)
    time1 = np.linspace(
        0, (file1.attrs["Iterations"] - 1), num=file1.attrs["Iterations"]
    )
    time2 = np.zeros((file2.attrs["Iterations"]), dtype=floattype2)
    time2 = np.linspace(
        0, (file2.attrs["Iterations"] - 1), num=file2.attrs["Iterations"]
    )

    # Arrays for storing field data
    data1 = np.zeros((file1.attrs["Iterations"], len(outputs1)), dtype=floattype1)
    data2 = np.zeros((file2.attrs["Iterations"], len(outputs2)), dtype=floattype2)
    for ID, name in enumerate(outputs1):
        data1[:, ID] = file1[path + str(name)][:]
        data2[:, ID] = file2[path + str(name)][:]
        if np.any(np.isnan(data1[:, ID])) or np.any(np.isnan(data2[:, ID])):
            logger.exception("Data contains NaNs")
            raise ValueError

    file1.close()
    file2.close()

    # Diffs
    datadiffs = np.zeros(data1.shape, dtype=np.float64)
    for i in range(len(outputs2)):
        maxi = np.amax(np.abs(data1[:, i]))
        datadiffs[:, i] = np.divide(
            np.abs(data2[:, i] - data1[:, i]),
            maxi,
            out=np.zeros_like(data1[:, i]),
            where=maxi != 0,
        )  # Replace any division by zero with zero

        # Calculate power (ignore warning from taking a log of any zero values)
        with np.errstate(divide="ignore"):
            datadiffs[:, i] = 20 * np.log10(datadiffs[:, i])
        # Replace any NaNs or Infs from zero division
        datadiffs[:, i][np.invert(np.isfinite(datadiffs[:, i]))] = 0

    return time1, datadiffs
