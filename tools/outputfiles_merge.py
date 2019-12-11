# Copyright (C) 2015-2019: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
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
import os
from pathlib import Path

import h5py
import numpy as np

from gprMax._version import __version__


def get_output_data(filename, rxnumber, rxcomponent):
    """Gets B-scan output data from a model.

    Args:
        filename (string): Filename (including path) of output file.
        rxnumber (int): Receiver output number.
        rxcomponent (str): Receiver output field/current component.

    Returns:
        outputdata (array): Array of A-scans, i.e. B-scan data.
        dt (float): Temporal resolution of the model.
    """

    # Open output file and read some attributes
    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']
    dt = f.attrs['dt']

    # Check there are any receivers
    if nrx == 0:
        raise CmdInputError(f'No receivers found in {filename}')

    path = '/rxs/rx' + str(rxnumber) + '/'
    availableoutputs = list(f[path].keys())

    # Check if requested output is in file
    if rxcomponent not in availableoutputs:
        raise CmdInputError(f"{rxcomponent} output requested to plot, but the available output for receiver 1 is {', '.join(availableoutputs)}")

    outputdata = f[path + '/' + rxcomponent]
    outputdata = np.array(outputdata)
    f.close()

    return outputdata, dt


def merge_files(basefilename, removefiles=False):
    """Merges traces (A-scans) from multiple output files into one new file,
        then optionally removes the series of output files.

    Args:
        basefilename (string): Base name of output file series including path.
        outputs (boolean): Flag to remove individual output files after merge.
    """

    outputfile = basefilename + '_merged.h5'
    files = glob.glob(basefilename + '*.h5')
    outputfiles = [filename for filename in files if '_merged' not in filename]
    modelruns = len(outputfiles)

    # Combined output file
    fout = h5py.File(outputfile, 'w')

    for model in range(modelruns):
        fin = h5py.File(basefilename + str(model + 1) + '.h5', 'r')
        nrx = fin.attrs['nrx']

        # Write properties for merged file on first iteration
        if model == 0:
            fout.attrs['gprMax'] = __version__
            fout.attrs['Iterations'] = fin.attrs['Iterations']
            fout.attrs['nx_ny_nz'] = fin.attrs['nx_ny_nz']
            fout.attrs['dx_dy_dz'] = fin.attrs['dx_dy_dz']
            fout.attrs['dt'] = fin.attrs['dt']
            fout.attrs['nsrc'] = fin.attrs['nsrc']
            fout.attrs['nrx'] = fin.attrs['nrx']
            fout.attrs['srcsteps'] = fin.attrs['srcsteps']
            fout.attrs['rxsteps'] = fin.attrs['rxsteps']

            for rx in range(1, nrx + 1):
                path = '/rxs/rx' + str(rx)
                grp = fout.create_group(path)
                availableoutputs = list(fin[path].keys())
                for output in availableoutputs:
                    grp.create_dataset(output, (fout.attrs['Iterations'], modelruns), dtype=fin[path + '/' + output].dtype)

        # For all receivers
        for rx in range(1, nrx + 1):
            path = '/rxs/rx' + str(rx) + '/'
            availableoutputs = list(fin[path].keys())
            # For all receiver outputs
            for output in availableoutputs:
                fout[path + '/' + output][:, model] = fin[path + '/' + output][:]

        fin.close()

    fout.close()

    if removefiles:
        for model in range(modelruns):
            file = basefilename + str(model + 1) + '.h5'
            os.remove(file)

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Merges traces (A-scans) from multiple output files into one new file, then optionally removes the series of output files.', usage='cd gprMax; python -m tools.outputfiles_merge basefilename')
    parser.add_argument('basefilename', help='base name of output file series including path')
    parser.add_argument('--remove-files', action='store_true', default=False, help='flag to remove individual output files after merge')
    args = parser.parse_args()

    merge_files(args.basefilename, removefiles=args.remove_files)
