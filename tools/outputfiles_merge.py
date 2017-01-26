# Copyright (C) 2015-2017: The University of Edinburgh
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

import h5py
import numpy as np

"""Merges traces (A-scans) from multiple output files into one new file, then removes the series of output files."""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Merges traces (A-scans) from multiple output files into one new file, then removes the series of output files.', usage='cd gprMax; python -m tools.outputfiles_merge basefilename')
parser.add_argument('basefilename', help='base name of output file series including path')
args = parser.parse_args()

basefilename = args.basefilename
outputfile = basefilename + '_merged.out'
files = glob.glob(basefilename + '*.out')
outputfiles = [filename for filename in files if '_merged' not in filename]
modelruns = len(outputfiles)
print('Found {} files to merge'.format(modelruns))

# Combined output file
fout = h5py.File(outputfile, 'w')

# Add positional data for rxs
for model in range(modelruns):
    fin = h5py.File(basefilename + str(model + 1) + '.out', 'r')
    nrx = fin.attrs['nrx']

    # Write properties for merged file on first iteration
    if model == 0:
        fout.attrs['Iterations'] = fin.attrs['Iterations']
        fout.attrs['dt'] = fin.attrs['dt']
        fout.attrs['nrx'] = fin.attrs['nrx']
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

check = input('Do you want to remove the multiple individual output files? [y] or n:')
if not check or check == 'y':
    for model in range(modelruns):
        file = basefilename + str(model + 1) + '.out'
        os.remove(file)
