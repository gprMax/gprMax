# Copyright (C) 2015: The University of Edinburgh
#            Authors: Craig Warren and Antonis Giannopoulos
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

import os, argparse
import h5py
import numpy as np

"""Merges traces (A-scans) from multiple output files into one new file, then removes the series of output files."""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Merges traces (A-scans) from multiple output files into one new file, then removes the series of output files.', usage='cd gprMax; python -m tools.outputfiles_mergehdf5 basefilename modelruns')
parser.add_argument('basefilename', help='base name of output file series including path')
parser.add_argument('modelruns', type=int, help='number of model runs, i.e. number of output files to merge')
args = parser.parse_args()

basefilename = args.basefilename
modelruns = args.modelruns
outputfile = basefilename + '_all.out'
path = '/rxs/rx1'

# Combined output file
fout = h5py.File(outputfile, 'w')

# Add positional data for rxs
for rx in range(modelruns):
    fin = h5py.File(basefilename + str(rx + 1) + '.out', 'r')
    if rx == 0:
        fout.attrs['Iterations'] = fin.attrs['Iterations']
        fout.attrs['dt'] = fin.attrs['dt']
        fields = fout.create_group(path)
        fields['Ex'] = np.zeros((fout.attrs['Iterations'], modelruns), dtype=fin[path + '/Ex'].dtype)
        fields['Ey'] = np.zeros((fout.attrs['Iterations'], modelruns), dtype=fin[path + '/Ex'].dtype)
        fields['Ez'] = np.zeros((fout.attrs['Iterations'], modelruns), dtype=fin[path + '/Ex'].dtype)
        fields['Hx'] = np.zeros((fout.attrs['Iterations'], modelruns), dtype=fin[path + '/Ex'].dtype)
        fields['Hy'] = np.zeros((fout.attrs['Iterations'], modelruns), dtype=fin[path + '/Ex'].dtype)
        fields['Hz'] = np.zeros((fout.attrs['Iterations'], modelruns), dtype=fin[path + '/Ex'].dtype)
    
    fields[path + '/Ex'][:,rx] = fin[path + '/Ex'][:]
    fields[path + '/Ey'][:,rx] = fin[path + '/Ey'][:]
    fields[path + '/Ez'][:,rx] = fin[path + '/Ez'][:]
    fields[path + '/Hx'][:,rx] = fin[path + '/Hx'][:]
    fields[path + '/Hy'][:,rx] = fin[path + '/Hy'][:]
    fields[path + '/Hz'][:,rx] = fin[path + '/Hz'][:]
    fin.close()

fout.close()

check = input('Do you want to remove the multiple individual output files? [y] or n:')
if not check or check == 'y':
    for rx in range(modelruns):
        file = basefilename + str(rx + 1) + '.out'
        os.remove(file)




