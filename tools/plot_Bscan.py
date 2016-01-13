# Copyright (C) 2015-2016: The University of Edinburgh
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

import os, argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

from gprMax.exceptions import CmdInputError

"""Plots a B-scan image."""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plots a B-scan image.', usage='cd gprMax; python -m tools.plot_Bscan outputfile --field fieldcomponent')
parser.add_argument('outputfile', help='name of output file including path')
parser.add_argument('--output', help='name of output to be plotted, i.e. Ex Ey Ez')
args = parser.parse_args()

# Open output file and read some attributes
f = h5py.File(args.outputfile, 'r')
path = '/rxs/rx1'
availablecomponents = list(f[path].keys())
    
# Check if requested output is in file
if args.output[0] not in availablecomponents:
    raise CmdInputError('{} output requested to plot, but the available output for receiver 1 is {}'.format(args.outputs[0], ', '.join(availablecomponents)))

outputdata = f[path + '/' + args.output]
f.close()

# Check that there is more than one A-scan present
if outputdata.shape[1] == 1:
    raise CmdInputError('{} contains only a single A-scan.'.format(args.outputfile))

# Plot B-scan image
fig = plt.figure(num=args.outputfile, figsize=(20, 10), facecolor='w', edgecolor='w')
plt.imshow(outputdata, extent=[0, outputdata.shape[1], outputdata.shape[0]*f.attrs['dt']*1e9, 0], interpolation='nearest', aspect='auto', cmap='seismic', vmin=-np.amax(np.abs(outputdata)), vmax=np.amax(np.abs(outputdata)))
plt.xlabel('Trace number')
plt.ylabel('Time [ns]')
plt.grid()
cb = plt.colorbar()
if 'E' in args.output:
    cb.set_label('Field strength [V/m]')
elif 'H' in args.output:
    cb.set_label('Field strength [A/m]')
elif 'I' in args.output:
    cb.set_label('Current [A]')

plt.show()

# Save a PDF/PNG of the figure
#fig.savefig(os.path.splitext(os.path.abspath(file))[0] + '.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
#fig.savefig(os.path.splitext(os.path.abspath(file))[0] + '.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

