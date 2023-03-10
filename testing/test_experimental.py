# Copyright (C) 2015-2023: The University of Edinburgh, United Kingdom
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

logger = logging.getLogger(__name__)

"""Plots a comparison of fields between given simulation output and experimental 
    data files.
"""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plots a comparison of fields between ' +
                                    'given simulation output and experimental data files.', 
                                 usage='cd gprMax; python -m testing.test_experimental modelfile realfile output')
parser.add_argument('modelfile', help='name of model output file including path')
parser.add_argument('realfile', help='name of file containing experimental data including path')
parser.add_argument('output', help='output to be plotted, i.e. Ex Ey Ez', nargs='+')
args = parser.parse_args()

modelfile = Path(args.modelfile)
realfile = Path(args.realfile)

# Model results
f = h5py.File(Path(modelfile), 'r')
path = '/rxs/rx1/'
availablecomponents = list(f[path].keys())

# Check for polarity of output and if requested output is in file
if args.output[0][0] == 'm':
    polarity = -1
    args.outputs[0] = args.output[0][1:]
else:
    polarity = 1

if args.output[0] not in availablecomponents:
    logger.exception(f"{args.output[0]} output requested to plot, but the " +
                     f"available output for receiver 1 is {', '.join(availablecomponents)}")
    raise ValueError

floattype = f[path + args.output[0]].dtype
iterations = f.attrs['Iterations']
dt = f.attrs['dt']
model = np.zeros(iterations, dtype=floattype)
model = f[path + args.output[0]][:] * polarity
model /= np.amax(np.abs(model))
timemodel = np.linspace(0, 1, iterations)
timemodel *= (iterations * dt)
f.close()

# Find location of maximum value from model
modelmax = np.where(np.abs(model) == 1)[0][0]

# Real results
with open(realfile, 'r') as f:
    real = np.loadtxt(f)
real[:, 1] = real[:, 1] / np.amax(np.abs(real[:, 1]))
realmax = np.where(np.abs(real[:, 1]) == 1)[0][0]

difftime = - (timemodel[modelmax] - real[realmax, 0])

# Plot modelled and real data
fig, ax = plt.subplots(num=modelfile.stem + '_vs_' + realfile.stem, 
                       figsize=(20, 10), facecolor='w', edgecolor='w')
ax.plot(timemodel + difftime, model, 'r', lw=2, label='Model')
ax.plot(real[:, 0], real[:, 1], 'r', ls='--', lw=2, label='Experiment')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude')
ax.set_xlim([0, timemodel[-1]])
# ax.set_ylim([-1, 1])
ax.legend()
ax.grid()

# Save a PDF/PNG of the figure
savename =  modelfile.stem + '_vs_' + realfile.stem
savename = modelfile.parent / savename
# fig.savefig(savename.with_suffix('.pdf'), dpi=None, format='pdf', 
#             bbox_inches='tight', pad_inches=0.1)
# fig.savefig(savename.with_suffix('.png'), dpi=150, format='png', 
#             bbox_inches='tight', pad_inches=0.1)

plt.show()
