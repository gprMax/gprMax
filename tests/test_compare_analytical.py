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

import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tests.analytical_solutions import hertzian_dipole_fs

"""Plots a comparison of analytical solutions and given simulated output."""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plots a comparison of analytical solutions and given simulated output.', usage='cd gprMax; python -m tests.test_compare_analytical modelfile')
parser.add_argument('modelfile', help='name of model output file including path')
args = parser.parse_args()

# Model results
f = h5py.File(args.modelfile, 'r')
path = '/rxs/rx1/'
availablecomponents = list(f[path].keys())

floattype = f[path + availablecomponents[0]].dtype
iterations = f.attrs['Iterations']
dt = f.attrs['dt']
dxdydz = f.attrs['dx, dy, dz']
time = np.linspace(0, 1, iterations)
time *= (iterations * dt)
rxpos = f[path].attrs['Position']
txpos = f['/srcs/src1/'].attrs['Position']
rxposrelative = ((rxpos[0] - txpos[0]), (rxpos[1] - txpos[1]), (rxpos[2] - txpos[2]))
model = np.zeros((iterations, len(availablecomponents)), dtype=floattype)

# Analytical solution of a dipole in free space
analytical = hertzian_dipole_fs(iterations, dt, dxdydz, rxposrelative)

# Read modelled fields and calculate differences
threshold = 1e-4 # Threshold, below which ignore differences
diffs = np.zeros((iterations, len(availablecomponents)), dtype=floattype)
for index in range(len(availablecomponents)):
    model[:,index] = f[path + availablecomponents[index]][:]
    max = np.amax(np.abs(analytical[:,index]))
    if max < threshold:
        diffs[:,index] = 0
        diffsum = 0
        print('Detected differences of less than threshold {}, when comparing {} field component, therefore set as zero.'.format(threshold, availablecomponents[index]))
    else:
        diffs[:,index] = (np.abs(analytical[:,index] - model[:,index]) / max) * 100
        diffsum = (np.sum(np.abs(analytical[:,index] - model[:,index])) / np.sum(np.abs(analytical[:,index]))) * 100
    print('Total differences in field component {}: {:.1f}%'.format(availablecomponents[index], diffsum))

f.close()

# Plot modelled and analytical solutions
fig1, ax = plt.subplots(subplot_kw=dict(xlabel='Time [s]'), num=args.modelfile + ' versus analytical solution', figsize=(20, 10), facecolor='w', edgecolor='w')
gs1 = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.3)

for index in range(len(availablecomponents)):
    i = int(index % 3)
    j = int((index - i) / 3 % 2)
    ax = plt.subplot(gs1[i, j])
    line1, = ax.plot(time, model[:,index],'r', lw=2, label='Model')
    line2, = ax.plot(time, analytical[:,index],'r', lw=2, ls='--', label='Analytical')
    ax.set_ylim(1.1 * np.amin(np.amin(model[:, 0:3], axis=1)), 1.1 * np.amax(np.amax(model[:, 0:3], axis=1)))
    
    if index > 2:
        plt.setp(line1, color='g')
        plt.setp(line2, color='g')
        ax.set_ylim(1.1 * np.amin(np.amin(model[:, 3:6], axis=1)), 1.1 * np.amax(np.amax(model[:, 3:6], axis=1)))

    ax.set_xlim(0, time[-1])
    ax.grid()
    ax.legend()

# Set axes labels, limits and turn on grid
ylabels = ['Ex, field strength [V/m]', 'Ey, field strength [V/m]', 'Ez, field strength [V/m]', 'Hx, field strength [A/m]', 'Hy, field strength [A/m]', 'Hz, field strength [A/m]']
[ax.set_ylabel(ylabels[index]) for index, ax in enumerate(fig1.axes)]

# Plot differences of modelled and analytical solutions
fig2, ax = plt.subplots(subplot_kw=dict(xlabel='Time [s]'), num=args.modelfile + ' versus analytical solution differences', figsize=(20, 10), facecolor='w', edgecolor='w')
gs2 = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.3)

for index in range(len(availablecomponents)):
    i = int(index % 3)
    j = int((index - i) / 3 % 2)
    ax = plt.subplot(gs2[i, j])
    line1, = ax.plot(time, diffs[:, index],'r', lw=2)
    ax.set_ylim(0, 1.1 * np.amax(np.amax(diffs[:, 0:3], axis=1)))
    
    if index > 2:
        plt.setp(line1, color='g')
        ax.set_ylim(0, 1.1 * np.amax(np.amax(diffs[:, 3:6], axis=1)))

    ax.set_ylim(0, 2)
    ax.set_xlim(0, time[-1])
    ax.grid()

# Set axes labels, limits and turn on grid
ylabels = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']
ylabels = [ylabel + ', percentage difference [%]' for ylabel in ylabels]
[ax.set_ylabel(ylabels[index]) for index, ax in enumerate(fig2.axes)]

# Save a PDF/PNG of the figure
savename = os.path.abspath(os.path.dirname(args.modelfile)) + os.sep + os.path.splitext(os.path.split(args.modelfile)[1])[0] + '_vs_analytical'
#fig1.savefig(savename + '.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
#fig2.savefig(savename + '_diffs.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
fig1.savefig(savename + '.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
fig2.savefig(savename + '_diffs.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

plt.show()
