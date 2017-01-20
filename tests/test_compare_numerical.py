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

import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt

"""Compare field outputs
    
Usage:
    cd gprMax
    python -m tests.test_compare_numerical path_to_file1 path_to_file2
    
"""

filename1 = sys.argv[1]
filename2 = sys.argv[2]
path = '/rxs/rx1/'
# Key refers to subplot location
fields = {0: 'Ex', 2: 'Ey', 4: 'Ez', 1: 'Hx', 3: 'Hy', 5: 'Hz'}
plotorder = list(fields.keys())

# File 1 results
f = h5py.File(filename1, 'r')
floattype = f[path + 'Ex'].dtype
data1 = np.zeros((f.attrs['Iterations'], 6), dtype=floattype)
time1 = np.zeros((f.attrs['Iterations']), dtype=floattype)
time1 = np.arange(0, f.attrs['dt'] * f.attrs['Iterations'], f.attrs['dt']) / 1e-9
for ID, name in fields.items():
    data1[:,ID] = f[path + str(name)][:]
f.close()

# File 2 results
f = h5py.File(filename2, 'r')
data2 = np.zeros((f.attrs['Iterations'], 6), dtype=floattype)
time2 = np.zeros((f.attrs['Iterations']), dtype=floattype)
time2 = np.arange(0, f.attrs['dt'] * f.attrs['Iterations'], f.attrs['dt']) / 1e-9
for ID, name in fields.items():
    data2[:,ID] = f[path + str(name)][:]
f.close()

# Differences
# In case there is any difference in the number of iterations, take the smaller
timesmallest = np.amin((time2.shape, time1.shape))
fieldssmallest = np.amin((data2.shape[0], data1.shape[0]))

threshold = 1e-4 # Threshold, below which ignore differences
diffs = np.zeros((fieldssmallest, 6), dtype=floattype)
for ID, name in fields.items():
    max = np.amax(np.abs(data1[:fieldssmallest,ID]))
    if max < threshold:
        diffs[:,ID] = 0
        diffsum = 0
        print('Detected differences of less than {} when comparing {} field component, therefore set as zero.'.format(threshold, fields[ID]))
    else:
        diffs[:,ID] = (np.abs(data1[:fieldssmallest,ID] - data2[:fieldssmallest,ID]) / max) * 100
        diffsum = (np.sum(np.abs(data1[:fieldssmallest,ID] - data2[:fieldssmallest,ID])) / np.sum(np.abs(data1[:fieldssmallest,ID]))) * 100
    print('Total differences in field component {}: {:.1f}%'.format(name, diffsum))

# Plot data1
fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, sharex=False, sharey='col', subplot_kw=dict(xlabel='Time [ns]'), num=filename1 + ' versus ' + filename2, figsize=(20, 10), facecolor='w', edgecolor='w')
ax1.plot(time1, data1[:,0],'r', lw=2, label='Ex')
ax3.plot(time1, data1[:,2],'r', lw=2, label='Ey')
ax5.plot(time1, data1[:,4],'r', lw=2, label='Ez')
ax2.plot(time1, data1[:,1],'b', lw=2, label='Hx')
ax4.plot(time1, data1[:,3],'b', lw=2, label='Hy')
ax6.plot(time1, data1[:,5],'b', lw=2, label='Hz')
    
# Set ylabels
ylabels = ['$E_x$, field strength [V/m]', '$H_x$, field strength [A/m]', '$E_y$, field strength [V/m]', '$H_y$, field strength [A/m]', '$E_z$, field strength [V/m]', '$H_z$, field strength [A/m]']
[ax.set_ylabel(ylabels[index]) for index, ax in enumerate(fig1.axes)]

# Turn on grid
[ax.grid() for ax in fig1.axes]

# Add data2 and set legend
for index, ax in enumerate(fig1.axes):
    if plotorder[index] in [0, 2, 4]:
        ax.plot(time2, data2[:,plotorder[index]], 'r', label='data2', lw=2, ls='--')
    else:
        ax.plot(time2, data2[:,plotorder[index]], label='data2', lw=2, ls='--')
    ax.set_xlim(0, time2[-1])
    handles, existlabels = ax.get_legend_handles_labels()
    ax.legend(handles, [os.path.split(filename1)[1], os.path.split(filename2)[1]])

# Plots of differences
fig2, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, sharex=False, sharey='col', subplot_kw=dict(xlabel='Time [ns]'), num='Deltas: ' + filename1 + ' versus ' + filename2, figsize=(20, 10), facecolor='w', edgecolor='w')
ax1.plot(time1[:timesmallest], diffs[:,0],'r', lw=2, label='Ex')
ax3.plot(time1[:timesmallest], diffs[:,2],'r', lw=2, label='Ey')
ax5.plot(time1[:timesmallest], diffs[:,4],'r', lw=2, label='Ez')
ax2.plot(time1[:timesmallest], diffs[:,1],'b', lw=2, label='Hx')
ax4.plot(time1[:timesmallest], diffs[:,3],'b', lw=2, label='Hy')
ax6.plot(time1[:timesmallest], diffs[:,5],'b', lw=2, label='Hz')

# Set ylabels
ylabels = ['$E_x$', '$H_x$', '$E_y$', '$H_y$', '$E_z$', '$H_z$']
ylabels = [ylabel + ', percentage difference [%]' for ylabel in ylabels]
[ax.set_ylabel(ylabels[index]) for index, ax in enumerate(fig2.axes)]

# Set axes limits and turn on grid
[ax.grid() for ax in fig2.axes]
[ax.set_xlim(0, time1[timesmallest - 1]) for ax in fig2.axes]
[ax.set_ylim(0, np.ceil(np.amax(np.abs(diffs)))) for ax in fig2.axes]

# Show/print plots
savename = os.path.abspath(os.path.dirname(filename1)) + os.sep + os.path.splitext(os.path.split(filename1)[1])[0] + '_vs_' + os.path.splitext(os.path.split(filename2)[1])[0]
#fig1.savefig(savename + '.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
#fig2.savefig(savename + '_diffs.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()

