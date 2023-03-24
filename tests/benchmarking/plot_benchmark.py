# Copyright (C) 2015-2023: The University of Edinburgh
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
import itertools
import os
import sys

import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from gprMax._version import __version__
from gprMax.utilities import get_host_info
from gprMax.utilities import human_size


"""Plots execution times and speedup factors from benchmarking models run with different numbers of CPU (OpenMP) threads. Can also benchmark GPU(s) if required. Results are read from a NumPy archive."""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plots execution times and speedup factors from benchmarking models run with different numbers of CPU (OpenMP) threads. Can also benchmark GPU(s) if required. Results are read from a NumPy archive.', usage='cd gprMax; python -m tests.benchmarking.plot_benchmark numpyfile')
parser.add_argument('baseresult', help='name of NumPy archive file including path')
parser.add_argument('--otherresults', default=None, help='list of NumPy archives file including path', nargs='+')
args = parser.parse_args()

# Load base result
baseresult = dict(np.load(args.baseresult))

# Get machine/CPU/OS details
hostinfo = get_host_info()
try:
    machineIDlong = str(baseresult['machineID'])
    # machineIDlong = 'Dell PowerEdge R630; Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz; Linux (3.10.0-327.18.2.el7.x86_64)' # Use to manually describe machine
    machineID = machineIDlong.split(';')[0]
    cpuID = machineIDlong.split(';')[1]
    cpuID = cpuID.split('GHz')[0].split('x')[1][1::] + 'GHz'
except KeyError:
    hyperthreading = ', {} cores with Hyper-Threading'.format(hostinfo['logicalcores']) if hostinfo['hyperthreading'] else ''
    machineIDlong = '{}; {} x {} ({} cores{}); {} RAM; {}'.format(hostinfo['machineID'], hostinfo['sockets'], hostinfo['cpuID'], hostinfo['physicalcores'], hyperthreading, human_size(hostinfo['ram'], a_kilobyte_is_1024_bytes=True), hostinfo['osversion'])
print('Host: {}'.format(machineIDlong))

# Base result - general info
print('Model: {}'.format(args.baseresult))
cells = np.array([baseresult['numcells'][0]]) # Length of cubic model side for cells per second metric
baseplotlabel = os.path.splitext(os.path.split(args.baseresult)[1])[0] + '.in'

# Base result - CPU threads and times info from Numpy archive
if baseresult['cputhreads'].size != 0:
    for i in range(len(baseresult['cputhreads'])):
        print('{} CPU (OpenMP) thread(s): {:g} s'.format(baseresult['cputhreads'][i], baseresult['cputimes'][i]))
    cpucellspersec = np.array([(baseresult['numcells'][0] * baseresult['numcells'][1] * baseresult['numcells'][2] * baseresult['iterations']) / baseresult['cputimes'][0]])

# Base result - GPU time info
gpuIDs = baseresult['gpuIDs'].tolist()
if gpuIDs:
    gpucellspersec = np.zeros((len(gpuIDs), 1))
    for i in range(len(gpuIDs)):
        print('NVIDIA {}: {:g} s'.format(gpuIDs[i], baseresult['gputimes'][i]))
        gpucellspersec[i] = (baseresult['numcells'][0] * baseresult['numcells'][1] * baseresult['numcells'][2] * baseresult['iterations']) / baseresult['gputimes'][i]

# Load any other results and info
otherresults = []
otherplotlabels = []
if args.otherresults is not None:
    for i, result in enumerate(args.otherresults):
        otherresults.append(dict(np.load(result)))
        print('\nModel: {}'.format(result))
        cells = np.append(cells, otherresults[i]['numcells'][0]) # Length of cubic model side for cells per second metric
        otherplotlabels.append(os.path.splitext(os.path.split(result)[1])[0] + '.in')
        
        # CPU
        if otherresults[i]['cputhreads'].size != 0:
            for thread in range(len(otherresults[i]['cputhreads'])):
                print('{} CPU (OpenMP) thread(s): {:g} s'.format(otherresults[i]['cputhreads'][thread], otherresults[i]['cputimes'][thread]))
            cpucellspersec = np.append(cpucellspersec, (otherresults[i]['numcells'][0] * otherresults[i]['numcells'][1] * otherresults[i]['numcells'][2] * otherresults[i]['iterations']) / otherresults[i]['cputimes'][0])
        
        # GPU
        othergpuIDs = otherresults[i]['gpuIDs'].tolist()
        if othergpuIDs:
            # Array for cells per second metric
            tmp = np.zeros((len(gpuIDs), len(args.otherresults) + 1))
            tmp[:gpucellspersec.shape[0],:gpucellspersec.shape[1]] = gpucellspersec
            gpucellspersec = tmp
            for j in range(len(othergpuIDs)):
                print('NVIDIA {}: {:g} s'.format(othergpuIDs[j], otherresults[i]['gputimes'][j]))
                gpucellspersec[j,i+1] = (otherresults[i]['numcells'][0] * otherresults[i]['numcells'][1] * otherresults[i]['numcells'][2] * otherresults[i]['iterations']) / otherresults[i]['gputimes'][j]

# Get gprMax version
try:
    version = str(baseresult['version'])
except KeyError:
    version = __version__

# Create/setup plot figure
#colors = ['#E60D30', '#5CB7C6', '#A21797', '#A3B347'] # Plot colours from http://tools.medialab.sciences-po.fr/iwanthue/index.php
colorIDs = ['#015dbb', '#c23100', '#00a15a', '#c84cd0', '#ff9aa0']
colors = itertools.cycle(colorIDs)
lines = itertools.cycle(('--', ':', '-.', '-'))
markers = ['o', 'd', '^', 's', '*']
fig, ax = plt.subplots(num=machineID, figsize=(30, 10), facecolor='w', edgecolor='w')
fig.suptitle(machineIDlong + '\ngprMax v' + version)
gs = gridspec.GridSpec(1, 3, hspace=0.5)
plotcount = 0

###########################################
# Subplot of CPU (OpenMP) threads vs time #
###########################################
if baseresult['cputhreads'].size != 0:
    ax = plt.subplot(gs[0, plotcount])
    ax.plot(baseresult['cputhreads'], baseresult['cputimes'], color=next(colors), marker=markers[0], markeredgecolor='none', ms=8, lw=2, label=baseplotlabel)

    if args.otherresults is not None:
        for i, result in enumerate(otherresults):
            ax.plot(result['cputhreads'], result['cputimes'], color=next(colors), marker=markers[0], markeredgecolor='none', ms=8, lw=2, ls=next(lines), label=otherplotlabels[i])

    ax.set_xlabel('Number of CPU (OpenMP) threads')
    ax.set_ylabel('Time [s]')
    ax.grid()
    legend = ax.legend(loc=1)
    frame = legend.get_frame()
    frame.set_edgecolor('white')
    ax.set_xlim([0, baseresult['cputhreads'][0] * 1.1])
    ax.set_xticks(np.append(baseresult['cputhreads'], 0))
    ax.set_ylim(0, top=ax.get_ylim()[1] * 1.1)
    plotcount += 1

######################################################
# Subplot of CPU (OpenMP) threads vs speed-up factor #
######################################################
colors = itertools.cycle(colorIDs) # Reset color iterator
if baseresult['cputhreads'].size != 0:
    ax = plt.subplot(gs[0, plotcount])
    ax.plot(baseresult['cputhreads'], baseresult['cputimes'][-1] / baseresult['cputimes'], color=next(colors), marker=markers[0], markeredgecolor='none', ms=8, lw=2, label=baseplotlabel)

    if args.otherresults is not None:
        for i, result in enumerate(otherresults):
            ax.plot(result['cputhreads'], result['cputimes'][-1] / result['cputimes'], color=next(colors), marker=markers[0], markeredgecolor='none', ms=8, lw=2, ls=next(lines), label=otherplotlabels[i])

    ax.set_xlabel('Number of CPU (OpenMP) threads')
    ax.set_ylabel('Speed-up factor')
    ax.grid()
    legend = ax.legend(loc=2)
    frame = legend.get_frame()
    frame.set_edgecolor('white')
    ax.set_xlim([0, baseresult['cputhreads'][0] * 1.1])
    ax.set_xticks(np.append(baseresult['cputhreads'], 0))
    ax.set_ylim(bottom=1, top=ax.get_ylim()[1] * 1.1)
    plotcount += 1

###########################################
# Subplot of simulation size vs cells/sec #
###########################################

def autolabel(rects):
    """Attach a text label above each bar on a matplotlib bar chart displaying its height.
        
        Args:
        rects: Handle to bar chart
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height,
                '%d' % int(height),
                ha='center', va='bottom', fontsize=10, rotation=90)

colors = itertools.cycle(colorIDs) # Reset color iterator
ax = plt.subplot(gs[0, plotcount])
barwidth = 8 # the width of the bars

if baseresult['cputhreads'].size != 0:
    cpu = ax.bar(cells - (1/2) * barwidth, cpucellspersec / 1e6, barwidth, color=next(colors), edgecolor='none', label=cpuID)
    autolabel(cpu)

if gpuIDs:
    positions = np.arange(-gpucellspersec.shape[0] / 2, gpucellspersec.shape[0] / 2, 1)
    for i in range(gpucellspersec.shape[0]):
        gpu = ax.bar(cells + positions[i] * barwidth, gpucellspersec[i,:] / 1e6, barwidth, color=next(colors), edgecolor='none', label='NVIDIA ' + gpuIDs[i])
        autolabel(gpu)

ax.set_xlabel('Side length of cubic domain [cells]')
ax.set_ylabel('Performance [Mcells/s]')
ax.grid()
legend = ax.legend(loc=2)
frame = legend.get_frame()
frame.set_edgecolor('white')
ax.set_xticks(cells)
ax.set_xticklabels(cells)
ax.set_xlim([0, cells[-1] * 1.1])
ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1)

##########################
# Save a png of the plot #
##########################
fig.savefig(os.path.join(os.path.dirname(args.baseresult), machineID.replace(' ', '_') + '.png'), dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
#fig.savefig(os.path.join(os.path.dirname(args.baseresult), machineID.replace(' ', '_') + '.pdf'), dpi='none', format='pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()
