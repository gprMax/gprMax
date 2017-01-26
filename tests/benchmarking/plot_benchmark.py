import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gprMax._version import __version__
from gprMax.utilities import get_host_info

"""Plots execution times and speedup factors from benchmarking models run with different numbers of threads. Results are read from a NumPy archive."""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plots execution times and speedup factors from benchmarking models run with different numbers of threads. Results are read from a NumPy archive.', usage='cd gprMax; python -m tests.benchmarking.plot_benchmark numpyfile')
parser.add_argument('baseresult', help='name of NumPy archive file including path')
parser.add_argument('--otherresults', default=None, help='list of NumPy archives file including path', nargs='+')
args = parser.parse_args()

# Load base result
baseresult = np.load(args.baseresult)

# Get machine/CPU/OS details
try:
    machineIDlong = str(baseresult['machineID'])
    # machineIDlong = 'Dell PowerEdge R630; Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz; Linux (3.10.0-327.18.2.el7.x86_64)' # Use to manually describe machine
    machineID = machineIDlong.split(';')[0]
except KeyError:
    hostinfo = get_host_info()
    machineIDlong = '; '.join([hostinfo['machineID'], hostinfo['cpuID'], hostinfo['osversion']])
print('MachineID: {}'.format(machineIDlong))

# Base result info
print('Model: {}'.format(args.baseresult))
for thread in range(len(baseresult['threads'])):
    print('{} thread(s): {:g} s'.format(baseresult['threads'][thread], baseresult['benchtimes'][thread]))
baseplotlabel = os.path.splitext(os.path.split(args.baseresult)[1])[0] + '.in'

# Load other results and info
otherresults = []
otherplotlabels = []
if args.otherresults is not None:
    for i, result in enumerate(args.otherresults):
        otherresults.append(np.load(result))
        print('Model: {}'.format(result))
        for thread in range(len(otherresults[i]['threads'])):
            print('{} thread(s): {:g} s'.format(otherresults[i]['threads'][thread], otherresults[i]['benchtimes'][thread]))
        otherplotlabels.append(os.path.splitext(os.path.split(result)[1])[0] + '.in')

# Get gprMax version
try:
    version = str(baseresult['version'])
except KeyError:
    version = __version__

# Plot colours from http://tools.medialab.sciences-po.fr/iwanthue/index.php
colors = ['#5CB7C6', '#E60D30', '#A21797', '#A3B347']
lines = ['--', ':', '-.']

fig, ax = plt.subplots(num=machineIDlong, figsize=(20, 10), facecolor='w', edgecolor='w')
fig.suptitle(machineIDlong)
gs = gridspec.GridSpec(1, 2, hspace=0.5)
ax = plt.subplot(gs[0, 0])
ax.plot(baseresult['threads'], baseresult['benchtimes'], color=colors[1], marker='.', ms=10, lw=2, label=baseplotlabel + ' (v' + version + ')')

if args.otherresults is not None:
    for i, result in enumerate(otherresults):
        ax.plot(result['threads'], result['benchtimes'], color=colors[1], marker='.', ms=10, lw=2, ls=lines[i], label=otherplotlabels[i] + ' (v' + version + ')')

#ax.plot(results['threads'], results['bench1'], color=colors[1], marker='.', ms=10, lw=2, label='bench_100x100x100.in (v3.0.0b21)')
#ax.plot(results['threads'], results['bench1c'], color=colors[0], marker='.', ms=10, lw=2, label='bench_100x100x100.in (v2)')
#ax.plot(results['threads'], results['bench2'], color=colors[1], marker='.', ms=10, lw=2, ls='--', label='bench_150x150x150.in (v3.0.0b21)')
#ax.plot(results['threads'], results['bench2c'], color=colors[0], marker='.', ms=10, lw=2, ls='--', label='bench_150x150x150.in (v2)')

ax.set_xlabel('Number of threads')
ax.set_ylabel('Time [s]')
ax.grid()

legend = ax.legend(loc=1)
frame = legend.get_frame()
frame.set_edgecolor('white')

ax.set_xlim([0, baseresult['threads'][0] * 1.1])
ax.set_xticks(np.append(baseresult['threads'], 0))
ax.set_ylim(0, top=ax.get_ylim()[1] * 1.1)

ax = plt.subplot(gs[0, 1])
ax.plot(baseresult['threads'], baseresult['benchtimes'][-1] / baseresult['benchtimes'], color=colors[1], marker='.', ms=10, lw=2, label=baseplotlabel + ' (v' + version + ')')

if args.otherresults is not None:
    for i, result in enumerate(otherresults):
        ax.plot(result['threads'], result['benchtimes'][-1] / result['benchtimes'], color=colors[1], marker='.', ms=10, lw=2, ls=lines[i], label=otherplotlabels[i] + ' (v' + version + ')')

#ax.plot(results['threads'], results['bench1'][0] / results['bench1'], color=colors[1], marker='.', ms=10, lw=2, label='bench_100x100x100.in (v3.0.0b21)')
#ax.plot(results['threads'], results['bench1c'][1] / results['bench1c'], color=colors[0], marker='.', ms=10, lw=2, label='bench_100x100x100.in (v2)')
#ax.plot(results['threads'], results['bench2'][0] / results['bench2'], color=colors[1], marker='.', ms=10, lw=2, ls='--', label='bench_150x150x150.in (v3.0.0b21)')
#ax.plot(results['threads'], results['bench2c'][1] / results['bench2c'], color=colors[0], marker='.', ms=10, lw=2, ls='--', label='bench_150x150x150.in (v2)')

ax.set_xlabel('Number of threads')
ax.set_ylabel('Speed-up factor')
ax.grid()

legend = ax.legend(loc=1)
frame = legend.get_frame()
frame.set_edgecolor('white')

ax.set_xlim([0, baseresult['threads'][0] * 1.1])
ax.set_xticks(np.append(baseresult['threads'], 0))
ax.set_ylim(bottom=1, top=ax.get_ylim()[1] * 1.1)

# Save a pdf of the plot
fig.savefig(os.path.join(os.path.dirname(args.baseresult), machineID.replace(' ', '_') + '.png'), dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

plt.show()
