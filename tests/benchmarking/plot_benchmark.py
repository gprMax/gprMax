import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gprMax._version import __version__
from gprMax.utilities import get_machine_cpu_os

"""Plots execution times and speedup factors from benchmarking models run with different numbers of threads. Results are read from a NumPy archive."""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plots execution times and speedup factors from benchmarking models run with different numbers of threads. Results are read from a NumPy archive.', usage='cd gprMax; python -m tests.benchmarking.plot_benchmark numpyfile')
parser.add_argument('numpyfile1', help='name of NumPy archive file including path')
parser.add_argument('--numpyfile2', default=None, help='name of NumPy archive file including path')
args = parser.parse_args()

# Load results1
results1 = np.load(args.numpyfile1)

# Get machine/CPU/OS details
try:
    machineIDlong = str(results1['machineID'])
    machineID = machineIDlong.split(';')[0]
except KeyError:
    machineID, cpuID, osversion = get_machine_cpu_os()
    machineIDlong = machineID + '; ' + cpuID + '; ' + osversion
print('MachineID: {}'.format(machineIDlong))

# Results1 info
print('Model: {}'.format(args.numpyfile1))
for thread in range(len(results1['threads'])):
    print('{} thread(s): {:g} s'.format(results1['threads'][thread], results1['benchtimes'][thread]))
plotlabel1 = os.path.splitext(os.path.split(args.numpyfile1)[1])[0] + '.in'

# Load results2 and info
if args.numpyfile2 is not None:
    results2 = np.load(args.numpyfile2)
    print('Model: {}'.format(args.numpyfile2))
    for thread in range(len(results2['threads'])):
        print('{} thread(s): {:g} s'.format(results2['threads'][thread], results2['benchtimes'][thread]))
    plotlabel2 = os.path.splitext(os.path.split(args.numpyfile2)[1])[0] + '.in'

# Get gprMax version
try:
    version = str(results1['version'])
except KeyError:
    version = __version__

# Plot colours from http://tools.medialab.sciences-po.fr/iwanthue/index.php
colors = ['#5CB7C6', '#E60D30', '#A21797', '#A3B347']

fig, ax = plt.subplots(num=machineIDlong, figsize=(20, 10), facecolor='w', edgecolor='w')
fig.suptitle(machineIDlong)
gs = gridspec.GridSpec(1, 2, hspace=0.5)
ax = plt.subplot(gs[0, 0])
ax.plot(results1['threads'], results1['benchtimes'], color=colors[1], marker='.', ms=10, lw=2, label=plotlabel1 + ' (v' + version + ')')

if args.numpyfile2 is not None:
    ax.plot(results2['threads'], results2['benchtimes'], color=colors[1], marker='.', ms=10, lw=2, ls='--', label=plotlabel2 + ' (v' + version + ')')

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

ax.set_xlim([0, results1['threads'][0] * 1.1])
ax.set_xticks(np.append(results1['threads'], 0))
ax.set_ylim(0, top=ax.get_ylim()[1] * 1.1)

ax = plt.subplot(gs[0, 1])
ax.plot(results1['threads'], results1['benchtimes'][-1] / results1['benchtimes'], color=colors[1], marker='.', ms=10, lw=2, label=plotlabel1 + ' (v' + version + ')')

if args.numpyfile2 is not None:
    ax.plot(results2['threads'], results2['benchtimes'][-1] / results2['benchtimes'], color=colors[1], marker='.', ms=10, lw=2, ls='--', label=plotlabel2 + ' (v' + version + ')')

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

ax.set_xlim([0, results1['threads'][0] * 1.1])
ax.set_xticks(np.append(results1['threads'], 0))
ax.set_ylim(bottom=1, top=ax.get_ylim()[1] * 1.1)

# Save a pdf of the plot
fig.savefig(os.path.join(os.path.dirname(args.numpyfile1), machineID.replace(' ', '_') + '.png'), dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

plt.show()


