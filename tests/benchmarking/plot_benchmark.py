import argparse, os, platform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gprMax._version import __version__

"""Plots execution times and speedup factors from benchmarking models run with different numbers of threads. Results are read from a NumPy archive."""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plots execution times and speedup factors from benchmarking models run with different numbers of threads. Results are read from a NumPy archive.', usage='cd gprMax; python -m tests.benchmarking.plot_benchmark numpyfile')
parser.add_argument('numpyfile', help='name of NumPy archive file including path')
args = parser.parse_args()

# Machine identifier
platformID = platform.platform()
machineID = input('Enter manufacturer and short machine description, e.g. Apple_MacPro1,1 or Dell_Z420: ')
cpuspeed = input ('Enter CPU number, type and speed, e.g. 2 x 2.66GHz Quad-Core Intel Xeon or 1 x 4GHz Quad-Core Intel Core i7: ')
machineIDextra = input('Enter any additional machine description, e.g. Retina 5K, 27-inch, Late 2014 or leave empty: ')
if machineIDextra:
    machineIDextra = '(' + machineIDextra + ')'
osversion = input('Enter operating system version, e.g. Windows 7 64-bit: ')
machineIDlong = machineID + ' ' + machineIDextra + '; ' + cpuspeed + '; ' + osversion

# Load results
results = np.load(args.numpyfile)
plotlabel = os.path.splitext(os.path.split(args.numpyfile)[1])[0] + '.in'

# Plot colours from http://tools.medialab.sciences-po.fr/iwanthue/index.php
colors = ['#5CB7C6', '#E60D30', '#A21797', '#A3B347']

fig, ax = plt.subplots(num=machineIDlong, figsize=(20, 10), facecolor='w', edgecolor='w')
fig.suptitle(machineIDlong)
gs = gridspec.GridSpec(1, 2, hspace=0.5)
ax = plt.subplot(gs[0, 0])
ax.plot(results['threads'], results['benchtimes'], color=colors[1], marker='.', ms=10, lw=2, label=plotlabel + ' (v' + __version__ + ')')
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

ax.set_xlim([0, results['threads'][0] * 1.1])
ax.set_xticks(np.append(results['threads'], 0))
ax.set_ylim(0, top=ax.get_ylim()[1] * 1.1)

ax = plt.subplot(gs[0, 1])
ax.plot(results['threads'], results['benchtimes'][-1] / results['benchtimes'], color=colors[1], marker='.', ms=10, lw=2, label=plotlabel + ' (v' + __version__ + ')')
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

ax.set_xlim([0, results['threads'][0] * 1.1])
ax.set_xticks(np.append(results['threads'], 0))
ax.set_ylim(bottom=1, top=ax.get_ylim()[1] * 1.1)

# Save a pdf of the plot
fig.savefig(os.path.splitext(args.numpyfile)[0] + '.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

plt.show()


