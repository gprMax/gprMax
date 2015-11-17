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
import numpy as np
import matplotlib.pyplot as plt

from gprMax.waveforms import Waveform


"""Plot waveforms that can be used for sources."""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot waveforms that can be used for sources.', usage='cd gprMax; python -m tools.plot_waveform type amp freq timewindow dt')
parser.add_argument('type', help='type of waveform, e.g. gaussian, ricker etc...')
parser.add_argument('amp', type=float, help='amplitude of waveform')
parser.add_argument('freq', type=float, help='centre frequency of waveform')
parser.add_argument('timewindow', type=float, help='time window to view waveform')
parser.add_argument('dt', type=float, help='time step to view waveform')
args = parser.parse_args()

w = Waveform()
w.type = args.type
w.amp = args.amp
w.freq = args.freq
timewindow = args.timewindow
dt = args.dt

time = np.arange(0, timewindow, dt)
waveform = np.zeros(len(time))
timeiter = np.nditer(time, flags=['c_index'])

while not timeiter.finished:
    waveform[timeiter.index] = w.calculate_value(timeiter[0], dt)
    timeiter.iternext()

# Calculate frequency spectra of waveform
power = 20 * np.log10(np.abs(np.fft.fft(waveform))**2)
f = np.fft.fftfreq(power.size, d=dt)

# Shift powers so any spectra with negative DC component will start at zero
power -= np.amax(power)

# Set plotting range to 4 * centre frequency
pltrange = np.where(f > (4 * w.freq))[0][0]

# Plot waveform
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, num=w.type, figsize=(20, 10), facecolor='w', edgecolor='w')
ax1.plot(time, waveform, 'r', lw=2)
ax1.set_xlabel('Time [ns]')
ax1.set_ylabel('Amplitude')
[label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65)) for label in ax1.get_xticklabels() + ax1.get_yticklabels()]

# Plot frequency spectra
ax2.stem(f[0:pltrange]/1e9, power[0:pltrange],'b', lw=2)
ax2.set_xlabel('Frequency [GHz]')
ax2.set_ylabel('Power [dB]')
[ax.grid() for ax in fig.axes] # Turn on grid
plt.show()

# Save a PDF of the figure
#fig.savefig(os.path.dirname(os.path.abspath(__file__)) + os.sep + w.type + '.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)