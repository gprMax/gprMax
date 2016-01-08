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

import argparse
import h5py
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gprMax.exceptions import CmdInputError

"""Plots the s11 scattering parameter (input port voltage reflection coefficient) from an output file containing a transmission line source."""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plots the s11 scattering parameter (input port voltage reflection coefficient) from an output file containing a transmission line source.', usage='cd gprMax; python -m tools.plot_s11 outputfile')
parser.add_argument('outputfile', help='name of output file including path')
parser.add_argument('-tln', default=1, type=int, help='transmission line number')
args = parser.parse_args()

# Open output file and read some attributes
file = args.outputfile
f = h5py.File(file, 'r')
dt = f.attrs['dt']
iterations = f.attrs['Iterations']
time = np.arange(0, dt * iterations, dt)
time = time / 1e-9

path = '/tls/tl' + str(args.tln) + '/'
Vinc = f[path + 'Vinc'][:]
Vscat = f[path + 'Vscat'][:]
Vtotal = f[path +'Vtotal'][:]

# Calculate magnitude of frequency spectra
Vincp = np.abs(np.fft.fft(Vinc))**2
freqs = np.fft.fftfreq(Vincp.size, d=dt)
Vscatp = np.abs(np.fft.fft(Vscat))**2
s11 = Vscatp / Vincp

# Convert to decibels
Vincp = 10 * np.log10(Vincp)
Vscatp = 10 * np.log10(Vscatp)
s11 = 10 * np.log10(s11)

# Set plotting range to -60dB from maximum power
pltrange = np.where((np.amax(Vincp) - Vincp) > 60)[0][0] + 1
pltrange = np.s_[0:pltrange]

# Plot incident voltage
plt.subplots(num='Transmission line voltages & s11 parameter', figsize=(20, 10), facecolor='w', edgecolor='w')
gs = gridspec.GridSpec(3, 2)
ax1 = plt.subplot(gs[0, 0])
ax1.plot(time, Vinc, 'r', lw=2, label='Vinc')
ax1.set_xlabel('Time [ns]')
ax1.set_ylabel('Incident voltage [V]')
ax1.set_xlim([0, np.amax(time)])
ax1.grid()

# Plot frequency spectra of incident voltage
ax2 = plt.subplot(gs[0, 1])
markerline, stemlines, baseline = ax2.stem(freqs[pltrange]/1e9, Vincp[pltrange], '-.')
plt.setp(baseline, 'linewidth', 0)
plt.setp(stemlines, 'color', 'r')
plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
ax2.plot(freqs[pltrange]/1e9, Vincp[pltrange], 'r', lw=2)
ax2.set_xlabel('Frequency [GHz]')
ax2.set_ylabel('Incident voltage spectra [dB]')
ax2.grid()

# Plot scattered (field) voltage
ax3 = plt.subplot(gs[1, 0])
ax3.plot(time, Vscat, 'r', lw=2, label='Vscat')
ax3.set_xlabel('Time [ns]')
ax3.set_ylabel('Scattered (field) voltage [V]')
ax3.set_xlim([0, np.amax(time)])
ax3.grid()

# Plot frequency spectra of scattered voltage
ax4 = plt.subplot(gs[1, 1])
markerline, stemlines, baseline = ax4.stem(freqs[pltrange]/1e9, Vscatp[pltrange], '-.')
plt.setp(baseline, 'linewidth', 0)
plt.setp(stemlines, 'color', 'r')
plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
ax4.plot(freqs[pltrange]/1e9, Vscatp[pltrange], 'r', lw=2)
ax4.set_xlabel('Frequency [GHz]')
ax4.set_ylabel('Scattered (field) voltage spectra [dB]')
ax4.grid()

# Plot frequency spectra of s11
ax5 = plt.subplot(gs[2, 1])
markerline, stemlines, baseline = ax5.stem(freqs[pltrange]/1e9, s11[pltrange], '-.')
plt.setp(baseline, 'linewidth', 0)
plt.setp(stemlines, 'color', 'r')
plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
ax5.plot(freqs[pltrange]/1e9, s11[pltrange], 'r', lw=2)
ax5.set_xlabel('Frequency [GHz]')
ax5.set_ylabel('s11 [dB]')
ax5.grid()

plt.show()
f.close()