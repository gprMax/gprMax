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

from gprMax.exceptions import CmdInputError

"""Plots the s11 scattering parameter (input port voltage reflection coefficient) from an output file containing a transmission line source."""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plots the s11 scattering parameter (input port voltage reflection coefficient) from an output file containing a transmission line source.', usage='cd gprMax; python -m tools.plot_s11 outputfile')
parser.add_argument('outputfile', help='name of output file including path')
args = parser.parse_args()

# Open output file and read some attributes
file = args.outputfile
f = h5py.File(file, 'r')
dt = f.attrs['dt']
iterations = f.attrs['Iterations']
time = np.arange(0, dt * iterations, dt)
time = time / 1e-9

path = '/tls/tl1/'
Vinc = f[path + 'Vinc'][:]
Vscat = f[path + 'Vscat'][:]
Vtotal = f[path +'Vtotal'][:]

# Calculate magnitude of frequency spectra
Vincp = np.abs(np.fft.fft(Vinc))**2
freqs = np.fft.fftfreq(Vincp.size, d=dt)
Vscatp = np.abs(np.fft.fft(Vscat))**2
s11 = np.abs(Vscatp / Vincp)

# Convert to decibels
Vincp = 10 * np.log10(Vincp)
Vscatp = 10 * np.log10(Vscatp)
s11 = 10 * np.log10(s11)

# Set plotting range to a frequency
pltrange = np.where(freqs > 2e9)[0][0]
pltrange = np.s_[1:pltrange]

# Plot incident voltage
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, num='Incident and scattered voltages', figsize=(20, 10), facecolor='w', edgecolor='w')
ax1.plot(time, Vinc, 'r', lw=2, label='Vinc')
ax1.set_xlabel('Time [ns]')
ax1.set_ylabel('Incident (field) voltage [V]')
ax1.set_xlim([0, np.amax(time)])
ax1.grid()

# Plot frequency spectra of incident voltage
markerline, stemlines, baseline = ax2.stem(freqs[pltrange]/1e9, Vincp[pltrange], '-.')
plt.setp(stemlines, 'color', 'r')
plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
ax2.set_xlabel('Frequency [GHz]')
ax2.set_ylabel('Power [dB]')
ax2.grid()

# Plot scattered voltage
ax3.plot(time, Vscat, 'r', lw=2, label='Vscat')
ax3.set_xlabel('Time [ns]')
ax3.set_ylabel('Scattered (field) voltage [V]')
ax3.set_xlim([0, np.amax(time)])
ax3.grid()

# Plot frequency spectra of scattered voltage
markerline, stemlines, baseline = ax4.stem(freqs[pltrange]/1e9, Vscatp[pltrange], '-.')
plt.setp(stemlines, 'color', 'r')
plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
ax4.set_xlabel('Frequency [GHz]')
ax4.set_ylabel('Power [dB]')
ax4.grid()

# Plot frequency spectra of s11
markerline, stemlines, baseline = ax6.stem(freqs[pltrange]/1e9, s11[pltrange], '-.')
plt.setp(stemlines, 'color', 'r')
plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
ax6.set_xlabel('Frequency [GHz]')
ax6.set_ylabel('Power [dB]')
ax6.grid()

plt.show()
f.close()