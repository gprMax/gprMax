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
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gprMax.constants import complextype

"""Plots antenna parameters (s11 parameter and input impedance and admittance) from an output file containing a transmission line source."""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plots antenna parameters (s11 parameter and input impedance and admittance) from an output file containing a transmission line source.', usage='cd gprMax; python -m tools.plot_antenna_params outputfile')
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

# Read/calculate voltages and currents
path = '/tls/tl' + str(args.tln) + '/'
Vinc = f[path + 'Vinc'][:]
Iinc = f[path + 'Iinc'][:]
Vtotal = f[path +'Vtotal'][:]
Itotal = f[path +'Itotal'][:]
Vref = Vtotal - Vinc
Iref = Itotal - Iinc

# Calculate magnitude of FFTs of voltages and currents
Vincp = np.abs(np.fft.fft(Vinc))
freqs = np.fft.fftfreq(Vincp.size, d=dt)
delaycorrection = np.zeros(Vincp.size, dtype=complextype)
delaycorrection = np.exp(-1j * np.pi * freqs * dt)
Iincp = np.abs(np.fft.fft(Iinc))
Vrefp = np.abs(np.fft.fft(Vref))
Irefp = np.abs(np.fft.fft(Iref))
Vtotalp = np.abs(np.fft.fft(Vtotal))
Itotalp = np.abs(np.fft.fft(Itotal))

# Calculate s11
s11 = Vrefp / Vincp

# Calculate input impedance
zin = np.zeros(iterations, dtype=complextype)
zin = (np.fft.fft(Vtotal) * delaycorrection) / np.fft.fft(Itotal)

# Calculate input admittance
yin = np.zeros(iterations, dtype=complextype)
yin = np.fft.fft(Itotal) / (np.fft.fft(Vtotal) * delaycorrection)

# Convert to decibels
Vincp = 20 * np.log10(Vincp)
Iincp = 20 * np.log10(Iincp)
Vrefp = 20 * np.log10(Vrefp)
Irefp = 20 * np.log10(Irefp)
Vtotalp = 20 * np.log10(Vtotalp)
Itotalp = 20 * np.log10(Itotalp)
s11 = 20 * np.log10(s11)

# Set plotting range
# To a certain drop from maximum power
#pltrange = np.where((np.amax(Vincp) - Vincp) > 30)[0][0] + 1
# To a maximum frequency
pltrange = np.where(freqs > 2e9)[0][0]
pltrange = np.s_[1:pltrange]

# Figure 1
# Plot incident voltage
fig1, ax = plt.subplots(num='Transmission line parameters', figsize=(20, 12), facecolor='w', edgecolor='w')
gs1 = gridspec.GridSpec(4, 2, hspace=0.7)
ax = plt.subplot(gs1[0, 0])
ax.plot(time, Vinc, 'r', lw=2, label='Vinc')
ax.set_title('Incident voltage')
ax.set_xlabel('Time [ns]')
ax.set_ylabel('Voltage [V]')
ax.set_xlim([0, np.amax(time)])
ax.grid()

# Plot frequency spectra of incident voltage
ax = plt.subplot(gs1[0, 1])
markerline, stemlines, baseline = ax.stem(freqs[pltrange]/1e9, Vincp[pltrange], '-.')
plt.setp(baseline, 'linewidth', 0)
plt.setp(stemlines, 'color', 'r')
plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
ax.plot(freqs[pltrange]/1e9, Vincp[pltrange], 'r', lw=2)
ax.set_title('Incident voltage')
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Power [dB]')
ax.grid()

# Plot incident current
ax = plt.subplot(gs1[1, 0])
ax.plot(time, Iinc, 'b', lw=2, label='Vinc')
ax.set_title('Incident current')
ax.set_xlabel('Time [ns]')
ax.set_ylabel('Current [A]')
ax.set_xlim([0, np.amax(time)])
ax.grid()

# Plot frequency spectra of incident current
ax = plt.subplot(gs1[1, 1])
markerline, stemlines, baseline = ax.stem(freqs[pltrange]/1e9, Iincp[pltrange], '-.')
plt.setp(baseline, 'linewidth', 0)
plt.setp(stemlines, 'color', 'b')
plt.setp(markerline, 'markerfacecolor', 'b', 'markeredgecolor', 'b')
ax.plot(freqs[pltrange]/1e9, Iincp[pltrange], 'b', lw=2)
ax.set_title('Incident current')
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Power [dB]')
ax.grid()

# Plot total voltage
ax = plt.subplot(gs1[2, 0])
ax.plot(time, Vtotal, 'r', lw=2, label='Vinc')
ax.set_title('Total (incident + reflected) voltage')
ax.set_xlabel('Time [ns]')
ax.set_ylabel('Voltage [V]')
ax.set_xlim([0, np.amax(time)])
ax.grid()

# Plot frequency spectra of total voltage
ax = plt.subplot(gs1[2, 1])
markerline, stemlines, baseline = ax.stem(freqs[pltrange]/1e9, Vtotalp[pltrange], '-.')
plt.setp(baseline, 'linewidth', 0)
plt.setp(stemlines, 'color', 'r')
plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
ax.plot(freqs[pltrange]/1e9, Vtotalp[pltrange], 'r', lw=2)
ax.set_title('Total (incident + reflected) voltage')
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Power [dB]')
ax.grid()

# Plot total current
ax = plt.subplot(gs1[3, 0])
ax.plot(time, Itotal, 'b', lw=2, label='Vinc')
ax.set_title('Total (incident + reflected) current')
ax.set_xlabel('Time [ns]')
ax.set_ylabel('Current [A]')
ax.set_xlim([0, np.amax(time)])
ax.grid()

# Plot frequency spectra of reflected current
ax = plt.subplot(gs1[3, 1])
markerline, stemlines, baseline = ax.stem(freqs[pltrange]/1e9, Itotalp[pltrange], '-.')
plt.setp(baseline, 'linewidth', 0)
plt.setp(stemlines, 'color', 'b')
plt.setp(markerline, 'markerfacecolor', 'b', 'markeredgecolor', 'b')
ax.plot(freqs[pltrange]/1e9, Itotalp[pltrange], 'b', lw=2)
ax.set_title('Total (incident + reflected) current')
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Power [dB]')
ax.grid()

## Plot reflected (reflected) voltage
#ax = plt.subplot(gs1[4, 0])
#ax.plot(time, Vref, 'r', lw=2, label='Vref')
#ax.set_title('Reflected voltage')
#ax.set_xlabel('Time [ns]')
#ax.set_ylabel('Voltage [V]')
#ax.set_xlim([0, np.amax(time)])
#ax.grid()
#
## Plot frequency spectra of reflected voltage
#ax = plt.subplot(gs1[4, 1])
#markerline, stemlines, baseline = ax.stem(freqs[pltrange]/1e9, Vrefp[pltrange], '-.')
#plt.setp(baseline, 'linewidth', 0)
#plt.setp(stemlines, 'color', 'r')
#plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
#ax.plot(freqs[pltrange]/1e9, Vrefp[pltrange], 'r', lw=2)
#ax.set_title('Reflected voltage')
#ax.set_xlabel('Frequency [GHz]')
#ax.set_ylabel('Power [dB]')
#ax.grid()
#
## Plot reflected (reflected) current
#ax = plt.subplot(gs1[5, 0])
#ax.plot(time, Iref, 'b', lw=2, label='Iref')
#ax.set_title('Reflected current')
#ax.set_xlabel('Time [ns]')
#ax.set_ylabel('Current [A]')
#ax.set_xlim([0, np.amax(time)])
#ax.grid()
#
## Plot frequency spectra of reflected current
#ax = plt.subplot(gs1[5, 1])
#markerline, stemlines, baseline = ax.stem(freqs[pltrange]/1e9, Irefp[pltrange], '-.')
#plt.setp(baseline, 'linewidth', 0)
#plt.setp(stemlines, 'color', 'b')
#plt.setp(markerline, 'markerfacecolor', 'b', 'markeredgecolor', 'b')
#ax.plot(freqs[pltrange]/1e9, Irefp[pltrange], 'b', lw=2)
#ax.set_title('Reflected current')
#ax.set_xlabel('Frequency [GHz]')
#ax.set_ylabel('Power [dB]')
#ax.grid()

# Figure 2
# Plot frequency spectra of s11
fig2, ax = plt.subplots(num='Antenna parameters', figsize=(20, 12), facecolor='w', edgecolor='w')
gs2 = gridspec.GridSpec(3, 2, hspace=0.5)
ax = plt.subplot(gs2[0, 0])
markerline, stemlines, baseline = ax.stem(freqs[pltrange]/1e9, s11[pltrange], '-.')
plt.setp(baseline, 'linewidth', 0)
plt.setp(stemlines, 'color', 'g')
plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
ax.plot(freqs[pltrange]/1e9, s11[pltrange], 'g', lw=2)
ax.set_title('s11')
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Power [dB]')
ax.grid()

# Plot input resistance (real part of impedance)
ax = plt.subplot(gs2[1, 0])
markerline, stemlines, baseline = ax.stem(freqs[pltrange]/1e9, np.abs(zin[pltrange]), '-.')
plt.setp(baseline, 'linewidth', 0)
plt.setp(stemlines, 'color', 'g')
plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
ax.plot(freqs[pltrange]/1e9, np.abs(zin[pltrange]), 'g', lw=2)
ax.set_title('Input impedance (resistive)')
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Resistance [Ohms]')
ax.set_ylim([0, 2000])
ax.grid()

# Plot input reactance (imaginery part of impedance)
ax = plt.subplot(gs2[1, 1])
markerline, stemlines, baseline = ax.stem(freqs[pltrange]/1e9, zin[pltrange].imag, '-.')
plt.setp(baseline, 'linewidth', 0)
plt.setp(stemlines, 'color', 'g')
plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
ax.plot(freqs[pltrange]/1e9, zin[pltrange].imag, 'g', lw=2)
ax.set_title('Input impedance (reactive)')
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Reactance [Ohms]')
ax.set_ylim(-2000, 2000)
ax.grid()

# Plot input admittance (magnitude)
ax = plt.subplot(gs2[2, 0])
markerline, stemlines, baseline = ax.stem(freqs[pltrange]/1e9, np.abs(yin[pltrange]), '-.')
plt.setp(baseline, 'linewidth', 0)
plt.setp(stemlines, 'color', 'g')
plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
ax.plot(freqs[pltrange]/1e9, np.abs(yin[pltrange]), 'g', lw=2)
ax.set_title('Input admittance (magnitude)')
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Admittance [Siemens]')
ax.grid()

# Plot input admittance (phase)
ax = plt.subplot(gs2[2, 1])
markerline, stemlines, baseline = ax.stem(freqs[pltrange]/1e9, np.angle(yin[pltrange], deg=True), '-.')
plt.setp(baseline, 'linewidth', 0)
plt.setp(stemlines, 'color', 'g')
plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
ax.plot(freqs[pltrange]/1e9, np.angle(yin[pltrange], deg=True), 'g', lw=2)
ax.set_title('Input admittance (phase)')
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Phase [degrees]')
ax.grid()

plt.show()
#fig1.savefig(os.path.splitext(os.path.abspath(file))[0] + '_tl_params.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
#fig2.savefig(os.path.splitext(os.path.abspath(file))[0] + '_ant_params.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
#fig1.savefig(os.path.splitext(os.path.abspath(file))[0] + '_tl_params.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
#fig2.savefig(os.path.splitext(os.path.abspath(file))[0] + '_ant_params.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
f.close()