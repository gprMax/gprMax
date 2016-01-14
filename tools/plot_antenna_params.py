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

import os, argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

"""Plots antenna parameters (s11 parameter and input impedance and admittance) from an output file containing a transmission line source."""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plots antenna parameters (s11 parameter and input impedance and admittance) from an output file containing a transmission line source.', usage='cd gprMax; python -m tools.plot_antenna_params outputfile')
parser.add_argument('outputfile', help='name of output file including path')
parser.add_argument('-tln', default=1, type=int, help='transmission line number')
args = parser.parse_args()

print("Antenna parameter analysis from file '{}'...".format(args.outputfile))

# Open output file and read some attributes
file = args.outputfile
f = h5py.File(file, 'r')
dt = f.attrs['dt']
iterations = f.attrs['Iterations']

# Choose a specific frequency bin spacing
#df = 1.5e6
#iterations = int((1 / df) / dt)

# Calculate time array and frequency bin spacing
time = np.arange(0, dt * iterations, dt)
time = time[0:iterations]
df = 1 / np.amax(time)
time = time / 1e-9

print('Time window: {:g} secs ({} iterations)'.format(np.amax(time) * 1e-9, iterations))
print('Time step: {:g} secs'.format(dt))
print('Frequency bin spacing: {:g} MHz'.format(df / 1e6))

# Read/calculate voltages and currents
path = '/tls/tl' + str(args.tln) + '/'
Vinc = f[path + 'Vinc'][0:iterations]
Iinc = f[path + 'Iinc'][0:iterations]
Vtotal = f[path +'Vtotal'][0:iterations]
Itotal = f[path +'Itotal'][0:iterations]
f.close()
Vref = Vtotal - Vinc
Iref = Itotal - Iinc

# Frequency bins
freqs = np.fft.fftfreq(Vinc.size, d=dt)

# Delay correction to ensure voltage and current are at same time step
delaycorrection = np.exp(-1j * 2 * np.pi * freqs * (dt /2 ))

# Calculate s11
s11 = np.abs(np.fft.fft(Vref) * delaycorrection) / np.abs(np.fft.fft(Vinc) * delaycorrection)

# Calculate input impedance
zin = (np.fft.fft(Vtotal) * delaycorrection) / np.fft.fft(Itotal)

# Calculate input admittance
yin = np.fft.fft(Itotal) / (np.fft.fft(Vtotal) * delaycorrection)

# Convert to decibels
Vincp = 20 * np.log10(np.abs((np.fft.fft(Vinc) * delaycorrection)))
Iincp = 20 * np.log10(np.abs(np.fft.fft(Iinc)))
Vrefp = 20 * np.log10(np.abs((np.fft.fft(Vref) * delaycorrection)))
Irefp = 20 * np.log10(np.abs(np.fft.fft(Iref)))
Vtotalp = 20 * np.log10(np.abs((np.fft.fft(Vtotal) * delaycorrection)))
Itotalp = 20 * np.log10(np.abs(np.fft.fft(Itotal)))
s11 = 20 * np.log10(s11)

# Set plotting range
pltrangemin = 1
# To a certain drop from maximum power
pltrangemax = np.where((np.amax(Vincp[1::]) - Vincp[1::]) > 60)[0][0] + 1
# To a maximum frequency
pltrangemax = np.where(freqs > 2e9)[0][0]
pltrange = np.s_[pltrangemin:pltrangemax]

# Print some useful values from s11, input impedance and admittance
s11minfreq = np.where(s11[pltrange] == np.amin(s11[pltrange]))[0][0]
print('s11 minimum: {:g} dB at {:g} MHz'.format(np.amin(s11[pltrange]), freqs[s11minfreq + pltrangemin] / 1e6))
print('At {:g} MHz...'.format(freqs[s11minfreq + pltrangemin] / 1e6))
print('Input impedance: {:.1f}{:+.1f}j Ohms'.format(np.abs(zin[s11minfreq + pltrangemin]), zin[s11minfreq + pltrangemin].imag))
print('Input admittance (mag): {:g} S'.format(np.abs(yin[s11minfreq + pltrangemin])))
print('Input admittance (phase): {:.1f} deg'.format(np.angle(yin[s11minfreq + pltrangemin], deg=True)))

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
#ax.set_xlim([0.88, 1.02])
#ax.set_ylim([-50, -8])
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
#ax.set_xlim([0.88, 1.02])
#ax.set_ylim([0, 1000])
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
#ax.set_xlim([0.88, 1.02])
#ax.set_ylim([-1000, 1000])
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
#ax.set_xlim([0.88, 1.02])
#ax.set_ylim([0.009, 0.015])
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
#ax.set_xlim([0.88, 1.02])
#ax.set_ylim([-45, 45])
ax.grid()

plt.show()

# Save a PDF/PNG of the figure
#fig1.savefig(os.path.splitext(os.path.abspath(file))[0] + '_tl_params.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
#fig2.savefig(os.path.splitext(os.path.abspath(file))[0] + '_ant_params.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
#fig1.savefig(os.path.splitext(os.path.abspath(file))[0] + '_tl_params.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
#fig2.savefig(os.path.splitext(os.path.abspath(file))[0] + '_ant_params.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
