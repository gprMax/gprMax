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

import argparse, os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gprMax.exceptions import CmdInputError


def plot_antenna_params(filename, tlnumber=1, rxnumber=None, rxcomponent=None):
    """Calculates and plots antenna parameters - s11, (s21) and input impedance.
            
    Args:
        filename (string): Filename (including path) of output file.
        tlnumber (int): Transmitting antenna - transmission line number
        rxnumber (int): Receiver antenna - output number
        rxcomponent (str): Receiver antenna - output electric field component
    """

    # Open output file and read some attributes
    f = h5py.File(filename, 'r')
    dxdydz = f.attrs['dx, dy, dz']
    dt = f.attrs['dt']
    iterations = f.attrs['Iterations']

    # Calculate time array and frequency bin spacing
    time = np.linspace(0, 1, iterations)
    time *= (iterations * dt)
    df = 1 / np.amax(time)

    print('Time window: {:g} s ({} iterations)'.format(np.amax(time), iterations))
    print('Time step: {:g} s'.format(dt))
    print('Frequency bin spacing: {:g} Hz'.format(df))

    # Read/calculate voltages and currents
    tlpath = '/tls/tl' + str(tlnumber) + '/'

    # Incident voltages/currents
    Vinc = f[tlpath + 'Vinc'][:]
    Iinc = f[tlpath + 'Iinc'][:]

    # Total (incident + reflected) voltages/currents
    Vtotal = f[tlpath +'Vtotal'][:]
    Itotal = f[tlpath +'Itotal'][:]

    # Reflected voltages/currents
    Vref = Vtotal - Vinc
    Iref = Itotal - Iinc

    # If a receiver number for a receiever antenna is given can get received voltage for s21
    if rxnumber:       
        rxpath = '/rxs/rx' + str(rxnumber) + '/'
        availableoutputs = list(f[rxpath].keys())
        
        if rxcomponent not in availableoutputs:
            raise CmdInputError('{} output requested, but the available output for receiver {} is {}'.format(rxcomponent, rxnumber, ', '.join(availableoutputs)))
        
        rxpath += rxcomponent
        
        # Received voltage
        if rxcomponent == 'Ex':
            Vrec = f[rxpath][:] * -1 * dxdydz[0]
        elif rxcomponent == 'Ey':
            Vrec = f[rxpath][:] * -1 * dxdydz[1]
        elif rxcomponent == 'Ez':
            Vrec = f[rxpath][:] * -1 * dxdydz[2]
    f.close()

    # Frequency bins
    freqs = np.fft.fftfreq(Vinc.size, d=dt)

    # Delay correction - current lags voltage, so delay voltage to match current timestep
    delaycorrection = np.exp(-1j * 2 * np.pi * freqs * (dt / 2))

    # Calculate s11
    s11 = np.abs(np.fft.fft(Vref) * delaycorrection) / np.abs(np.fft.fft(Vinc) * delaycorrection)
    if rxnumber:
        s21 = np.abs(np.fft.fft(Vrec)) / np.abs(np.fft.fft(Vinc) * delaycorrection)

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
    if rxnumber:
        s21 = 20 * np.log10(s21)

    # Set plotting range
    pltrangemin = 1
    # To a certain drop from maximum power
    pltrangemax = np.where((np.amax(Vincp[1::]) - Vincp[1::]) > 60)[0][0] + 1
    # To a maximum frequency
    #pltrangemax = np.where(freqs > 6e9)[0][0]
    pltrange = np.s_[pltrangemin:pltrangemax]

    # Print some useful values from s11, and input impedance
    s11minfreq = np.where(s11[pltrange] == np.amin(s11[pltrange]))[0][0]
    print('s11 minimum: {:g} dB at {:g} Hz'.format(np.amin(s11[pltrange]), freqs[s11minfreq + pltrangemin]))
    print('At {:g} Hz...'.format(freqs[s11minfreq + pltrangemin]))
    print('Input impedance: {:.1f}{:+.1f}j Ohms'.format(np.abs(zin[s11minfreq + pltrangemin]), zin[s11minfreq + pltrangemin].imag))
    #print('Input admittance (mag): {:g} S'.format(np.abs(yin[s11minfreq + pltrangemin])))
    #print('Input admittance (phase): {:.1f} deg'.format(np.angle(yin[s11minfreq + pltrangemin], deg=True)))

    # Figure 1
    # Plot incident voltage
    fig1, ax = plt.subplots(num='Transmission line parameters', figsize=(20, 12), facecolor='w', edgecolor='w')
    gs1 = gridspec.GridSpec(4, 2, hspace=0.7)
    ax = plt.subplot(gs1[0, 0])
    ax.plot(time, Vinc, 'r', lw=2, label='Vinc')
    ax.set_title('Incident voltage')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Voltage [V]')
    ax.set_xlim([0, np.amax(time)])
    ax.grid()

    # Plot frequency spectra of incident voltage
    ax = plt.subplot(gs1[0, 1])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], Vincp[pltrange], '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'r')
    plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
    ax.plot(freqs[pltrange], Vincp[pltrange], 'r', lw=2)
    ax.set_title('Incident voltage')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    ax.grid()

    # Plot incident current
    ax = plt.subplot(gs1[1, 0])
    ax.plot(time, Iinc, 'b', lw=2, label='Vinc')
    ax.set_title('Incident current')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Current [A]')
    ax.set_xlim([0, np.amax(time)])
    ax.grid()

    # Plot frequency spectra of incident current
    ax = plt.subplot(gs1[1, 1])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], Iincp[pltrange], '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'b')
    plt.setp(markerline, 'markerfacecolor', 'b', 'markeredgecolor', 'b')
    ax.plot(freqs[pltrange], Iincp[pltrange], 'b', lw=2)
    ax.set_title('Incident current')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    ax.grid()

    # Plot total voltage
    ax = plt.subplot(gs1[2, 0])
    ax.plot(time, Vtotal, 'r', lw=2, label='Vinc')
    ax.set_title('Total (incident + reflected) voltage')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Voltage [V]')
    ax.set_xlim([0, np.amax(time)])
    ax.grid()

    # Plot frequency spectra of total voltage
    ax = plt.subplot(gs1[2, 1])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], Vtotalp[pltrange], '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'r')
    plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
    ax.plot(freqs[pltrange], Vtotalp[pltrange], 'r', lw=2)
    ax.set_title('Total (incident + reflected) voltage')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    ax.grid()

    # Plot total current
    ax = plt.subplot(gs1[3, 0])
    ax.plot(time, Itotal, 'b', lw=2, label='Vinc')
    ax.set_title('Total (incident + reflected) current')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Current [A]')
    ax.set_xlim([0, np.amax(time)])
    ax.grid()

    # Plot frequency spectra of reflected current
    ax = plt.subplot(gs1[3, 1])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], Itotalp[pltrange], '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'b')
    plt.setp(markerline, 'markerfacecolor', 'b', 'markeredgecolor', 'b')
    ax.plot(freqs[pltrange], Itotalp[pltrange], 'b', lw=2)
    ax.set_title('Total (incident + reflected) current')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    ax.grid()

    ## Plot reflected (reflected) voltage
    #ax = plt.subplot(gs1[4, 0])
    #ax.plot(time, Vref, 'r', lw=2, label='Vref')
    #ax.set_title('Reflected voltage')
    #ax.set_xlabel('Time [s]')
    #ax.set_ylabel('Voltage [V]')
    #ax.set_xlim([0, np.amax(time)])
    #ax.grid()
    #
    ## Plot frequency spectra of reflected voltage
    #ax = plt.subplot(gs1[4, 1])
    #markerline, stemlines, baseline = ax.stem(freqs[pltrange], Vrefp[pltrange], '-.')
    #plt.setp(baseline, 'linewidth', 0)
    #plt.setp(stemlines, 'color', 'r')
    #plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
    #ax.plot(freqs[pltrange], Vrefp[pltrange], 'r', lw=2)
    #ax.set_title('Reflected voltage')
    #ax.set_xlabel('Frequency [Hz]')
    #ax.set_ylabel('Power [dB]')
    #ax.grid()
    #
    ## Plot reflected (reflected) current
    #ax = plt.subplot(gs1[5, 0])
    #ax.plot(time, Iref, 'b', lw=2, label='Iref')
    #ax.set_title('Reflected current')
    #ax.set_xlabel('Time [s]')
    #ax.set_ylabel('Current [A]')
    #ax.set_xlim([0, np.amax(time)])
    #ax.grid()
    #
    ## Plot frequency spectra of reflected current
    #ax = plt.subplot(gs1[5, 1])
    #markerline, stemlines, baseline = ax.stem(freqs[pltrange], Irefp[pltrange], '-.')
    #plt.setp(baseline, 'linewidth', 0)
    #plt.setp(stemlines, 'color', 'b')
    #plt.setp(markerline, 'markerfacecolor', 'b', 'markeredgecolor', 'b')
    #ax.plot(freqs[pltrange], Irefp[pltrange], 'b', lw=2)
    #ax.set_title('Reflected current')
    #ax.set_xlabel('Frequency [Hz]')
    #ax.set_ylabel('Power [dB]')
    #ax.grid()

    # Figure 2
    # Plot frequency spectra of s11
    fig2, ax = plt.subplots(num='Antenna parameters', figsize=(20, 12), facecolor='w', edgecolor='w')
    gs2 = gridspec.GridSpec(2, 2, hspace=0.5)
    ax = plt.subplot(gs2[0, 0])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], s11[pltrange], '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'g')
    plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
    ax.plot(freqs[pltrange], s11[pltrange], 'g', lw=2)
    ax.set_title('s11')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    #ax.set_xlim([0.88e9, 1.02e9])
    #ax.set_ylim([-20, 0])
    ax.grid()

    # Plot frequency spectra of s21
    if rxnumber:
        ax = plt.subplot(gs2[0, 1])
        markerline, stemlines, baseline = ax.stem(freqs[pltrange], s21[pltrange], '-.')
        plt.setp(baseline, 'linewidth', 0)
        plt.setp(stemlines, 'color', 'g')
        plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
        ax.plot(freqs[pltrange], s21[pltrange], 'g', lw=2)
        ax.set_title('s21')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Power [dB]')
        #ax.set_xlim([0.88e9, 1.02e9])
        #ax.set_ylim([-25, 50])
        ax.grid()

    # Plot input resistance (real part of impedance)
    ax = plt.subplot(gs2[1, 0])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], zin[pltrange].real, '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'g')
    plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
    ax.plot(freqs[pltrange], zin[pltrange].real, 'g', lw=2)
    ax.set_title('Input impedance (resistive)')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Resistance [Ohms]')
    #ax.set_xlim([0.88e9, 1.02e9])
    ax.set_ylim(bottom=0)
    #ax.set_ylim([0, 300])
    ax.grid()

    # Plot input reactance (imaginery part of impedance)
    ax = plt.subplot(gs2[1, 1])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], zin[pltrange].imag, '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'g')
    plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
    ax.plot(freqs[pltrange], zin[pltrange].imag, 'g', lw=2)
    ax.set_title('Input impedance (reactive)')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Reactance [Ohms]')
    #ax.set_xlim([0.88e9, 1.02e9])
    #ax.set_ylim([-200, 100])
    ax.grid()

    ## Plot input admittance (magnitude)
    #ax = plt.subplot(gs2[2, 0])
    #markerline, stemlines, baseline = ax.stem(freqs[pltrange], np.abs(yin[pltrange]), '-.')
    #plt.setp(baseline, 'linewidth', 0)
    #plt.setp(stemlines, 'color', 'g')
    #plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
    #ax.plot(freqs[pltrange], np.abs(yin[pltrange]), 'g', lw=2)
    #ax.set_title('Input admittance (magnitude)')
    #ax.set_xlabel('Frequency [Hz]')
    #ax.set_ylabel('Admittance [Siemens]')
    ##ax.set_xlim([0.88e9, 1.02e9])
    ##ax.set_ylim([0, 0.035])
    #ax.grid()
    #
    ## Plot input admittance (phase)
    #ax = plt.subplot(gs2[2, 1])
    #markerline, stemlines, baseline = ax.stem(freqs[pltrange], np.angle(yin[pltrange], deg=True), '-.')
    #plt.setp(baseline, 'linewidth', 0)
    #plt.setp(stemlines, 'color', 'g')
    #plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
    #ax.plot(freqs[pltrange], np.angle(yin[pltrange], deg=True), 'g', lw=2)
    #ax.set_title('Input admittance (phase)')
    #ax.set_xlabel('Frequency [Hz]')
    #ax.set_ylabel('Phase [degrees]')
    ##ax.set_xlim([0.88e9, 1.02e9])
    ##ax.set_ylim([-40, 100])
    #ax.grid()

    # Save a PDF/PNG of the figure
    #fig1.savefig(os.path.splitext(os.path.abspath(filename))[0] + '_tl_params.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    #fig2.savefig(os.path.splitext(os.path.abspath(filename))[0] + '_ant_params.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    #fig1.savefig(os.path.splitext(os.path.abspath(filename))[0] + '_tl_params.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
    #fig2.savefig(os.path.splitext(os.path.abspath(filename))[0] + '_ant_params.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)

    plt.show()


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plots antenna parameters (s11, s21 parameters and input impedance) from an output file containing a transmission line source.', usage='cd gprMax; python -m tools.plot_antenna_params outputfile')
    parser.add_argument('outputfile', help='name of output file including path')
    parser.add_argument('--tl-num', default=1, type=int, help='transmitting antenna - transmission line number')
    parser.add_argument('--rx-num', type=int, help='receiver antenna - output number')
    parser.add_argument('--rx-component', type=str, help='receiver antenna - output electric field component', choices=['Ex', 'Ey', 'Ez'])
    args = parser.parse_args()

    plot_antenna_params(args.outputfile, args.tl_num, args.rx_num, args.rx_component)

