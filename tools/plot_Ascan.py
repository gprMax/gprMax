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

from gprMax.exceptions import CmdInputError

"""Plots electric and magnetic fields and currents from all receiver points in the given output file. Each receiver point is plotted in a new figure window."""

# Outputs that can be plotted
outputslist = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 'Ix', 'Iy', 'Iz']

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plots electric and magnetic fields and currents from all receiver points in the given output file. Each receiver point is plotted in a new figure window.', usage='cd gprMax; python -m tools.plot_Ascan outputfile')
parser.add_argument('outputfile', help='name of output file including path')
parser.add_argument('--outputs', help='list of outputs to be plotted, i.e. Ex Ey Ez', default=outputslist, nargs='+')
parser.add_argument('-fft', action='store_true', default=False, help='plot FFT (single output must be specified)')
args = parser.parse_args()

# Open output file and read some attributes
file = args.outputfile
f = h5py.File(file, 'r')
nrx = f.attrs['nrx']
dt = f.attrs['dt']
iterations = f.attrs['Iterations']
time = np.arange(0, dt * iterations, dt)
time = time / 1e-9

# Check for valid output names
for output in args.outputs:
    if output not in outputslist:
        raise CmdInputError('{} not allowed. Options are: Ex Ey Ez Hx Hy Hz'.format(output))

# Check for single output component when doing a FFT
if args.fft:
    if not len(args.outputs) == 1:
        raise CmdInputError('A single output must be specified when using the -fft option')

# New plot for each receiver
for rx in range(1, nrx + 1):
    path = '/rxs/rx' + str(rx) + '/'
    
    # If only a single output is required, create one subplot
    if len(args.outputs) == 1:
        outputdata = f[path + args.outputs[0]][:]
        
        # Plotting if FFT required
        if args.fft:
            # Calculate magnitude of frequency spectra of waveform
            power = 10 * np.log10(np.abs(np.fft.fft(outputdata))**2)
            freqs = np.fft.fftfreq(power.size, d=dt)

            # Shift powers so that frequency with maximum power is at zero decibels
            power -= np.amax(power)

            # Set plotting range to -60dB from maximum power
            pltrange = np.where((np.amax(power[1::]) - power[1::]) > 60)[0][0] + 1
            pltrange = np.s_[0:pltrange]

            # Plot time history of output component
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, num='rx' + str(rx), figsize=(20, 10), facecolor='w', edgecolor='w')
            line1 = ax1.plot(time, outputdata, 'r', lw=2, label=args.outputs[0])
            ax1.set_xlabel('Time [ns]')
            ax1.set_ylabel(args.outputs[0] + ' field strength [V/m]')
            ax1.set_xlim([0, np.amax(time)])
            ax1.grid()

            # Plot frequency spectra
            markerline, stemlines, baseline = ax2.stem(freqs[pltrange]/1e9, power[pltrange], '-.')
            plt.setp(baseline, 'linewidth', 0)
            plt.setp(stemlines, 'color', 'r')
            plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
            line2 = ax2.plot(freqs[pltrange]/1e9, power[pltrange], 'r', lw=2)
            ax2.set_xlabel('Frequency [GHz]')
            ax2.set_ylabel('Power [dB]')
            ax2.grid()
            
            # Change colours and labels for magnetic field components
            if 'H' in args.outputs[0]:
                plt.setp(line1, color='b')
                plt.setp(line2, color='b')
                plt.setp(ax1, ylabel=args.outputs[0] + ' field strength [A/m]')
                plt.setp(stemlines, 'color', 'b')
                plt.setp(markerline, 'markerfacecolor', 'b', 'markeredgecolor', 'b')
            
            plt.show()
    
        # Plotting if no FFT required
        else:
            fig, ax = plt.subplots(subplot_kw=dict(xlabel='Time [ns]', ylabel=args.outputs[0] + ' field strength [V/m]'), num='rx' + str(rx), figsize=(20, 10), facecolor='w', edgecolor='w')
            line = ax.plot(time, outputdata,'r', lw=2, label=args.outputs[0])
            ax.set_xlim([0, np.amax(time)])
            ax.grid()
            
            if 'H' in args.outputs[0]:
                plt.setp(line, color='b')
                plt.setp(ax, ylabel=args.outputs[0] + ', field strength [A/m]')
            elif 'I' in args.outputs[0]:
                plt.setp(line, color='b')
                plt.setp(ax, ylabel=args.outputs[0] + ', current [A]')

    # If multiple fields required, creat all nine subplots and populate only the specified ones
    else:
        fig, ax = plt.subplots(subplot_kw=dict(xlabel='Time [ns]'), num='rx' + str(rx), figsize=(20, 10), facecolor='w', edgecolor='w')
        gs = gridspec.GridSpec(3, 3, hspace=0.3)
#        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, sharex=False, sharey='col', subplot_kw=dict(xlabel='Time [ns]'), num='rx' + str(rx), figsize=(20, 10), facecolor='w', edgecolor='w')
        for output in args.outputs:
            outputdata = f[path + output][:]
            if output == 'Ex':
                ax = plt.subplot(gs[0, 0])
                ax.plot(time, outputdata,'r', lw=2, label=output)
                ax.set_ylabel(output + ', field strength [V/m]')
            elif output == 'Ey':
                ax = plt.subplot(gs[1, 0])
                ax.plot(time, outputdata,'r', lw=2, label=output)
                ax.set_ylabel(output + ', field strength [V/m]')
            elif output == 'Ez':
                ax = plt.subplot(gs[2, 0])
                ax.plot(time, outputdata,'r', lw=2, label=output)
                ax.set_ylabel(output + ', field strength [V/m]')
            elif output == 'Hx':
                ax = plt.subplot(gs[0, 1])
                ax.plot(time, outputdata,'b', lw=2, label=output)
                ax.set_ylabel(output + ', field strength [A/m]')
            elif output == 'Hy':
                ax = plt.subplot(gs[1, 1])
                ax.plot(time, outputdata,'b', lw=2, label=output)
                ax.set_ylabel(output + ', field strength [A/m]')
            elif output == 'Hz':
                ax = plt.subplot(gs[2, 1])
                ax.plot(time, outputdata,'b', lw=2, label=output)
                ax.set_ylabel(output + ', field strength [A/m]')
            elif output == 'Ix':
                ax = plt.subplot(gs[0, 2])
                ax.plot(time, outputdata,'b', lw=2, label=output)
                ax.set_ylabel(output + ', current [A]')
            elif output == 'Iy':
                ax = plt.subplot(gs[1, 2])
                ax.plot(time, outputdata,'b', lw=2, label=output)
                ax.set_ylabel(output + ', current [A]')
            elif output == 'Iz':
                ax = plt.subplot(gs[2, 2])
                ax.plot(time, outputdata,'b', lw=2, label=output)
                ax.set_ylabel(output + ', current [A]')
        for ax in fig.axes:
            ax.set_xlim([0, np.amax(time)])
            ax.grid()

    # Save a PDF of the figure
    #fig.savefig(os.path.splitext(os.path.abspath(file))[0] + '.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()
f.close()