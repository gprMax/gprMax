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
import numpy as np
np.seterr(divide='ignore')
import matplotlib.pyplot as plt

from gprMax.exceptions import CmdInputError
from gprMax.utilities import round_value
from gprMax.waveforms import Waveform


def check_timewindow(timewindow, dt):
    """Checks and sets time window and number of iterations.
            
    Args:
        timewindow (float): Time window.
        dt (float): Time discretisation.
            
    Returns:
        timewindow (float): Time window.
        iterations (int): Number of interations.
    """
    
    # Time window could be a string, float or int, so convert to string then check
    timewindow = str(timewindow)
    
    try:
        timewindow = int(timewindow)
        iterations = timewindow
        timewindow = (timewindow - 1) * dt

    except:
        timewindow = float(timewindow)
        if timewindow > 0:
            iterations = round_value((timewindow / dt)) + 1
        else:
            raise CmdInputError('Time window must have a value greater than zero')

    return timewindow, iterations


def mpl_plot(w, timewindow, dt, iterations, fft=False):
    """Plots waveform and prints useful information about its properties.
            
    Args:
        w (class): Waveform class instance.
        timewindow (float): Time window.
        dt (float): Time discretisation.
        iterations (int): Number of iterations.
        fft (boolean): Plot FFT switch.
        
    Returns:
        plt (object): matplotlib plot object.
    """
    
    time = np.linspace(0, 1, iterations)
    time *= (iterations * dt)
    waveform = np.zeros(len(time))
    timeiter = np.nditer(time, flags=['c_index'])

    while not timeiter.finished:
        waveform[timeiter.index] = w.calculate_value(timeiter[0], dt)
        timeiter.iternext()

    print('Waveform characteristics...')
    print('Type: {}'.format(w.type))
    print('Amplitude: {:g}'.format(w.amp))
    print('Centre frequency: {:g} Hz'.format(w.freq))
    print('Time to centre of pulse: {:g} s'.format(1 / w.freq))

    # Calculate pulse width for gaussian
    if w.type == 'gaussian':
        powerdrop = -3 #dB
        start = np.where((10 * np.log10(waveform / np.amax(waveform))) > powerdrop)[0][0]
        stop = np.where((10 * np.log10(waveform[start:] / np.amax(waveform))) < powerdrop)[0][0] + start
        print('Pulse width at {:d}dB, i.e. FWHM: {:g} s'.format(powerdrop, time[stop] - time[start]))

    print('Time window: {:g} s ({} iterations)'.format(timewindow, iterations))
    print('Time step: {:g} s'.format(dt))

    if fft:
        # Calculate magnitude of frequency spectra of waveform
        power = 10 * np.log10(np.abs(np.fft.fft(waveform))**2)
        freqs = np.fft.fftfreq(power.size, d=dt)

        # Shift powers so that frequency with maximum power is at zero decibels
        power -= np.amax(power)

        # Set plotting range to 4 times centre frequency of waveform
        pltrange = np.where(freqs > 4 * w.freq)[0][0]
        pltrange = np.s_[0:pltrange]

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, num=w.type, figsize=(20, 10), facecolor='w', edgecolor='w')
        
        # Plot waveform
        ax1.plot(time, waveform, 'r', lw=2)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Amplitude')

        # Plot frequency spectra
        markerline, stemlines, baseline = ax2.stem(freqs[pltrange], power[pltrange], '-.')
        plt.setp(baseline, 'linewidth', 0)
        plt.setp(stemlines, 'color', 'r')
        plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
        ax2.plot(freqs[pltrange], power[pltrange], 'r', lw=2)
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_ylabel('Power [dB]')

    else:
        fig, ax1 = plt.subplots(num=w.type, figsize=(20, 10), facecolor='w', edgecolor='w')

        # Plot waveform
        ax1.plot(time, waveform, 'r', lw=2)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Amplitude')

    [ax.grid() for ax in fig.axes] # Turn on grid

    # Save a PDF/PNG of the figure
    #fig.savefig(os.path.dirname(os.path.abspath(__file__)) + os.sep + w.type + '.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
    #fig.savefig(os.path.dirname(os.path.abspath(__file__)) + os.sep + w.type + '.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

    return plt


if __name__ == "__main__":
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot built-in waveforms that can be used for sources.', usage='cd gprMax; python -m tools.plot_builtin_wave type amp freq timewindow dt')
    parser.add_argument('type', help='type of waveform', choices=Waveform.types)
    parser.add_argument('amp', type=float, help='amplitude of waveform')
    parser.add_argument('freq', type=float, help='centre frequency of waveform')
    parser.add_argument('timewindow', help='time window to view waveform')
    parser.add_argument('dt', type=float, help='time step to view waveform')
    parser.add_argument('-fft', action='store_true', help='plot FFT of waveform', default=False)
    args = parser.parse_args()

    # Check waveform parameters
    if args.type.lower() not in Waveform.types:
        raise CmdInputError('The waveform must have one of the following types {}'.format(', '.join(Waveform.types)))
    if args.freq <= 0:
        raise CmdInputError('The waveform requires an excitation frequency value of greater than zero')

    # Create waveform instance
    w = Waveform()
    w.type = args.type
    w.amp = args.amp
    w.freq = args.freq

    timewindow, iterations = check_timewindow(args.timewindow, args.dt)
    plt = mpl_plot(w, timewindow, args.dt, iterations, args.fft)
    plt.show()

