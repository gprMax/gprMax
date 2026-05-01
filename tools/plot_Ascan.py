# Copyright (C) 2015-2023: The University of Edinburgh
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

import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

from gprMax.exceptions import CmdInputError
from gprMax.receivers import Rx
from gprMax.utilities import fft_power


def _normalize_data(data):
    """Normalize data to [0, 1] range with safe division.

    Args:
        data (ndarray): Input signal data.

    Returns:
        ndarray: Normalized data, or zeros if max amplitude is zero.
    """
    max_val = np.amax(np.abs(data))
    if max_val == 0:
        return np.zeros_like(data)
    return data / max_val


def mpl_plot(filename, outputs=Rx.defaultoutputs, fft=False,
             show_grid=True, normalize=False):
    """Plots electric and magnetic fields and currents from all receiver points
    in the given output file. Each receiver point is plotted in a new figure
    window.

    Args:
        filename (string): Filename (including path) of output file.
        outputs (list): List of field/current components to plot.
        fft (boolean): Plot FFT switch.
        show_grid (boolean): Whether to display grid lines on plots.
        normalize (boolean): Normalize output signals to [0, 1] range.

    Returns:
        plt (object): matplotlib plot object.
    """

    # Derive display name from filename for titles
    basename = os.path.basename(filename)

    # Open output file and read some attributes
    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']
    dt = f.attrs['dt']
    iterations = f.attrs['Iterations']
    time = np.linspace(0, (iterations - 1) * dt, num=iterations)

    # Convert time to nanoseconds for display
    time_ns = time * 1e9

    # Check there are any receivers
    if nrx == 0:
        raise CmdInputError('No receivers found in {}'.format(filename))

    # Check for single output component when doing a FFT
    if fft:
        if not len(outputs) == 1:
            raise CmdInputError('A single output must be specified when using the -fft option')

    # New plot for each receiver
    for rx in range(1, nrx + 1):
        path = '/rxs/rx' + str(rx) + '/'
        availableoutputs = list(f[path].keys())

        # If only a single output is required, create one subplot
        if len(outputs) == 1:

            # Check for polarity of output and if requested output is in file
            if outputs[0][-1] == '-':
                polarity = -1
                outputtext = '-' + outputs[0][0:-1]
                output = outputs[0][0:-1]
            else:
                polarity = 1
                outputtext = outputs[0]
                output = outputs[0]

            if output not in availableoutputs:
                raise CmdInputError('{} output requested to plot, but the available output for receiver 1 is {}'.format(output, ', '.join(availableoutputs)))

            outputdata = f[path + output][:] * polarity

            # Optionally normalize the output data
            if normalize:
                outputdata = _normalize_data(outputdata)

            # Plotting if FFT required
            if fft:
                # FFT
                freqs, power = fft_power(outputdata, dt)
                freqmaxpower = np.where(np.isclose(power, 0))[0][0]

                # Set plotting range to -60dB from maximum power or 4 times
                # frequency at maximum power
                try:
                    pltrange = np.where(power[freqmaxpower:] < -60)[0][0] + freqmaxpower + 1
                except:
                    pltrange = freqmaxpower * 4

                pltrange = np.s_[0:pltrange]

                # Plot time history of output component
                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, num='rx' + str(rx), figsize=(20, 10), facecolor='w', edgecolor='w')

                fig.suptitle('A-scan Output - {} (rx{})'.format(basename, rx))

                ylabel = 'Normalized amplitude' if normalize else outputtext + ' field strength [V/m]'
                line1 = ax1.plot(time_ns, outputdata, 'r', lw=2, label=outputtext)
                ax1.set_xlabel('Time (ns)')
                ax1.set_ylabel(ylabel)
                ax1.set_xlim([0, np.amax(time_ns)])
                if show_grid:
                    ax1.grid(which='both', axis='both', linestyle='-.')

                # Plot frequency spectra
                markerline, stemlines, baseline = ax2.stem(freqs[pltrange], power[pltrange], '-.')
                plt.setp(baseline, 'linewidth', 0)
                plt.setp(stemlines, 'color', 'r')
                plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
                line2 = ax2.plot(freqs[pltrange], power[pltrange], 'r', lw=2)
                ax2.set_xlabel('Frequency [Hz]')
                ax2.set_ylabel('Power [dB]')
                if show_grid:
                    ax2.grid(which='both', axis='both', linestyle='-.')

                # Change colours and labels for magnetic field components or currents
                if 'H' in outputs[0]:
                    plt.setp(line1, color='g')
                    plt.setp(line2, color='g')
                    if not normalize:
                        plt.setp(ax1, ylabel=outputtext + ' field strength [A/m]')
                    plt.setp(stemlines, 'color', 'g')
                    plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
                elif 'I' in outputs[0]:
                    plt.setp(line1, color='b')
                    plt.setp(line2, color='b')
                    if not normalize:
                        plt.setp(ax1, ylabel=outputtext + ' current [A]')
                    plt.setp(stemlines, 'color', 'b')
                    plt.setp(markerline, 'markerfacecolor', 'b', 'markeredgecolor', 'b')

                plt.tight_layout()
                plt.show()

            # Plotting if no FFT required
            else:
                ylabel = 'Normalized amplitude' if normalize else outputtext + ' field strength [V/m]'
                fig, ax = plt.subplots(subplot_kw=dict(xlabel='Time (ns)', ylabel=ylabel), num='rx' + str(rx), figsize=(20, 10), facecolor='w', edgecolor='w')

                fig.suptitle('A-scan Output - {} (rx{})'.format(basename, rx))

                line = ax.plot(time_ns, outputdata, 'r', lw=2, label=outputtext)
                ax.set_xlim([0, np.amax(time_ns)])
                if show_grid:
                    ax.grid(which='both', axis='both', linestyle='-.')

                if 'H' in output:
                    plt.setp(line, color='g')
                    if not normalize:
                        plt.setp(ax, ylabel=outputtext + ', field strength [A/m]')
                elif 'I' in output:
                    plt.setp(line, color='b')
                    if not normalize:
                        plt.setp(ax, ylabel=outputtext + ', current [A]')

        # If multiple outputs required, create all subplots and populate
        else:
            plt_cols = 3 if len(outputs) == 9 else 2

            fig, axs = plt.subplots(
                subplot_kw=dict(xlabel="Time (ns)"),
                num='rx' + str(rx),
                figsize=(20, 10),
                nrows=3,
                ncols=plt_cols,
                facecolor="w",
                edgecolor="w",
            )

            fig.suptitle('A-scan Output - {} (rx{})'.format(basename, rx))

            for output in outputs:
                # Check for polarity of output and if requested output is in file
                if output[-1] == '-':
                    polarity = -1
                    outputtext = '-' + output[0:-1]
                    output = output[0:-1]
                else:
                    polarity = 1
                    outputtext = output

                # Check if requested output is in file
                if output not in availableoutputs:
                    raise CmdInputError('Output(s) requested to plot: {}, but available output(s) for receiver {} in the file: {}'.format(', '.join(outputs), rx, ', '.join(availableoutputs)))

                outputdata = f[path + output][:] * polarity

                # Optionally normalize the output data
                if normalize:
                    outputdata = _normalize_data(outputdata)

                # Determine y-axis label based on component type and normalization
                if normalize:
                    ylabel = 'Normalized amplitude'
                elif output.startswith(('E',)):
                    ylabel = outputtext + ", field strength [V/m]"
                elif output.startswith(('H',)):
                    ylabel = outputtext + ", field strength [A/m]"
                elif output.startswith(('I',)):
                    ylabel = outputtext + ", current [A]"
                else:
                    ylabel = "Amplitude"

                if output == "Ex":
                    axs[0, 0].plot(time_ns, outputdata, "r", lw=2, label=outputtext)
                    axs[0, 0].set_ylabel(ylabel)
                elif output == "Ey":
                    axs[1, 0].plot(time_ns, outputdata, "r", lw=2, label=outputtext)
                    axs[1, 0].set_ylabel(ylabel)
                elif output == "Ez":
                    axs[2, 0].plot(time_ns, outputdata, "r", lw=2, label=outputtext)
                    axs[2, 0].set_ylabel(ylabel)
                elif output == "Hx":
                    axs[0, 1].plot(time_ns, outputdata, "g", lw=2, label=outputtext)
                    axs[0, 1].set_ylabel(ylabel)
                elif output == "Hy":
                    axs[1, 1].plot(time_ns, outputdata, "g", lw=2, label=outputtext)
                    axs[1, 1].set_ylabel(ylabel)
                elif output == "Hz":
                    axs[2, 1].plot(time_ns, outputdata, "g", lw=2, label=outputtext)
                    axs[2, 1].set_ylabel(ylabel)
                elif output == "Ix":
                    axs[0, 2].plot(time_ns, outputdata, "b", lw=2, label=outputtext)
                    axs[0, 2].set_ylabel(ylabel)
                elif output == "Iy":
                    axs[1, 2].plot(time_ns, outputdata, "b", lw=2, label=outputtext)
                    axs[1, 2].set_ylabel(ylabel)
                elif output == "Iz":
                    axs[2, 2].plot(time_ns, outputdata, "b", lw=2, label=outputtext)
                    axs[2, 2].set_ylabel(ylabel)

                for ax in fig.axes:
                    ax.set_xlim([0, np.amax(time_ns)])
                    if show_grid:
                        ax.grid(which="both", axis="both", linestyle="-.")

        # Apply tight layout to prevent label clipping
        fig.tight_layout()

        # Save a PDF/PNG of the figure
        # fig.savefig(os.path.splitext(os.path.abspath(filename))[0] + '_rx' + str(rx) + '.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
        # fig.savefig(os.path.splitext(os.path.abspath(filename))[0] + '_rx' + str(rx) + '.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

    f.close()

    return plt


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Plots electric and magnetic fields and currents from all '
                    'receiver points in the given output file. Each receiver '
                    'point is plotted in a new figure window.',
        usage='cd gprMax; python -m tools.plot_Ascan outputfile'
    )
    parser.add_argument('outputfile', help='name of output file including path')
    parser.add_argument('--outputs', help='outputs to be plotted',
                        default=Rx.defaultoutputs,
                        choices=['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz',
                                 'Ix', 'Iy', 'Iz', 'Ex-', 'Ey-', 'Ez-',
                                 'Hx-', 'Hy-', 'Hz-', 'Ix-', 'Iy-', 'Iz-'],
                        nargs='+')
    parser.add_argument('-fft', action='store_true',
                        help='plot FFT (single output must be specified)',
                        default=False)
    parser.add_argument('--save', metavar='FILENAME',
                        help='save plot to file (e.g. output.png, output.pdf)')
    parser.add_argument('--no-grid', action='store_true', default=False,
                        help='disable grid lines on plots')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='normalize signal amplitudes to [0, 1] range')
    args = parser.parse_args()

    plthandle = mpl_plot(args.outputfile, args.outputs, fft=args.fft,
                         show_grid=not args.no_grid,
                         normalize=args.normalize)

    # Save plot to file if --save is provided, otherwise show interactively
    if args.save:
        plthandle.savefig(args.save, dpi=150, bbox_inches='tight',
                          pad_inches=0.1)
        print('Plot saved to: {}'.format(os.path.abspath(args.save)))
    else:
        plthandle.show()
