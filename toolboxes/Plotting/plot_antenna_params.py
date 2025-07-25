# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, and John Hartley
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
import logging
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def calculate_antenna_params(
    filename, tltxnumber=1, tlrxnumber=None, rxnumber=None, rxcomponent=None
):
    """Calculates antenna parameters - incident, reflected and total volatges
        and currents; s11, (s21) and input impedance.

    Args:
        filename: string of filename (including path) of output file.
        tltxnumber: int for transmitter antenna - transmission line number
        tlrxnumber: int for receiver antenna - transmission line number
        rxnumber: int for receiver antenna - output number
        rxcomponent: in for receiver antenna - output electric field component

    Returns:
        antennaparams: dict of antenna parameters.
    """

    # Open output file and read some attributes
    file = Path(filename)
    f = h5py.File(file, "r")
    dxdydz = f.attrs["dx_dy_dz"]
    dt = f.attrs["dt"]
    iterations = f.attrs["Iterations"]

    # Calculate time array and frequency bin spacing
    time = np.linspace(0, (iterations - 1) * dt, num=iterations)
    df = 1 / np.amax(time)

    logger.info(f"Time window: {np.amax(time):g} s ({iterations} iterations)")
    logger.info(f"Time step: {dt:g} s")
    logger.info(f"Frequency bin spacing: {df:g} Hz")

    # Read/calculate voltages and currents from transmitter antenna
    tltxpath = "/tls/tl" + str(tltxnumber) + "/"

    # Incident voltages/currents
    Vinc = f[tltxpath + "Vinc"][:]
    Iinc = f[tltxpath + "Iinc"][:]

    # Total (incident + reflected) voltages/currents
    Vtotal = f[tltxpath + "Vtotal"][:]
    Itotal = f[tltxpath + "Itotal"][:]

    # Reflected voltages/currents
    Vref = Vtotal - Vinc
    Iref = Itotal - Iinc

    # If a receiver antenna is used (with a transmission line or receiver),
    # get received voltage for s21
    if tlrxnumber:
        tlrxpath = "/tls/tl" + str(tlrxnumber) + "/"
        Vrec = f[tlrxpath + "Vtotal"][:]

    elif rxnumber:
        rxpath = "/rxs/rx" + str(rxnumber) + "/"
        availableoutputs = list(f[rxpath].keys())

        if rxcomponent not in availableoutputs:
            logger.exception(
                f"{rxcomponent} output requested, but the available "
                + f"output for receiver {rxnumber} is "
                + f"{', '.join(availableoutputs)}"
            )
            raise ValueError

        rxpath += rxcomponent

        # Received voltage
        if rxcomponent == "Ex":
            Vrec = f[rxpath][:] * -1 * dxdydz[0]
        elif rxcomponent == "Ey":
            Vrec = f[rxpath][:] * -1 * dxdydz[1]
        elif rxcomponent == "Ez":
            Vrec = f[rxpath][:] * -1 * dxdydz[2]
    f.close()

    # Frequency bins
    freqs = np.fft.fftfreq(Vinc.size, d=dt)

    # Delay correction - current lags voltage, so delay voltage to match
    # current timestep
    delaycorrection = np.exp(1j * 2 * np.pi * freqs * (dt / 2))

    # Calculate s11 and (optionally) s21
    s11 = np.abs(np.fft.fft(Vref) / np.fft.fft(Vinc))
    if tlrxnumber or rxnumber:
        s21 = np.abs(np.fft.fft(Vrec) / np.fft.fft(Vinc))

    # Calculate input impedance
    zin = (np.fft.fft(Vtotal) * delaycorrection) / np.fft.fft(Itotal)

    # Calculate input admittance
    yin = np.fft.fft(Itotal) / (np.fft.fft(Vtotal) * delaycorrection)

    # Convert to decibels (ignore warning from taking a log of any zero values)
    with np.errstate(divide="ignore"):
        Vincp = 20 * np.log10(np.abs((np.fft.fft(Vinc) * delaycorrection)))
        Iincp = 20 * np.log10(np.abs(np.fft.fft(Iinc)))
        Vrefp = 20 * np.log10(np.abs((np.fft.fft(Vref) * delaycorrection)))
        Irefp = 20 * np.log10(np.abs(np.fft.fft(Iref)))
        Vtotalp = 20 * np.log10(np.abs((np.fft.fft(Vtotal) * delaycorrection)))
        Itotalp = 20 * np.log10(np.abs(np.fft.fft(Itotal)))
        s11 = 20 * np.log10(s11)

    # Replace any NaNs or Infs from zero division
    Vincp[np.invert(np.isfinite(Vincp))] = 0
    Iincp[np.invert(np.isfinite(Iincp))] = 0
    Vrefp[np.invert(np.isfinite(Vrefp))] = 0
    Irefp[np.invert(np.isfinite(Irefp))] = 0
    Vtotalp[np.invert(np.isfinite(Vtotalp))] = 0
    Itotalp[np.invert(np.isfinite(Itotalp))] = 0
    s11[np.invert(np.isfinite(s11))] = 0

    # Create dictionary of antenna parameters
    antennaparams = {
        "time": time,
        "freqs": freqs,
        "Vinc": Vinc,
        "Vincp": Vincp,
        "Iinc": Iinc,
        "Iincp": Iincp,
        "Vref": Vref,
        "Vrefp": Vrefp,
        "Iref": Iref,
        "Irefp": Irefp,
        "Vtotal": Vtotal,
        "Vtotalp": Vtotalp,
        "Itotal": Itotal,
        "Itotalp": Itotalp,
        "s11": s11,
        "zin": zin,
        "yin": yin,
    }
    if tlrxnumber or rxnumber:
        with np.errstate(divide="ignore"):  # Ignore warning from taking a log of any zero values
            s21 = 20 * np.log10(s21)
        s21[np.invert(np.isfinite(s21))] = 0
        antennaparams["s21"] = s21

    return antennaparams


def mpl_plot(
    filename,
    time,
    freqs,
    Vinc,
    Vincp,
    Iinc,
    Iincp,
    Vref,
    Vrefp,
    Iref,
    Irefp,
    Vtotal,
    Vtotalp,
    Itotal,
    Itotalp,
    s11,
    zin,
    yin,
    s21=None,
    save=False,
):
    """Plots antenna parameters - incident, reflected and total voltages and
        currents; s11, (s21) and input impedance.

    Args:
        filename: string of filename (including path) of output file.
        time: array of simulation time.
        freq: array of frequencies for FFTs.
        Vinc, Vincp, Iinc, Iincp: arrays of time and frequency domain
                                    representations of incident voltage and
                                    current.
        Vref, Vrefp, Iref, Irefp: arrays of time and frequency domain
                                    representations of reflected voltage and
                                    current.
        Vtotal, Vtotalp, Itotal, Itotalp: arrays of time and frequency domain
                                            representations of total voltage and
                                            current.
        s11, s21: array(s) of s11 and, optionally, s21 parameters.
        zin, yin: arrays of input impedance and input admittance parameters.
        save: boolean flag to save plot to file.

    Returns:
        plt: matplotlib plot object.
    """

    # Set plotting range
    pltrangemin = 1
    # To a certain drop from maximum power
    pltrangemax = np.where((np.amax(Vincp[1::]) - Vincp[1::]) > 60)[0][0] + 1
    # To a maximum frequency
    pltrangemax = np.where(freqs > 3e9)[0][0]
    pltrange = np.s_[pltrangemin:pltrangemax]

    # Print some useful values from s11, and input impedance
    s11minfreq = np.where(s11[pltrange] == np.amin(s11[pltrange]))[0][0]
    logger.info(
        f"s11 minimum: {np.amin(s11[pltrange]):g} dB at "
        + f"{freqs[s11minfreq + pltrangemin]:g} Hz"
    )
    logger.info(f"At {freqs[s11minfreq + pltrangemin]:g} Hz...")
    logger.info(
        f"Input impedance: {np.abs(zin[s11minfreq + pltrangemin]):.1f}"
        + f"{zin[s11minfreq + pltrangemin].imag:+.1f}j Ohms"
    )
    # logger.info(f'Input admittance (mag): {np.abs(yin[s11minfreq + pltrangemin]):g} S')
    # logger.info(f'Input admittance (phase): {np.angle(yin[s11minfreq + pltrangemin], deg=True):.1f} deg')

    # Figure 1
    # Plot incident voltage
    fig1, axs = plt.subplots(
        num="Transmitter transmission line parameters",
        figsize=(20, 12),
        nrows=4,
        ncols=2,
        facecolor="w",
        edgecolor="w",
    )
    plt.subplots_adjust(hspace=0.75)
    axs[0, 0].plot(time, Vinc, "r", lw=2, label="Vinc")
    axs[0, 0].set_title("Incident voltage")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Voltage [V]")
    axs[0, 0].set_xlim([0, np.amax(time)])
    axs[0, 0].grid(which="both", axis="both", linestyle="-.")

    # Plot frequency spectra of incident voltage
    markerline, stemlines, baseline = axs[0, 1].stem(freqs[pltrange], Vincp[pltrange], "-.")
    plt.setp(baseline, "linewidth", 0)
    plt.setp(stemlines, "color", "r")
    plt.setp(markerline, "markerfacecolor", "r", "markeredgecolor", "r")
    axs[0, 1].plot(freqs[pltrange], Vincp[pltrange], "r", lw=2)
    axs[0, 1].set_title("Incident voltage")
    axs[0, 1].set_xlabel("Frequency [Hz]")
    axs[0, 1].set_ylabel("Power [dB]")
    axs[0, 1].grid(which="both", axis="both", linestyle="-.")

    # Plot incident current
    axs[1, 0].plot(time, Iinc, "b", lw=2, label="Vinc")
    axs[1, 0].set_title("Incident current")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Current [A]")
    axs[1, 0].set_xlim([0, np.amax(time)])
    axs[1, 0].grid(which="both", axis="both", linestyle="-.")

    # Plot frequency spectra of incident current
    markerline, stemlines, baseline = axs[1, 1].stem(freqs[pltrange], Iincp[pltrange], "-.")
    plt.setp(baseline, "linewidth", 0)
    plt.setp(stemlines, "color", "b")
    plt.setp(markerline, "markerfacecolor", "b", "markeredgecolor", "b")
    axs[1, 1].plot(freqs[pltrange], Iincp[pltrange], "b", lw=2)
    axs[1, 1].set_title("Incident current")
    axs[1, 1].set_xlabel("Frequency [Hz]")
    axs[1, 1].set_ylabel("Power [dB]")
    axs[1, 1].grid(which="both", axis="both", linestyle="-.")

    # Plot total voltage
    axs[2, 0].plot(time, Vtotal, "r", lw=2, label="Vinc")
    axs[2, 0].set_title("Total (incident + reflected) voltage")
    axs[2, 0].set_xlabel("Time [s]")
    axs[2, 0].set_ylabel("Voltage [V]")
    axs[2, 0].set_xlim([0, np.amax(time)])
    axs[2, 0].grid(which="both", axis="both", linestyle="-.")

    # Plot frequency spectra of total voltage
    markerline, stemlines, baseline = axs[2, 1].stem(freqs[pltrange], Vtotalp[pltrange], "-.")
    plt.setp(baseline, "linewidth", 0)
    plt.setp(stemlines, "color", "r")
    plt.setp(markerline, "markerfacecolor", "r", "markeredgecolor", "r")
    axs[2, 1].plot(freqs[pltrange], Vtotalp[pltrange], "r", lw=2)
    axs[2, 1].set_title("Total (incident + reflected) voltage")
    axs[2, 1].set_xlabel("Frequency [Hz]")
    axs[2, 1].set_ylabel("Power [dB]")
    axs[2, 1].grid(which="both", axis="both", linestyle="-.")

    # Plot total current
    axs[3, 0].plot(time, Itotal, "b", lw=2, label="Vinc")
    axs[3, 0].set_title("Total (incident + reflected) current")
    axs[3, 0].set_xlabel("Time [s]")
    axs[3, 0].set_ylabel("Current [A]")
    axs[3, 0].set_xlim([0, np.amax(time)])
    axs[3, 0].grid(which="both", axis="both", linestyle="-.")

    # Plot frequency spectra of total current
    markerline, stemlines, baseline = axs[3, 1].stem(freqs[pltrange], Itotalp[pltrange], "-.")
    plt.setp(baseline, "linewidth", 0)
    plt.setp(stemlines, "color", "b")
    plt.setp(markerline, "markerfacecolor", "b", "markeredgecolor", "b")
    axs[3, 1].plot(freqs[pltrange], Itotalp[pltrange], "b", lw=2)
    axs[3, 1].set_title("Total (incident + reflected) current")
    axs[3, 1].set_xlabel("Frequency [Hz]")
    axs[3, 1].set_ylabel("Power [dB]")
    axs[3, 1].grid(which="both", axis="both", linestyle="-.")

    # Plot reflected (reflected) voltage
    # axs[4, 0].plot(time, Vref, 'r', lw=2, label='Vref')
    # axs[4, 0].set_title('Reflected voltage')
    # axs[4, 0].set_xlabel('Time [s]')
    # axs[4, 0].set_ylabel('Voltage [V]')
    # axs[4, 0].set_xlim([0, np.amax(time)])
    # axs[4, 0].grid(which='both', axis='both', linestyle='-.')

    # Plot frequency spectra of reflected voltage
    # markerline, stemlines, baseline = axs[4, 1].stem(freqs[pltrange], Vrefp[pltrange], '-.')
    # plt.setp(baseline, 'linewidth', 0)
    # plt.setp(stemlines, 'color', 'r')
    # plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
    # axs[4, 1].plot(freqs[pltrange], Vrefp[pltrange], 'r', lw=2)
    # axs[4, 1].set_title('Reflected voltage')
    # axs[4, 1].set_xlabel('Frequency [Hz]')
    # axs[4, 1].set_ylabel('Power [dB]')
    # axs[4, 1].grid(which='both', axis='both', linestyle='-.')

    # Plot reflected (reflected) current
    # axs[5, 0].plot(time, Iref, 'b', lw=2, label='Iref')
    # axs[5, 0].set_title('Reflected current')
    # axs[5, 0].set_xlabel('Time [s]')
    # axs[5, 0].set_ylabel('Current [A]')
    # axs[5, 0].set_xlim([0, np.amax(time)])
    # axs[5, 0].grid(which='both', axis='both', linestyle='-.')

    # Plot frequency spectra of reflected current
    # markerline, stemlines, baseline = axs[5, 1].stem(freqs[pltrange], Irefp[pltrange], '-.')
    # plt.setp(baseline, 'linewidth', 0)
    # plt.setp(stemlines, 'color', 'b')
    # plt.setp(markerline, 'markerfacecolor', 'b', 'markeredgecolor', 'b')
    # axs[5, 1].plot(freqs[pltrange], Irefp[pltrange], 'b', lw=2)
    # axs[5, 1].set_title('Reflected current')
    # axs[5, 1].set_xlabel('Frequency [Hz]')
    # axs[5, 1].set_ylabel('Power [dB]')
    # axs[5, 1].grid(which='both', axis='both', linestyle='-.')

    # Figure 2
    # Plot frequency spectra of s11
    fig2, axs = plt.subplots(num="Antenna parameters", 
                             figsize=(20, 12),
                             nrows=2,
                             ncols=2,
                             facecolor="w",
                             edgecolor="w"
    )
    plt.subplots_adjust(hspace=0.75)
    markerline, stemlines, baseline = axs[0, 0].stem(freqs[pltrange], s11[pltrange], "-.")
    plt.setp(baseline, "linewidth", 0)
    plt.setp(stemlines, "color", "g")
    plt.setp(markerline, "markerfacecolor", "g", "markeredgecolor", "g")
    axs[0, 0].plot(freqs[pltrange], s11[pltrange], "g", lw=2)
    axs[0, 0].set_title("s11")
    axs[0, 0].set_xlabel("Frequency [Hz]")
    axs[0, 0].set_ylabel("Power [dB]")
    axs[0, 0].grid(which="both", axis="both", linestyle="-.")

    # Plot frequency spectra of s21
    if s21 is not None:
        markerline, stemlines, baseline = axs[0, 1].stem(freqs[pltrange], s21[pltrange], "-.")
        plt.setp(baseline, "linewidth", 0)
        plt.setp(stemlines, "color", "g")
        plt.setp(markerline, "markerfacecolor", "g", "markeredgecolor", "g")
        axs[0, 1].plot(freqs[pltrange], s21[pltrange], "g", lw=2)
        axs[0, 1].set_title("s21")
        axs[0, 1].set_xlabel("Frequency [Hz]")
        axs[0, 1].set_ylabel("Power [dB]")
        axs[0, 1].grid(which="both", axis="both", linestyle="-.")

    # Plot input resistance (real part of impedance)
    markerline, stemlines, baseline = axs[1, 0].stem(freqs[pltrange], zin[pltrange].real, "-.")
    plt.setp(baseline, "linewidth", 0)
    plt.setp(stemlines, "color", "g")
    plt.setp(markerline, "markerfacecolor", "g", "markeredgecolor", "g")
    axs[1, 0].plot(freqs[pltrange], zin[pltrange].real, "g", lw=2)
    axs[1, 0].set_title("Input impedance (resistive)")
    axs[1, 0].set_xlabel("Frequency [Hz]")
    axs[1, 0].set_ylabel("Resistance [Ohms]")
    axs[1, 0].set_ylim(bottom=0)
    axs[1, 0].grid(which="both", axis="both", linestyle="-.")

    # Plot input reactance (imaginery part of impedance)
    markerline, stemlines, baseline = axs[1, 1].stem(freqs[pltrange], zin[pltrange].imag, "-.")
    plt.setp(baseline, "linewidth", 0)
    plt.setp(stemlines, "color", "g")
    plt.setp(markerline, "markerfacecolor", "g", "markeredgecolor", "g")
    axs[1, 1].plot(freqs[pltrange], zin[pltrange].imag, "g", lw=2)
    axs[1, 1].set_title("Input impedance (reactive)")
    axs[1, 1].set_xlabel("Frequency [Hz]")
    axs[1, 1].set_ylabel("Reactance [Ohms]")
    axs[1, 1].grid(which="both", axis="both", linestyle="-.")

    # Plot input admittance (magnitude)
    # markerline, stemlines, baseline = axs[2, 0].stem(freqs[pltrange], np.abs(yin[pltrange]), '-.')
    # plt.setp(baseline, 'linewidth', 0)
    # plt.setp(stemlines, 'color', 'g')
    # plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
    # axs[2, 0].plot(freqs[pltrange], np.abs(yin[pltrange]), 'g', lw=2)
    # axs[2, 0].set_title('Input admittance (magnitude)')
    # axs[2, 0].set_xlabel('Frequency [Hz]')
    # axs[2, 0].set_ylabel('Admittance [Siemens]')
    # axs[2, 0].grid(which='both', axis='both', linestyle='-.')

    # Plot input admittance (phase)
    # markerline, stemlines, baseline = axs[2, 1].stem(freqs[pltrange], np.angle(yin[pltrange], deg=True), '-.')
    # plt.setp(baseline, 'linewidth', 0)
    # plt.setp(stemlines, 'color', 'g')
    # plt.setp(markerline, 'markerfacecolor', 'g', 'markeredgecolor', 'g')
    # axs[2, 1].plot(freqs[pltrange], np.angle(yin[pltrange], deg=True), 'g', lw=2)
    # axs[2, 1].set_title('Input admittance (phase)')
    # axs[2, 1].set_xlabel('Frequency [Hz]')
    # axs[2, 1].set_ylabel('Phase [degrees]')
    # axs[2, 1].grid(which='both', axis='both', linestyle='-.')

    if save:
        filepath = Path(filename)
        savename1 = filepath.stem + "_tl_params"
        savename1 = filepath.parent / savename1
        savename2 = filepath.stem + "_ant_params"
        savename2 = filepath.parent / savename2
        # Save a PDF of the figure
        fig1.savefig(
            savename1.with_suffix(".pdf"),
            dpi=None,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )
        fig2.savefig(
            savename2.with_suffix(".pdf"),
            dpi=None,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.1,
        )
        # Save a PNG of the figure
        # fig1.savefig(savename1.with_suffix('.png'), dpi=150, format='png',
        #             bbox_inches='tight', pad_inches=0.1)
        # fig2.savefig(savename2.with_suffix('.png'), dpi=150, format='png',
        #             bbox_inches='tight', pad_inches=0.1)

    return plt


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Plots antenna parameters - "
        + "incident, reflected and total voltages "
        + "and currents; s11, (s21) and input impedance "
        + "from an output file containing a transmission "
        + "line source.",
        usage="cd gprMax; python -m toolboxes.Plotting.plot_antenna_params outputfile",
    )
    parser.add_argument("outputfile", help="name of output file including path")
    parser.add_argument(
        "--tltx-num", default=1, type=int, help="transmitter antenna - transmission line number"
    )
    parser.add_argument("--tlrx-num", type=int, help="receiver antenna - transmission line number")
    parser.add_argument("--rx-num", type=int, help="receiver antenna - output number")
    parser.add_argument(
        "--rx-component",
        type=str,
        help="receiver antenna - output electric field component",
        choices=["Ex", "Ey", "Ez"],
    )
    parser.add_argument(
        "-save",
        action="store_true",
        default=False,
        help="save plot directly to file, i.e. do not display",
    )
    args = parser.parse_args()

    antennaparams = calculate_antenna_params(
        args.outputfile, args.tltx_num, args.tlrx_num, args.rx_num, args.rx_component
    )
    plthandle = mpl_plot(args.outputfile, **antennaparams, save=args.save)
    plthandle.show()
