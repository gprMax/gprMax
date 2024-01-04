# Copyright (C) 2015-2024: The University of Edinburgh, United Kingdom
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

import datetime
import decimal as d
import logging
import re
import textwrap
from shutil import get_terminal_size

import numpy as np
from colorama import Fore, Style, init

init()

logger = logging.getLogger(__name__)

try:
    from time import thread_time as timer_fn
except ImportError:
    # thread_time not always available in macOS
    from time import process_time as timer_fn


def get_terminal_width():
    """Gets/sets width of terminal being used.

    Returns:
        terminalwidth: int for the terminal width.
    """

    terminalwidth = get_terminal_size()[0]
    if terminalwidth == 0:
        terminalwidth = 100

    return terminalwidth


def atoi(text):
    """Converts a string into an integer."""
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """Human sorting of a string."""
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def logo(version):
    """Prints gprMax logo, version, and licencing/copyright information.

    Args:
        version: string for version number.

    Returns:
        str: string containing logo, version, and licencing/copyright info.
    """

    description = "\n=== Electromagnetic modelling software based on the " "Finite-Difference Time-Domain (FDTD) method"
    current_year = datetime.datetime.now().year
    copyright = f"Copyright (C) 2015-{current_year}: The University of " "Edinburgh, United Kingdom"
    authors = "Authors: Craig Warren, Antonis Giannopoulos, and John Hartley"
    licenseinfo1 = (
        "gprMax is free software: you can redistribute it and/or "
        "modify it under the terms of the GNU General Public "
        "License as published by the Free Software Foundation, "
        "either version 3 of the License, or (at your option) any "
        "later version.\n"
    )
    licenseinfo2 = (
        "gprMax is distributed in the hope that it will be useful, "
        "but WITHOUT ANY WARRANTY; without even the implied "
        "warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR "
        "PURPOSE.  See the GNU General Public License for more "
        "details."
    )
    licenseinfo3 = (
        "You should have received a copy of the GNU General Public "
        "License along with gprMax.  If not, "
        "see www.gnu.org/licenses."
    )

    logo = (
        """    www.gprmax.com   __  __
     __ _ _ __  _ __|  \/  | __ ___  __
    / _` | '_ \| '__| |\/| |/ _` \ \/ /
   | (_| | |_) | |  | |  | | (_| |>  <
    \__, | .__/|_|  |_|  |_|\__,_/_/\_\\
    |___/|_|
                     v"""
        + version
        + "\n\n"
    )

    str = f"{description} {'=' * (get_terminal_width() - len(description) - 1)}\n\n"
    str += f"{Fore.CYAN}{logo}"
    str += Style.RESET_ALL + textwrap.fill(copyright, width=get_terminal_width() - 1, initial_indent=" ") + "\n"
    str += textwrap.fill(authors, width=get_terminal_width() - 1, initial_indent=" ") + "\n\n"
    str += (
        textwrap.fill(licenseinfo1, width=get_terminal_width() - 1, initial_indent=" ", subsequent_indent="  ") + "\n"
    )
    str += (
        textwrap.fill(licenseinfo2, width=get_terminal_width() - 1, initial_indent=" ", subsequent_indent="  ") + "\n"
    )
    str += textwrap.fill(licenseinfo3, width=get_terminal_width() - 1, initial_indent=" ", subsequent_indent="  ")

    return str


def round_value(value, decimalplaces=0):
    """Rounding function.

    Args:
        value: float of number to round.
        decimalplaces: int for number of decimal places of float to represent
                        rounded value.

    Returns:
        rounded: int/float of rounded value.
    """

    # Rounds to nearest integer (half values are rounded downwards)
    if decimalplaces == 0:
        rounded = int(d.Decimal(value).quantize(d.Decimal("1"), rounding=d.ROUND_HALF_DOWN))

    # Rounds down to nearest float represented by number of decimal places
    else:
        precision = f"1.{'0' * decimalplaces}"
        rounded = float(d.Decimal(value).quantize(d.Decimal(precision), rounding=d.ROUND_FLOOR))

    return rounded


def round32(value):
    """Rounds up to nearest multiple of 32."""
    return int(32 * np.ceil(float(value) / 32))


def fft_power(waveform, dt):
    """Calculates FFT of the given waveform of amplitude values;
        converted to decibels and shifted so that maximum power is 0dB

    Args:
        waveform: array containing time domain waveform.
        dt: float of time step.

    Returns:
        freq: array of frequency bins.
        power: array containing power spectra.
    """

    # Calculate magnitude of frequency spectra of waveform (ignore warning from
    # taking a log of any zero values)
    with np.errstate(divide="ignore"):
        power = 10 * np.log10(np.abs(np.fft.fft(waveform)) ** 2)

    # Replace any NaNs or Infs from zero division
    power[np.invert(np.isfinite(power))] = 0

    # Frequency bins
    freqs = np.fft.fftfreq(power.size, d=dt)

    # Shift powers so that frequency with maximum power is at zero decibels
    power -= np.amax(power)

    return freqs, power


def timer():
    """Time in fractional seconds."""
    return timer_fn()
