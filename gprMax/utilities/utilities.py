# Copyright (C) 2015-2021: The University of Edinburgh
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
import os
import re
import textwrap
import xml.dom.minidom
from shutil import get_terminal_size

import numpy as np
from colorama import Fore, Style, init
init()

logger = logging.getLogger(__name__)

try:
    from time import thread_time as timer_fn
except ImportError:
    from time import perf_counter as timer_fn
    logger.debug('"thread_time" not currently available in macOS and bug'\
                   ' (https://bugs.python.org/issue36205) with "process_time", so use "perf_counter".')


def get_terminal_width():
    """Get/set width of terminal being used.

    Returns:
        terminalwidth (int): Terminal width
    """

    terminalwidth = get_terminal_size()[0]
    if terminalwidth == 0:
        terminalwidth = 100

    return terminalwidth


def logo(version):
    """Print gprMax logo, version, and licencing/copyright information.

    Args:
        version (str): Version number.

    Returns:
        (str): Containing logo, version, and licencing/copyright information.
    """

    description = '\n=== Electromagnetic modelling software based on the Finite-Difference Time-Domain (FDTD) method'
    current_year = datetime.datetime.now().year
    copyright = f'Copyright (C) 2015-{current_year}: The University of Edinburgh'
    authors = 'Authors: Craig Warren, Antonis Giannopoulos, and John Hartley'
    licenseinfo1 = 'gprMax is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.\n'
    licenseinfo2 = 'gprMax is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.'
    licenseinfo3 = 'You should have received a copy of the GNU General Public License along with gprMax.  If not, see www.gnu.org/licenses.'

    logo = """    www.gprmax.com   __  __
     __ _ _ __  _ __|  \/  | __ ___  __
    / _` | '_ \| '__| |\/| |/ _` \ \/ /
   | (_| | |_) | |  | |  | | (_| |>  <
    \__, | .__/|_|  |_|  |_|\__,_/_/\_\\
    |___/|_|
                     v""" + version + '\n\n'

    str = f"{description} {'=' * (get_terminal_width() - len(description) - 1)}\n\n"
    str += Fore.CYAN + f'{logo}'
    str += Style.RESET_ALL + textwrap.fill(copyright, width=get_terminal_width() - 1, initial_indent=' ') + '\n'
    str += textwrap.fill(authors, width=get_terminal_width() - 1, initial_indent=' ') + '\n\n'
    str += textwrap.fill(licenseinfo1, width=get_terminal_width() - 1, initial_indent=' ', subsequent_indent='  ') + '\n'
    str += textwrap.fill(licenseinfo2, width=get_terminal_width() - 1, initial_indent=' ', subsequent_indent='  ') + '\n'
    str += textwrap.fill(licenseinfo3, width=get_terminal_width() - 1, initial_indent=' ', subsequent_indent='  ')

    return str


def pretty_xml(roughxml):
    """Nicely format XML string.

    Args:
        roughxml (str): XML string to format

    Returns:
        prettyxml (str): nicely formatted XML string
    """

    prettyxml = xml.dom.minidom.parseString(roughxml).toprettyxml()
    # Remove the weird newline issue
    prettyxml = os.linesep.join(
        [s for s in prettyxml.splitlines() if s.strip()])

    return prettyxml


def round_value(value, decimalplaces=0):
    """Rounding function.

    Args:
        value (float): Number to round.
        decimalplaces (int): Number of decimal places of float to represent
                                rounded value.

    Returns:
        rounded (int/float): Rounded value.
    """

    # Rounds to nearest integer (half values are rounded downwards)
    if decimalplaces == 0:
        rounded = int(d.Decimal(value).quantize(d.Decimal('1'), rounding=d.ROUND_HALF_DOWN))

    # Rounds down to nearest float represented by number of decimal places
    else:
        precision = '1.{places}'.format(places='0' * decimalplaces)
        rounded = float(d.Decimal(value).quantize(d.Decimal(precision), rounding=d.ROUND_FLOOR))

    return rounded


def round32(value):
    """Rounds up to nearest multiple of 32."""
    return int(32 * np.ceil(float(value) / 32))


def fft_power(waveform, dt):
    """Calculate a FFT of the given waveform of amplitude values;
        converted to decibels and shifted so that maximum power is 0dB

    Args:
        waveform (ndarray): time domain waveform
        dt (float): time step

    Returns:
        freqs (ndarray): frequency bins
        power (ndarray): power
    """

    # Calculate magnitude of frequency spectra of waveform (ignore warning from taking a log of any zero values)
    with np.errstate(divide='ignore'):
        power = 10 * np.log10(np.abs(np.fft.fft(waveform))**2)

    # Replace any NaNs or Infs from zero division
    power[np.invert(np.isfinite(power))] = 0

    # Frequency bins
    freqs = np.fft.fftfreq(power.size, d=dt)

    # Shift powers so that frequency with maximum power is at zero decibels
    power -= np.amax(power)

    return freqs, power


def human_size(size, a_kilobyte_is_1024_bytes=False):
    """Convert a file size to human-readable form.

    Args:
        size (int): file size in bytes.
        a_kilobyte_is_1024_bytes (boolean) - true for multiples of 1024, false for multiples of 1000.

    Returns:
        Human-readable (string).
    """

    suffixes = {1000: ['KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'], 1024: ['KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB']}

    if size < 0:
        raise ValueError('Number must be non-negative.')

    multiple = 1024 if a_kilobyte_is_1024_bytes else 1000
    for suffix in suffixes[multiple]:
        size /= multiple
        if size < multiple:
            return '{:.3g}{}'.format(size, suffix)

    raise ValueError('Number is too large.')


def atoi(text):
    """Converts a string into an integer."""
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """Human sorting of a string."""
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def timer():
    """Function to return time in fractional seconds."""
    return timer_fn()
