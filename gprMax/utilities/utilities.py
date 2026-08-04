# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
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
from shutil import get_terminal_size
from time import perf_counter as timer_fn
from typing import Union

import numpy as np
from colorama import Fore, Style, init

init()

logger = logging.getLogger(__name__)


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

    description = (
        "\n=== Electromagnetic modelling software based on the "
        "Finite-Difference Time-Domain (FDTD) method"
    )
    current_year = datetime.datetime.now().year
    copyright = (
        f"Copyright (C) 2015-{current_year}: The University of " f"Edinburgh, United Kingdom"
    )
    authors = "Authors: Craig Warren, Antonis Giannopoulos, John Hartley and Nathan Mannall"
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
        r"""    www.gprmax.com   __  __
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
    str += (
        Style.RESET_ALL
        + textwrap.fill(copyright, width=get_terminal_width() - 1, initial_indent=" ")
        + "\n"
    )
    str += textwrap.fill(authors, width=get_terminal_width() - 1, initial_indent=" ") + "\n\n"
    str += (
        textwrap.fill(
            licenseinfo1,
            width=get_terminal_width() - 1,
            initial_indent=" ",
            subsequent_indent="  ",
        )
        + "\n"
    )
    str += (
        textwrap.fill(
            licenseinfo2,
            width=get_terminal_width() - 1,
            initial_indent=" ",
            subsequent_indent="  ",
        )
        + "\n"
    )
    str += (
        textwrap.fill(
            licenseinfo3,
            width=get_terminal_width() - 1,
            initial_indent=" ",
            subsequent_indent="  ",
        )
        + "\n"
    )

    return str


def round_int(value: float) -> int:
    """Round number to nearest integer (half values are rounded down).

    Args:
        value: Number to round.

    Returns:
        rounded: Rounded value.
    """
    return int(d.Decimal(float(value)).quantize(d.Decimal("1"), rounding=d.ROUND_HALF_DOWN))


def round_float(value: float, decimalplaces: int) -> float:
    """Round down to a specific number of decimal places.

    Args:
        value: Number to round.
        decimalplaces: Number of decimal places of float to represent
            rounded value.

    Returns:
        rounded: Rounded value.
    """
    precision = f"1.{'0' * decimalplaces}"
    return float(d.Decimal(value).quantize(d.Decimal(precision), rounding=d.ROUND_FLOOR))


def round_value(value: float, decimalplaces: int = 0) -> Union[float, int]:
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
        rounded = round_int(value)

    # Rounds down to nearest float represented by number of decimal places
    else:
        rounded = round_float(value, decimalplaces)

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

    waveform = np.asarray(waveform)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("waveform must be a non-empty one-dimensional array")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and greater than zero")

    magnitude = np.abs(np.fft.fft(waveform))
    maximum = np.max(magnitude)
    if maximum == 0:
        power = np.full(magnitude.shape, -np.inf, dtype=float)
    else:
        # Normalise before taking the logarithm. Spectral nulls intentionally
        # remain -inf rather than being turned into apparent 0 dB peaks.
        with np.errstate(divide="ignore", invalid="ignore"):
            power = 20 * np.log10(magnitude / maximum)

    # Frequency bins
    freqs = np.fft.fftfreq(power.size, d=dt)

    return freqs, power


def timer():
    """Time in fractional seconds."""
    return timer_fn()


def handle_plot_output(plt, fig, base_filename, suffix="", show=True):
    """Shows a figure interactively, or saves it to disk when that is not
        possible (e.g. a headless/non-interactive matplotlib backend), so
        plotting tools do not crash under a headless backend and do not
        silently produce nothing when running non-interactively.

    Args:
        plt: matplotlib.pyplot module (passed in so this can be used from
            tools that manage their own pyplot import).
        fig: matplotlib Figure to show or save.
        base_filename: string/path used to derive the saved PNG's filename.
        suffix: string appended to the base filename before the extension,
            e.g. to distinguish multiple figures from the same run.
        show: bool, whether the caller wants the figure displayed
            interactively. Ignored (falls back to saving) when the current
            matplotlib backend is not interactive.
    """

    is_interactive = plt.get_backend().lower() not in ["agg", "pdf", "svg", "ps", "template"]
    save_path = os.path.splitext(os.path.abspath(base_filename))[0] + suffix + ".png"

    if not show or not is_interactive:
        fig.savefig(save_path, dpi=150, format="png", bbox_inches="tight", pad_inches=0.1)
        logger.info(Fore.GREEN + f"Plot saved to: {save_path}" + Style.RESET_ALL)
        if show and not is_interactive:
            logger.warning(
                Fore.YELLOW
                + "Non-interactive backend detected. Plot was automatically "
                + "saved instead of shown."
                + Style.RESET_ALL
            )
    else:
        plt.show()
