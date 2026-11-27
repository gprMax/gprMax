# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

import datetime
import decimal as d
import logging
import os
import re
import textwrap
from numbers import Integral
from pathlib import Path
from shutil import get_terminal_size
from time import perf_counter as timer_fn
from typing import Union

import numpy as np
from colorama import Fore, Style, init

from gprMax.toolboxes.Utilities.output_paths import atomic_output_path, validate_output_path

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
        r"""    www.gprmax.org   __  __
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
    str += "\n"
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

    Raises:
        ValueError: If value is not finite (inf or NaN).
    """
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"round_int requires a finite value, got {value!r}")
    return int(d.Decimal(number).quantize(d.Decimal("1"), rounding=d.ROUND_HALF_DOWN))


def round_float(value: float, decimalplaces: int) -> float:
    """Round down to a specific number of decimal places.

    Args:
        value: Number to round.
        decimalplaces: Number of decimal places of float to represent
            rounded value.

    Returns:
        rounded: Rounded value.

    Raises:
        ValueError: If value is not finite (inf or NaN).
    """
    if (
        isinstance(decimalplaces, bool)
        or not isinstance(decimalplaces, Integral)
        or decimalplaces < 0
    ):
        raise ValueError(f"decimalplaces must be a non-negative integer, got {decimalplaces!r}")
    if not np.isfinite(value):
        raise ValueError(f"round_float requires a finite value, got {value!r}")
    precision = f"1.{'0' * decimalplaces}"
    # str() first: Decimal(0.99) is 0.98999... and would floor to 0.98.
    return float(d.Decimal(str(value)).quantize(d.Decimal(precision), rounding=d.ROUND_FLOOR))


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
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"round32 requires a numeric value, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"round32 requires a numeric value, got {value!r}") from exc
    if not np.isfinite(number) or number < 0:
        raise ValueError(f"round32 requires a finite value >= 0, got {value!r}")
    return int(32 * np.ceil(number / 32))


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


def handle_plot_output(plt, fig, base_filename, suffix="", show=True, *, input_filenames=None):
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
        input_filenames: optional collection of input files that the saved
            plot destination must not overwrite.
    """

    is_interactive = plt.get_backend().lower() not in ["agg", "pdf", "svg", "ps", "template"]
    base_path = Path(os.path.abspath(base_filename))
    if base_path.suffix.lower() in {".h5", ".hdf5", ".png"}:
        base_stem = base_path.stem
    else:
        base_stem = base_path.name
    destination = base_path.with_name(f"{base_stem}{suffix}.png")

    if input_filenames is not None:
        if isinstance(input_filenames, (str, Path)):
            actual_inputs = [Path(input_filenames)]
        else:
            actual_inputs = [Path(p) for p in input_filenames]
    elif base_path.suffix.lower() in {".h5", ".hdf5", ".out"}:
        actual_inputs = [base_path]
    else:
        actual_inputs = []

    if not show or not is_interactive:
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if actual_inputs:
                validate_output_path(
                    destination,
                    actual_inputs,
                    message="plot destination must not overwrite an input file",
                )
            with atomic_output_path(destination) as temporary:
                fig.savefig(temporary, dpi=150, format="png", bbox_inches="tight", pad_inches=0.1)
        except Exception:
            plt.close(fig)
            raise
        logger.info(Fore.GREEN + f"Plot saved to: {destination}" + Style.RESET_ALL)
        if show and not is_interactive:
            logger.warning(
                Fore.YELLOW
                + "Non-interactive backend detected. Plot was automatically "
                + "saved instead of shown."
                + Style.RESET_ALL
            )
    else:
        plt.show()
