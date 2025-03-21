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

import datetime
import logging
import sys
from copy import copy

from colorama import Back, Fore, Style, init

init()

logger = logging.getLogger(__name__)


# Adds a custom log level to the root logger from which new loggers are derived
BASIC_NUM = 25
logging.addLevelName(BASIC_NUM, "BASIC")
logging.BASIC = BASIC_NUM


def basic(self, message, *args, **kws):
    if self.isEnabledFor(BASIC_NUM):
        self._log(BASIC_NUM, message, args, **kws)


logging.Logger.basic = basic


# Colour mapping for different log levels
MAPPING = {
    "DEBUG": Fore.WHITE,
    "INFO": Fore.WHITE,
    "BASIC": Fore.WHITE,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.WHITE + Back.RED,  # white on red bg
}


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors
    (https://stackoverflow.com/a/46482050)."""

    def __init__(self, pattern: str):
        logging.Formatter.__init__(self, pattern)

    def format(self, record: logging.LogRecord) -> str:
        colored_record = copy(record)
        levelname = colored_record.levelname
        colour = MAPPING.get(levelname, Fore.BLUE)  # default white
        colored_levelname = f"{colour}{levelname}{Style.RESET_ALL}"
        colored_record.levelname = colored_levelname
        colored_record.msg = f"{colour}{colored_record.getMessage()}{Style.RESET_ALL}"
        return logging.Formatter.format(self, colored_record)


def logging_config(
    name="gprMax",
    level=logging.INFO,
    format_style="std",
    log_file=False,
    mpi_logger=False,
    log_all_ranks=False,
):
    """Setup and configure logging.

    Args:
        name: string of name of logger to create.
        level: logging level to set logging level to stdout.
        format_style: string to set formatting - 'std' or 'full'
        log_file: bool for additional logging to file.
    """

    format_std = "%(message)s"
    format_full = "%(asctime)s:%(levelname)s:%(name)s:%(lineno)d: %(message)s"

    # Set format style
    if format_style == "full" or level == logging.DEBUG:
        format = format_full
    elif format_style == "std":
        format = format_std

    # Create main top-level logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    # Don't add handlers for non-zero ranks unless logging is turned on
    # for all ranks
    if mpi_logger:
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.rank
        if not log_all_ranks and not rank == 0:
            return

    # Config for logging to console
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    if mpi_logger and log_all_ranks and format == format_full:
        handler.setFormatter(CustomFormatter(f"[Rank {rank}] {format}"))
    else:
        handler.setFormatter(CustomFormatter(format))
    logger.addHandler(handler)

    # Config for logging to file if required
    if log_file:
        if mpi_logger and log_all_ranks:
            filename = f"{name}-log-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{MPI.COMM_WORLD.rank}.txt"
        else:
            filename = name + "-log-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt"
        handler = logging.FileHandler(filename, mode="w")
        formatter = logging.Formatter(format_full)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
