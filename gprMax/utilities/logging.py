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
import logging
import sys
from copy import copy

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
    'DEBUG'   : 37, # white
    'BASIC'   : 37, # white
    'INFO'    : 37, # white
    'WARNING' : 33, # yellow
    'ERROR'   : 31, # red
    'CRITICAL': 41, # white on red bg
}
PREFIX = '\033['
SUFFIX = '\033[0m'
    

class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors 
        (https://stackoverflow.com/a/46482050)."""

    def __init__(self, pattern):
        logging.Formatter.__init__(self, pattern)

    def format(self, record):
        colored_record = copy(record)
        levelname = colored_record.levelname
        seq = MAPPING.get(levelname, 37) # default white
        colored_levelname = ('{0}{1}m{2}{3}').format(PREFIX, seq, levelname, SUFFIX)
        colored_record.levelname = colored_levelname
        return logging.Formatter.format(self, colored_record)


def logging_config(name='gprMax', level=logging.INFO, format_style='std', log_file=False):
    """Setup and configure logging.

    Args:
        name (str): name of logger to create.
        level (logging level): set logging level to stdout.
        format_style (str): set formatting - 'std' or 'full'
        log_file (bool): additional logging to file.
    """

    # Set format style
    if format_style == 'std':
        format = "%(message)s"
    elif format_style == 'full':
        format = "%(asctime)s:%(levelname)s:%(name)s:%(lineno)d: %(message)s"

    # Create main top-level logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Config for logging to console
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(CustomFormatter(format))
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(handler)

    # Config for logging to file if required
    if log_file:
        filename = name + '-log-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.txt'
        handler = logging.FileHandler(filename, mode='w')
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s (%(filename)s:%(lineno)d)')
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)