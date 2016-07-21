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

import sys
import decimal as d


def logo(version):
    """Print gprMax logo, version, and licencing/copyright information.
        
    Args:
        version (str): Version number.
    """

    licenseinfo = """
Copyright (C) 2015-2016: The University of Edinburgh
                Authors: Craig Warren and Antonis Giannopoulos
        
gprMax is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
                    
gprMax is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
                                    
You should have received a copy of the GNU General Public License
along with gprMax.  If not, see <http://www.gnu.org/licenses/>."""

    gprMaxlogo = """
                                 __  __
                 __ _ _ __  _ __|  \/  | __ ___  __
                / _` | '_ \| '__| |\/| |/ _` \ \/ /
               | (_| | |_) | |  | |  | | (_| |>  <
                \__, | .__/|_|  |_|  |_|\__,_/_/\_\\
                |___/|_|
        """

    width = 65
    url = 'www.gprmax.com'

    print('\nElectromagnetic modelling software based on the Finite-Difference \nTime-Domain (FDTD) method')
    print('\n{} {} {}'.format('*' * round((width - len(url)) / 2), url, '*' * round((width - len(url)) / 2)))
    print('{}'.format(gprMaxlogo))
    print('{} v{} {}'.format('*' * round((width - len(version)) / 2), (version), '*' * round((width - len(version)) / 2)))
    print(licenseinfo)


def update_progress(progress):
    """Displays or updates a console progress bar.
        
    Args:
        progress (float): Number between zero and one to signify progress.
    """

    # Modify this to change the length of the progress bar
    barLength = 50
    block = round_value(barLength * progress)
    text = '\r|{}| {:2.1f}%'.format('#' * block + '-' * (barLength - block), progress * 100)
    sys.stdout.write(text)
    sys.stdout.flush()


def round_value(value, decimalplaces=0):
    """Rounding function.
        
    Args:
        value (float): Number to round.
        decimalplaces (int): Number of decimal places of float to represent rounded value.
        
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


def human_size(size, a_kilobyte_is_1024_bytes=True):
    """Convert a file size to human-readable form.
        
    Args:
        size (int): file size in bytes
        a_kilobyte_is_1024_bytes (boolean) - true for multiples of 1024, false for multiples of 1000
        
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
            return '{0:.1f}{1}'.format(size, suffix)

    raise ValueError('Number is too large.')
