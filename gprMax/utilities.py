# Copyright (C) 2015-2017: The University of Edinburgh
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

from contextlib import contextmanager
import decimal as d
import platform
import psutil
import re
import subprocess
from shutil import get_terminal_size
import sys
import textwrap

from colorama import init, Fore, Style
init()
import numpy as np

from gprMax.constants import floattype


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
    """

    description = '\n=== Electromagnetic modelling software based on the Finite-Difference Time-Domain (FDTD) method'
    copyright = 'Copyright (C) 2015-2017: The University of Edinburgh'
    authors = 'Authors: Craig Warren and Antonis Giannopoulos'
    licenseinfo1 = 'gprMax is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.\n'
    licenseinfo2 = 'gprMax is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.'
    licenseinfo3 = 'You should have received a copy of the GNU General Public License along with gprMax.  If not, see www.gnu.org/licenses.'

    logo = """    www.gprmax.com   __  __
     __ _ _ __  _ __|  \/  | __ ___  __
    / _` | '_ \| '__| |\/| |/ _` \ \/ /
   | (_| | |_) | |  | |  | | (_| |>  <
    \__, | .__/|_|  |_|  |_|\__,_/_/\_\\
    |___/|_|
                       v""" + version

    print('{} {}\n'.format(description, '=' * (get_terminal_width() - len(description) - 1)))
    print(Fore.CYAN + '{}\n'.format(logo))
    print(Style.RESET_ALL + textwrap.fill(copyright, width=get_terminal_width() - 1, initial_indent=' '))
    print(textwrap.fill(authors, width=get_terminal_width() - 1, initial_indent=' '))
    print()
    print(textwrap.fill(licenseinfo1, width=get_terminal_width() - 1, initial_indent=' ', subsequent_indent='  '))
    print(textwrap.fill(licenseinfo2, width=get_terminal_width() - 1, initial_indent=' ', subsequent_indent='  '))
    print(textwrap.fill(licenseinfo3, width=get_terminal_width() - 1, initial_indent=' ', subsequent_indent='  '))


@contextmanager
def open_path_file(path_or_file):
    """Accepts either a path as a string or a file object and returns a file object (http://stackoverflow.com/a/6783680).

    Args:
        path_or_file: path as a string or a file object.

    Returns:
        f (object): File object.
    """

    if isinstance(path_or_file, str):
        f = file_to_close = open(path_or_file, 'r')
    else:
        f = path_or_file
        file_to_close = None

    try:
        yield f
    finally:
        if file_to_close:
            file_to_close.close()


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


def get_host_info():
    """Get information about the machine, CPU, RAM, and OS.

    Returns:
        hostinfo (dict): Manufacturer and model of machine; description of CPU type, speed, cores; RAM; name and version of operating system.
    """

    # Windows
    if sys.platform == 'win32':
        manufacturer = subprocess.check_output("wmic csproduct get vendor", shell=True).decode('utf-8').strip()
        manufacturer = manufacturer.split('\n')[1]
        model = subprocess.check_output("wmic computersystem get model", shell=True).decode('utf-8').strip()
        model = model.split('\n')[1]
        machineID = manufacturer + ' ' + model
        cpuID = subprocess.check_output("wmic cpu get Name", shell=True).decode('utf-8').strip()
        cpuID = cpuID.split('\n')[1]
        if platform.machine().endswith('64'):
            osbit = '(64-bit)'
        else:
            osbit = '(32-bit)'
        osversion = 'Windows ' + platform.release() + osbit

    # Mac OS X/macOS
    elif sys.platform == 'darwin':
        manufacturer = 'Apple'
        model = subprocess.check_output("sysctl -n hw.model", shell=True).decode('utf-8').strip()
        machineID = manufacturer + ' ' + model
        cpuID = subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True).decode('utf-8').strip()
        cpuID = ' '.join(cpuID.split())
        if int(platform.mac_ver()[0].split('.')[1]) < 12:
            osversion = 'Mac OS X (' + platform.mac_ver()[0] + ')'
        else:
            osversion = 'macOS (' + platform.mac_ver()[0] + ')'

    # Linux
    elif sys.platform == 'linux':
        manufacturer = subprocess.check_output("cat /sys/class/dmi/id/sys_vendor", shell=True).decode('utf-8').strip()
        model = subprocess.check_output("cat /sys/class/dmi/id/product_name", shell=True).decode('utf-8').strip()
        machineID = manufacturer + ' ' + model
        allcpuinfo = subprocess.check_output("cat /proc/cpuinfo", shell=True).decode('utf-8').strip()
        for line in allcpuinfo.split('\n'):
            if 'model name' in line:
                cpuID = re.sub('.*model name.*:', '', line, 1)
        osversion = 'Linux (' + platform.release() + ')'

    machineID = machineID.strip()
    cpuID = cpuID.strip()
    # Get number of physical CPU cores, i.e. avoid hyperthreading with OpenMP
    cpucores = psutil.cpu_count(logical=False)
    ram = psutil.virtual_memory().total

    hostinfo = {'machineID': machineID, 'cpuID': cpuID, 'cpucores': cpucores, 'ram': ram, 'osversion': osversion}

    return hostinfo


def memory_usage(G):
    """Estimate the amount of memory (RAM) required to run a model.

    Args:
        G (class): Grid class instance - holds essential parameters describing the model.

    Returns:
        memestimate (int): Estimate of required memory in bytes
    """

    stdoverhead = 50e6

    # 6 x field arrays + 6 x ID arrays
    fieldarrays = (6 + 6) * (G.nx + 1) * (G.ny + 1) * (G.nz + 1) * np.dtype(floattype).itemsize

    solidarray = G.nx * G.ny * G.nz * np.dtype(np.uint32).itemsize

    # 12 x rigidE array components + 6 x rigidH array components
    rigidarrays = (12 + 6) * G.nx * G.ny * G.nz * np.dtype(np.int8).itemsize

    pmlarrays = 0
    for (k, v) in G.pmlthickness.items():
        if v > 0:
            if 'x' in k:
                pmlarrays += ((v + 1) * G.ny * (G.nz + 1))
                pmlarrays += ((v + 1) * (G.ny + 1) * G.nz)
                pmlarrays += (v * G.ny * (G.nz + 1))
                pmlarrays += (v * (G.ny + 1) * G.nz)
            elif 'y' in k:
                pmlarrays += (G.nx * (v + 1) * (G.nz + 1))
                pmlarrays += ((G.nx + 1) * (v + 1) * G.nz)
                pmlarrays += ((G.nx + 1) * v * G.nz)
                pmlarrays += (G.nx * v * (G.nz + 1))
            elif 'z' in k:
                pmlarrays += (G.nx * (G.ny + 1) * (v + 1))
                pmlarrays += ((G.nx + 1) * G.ny * (v + 1))
                pmlarrays += ((G.nx + 1) * G.ny * v)
                pmlarrays += (G.nx * (G.ny + 1) * v)

    memestimate = int(stdoverhead + fieldarrays + solidarray + rigidarrays + pmlarrays)

    return memestimate
