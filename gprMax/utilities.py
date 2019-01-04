# Copyright (C) 2015-2019: The University of Edinburgh
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
import codecs
import decimal as d
import platform
import psutil
import re
import subprocess
from shutil import get_terminal_size
import sys
import textwrap

from colorama import init
from colorama import Fore
from colorama import Style
init()
import numpy as np

from gprMax.constants import complextype
from gprMax.constants import floattype
from gprMax.exceptions import GeneralError
from gprMax.materials import Material


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
    copyright = 'Copyright (C) 2015-2019: The University of Edinburgh'
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
    """
    Accepts either a path as a string or a file object and returns a file
    object (http://stackoverflow.com/a/6783680).

    Args:
        path_or_file: path as a string or a file object.

    Returns:
        f (object): File object.
    """

    if isinstance(path_or_file, str):
        f = file_to_close = codecs.open(path_or_file, 'r', encoding='utf-8')
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


def get_host_info():
    """Get information about the machine, CPU, RAM, and OS.

    Returns:
        hostinfo (dict): Manufacturer and model of machine; description of CPU
                type, speed, cores; RAM; name and version of operating system.
    """

    # Default to 'unknown' if any of the detection fails
    manufacturer = model = cpuID = sockets = threadspercore = 'unknown'

    # Windows
    if sys.platform == 'win32':
        # Manufacturer/model
        try:
            manufacturer = subprocess.check_output("wmic csproduct get vendor", shell=True, stderr=subprocess.STDOUT).decode('utf-8').strip()
            manufacturer = manufacturer.split('\n')
            if len(manufacturer) > 1:
                manufacturer = manufacturer[1]
            else:
                manufacturer = manufacturer[0]
            model = subprocess.check_output("wmic computersystem get model", shell=True, stderr=subprocess.STDOUT).decode('utf-8').strip()
            model = model.split('\n')
            if len(model) > 1:
                model = model[1]
            else:
                model = model[0]
        except subprocess.CalledProcessError:
            pass
        machineID = manufacturer + ' ' + model

        # CPU information
        try:
            allcpuinfo = subprocess.check_output("wmic cpu get Name", shell=True, stderr=subprocess.STDOUT).decode('utf-8').strip()
            allcpuinfo = allcpuinfo.split('\n')
            sockets = 0
            for line in allcpuinfo:
                if 'CPU' in line:
                    cpuID = line.strip()
                    sockets += 1
        except subprocess.CalledProcessError:
            pass

        # Hyperthreading
        if psutil.cpu_count(logical=False) != psutil.cpu_count(logical=True):
            hyperthreading = True
        else:
            hyperthreading = False

        # OS version
        if platform.machine().endswith('64'):
            osbit = ' (64-bit)'
        else:
            osbit = ' (32-bit)'
        osversion = 'Windows ' + platform.release() + osbit

    # Mac OS X/macOS
    elif sys.platform == 'darwin':
        # Manufacturer/model
        manufacturer = 'Apple'
        try:
            model = subprocess.check_output("sysctl -n hw.model", shell=True, stderr=subprocess.STDOUT).decode('utf-8').strip()
        except subprocess.CalledProcessError:
            pass
        machineID = manufacturer + ' ' + model

        # CPU information
        try:
            sockets = subprocess.check_output("sysctl -n hw.packages", shell=True, stderr=subprocess.STDOUT).decode('utf-8').strip()
            sockets = int(sockets)
            cpuID = subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True, stderr=subprocess.STDOUT).decode('utf-8').strip()
            cpuID = ' '.join(cpuID.split())
        except subprocess.CalledProcessError:
            pass

        # Hyperthreading
        if psutil.cpu_count(logical=False) != psutil.cpu_count(logical=True):
            hyperthreading = True
        else:
            hyperthreading = False

        # OS version
        if int(platform.mac_ver()[0].split('.')[1]) < 12:
            osversion = 'Mac OS X (' + platform.mac_ver()[0] + ')'
        else:
            osversion = 'macOS (' + platform.mac_ver()[0] + ')'

    # Linux
    elif sys.platform == 'linux':
        # Manufacturer/model
        try:
            manufacturer = subprocess.check_output("cat /sys/class/dmi/id/sys_vendor", shell=True, stderr=subprocess.STDOUT).decode('utf-8').strip()
            model = subprocess.check_output("cat /sys/class/dmi/id/product_name", shell=True, stderr=subprocess.STDOUT).decode('utf-8').strip()
        except subprocess.CalledProcessError:
            pass
        machineID = manufacturer + ' ' + model

        # CPU information
        try:
            cpuIDinfo = subprocess.check_output("cat /proc/cpuinfo", shell=True, stderr=subprocess.STDOUT).decode('utf-8').strip()
            for line in cpuIDinfo.split('\n'):
                if re.search('model name', line):
                    cpuID = re.sub('.*model name.*:', '', line, 1).strip()
            allcpuinfo = subprocess.check_output("lscpu", shell=True, stderr=subprocess.STDOUT).decode('utf-8').strip()
            for line in allcpuinfo.split('\n'):
                if 'Socket(s)' in line:
                    sockets = int(line.strip()[-1])
                if 'Thread(s) per core' in line:
                    threadspercore = int(line.strip()[-1])
        except subprocess.CalledProcessError:
            pass

        # Hyperthreading
        if threadspercore == 2:
            hyperthreading = True
        else:
            hyperthreading = False

        # OS version
        osrelease = subprocess.check_output("cat /proc/sys/kernel/osrelease", shell=True).decode('utf-8').strip()
        osversion = 'Linux (' + osrelease + ', ' + platform.linux_distribution()[0] + ')'

    # Dictionary of host information
    hostinfo = {}
    hostinfo['hostname'] = platform.node()
    hostinfo['machineID'] = machineID.strip()
    hostinfo['sockets'] = sockets
    hostinfo['cpuID'] = cpuID
    hostinfo['osversion'] = osversion
    hostinfo['hyperthreading'] = hyperthreading
    hostinfo['logicalcores'] = psutil.cpu_count()
    try:
        # Get number of physical CPU cores, i.e. avoid hyperthreading with OpenMP
        hostinfo['physicalcores'] = psutil.cpu_count(logical=False)
    except ValueError:
        hostinfo['physicalcores'] = hostinfo['logicalcores']
    # Handle case where cpu_count returns None on some machines
    if not hostinfo['physicalcores']:
        hostinfo['physicalcores'] = hostinfo['logicalcores']
    hostinfo['ram'] = psutil.virtual_memory().total

    return hostinfo


class GPU(object):
    """GPU information."""

    def __init__(self, deviceID):
        """
        Args:
            deviceID (int): Device ID for GPU.
        """

        self.deviceID = deviceID
        self.name = None
        self.pcibusID = None
        self.constmem = None
        self.totalmem = None

    def get_gpu_info(self, drv):
        """Set information about GPU.

        Args:
            drv (object): PyCuda driver.
        """

        self.name = drv.Device(self.deviceID).name()
        self.pcibusID = drv.Device(self.deviceID).pci_bus_id()
        self.constmem = drv.Device(self.deviceID).total_constant_memory
        self.totalmem = drv.Device(self.deviceID).total_memory()


def detect_gpus():
    """Get information about Nvidia GPU(s).

    Returns:
        gpus (list): Detected GPU(s) object(s).
    """

    try:
        import pycuda.driver as drv
    except ImportError:
        raise ImportError('To use gprMax in GPU mode the pycuda package must be installed, and you must have a NVIDIA CUDA-Enabled GPU (https://developer.nvidia.com/cuda-gpus).')
    drv.init()

    # Check if there are any CUDA-Enabled GPUs
    if drv.Device.count() == 0:
        raise GeneralError('No NVIDIA CUDA-Enabled GPUs detected (https://developer.nvidia.com/cuda-gpus)')

    # Print information about all detected GPUs
    gpus = []
    gputext = []
    for i in range(drv.Device.count()):
        gpu = GPU(deviceID=i)
        gpu.get_gpu_info(drv)
        gpus.append(gpu)
        gputext.append('{} - {}, {}'.format(gpu.deviceID, gpu.name, human_size(gpu.totalmem, a_kilobyte_is_1024_bytes=True)))
    print('GPU(s) detected: {}'.format(' | '.join(gputext)))

    return gpus
