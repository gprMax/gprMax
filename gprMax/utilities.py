# Copyright (C) 2015-2021: The University of Edinburgh
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

import datetime
import decimal as d
import logging
import os
import platform
import re
import subprocess
import sys
import textwrap
import xml.dom.minidom
from shutil import get_terminal_size

import gprMax.config as config
import numpy as np
import psutil
from colorama import Fore, Style, init
init()

logger = logging.getLogger(__name__)

try:
    from time import thread_time as timer_fn
except ImportError:
    from time import perf_counter as timer_fn
    logger.debug('"thread_time" not currently available in macOS and bug'\
                   ' (https://bugs.python.org/issue36205) with "process_time", so use "perf_counter".')
    

class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors 
        (https://stackoverflow.com/a/56944256)."""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    format = "%(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def logging_config(name='gprMax', level=logging.INFO, log_file=False):
    """Setup and configure logging.

    Args:
        name (str): name of logger to create.
        level (logging level): set logging level to stdout.
        log_file (bool): additional logging to file.
    """

    # Adds a custom log level to the root logger 
    # from which new loggers are derived
    BASIC_NUM = 25
    logging.addLevelName(BASIC_NUM, "BASIC")
    def basic(self, message, *args, **kws):
        if self.isEnabledFor(BASIC_NUM):
            self._log(BASIC_NUM, message, args, **kws)
    logging.Logger.basic = basic

    # Create main top-level logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Config for logging to console
    handler = logging.StreamHandler(sys.stdout)
    formatter = CustomFormatter()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Config for logging to file if required
    if log_file:
        filename = name + '-log-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        handler = logging.FileHandler(filename, mode='w')
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s')
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


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


def get_host_info():
    """Get information about the machine, CPU, RAM, and OS.

    Returns:
        hostinfo (dict): Manufacturer and model of machine; description of CPU
                            type, speed, cores; RAM; name and
                            version of operating system.
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
            # Locale to ensure English
            myenv = {**os.environ, 'LANG': 'en_US.utf8'}
            cpuIDinfo = subprocess.check_output("cat /proc/cpuinfo", shell=True, stderr=subprocess.STDOUT, env=myenv).decode('utf-8').strip()
            for line in cpuIDinfo.split('\n'):
                if re.search('model name', line):
                    cpuID = re.sub('.*model name.*:', '', line, 1).strip()
            allcpuinfo = subprocess.check_output("lscpu", shell=True, stderr=subprocess.STDOUT, env=myenv).decode('utf-8').strip()
            for line in allcpuinfo.split('\n'):
                if 'Socket(s)' in line:
                    sockets = int(line.strip()[-1])
                if 'Thread(s) per core' in line:
                    threadspercore = int(line.strip()[-1])
        except subprocess.CalledProcessError:
            pass

        # Hyperthreading
        hyperthreading = True if threadspercore == 2 else False

        # OS version
        osversion = platform.platform()

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


def set_omp_threads(nthreads=None):
    """Sets the number of OpenMP CPU threads for parallelised parts of code.

    Returns:
        nthreads (int): Number of OpenMP threads.
    """

    if sys.platform == 'darwin':
        # Should waiting threads consume CPU power (can drastically effect
        # performance)
        os.environ['OMP_WAIT_POLICY'] = 'ACTIVE'

    # Number of threads may be adjusted by the run time environment to best
    # utilize system resources
    os.environ['OMP_DYNAMIC'] = 'FALSE'

    # Each place corresponds to a single core (having one or more hardware threads)
    os.environ['OMP_PLACES'] = 'cores'

    # Bind threads to physical cores
    os.environ['OMP_PROC_BIND'] = 'TRUE'

    # Prints OMP version and environment variables (useful for debug)
    # os.environ['OMP_DISPLAY_ENV'] = 'TRUE'

    # Catch bug with Windows Subsystem for Linux (https://github.com/Microsoft/BashOnWindows/issues/785)
    if 'Microsoft' in config.sim_config.hostinfo['osversion']:
        os.environ['KMP_AFFINITY'] = 'disabled'
        del os.environ['OMP_PLACES']
        del os.environ['OMP_PROC_BIND']

    if nthreads:
        os.environ['OMP_NUM_THREADS'] = str(nthreads)
    elif os.environ.get('OMP_NUM_THREADS'):
        nthreads = int(os.environ.get('OMP_NUM_THREADS'))
    else:
        # Set number of threads to number of physical CPU cores
        nthreads = config.sim_config.hostinfo['physicalcores']
        os.environ['OMP_NUM_THREADS'] = str(nthreads)

    return nthreads


def mem_check_host(mem):
    """Check if the required amount of memory (RAM) is available on host.

    Args:
        mem (int): Memory required (bytes).
    """
    if mem > config.sim_config.hostinfo['ram']:
        logger.exception(f"Memory (RAM) required ~{human_size(mem)} exceeds {human_size(config.sim_config.hostinfo['ram'], a_kilobyte_is_1024_bytes=True)} detected!\n")
        raise ValueError


def mem_check_gpu_snaps(total_mem, snaps_mem):
    """Check if the required amount of memory (RAM) for all snapshots can fit
        on specified GPU.

    Args:
        total_mem (int): Total memory required for model (bytes).
        snaps_mem (int): Memory required for all snapshots (bytes).
    """
    if total_mem - snaps_mem > config.get_model_config().cuda['gpu'].totalmem:
        logger.exception(f"Memory (RAM) required ~{human_size(total_mem)} exceeds {human_size(config.get_model_config().cuda['gpu'].totalmem, a_kilobyte_is_1024_bytes=True)} detected on specified {config.get_model_config().cuda['gpu'].deviceID} - {config.get_model_config().cuda['gpu'].name} GPU!\n")
        raise ValueError

    # If the required memory without the snapshots will fit on the GPU then
    # transfer and store snaphots on host
    if snaps_mem != 0 and total_mem - snaps_mem < config.get_model_config().cuda['gpu'].totalmem:
        config.get_model_config().cuda['snapsgpu2cpu'] = True


def mem_check_all(grids):
    """Check memory for all grids, including for any dispersive materials,
        snapshots, and if solver with GPU, whether snapshots will fit on GPU
        memory.

    Args:
        grids (list): FDTDGrid objects.

    Returns:
        total_mem (int): Total memory required for all grids.
        mem_strs (list): Strings containing text of memory requirements for
                            each grid.
    """

    total_snaps_mem = 0
    mem_strs = []

    for grid in grids:
        # Memory required for main grid arrays
        config.get_model_config().mem_use += grid.mem_est_basic()
        grid.mem_use += grid.mem_est_basic()

        # Additional memory required if there are any dispersive materials.
        if config.get_model_config().materials['maxpoles'] != 0:
            config.get_model_config().mem_use += grid.mem_est_dispersive()
            grid.mem_use += grid.mem_est_dispersive()

        # Additional memory required if there are any snapshots
        if grid.snapshots:
            for snap in grid.snapshots:
                # 2 x required to account for electric and magnetic fields
                snap_mem = int(2 * snap.datasizefield)
                config.get_model_config().mem_use += snap_mem
                total_snaps_mem += snap_mem
                grid.mem_use += snap_mem

        mem_strs.append(f'~{human_size(grid.mem_use)} [{grid.name}]')

    total_mem = config.get_model_config().mem_use

    # Check if there is sufficient memory on host
    mem_check_host(total_mem)

    # Check if there is sufficient memory for any snapshots on GPU
    if total_snaps_mem > 0 and config.sim_config.general['cuda']:
        mem_check_gpu_snaps(total_mem, total_snaps_mem)

    return total_mem, mem_strs


class GPU:
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
        logger.exception('To use gprMax in GPU mode the pycuda package must be installed, and you must have a NVIDIA CUDA-Enabled GPU (https://developer.nvidia.com/cuda-gpus).')
        raise
    drv.init()

    # Check and list any CUDA-Enabled GPUs
    if drv.Device.count() == 0:
        logger.exception('No NVIDIA CUDA-Enabled GPUs detected (https://developer.nvidia.com/cuda-gpus)')
        raise ValueError
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
        deviceIDsavail = os.environ.get('CUDA_VISIBLE_DEVICES')
        deviceIDsavail = [int(s) for s in deviceIDsavail.split(',')]
    else:
        deviceIDsavail = range(drv.Device.count())

    # Gather information about detected GPUs
    gpus = []
    for ID in deviceIDsavail:
        gpu = GPU(deviceID=ID)
        gpu.get_gpu_info(drv)
        gpus.append(gpu)

    return gpus


def timer():
    """Function to return time in fractional seconds."""
    return timer_fn()
