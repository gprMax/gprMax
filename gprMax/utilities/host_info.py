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

import logging
import os
import platform
import re
import subprocess
import sys

import gprMax.config as config
import psutil

from .utilities import human_size

logger = logging.getLogger(__name__)


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
        if 'Apple' in config.sim_config.hostinfo['cpuID']:
            # https://developer.apple.com/documentation/apple-silicon/tuning-your-code-s-performance-for-apple-silicon
            os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
        else:
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

    def __init__(self):

        self.deviceID = None
        self.name = None
        self.pcibusID = None
        self.constmem = None
        self.totalmem = None

    def get_cuda_gpu_info(self, drv, deviceID):
        """Set information about GPU.

        Args:
            drv (object): pycuda driver.
            deviceID (int): Device ID for GPU.
        """

        self.deviceID = deviceID
        self.name = drv.Device(self.deviceID).name()
        self.pcibusID = drv.Device(self.deviceID).pci_bus_id()
        self.constmem = drv.Device(self.deviceID).total_constant_memory
        self.totalmem = drv.Device(self.deviceID).total_memory()


def detect_cuda_gpus():
    """Get information about Nvidia GPU(s).

    Returns:
        gpus (list): Detected GPU(s) object(s).
    """

    try:
        import pycuda.driver as drv
        has_pycuda = True
    except ImportError:
        logger.warning('pycuda not detected - to use gprMax in GPU mode the pycuda package must be installed, and you must have a NVIDIA CUDA-Enabled GPU (https://developer.nvidia.com/cuda-gpus).')
        has_pycuda = False
    
    if has_pycuda:
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
            gpu = GPU()
            gpu.get_cuda_gpu_info(drv, ID)
            gpus.append(gpu)

    else:
        gpus = None

    return gpus


def detect_opencl():
    """Get information about OpenCL platforms and devices.

    Returns:
        gpus (list): Detected GPU(s) object(s).
    """

    try:
        import pyopencl as cl
        has_pyopencl = True
    except ImportError:
        logger.warning('pyopencl not detected - to use gprMax with OpenCL, the pyopencl package must be installed, and you must have at least one OpenCL capable platform.')
        has_pyopencl = False

    if has_pyopencl:
        platforms = cl.get_platforms()
        platform_names = [p.name for p in platforms]
        logger.info(platform_names)
