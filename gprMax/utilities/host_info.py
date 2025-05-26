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

import logging
import os
import platform
import re
import subprocess
import sys

import humanize
import psutil

from .. import config

logger = logging.getLogger(__name__)


def get_host_info():
    """Gets information about the machine, CPU, RAM, and OS.

    Returns:
        hostinfo: dict containing manufacturer and model of machine;
                    description of CPU type, speed, cores; RAM; name and
                    version of operating system.
    """

    # Default to 'unknown' if any of the detection fails
    manufacturer = model = cpuID = sockets = threadspercore = "unknown"

    # Windows
    if sys.platform == "win32":
        # Manufacturer/model
        try:
            manufacturer = (
                subprocess.check_output(
                    ["wmic", "csproduct", "get", "vendor"], shell=False, stderr=subprocess.STDOUT
                )
                .decode("utf-8")
                .strip()
            )
            manufacturer = manufacturer.split("\n")
            if len(manufacturer) > 1:
                manufacturer = manufacturer[1]
            else:
                manufacturer = manufacturer[0]
            model = (
                subprocess.check_output(
                    ["wmic", "computersystem", "get", "model"],
                    shell=False,
                    stderr=subprocess.STDOUT,
                )
                .decode("utf-8")
                .strip()
            )
            model = model.split("\n")
            if len(model) > 1:
                model = model[1]
            else:
                model = model[0]
        except subprocess.CalledProcessError:
            pass
        machineID = " ".join(manufacturer.split()) + " " + " ".join(model.split())

        # CPU information
        try:
            allcpuinfo = (
                subprocess.check_output(
                    ["wmic", "cpu", "get", "Name"], shell=False, stderr=subprocess.STDOUT
                )
                .decode("utf-8")
                .strip()
            )
            allcpuinfo = allcpuinfo.split("\n")
            sockets = 0
            for line in allcpuinfo:
                if "CPU" in line:
                    cpuID = line.strip()
                    cpuID = " ".join(cpuID.split())
                    sockets += 1
        except subprocess.CalledProcessError:
            pass

        # Hyperthreading
        if psutil.cpu_count(logical=False) != psutil.cpu_count(logical=True):
            hyperthreading = True
        else:
            hyperthreading = False

        # OS version
        if platform.machine().endswith("64"):
            osbit = " (64-bit)"
        else:
            osbit = " (32-bit)"
        osversion = "Windows " + platform.release() + osbit

    # Mac OS X/macOS
    elif sys.platform == "darwin":
        # Manufacturer/model
        manufacturer = "Apple"
        try:
            model = (
                subprocess.check_output(
                    ["sysctl", "-n", "hw.model"], shell=False, stderr=subprocess.STDOUT
                )
                .decode("utf-8")
                .strip()
            )
        except subprocess.CalledProcessError:
            pass
        machineID = " ".join(manufacturer.split()) + " " + " ".join(model.split())

        # CPU information
        try:
            sockets = (
                subprocess.check_output(
                    ["sysctl", "-n", "hw.packages"], shell=False, stderr=subprocess.STDOUT
                )
                .decode("utf-8")
                .strip()
            )
            sockets = int(sockets)
            cpuID = (
                subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    shell=False,
                    stderr=subprocess.STDOUT,
                )
                .decode("utf-8")
                .strip()
            )
            cpuID = " ".join(cpuID.split())
        except subprocess.CalledProcessError:
            pass

        # Hyperthreading
        if psutil.cpu_count(logical=False) != psutil.cpu_count(logical=True):
            hyperthreading = True
        else:
            hyperthreading = False

        # OS version
        osversion = "macOS (" + platform.mac_ver()[0] + ")"

    # Linux
    elif sys.platform == "linux":
        # Manufacturer/model
        try:
            manufacturer = (
                subprocess.check_output(
                    ["cat", "/sys/class/dmi/id/sys_vendor"], shell=False, stderr=subprocess.STDOUT
                )
                .decode("utf-8")
                .strip()
            )
            model = (
                subprocess.check_output(
                    ["cat", "/sys/class/dmi/id/product_name"],
                    shell=False,
                    stderr=subprocess.STDOUT,
                )
                .decode("utf-8")
                .strip()
            )
        except subprocess.CalledProcessError:
            pass
        machineID = " ".join(manufacturer.split()) + " " + " ".join(model.split())

        # CPU information
        try:
            # Locale to ensure English
            myenv = {**os.environ, "LANG": "en_US.utf8"}
            cpuIDinfo = (
                subprocess.check_output(
                    ["cat", "/proc/cpuinfo"], shell=False, stderr=subprocess.STDOUT, env=myenv
                )
                .decode("utf-8")
                .strip()
            )
            for line in cpuIDinfo.split("\n"):
                if re.search("model name", line):
                    cpuID = re.sub(".*model name.*:", "", line, 1).strip()
                    cpuID = " ".join(cpuID.split())
            allcpuinfo = (
                subprocess.check_output(["lscpu"], shell=False, stderr=subprocess.STDOUT, env=myenv)
                .decode("utf-8")
                .strip()
            )
            for line in allcpuinfo.split("\n"):
                if "Socket(s)" in line:
                    sockets = int(line.strip()[-1])
                if "Thread(s) per core" in line:
                    threadspercore = int(line.strip()[-1])
        except subprocess.CalledProcessError:
            pass

        # Hyperthreading
        hyperthreading = True if threadspercore == 2 else False

        # OS version
        osversion = platform.platform()

    # Dictionary of host information
    hostinfo = {}
    hostinfo["hostname"] = platform.node()
    hostinfo["machineID"] = machineID.strip()
    hostinfo["sockets"] = sockets
    hostinfo["cpuID"] = cpuID
    hostinfo["osversion"] = osversion
    hostinfo["hyperthreading"] = hyperthreading
    hostinfo["logicalcores"] = psutil.cpu_count()

    try:
        # Get number of physical CPU cores, i.e. avoid hyperthreading with OpenMP
        hostinfo["physicalcores"] = psutil.cpu_count(logical=False)
    except ValueError:
        hostinfo["physicalcores"] = hostinfo["logicalcores"]

    # Handle case where cpu_count returns None on some machines
    if not hostinfo["physicalcores"]:
        hostinfo["physicalcores"] = hostinfo["logicalcores"]

    hostinfo["ram"] = psutil.virtual_memory().total

    return hostinfo


def print_host_info(hostinfo):
    """Prints information about the machine, CPU, RAM, and OS.

    Args:
        hostinfo: dict containing manufacturer and model of machine;
                    description of CPU type, speed, cores; RAM; name and
                    version of operating system.
    """

    hyperthreadingstr = (
        f", {config.sim_config.hostinfo['logicalcores']} cores with Hyper-Threading"
        if config.sim_config.hostinfo["hyperthreading"]
        else ""
    )
    logger.basic(
        f"{config.sim_config.hostinfo['hostname']} | "
        f"{config.sim_config.hostinfo['machineID']} | "
        f"{hostinfo['sockets']} x {hostinfo['cpuID']} "
        f"({hostinfo['physicalcores']} cores{hyperthreadingstr}) | "
        f"{humanize.naturalsize(hostinfo['ram'], True)} | "
        f"{hostinfo['osversion']}"
    )
    logger.basic(f"|--->OpenMP: {hostinfo['physicalcores']} threads")


def set_omp_threads(nthreads=None):
    """Sets the number of OpenMP CPU threads for parallelised parts of code.

    Returns:
        nthreads: int for number of OpenMP threads.
    """

    if sys.platform == "darwin":
        # Should waiting threads consume CPU power (can drastically effect
        # performance)
        if "Apple" in config.sim_config.hostinfo["cpuID"]:
            # https://developer.apple.com/documentation/apple-silicon/tuning-your-code-s-performance-for-apple-silicon
            os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
        else:
            os.environ["OMP_WAIT_POLICY"] = "ACTIVE"

    # Number of threads may be adjusted by the run time environment to best
    # utilize system resources
    os.environ["OMP_DYNAMIC"] = "FALSE"

    # Each place corresponds to a single core (having one or more hardware threads)
    os.environ["OMP_PLACES"] = "cores"

    # Bind threads to physical cores
    os.environ["OMP_PROC_BIND"] = "TRUE"

    # Prints OMP version and environment variables (useful for debug)
    # os.environ['OMP_DISPLAY_ENV'] = 'TRUE'

    # Catch bug with Windows Subsystem for Linux (https://github.com/Microsoft/BashOnWindows/issues/785)
    if "Microsoft" in config.sim_config.hostinfo["osversion"]:
        os.environ["KMP_AFFINITY"] = "disabled"
        del os.environ["OMP_PLACES"]
        del os.environ["OMP_PROC_BIND"]

    if nthreads:
        os.environ["OMP_NUM_THREADS"] = str(nthreads)
    elif os.environ.get("OMP_NUM_THREADS"):
        nthreads = int(os.environ.get("OMP_NUM_THREADS"))
    else:
        # Set number of threads to number of physical CPU cores
        nthreads = config.sim_config.hostinfo["physicalcores"]
        os.environ["OMP_NUM_THREADS"] = str(nthreads)

    return nthreads


def mem_check_host(mem):
    """Checks if the required amount of memory (RAM) is available on host.

    Args:
        mem: int for memory required (bytes).
    """
    if mem > config.sim_config.hostinfo["ram"]:
        logger.warning(
            f"Memory (RAM) required (~{humanize.naturalsize(mem)}) exceeds "
            f"({humanize.naturalsize(config.sim_config.hostinfo['ram'], True)}) "
            " physical memory detected!\n"
        )


def mem_check_device_snaps(total_mem, snaps_mem):
    """Checks if the required amount of memory (RAM) for all snapshots can fit
        on specified device.

    Args:
        total_mem: int for total memory required for model (bytes).
        snaps_mem: int for memory required for all snapshots (bytes).
    """

    if config.sim_config.general["solver"] == "cuda":
        device_mem = config.get_model_config().device["dev"].total_memory()
    elif config.sim_config.general["solver"] == "opencl":
        device_mem = config.get_model_config().device["dev"].global_mem_size

    if total_mem - snaps_mem > device_mem:
        logger.warning(
            f"Memory (RAM) required (~{humanize.naturalsize(total_mem)}) exceeds "
            f"({humanize.naturalsize(device_mem, True)}) physical memory detected "
            f"on specified {' '.join(config.get_model_config().device['dev'].name.split())} device!\n"
        )

    # If the required memory without the snapshots will fit on the GPU then
    # transfer and store snaphots on host
    if snaps_mem != 0 and total_mem - snaps_mem < device_mem:
        config.get_model_config().device["snapsgpu2cpu"] = True


def mem_check_run_all(grids):
    """Checks memory required to run model for all grids, including for any
        dispersive materials, snapshots, and if solver with GPU, whether
        snapshots will fit on GPU memory.

    Args:
        grids: list of FDTDGrid objects.

    Returns:
        total_mem: int for total memory required for all grids.
        mem_str: list of strings containing text of memory requirements for
                    each grid.
    """

    total_mem_snaps = 0
    mem_strs = []

    for grid in grids:
        # Keep track of total memory for each model in
        # config.get_model_config().mem_use, which can contain multiple grids,
        # and also total memory per grid in grid.mem_use

        # Memory required for main grid arrays
        config.get_model_config().mem_use += grid.mem_est_basic()
        grid.mem_use += grid.mem_est_basic()

        # Additional memory required if there are any dispersive materials.
        if config.get_model_config().materials["maxpoles"] != 0:
            config.get_model_config().mem_use += grid.mem_est_dispersive()
            grid.mem_use += grid.mem_est_dispersive()

        # Additional memory required if there are any snapshots
        if grid.snapshots:
            for snap in grid.snapshots:
                config.get_model_config().mem_use += snap.nbytes
                grid.mem_use += snap.nbytes
                total_mem_snaps += snap.nbytes

        mem_strs.append(f"~{humanize.naturalsize(grid.mem_use)} [{grid.name}]")

    total_mem_model = config.get_model_config().mem_use

    # Check if there is sufficient memory on host
    mem_check_host(total_mem_model)

    # Check if there is sufficient memory for any snapshots on GPU
    if (
        total_mem_snaps > 0
        and config.sim_config.general["solver"] == "cuda"
        or config.sim_config.general["solver"] == "opencl"
    ):
        mem_check_device_snaps(total_mem_model, total_mem_snaps)

    return total_mem_model, mem_strs


def mem_check_build_all(grids):
    """Checks memory required to build all grids - primarily memory required to
        initialise grid arrays and to build any FractalVolume or FractalSurface
        objects which can require significant amounts of memory.

    Args:
        grids: list of FDTDGrid objects.

    Returns:
        total_mem: int for total memory required for all grids.
        mem_str: list of strings containing text of memory requirements for
                    each grid.
    """

    total_mem_model = config.get_model_config().mem_use
    mem_strs = []

    for grid in grids:
        grid_mem = 0

        # Memory required for main grid arrays
        grid_mem += grid.mem_est_basic()

        # Memory required for any FractalVolumes/FractalSurfaces
        if grid.fractalvolumes:
            grid_mem += grid.mem_est_fractals()

        mem_strs.append(f"~{humanize.naturalsize(grid_mem)} [{grid.name}]")
        total_mem_model += grid_mem

    # Check if there is sufficient memory on host
    mem_check_host(total_mem_model)

    return total_mem_model, mem_strs


def has_pycuda():
    """Checks if pycuda module is installed."""
    pycuda = True
    try:
        import pycuda
    except ImportError:
        pycuda = False
    return pycuda


def has_pyopencl():
    """Checks if pyopencl module is installed."""
    pyopencl = True
    try:
        import pyopencl
    except ImportError:
        pyopencl = False
    return pyopencl

def has_hip():
    """Checks if HIP module is installed."""
    hip = True
    try:
        import hip
    except ImportError:
        hip = False
    return hip

def detect_hip_gpus():
    """Gets information about HIP-capable GPU(s).

    Returns:
        gpus: dict of detected hip device object(s) where where device ID(s)
                are keys.
    """

    gpus = {}

    hip_reqs = (
        "To use gprMax with HIP you must:"
        "\n 1) install hipify"
        "\n 2) install AMD ROCm (https://rocm.docs.amd.com/en/latest/Installation.html)"
        "\n 3) have an AMD HIP-Enabled GPU (https://rocm.docs.amd.com/en/latest/Installation.html#supported-hardware)"
    )

    if has_hip():
        # import hip

        # # Check and list any HIP-Enabled GPUs
        # deviceIDsavail = []
        # if hip.Device.count() == 0:
        #     logger.warning("No AMD HIP-Enabled GPUs detected!\n" + hip_reqs)
        # elif "HIP_VISIBLE_DEVICES" in os.environ:
        #     deviceIDsavail = os.environ.get("HIP_VISIBLE_DEVICES")
        #     deviceIDsavail = [int(s) for s in deviceIDsavail.split(",")]
        # else:
        #     deviceIDsavail = range(hip.Device.count())

        # # Gather information about detected GPUs
        # for ID in deviceIDsavail:
            gpus[0] = gpu(0, "AMD GPU", 65536)  # Placeholder for actual device info

    else:
        logger.warning("hip not detected!\n" + hip_reqs)

    return gpus

def detect_cuda_gpus():
    """Gets information about CUDA-capable GPU(s).

    Returns:
        gpus: dict of detected pycuda device object(s) where where device ID(s)
                are keys.
    """

    gpus = {}

    cuda_reqs = (
        "To use gprMax with CUDA you must:"
        "\n 1) install pycuda"
        "\n 2) install NVIDIA CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit)"
        "\n 3) have an NVIDIA CUDA-Enabled GPU (https://developer.nvidia.com/cuda-gpus)"
    )

    if has_pycuda():
        import pycuda.driver as drv

        drv.init()

        # Check and list any CUDA-Enabled GPUs
        deviceIDsavail = []
        if drv.Device.count() == 0:
            logger.warning("No NVIDIA CUDA-Enabled GPUs detected!\n" + cuda_reqs)
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            deviceIDsavail = os.environ.get("CUDA_VISIBLE_DEVICES")
            deviceIDsavail = [int(s) for s in deviceIDsavail.split(",")]
        else:
            deviceIDsavail = range(drv.Device.count())

        # Gather information about detected GPUs
        for ID in deviceIDsavail:
            gpus[ID] = drv.Device(ID)

    else:
        logger.warning("pycuda not detected!\n" + cuda_reqs)

    return gpus


def print_cuda_info(devs):
    """Prints info about detected CUDA-capable GPU(s).

    Args:
        devs: dict of detected pycuda device object(s) where where device ID(s)
                are keys.
    """

    import pycuda

    logger.basic("|--->CUDA:")
    logger.debug(f"PyCUDA: {pycuda.VERSION_TEXT}")

    for ID, gpu in devs.items():
        logger.basic(
            f"     |--->Device {ID}: {' '.join(gpu.name().split())} | "
            f"{humanize.naturalsize(gpu.total_memory(), True)}"
        )


def detect_opencl():
    """Gets information about OpenCL platforms and devices.

    Returns:
        devs: dict of detected pyopencl device object(s) where where device ID(s)
                are keys.
    """

    devs = {}

    ocl_reqs = (
        "To use gprMax with OpenCL you must:"
        "\n 1) install pyopencl"
        "\n 2) install appropriate OpenCL device driver(s)"
        "\n 3) have at least one OpenCL-capable platform."
    )

    if has_pyopencl():
        import pyopencl as cl

        try:
            i = 0
            for platform in cl.get_platforms():
                for device in platform.get_devices():
                    devs[i] = device
                    i += 1
        except:
            logger.warning("No OpenCL-capable platforms detected!\n" + ocl_reqs)

    else:
        logger.warning("pyopencl not detected!\n" + ocl_reqs)

    return devs


def print_opencl_info(devs):
    """Prints info about detected OpenCL-capable device(s).

    Args:
        devs: dict of detected pyopencl device object(s) where where device ID(s)
                are keys.
    """

    import pyopencl as cl

    logger.basic("|--->OpenCL:")
    logger.debug(f"PyOpenCL: {cl.VERSION_TEXT}")

    for i, (ID, dev) in enumerate(devs.items()):
        if i == 0:
            platform = dev.platform.name
            logger.basic(f"     |--->Platform: {platform}")
        if platform != dev.platform.name:
            logger.basic(f"     |--->Platform: {dev.platform.name}")
        types = cl.device_type.to_string(dev.type)
        if "CPU" in types:
            type = "CPU"
        if "GPU" in types:
            type = "GPU"
        logger.basic(
            f"          |--->Device {ID}: {type} | {' '.join(dev.name.split())} | "
            f"{humanize.naturalsize(dev.global_mem_size, True)}"
        )
def print_hip_info(devs):
    """Prints info about detected HIP-capable device(s).

    Args:
        devs: dict of detected hip device object(s) where where device ID(s)
                are keys.
    """

    import hip

    logger.basic("|--->HIP:")
    logger.debug(f"HIP: {hip.__version__}")

    for ID, dev in devs.items():
        logger.basic(
            f"     |--->Device {ID}: {' '.join(dev.name.split())} | "
            f"{humanize.naturalsize(dev.total_memory(), True)}"
        )

class gpu(object):
    """Class to hold information about a GPU."""

    def __init__(self, ID, name, total_memory):
        self.ID = ID
        self.name = name
        self.total_memory = total_memory

    def __str__(self):
        return f"GPU {self.ID}: {self.name} | {humanize.naturalsize(self.total_memory, True)}"