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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <http://www.gnu.org/licenses/>.

import logging

import humanize

from gprMax.utilities.host_info import (
    detect_cuda_gpus,
    detect_metal,
    detect_opencl,
    get_host_info,
    print_cuda_info,
    print_metal_info,
    print_opencl_info,
)
from gprMax.utilities.utilities import get_terminal_width

logger = logging.getLogger(__name__)


def main():
    """Print host, CPU, memory, and available accelerator information."""
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    hostinfo = get_host_info()
    hyperthreading = (
        f", {hostinfo['logicalcores']} logical cores with Hyper-Threading"
        if hostinfo["hyperthreading"]
        else ""
    )
    hostname = f"\n=== {hostinfo['hostname']}"
    logger.info("%s %s", hostname, "=" * max(0, get_terminal_width() - len(hostname) - 1))
    logger.info("\n%-12s %s", "Mfr/model:", hostinfo["machineID"])
    logger.info(
        "%-12s %s x %s (%s physical cores%s)",
        "CPU:",
        hostinfo["sockets"],
        hostinfo["cpuID"],
        hostinfo["physicalcores"],
        hyperthreading,
    )
    logger.info("%-12s %s", "RAM:", humanize.naturalsize(hostinfo["ram"], True))
    logger.info("%-12s %s", "OS/Version:", hostinfo["osversion"])

    logger.info("\n\n=== OpenMP capabilities\n")
    logger.info("OpenMP threads: %s", hostinfo["physicalcores"])

    logger.info("\n\n=== CUDA capabilities\n")
    devices = detect_cuda_gpus()
    print_cuda_info(devices) if devices else logger.info("Nothing detected.")

    logger.info("\n\n=== OpenCL capabilities\n")
    devices = detect_opencl()
    print_opencl_info(devices) if devices else logger.info("Nothing detected.")

    logger.info("\n\n=== Apple Metal capabilities\n")
    devices = detect_metal()
    print_metal_info(devices) if devices else logger.info("Nothing detected.")

    logger.info("\n%s\n", "=" * max(0, get_terminal_width() - 1))


if __name__ == "__main__":
    main()
