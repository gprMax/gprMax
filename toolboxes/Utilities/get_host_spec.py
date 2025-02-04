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

import logging

import humanize

from gprMax.utilities.host_info import (
    detect_cuda_gpus,
    detect_opencl,
    get_host_info,
    print_cuda_info,
    print_opencl_info,
)
from gprMax.utilities.utilities import get_terminal_width

logging.basicConfig(format="%(message)s", level=logging.INFO)

# Host machine info.
hostinfo = get_host_info()
hyperthreadingstr = f", {hostinfo['logicalcores']} cores with Hyper-Threading" if hostinfo["hyperthreading"] else ""
hostname = f"\n=== {hostinfo['hostname']}"
logging.info(f"{hostname} {'=' * (get_terminal_width() - len(hostname) - 1)}")
logging.info(f"\n{'Mfr/model:':<12} {hostinfo['machineID']}")
logging.info(
    f"{'CPU:':<12} {hostinfo['sockets']} x {hostinfo['cpuID']} "
    + f"({hostinfo['physicalcores']} cores{hyperthreadingstr})"
)
logging.info(f"{'RAM:':<12} {humanize.naturalsize(hostinfo['ram'], True)}")
logging.info(f"{'OS/Version:':<12} {hostinfo['osversion']}")

# OpenMP
logging.info(
    "\n\n=== OpenMP capabilities (gprMax will not use Hyper-Threading " + "as there is no performance advantage)\n"
)
logging.info(f"{'OpenMP threads: '} {hostinfo['physicalcores']}")

# CUDA
logging.info("\n\n=== CUDA capabilities\n")
gpus = detect_cuda_gpus()
if gpus:
    print_cuda_info(gpus)
else:
    logging.info("Nothing detected.")

# OpenCL
logging.info("\n\n=== OpenCL capabilities\n")
devs = detect_opencl()
if devs:
    print_opencl_info(devs)
else:
    logging.info("Nothing detected.")

logging.info(f"\n{'=' * (get_terminal_width() - 1)}\n")
