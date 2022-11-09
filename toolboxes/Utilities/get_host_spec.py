# Copyright (C) 2015-2022: The University of Edinburgh, United Kingdom
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

from gprMax.utilities.host_info import (detect_cuda_gpus, detect_opencl,
                                        get_host_info, print_cuda_info,
                                        print_opencl_info)
from gprMax.utilities.utilities import get_terminal_width, human_size

# Host machine info.
hostinfo = get_host_info()
hyperthreadingstr = f", {hostinfo['logicalcores']} cores with Hyper-Threading" if hostinfo['hyperthreading'] else ''
hostname = (f"\n=== {hostinfo['hostname']}")
print(f"{hostname} {'=' * (get_terminal_width() - len(hostname) - 1)}")
print(f"\n{'Mfr/model:':<12} {hostinfo['machineID']}")
print(f"{'CPU:':<12} {hostinfo['sockets']} x {hostinfo['cpuID']} ({hostinfo['physicalcores']} cores{hyperthreadingstr})")
print(f"{'RAM:':<12} {human_size(hostinfo['ram'], a_kilobyte_is_1024_bytes=True)}")
print(f"{'OS/Version:':<12} {hostinfo['osversion']}")

# OpenMP
print("\n\n=== OpenMP capabilities (gprMax will not use Hyper-Threading with OpenMP as there is no performance advantage)\n")
print(f"{'OpenMP threads: '} {hostinfo['physicalcores']}")

# CUDA
print("\n\n=== CUDA capabilities\n")
gpus = detect_cuda_gpus()
if gpus:
    print_cuda_info(gpus)

# OpenCL
print("\n\n=== OpenCL capabilities\n")
devs = detect_opencl()
if devs:
    print_opencl_info(devs)

print(f"\n{'=' * (get_terminal_width() - 1)}\n")
