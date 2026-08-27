# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
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
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

"""gprMax.__main__: executed when gprMax directory is called as script."""

import gprMax.gprMax

if __name__ == "__main__":
    gprMax.gprMax.cli()

# Code profiling
# Time profiling
# import cProfile, pstats
# cProfile.run('gprMax.gprMax.main()','stats')
# p = pstats.Stats('stats')
# p.sort_stats('time').print_stats(25)

# Memory profiling - use in gprMax.py
# from memory profiler import profile
# add @profile before function to profile
