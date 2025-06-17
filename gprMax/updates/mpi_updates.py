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

from gprMax.grid.mpi_grid import MPIGrid
from gprMax.updates.cpu_updates import CPUUpdates


class MPIUpdates(CPUUpdates[MPIGrid]):
    """Defines update functions for MPI CPU-based solver."""

    def halo_swap_electric(self):
        self.grid.halo_swap_electric()

    def halo_swap_magnetic(self):
        self.grid.halo_swap_magnetic()
