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

    def update_magnetic_sources(self, iteration):
        """Apply magnetic-field writers before exchanging magnetic halos.

        Transmission lines do not write magnetic fields: they sample an
        Ampere contour. Their update is deliberately deferred to
        ``update_magnetic_edge_devices`` after the halo exchange.
        """

        for source in self.grid.magneticdipoles:
            source.update_magnetic(
                iteration,
                self.grid.updatecoeffsH,
                self.grid.ID,
                self.grid.Hx,
                self.grid.Hy,
                self.grid.Hz,
                self.grid,
            )
        # Each rank owns only the frill H-edge terms that fall inside its
        # interior. The distributed update sums their Ampere-loop
        # contributions, advances one identical terminal state, and deposits
        # only to locally owned terms before the H halo exchange.
        for source in self.grid.magneticfrillsources:
            source.update_magnetic_mpi(
                iteration,
                self.grid.updatecoeffsH,
                self.grid.ID,
                self.grid.Hx,
                self.grid.Hy,
                self.grid.Hz,
                self.grid,
            )

    def update_magnetic_edge_devices(self, iteration):
        """Complete devices that require synchronised magnetic fields."""

        for source in self.grid.transmissionlines:
            source.update_magnetic(
                iteration,
                self.grid.updatecoeffsH,
                self.grid.ID,
                self.grid.Hx,
                self.grid.Hy,
                self.grid.Hz,
                self.grid,
            )
        for guide in getattr(self.grid, "virtual_waveguides", ()):
            guide.complete_magnetic_mpi()

    def finalise(self):
        """Complete the last halo exchange before MPI/Python teardown."""

        self.grid.complete_halo_swaps()
        super().finalise()

    def halo_swap_electric(self):
        self.grid.halo_swap_electric()

    def halo_swap_magnetic(self):
        self.grid.halo_swap_magnetic()
