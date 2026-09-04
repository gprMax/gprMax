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

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from gprMax.grid.fdtd_grid import FDTDGrid

GridType = TypeVar("GridType", bound=FDTDGrid)


class Updates(Generic[GridType], ABC):
    """Defines update functions for a solver."""

    is_distributed = False

    def __init__(self, G: GridType):
        """
        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        self.grid = G

    @abstractmethod
    def store_outputs(self, iteration: int) -> None:
        """Stores field component values for every receiver and transmission line."""
        pass

    @abstractmethod
    def store_snapshots(self, iteration: int) -> None:
        """Store snapshots at the current electric-field time level.

        Args:
            iteration: int for iteration number.
        """
        pass

    def observe_ntff_electric(self, iteration: int) -> None:
        """Observe electric fields for any KSIR monitors.

        Local solver implementations override this hook; unsupported backends
        retain the no-op.
        """

        pass

    def observe_ntff_magnetic(self, iteration: int) -> None:
        """Observe magnetic fields for any KSIR monitors."""

        pass

    def observe_sar_electric(self, iteration: int) -> None:
        """Observe electric fields for tagged-cell SAR monitors."""

        pass

    @abstractmethod
    def update_magnetic(self) -> None:
        """Updates magnetic field components."""
        pass

    @abstractmethod
    def update_magnetic_pml(self) -> None:
        """Updates magnetic field components with the PML correction."""
        pass

    @abstractmethod
    def update_magnetic_sources(self, iteration: int) -> None:
        """Updates magnetic field components from sources."""
        pass

    def update_plane_waves_magnetic(self, iteration: int) -> None:
        """Advance auxiliary plane waves and apply magnetic TFSF corrections."""

        pass

    @abstractmethod
    def update_electric_a(self) -> None:
        """Updates electric field components."""
        pass

    def update_symmetry_boundaries_electric(self) -> None:
        """Apply any PMC ghost-image electric boundary correction."""

        pass

    @abstractmethod
    def update_electric_pml(self) -> None:
        """Updates electric field components with the PML correction."""
        pass

    @abstractmethod
    def update_electric_sources(self, iteration: int) -> None:
        """Updates electric field components from sources -
        update any Hertzian dipole sources last.
        """
        pass

    def update_plane_waves_electric(self, iteration: int) -> None:
        """Advance auxiliary plane waves and apply electric TFSF corrections."""

        pass

    def update_eigenmode_sources_magnetic(self, iteration: int) -> None:
        """Apply modal magnetic TF/SF corrections and virtual-guide coupling."""

        pass

    def update_eigenmode_sources_electric(self, iteration: int) -> None:
        """Apply modal electric TF/SF corrections and virtual-guide coupling."""

        pass

    def observe_eigenmode_ports(self, iteration: int) -> None:
        """Accumulate modal port DFTs."""

        pass

    def update_network_terminals(self, iteration: int) -> None:
        """Apply sparse rational-network corrections to electric edges."""

        pass

    def update_impedance_surfaces(self) -> None:
        """Apply sparse surface-impedance corrections to boundary E edges."""

        pass

    @abstractmethod
    def update_electric_b(self) -> None:
        """If there are any dispersive materials do 2nd part of dispersive
        update - it is split into two parts as it requires present and
        updated electric field values. Therefore it can only be completely
        updated after the electric field has been updated by the PML and
        source updates.
        """
        pass

    def update_symmetry_boundaries_electric_b(self) -> None:
        """Apply phase B of any dispersive PMC boundary correction."""

        pass

    @abstractmethod
    def time_start(self) -> None:
        """Starts timer used to calculate solving time for model."""
        pass

    @abstractmethod
    def calculate_solve_time(self) -> float:
        """Calculates solving time for model."""
        pass

    def finalise(self) -> None:
        pass

    def cleanup(self) -> None:
        pass
