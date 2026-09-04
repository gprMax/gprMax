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

import logging

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.model import Model
from gprMax.subgrids.grid import SubGridBaseGrid

from ..updates.cpu_updates import CPUUpdates
from .precursor_nodes import (
    PrecursorNodes,
    PrecursorNodesEqualResolution,
    PrecursorNodesFiltered,
)
from .subgrid_hsg import SubGridHSG

logger = logging.getLogger(__name__)


def create_updates(model: Model):
    """Return the solver for the given subgrids."""
    updaters = []

    for sg in model.subgrids:
        sg_type = type(sg)
        if sg_type == SubGridHSG and sg.equal_resolution:
            precursors = PrecursorNodesEqualResolution(model.G, sg)
        elif sg_type == SubGridHSG and sg.filter:
            precursors = PrecursorNodesFiltered(model.G, sg)
        elif sg_type == SubGridHSG:
            precursors = PrecursorNodes(model.G, sg)
        else:
            logger.exception(f"{str(sg)} is not a subgrid type")
            raise ValueError

        sgu = SubgridUpdater(sg, precursors, model.G)
        updaters.append(sgu)

    updates = SubgridUpdates(model.G, updaters)
    return updates


class SubgridUpdates(CPUUpdates):
    """Updates for subgrids."""

    def __init__(self, G, updaters):
        super().__init__(G)
        self.updaters = updaters

    def hsg_1(self):
        """Updates the subgrids over the first phase."""
        for sg_updater in self.updaters:
            sg_updater.hsg_1()

    def hsg_2(self):
        """Updates the subgrids over the second phase."""
        for sg_updater in self.updaters:
            sg_updater.hsg_2()


class SubgridUpdater(CPUUpdates[SubGridBaseGrid]):
    """Handles updating the electric and magnetic fields of an HSG subgrid.
    The IS, OS, subgrid region and the electric/magnetic sources are updated
    using the precursor regions.
    """

    def __init__(self, subgrid: SubGridBaseGrid, precursors: PrecursorNodes, G: FDTDGrid):
        """
        Args:
            subgrid: SubGrid3d instance to be updated.
            precursors (PrecursorNodes): PrecursorNodes instance nodes associated
                                            with the subgrid - contain interpolated
                                            fields.
            G: FDTDGrid class describing a grid in a model.
        """
        super().__init__(subgrid)
        self.precursors = precursors
        self.G = G
        self.iteration = 0

    def store_outputs(self):
        """Store one complete fine-grid Yee time level.

        The HSG choreography reaches this method with electric fields at
        ``iteration * dt`` and magnetic fields at
        ``(iteration - 1/2) * dt``.  This is the same convention used by the
        main-grid solver and by the receiver/snapshot output metadata.

        The final electric update advances the subgrid to ``iterations``
        solely to complete the main-grid coupling step.  That time level is
        outside the requested output range, so do not write it.
        """

        if self.iteration >= self.grid.iterations:
            return
        super().store_outputs(self.iteration)
        super().store_snapshots(self.iteration)
        super().observe_sar_electric(self.iteration)

    def update_electric_sources(self):
        iteration = self.iteration
        super().update_electric_sources(iteration)
        super().update_eigenmode_sources_electric(iteration)
        self.iteration += 1

    def update_magnetic_sources(self):
        super().update_magnetic_sources(self.iteration)
        super().update_eigenmode_sources_magnetic(self.iteration)
        super().observe_eigenmode_ports(self.iteration)

    def update_network_terminals(self):
        """Update sparse terminals at the fine-grid electric time step."""

        # update_electric_sources() has just advanced the shared fine-grid
        # iteration counter, so the terminal history index is one behind it.
        return super().update_network_terminals(self.iteration - 1)

    def hsg_1(self):
        """First half of the subgrid update. Takes the time step up to the main
        grid magnetic update.
        """

        G = self.G
        subgrid = self.grid
        precursors = self.precursors

        # Copy the main grid electric fields at the IS position
        precursors.update_electric()

        upper_m = int(subgrid.ratio / 2 - 0.5)

        for m in range(1, upper_m + 1):
            self.update_electric_a()
            self.update_electric_pml()
            precursors.interpolate_magnetic_in_time(int(m + subgrid.ratio / 2 - 0.5))
            subgrid.update_electric_is(precursors)
            self.update_electric_sources()
            self.update_electric_b()
            self.update_network_terminals()
            self.store_outputs()
            self.update_magnetic()
            self.update_magnetic_pml()
            precursors.interpolate_electric_in_time(m)
            subgrid.update_magnetic_is(precursors)
            self.update_magnetic_sources()

        self.update_electric_a()
        if not subgrid.equal_resolution:
            self.update_electric_pml()
        precursors.calc_exact_magnetic_in_time()
        subgrid.update_electric_is(precursors)
        self.update_electric_sources()
        self.update_electric_b()
        self.update_network_terminals()
        self.store_outputs()
        subgrid.update_electric_os(G)

    def hsg_2(self):
        """Second half of the subgrid update. Takes the time step up to the main
        grid electric update.
        """

        G = self.G
        subgrid = self.grid
        precursors = self.precursors

        # Copy the main grid magnetic fields at the IS position
        precursors.update_magnetic()

        # Before the first fine-grid magnetic update the subgrid is at its
        # initial E(0), H(-1/2) state.  Record that state explicitly; all
        # subsequent output is recorded immediately after a complete electric
        # update, when the same Yee staggering is present again.
        if self.iteration == 0:
            self.store_outputs()

        upper_m = int(subgrid.ratio / 2 - 0.5)

        for m in range(1, upper_m + 1):
            self.update_magnetic()
            self.update_magnetic_pml()
            precursors.interpolate_electric_in_time(int(m + subgrid.ratio / 2 - 0.5))
            subgrid.update_magnetic_is(precursors)
            self.update_magnetic_sources()
            self.update_electric_a()
            self.update_electric_pml()
            precursors.interpolate_magnetic_in_time(m)
            subgrid.update_electric_is(precursors)
            self.update_electric_sources()
            self.update_electric_b()
            self.update_network_terminals()
            self.store_outputs()

        self.update_magnetic()
        if not subgrid.equal_resolution:
            self.update_magnetic_pml()
        precursors.calc_exact_electric_in_time()
        subgrid.update_magnetic_is(precursors)
        self.update_magnetic_sources()
        subgrid.update_magnetic_os(G)
