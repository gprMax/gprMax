# Copyright (C) 2015-2020: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
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

from ..exceptions import GeneralError
from ..updates import CPUUpdates
from .precursor_nodes import PrecursorNodes, PrecursorNodesFiltered
from .subgrid_hsg import SubGridHSG

log = logging.getLogger(__name__)


def create_updates(G):
    """Return the solver for the given subgrids."""
    updaters = []

    for sg in G.subgrids:
        sg_type = type(sg)
        if sg_type == SubGridHSG and sg.filter:
            precursors = PrecursorNodesFiltered(G, sg)
        elif sg_type == SubGridHSG and not sg.filter:
            precursors = PrecursorNodes(G, sg)
        else:
            raise GeneralError(str(sg) + ' is not a subgrid type')

        sgu = SubgridUpdater(sg, precursors, G)
        updaters.append(sgu)

    updates = SubgridUpdates(G, updaters)
    return updates


class SubgridUpdates(CPUUpdates):
    """Provides update functions for the Sub gridding simulation."""

    def __init__(self, G, updaters):
        super().__init__(G)
        self.updaters = updaters

    def hsg_1(self):
        """Method to update the subgrids over the first phase."""
        # updaters update each subgrid
        for sg_updater in self.updaters:
            sg_updater.hsg_1()

    def hsg_2(self):
        """Method to update the subgrids over the second phase."""
        for sg_updater in self.updaters:
            sg_updater.hsg_2()


class SubgridUpdater(CPUUpdates):
    """Class to handle updating the electric and magnetic fields of an HSG
        subgrid. The IS, OS, subgrid region and the electric/magnetic sources
        are updated using the precursor regions.
    """

    def __init__(self, subgrid, precursors, G):
        """
        Args:
            subgrid (SubGrid3d): Subgrid to be updated.
            precursors (PrecursorNodes): Precursor nodes associated with
            the subgrid - contain interpolated fields.
            G (FDTDGrid): Parameters describing a grid in a model.
        """
        super().__init__(subgrid)
        self.precursors = precursors
        self.G = G
        self.source_iteration = 0

    def hsg_1(self):
        """This is the first half of the subgrid update. Takes the time step
            up to the main grid magnetic update.
        """

        G = self.G
        sub_grid = self.grid
        precursors = self.precursors

        # copy the main grid electric fields at the IS position
        precursors.update_electric()

        upper_m = int(sub_grid.ratio / 2 - 0.5)

        for m in range(1, upper_m + 1):

            self.store_outputs()
            self.update_electric_a()
            self.update_electric_pml()
            precursors.interpolate_magnetic_in_time(int(m + sub_grid.ratio / 2 - 0.5))
            sub_grid.update_electric_is(precursors)
            self.update_electric_sources()
            # second dispersive update
            self.update_electric_b()

            # STD update, interpolate inc. field in time, apply correction
            self.update_magnetic()
            self.update_magnetic_pml()
            precursors.interpolate_electric_in_time(m)
            sub_grid.update_magnetic_is(precursors)
            self.update_magnetic_sources()

        self.store_outputs()
        self.update_electric_a()
        self.update_electric_pml()
        precursors.calc_exact_magnetic_in_time()
        sub_grid.update_electric_is(precursors)
        self.update_electric_sources()
        self.update_electric_b()
        sub_grid.update_electric_os(G)

    def hsg_2(self):
        """This is the first half of the subgrid update. Takes the time step
            up to the main grid electric update.
        """

        G = self.G
        sub_grid = self.grid
        precursors = self.precursors

        precursors.update_magnetic()

        upper_m = int(sub_grid.ratio / 2 - 0.5)

        for m in range(1, upper_m + 1):

            self.update_magnetic()
            self.update_magnetic_pml()

            precursors.interpolate_electric_in_time(int(m + sub_grid.ratio / 2 - 0.5))
            sub_grid.update_magnetic_is(precursors)
            self.update_magnetic_sources()

            self.store_outputs()
            self.update_electric_a()
            self.update_electric_pml()

            precursors.interpolate_magnetic_in_time(m)
            sub_grid.update_electric_is(precursors)
            self.update_electric_sources()
            self.update_electric_b()

        self.update_magnetic()
        self.update_magnetic_pml()
        precursors.calc_exact_electric_in_time()
        sub_grid.update_magnetic_is(precursors)
        self.update_magnetic_sources()
        sub_grid.update_magnetic_os(G)
