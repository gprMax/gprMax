# Copyright (C) 2015-2019: The University of Edinburgh
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
from ..updates import CPUUpdates


def create_subgrid_updates(G):
    # collection responsible for updating each subgrid
    updaters = []

    # precursor nodes can use low pass filtering at the IS. Filters are either
    # on or off for all subgrids
    if G.subgrids[0].filter:
        from .precursor_nodes import PrecursorNodesFiltered as PrecursorNodes
    else:
        from .precursor_nodes import PrecursorNodes

    # make an updater and precursor nodes for each sub grid
    for sg in G.subgrids:
        precursors = PrecursorNodes(G, sg)
        sgu = SubgridUpdater(sg, precursors, G)
        updaters.append(sgu)

    # responsible for updating the subgridding simulation as a whole
    updates = SubgridUpdates(G, updaters)

    return updates


class SubgridUpdates(CPUUpdates):
    """Top level subgrid updater. Manages the collection of subgrids."""

    def __init__(self, G, subgrid_updaters):
        super().__init__(G)
        self.subgrid_updaters = subgrid_updaters

    def hsg_1(self):
        """Method to update the subgrids over the first phase"""
        for sg_updater in self.subgrid_updaters:
            sg_updater.hsg_1()

    def hsg_2(self):
        """Method to update the subgrids over the second phase"""
        for sg_updater in self.subgrid_updaters:
            sg_updater.hsg_2()

    def update_electric_a(self):
        super().update_electric_a()
        self.hsg_1()

    def update_magnetic(self):
        super().update_magnetic()
        self.hsg_2()

    def set_dispersive_updates(self, props):
        # set the dispersive update functions for the main grid updates
        super().set_dispersive_updates(props)
        # set the same dispersive update functions for all fields in the subgrids
        for updater in self.subgrid_updaters:
            updater.set_dispersive_updates(props)


class SubgridUpdater(CPUUpdates):
    """Class to handle updating the electric and magnetic fields of an HSG
    subgrid. The IS, OS, subgrid region and the electric/magnetic sources are updated
    using the precursor regions.
    """

    def __init__(self, subgrid, precursors, G):
        """
        Args:
            subgrid (SubGrid3d): Subgrid to be updated
            precursors (PrecursorNodes): Precursor nodes associated with
            the subgrid
            G (class): Grid class instance - holds essential parameters
            describing the model.
        """
        super().__init__(subgrid)
        self.precursors = precursors
        self.source_iteration = 0
        self.G = G

    def hsg_1(self):
        """This is the first half of the subgrid update. Takes the time step
        up to the main grid magnetic update"""
        precursors = self.precursors

        precursors.update_electric()

        # the subgrid
        sub_grid = self.grid

        upper_m = int(sub_grid.ratio / 2 - 0.5)

        for m in range(1, upper_m + 1):

            # store_outputs(self.grid)
            # STD update, interpolate inc. field in time, apply correction
            self.update_electric_a()

            for pml in sub_grid.pmls:
                pml.update_electric(sub_grid)
            self.update_sub_grid_electric_sources()
            precursors.interpolate_magnetic_in_time(int(m + sub_grid.ratio / 2 - 0.5))
            sub_grid.update_electric_is(precursors)
            self.update_electric_b()

            # STD update, interpolate inc. field in time, apply correction
            self.update_magnetic()

            for pml in sub_grid.pmls:
                pml.update_magnetic(sub_grid)
            precursors.interpolate_electric_in_time(m)
            sub_grid.update_magnetic_is(precursors)
            self.update_sub_grid_magnetic_sources()

        # store_outputs(self.grid)
        self.update_electric_a()
        for pml in sub_grid.pmls:
            pml.update_electric(sub_grid)

        self.update_sub_grid_electric_sources()

        precursors.calc_exact_magnetic_in_time()
        sub_grid.update_electric_is(precursors)
        self.update_electric_b()

        sub_grid.update_electric_os(self.G)

    def hsg_2(self):
        """This is the first half of the subgrid update. Takes the time step
        up to the main grid electric update"""
        sub_grid = self.grid
        precursors = self.precursors

        precursors.update_magnetic()

        upper_m = int(sub_grid.ratio / 2 - 0.5)

        for m in range(1, upper_m + 1):

            self.update_magnetic()

            for pml in sub_grid.pmls:
                pml.update_magnetic(sub_grid)
            precursors.interpolate_electric_in_time(int(m + sub_grid.ratio / 2 - 0.5))
            sub_grid.update_magnetic_is(precursors)

            self.update_sub_grid_magnetic_sources()

            # store_outputs(self.grid)
            self.update_electric_a()

            for pml in sub_grid.pmls:
                pml.update_electric(sub_grid)
            self.update_sub_grid_electric_sources()

            precursors.interpolate_magnetic_in_time(m)
            sub_grid.update_electric_is(precursors)
            self.update_electric_b()

        self.update_magnetic()

        for pml in sub_grid.pmls:
            pml.update_magnetic(sub_grid)

        precursors.calc_exact_electric_in_time()
        sub_grid.update_magnetic_is(precursors)
        self.update_sub_grid_magnetic_sources()

        sub_grid.update_magnetic_os(self.G)

    def update_sub_grid_electric_sources(self):
        """Update any electric sources in the subgrid"""
        sg = self.grid
        for source in sg.voltagesources + sg.transmissionlines + sg.hertziandipoles:
            source.update_electric(self.source_iteration, sg.updatecoeffsE, sg.ID, sg.Ex, sg.Ey, sg.Ez, sg)

        self.source_iteration += 1

    def update_sub_grid_magnetic_sources(self):
        """Update any magnetic sources in the subgrid"""
        sg = self.grid
        for source in sg.transmissionlines + sg.magneticdipoles:
            source.update_magnetic(self.source_iteration, sg.updatecoeffsH, sg.ID,
                                   sg.Hx, sg.Hy, sg.Hz, sg)
