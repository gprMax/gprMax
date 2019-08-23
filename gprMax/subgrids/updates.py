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
from gprMax.utilities import timer
from ..cython.fields_updates_normal import update_electric
from ..cython.fields_updates_normal import update_magnetic


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


class SubgridSolver:

    """Generic solver for Update objects"""

    def __init__(self, updates):
        """Context for the model to run in. Sub-class this with contexts
        i.e. an MPI context.

        Args:
            updates (Updates): updates contains methods to run FDTD algorithm
            iterator (iterator): can be range() or tqdm()
        """
        self.updates = updates

    def get_G(self):
        return self.updates.grid

    def update_electric_1(self):
        """Method to update E fields, PML and electric sources"""
        G = self.get_G()
        update_electric(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsE, G.ID, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)

        # Update electric field components with the PML correction
        for pml in G.pmls:
            pml.update_electric(G)
        # Update electric field components from sources (update any Hertzian dipole sources last)
        for source in G.voltagesources + G.transmissionlines + G.hertziandipoles:
            source.update_electric(G.iteration, G.updatecoeffsE, G.ID, G.Ex, G.Ey, G.Ez, G)

    def update_electric_2(self):
        return

    def update_magnetic(self):
        """Method to update H fields, PML and magnetic sources"""
        # Update magnetic field components
        G = self.get_G()

        update_magnetic(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsH, G.ID, G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)

        # Update magnetic field components with the PML correction
        for pml in G.pmls:
            pml.update_magnetic(G)

        # Update magnetic field components from sources
        for source in G.transmissionlines + G.magneticdipoles:
            source.update_magnetic(G.iteration, G.updatecoeffsH, G.ID, G.Hx, G.Hy, G.Hz, G)

    def solve(self, iterator):
        """Time step the FDTD model."""
        tsolvestart = timer()

        for iteration in iterator:
            # Update main grid electric field components including sources
            self.updates.store_outputs(iteration)
            self.update_electric_1()
            #self.updates.update_electric_pml()
            #self.updates.update_electric_sources(iteration)
            # Update the fields in the subgrids / main grid
            self.updates.hsg_1()
            # apply 2nd dispersive update after OS updates
            #self.updates.update_electric_b()
            # Update main grid electric field components including sources
            self.update_magnetic()
            #self.updates.update_magnetic_pml()
            #self.updates.update_magnetic_sources(iteration)
            # Update the fields in the subgrids / main grid
            self.updates.hsg_2()

        tsolve = timer() - tsolvestart
        return tsolve


class SubgridUpdates(CPUUpdates):
    """Top level subgrid updater. Manages the collection of subgrids."""

    def __init__(self, G, subgrid_updaters):
        super().__init__(G)
        self.subgrid_updaters = subgrid_updaters

    def store_outputs(self, iteration):
        super().store_outputs(iteration)
        for updater in self.subgrid_updaters:
            updater.store_outputs(iteration)

    def hsg_1(self):
        """Method to update the subgrids over the first phase"""
        for sg_updater in self.subgrid_updaters:
            sg_updater.hsg_1()

    def hsg_2(self):
        """Method to update the subgrids over the second phase"""
        for sg_updater in self.subgrid_updaters:
            sg_updater.hsg_2()

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

    def update_electric_2(self):
        return

    def update_electric_1(self):
        subgrid = self.grid
        # Update electric field components
        # All materials are non-dispersive so do standard update
        update_electric(subgrid.nx, subgrid.ny, subgrid.nz, subgrid.nthreads, subgrid.updatecoeffsE, subgrid.ID, subgrid.Ex, subgrid.Ey, subgrid.Ez, subgrid.Hx, subgrid.Hy, subgrid.Hz)

    def hsg_1(self):
        G = self.G
        sub_grid = self.grid

        precursors = self.precursors

        precursors.update_electric()

        upper_m = int(sub_grid.ratio / 2 - 0.5)

        for m in range(1, upper_m + 1):

            # store_outputs(self.grid)
            # STD update, interpolate inc. field in time, apply correction
            self.update_electric_1()

            for pml in sub_grid.pmls:
                pml.update_electric(sub_grid)
            self.update_sub_grid_electric_sources()
            precursors.interpolate_magnetic_in_time(int(m + sub_grid.ratio / 2 - 0.5))
            sub_grid.update_electric_is(precursors)
            self.update_electric_2()

            # STD update, interpolate inc. field in time, apply correction
            update_magnetic(sub_grid.nx,
                            sub_grid.ny,
                            sub_grid.nz,
                            G.nthreads,
                            sub_grid.updatecoeffsH,
                            sub_grid.ID,
                            sub_grid.Ex,
                            sub_grid.Ey,
                            sub_grid.Ez,
                            sub_grid.Hx,
                            sub_grid.Hy,
                            sub_grid.Hz)

            for pml in sub_grid.pmls:
                pml.update_magnetic(sub_grid)
            precursors.interpolate_electric_in_time(m)
            sub_grid.update_magnetic_is(precursors)
            self.update_sub_grid_magnetic_sources()

        # store_outputs(self.grid)
        self.update_electric_1()
        for pml in sub_grid.pmls:
            pml.update_electric(sub_grid)

        self.update_sub_grid_electric_sources()

        precursors.calc_exact_magnetic_in_time()
        sub_grid.update_electric_is(precursors)
        self.update_electric_2()

        sub_grid.update_electric_os(G)

    def hsg_2(self):
        G = self.G
        sub_grid = self.grid
        precursors = self.precursors

        precursors.update_magnetic()

        upper_m = int(sub_grid.ratio / 2 - 0.5)

        for m in range(1, upper_m + 1):

            update_magnetic(sub_grid.nx,
                            sub_grid.ny,
                            sub_grid.nz,
                            G.nthreads,
                            sub_grid.updatecoeffsH,
                            sub_grid.ID,
                            sub_grid.Ex,
                            sub_grid.Ey,
                            sub_grid.Ez,
                            sub_grid.Hx,
                            sub_grid.Hy,
                            sub_grid.Hz)

            for pml in sub_grid.pmls:
                pml.update_magnetic(sub_grid)
            precursors.interpolate_electric_in_time(int(m + sub_grid.ratio / 2 - 0.5))
            sub_grid.update_magnetic_is(precursors)

            self.update_sub_grid_magnetic_sources()

            # store_outputs(self.grid)
            self.update_electric_1()

            for pml in sub_grid.pmls:
                pml.update_electric(sub_grid)
            self.update_sub_grid_electric_sources()

            precursors.interpolate_magnetic_in_time(m)
            sub_grid.update_electric_is(precursors)
            self.update_electric_2()

        #sub_grid.update_magnetic_os(G)

        update_magnetic(sub_grid.nx,
                        sub_grid.ny,
                        sub_grid.nz,
                        G.nthreads,
                        sub_grid.updatecoeffsH,
                        sub_grid.ID,
                        sub_grid.Ex,
                        sub_grid.Ey,
                        sub_grid.Ez,
                        sub_grid.Hx,
                        sub_grid.Hy,
                        sub_grid.Hz)

        for pml in sub_grid.pmls:
            pml.update_magnetic(sub_grid)

        precursors.calc_exact_electric_in_time()
        sub_grid.update_magnetic_is(precursors)
        self.update_sub_grid_magnetic_sources()

        sub_grid.update_magnetic_os(G)

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
