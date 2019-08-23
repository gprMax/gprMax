from ..cython.fields_updates_normal import update_electric
from ..cython.fields_updates_normal import update_magnetic
from ..fields_outputs import store_outputs
from ..utilities import get_terminal_width
from ..exceptions import GeneralError

from .subgrid_hsg import SubGridHSG
from .precursor_nodes import PrecursorNodes as PrecursorNodesHSG
from .precursor_nodes_filtered import PrecursorNodes as PrecursorNodesFilteredHSG

from tqdm import tqdm
from time import perf_counter

import os
import sys

from ..updates import CPUUpdates



def create_solver(G):
    """Return the solver for the given subgrids."""
    updaters = []

    for sg in G.subgrids:
        print(sg)
        sg_type = type(sg)
        if sg_type == SubGridHSG and sg.filter:
            precursors = PrecursorNodesFilteredHSG(G, sg)
        elif sg_type == SubGridHSG and not sg.filter:
            precursors = PrecursorNodesHSG(G, sg)
        else:
            raise GeneralError(str(sg) + ' is not a subgrid type')

        sgu = SubgridUpdater(sg, precursors, G)
        updaters.append(sgu)

    updates = SubgridUpdates(G, updaters)
    solver = SubGridSolver(G, updates)
    return solver


class SubgridUpdates(CPUUpdates):

    def __init__(self, G, updaters):
        super().__init__(G)
        self.updaters = updaters

    def hsg_1(self):
        """Method to update the subgrids over the first phase."""
        for sg_updater in self.updaters:
            sg_updater.hsg_1()

    def hsg_2(self):
        """Method to update the subgrids over the second phase."""
        for sg_updater in self.updaters:
            sg_updater.hsg_2()

    def store_outputs(self, iteration):
        super().store_outputs(iteration)
        """Method to store field outputs for all grid for each main grid iteration."""
        for updater in self.updaters:
            updater.store_outputs(iteration)

class SubGridSolver:
    """Solver for subgridding simulations."""

    """Class to call the various update methods required for an HSG-Subgrid simulation.
    Multiple subgrids can be updated by adding more subgrid_updater objects to the subgrid_updater
    array.
    """

    def __init__(self, G, updates, hsg=True):
        """
        Args:
            G (G): Grid class instance - holds essential parameters
            describing the model.
            updates: (list): list of subgrid_updaters used for updating
            the subgrids
            hsg (bool): HSG methods for subgrids will not be called if False.
        """
        self.G = G
        self.updates = updates
        self.hsg = hsg

    def store_snapshots(self):
        """Store any snapshots."""
        for snap in self.G.snapshots:
            if snap.time == self.G.iteration + 1:
                snap.store(self.G)

    def solve(self, iterations):
        """Run timestepping."""
        tsolvestart = perf_counter()
        self.iterations = iterations
        # for time step in range(self.G.iterations):

        # The main grid FDTD loop
        for iteration in self.iterations:

            self.updates.store_outputs(iteration)
            #self.updates.store_snapshots(iteration)
            self.updates.update_magnetic()
            self.updates.update_magnetic_pml()
            self.updates.update_magnetic_sources(iteration)
            self.updates.hsg_2()
            self.updates.update_electric_a()
            self.updates.update_electric_pml()
            self.updates.update_electric_sources(iteration)
            self.updates.hsg_1()
            self.updates.update_electric_b()

            # Keep track of the index. Required for saving output correctly
            self.G.iteration = iteration

        # Return the elapsed time
        tsolve = perf_counter() - tsolvestart

        return tsolve

    def write_snapshots(self, iteration):
        # Write any snapshots to file
        for i, snap in enumerate(self.G.snapshots):
            if snap.time == iteration + 1:
                snapiters = 36 * (((snap.xf - snap.xs) / snap.dx) * ((snap.yf - snap.ys) / snap.dy) * ((snap.zf - snap.zs) / snap.dz))
                pbar = tqdm(total=snapiters, leave=False, unit='byte', unit_scale=True, desc='  Writing snapshot file {} of {}, {}'.format(i + 1, len(self.G.snapshots), os.path.split(snap.filename)[1]), ncols=get_terminal_width() - 1, file=sys.stdout, disable=self.G.tqdmdisable)

                # Use this call to print out main grid and subgrids
                snap.write_vtk_imagedata(self.G.Ex, self.G.Ey, self.G.Ez, self.G.Hx, self.G.Hy, self.G.Hz, self.G, pbar, sub_grids=self.G.subgrids)

                # Use this call to print out the standard grid without subgrid
                # snap.write_vtk_imagedata(self.G.Ex, self.G.Ey, self.G.Ez, self.G.Hx, self.G.Hy, self.G.Hz, self.G, pbar)

                # Use this call to print out only the subgrid - use in combination with commented code in .multi_cmds/snapshots.py
                # snap.write_vtk_imagedata_fast(self.grid)
                pbar.close()


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
        self.G = G
        self.source_iteration = 0

    def hsg_1(self):
        """This is the first half of the subgrid update. Takes the time step
        up to the main grid magnetic update"""
        G = self.G
        sub_grid = self.grid
        precursors = self.precursors

        precursors.update_electric()

        upper_m = int(sub_grid.ratio / 2 - 0.5)

        for m in range(1, upper_m + 1):

            # STD update, interpolate inc. field in time, apply correction
            self.update_electric_a()
            self.update_electric_pml()
            precursors.interpolate_magnetic_in_time(int(m + sub_grid.ratio / 2 - 0.5))
            sub_grid.update_electric_is(precursors)

            self.update_sub_grid_electric_sources()

            # STD update, interpolate inc. field in time, apply correction
            self.update_magnetic()
            self.update_magnetic_pml()
            precursors.interpolate_electric_in_time(m)
            sub_grid.update_magnetic_is(precursors)
            self.update_sub_grid_magnetic_sources()

        self.update_electric_a()
        self.update_electric_pml()
        precursors.calc_exact_magnetic_in_time()
        sub_grid.update_electric_is(precursors)
        self.update_sub_grid_electric_sources()
        sub_grid.update_electric_os(G)

    def hsg_2(self):
        """This is the first half of the subgrid update. Takes the time step
        up to the main grid electric update"""
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
            self.update_sub_grid_magnetic_sources()

            self.update_electric_a()
            self.update_electric_pml()

            precursors.interpolate_magnetic_in_time(m)
            sub_grid.update_electric_is(precursors)
            self.update_sub_grid_electric_sources()

        self.update_magnetic()
        self.update_magnetic_pml()
        precursors.calc_exact_electric_in_time()
        sub_grid.update_magnetic_is(precursors)
        self.update_sub_grid_magnetic_sources()
        sub_grid.update_magnetic_os(G)

    def update_sub_grid_electric_sources(self):
        """Update any electric sources in the subgrid"""
        sg = self.grid
        for source in sg.voltagesources + sg.transmissionlines + sg.hertziandipoles:
            source.update_electric(self.source_iteration, sg.updatecoeffsE, sg.ID,
                                   sg.Ex, sg.Ey, sg.Ez, sg)
        self.source_iteration += 1

    def update_sub_grid_magnetic_sources(self):
        """Update any magnetic sources in the subgrid"""
        sg = self.grid
        for source in sg.transmissionlines + sg.magneticdipoles:
            source.update_magnetic(self.source_iteration, sg.updatecoeffsH, sg.ID,
                                   sg.Hx, sg.Hy, sg.Hz, sg)
