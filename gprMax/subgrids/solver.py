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

    solver = SubGridSolver(G, updaters)
    return solver


class SubGridSolver:
    """Solver for subgridding simulations."""

    """Class to call the various update methods required for an HSG-Subgrid simulation.
    Multiple subgrids can be updated by adding more subgrid_updater objects to the subgrid_updater
    array.
    """

    def __init__(self, G, subgrid_updaters, hsg=True):
        """
        Args:
            G (G): Grid class instance - holds essential parameters
            describing the model.
            subgrid_updaters: (list): list of subgrid_updaters used for updating
            the subgrids
            iterations (int): number of iterations for the simulation.
            hsg (bool): HSG methods for subgrids will not be called if False.

        """

        self.G = G
        self.grids = [self.G] + G.subgrids
        self.subgrid_updaters = subgrid_updaters
        self.materials = G.materials
        self.abs_time = 0
        self.hsg = hsg

    def store_outputs(self):
        """Method to store field outputs for all grid for each main grid iteration."""
        for grid in self.grids:
            store_outputs(self.G.iteration, grid.Ex, grid.Ey, grid.Ez,
                          grid.Hx, grid.Hy, grid.Hz, grid)

    def store_snapshots(self):
        """Store any snapshots."""
        for snap in self.G.snapshots:
            if snap.time == self.G.iteration + 1:
                snap.store(self.G)

    def hsg_1(self):
        """Method to update the subgrids over the first phase."""
        for sg_updater in self.subgrid_updaters:
            sg_updater.hsg_1()

    def hsg_2(self):
        """Method to update the subgrids over the second phase."""
        for sg_updater in self.subgrid_updaters:
            sg_updater.hsg_2()

    def solve(self, iterations):
        """Run timestepping."""
        tsolvestart = perf_counter()
        self.iterations = iterations
        # for time step in range(self.G.iterations):

        # The main grid FDTD loop
        for iteration in self.iterations:

            # Keep track of the index. Required for saving output correctly
            self.G.iteration = iteration

            # Write any snapshots of the E, H, I fields/currents
            self.write_snapshots(iteration)

            self.store_outputs()
            # Update main grid electric field components including sources
            self.update_magnetic()
            # Update the fields in the subgrids / main grid
            if self.hsg:
                self.hsg_2()

            # Update main grid electric field components including sources
            self.update_electric()
            # Update the fields in the subgrids / main grid
            if self.hsg:
                self.hsg_1()

        # Return the elapsed time
        tsolve = perf_counter() - tsolvestart

        return tsolve

    def update_electric(self):
        """Method to update E fields, PML and electric sources."""
        # All materials are non-dispersive so do standard update
        G = self.G

        update_electric(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsE, G.ID,
                        G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)

        # Update electric field components with the PML correction
        for pml in G.pmls:
            pml.update_electric(G)
        # Update electric field components from sources (update any Hertzian dipole sources last)
        for source in G.voltagesources + G.transmissionlines + G.hertziandipoles:
            source.update_electric(G.iteration, G.updatecoeffsE, G.ID, G.Ex,
                                   G.Ey, G.Ez, G)

    def update_magnetic(self):
        """Method to update H fields, PML and magnetic sources."""
        # Update magnetic field components
        G = self.G

        update_magnetic(G.nx, G.ny, G.nz, G.nthreads, G.updatecoeffsH, G.ID,
                        G.Ex, G.Ey, G.Ez, G.Hx, G.Hy, G.Hz)

        # Update magnetic field components with the PML correction
        for pml in G.pmls:
            pml.update_magnetic(G)

        # Update magnetic field components from sources
        for source in G.transmissionlines + G.magneticdipoles:
            source.update_magnetic(G.iteration, G.updatecoeffsH, G.ID,
                                   G.Hx, G.Hy, G.Hz, G)

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


class SubgridUpdater:
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
        self.subgrid = subgrid
        self.precursors = precursors
        self.G = G
        self.source_iteration = 0

    def hsg_1(self):
        """This is the first half of the subgrid update. Takes the time step
        up to the main grid magnetic update"""
        G = self.G
        sub_grid = self.subgrid
        precursors = self.precursors

        precursors.update_electric()

        upper_m = int(sub_grid.ratio / 2 - 0.5)

        for m in range(1, upper_m + 1):

            # store_outputs(self.grid)
            # STD update, interpolate inc. field in time, apply correction
            update_electric(sub_grid.nx,
                            sub_grid.ny,
                            sub_grid.nz,
                            G.nthreads,
                            sub_grid.updatecoeffsE,
                            sub_grid.ID,
                            sub_grid.Ex,
                            sub_grid.Ey,
                            sub_grid.Ez,
                            sub_grid.Hx,
                            sub_grid.Hy,
                            sub_grid.Hz)
            for pml in sub_grid.pmls:
                pml.update_electric(sub_grid)
            precursors.interpolate_magnetic_in_time(int(m + sub_grid.ratio / 2 - 0.5))
            sub_grid.update_electric_is(precursors)

            self.update_sub_grid_electric_sources()

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
        update_electric(sub_grid.nx,
                        sub_grid.ny,
                        sub_grid.nz,
                        G.nthreads,
                        sub_grid.updatecoeffsE,
                        sub_grid.ID,
                        sub_grid.Ex,
                        sub_grid.Ey,
                        sub_grid.Ez,
                        sub_grid.Hx,
                        sub_grid.Hy,
                        sub_grid.Hz)
        for pml in sub_grid.pmls:
            pml.update_electric(sub_grid)
        precursors.calc_exact_magnetic_in_time()
        sub_grid.update_electric_is(precursors)
        self.update_sub_grid_electric_sources()

        sub_grid.update_electric_os(G)

    def hsg_2(self):
        """This is the first half of the subgrid update. Takes the time step
        up to the main grid electric update"""
        G = self.G
        sub_grid = self.subgrid
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
            update_electric(sub_grid.nx,
                            sub_grid.ny,
                            sub_grid.nz,
                            G.nthreads,
                            sub_grid.updatecoeffsE,
                            sub_grid.ID,
                            sub_grid.Ex,
                            sub_grid.Ey,
                            sub_grid.Ez,
                            sub_grid.Hx,
                            sub_grid.Hy,
                            sub_grid.Hz)

            for pml in sub_grid.pmls:
                pml.update_electric(sub_grid)

            precursors.interpolate_magnetic_in_time(m)
            sub_grid.update_electric_is(precursors)
            self.update_sub_grid_electric_sources()
    
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
        sg = self.subgrid
        for source in sg.voltagesources + sg.transmissionlines + sg.hertziandipoles:
            source.update_electric(self.source_iteration, sg.updatecoeffsE, sg.ID,
                                   sg.Ex, sg.Ey, sg.Ez, sg)
        self.source_iteration += 1

    def update_sub_grid_magnetic_sources(self):
        """Update any magnetic sources in the subgrid"""
        sg = self.subgrid
        for source in sg.transmissionlines + sg.magneticdipoles:
            source.update_magnetic(self.source_iteration, sg.updatecoeffsH, sg.ID,
                                   sg.Hx, sg.Hy, sg.Hz, sg)
