# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom


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

import gprMax.config as config
from gprMax.cuda_opencl import knl_subgrid_hsg, knl_subgrid_precursors
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.subgrids.grid import SubGridBaseGrid
from gprMax.subgrids.precursor_nodes import PrecursorNodes

from ..updates.cuda_updates import CUDAUpdates

logger = logging.getLogger(__name__)

INTERFACE_KNLS = {
    "update_is": knl_subgrid_hsg.update_is,
    "update_electric_os": knl_subgrid_hsg.update_electric_os,
    "update_magnetic_os": knl_subgrid_hsg.update_magnetic_os,
}

PRECURSOR_KNLS = {
    "gather_taps": knl_subgrid_precursors.gather_taps,
    "bilinear_interp": knl_subgrid_precursors.bilinear_interp,
    "time_blend": knl_subgrid_precursors.time_blend,
}


def _upload_mat_coeffs(grid):
    """Put updatecoeffsE/H in global memory, once per grid.

    The plane-wave setup uploads these too, so skip it when they are already
    there rather than paying for a second transfer.
    """
    if getattr(grid, "updatecoeffsE_dev", None) is None:
        grid.htod_mat_coeff_arrays()


class CUDASubgridUpdater(CUDAUpdates):
    """Handles updating the electric and magnetic fields of an HSG subgrid on
    the GPU. The IS, OS, subgrid region and the electric/magnetic sources are
    updated using the precursor regions.

    The subgrid it drives is a CUDASubGridHSG, whose update_*_is/os methods
    launch kernels; the precursors are CUDAPrecursorNodes, which keep their
    slices on the device. Neither appears in the phase sequence, so the copy
    below stays textually identical to the CPU original.
    """

    def __init__(self, subgrid: SubGridBaseGrid, precursors: PrecursorNodes,
                 G: FDTDGrid, shared=None):
        """
        Args:
            subgrid: SubGrid3d instance to be updated.
            precursors (PrecursorNodes): PrecursorNodes instance nodes associated
                                            with the subgrid - contain interpolated
                                            fields.
            G: FDTDGrid class describing a grid in a model.
            shared: CUDAUpdates whose context this updater attaches to. The
                        main grid and the subgrid must share one context or
                        they cannot address each other's arrays.
        """
        super().__init__(subgrid, shared=shared)
        self.precursors = precursors
        self.G = G
        self.iteration = 0

        self._set_subgrid_knls()

    def _set_subgrid_knls(self):
        """HSG interface and precursor updates - prepares kernels, and gets
        kernel functions.

        The kernels take every array dimension as a runtime argument rather
        than through the macro preamble: a model may hold several subgrids of
        different sizes, so the dimensions cannot be baked in the way the bulk
        field kernels bake theirs.
        """
        # The interface kernels look material coefficients up per node and
        # take the tables as pointer arguments, so both grids need them in
        # global memory. _set_macros() does not upload them - only
        # _set_planewave_knls() does, and only when the model has a plane
        # wave - so a plain subgrid model would otherwise have no
        # updatecoeffsE_dev at all.
        _upload_mat_coeffs(self.grid)

        opts = config.sim_config.devices["nvcc_opts"]

        self.knls_interface = {}
        for name, knl in INTERFACE_KNLS.items():
            bld = self._build_knl(knl, self.subs_name_args, self.subs_func)
            module = self.source_module(bld, options=opts)
            self.knls_interface[name] = module.get_function(name)

        self.knls_precursor = {}
        for name, knl in PRECURSOR_KNLS.items():
            bld = self._build_knl(knl, self.subs_name_args, self.subs_func)
            module = self.source_module(bld, options=opts)
            self.knls_precursor[name] = module.get_function(name)

    # ---- verbatim from SubgridUpdater; see the module docstring ----------

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
        super().update_magnetic_edge_devices(self.iteration)
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
        self.update_magnetic_pml()
        precursors.calc_exact_electric_in_time()
        subgrid.update_magnetic_is(precursors)
        self.update_magnetic_sources()
        subgrid.update_magnetic_os(G)


class CUDASubgridUpdates(CUDAUpdates):
    """Updates for subgrids on the GPU.

    Owns the CUDA context; every subgrid updater attaches to it so that the
    main grid and the subgrids share one address space.
    """

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

    def finalise(self):
        """Copy device data back for the main grid and every sub-grid.

        Solver.solve() calls finalise() on this object only, but each sub-grid
        updater owns its own receiver and snapshot device arrays. Without
        fanning out, those host arrays are never written and the sub-grid's
        outputs come back as zeros - which looks like a solver bug rather than
        a missing copy.
        """
        super().finalise()
        for sg_updater in self.updaters:
            sg_updater.finalise()

    def cleanup(self):
        """Release the sub-grid updaters before the shared context.

        Safe to fan out: the updaters have _owns_context False, so they clear
        their reference without popping the context the main grid owns.
        """
        for sg_updater in self.updaters:
            sg_updater.cleanup()
        super().cleanup()


def create_cuda_updates(model, subgrid_hsg_cls):
    """Build the CUDA subgrid solver.

    Mirrors create_updates() in subgrids/updates.py, but constructs the main
    updates object first so its context can be shared with each updater -
    separate contexts would not share device memory, and the two grids could
    not exchange fields at all.

    Args:
        model: model containing the main grid and subgrids.
        subgrid_hsg_cls: the SubGridHSG class, for the type check.

    Returns:
        CUDASubgridUpdates instance.
    """
    from .cuda_precursor_nodes import (
        CUDAPrecursorNodes,
        CUDAPrecursorNodesFiltered,
    )

    updates = CUDASubgridUpdates(model.G, [])

    # The OS kernels read the MAIN grid's coefficients as pointer arguments
    _upload_mat_coeffs(model.G)

    for sg in model.subgrids:
        if not issubclass(type(sg), subgrid_hsg_cls):
            logger.exception(f"{str(sg)} is not a subgrid type")
            raise ValueError

        # Upstream added a third precursor class for ratio-one embedded
        # regions, where the two grids share a lattice and no spatial
        # interpolation is needed. The device path has no equivalent yet,
        # so refuse it rather than silently running the wrong precursors.
        if getattr(sg, "equal_resolution", False):
            raise NotImplementedError(
                f"{sg} uses equal_resolution, which the CUDA sub-grid path "
                "does not support yet - run this model on the CPU solver."
            )

        cls = CUDAPrecursorNodesFiltered if sg.filter else CUDAPrecursorNodes
        precursors = cls(model.G, sg)

        sgu = CUDASubgridUpdater(sg, precursors, model.G, shared=updates)

        # Both grids' arrays are on the device by now, so the interface and
        # the precursor descriptors can be built against them.
        sg.setup_cuda_interface(sgu.knls_interface, model.G)
        precursors.setup_device(sgu.knls_precursor, sg.gpuarray, tpb=sg.tpb[0])

        # Mirrors create_updates() on the CPU: the main grid may now carry a
        # nonzero hard-source E(0), so seed the current electric precursor
        # level before hsg_2 first reads it. Unlike the CPU, this has to come
        # after setup_device() - there are no device buffers to gather into
        # until then.
        precursors.update_electric()

        updates.updaters.append(sgu)

    return updates
