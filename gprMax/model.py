# Copyright (C) 2015-2024: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, and John Hartley
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

import datetime
import logging
import sys
from typing import Dict, List, Optional, Sequence

import humanize
import numpy as np
import numpy.typing as npt
import psutil
from colorama import Fore, Style, init

from gprMax.geometry_outputs.geometry_objects import GeometryObject
from gprMax.geometry_outputs.geometry_view_lines import GeometryViewLines
from gprMax.geometry_outputs.geometry_view_voxels import GeometryViewVoxels
from gprMax.geometry_outputs.geometry_views import GeometryView, save_geometry_views
from gprMax.grid.cuda_grid import CUDAGrid
from gprMax.grid.metal_grid import MetalGrid
from gprMax.grid.opencl_grid import OpenCLGrid
from gprMax.subgrids.grid import SubGridBaseGrid

init()

from tqdm import tqdm

import gprMax.config as config
from gprMax.fields_outputs import write_hdf5_outputfile
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.snapshots import Snapshot, save_snapshots
from gprMax.utilities.host_info import mem_check_build_all, mem_check_run_all, set_omp_threads
from gprMax.utilities.utilities import get_terminal_width

logger = logging.getLogger(__name__)


class Model:
    """Builds and runs (solves) a model."""

    def __init__(self):
        self.title = ""

        self.dt_mod = 1.0  # Time step stability factor

        self.iteration = 0  # Current iteration number

        self.G = self._create_grid()
        self.subgrids: List[SubGridBaseGrid] = []

        self.geometryviews: List[GeometryView] = []
        self.geometryobjects: List[GeometryObject] = []

        # Monitor memory usage
        self.p = None

        # Set number of OpenMP threads to physical threads at this point to be
        # used with threaded model building methods, e.g. fractals. Can be
        # changed by the user via #omp_threads command in input file or via API
        # later for use with CPU solver.
        config.get_model_config().ompthreads = set_omp_threads(config.get_model_config().ompthreads)

    @property
    def nx(self) -> int:
        return self.G.nx

    @nx.setter
    def nx(self, value: int):
        self.G.nx = value

    @property
    def ny(self) -> int:
        return self.G.ny

    @ny.setter
    def ny(self, value: int):
        self.G.ny = value

    @property
    def nz(self) -> int:
        return self.G.nz

    @nz.setter
    def nz(self, value: int):
        self.G.nz = value

    @property
    def cells(self) -> np.uint64:
        return np.prod(self.G.size, dtype=np.uint64)

    @property
    def dx(self) -> float:
        return self.G.dl[0]

    @dx.setter
    def dx(self, value: float):
        self.G.dl[0] = value

    @property
    def dy(self) -> float:
        return self.G.dl[0]

    @dy.setter
    def dy(self, value: float):
        self.G.dl[1] = value

    @property
    def dz(self) -> float:
        return self.G.dl[0]

    @dz.setter
    def dz(self, value: float):
        self.G.dl[2] = value

    @property
    def dl(self) -> npt.NDArray[np.float64]:
        return self.G.dl

    @dl.setter
    def dl(self, value: npt.NDArray[np.float64]):
        self.G.dl = value

    @property
    def dt(self) -> float:
        return self.G.dt

    @dt.setter
    def dt(self, value: float):
        self.G.dt = value

    @property
    def iterations(self) -> int:
        return self.G.iterations

    @iterations.setter
    def iterations(self, value: int):
        self.G.iterations = value

    @property
    def timewindow(self) -> float:
        return self.G.timewindow

    @timewindow.setter
    def timewindow(self, value: float):
        self.G.timewindow = value

    @property
    def srcsteps(self) -> npt.NDArray[np.int32]:
        return self.G.srcsteps

    @srcsteps.setter
    def srcsteps(self, value: npt.NDArray[np.int32]):
        self.G.srcsteps = value

    @property
    def rxsteps(self) -> npt.NDArray[np.int32]:
        return self.G.rxsteps

    @rxsteps.setter
    def rxsteps(self, value: npt.NDArray[np.int32]):
        self.G.rxsteps = value

    def _create_grid(self) -> FDTDGrid:
        """Create grid object according to solver.

        Returns:
            grid: FDTDGrid class describing a grid in a model.
        """
        if config.sim_config.general["solver"] == "cpu":
            grid = FDTDGrid()
        elif config.sim_config.general["solver"] == "cuda":
            grid = CUDAGrid()
        elif config.sim_config.general["solver"] == "opencl":
            grid = OpenCLGrid()
        elif config.sim_config.general["solver"] == "metal":
            grid = MetalGrid()

        return grid

    def set_size(self, size: npt.NDArray[np.int32]):
        """Set size of the model.

        Args:
            size: Array to set the size (3 dimensional).
        """
        self.nx, self.ny, self.nz = size

    def add_geometry_object(
        self,
        grid: FDTDGrid,
        start: npt.NDArray[np.int32],
        stop: npt.NDArray[np.int32],
        basefilename: str,
    ) -> Optional[GeometryObject]:
        """Add a geometry object to the model.

        Args:
            grid: Grid to create a geometry object for.
            start: Lower extent of the geometry object (x, y, z).
            stop: Upper extent of the geometry object (x, y, z).
            basefilename: Output filename of the geometry object.

        Returns:
            geometry_object: The created geometry object.
        """
        geometry_object = GeometryObject(
            grid, start[0], start[1], start[2], stop[0], stop[1], stop[2], basefilename
        )
        self.geometryobjects.append(geometry_object)
        return geometry_object

    def add_geometry_view_voxels(
        self,
        grid: FDTDGrid,
        start: npt.NDArray[np.int32],
        stop: npt.NDArray[np.int32],
        dl: npt.NDArray[np.int32],
        filename: str,
    ) -> Optional[GeometryViewVoxels]:
        """Add a voxel geometry view to the model.

        Args:
            grid: Grid to create a geometry view for.
            start: Lower extent of the geometry view (x, y, z).
            stop: Upper extent of the geometry view (x, y, z).
            dl: Discritisation of the geometry view (x, y, z).
            filename: Output filename of the geometry view.

        Returns:
            geometry_view: The created geometry view.
        """
        geometry_view = GeometryViewVoxels(
            start[0],
            start[1],
            start[2],
            stop[0],
            stop[1],
            stop[2],
            dl[0],
            dl[1],
            dl[2],
            filename,
            grid,
        )
        self.geometryviews.append(geometry_view)
        return geometry_view

    def add_geometry_view_lines(
        self,
        grid: FDTDGrid,
        start: npt.NDArray[np.int32],
        stop: npt.NDArray[np.int32],
        filename: str,
    ) -> Optional[GeometryViewLines]:
        """Add a lines geometry view to the model.

        Args:
            grid: Grid to create a geometry view for.
            start: Lower extent of the geometry view (x, y, z).
            stop: Upper extent of the geometry view (x, y, z).
            filename: Output filename of the geometry view.

        Returns:
            geometry_view: The created geometry view.
        """
        geometry_view = GeometryViewLines(
            start[0],
            start[1],
            start[2],
            stop[0],
            stop[1],
            stop[2],
            filename,
            grid,
        )
        self.geometryviews.append(geometry_view)
        return geometry_view

    def add_snapshot(
        self,
        grid: FDTDGrid,
        start: npt.NDArray[np.int32],
        stop: npt.NDArray[np.int32],
        dl: npt.NDArray[np.int32],
        time: int,
        filename: str,
        fileext: str,
        outputs: Dict[str, bool],
    ) -> Optional[Snapshot]:
        """Add a snapshot to the provided grid.

        Args:
            grid: Grid to create a snapshot for.
            start: Lower extent of the snapshot (x, y, z).
            stop: Upper extent of the snapshot (x, y, z).
            dl: Discritisation of the snapshot (x, y, z).
            time: Iteration number to take the snapshot on
            filename: Output filename of the snapshot.
            fileext: File extension of the snapshot.
            outputs: Fields to use in the snapshot.

        Returns:
            snapshot: The created snapshot.
        """
        snapshot = Snapshot(
            start[0],
            start[1],
            start[2],
            stop[0],
            stop[1],
            stop[2],
            dl[0],
            dl[1],
            dl[2],
            time,
            filename,
            fileext,
            outputs,
            grid,
        )
        # TODO: Move snapshots into the Model
        grid.snapshots.append(snapshot)
        return snapshot

    def build(self):
        """Builds the Yee cells for a model."""

        # Monitor memory usage
        self.p = psutil.Process()

        # Normal model reading/building process; bypassed if geometry information to be reused
        if config.get_model_config().reuse_geometry():
            self.reuse_geometry()
        else:
            self.build_geometry()

        # TFSF corrections are applied on the main grid. A box may contain a
        # complete subgrid, but it must not cut through the HSG outer coupling
        # surface where main- and fine-grid fields are exchanged.
        if self.subgrids and self.G.discreteplanewaves:
            from gprMax.ntff.interface import validate_tfsf_subgrid_enclosure

            validate_tfsf_subgrid_enclosure(self)

        # NTFF surface definitions are registered while the scene is parsed,
        # but formulation-specific component surfaces can only be compiled
        # after grid.build() has finalised the Yee material IDs.
        if getattr(self.G, "ntff_surface_specs", None):
            from gprMax.ntff.interface import compile_ntff_outputs

            if not self.G.ntff_output_writers:
                compile_ntff_outputs(self, self.G)

        logger.info(
            f"Output directory: {config.get_model_config().output_file_path.parent.resolve()}\n"
        )

        grids = [self.G] + self.subgrids
        for grid in grids:
            grid.update_sources_and_recievers()

        # A Study supplies absolute per-case state after the legacy linear
        # src/rx stepping hooks. This ordering makes Study authoritative and,
        # because it restores its captured baseline first, prevents changes
        # leaking from one reused-geometry run into the next.
        if config.sim_config.study is not None:
            config.sim_config.study.apply_case(self)

        # Magnetic-frill sources bind the attached thin-wire radius, resolve
        # symmetry, validate the PEC ground plane, and precompute their
        # feed-cell recurrence here. This needs final material coefficients
        # and symmetry boundaries, which do not exist during scene parsing.
        # On MPI grids a frill is replicated so its four-edge feed stencil can
        # span ranks; on a CPU subgrid it is prepared against that fine grid.
        for grid in grids:
            for frill in grid.magneticfrillsources:
                frill.finalise_setup(grid)

        # Rational networks are local terminal corrections. Bind their final
        # Yee material/source coefficient and initialise sparse pole state only
        # after grid.build() has completed.
        for grid in grids:
            for terminal in getattr(grid, "networkterminals", ()):
                terminal.prepare(grid)

        # Voltage-source ports bind their receiver during scene processing,
        # but their effective edge material and update coefficient only exist
        # after grid.build() has completed.
        for grid in grids:
            for port in getattr(grid, "port_monitors", ()):
                port.prepare(grid)

        # Transmission lines already record incident and terminal voltage and
        # current histories. Prepare their automatic S11/impedance outputs
        # after materials and the native time/frequency axes are finalised.
        from gprMax.ports import prepare_magnetic_frill_ports, prepare_transmission_line_ports

        # MPI transmission-line objects are gathered only after solving. Do
        # not attach cached spectral arrays before that transfer; the
        # coordinator prepares and finalises them after the gather instead.
        if not hasattr(self.G, "comm"):
            for grid in grids:
                prepare_transmission_line_ports(grid)
                prepare_magnetic_frill_ports(grid)

        # Source stepping is applied immediately above, so enclosure is
        # checked against the positions actually used by this model run.
        if self.G.ntff_monitors:
            from gprMax.ntff.interface import validate_ntff_source_enclosure

            validate_ntff_source_enclosure(self, self.G)

        self._output_geometry()

    def _output_geometry(self):
        # Write files for any geometry views and geometry object outputs
        if (
            not self.geometryviews
            and not self.geometryobjects
            and config.sim_config.args.geometry_only
        ):
            logger.warning(
                "Geometry only run specified, but no geometry views or geometry objects found."
            )
            return

        save_geometry_views(self.geometryviews)

        if self.geometryobjects:
            logger.info("")
            for i, go in enumerate(self.geometryobjects):
                pbar = tqdm(
                    total=go.datawritesize,
                    unit="byte",
                    unit_scale=True,
                    desc=f"Writing geometry object file {i + 1}/{len(self.geometryobjects)}, "
                    + f"{go.filename_hdf5.name}",
                    ncols=get_terminal_width() - 1,
                    file=sys.stdout,
                    disable=not config.sim_config.general["progressbars"],
                )
                go.write_hdf5(self.title, pbar)
                pbar.close()
            logger.info("")

    def build_geometry(self):
        logger.info(config.get_model_config().inputfilestr)

        # Print info on any subgrids
        for subgrid in self.subgrids:
            subgrid.print_info()

        # Combine available grids
        grids = [self.G] + self.subgrids

        self._check_stateful_sources_with_geometry_fixed(grids)
        # The arithmetic average of several dispersive media can contain more
        # inclusive terms than any constituent. Resolve electric compound
        # materials before selecting dispersive storage and checking memory
        # so the dense model-wide maxpoles allocation is estimated correctly.
        if config.get_model_config().dispersive_averaging:
            for grid in grids:
                if any(
                    material.averagable and getattr(material, "poles", 0) > 0
                    for material in grid.materials
                ):
                    grid.prepare_electric_components()
        self._check_for_dispersive_materials(grids)
        self._check_accelerator_symmetry_boundaries(grids)
        self._check_memory_requirements(grids)

        for grid in grids:
            grid.build()
            grid.dispersion_analysis(self.iterations)

    def _check_stateful_sources_with_geometry_fixed(self, grids: Sequence[FDTDGrid]):
        # TransmissionLine and DiscretePlaneWave each carry their own
        # persistent internal state (TL: voltage/current/ABC history;
        # DPW: its own internal 1D E/H-field and PML-integral arrays) that
        # is never reset between geometry_fixed reuse runs (only grid
        # fields/PMLs are). Neither source can even vary between those
        # runs in the first place - TransmissionLine is explicitly
        # excluded from #src_steps repositioning
        # (FDTDGrid.update_simple_source_positions()), and a
        # DiscretePlaneWave's angle/direction is fixed once at scene-parse
        # time with no per-run mechanism to change it. So with
        # geometry_fixed and more than one model requested, every run
        # after the first would silently reuse the exact same source at
        # the exact same configuration, contaminated by the previous
        # run's leftover internal state - not a "reuse the geometry, vary
        # something else" scenario at all, just a broken repeat of an
        # identical model. (A stepped receiver/dipole elsewhere in the
        # same scene doesn't need multiple runs either - use multiple
        # #rx commands, or #src_steps/#rx_steps, in a single run instead.)
        # Eigenmode sources and receivers also attach persistent modal DFT
        # accumulators, recursive phase state, and derived S-parameters to
        # their grids. grid.reset_fields() does not reset any of that state.
        if config.sim_config.geometry_fixed and config.sim_config.number_of_models > 1:
            if any(grid.transmissionlines for grid in grids):
                raise ValueError(
                    "#transmission_line cannot be used with geometry_fixed when more "
                    "than one model is requested (n > 1) - a transmission line's "
                    "internal state (voltage/current/ABC history) is not reset "
                    "between reused-geometry runs, and it cannot be repositioned "
                    "via #src_steps, so every run after the first would silently "
                    "repeat the identical, contaminated source. Run a single model "
                    "instead, or use #voltage_source if you need geometry_fixed."
                )
            if any(grid.discreteplanewaves for grid in grids):
                raise ValueError(
                    "A discrete plane wave command cannot be used with "
                    "geometry_fixed when more than one model is requested (n > 1) "
                    "- its internal state (the plane wave's own 1D E/H-field and "
                    "PML-integral arrays) is not reset between reused-geometry "
                    "runs, and its angle/direction cannot vary between runs, so "
                    "every run after the first would silently repeat the "
                    "identical, contaminated source. Run a single model instead."
                )
            if any(grid.magneticfrillsources for grid in grids):
                raise ValueError(
                    "#magnetic_frill_source cannot be used with geometry_fixed "
                    "when more than one model is requested (n > 1) - its "
                    "internal voltage/current history arrays are not reset "
                    "between reused-geometry runs, so every run after the "
                    "first would retain state from the previous run and "
                    "contaminate "
                    "Vtotal/S11/Zin output with no error. Run a single model "
                    "instead."
                )
            if any(
                grid.eigenmodesources or grid.eigenmodereceivers or grid.virtual_waveguide_specs
                for grid in grids
            ):
                raise ValueError(
                    "EigenmodeBand, EigenmodePort, EigenmodeExcitation, and "
                    "VirtualWaveguide cannot "
                    "be used with geometry_fixed when more "
                    "than one model is requested (n > 1) - their modal DFT "
                    "accumulators, recursive phase state, and derived "
                    "S-parameters are not reset between reused-geometry runs. "
                    "Run a single model instead."
                )

    def _check_for_dispersive_materials(self, grids: Sequence[FDTDGrid]):
        # Check for dispersive materials (and specific type)
        if config.get_model_config().materials["maxpoles"] != 0:
            # dispersivedtype/dispersiveCdtype are single, model-wide
            # settings (not per-grid) - every grid's dispersive arrays are
            # allocated using them (FDTDGrid.initialise_dispersive_arrays()),
            # so drudelorentz must be True if ANY grid (main or subgrid)
            # contains a Drude/Lorentz material, not just the last one
            # checked. Getting this wrong doesn't just pick the wrong
            # update kernel - it allocates a real-dtype updatecoeffsdispersive
            # array for a grid whose materials need complex pole
            # coefficients, silently truncating them (numpy raises only a
            # ComplexWarning on such an assignment, not an error).
            config.get_model_config().materials["drudelorentz"] = any(
                "drude" in m.type or "lorentz" in m.type for grid in grids for m in grid.materials
            )

            # Set data type if any dispersive materials (must be done before memory checks)
            config.get_model_config().set_dispersive_material_types()

            # TODO: This is the correct, model-wide-safe fix (every grid
            # gets a dtype capable of any dispersive material present
            # anywhere), but it's not the ideal one. A Debye-only subgrid
            # still pays for complex-arithmetic dispersive kernels/arrays
            # just because the main grid (or another subgrid) has a
            # Lorentz/Drude material. Doing this properly would mean
            # making drudelorentz/dispersivedtype/dispersiveCdtype/
            # crealfunc per-grid attributes instead of singleton
            # ModelConfig ones, and updating every backend's dispersive
            # kernel-selection code (e.g. CPUUpdates.set_dispersive_updates(),
            # the CUDA/OpenCL/Metal equivalents) to read the owning grid's
            # own flag rather than the global one.

    def _check_accelerator_symmetry_boundaries(self, grids: Sequence[FDTDGrid]):
        """Reserved parity check after material dispersion is resolved."""

        # All local backends now implement both phases of the dispersive PMC
        # ADE update. Keep this hook because future backends may need an
        # explicit capability check here.

    def _check_memory_requirements(self, grids: Sequence[FDTDGrid]):
        # Check memory requirements to build model/scene (different to memory
        # requirements to run model when FractalVolumes/FractalSurfaces are
        # used as these can require significant additional memory)
        total_mem_build, mem_strs_build = mem_check_build_all(grids)

        # Check memory requirements to run model
        total_mem_run, mem_strs_run = mem_check_run_all(grids)

        if total_mem_build > total_mem_run:
            logger.info(
                f'Memory required (estimated): {" + ".join(mem_strs_build)} + '
                f"~{humanize.naturalsize(config.get_model_config().mem_overhead)} "
                f"overhead = {humanize.naturalsize(total_mem_build)}\n"
            )
        else:
            logger.info(
                f'Memory required (estimated): {" + ".join(mem_strs_run)} + '
                f"~{humanize.naturalsize(config.get_model_config().mem_overhead)} "
                f"overhead = {humanize.naturalsize(total_mem_run)}\n"
            )

    def reuse_geometry(self):
        s = (
            f"\n--- Model {config.get_model_config().appendmodelnumber}/{config.sim_config.model_end}, "
            f"input file (not re-processed, i.e. geometry fixed): "
            f"{config.sim_config.input_file_path}"
        )
        config.get_model_config().inputfilestr = (
            Fore.GREEN + f"{s} {'-' * (get_terminal_width() - 1 - len(s))}\n\n" + Style.RESET_ALL
        )
        logger.basic(config.get_model_config().inputfilestr)
        self.iteration = 0  # Reset current iteration number
        for grid in [self.G] + self.subgrids:
            grid.reset_fields()

    def write_output_data(self):
        """Writes output data, i.e. field data for receivers and snapshots to
        file(s).
        """

        # Device finalisation has already copied receiver histories to the
        # host. Complete derived port spectra before opening the HDF5 file so
        # a calculation error cannot leave a partially written port group.
        grids = [self.G] + self.subgrids
        from gprMax.eigenmode_ports import finalise_eigenmode_ports

        for grid in grids:
            finalise_eigenmode_ports(grid)
        for grid in grids:
            for port in getattr(grid, "port_monitors", ()):
                port.finalise(grid)

        from gprMax.ports import finalise_magnetic_frill_ports, finalise_transmission_line_ports

        for grid in [self.G] + self.subgrids:
            finalise_transmission_line_ports(grid)
            finalise_magnetic_frill_ports(grid)

        if config.sim_config.study is not None:
            config.sim_config.study.collect_case(self)

        # Write output data to file if they are any receivers in any grids
        sg_rxs = [True for sg in self.subgrids if sg.rxs]
        sg_tls = [True for sg in self.subgrids if sg.transmissionlines]
        sg_frills = [True for sg in self.subgrids if sg.magneticfrillsources]
        sg_ports = [True for sg in self.subgrids if sg.port_monitors]
        ntff_outputs = [
            monitor
            for monitor in self.G.ntff_monitors
            if getattr(monitor, "write_hdf5", None) is not None
        ]
        ntff_outputs.extend(getattr(self.G, "ntff_output_writers", ()))
        if (
            self.G.rxs
            or sg_rxs
            or self.G.transmissionlines
            or sg_tls
            or self.G.magneticfrillsources
            or sg_frills
            or sg_ports
            or ntff_outputs
            or self.G.port_monitors
            or self.G.eigenmodeports
            or any(grid.eigenmodeports for grid in self.subgrids)
        ):
            write_hdf5_outputfile(config.get_model_config().output_file_path_ext, self.title, self)

        # Write any snapshots to file for each grid
        for grid in [self.G] + self.subgrids:
            if grid.snapshots:
                save_snapshots(grid.snapshots)

    def solve(self, solver):
        """Solve using FDTD method.

        Args:
            solver: solver object.
        """

        # Print information about and check OpenMP threads
        if config.sim_config.general["solver"] == "cpu":
            if config.sim_config.mpi:
                backend = "MPI+OpenMP"
                layout = f"{np.prod(config.sim_config.mpi)} MPI rank(s) and {config.get_model_config().ompthreads} thread(s) per rank"
            else:
                backend = "OpenMP"
                layout = f"{config.get_model_config().ompthreads} thread(s)"
            logger.basic(
                f"Model {config.sim_config.current_model + 1}/{config.sim_config.model_end} "
                f"on {config.sim_config.hostinfo['hostname']} "
                f"with {backend} backend using {layout}"
            )
            if config.get_model_config().ompthreads > config.sim_config.hostinfo["physicalcores"]:
                logger.warning(
                    f"You have specified more threads ({config.get_model_config().ompthreads}) "
                    f"than available physical CPU cores ({config.sim_config.hostinfo['physicalcores']}). "
                    f"This may lead to degraded performance."
                )
        elif config.sim_config.general["solver"] in ["cuda", "opencl", "metal"]:
            if config.sim_config.general["solver"] == "opencl":
                solvername = "OpenCL"
                platformname = (
                    " ".join(config.get_model_config().device["dev"].platform.name.split())
                    + " with "
                )
                devicename = (
                    f'Device {config.get_model_config().device["deviceID"]}: '
                    f'{" ".join(config.get_model_config().device["dev"].name.split())}'
                )
            elif config.sim_config.general["solver"] == "cuda":
                solvername = "CUDA"
                platformname = ""
                devicename = (
                    f'Device {config.get_model_config().device["deviceID"]}: '
                    f'{" ".join(config.get_model_config().device["dev"].name().split())}'
                )
            else:  # Metal
                solvername = "Apple Metal"
                platformname = ""
                devicename = (
                    f'Device {config.get_model_config().device["deviceID"]}: '
                    f'{" ".join(config.get_model_config().device["dev"].name().split())}'
                )
            logger.basic(
                f"\nModel {config.sim_config.current_model + 1}/{config.sim_config.model_end} "
                f"solving on {config.sim_config.hostinfo['hostname']} "
                f"with {solvername} backend using {platformname}{devicename}"
            )

        # Prepare iterator
        if config.sim_config.general["progressbars"]:
            iterator = tqdm(
                range(self.iterations),
                desc="|--->",
                ncols=get_terminal_width() - 1,
                file=sys.stdout,
                disable=not config.sim_config.general["progressbars"],
            )
        else:
            iterator = range(self.iterations)

        # Run solver
        solver.solve(iterator)

        # Write output data, i.e. field data for receivers and snapshots to file(s)
        self.write_output_data()

        # Print information about memory usage and solving time for a model
        # Add a string on device (GPU) memory usage if applicable
        mem_str = ""
        if config.sim_config.general["solver"] == "cuda":
            mem_str = f" host + ~{humanize.naturalsize(solver.memused)} device"
        elif config.sim_config.general["solver"] == "opencl":
            mem_str = f" host + unknown for device"

        logger.info(
            f"Memory used (estimated): "
            + f"~{humanize.naturalsize(self.p.memory_full_info().uss)}{mem_str}"
        )
        logger.info(
            f"Time taken: "
            + f"{humanize.precisedelta(datetime.timedelta(seconds=solver.solvetime), format='%0.4f')}\n"
        )
