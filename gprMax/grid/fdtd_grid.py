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

import decimal
import itertools
import logging
import sys
from collections import OrderedDict
from typing import Any, Iterable, List, Union

import humanize
import numpy as np
from terminaltables import SingleTable
from tqdm import tqdm

from gprMax import config
from gprMax.cython.yee_cell_build import build_electric_components, build_magnetic_components

# from gprMax.geometry_outputs import GeometryObjects, GeometryView
from gprMax.materials import process_materials
from gprMax.pml import CFS, PML, build_pml, print_pml_info
from gprMax.receivers import Rx
from gprMax.snapshots import Snapshot
from gprMax.sources import HertzianDipole, MagneticDipole, Source, TransmissionLine, VoltageSource

# from gprMax.subgrids.grid import SubGridBaseGrid
from gprMax.utilities.host_info import mem_check_build_all, mem_check_run_all
from gprMax.utilities.utilities import fft_power, get_terminal_width, round_value
from gprMax.waveforms import Waveform

logger = logging.getLogger(__name__)


class FDTDGrid:
    """Holds attributes associated with entire grid. A convenient way for
    accessing regularly used parameters.
    """

    def __init__(self):
        self.name = "main_grid"
        self.mem_use = 0

        self.nx = 0
        self.ny = 0
        self.nz = 0
        self.dl: np.ndarray[Any, np.dtype[np.single]]
        self.dt = 0.0

        # PML parameters - set some defaults to use if not user provided
        self.pmls = {}
        self.pmls["formulation"] = "HORIPML"
        self.pmls["cfs"] = []
        self.pmls["slabs"] = []
        # Ordered dictionary required so *updating* the PMLs always follows the
        # same order (the order for *building* PMLs does not matter). The order
        # itself does not matter, however, if must be the same from model to
        # model otherwise the numerical precision from adding the PML
        # corrections will be different.
        self.pmls["thickness"] = OrderedDict((key, 10) for key in PML.boundaryIDs)

        # TODO: Add type information.
        # Currently importing GeometryObjects, GeometryView, and
        # SubGridBaseGrid cause cyclic dependencies
        self.averagevolumeobjects = True
        self.fractalvolumes = []
        self.geometryviews = []
        self.geometryobjectswrite = []
        self.waveforms: List[Waveform] = []
        self.voltagesources: List[VoltageSource] = []
        self.hertziandipoles: List[HertzianDipole] = []
        self.magneticdipoles: List[MagneticDipole] = []
        self.transmissionlines: List[TransmissionLine] = []
        self.rxs: List[Rx] = []
        self.srcsteps: List[int] = [0, 0, 0]
        self.rxsteps: List[int] = [0, 0, 0]
        self.snapshots: List[Snapshot] = []

    @property
    def dx(self) -> float:
        return self.dl[0]

    @dx.setter
    def dx(self, value: float):
        self.dl[0] = value

    @property
    def dy(self) -> float:
        return self.dl[1]

    @dy.setter
    def dy(self, value: float):
        self.dl[1] = value

    @property
    def dz(self) -> float:
        return self.dl[2]

    @dz.setter
    def dz(self, value: float):
        self.dl[2] = value

    def build(self) -> None:
        # Print info on any subgrids
        for subgrid in self.subgrids:
            subgrid.print_info()

        # Combine available grids
        grids = [self] + self.subgrids

        # Check for dispersive materials (and specific type)
        if config.get_model_config().materials["maxpoles"] != 0:
            # TODO: This sets materials["drudelorentz"] based only the
            # last grid/subgrid. Is that correct?
            for grid in grids:
                config.get_model_config().materials["drudelorentz"] = any(
                    [m for m in grid.materials if "drude" in m.type or "lorentz" in m.type]
                )

            # Set data type if any dispersive materials (must be done before memory checks)
            config.get_model_config().set_dispersive_material_types()

        # Check memory requirements to build model/scene (different to memory
        # requirements to run model when FractalVolumes/FractalSurfaces are
        # used as these can require significant additional memory)
        total_mem_build, mem_strs_build = mem_check_build_all(grids)

        # Check memory requirements to run model
        total_mem_run, mem_strs_run = mem_check_run_all(grids)

        if total_mem_build > total_mem_run:
            logger.info(
                f'\nMemory required (estimated): {" + ".join(mem_strs_build)} + '
                f"~{humanize.naturalsize(config.get_model_config().mem_overhead)} "
                f"overhead = {humanize.naturalsize(total_mem_build)}"
            )
        else:
            logger.info(
                f'\nMemory required (estimated): {" + ".join(mem_strs_run)} + '
                f"~{humanize.naturalsize(config.get_model_config().mem_overhead)} "
                f"overhead = {humanize.naturalsize(total_mem_run)}"
            )

        # Build grids
        for grid in grids:
            # Set default CFS parameter for PMLs if not user provided
            if not grid.pmls["cfs"]:
                grid.pmls["cfs"] = [CFS()]
            logger.info(print_pml_info(grid))
            if not all(value == 0 for value in grid.pmls["thickness"].values()):
                grid._build_pmls()
            if grid.averagevolumeobjects:
                grid._build_components()
            grid._tm_grid_update()
            grid._update_voltage_source_materials()
            grid.initialise_field_arrays()
            grid.initialise_std_update_coeff_arrays()
            if config.get_model_config().materials["maxpoles"] > 0:
                grid.initialise_dispersive_arrays()
                grid.initialise_dispersive_update_coeff_array()
            grid._build_materials()

            # Check to see if numerical dispersion might be a problem
            results = dispersion_analysis(grid)
            if results["error"]:
                logger.warning(
                    f"\nNumerical dispersion analysis [{grid.name}] "
                    f"not carried out as {results['error']}"
                )
            elif results["N"] < config.get_model_config().numdispersion["mingridsampling"]:
                logger.exception(
                    f"\nNon-physical wave propagation in [{grid.name}] "
                    f"detected. Material '{results['material'].ID}' "
                    f"has wavelength sampled by {results['N']} cells, "
                    f"less than required minimum for physical wave "
                    f"propagation. Maximum significant frequency "
                    f"estimated as {results['maxfreq']:g}Hz"
                )
                raise ValueError
            elif (
                results["deltavp"]
                and np.abs(results["deltavp"])
                > config.get_model_config().numdispersion["maxnumericaldisp"]
            ):
                logger.warning(
                    f"\n[{grid.name}] has potentially significant "
                    f"numerical dispersion. Estimated largest physical "
                    f"phase-velocity error is {results['deltavp']:.2f}% "
                    f"in material '{results['material'].ID}' whose "
                    f"wavelength sampled by {results['N']} cells. "
                    f"Maximum significant frequency estimated as "
                    f"{results['maxfreq']:g}Hz"
                )
            elif results["deltavp"]:
                logger.info(
                    f"\nNumerical dispersion analysis [{grid.name}]: "
                    f"estimated largest physical phase-velocity error is "
                    f"{results['deltavp']:.2f}% in material '{results['material'].ID}' "
                    f"whose wavelength sampled by {results['N']} cells. "
                    f"Maximum significant frequency estimated as "
                    f"{results['maxfreq']:g}Hz"
                )

    def _build_pmls(self) -> None:
        pbar = tqdm(
            total=sum(1 for value in self.pmls["thickness"].values() if value > 0),
            desc=f"Building PML boundaries [{self.name}]",
            ncols=get_terminal_width() - 1,
            file=sys.stdout,
            disable=not config.sim_config.general["progressbars"],
        )
        for pml_id, thickness in self.pmls["thickness"].items():
            if thickness > 0:
                build_pml(self, pml_id, thickness)
                pbar.update()
        pbar.close()

    def _build_components(self) -> None:
        # Build the model, i.e. set the material properties (ID) for every edge
        # of every Yee cell
        logger.info("")
        pbar = tqdm(
            total=2,
            desc=f"Building Yee cells [{self.name}]",
            ncols=get_terminal_width() - 1,
            file=sys.stdout,
            disable=not config.sim_config.general["progressbars"],
        )
        build_electric_components(self.solid, self.rigidE, self.ID, self)
        pbar.update()
        build_magnetic_components(self.solid, self.rigidH, self.ID, self)
        pbar.update()
        pbar.close()

    def _tm_grid_update(self) -> None:
        if config.get_model_config().mode == "2D TMx":
            self.tmx()
        elif config.get_model_config().mode == "2D TMy":
            self.tmy()
        elif config.get_model_config().mode == "2D TMz":
            self.tmz()

    def _update_voltage_source_materials(self):
        # Process any voltage sources (that have resistance) to create a new
        # material at the source location
        for voltagesource in self.voltagesources:
            voltagesource.create_material(self)

    def _build_materials(self) -> None:
        # Process complete list of materials - calculate update coefficients,
        # store in arrays, and build text list of materials/properties
        materialsdata = process_materials(self)
        materialstable = SingleTable(materialsdata)
        materialstable.outer_border = False
        materialstable.justify_columns[0] = "right"

        logger.info(f"\nMaterials [{self.name}]:")
        logger.info(materialstable.table)

    def _update_positions(
        self, items: Iterable[Union[Source, Rx]], step_size: List[int], step_number: int
    ) -> None:
        if step_size[0] != 0 or step_size[1] != 0 or step_size[2] != 0:
            for item in items:
                if step_number == 0:
                    if (
                        item.xcoord + self.srcsteps[0] * config.sim_config.model_end < 0
                        or item.xcoord + self.srcsteps[0] * config.sim_config.model_end > self.nx
                        or item.ycoord + self.srcsteps[1] * config.sim_config.model_end < 0
                        or item.ycoord + self.srcsteps[1] * config.sim_config.model_end > self.ny
                        or item.zcoord + self.srcsteps[2] * config.sim_config.model_end < 0
                        or item.zcoord + self.srcsteps[2] * config.sim_config.model_end > self.nz
                    ):
                        raise ValueError
                item.xcoord = item.xcoordorigin + step_number * step_size[0]
                item.ycoord = item.ycoordorigin + step_number * step_size[1]
                item.zcoord = item.zcoordorigin + step_number * step_size[2]

    def update_simple_source_positions(self, step: int = 0) -> None:
        try:
            self._update_positions(
                itertools.chain(self.hertziandipoles, self.magneticdipoles), self.srcsteps, step
            )
        except ValueError as e:
            logger.exception("Source(s) will be stepped to a position outside the domain.")
            raise ValueError from e

    def update_receiver_positions(self, step: int = 0) -> None:
        try:
            self._update_positions(self.rxs, self.rxsteps, step)
        except ValueError as e:
            logger.exception("Receiver(s) will be stepped to a position outside the domain.")
            raise ValueError from e

    def within_bounds(self, p):
        if p[0] < 0 or p[0] > self.nx:
            raise ValueError("x")
        if p[1] < 0 or p[1] > self.ny:
            raise ValueError("y")
        if p[2] < 0 or p[2] > self.nz:
            raise ValueError("z")

    def discretise_point(self, p):
        x = round_value(float(p[0]) / self.dx)
        y = round_value(float(p[1]) / self.dy)
        z = round_value(float(p[2]) / self.dz)
        return (x, y, z)

    def round_to_grid(self, p):
        p = self.discretise_point(p)
        p_r = (p[0] * self.dx, p[1] * self.dy, p[2] * self.dz)
        return p_r

    def within_pml(self, p):
        if (
            p[0] < self.pmls["thickness"]["x0"]
            or p[0] > self.nx - self.pmls["thickness"]["xmax"]
            or p[1] < self.pmls["thickness"]["y0"]
            or p[1] > self.ny - self.pmls["thickness"]["ymax"]
            or p[2] < self.pmls["thickness"]["z0"]
            or p[2] > self.nz - self.pmls["thickness"]["zmax"]
        ):
            return True
        else:
            return False

    def get_waveform_by_id(self, waveform_id: str) -> Waveform:
        return next(waveform for waveform in self.waveforms if waveform.ID == waveform_id)

    def initialise_geometry_arrays(self):
        """Initialise an array for volumetric material IDs (solid);
            boolean arrays for specifying whether materials can have dielectric
            smoothing (rigid); and an array for cell edge IDs (ID).
        Solid and ID arrays are initialised to free_space (one);
            rigid arrays to allow dielectric smoothing (zero).
        """
        self.solid = np.ones((self.nx, self.ny, self.nz), dtype=np.uint32)
        self.rigidE = np.zeros((12, self.nx, self.ny, self.nz), dtype=np.int8)
        self.rigidH = np.zeros((6, self.nx, self.ny, self.nz), dtype=np.int8)
        self.ID = np.ones((6, self.nx + 1, self.ny + 1, self.nz + 1), dtype=np.uint32)
        self.IDlookup = {"Ex": 0, "Ey": 1, "Ez": 2, "Hx": 3, "Hy": 4, "Hz": 5}

    def initialise_field_arrays(self):
        """Initialise arrays for the electric and magnetic field components."""
        self.Ex = np.zeros(
            (self.nx + 1, self.ny + 1, self.nz + 1),
            dtype=config.sim_config.dtypes["float_or_double"],
        )
        self.Ey = np.zeros(
            (self.nx + 1, self.ny + 1, self.nz + 1),
            dtype=config.sim_config.dtypes["float_or_double"],
        )
        self.Ez = np.zeros(
            (self.nx + 1, self.ny + 1, self.nz + 1),
            dtype=config.sim_config.dtypes["float_or_double"],
        )
        self.Hx = np.zeros(
            (self.nx + 1, self.ny + 1, self.nz + 1),
            dtype=config.sim_config.dtypes["float_or_double"],
        )
        self.Hy = np.zeros(
            (self.nx + 1, self.ny + 1, self.nz + 1),
            dtype=config.sim_config.dtypes["float_or_double"],
        )
        self.Hz = np.zeros(
            (self.nx + 1, self.ny + 1, self.nz + 1),
            dtype=config.sim_config.dtypes["float_or_double"],
        )

    def initialise_std_update_coeff_arrays(self):
        """Initialise arrays for storing update coefficients."""
        self.updatecoeffsE = np.zeros(
            (len(self.materials), 5), dtype=config.sim_config.dtypes["float_or_double"]
        )
        self.updatecoeffsH = np.zeros(
            (len(self.materials), 5), dtype=config.sim_config.dtypes["float_or_double"]
        )

    def initialise_dispersive_arrays(self):
        """Initialise field arrays when there are dispersive materials present."""
        self.Tx = np.zeros(
            (
                config.get_model_config().materials["maxpoles"],
                self.nx + 1,
                self.ny + 1,
                self.nz + 1,
            ),
            dtype=config.get_model_config().materials["dispersivedtype"],
        )
        self.Ty = np.zeros(
            (
                config.get_model_config().materials["maxpoles"],
                self.nx + 1,
                self.ny + 1,
                self.nz + 1,
            ),
            dtype=config.get_model_config().materials["dispersivedtype"],
        )
        self.Tz = np.zeros(
            (
                config.get_model_config().materials["maxpoles"],
                self.nx + 1,
                self.ny + 1,
                self.nz + 1,
            ),
            dtype=config.get_model_config().materials["dispersivedtype"],
        )

    def initialise_dispersive_update_coeff_array(self):
        """Initialise array for storing update coefficients when there are dispersive
        materials present.
        """
        self.updatecoeffsdispersive = np.zeros(
            (len(self.materials), 3 * config.get_model_config().materials["maxpoles"]),
            dtype=config.get_model_config().materials["dispersivedtype"],
        )

    def reset_fields(self):
        """Clear arrays for field components and PMLs."""
        # Clear arrays for field components
        self.initialise_field_arrays()
        if config.get_model_config().materials["maxpoles"] > 0:
            self.initialise_dispersive_arrays()

        # Clear arrays for fields in PML
        for pml in self.pmls["slabs"]:
            pml.initialise_field_arrays()

    def mem_est_basic(self):
        """Estimates the amount of memory (RAM) required for grid arrays.

        Returns:
            mem_use: int of memory (bytes).
        """

        solidarray = self.nx * self.ny * self.nz * np.dtype(np.uint32).itemsize

        # 12 x rigidE array components + 6 x rigidH array components
        rigidarrays = (12 + 6) * self.nx * self.ny * self.nz * np.dtype(np.int8).itemsize

        # 6 x field arrays + 6 x ID arrays
        fieldarrays = (
            (6 + 6)
            * (self.nx + 1)
            * (self.ny + 1)
            * (self.nz + 1)
            * np.dtype(config.sim_config.dtypes["float_or_double"]).itemsize
        )

        # PML arrays
        pmlarrays = 0
        for k, v in self.pmls["thickness"].items():
            if v > 0:
                if "x" in k:
                    pmlarrays += (v + 1) * self.ny * (self.nz + 1)
                    pmlarrays += (v + 1) * (self.ny + 1) * self.nz
                    pmlarrays += v * self.ny * (self.nz + 1)
                    pmlarrays += v * (self.ny + 1) * self.nz
                elif "y" in k:
                    pmlarrays += self.nx * (v + 1) * (self.nz + 1)
                    pmlarrays += (self.nx + 1) * (v + 1) * self.nz
                    pmlarrays += (self.nx + 1) * v * self.nz
                    pmlarrays += self.nx * v * (self.nz + 1)
                elif "z" in k:
                    pmlarrays += self.nx * (self.ny + 1) * (v + 1)
                    pmlarrays += (self.nx + 1) * self.ny * (v + 1)
                    pmlarrays += (self.nx + 1) * self.ny * v
                    pmlarrays += self.nx * (self.ny + 1) * v

        mem_use = int(fieldarrays + solidarray + rigidarrays + pmlarrays)

        return mem_use

    def mem_est_dispersive(self):
        """Estimates the amount of memory (RAM) required for dispersive grid arrays.

        Returns:
            mem_use: int of memory (bytes).
        """

        mem_use = int(
            3
            * config.get_model_config().materials["maxpoles"]
            * (self.nx + 1)
            * (self.ny + 1)
            * (self.nz + 1)
            * np.dtype(config.get_model_config().materials["dispersivedtype"]).itemsize
        )
        return mem_use

    def mem_est_fractals(self):
        """Estimates the amount of memory (RAM) required to build any objects
            which use the FractalVolume/FractalSurface classes.

        Returns:
            mem_use: int of memory (bytes).
        """

        mem_use = 0

        for vol in self.fractalvolumes:
            mem_use += vol.nx * vol.ny * vol.nz * vol.dtype.itemsize
            for surface in vol.fractalsurfaces:
                surfacedims = surface.get_surface_dims()
                mem_use += surfacedims[0] * surfacedims[1] * surface.dtype.itemsize

        return mem_use

    def tmx(self):
        """Add PEC boundaries to invariant direction in 2D TMx mode.
        N.B. 2D modes are a single cell slice of 3D grid.
        """
        # Ey & Ez components
        self.ID[1, 0, :, :] = 0
        self.ID[1, 1, :, :] = 0
        self.ID[2, 0, :, :] = 0
        self.ID[2, 1, :, :] = 0

    def tmy(self):
        """Add PEC boundaries to invariant direction in 2D TMy mode.
        N.B. 2D modes are a single cell slice of 3D grid.
        """
        # Ex & Ez components
        self.ID[0, :, 0, :] = 0
        self.ID[0, :, 1, :] = 0
        self.ID[2, :, 0, :] = 0
        self.ID[2, :, 1, :] = 0

    def tmz(self):
        """Add PEC boundaries to invariant direction in 2D TMz mode.
        N.B. 2D modes are a single cell slice of 3D grid.
        """
        # Ex & Ey components
        self.ID[0, :, :, 0] = 0
        self.ID[0, :, :, 1] = 0
        self.ID[1, :, :, 0] = 0
        self.ID[1, :, :, 1] = 0

    def calculate_dt(self):
        """Calculate time step at the CFL limit."""
        if config.get_model_config().mode == "2D TMx":
            self.dt = 1 / (
                config.sim_config.em_consts["c"] * np.sqrt((1 / self.dy**2) + (1 / self.dz**2))
            )
        elif config.get_model_config().mode == "2D TMy":
            self.dt = 1 / (
                config.sim_config.em_consts["c"] * np.sqrt((1 / self.dx**2) + (1 / self.dz**2))
            )
        elif config.get_model_config().mode == "2D TMz":
            self.dt = 1 / (
                config.sim_config.em_consts["c"] * np.sqrt((1 / self.dx**2) + (1 / self.dy**2))
            )
        else:
            self.dt = 1 / (
                config.sim_config.em_consts["c"]
                * np.sqrt((1 / self.dx**2) + (1 / self.dy**2) + (1 / self.dz**2))
            )

        # Round down time step to nearest float with precision one less than
        # hardware maximum. Avoids inadvertently exceeding the CFL due to
        # binary representation of floating point number.
        self.dt = round_value(self.dt, decimalplaces=decimal.getcontext().prec - 1)

    def calculate_Ix(self, x: int, y: int, z: int) -> float:
        """Calculates the x-component of current at a grid position.

        Args:
            x: x coordinate of position in grid
            y: y coordinate of position in grid
            z: z coordinate of position in grid
        """

        if y == 0 or z == 0:
            Ix = 0
        else:
            Ix = self.dy * (self.Hy[x, y, z - 1] - self.Hy[x, y, z]) + self.dz * (
                self.Hz[x, y, z] - self.Hz[x, y - 1, z]
            )

        return Ix

    def calculate_Iy(self, x: int, y: int, z: int) -> float:
        """Calculates the y-component of current at a grid position.

        Args:
            x: x coordinate of position in grid
            y: y coordinate of position in grid
            z: z coordinate of position in grid
        """

        if x == 0 or z == 0:
            Iy = 0
        else:
            Iy = self.dx * (self.Hx[x, y, z] - self.Hx[x, y, z - 1]) + self.dz * (
                self.Hz[x - 1, y, z] - self.Hz[x, y, z]
            )

        return Iy

    def calculate_Iz(self, x: int, y: int, z: int) -> float:
        """Calculates the y-component of current at a grid position.

        Args:
            x: x coordinate of position in grid
            y: y coordinate of position in grid
            z: z coordinate of position in grid
        """

        if x == 0 or y == 0:
            Iz = 0
        else:
            Iz = self.dx * (self.Hx[x, y - 1, z] - self.Hx[x, y, z]) + self.dy * (
                self.Hy[x, y, z] - self.Hy[x - 1, y, z]
            )

        return Iz


def dispersion_analysis(G):
    """Analysis of numerical dispersion (Taflove et al, 2005, p112) -
        worse case of maximum frequency and minimum wavelength

    Args:
        G: FDTDGrid class describing a grid in a model.

    Returns:
        results: dict of results from dispersion analysis.
    """

    # deltavp: physical phase velocity error (percentage)
    # N: grid sampling density
    # material: material with maximum permittivity
    # maxfreq: maximum significant frequency
    # error: error message
    results = {"deltavp": None, "N": None, "material": None, "maxfreq": [], "error": ""}

    # Find maximum significant frequency
    if G.waveforms:
        for waveform in G.waveforms:
            if waveform.type in ["sine", "contsine"]:
                results["maxfreq"].append(4 * waveform.freq)

            elif waveform.type == "impulse":
                results["error"] = "impulse waveform used."

            elif waveform.type == "user":
                results["error"] = "user waveform detected."

            else:
                # Time to analyse waveform - 4*pulse_width as using entire
                # time window can result in demanding FFT
                waveform.calculate_coefficients()
                iterations = round_value(4 * waveform.chi / G.dt)
                iterations = min(iterations, G.iterations)
                waveformvalues = np.zeros(G.iterations)
                for iteration in range(G.iterations):
                    waveformvalues[iteration] = waveform.calculate_value(iteration * G.dt, G.dt)

                # Ensure source waveform is not being overly truncated before attempting any FFT
                if np.abs(waveformvalues[-1]) < np.abs(np.amax(waveformvalues)) / 100:
                    # FFT
                    freqs, power = fft_power(waveformvalues, G.dt)
                    # Get frequency for max power
                    freqmaxpower = np.where(np.isclose(power, 0))[0][0]

                    # Set maximum frequency to a threshold drop from maximum power, ignoring DC value
                    try:
                        freqthres = (
                            np.where(
                                power[freqmaxpower:]
                                < -config.get_model_config().numdispersion["highestfreqthres"]
                            )[0][0]
                            + freqmaxpower
                        )
                        results["maxfreq"].append(freqs[freqthres])
                    except ValueError:
                        results["error"] = (
                            "unable to calculate maximum power "
                            + "from waveform, most likely due to "
                            + "undersampling."
                        )

                # Ignore case where someone is using a waveform with zero amplitude, i.e. on a receiver
                elif waveform.amp == 0:
                    pass

                # If waveform is truncated don't do any further analysis
                else:
                    results["error"] = (
                        "waveform does not fit within specified "
                        + "time window and is therefore being truncated."
                    )
    else:
        results["error"] = "no waveform detected."

    if results["maxfreq"]:
        results["maxfreq"] = max(results["maxfreq"])

        # Find minimum wavelength (material with maximum permittivity)
        maxer = 0
        matmaxer = ""
        for x in G.materials:
            if x.se != float("inf"):
                er = x.er
                # If there are dispersive materials calculate the complex
                # relative permittivity at maximum frequency and take the real part
                if x.__class__.__name__ == "DispersiveMaterial":
                    er = x.calculate_er(results["maxfreq"])
                    er = er.real
                if er > maxer:
                    maxer = er
                    matmaxer = x.ID
        results["material"] = next(x for x in G.materials if x.ID == matmaxer)

        # Minimum velocity
        minvelocity = config.c / np.sqrt(maxer)

        # Minimum wavelength
        minwavelength = minvelocity / results["maxfreq"]

        # Maximum spatial step
        if "3D" in config.get_model_config().mode:
            delta = max(G.dx, G.dy, G.dz)
        elif "2D" in config.get_model_config().mode:
            if G.nx == 1:
                delta = max(G.dy, G.dz)
            elif G.ny == 1:
                delta = max(G.dx, G.dz)
            elif G.nz == 1:
                delta = max(G.dx, G.dy)

        # Courant stability factor
        S = (config.c * G.dt) / delta

        # Grid sampling density
        results["N"] = minwavelength / delta

        # Check grid sampling will result in physical wave propagation
        if (
            int(np.floor(results["N"]))
            >= config.get_model_config().numdispersion["mingridsampling"]
        ):
            # Numerical phase velocity
            vp = np.pi / (results["N"] * np.arcsin((1 / S) * np.sin((np.pi * S) / results["N"])))

            # Physical phase velocity error (percentage)
            results["deltavp"] = (((vp * config.c) - config.c) / config.c) * 100

        # Store rounded down value of grid sampling density
        results["N"] = int(np.floor(results["N"]))

    return results
