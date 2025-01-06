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
from typing import Any, Iterable, List, Tuple, Union

import numpy as np
import numpy.typing as npt
from terminaltables import AsciiTable
from tqdm import tqdm
from typing_extensions import TypeVar

from gprMax import config
from gprMax.cython.pml_build import pml_average_er_mr
from gprMax.cython.yee_cell_build import build_electric_components, build_magnetic_components
from gprMax.fractals import FractalVolume
from gprMax.materials import ListMaterial, Material, PeplinskiSoil, RangeMaterial, process_materials
from gprMax.pml import CFS, PML, print_pml_info
from gprMax.receivers import Rx
from gprMax.snapshots import Snapshot
from gprMax.sources import HertzianDipole, MagneticDipole, Source, TransmissionLine, VoltageSource
from gprMax.utilities.utilities import fft_power, get_terminal_width, round_value
from gprMax.waveforms import Waveform

logger = logging.getLogger(__name__)


class FDTDGrid:
    """Holds attributes associated with entire grid. A convenient way for
    accessing regularly used parameters.
    """

    IDlookup = {"Ex": 0, "Ey": 1, "Ez": 2, "Hx": 3, "Hy": 4, "Hz": 5}

    def __init__(self):
        self.name = "main_grid"
        self.mem_use = 0

        self.nx = 0
        self.ny = 0
        self.nz = 0
        self.dl = np.ones(3, dtype=np.float64)
        self.dt = 0.0

        self.iterations = 0  # Total number of iterations
        self.timewindow = 0.0

        # Field Arrays
        self.Ex: npt.NDArray[np.float32]
        self.Ey: npt.NDArray[np.float32]
        self.Ez: npt.NDArray[np.float32]
        self.Hx: npt.NDArray[np.float32]
        self.Hy: npt.NDArray[np.float32]
        self.Hz: npt.NDArray[np.float32]

        # Dispersive Arrays
        self.Tx: npt.NDArray[np.float32]
        self.Ty: npt.NDArray[np.float32]
        self.Tz: npt.NDArray[np.float32]

        # Geometry Arrays
        self.solid: npt.NDArray[np.uint32]
        self.rigidE: npt.NDArray[np.int8]
        self.rigidH: npt.NDArray[np.int8]
        self.ID: npt.NDArray[np.uint32]

        # Update Coefficient Arrays
        self.updatecoeffsE: npt.NDArray[np.float32]
        self.updatecoeffsH: npt.NDArray[np.float32]
        self.updatecoeffsdispersive: npt.NDArray[np.float32]

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

        # Materials used by this grid
        self.materials: List[Material] = []
        self.mixingmodels: List[Union[PeplinskiSoil, RangeMaterial, ListMaterial]] = []
        self.fractalvolumes: List[FractalVolume] = []

        # Sources and receivers contained inside this grid
        self.waveforms: List[Waveform] = []
        self.voltagesources: List[VoltageSource] = []
        self.hertziandipoles: List[HertzianDipole] = []
        self.magneticdipoles: List[MagneticDipole] = []
        self.transmissionlines: List[TransmissionLine] = []
        self.rxs: List[Rx] = []
        self.snapshots: List[Snapshot] = []

        self.averagevolumeobjects = True

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
        """Build the grid."""

        # Set default CFS parameter for PMLs if not user provided
        if not self.pmls["cfs"]:
            self.pmls["cfs"] = [CFS()]
        logger.info(print_pml_info(self))
        if not all(value == 0 for value in self.pmls["thickness"].values()):
            self._build_pmls()
        for snapshot in self.snapshots:  # TODO: Remove if implement parallel build
            snapshot.initialise_snapfields()
        if self.averagevolumeobjects:
            self._build_components()
        self._tm_grid_update()
        self._create_voltage_source_materials()
        self.initialise_field_arrays()
        self.initialise_std_update_coeff_arrays()
        if config.get_model_config().materials["maxpoles"] > 0:
            self.initialise_dispersive_arrays()
            self.initialise_dispersive_update_coeff_array()
        self._build_materials()

    def _build_pmls(self) -> None:
        """Construct and calculate material properties of the PMLs."""

        pbar = tqdm(
            total=sum(1 for value in self.pmls["thickness"].values() if value > 0),
            desc=f"Building PML boundaries [{self.name}]",
            ncols=get_terminal_width() - 1,
            file=sys.stdout,
            disable=not config.sim_config.general["progressbars"],
        )
        for pml_id, thickness in self.pmls["thickness"].items():
            if thickness > 0:
                pml = self._construct_pml(pml_id, thickness)
                averageer, averagemr = self._calculate_average_pml_material_properties(pml)
                logger.debug(
                    f"PML {pml.ID}: Average permittivity = {averageer}, Average permeability ="
                    f" {averagemr}"
                )
                pml.calculate_update_coeffs(averageer, averagemr)
                self.pmls["slabs"].append(pml)
                pbar.update()
        pbar.close()

    PmlType = TypeVar("PmlType", bound=PML)

    def _construct_pml(self, pml_ID: str, thickness: int, pml_type: type[PmlType] = PML) -> PmlType:
        """Build PML instance of the specified ID, thickness and type.

        Constructs a PML of the specified type and thickness. Properties
        of the PML are set based on the provided identifier.

        Args:
            pml_ID: Identifier of PML slab.
            thickness: Thickness of PML slab in cells.
            pml_type: PML class to construct.
        """
        if pml_ID == "x0":
            pml = pml_type(
                self,
                ID=pml_ID,
                direction="xminus",
                xs=0,
                xf=thickness,
                ys=0,
                yf=self.ny,
                zs=0,
                zf=self.nz,
            )
        elif pml_ID == "xmax":
            pml = pml_type(
                self,
                ID=pml_ID,
                direction="xplus",
                xs=self.nx - thickness,
                xf=self.nx,
                ys=0,
                yf=self.ny,
                zs=0,
                zf=self.nz,
            )
        elif pml_ID == "y0":
            pml = pml_type(
                self,
                ID=pml_ID,
                direction="yminus",
                xs=0,
                xf=self.nx,
                ys=0,
                yf=thickness,
                zs=0,
                zf=self.nz,
            )
        elif pml_ID == "ymax":
            pml = pml_type(
                self,
                ID=pml_ID,
                direction="yplus",
                xs=0,
                xf=self.nx,
                ys=self.ny - thickness,
                yf=self.ny,
                zs=0,
                zf=self.nz,
            )
        elif pml_ID == "z0":
            pml = pml_type(
                self,
                ID=pml_ID,
                direction="zminus",
                xs=0,
                xf=self.nx,
                ys=0,
                yf=self.ny,
                zs=0,
                zf=thickness,
            )
        elif pml_ID == "zmax":
            pml = pml_type(
                self,
                ID=pml_ID,
                direction="zplus",
                xs=0,
                xf=self.nx,
                ys=0,
                yf=self.ny,
                zs=self.nz - thickness,
                zf=self.nz,
            )
        else:
            raise ValueError(f"Unknown PML ID '{pml_ID}'")

        return pml

    def _calculate_average_pml_material_properties(self, pml: PML) -> Tuple[float, float]:
        """Calculate average material properties for the provided PML.

        Args:
            pml: PML to calculate the properties of.

        Returns:
            averageer, averagemr: Average permittivity and permeability
                in the PML slab.
        """
        # Arrays to hold values of permittivity and permeability (avoids accessing
        # Material class in Cython.)
        ers = np.zeros(len(self.materials))
        mrs = np.zeros(len(self.materials))

        for i, m in enumerate(self.materials):
            ers[i] = m.er
            mrs[i] = m.mr

        if pml.ID[0] == "x":
            n1 = self.ny
            n2 = self.nz
            solid = self.solid[pml.xs, :, :]
        elif pml.ID[0] == "y":
            n1 = self.nx
            n2 = self.nz
            solid = self.solid[:, pml.ys, :]
        elif pml.ID[0] == "z":
            n1 = self.nx
            n2 = self.ny
            solid = self.solid[:, :, pml.zs]
        else:
            raise ValueError(f"Unknown PML ID '{pml.ID}'")

        return pml_average_er_mr(n1, n2, config.get_model_config().ompthreads, solid, ers, mrs)

    def _build_components(self) -> None:
        """Build electric and magnetic components of the grid.

        Set the material properties (stored in the ID array) for every
        edge of every Yee cell.
        """
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
        """Add PEC boundaries to invariant if in 2D mode."""
        if config.get_model_config().mode == "2D TMx":
            self.tmx()
        elif config.get_model_config().mode == "2D TMy":
            self.tmy()
        elif config.get_model_config().mode == "2D TMz":
            self.tmz()

    def _create_voltage_source_materials(self):
        """Create materials for voltage sources.

        Process any voltage sources (that have resistance) to create a
        new material at the source location.
        """
        # Process any voltage sources (that have resistance) to create a new
        # material at the source location
        for voltagesource in self.voltagesources:
            voltagesource.create_material(self)

    def _build_materials(self) -> None:
        """Calculate properties of materials in the grid.

        Log a summary of the material properties.
        """
        materialsdata = process_materials(self)
        # materialstable = SingleTable(materialsdata)
        materialstable = AsciiTable(materialsdata)
        materialstable.outer_border = False
        materialstable.justify_columns[0] = "right"

        logger.info("")
        logger.info(f"Materials [{self.name}]:\n{materialstable.table}\n")

    def _update_positions(
        self, items: Iterable[Union[Source, Rx]], step_size: npt.NDArray[np.int32], step_number: int
    ) -> None:
        """Update the grid positions of the provided items.

        Args:
            items: Sources and receivers to update.
            step_size: Number of grid cells to move the items each step.
            step_number: Number of steps to move the items by.

        Raises:
            ValueError: Raised if any of the items would be stepped
                outside of the grid.
        """
        if step_size[0] != 0 or step_size[1] != 0 or step_size[2] != 0:
            for item in items:
                if step_number == 0:
                    if (
                        item.xcoord + step_size[0] * config.sim_config.model_end < 0
                        or item.xcoord + step_size[0] * config.sim_config.model_end > self.nx
                        or item.ycoord + step_size[1] * config.sim_config.model_end < 0
                        or item.ycoord + step_size[1] * config.sim_config.model_end > self.ny
                        or item.zcoord + step_size[2] * config.sim_config.model_end < 0
                        or item.zcoord + step_size[2] * config.sim_config.model_end > self.nz
                    ):
                        raise ValueError
                item.coord = item.coordorigin + step_number * step_size

    def update_simple_source_positions(
        self, step_size: npt.NDArray[np.int32], step: int = 0
    ) -> None:
        """Update the positions of sources in the grid.

        Move hertzian dipole and magnetic dipole sources. Transmission
        line sources and voltage sources will not be moved.

        Args:
            step_size: Number of grid cells to move the sources each
                step.
            step: Number of steps to move the sources by.

        Raises:
            ValueError: Raised if any of the sources would be stepped
                outside of the grid.
        """
        try:
            self._update_positions(
                itertools.chain(self.hertziandipoles, self.magneticdipoles), step_size, step
            )
        except ValueError as e:
            logger.exception("Source(s) will be stepped to a position outside the domain.")
            raise ValueError from e

    def update_receiver_positions(self, step_size: npt.NDArray[np.int32], step: int = 0) -> None:
        """Update the positions of receivers in the grid.

        Args:
            step_size: Number of grid cells to move the receivers each
                step.
            step: Number of steps to move the receivers by.

        Raises:
            ValueError: Raised if any of the receivers would be stepped
                outside of the grid.
        """
        try:
            self._update_positions(self.rxs, step_size, step)
        except ValueError as e:
            logger.exception("Receiver(s) will be stepped to a position outside the domain.")
            raise ValueError from e

    IntPoint = Tuple[int, int, int]
    FloatPoint = Tuple[float, float, float]

    def within_bounds(self, p: IntPoint):
        """Check a point is within the grid.

        Args:
            p: Point to check.

        Raises:
            ValueError: Raised if the point is outside the grid.
        """
        if p[0] < 0 or p[0] > self.nx:
            raise ValueError("x")
        if p[1] < 0 or p[1] > self.ny:
            raise ValueError("y")
        if p[2] < 0 or p[2] > self.nz:
            raise ValueError("z")

    def discretise_point(self, p: FloatPoint) -> IntPoint:
        """Calculate the nearest grid cell to the given point.

        Args:
            p: Point to discretise.

        Returns:
            x, y, z: Discretised point.
        """
        x = round_value(float(p[0]) / self.dx)
        y = round_value(float(p[1]) / self.dy)
        z = round_value(float(p[2]) / self.dz)
        return (x, y, z)

    def round_to_grid(self, p: FloatPoint) -> FloatPoint:
        """Round the provided point to the nearest grid cell.

        Args:
            p: Point to round.

        Returns:
            p_r: Rounded point.
        """
        p = self.discretise_point(p)
        p_r = (p[0] * self.dx, p[1] * self.dy, p[2] * self.dz)
        return p_r

    def within_pml(self, p: IntPoint) -> bool:
        """Check if the provided point is within a PML.

        Args:
            p: Point to check.

        Returns:
            within_pml: True if the point is within a PML.
        """
        return (
            p[0] < self.pmls["thickness"]["x0"]
            or p[0] > self.nx - self.pmls["thickness"]["xmax"]
            or p[1] < self.pmls["thickness"]["y0"]
            or p[1] > self.ny - self.pmls["thickness"]["ymax"]
            or p[2] < self.pmls["thickness"]["z0"]
            or p[2] > self.nz - self.pmls["thickness"]["zmax"]
        )

    def get_waveform_by_id(self, waveform_id: str) -> Waveform:
        """Get waveform with the specified ID.

        Args:
            waveform_id: ID of the waveform.

        Returns:
            waveform: Requested waveform
        """
        return next(waveform for waveform in self.waveforms if waveform.ID == waveform_id)

    def initialise_geometry_arrays(self):
        """Initialise arrays to store geometry properties.

        Initialise an array for volumetric material IDs (solid); boolean
        arrays for specifying whether materials can have dielectric
        smoothing (rigid); and an array for cell edge IDs (ID).

        Solid and ID arrays are initialised to free_space (one); rigid
        arrays to allow dielectric smoothing (zero).
        """
        self.solid = np.ones((self.nx, self.ny, self.nz), dtype=np.uint32)
        self.rigidE = np.zeros((12, self.nx, self.ny, self.nz), dtype=np.int8)
        self.rigidH = np.zeros((6, self.nx, self.ny, self.nz), dtype=np.int8)
        self.ID = np.ones((6, self.nx + 1, self.ny + 1, self.nz + 1), dtype=np.uint32)

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
        """Calculate the memory required to build fractal objects.

        Estimates the amount of memory (RAM) required to build any
        objects which use the FractalVolume/FractalSurface classes.

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

    def dispersion_analysis(self, iterations: int):
        """Check to see if numerical dispersion might be a problem.

        Raises:
            ValueError: Raised if a problem is encountered.
        """
        results = self._dispersion_analysis(iterations)
        if results["error"]:
            logger.warning(
                f"Numerical dispersion analysis [{self.name}] not carried out as {results['error']}"
            )
        elif results["N"] < config.get_model_config().numdispersion["mingridsampling"]:
            logger.exception(
                f"\nNon-physical wave propagation in [{self.name}] "
                f"detected. Material '{results['material'].ID}' "
                f"has wavelength sampled by {results['N']} cells, "
                "less than required minimum for physical wave "
                "propagation. Maximum significant frequency "
                f"estimated as {results['maxfreq']:g}Hz"
            )
            raise ValueError
        elif (
            results["deltavp"]
            and np.abs(results["deltavp"])
            > config.get_model_config().numdispersion["maxnumericaldisp"]
        ):
            logger.warning(
                f"[{self.name}] has potentially significant "
                "numerical dispersion. Estimated largest physical "
                f"phase-velocity error is {results['deltavp']:.2f}% "
                f"in material '{results['material'].ID}' whose "
                f"wavelength sampled by {results['N']} cells. "
                "Maximum significant frequency estimated as "
                f"{results['maxfreq']:g}Hz\n"
            )
        elif results["deltavp"]:
            logger.info(
                f"Numerical dispersion analysis [{self.name}]: "
                "estimated largest physical phase-velocity error is "
                f"{results['deltavp']:.2f}% in material '{results['material'].ID}' "
                f"whose wavelength sampled by {results['N']} cells. "
                "Maximum significant frequency estimated as "
                f"{results['maxfreq']:g}Hz\n"
            )

    def _dispersion_analysis(self, iterations: int) -> dict[str, Any]:
        """Run dispersion analysis.

        Analysis of numerical dispersion (Taflove et al, 2005, p112) -
        worse case of maximum frequency and minimum wavelength.

        Args:
            iterations: Number of iterations the model will run for.

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
        if self.waveforms:
            for waveform in self.waveforms:
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
                    # TODO: Check max_iterations should be calculated (original code didn't go on to use it)
                    max_iterations = round_value(4 * waveform.chi / self.dt)
                    iterations = min(iterations, max_iterations)
                    waveformvalues = np.zeros(iterations)
                    for iteration in range(iterations):
                        waveformvalues[iteration] = waveform.calculate_value(
                            iteration * self.dt, self.dt
                        )

                    # Ensure source waveform is not being overly truncated before attempting any FFT
                    if np.abs(waveformvalues[-1]) < np.abs(np.amax(waveformvalues)) / 100:
                        # FFT
                        freqs, power = fft_power(waveformvalues, self.dt)
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
            for x in self.materials:
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
            results["material"] = next(x for x in self.materials if x.ID == matmaxer)

            # Minimum velocity
            minvelocity = config.c / np.sqrt(maxer)

            # Minimum wavelength
            minwavelength = minvelocity / results["maxfreq"]

            # Maximum spatial step
            if "3D" in config.get_model_config().mode:
                delta = max(self.dx, self.dy, self.dz)
            elif "2D" in config.get_model_config().mode:
                if self.nx == 1:
                    delta = max(self.dy, self.dz)
                elif self.ny == 1:
                    delta = max(self.dx, self.dz)
                elif self.nz == 1:
                    delta = max(self.dx, self.dy)

            # Courant stability factor
            S = (config.c * self.dt) / delta

            # Grid sampling density
            results["N"] = minwavelength / delta

            # Check grid sampling will result in physical wave propagation
            if (
                int(np.floor(results["N"]))
                >= config.get_model_config().numdispersion["mingridsampling"]
            ):
                # Numerical phase velocity
                vp = np.pi / (
                    results["N"] * np.arcsin((1 / S) * np.sin((np.pi * S) / results["N"]))
                )

                # Physical phase velocity error (percentage)
                results["deltavp"] = (((vp * config.c) - config.c) / config.c) * 100

            # Store rounded down value of grid sampling density
            results["N"] = int(np.floor(results["N"]))

        return results
