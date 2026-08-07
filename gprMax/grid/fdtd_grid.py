# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
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
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from terminaltables import AsciiTable
from tqdm import tqdm

from gprMax import config
from gprMax.cython.geometry_primitives import (
    build_edge_x,
    build_edge_y,
    build_edge_z,
    build_magnetic_edge_x,
    build_magnetic_edge_y,
    build_magnetic_edge_z,
)
from gprMax.cython.pml_build import pml_average_er_mr
from gprMax.cython.yee_cell_build import build_electric_components, build_magnetic_components
from gprMax.fractals.fractal_surface import FractalSurface
from gprMax.fractals.fractal_volume import FractalVolume
from gprMax.materials import ListMaterial, Material, PeplinskiSoil, RangeMaterial, process_materials
from gprMax.pml import CFS, InternalPMLSpec, PML, print_pml_info
from gprMax.receivers import Rx
from gprMax.sources import (
    DiscretePlaneWave,
    EigenmodeReceiver,
    EigenmodeSource,
    HertzianDipole,
    MagneticDipole,
    MagneticFrillSource,
    Source,
    TransmissionLine,
    VoltageSource,
)
from gprMax.symmetry_boundaries import (
    build_symmetry_boundary_edges,
    build_symmetry_boundary_edges_dispersive,
    build_symmetry_boundary_edges_dispersive_b,
)
from gprMax.utilities.utilities import fft_power, get_terminal_width, round_value
from gprMax.waveforms import Waveform

logger = logging.getLogger(__name__)


class FDTDGrid:
    """Holds attributes associated with entire grid. A convenient way for
    accessing regularly used parameters.
    """

    IDlookup = {"Ex": 0, "Ey": 1, "Ez": 2, "Hx": 3, "Hy": 4, "Hz": 5}
    pml_type = PML

    def __init__(self):
        self.name = "main_grid"
        self.mem_use = 0

        self.size = np.zeros(3, dtype=np.int64)
        self.dl = np.ones(3, dtype=np.float64)
        self.dt = 0.0

        self.iterations = 0  # Total number of iterations
        self.timewindow = 0.0

        self.srcsteps = np.zeros(3, dtype=np.int32)
        self.rxsteps = np.zeros(3, dtype=np.int32)

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
        self.pmls["internal_specs"] = []
        # Ordered dictionary required so *updating* the PMLs always follows the
        # same order (the order for *building* PMLs does not matter). The order
        # itself does not matter, however, if must be the same from model to
        # model otherwise the numerical precision from adding the PML
        # corrections will be different.
        self.pmls["thickness"] = OrderedDict()
        self.set_pml_thickness(10)

        # PEC/PMC symmetry boundaries, keyed by domain face. Edge dispatch
        # is resolved once after geometry and material IDs are finalised.
        self.symmetry_boundaries: dict = {}
        self.symmetry_boundary_edges: list = []
        self.symmetry_boundary_edges_dispersive: list = []
        self.symmetry_boundary_edges_dispersive_b: list = []

        # Materials used by this grid
        self.materials: List[Material] = []
        self.mixingmodels: List[Union[PeplinskiSoil, RangeMaterial, ListMaterial]] = []
        self.fractalvolumes: List[FractalVolume] = []
        # Thin-wire geometry is registered while parsing the scene, then
        # applied after normal Yee-component averaging has resolved the
        # actual background material of every affected H component.
        self.thinwires: list = []

        # Sources and receivers contained inside this grid
        self.waveforms: List[Waveform] = []
        self.voltagesources: List[VoltageSource] = []
        self.hertziandipoles: List[HertzianDipole] = []
        self.magneticdipoles: List[MagneticDipole] = []
        self.transmissionlines: List[TransmissionLine] = []
        self.magneticfrillsources: List[MagneticFrillSource] = []
        self.discreteplanewaves: List[DiscretePlaneWave] = []
        self.eigenmodeband = None
        self.eigenmodeportdefs = {}
        self.eigenmodeexcitation = None
        self.eigenmodesources: List[EigenmodeSource] = []
        self.eigenmodereceivers: List[EigenmodeReceiver] = []
        self.eigenmodeports = []
        self.rxs: List[Rx] = []
        self.port_monitors = []  # Source-bound S-parameter/impedance outputs
        self.snapshots = []  # List[Snapshot]
        self.ntff_monitors = []  # Time- and frequency-domain NTFF monitors
        # Reusable NTFF surface definitions are registered by user objects,
        # then compiled for the requested formulations after Yee material IDs
        # have been constructed.
        self.ntff_surface_specs = {}
        self.ksir_transform_specs = {}
        self.ntff_transform_specs = {}
        self.ksir_time_requests = []
        self.ksir_frequency_requests = []
        self.ksir_far_field_requests = []
        self.ntff_far_field_requests = []
        self.ntff_time_far_field_requests = []
        self.ksir_antenna_port_specs = {}
        self.ntff_antenna_port_specs = {}
        self.ksir_request_owners = {}
        self.ksir_transform_owners = {}
        self.ntff_request_owners = {}
        self.ntff_transform_owners = {}
        self.ntff_output_writers = []

        self.averagevolumeobjects = True

    @property
    def nx(self) -> int:
        return self.size[0]

    @nx.setter
    def nx(self, value: int):
        self.size[0] = value

    @property
    def ny(self) -> int:
        return self.size[1]

    @ny.setter
    def ny(self, value: int):
        self.size[1] = value

    @property
    def nz(self) -> int:
        return self.size[2]

    @nz.setter
    def nz(self, value: int):
        self.size[2] = value

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

    def set_pml_thickness(self, thickness: Union[int, Tuple[int, int, int, int, int, int]]):
        if isinstance(thickness, int) or len(thickness) == 1:
            for key in PML.boundaryIDs:
                self.pmls["thickness"][key] = int(thickness)
        elif len(thickness) == 6:
            self.pmls["thickness"]["x0"] = int(thickness[0])
            self.pmls["thickness"]["y0"] = int(thickness[1])
            self.pmls["thickness"]["z0"] = int(thickness[2])
            self.pmls["thickness"]["xmax"] = int(thickness[3])
            self.pmls["thickness"]["ymax"] = int(thickness[4])
            self.pmls["thickness"]["zmax"] = int(thickness[5])

    def add_fractal_volume(
        self,
        xs: int,
        xf: int,
        ys: int,
        yf: int,
        zs: int,
        zf: int,
        frac_dim: float,
        seed: Optional[int],
    ) -> FractalVolume:
        volume = FractalVolume(xs, xf, ys, yf, zs, zf, frac_dim, seed)
        self.fractalvolumes.append(volume)
        return volume

    def create_fractal_surface(
        self,
        xs: int,
        xf: int,
        ys: int,
        yf: int,
        zs: int,
        zf: int,
        frac_dim: float,
        seed: Optional[int],
    ) -> FractalSurface:
        return FractalSurface(xs, xf, ys, yf, zs, zf, frac_dim, seed)

    def add_source(self, source: Source):
        if isinstance(source, VoltageSource):
            self.voltagesources.append(source)
        elif isinstance(source, HertzianDipole):
            self.hertziandipoles.append(source)
        elif isinstance(source, MagneticDipole):
            self.magneticdipoles.append(source)
        elif isinstance(source, TransmissionLine):
            self.transmissionlines.append(source)
        elif isinstance(source, MagneticFrillSource):
            self.magneticfrillsources.append(source)
        elif isinstance(source, DiscretePlaneWave):
            self.discreteplanewaves.append(source)
        elif isinstance(source, EigenmodeSource):
            self.eigenmodesources.append(source)
        else:
            raise TypeError(f"Source of type '{type(source)}' is unknown to gprMax")

    def add_receiver(self, receiver: Rx):
        self.rxs.append(receiver)

    def build(self) -> None:
        """Build the grid."""

        # Set default CFS parameter for PMLs if not user provided
        if not self.pmls["cfs"]:
            self.pmls["cfs"] = [CFS()]
        logger.info(print_pml_info(self))
        if not all(value == 0 for value in self.pmls["thickness"].values()):
            self._validate_pml_thickness()
        if self.pmls["internal_specs"] or not all(
            value == 0 for value in self.pmls["thickness"].values()
        ):
            self._build_pmls()
        for snapshot in self.snapshots:  # TODO: Remove if implement parallel build
            snapshot.initialise_snapfields()
        if self.averagevolumeobjects:
            self._build_components()
        self._build_thin_wires()
        self._2d_mode_grid_update()
        self._terminate_pmls_with_pec()
        self._build_symmetry_boundaries()
        self._create_voltage_source_materials()
        self.initialise_field_arrays()
        self.initialise_std_update_coeff_arrays()
        if config.get_model_config().materials["maxpoles"] > 0:
            self.initialise_dispersive_arrays()
            self.initialise_dispersive_update_coeff_array()

        self._build_materials()
        self._apply_thin_wire_update_coefficients()
        self._DPW__source_grid_init()
        self._eigenmode_port_grid_init()

    def _validate_pml_thickness(self) -> None:
        """Check that no PML reaches or crosses the domain midpoint.

        ``PMLThickness.build()`` performs this check when the user supplies
        ``#pml_cells`` explicitly. Grids otherwise retain their default
        10-cell PML on every side, so small domains previously reached grid
        construction without an equivalent check. Running it here covers
        both explicit and default thicknesses.
        """
        thickness = self.pmls["thickness"]
        if (
            2 * thickness["x0"] >= self.nx
            or 2 * thickness["y0"] >= self.ny
            or 2 * thickness["z0"] >= self.nz
            or 2 * thickness["xmax"] >= self.nx
            or 2 * thickness["ymax"] >= self.ny
            or 2 * thickness["zmax"] >= self.nz
        ):
            raise ValueError("PML has too many cells for the domain size")

    def _build_pmls(self) -> None:
        """Construct and calculate material properties of the PMLs."""

        pbar = tqdm(
            total=(
                sum(1 for value in self.pmls["thickness"].values() if value > 0)
                + len(self.pmls["internal_specs"])
            ),
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
                    f"PML {pml.ID}: Average permittivity = {averageer}, Average permeability =" f" {averagemr}"
                )
                pml.calculate_update_coeffs(averageer, averagemr)
                self.pmls["slabs"].append(pml)
                pbar.update()

        for spec in self.pmls["internal_specs"]:
            pml = self._construct_internal_pml(spec)
            averageer, averagemr = self._calculate_average_pml_material_properties(pml)
            logger.debug(
                f"Internal PML {pml.ID}: Average permittivity = {averageer}, "
                f"Average permeability = {averagemr}"
            )
            pml.calculate_update_coeffs(averageer, averagemr)
            self.pmls["slabs"].append(pml)
            pbar.update()
        pbar.close()

    def _new_pml(self, **kwargs) -> PML:
        """Construct a backend-specific PML and perform backend setup."""
        return self._prepare_pml(self.pml_type(self, **kwargs))

    def _prepare_pml(self, pml: PML) -> PML:
        """Backend hook called after a PML instance has been constructed."""
        return pml

    def _construct_pml(self, pml_ID: str, thickness: int) -> PML:
        """Build PML instance of the specified ID, thickness and type.

        Constructs a PML of the specified type and thickness. Properties
        of the PML are set based on the provided identifier.

        Args:
            pml_ID: Identifier of PML slab.
            thickness: Thickness of PML slab in cells.
            pml_type: PML class to construct.
        """
        if pml_ID == "x0":
            pml = self._new_pml(
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
            pml = self._new_pml(
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
            pml = self._new_pml(
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
            pml = self._new_pml(
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
            pml = self._new_pml(
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
            pml = self._new_pml(
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

    def _construct_internal_pml(self, spec: InternalPMLSpec) -> PML:
        """Construct a user-positioned one-axis PML slab."""
        return self._new_pml(
            ID=spec.ID,
            direction=spec.direction,
            xs=spec.xs,
            xf=spec.xf,
            ys=spec.ys,
            yf=spec.yf,
            zs=spec.zs,
            zf=spec.zf,
            internal=True,
            maximum_face=spec.maximum_face,
        )

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

        if pml.internal:
            # Sample the zero-loss entrance plane, not the high-stretch end.
            # A portable slab is required to be a longitudinal extrusion, so
            # this plane represents the material cross-section throughout it.
            if pml.direction == "xminus":
                solid = self.solid[pml.xf - 1, pml.ys : pml.yf, pml.zs : pml.zf]
            elif pml.direction == "xplus":
                solid = self.solid[pml.xs, pml.ys : pml.yf, pml.zs : pml.zf]
            elif pml.direction == "yminus":
                solid = self.solid[pml.xs : pml.xf, pml.yf - 1, pml.zs : pml.zf]
            elif pml.direction == "yplus":
                solid = self.solid[pml.xs : pml.xf, pml.ys, pml.zs : pml.zf]
            elif pml.direction == "zminus":
                solid = self.solid[pml.xs : pml.xf, pml.ys : pml.yf, pml.zf - 1]
            elif pml.direction == "zplus":
                solid = self.solid[pml.xs : pml.xf, pml.ys : pml.yf, pml.zs]
            else:
                raise ValueError(f"Unknown PML direction '{pml.direction}'")
            n1, n2 = solid.shape
        elif pml.ID[0] == "x":
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
        harmonic = config.get_model_config().magnetic_averaging_mode == "harmonic"
        build_magnetic_components(self.solid, self.rigidH, self.ID, self, harmonic)
        pbar.update()
        pbar.close()

    def _thin_wire_material(
        self,
        wire,
        background: Material,
        *,
        role: str,
        radial_axis: Optional[str] = None,
    ) -> Material:
        """Return a reusable wire material for one Yee-component role.

        ``se = inf`` makes the material a PEC when assigned to the wire's E
        edge. A separate material row is used for each of the two surrounding
        H-component orientations because the Mäkinen projection factor is
        orientation dependent on a rectangular mesh. Magnetic properties are
        copied from the already-resolved H background.
        """

        if getattr(background, "thin_wire_axis", None) is not None:
            same_wire = (
                background.thin_wire_axis == wire.wire_axis
                and background.thin_wire_radius == wire.radius
                and background.thin_wire_role == role
                and background.thin_wire_radial_axis == radial_axis
            )
            if same_wire:
                return background
            raise ValueError(
                "Thin-wire magnetic stencils overlap with different axes or radii; "
                "sub-cell wire junctions are not yet supported."
            )

        cache = getattr(self, "_thin_wire_material_cache", None)
        if cache is None:
            cache = {}
            self._thin_wire_material_cache = cache

        key = (
            wire.wire_axis,
            float(wire.radius),
            role,
            radial_axis,
            int(background.numID),
        )
        material = cache.get(key)
        if material is not None:
            return material

        material_id = f"thin_wire_{wire.wire_axis}_{wire.radius:.12g}_{role}_bg" f"{background.numID}"
        material = Material(len(self.materials), material_id)
        material.type = "thin-wire"
        material.averagable = False
        material.er = background.er
        material.se = float("inf")
        material.mr = background.mr
        material.sm = background.sm
        material.thin_wire_axis = wire.wire_axis
        material.thin_wire_radius = float(wire.radius)
        material.thin_wire_role = role
        material.thin_wire_radial_axis = radial_axis
        material.thin_wire_background_numID = int(background.numID)
        material.thin_wire_background_ID = background.ID
        self.materials.append(material)
        cache[key] = material
        return material

    @staticmethod
    def _thin_wire_h_radial_axis(wire_axis: str, component_h: str) -> str:
        """Return the radial derivative direction represented by an H edge."""

        h_axis = component_h[1].lower()
        return next(axis for axis in "xyz" if axis not in (wire_axis, h_axis))

    def _thin_wire_h_targets(self, axis: str, i: int, j: int, k: int):
        """Yield active H components surrounding one thin-wire E edge."""

        targets = {
            "x": (
                ("Hy", i, j, k - 1),
                ("Hy", i, j, k),
                ("Hz", i, j - 1, k),
                ("Hz", i, j, k),
            ),
            "y": (
                ("Hx", i, j, k - 1),
                ("Hx", i, j, k),
                ("Hz", i - 1, j, k),
                ("Hz", i, j, k),
            ),
            "z": (
                ("Hy", i - 1, j, k),
                ("Hy", i, j, k),
                ("Hx", i, j - 1, k),
                ("Hx", i, j, k),
            ),
        }[axis]
        active_ranges = {
            "Hx": ((0, self.nx + 1), (0, self.ny), (0, self.nz)),
            "Hy": ((0, self.nx), (0, self.ny + 1), (0, self.nz)),
            "Hz": ((0, self.nx), (0, self.ny), (0, self.nz + 1)),
        }

        for component, x, y, z in targets:
            ranges = active_ranges[component]
            if all(low <= value < high for value, (low, high) in zip((x, y, z), ranges)):
                yield component, x, y, z

    def _validate_thin_wire_transverse_boundaries(self, wire) -> None:
        """Require PMC symmetry when a wire lies on a transverse wall."""

        axis_index = "xyz".index(wire.wire_axis)
        for index, axis in enumerate("xyz"):
            if index == axis_index:
                continue
            coordinate = int(wire.start[index])
            if coordinate == 0:
                face = f"{axis}0"
            elif coordinate == int(self.size[index]):
                face = f"{axis}max"
            else:
                continue
            if self.symmetry_boundaries.get(face) != "pmc":
                raise ValueError(
                    f"{wire} lies on transverse domain face {face}; a thin wire "
                    "can lie on a transverse wall only when that face is declared "
                    "as a PMC symmetry boundary."
                )

    def _build_thin_wires(self) -> None:
        """Apply registered thin wires after ordinary component averaging."""

        if not self.thinwires:
            return

        electric_builders = {"x": build_edge_x, "y": build_edge_y, "z": build_edge_z}
        magnetic_builders = {
            "Hx": build_magnetic_edge_x,
            "Hy": build_magnetic_edge_y,
            "Hz": build_magnetic_edge_z,
        }
        electric_components = {"x": "Ex", "y": "Ey", "z": "Ez"}
        occupied_e = {}
        occupied_h = {}

        for wire in self.thinwires:
            self._validate_thin_wire_transverse_boundaries(wire)
            component_e = electric_components[wire.wire_axis]
            component_e_index = self.IDlookup[component_e]

            for i, j, k in wire.cells():
                if self.within_pml(np.array((i, j, k), dtype=np.int32)):
                    raise ValueError(
                        f"{wire} enters a PML at grid position {(i, j, k)}; "
                        "thin wires inside PML regions are not supported."
                    )

                e_key = (component_e, i, j, k)
                previous_e = occupied_e.get(e_key)
                signature = (wire.wire_axis, wire.radius)
                if previous_e is not None and previous_e != signature:
                    raise ValueError("Thin-wire electric edges overlap with different axes or radii.")
                occupied_e[e_key] = signature

                h_targets = list(self._thin_wire_h_targets(wire.wire_axis, i, j, k))
                if any(self.within_pml(np.array((x, y, z), dtype=np.int32)) for _, x, y, z in h_targets):
                    raise ValueError(
                        f"{wire} has a surrounding magnetic component inside a PML "
                        f"at grid position {(i, j, k)}; thin-wire stencils cannot "
                        "touch PML regions."
                    )

                background_e = self.materials[int(self.ID[component_e_index, i, j, k])]
                material_e = self._thin_wire_material(wire, background_e, role=component_e)
                electric_builders[wire.wire_axis](i, j, k, material_e.numID, self.rigidE, self.rigidH, self.ID)

                for component_h, x, y, z in h_targets:
                    h_key = (component_h, x, y, z)
                    previous_h = occupied_h.get(h_key)
                    if previous_h is not None and previous_h != signature:
                        raise ValueError(
                            "Thin-wire magnetic stencils overlap with different "
                            "axes or radii; sub-cell wire junctions are not yet supported."
                        )
                    occupied_h[h_key] = signature

                    component_h_index = self.IDlookup[component_h]
                    background_h = self.materials[int(self.ID[component_h_index, x, y, z])]
                    if background_h.ID == "pmc" or background_h.sm == float("inf"):
                        raise ValueError(
                            f"{wire} has a PMC magnetic component in its surrounding " f"stencil at {(x, y, z)}."
                        )
                    radial_axis = self._thin_wire_h_radial_axis(wire.wire_axis, component_h)
                    material_h = self._thin_wire_material(
                        wire,
                        background_h,
                        role=component_h,
                        radial_axis=radial_axis,
                    )
                    magnetic_builders[component_h](x, y, z, material_h.numID, self.rigidH, self.ID)

    def _apply_thin_wire_update_coefficients(self) -> None:
        """Apply the Mäkinen thin-wire projection to magnetic material rows.

        Following equations (8)-(14) of Mäkinen et al. (2002), the stored
        field is the projected Yee-edge average
        ``H_tilde = k_H H``. Multiplying Mäkinen's H update by ``k_H`` and
        using ``k_E = 1 / k_H`` leaves the wire-axis curl coefficient
        unchanged, while the radial coefficient becomes ``F * k_H``. The
        magnetic-source coefficient is also multiplied by ``k_H`` so sources
        such as a co-located magnetic frill deposit the projected field.
        """

        coefficient_columns = {"x": 1, "y": 2, "z": 3}
        for material in self.materials:
            wire_axis = getattr(material, "thin_wire_axis", None)
            radial_axis = getattr(material, "thin_wire_radial_axis", None)
            if wire_axis is None or radial_axis is None:
                continue

            h_axis = material.thin_wire_role[1].lower()
            radial_step = float(self.dl["xyz".index(radial_axis)])
            h_step = float(self.dl["xyz".index(h_axis)])
            factor_f = 2.0 / np.log(radial_step / material.thin_wire_radius)
            factor_kh = (radial_step / h_step) * np.arctan(h_step / radial_step)
            factor_ke = 1.0 / factor_kh
            combined = factor_f * factor_kh

            self.updatecoeffsH[material.numID, coefficient_columns[radial_axis]] *= combined
            self.updatecoeffsH[material.numID, 4] *= factor_kh

            material.thin_wire_h_is_projected = True
            material.thin_wire_factors = {
                "F": float(factor_f),
                "kH": float(factor_kh),
                "kE": float(factor_ke),
                "F_kH": float(combined),
            }

    def _2d_mode_grid_update(self) -> None:
        """Set the invariant-axis boundary materials for a 2D mode."""
        mode = config.get_model_config().mode
        if mode == "2D TMx":
            self.tmx()
        elif mode == "2D TMy":
            self.tmy()
        elif mode == "2D TMz":
            self.tmz()
        elif mode == "2D TEx":
            self.tex()
        elif mode == "2D TEy":
            self.tey()
        elif mode == "2D TEz":
            self.tez()

    def _terminate_pmls_with_pec(self) -> None:
        """Mark the tangential E components at boundary-PML outer faces as PEC.

        The existing field-update bounds already make the outer PML wall a
        PEC termination. Updating the material IDs makes that termination
        explicit for inspection and for edges shared with a PMC symmetry
        face. This must run after geometry and component averaging.
        """
        pml_faces = [face for face, thickness in self.pmls["thickness"].items() if thickness > 0]
        if not pml_faces:
            return

        pec_numid = next(m.numID for m in self.materials if m.ID == "pec")
        for face in pml_faces:
            self._force_pec_tangential_e(face, pec_numid)

    def _build_symmetry_boundaries(self) -> None:
        """Apply PEC faces and resolve the per-iteration PMC edge dispatch."""
        if not self.symmetry_boundaries:
            return

        pec_numid = next(m.numID for m in self.materials if m.ID == "pec")
        for face, boundary_type in self.symmetry_boundaries.items():
            if boundary_type == "pec":
                self._force_pec_tangential_e(face, pec_numid)

        self.symmetry_boundary_edges = build_symmetry_boundary_edges(self)
        self.symmetry_boundary_edges_dispersive = build_symmetry_boundary_edges_dispersive(self)
        self.symmetry_boundary_edges_dispersive_b = build_symmetry_boundary_edges_dispersive_b(self)

    def _force_pec_tangential_e(self, face: str, pec_numid: int) -> None:
        """Force the two tangential E-component IDs on a domain face to PEC."""
        idx_ex, idx_ey, idx_ez = 0, 1, 2

        if face == "x0":
            self.ID[idx_ey, 0, 0 : self.ny, 0 : self.nz + 1] = pec_numid
            self.ID[idx_ez, 0, 0 : self.ny + 1, 0 : self.nz] = pec_numid
        elif face == "xmax":
            self.ID[idx_ey, self.nx, 0 : self.ny, 0 : self.nz + 1] = pec_numid
            self.ID[idx_ez, self.nx, 0 : self.ny + 1, 0 : self.nz] = pec_numid
        elif face == "y0":
            self.ID[idx_ex, 0 : self.nx, 0, 0 : self.nz + 1] = pec_numid
            self.ID[idx_ez, 0 : self.nx + 1, 0, 0 : self.nz] = pec_numid
        elif face == "ymax":
            self.ID[idx_ex, 0 : self.nx, self.ny, 0 : self.nz + 1] = pec_numid
            self.ID[idx_ez, 0 : self.nx + 1, self.ny, 0 : self.nz] = pec_numid
        elif face == "z0":
            self.ID[idx_ex, 0 : self.nx, 0 : self.ny + 1, 0] = pec_numid
            self.ID[idx_ey, 0 : self.nx + 1, 0 : self.ny, 0] = pec_numid
        elif face == "zmax":
            self.ID[idx_ex, 0 : self.nx, 0 : self.ny + 1, self.nz] = pec_numid
            self.ID[idx_ey, 0 : self.nx + 1, 0 : self.ny, self.nz] = pec_numid
        else:
            raise ValueError(f"Unknown symmetry boundary face '{face}'")

    def _create_voltage_source_materials(self):
        """Create materials for voltage sources.

        Process any voltage sources (that have resistance) to create a
        new material at the source location.
        """
        # Process any voltage sources (that have resistance) to create a new
        # material at the source location
        for voltagesource in self.voltagesources:
            voltagesource.create_material(self)

    def _DPW__source_grid_init(self):
        """Create IDs and materials for some discrete plane wave sources.

        Process any DPW sources that need grid information not available during initialization of a DPW
        This is used when axial propagation is used and the DPW needs the grid ID components to have been build first
        """
        # Process any Discrete plane wave sources that are need extra information

        for dpw in self.discreteplanewaves:
            dpw.grid_init(self)

    def _eigenmode_port_grid_init(self):
        """Process eigenmode sources and receivers after Yee IDs have been built."""
        if self.eigenmodeportdefs and self.eigenmodeband is None:
            raise ValueError('Eigenmode ports require exactly one EigenmodeBand.')
        if self.eigenmodeportdefs and self.eigenmodeexcitation is None:
            raise ValueError('Eigenmode ports require exactly one EigenmodeExcitation.')
        source_count = len(self.eigenmodesources)
        if (source_count or self.eigenmodereceivers) and source_count != 1:
            raise ValueError(
                "Eigenmode ports require one and only one eigenmode source on " f"{self.name}; found {source_count}."
            )
        ports = [*self.eigenmodesources, *self.eigenmodereceivers]
        port_indices = [int(port.port_index) for port in ports]
        if any(port_index < 1 for port_index in port_indices):
            raise ValueError("Eigenmode port indices must be one or greater.")
        if len(set(port_indices)) != len(port_indices):
            raise ValueError(
                "Eigenmode source and receiver port indices must be unique; " f"got {port_indices} on {self.name}."
            )
        from gprMax.sources import initialise_eigenmode_ports

        initialise_eigenmode_ports(self)

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

    def update_sources_and_recievers(self):
        """Update position of sources and receivers."""

        # Adjust position of simple sources and receivers if required
        model_num = config.sim_config.current_model
        if any(self.srcsteps != 0):
            self.update_simple_source_positions(model_num)
        if any(self.rxsteps != 0):
            self.update_receiver_positions(model_num)

    def _update_positions(
        self,
        items: Iterable[Union[Source, Rx]],
        step_size: npt.NDArray[np.int32],
        step_number: int,
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
        # `!= 0` (not `> 0`) - a negative step is a valid request to move
        # backward each model; the old `> 0` check silently skipped both
        # the bounds check and the actual repositioning below whenever
        # every component of step_size was <= 0 (e.g. an all-negative
        # step), even though SrcSteps/RxSteps accepted and logged it.
        if any(step_size != 0):
            for item in items:
                # The one-time "won't be stepped outside the grid" check
                # must run on the first model actually processed in this
                # run - step_number == config.sim_config.model_start, not
                # a literal 0. step_number is the ABSOLUTE model index, so
                # with a restart (-i/i=) the first model processed has
                # step_number == model_start (never 0), and a literal-0
                # check would never fire at all on any restarted run.
                # Degrades to the exact original check when there's no
                # restart (model_start defaults to 0).
                if step_number == config.sim_config.model_start:
                    # The last model actually run has index model_end - 1
                    # (models run over range(model_start, model_end)), not
                    # model_end itself - checking one step further than
                    # any real run would reject some valid scans that fit
                    # exactly within the domain boundary.
                    end_coord = item.coord + step_size * (config.sim_config.model_end - 1)
                    self.within_bounds(end_coord)
                # Always reposition (not just on step_number !=
                # model_start): step_number is the absolute model index,
                # so this is correct regardless of restart - for the
                # first model processed (step_number == model_start),
                # this must still run alongside the bounds check above,
                # not instead of it, since a restarted run's first model
                # (model_start != 0) genuinely needs real repositioning,
                # unlike a non-restarted run's model 0 (model_start == 0),
                # where this is a harmless no-op (coordorigin + 0*step).
                item.coord = item.coordorigin + step_number * step_size

    def update_simple_source_positions(self, step: int = 0) -> None:
        """Update the positions of sources in the grid.

        Move hertzian dipole and magnetic dipole sources. Transmission
        line sources and voltage sources will not be moved.

        Args:
            step: Number of steps to move the sources by.

        Raises:
            ValueError: Raised if any of the sources would be stepped
                outside of the grid.
        """
        try:
            self._update_positions(
                itertools.chain(self.hertziandipoles, self.magneticdipoles),
                self.srcsteps,
                step,
            )
        except ValueError as e:
            logger.exception("Source(s) will be stepped to a position outside the domain.")
            raise ValueError from e

    def update_receiver_positions(self, step: int = 0) -> None:
        """Update the positions of receivers in the grid.

        Args:
            step: Number of steps to move the receivers by.

        Raises:
            ValueError: Raised if any of the receivers would be stepped
                outside of the grid.
        """
        try:
            # Internal port receivers must remain on their fixed voltage
            # source; #rx_steps applies only to public receiver commands.
            public_receivers = (rx for rx in self.rxs if not getattr(rx, "source_bound", False))
            self._update_positions(public_receivers, self.rxsteps, step)
        except ValueError as e:
            logger.exception("Receiver(s) will be stepped to a position outside the domain.")
            raise ValueError from e

    def within_bounds(self, p: npt.NDArray[np.int32]) -> bool:
        """Check a point is within the grid.

        Args:
            p: Point to check.

        Returns:
            within_bounds: True if the point is within the grid bounds.

        Raises:
            ValueError: Raised if the point is outside the grid.
        """
        if p[0] < 0 or p[0] > self.nx:
            raise ValueError("x")
        if p[1] < 0 or p[1] > self.ny:
            raise ValueError("y")
        if p[2] < 0 or p[2] > self.nz:
            raise ValueError("z")

        return True

    def discretise_point(self, p: Tuple[float, float, float]) -> Tuple[int, int, int]:
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

    def round_to_grid(self, p: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Round the provided point to the nearest grid cell.

        Args:
            p: Point to round.

        Returns:
            p_r: Rounded point.
        """
        p = self.discretise_point(p)
        p_r = (p[0] * self.dx, p[1] * self.dy, p[2] * self.dz)
        return p_r

    def within_pml(self, p: npt.NDArray[np.int32]) -> bool:
        """Check if the provided point is within a PML.

        Args:
            p: Point to check.

        Returns:
            within_pml: True if the point is within a PML.
        """
        within_boundary_pml = (
            p[0] < self.pmls["thickness"]["x0"]
            or p[0] > self.nx - self.pmls["thickness"]["xmax"]
            or p[1] < self.pmls["thickness"]["y0"]
            or p[1] > self.ny - self.pmls["thickness"]["ymax"]
            or p[2] < self.pmls["thickness"]["z0"]
            or p[2] > self.nz - self.pmls["thickness"]["zmax"]
        )
        if within_boundary_pml:
            return True

        return any(
            spec.xs <= p[0] <= spec.xf
            and spec.ys <= p[1] <= spec.yf
            and spec.zs <= p[2] <= spec.zf
            for spec in self.pmls["internal_specs"]
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

        Solid and ID arrays are initialised to free_space; rigid arrays to
        allow dielectric smoothing (zero).
        """
        free_space_numid = next(m.numID for m in self.materials if m.ID == "free_space")
        self.solid = np.full((self.nx, self.ny, self.nz), free_space_numid, dtype=np.uint32)
        self.rigidE = np.zeros((12, self.nx, self.ny, self.nz), dtype=np.int8)
        self.rigidH = np.zeros((6, self.nx, self.ny, self.nz), dtype=np.int8)
        self.ID = np.full(
            (6, self.nx + 1, self.ny + 1, self.nz + 1),
            free_space_numid,
            dtype=np.uint32,
        )

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
        self.updatecoeffsE = np.zeros((len(self.materials), 5), dtype=config.sim_config.dtypes["float_or_double"])
        self.updatecoeffsH = np.zeros((len(self.materials), 5), dtype=config.sim_config.dtypes["float_or_double"])

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

        mem_use = fieldarrays + solidarray + rigidarrays + pmlarrays

        return mem_use

    def mem_est_dispersive(self):
        """Estimates the amount of memory (RAM) required for dispersive grid arrays.

        Returns:
            mem_use: int of memory (bytes).
        """

        mem_use = (
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
            mem_use += np.prod(vol.size) * vol.dtype.itemsize
            for surface in vol.fractalsurfaces:
                surfacedims = surface.get_surface_dims()
                mem_use += surfacedims[0] * surfacedims[1] * surface.dtype.itemsize

        return mem_use

    def tmx(self):
        """Add PEC boundaries to invariant direction in 2D TMx mode.
        N.B. 2D modes are a single cell slice of 3D grid.
        """
        pec_numid = next(m.numID for m in self.materials if m.ID == "pec")
        # Ey & Ez components
        self.ID[1, 0, :, :] = pec_numid
        self.ID[1, 1, :, :] = pec_numid
        self.ID[2, 0, :, :] = pec_numid
        self.ID[2, 1, :, :] = pec_numid

    def tmy(self):
        """Add PEC boundaries to invariant direction in 2D TMy mode.
        N.B. 2D modes are a single cell slice of 3D grid.
        """
        pec_numid = next(m.numID for m in self.materials if m.ID == "pec")
        # Ex & Ez components
        self.ID[0, :, 0, :] = pec_numid
        self.ID[0, :, 1, :] = pec_numid
        self.ID[2, :, 0, :] = pec_numid
        self.ID[2, :, 1, :] = pec_numid

    def tmz(self):
        """Add PEC boundaries to invariant direction in 2D TMz mode.
        N.B. 2D modes are a single cell slice of 3D grid.
        """
        pec_numid = next(m.numID for m in self.materials if m.ID == "pec")
        # Ex & Ey components
        self.ID[0, :, :, 0] = pec_numid
        self.ID[0, :, :, 1] = pec_numid
        self.ID[1, :, :, 0] = pec_numid
        self.ID[1, :, :, 1] = pec_numid

    def tex(self):
        """Set the invariant-axis boundary materials for 2D TEx mode."""
        pec_numid = next(m.numID for m in self.materials if m.ID == "pec")
        pmc_numid = next(m.numID for m in self.materials if m.ID == "pmc")

        # Ex and the transverse H components are inactive throughout the slice.
        self.ID[0, 0:2, :, :] = pec_numid
        self.ID[4, 0:2, :, :] = pmc_numid
        self.ID[5, 0:2, :, :] = pmc_numid
        # Mark the inactive outer-wall components explicitly.
        self.ID[1:3, (0, 2), :, :] = pec_numid
        self.ID[3, (0, 2), :, :] = pmc_numid

    def tey(self):
        """Set the invariant-axis boundary materials for 2D TEy mode."""
        pec_numid = next(m.numID for m in self.materials if m.ID == "pec")
        pmc_numid = next(m.numID for m in self.materials if m.ID == "pmc")

        self.ID[1, :, 0:2, :] = pec_numid
        self.ID[3, :, 0:2, :] = pmc_numid
        self.ID[5, :, 0:2, :] = pmc_numid
        self.ID[0, :, (0, 2), :] = pec_numid
        self.ID[2, :, (0, 2), :] = pec_numid
        self.ID[4, :, (0, 2), :] = pmc_numid

    def tez(self):
        """Set the invariant-axis boundary materials for 2D TEz mode."""
        pec_numid = next(m.numID for m in self.materials if m.ID == "pec")
        pmc_numid = next(m.numID for m in self.materials if m.ID == "pmc")

        self.ID[2, :, :, 0:2] = pec_numid
        self.ID[3, :, :, 0:2] = pmc_numid
        self.ID[4, :, :, 0:2] = pmc_numid
        self.ID[0, :, :, (0, 2)] = pec_numid
        self.ID[1, :, :, (0, 2)] = pec_numid
        self.ID[5, :, :, (0, 2)] = pmc_numid

    def calculate_dt(self):
        """Calculate time step at the CFL limit."""
        if config.get_model_config().mode in ("2D TMx", "2D TEx"):
            self.dt = 1 / (config.sim_config.em_consts["c"] * np.sqrt((1 / self.dy**2) + (1 / self.dz**2)))
        elif config.get_model_config().mode in ("2D TMy", "2D TEy"):
            self.dt = 1 / (config.sim_config.em_consts["c"] * np.sqrt((1 / self.dx**2) + (1 / self.dz**2)))
        elif config.get_model_config().mode in ("2D TMz", "2D TEz"):
            self.dt = 1 / (config.sim_config.em_consts["c"] * np.sqrt((1 / self.dx**2) + (1 / self.dy**2)))
        else:
            self.dt = 1 / (
                config.sim_config.em_consts["c"] * np.sqrt((1 / self.dx**2) + (1 / self.dy**2) + (1 / self.dz**2))
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
            logger.warning(f"Numerical dispersion analysis [{self.name}] not carried out as {results['error']}")
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
            and np.abs(results["deltavp"]) > config.get_model_config().numdispersion["maxnumericaldisp"]
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
        results = {
            "deltavp": None,
            "N": None,
            "material": None,
            "maxfreq": [],
            "error": "",
        }

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
                        waveformvalues[iteration] = waveform.calculate_value(iteration * self.dt, self.dt)

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
                                    power[freqmaxpower:] < -config.get_model_config().numdispersion["highestfreqthres"]
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
                            "waveform does not fit within specified " + "time window and is therefore being truncated."
                        )
        else:
            results["error"] = "no waveform detected."

        if results["maxfreq"]:
            results["maxfreq"] = max(results["maxfreq"])

            # Find minimum wavelength (material with maximum permittivity)
            maxer = 0
            matmaxer = ""
            for x in self.materials:
                if x.se == float("inf") or x.sm == float("inf"):
                    continue
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
            mode = config.get_model_config().mode
            if "3D" in mode:
                delta = max(self.dx, self.dy, self.dz)
            elif "2D" in mode:
                invariant_axis = mode[-1]
                if invariant_axis == "x":
                    delta = max(self.dy, self.dz)
                elif invariant_axis == "y":
                    delta = max(self.dx, self.dz)
                else:
                    delta = max(self.dx, self.dy)

            # Courant stability factor
            S = (config.c * self.dt) / delta

            # Grid sampling density
            results["N"] = minwavelength / delta

            # Check grid sampling will result in physical wave propagation
            if int(np.floor(results["N"])) >= config.get_model_config().numdispersion["mingridsampling"]:
                # Numerical phase velocity
                vp = np.pi / (results["N"] * np.arcsin((1 / S) * np.sin((np.pi * S) / results["N"])))

                # Physical phase velocity error (percentage)
                results["deltavp"] = (((vp * config.c) - config.c) / config.c) * 100

            # Store rounded down value of grid sampling density
            results["N"] = int(np.floor(results["N"]))

        return results
