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

import logging
import sys
from abc import ABC, abstractmethod
from io import TextIOWrapper
from itertools import chain
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import h5py
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

import gprMax.config as config
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.materials import Material
from gprMax.receivers import Rx
from gprMax.sources import Source
from gprMax.vtkhdf_filehandlers.vtk_image_data import VtkImageData
from gprMax.vtkhdf_filehandlers.vtk_unstructured_grid import VtkUnstructuredGrid
from gprMax.vtkhdf_filehandlers.vtkhdf import VtkCellType, VtkHdfFile

from ._version import __version__
from .cython.geometry_outputs import get_line_properties
from .subgrids.grid import SubGridBaseGrid
from .utilities.utilities import get_terminal_width

logger = logging.getLogger(__name__)


def save_geometry_views(gvs: "List[GeometryView]"):
    """Creates and saves geometryviews.

    Args:
        gvs: list of all GeometryViews.
    """

    logger.info("")
    for i, gv in enumerate(gvs):
        gv.set_filename()
        gv.prep_vtk()
        pbar = tqdm(
            total=gv.nbytes,
            unit="byte",
            unit_scale=True,
            desc=f"Writing geometry view file {i + 1}/{len(gvs)}, {gv.filename.name}",
            ncols=get_terminal_width() - 1,
            file=sys.stdout,
            disable=not config.sim_config.general["progressbars"],
        )
        gv.write_vtk()
        pbar.update(gv.nbytes)
        pbar.close()

    logger.info("")


class GeometryView(ABC):
    """Base class for Geometry Views."""

    FILE_EXTENSION = ".vtkhdf"

    def __init__(
        self,
        xs: int,
        ys: int,
        zs: int,
        xf: int,
        yf: int,
        zf: int,
        dx: int,
        dy: int,
        dz: int,
        filename: str,
        grid: FDTDGrid,
    ):
        """
        Args:
            xs, xf, ys, yf, zs, zf: ints for extent of geometry view in cells.
            dx, dy, dz: ints for spatial discretisation of geometry view in cells.
            filename: string for filename.
            grid: FDTDGrid class describing a grid in a model.
        """

        self.start = np.array([xs, ys, zs], dtype=np.intc)
        self.stop = np.array([xf, yf, zf], dtype=np.intc)
        self.step = np.array([dx, dy, dz], dtype=np.intc)
        self.size = (self.stop - self.start) // self.step

        self.filename = Path(filename)
        self.filenamebase = filename
        self.grid = grid
        self.nbytes = None

        self.material_data = None
        self.materials = None

    # Properties for backwards compatibility
    @property
    def xs(self) -> int:
        return self.start[0]

    @property
    def ys(self) -> int:
        return self.start[1]

    @property
    def zs(self) -> int:
        return self.start[2]

    @property
    def xf(self) -> int:
        return self.stop[0]

    @property
    def yf(self) -> int:
        return self.stop[1]

    @property
    def zf(self) -> int:
        return self.stop[2]

    @property
    def dx(self) -> int:
        return self.step[0]

    @property
    def dy(self) -> int:
        return self.step[1]

    @property
    def dz(self) -> int:
        return self.step[2]

    @property
    def nx(self) -> int:
        return self.size[0]

    @property
    def ny(self) -> int:
        return self.size[1]

    @property
    def nz(self) -> int:
        return self.size[2]

    def set_filename(self):
        """Construct filename from user-supplied name and model number."""
        parts = config.get_model_config().output_file_path.parts
        self.filename = Path(
            *parts[:-1], self.filenamebase + config.get_model_config().appendmodelnumber
        ).with_suffix(self.FILE_EXTENSION)

    @abstractmethod
    def prep_vtk(self):
        pass

    @abstractmethod
    def write_vtk(self):
        pass


class GeometryViewLines(GeometryView):
    """Unstructured grid for a per-cell-edge geometry view."""

    def prep_vtk(self):
        """Prepares data for writing to VTKHDF file."""

        # Sample ID array according to geometry view spatial discretisation
        # Only create a new array if subsampling is required
        if (
            self.grid.ID.shape != (self.xf, self.yf, self.zf)
            or (self.dx, self.dy, self.dz) != (1, 1, 1)
            or (self.xs, self.ys, self.zs) != (0, 0, 0)
        ):
            # Require contiguous array
            ID = np.ascontiguousarray(
                self.grid.ID[
                    :,
                    self.xs : self.xf : self.dx,
                    self.ys : self.yf : self.dy,
                    self.zs : self.zf : self.dz,
                ]
            )
        else:
            # This array is contiguous by design
            ID = self.grid.ID

        x = np.arange(self.nx + 1, dtype=np.float64)
        y = np.arange(self.ny + 1, dtype=np.float64)
        z = np.arange(self.nz + 1, dtype=np.float64)
        coords = np.meshgrid(x, y, z, indexing="ij")
        self.points = np.vstack(list(map(np.ravel, coords))).T
        self.points += self.start
        self.points *= self.step * self.grid.dl

        # Add offset to subgrid geometry to correctly locate within main grid
        if isinstance(self.grid, SubGridBaseGrid):
            offset = [self.grid.i0, self.grid.j0, self.grid.k0]
            self.points += offset * self.grid.dl * self.grid.ratio

        # Each point is the 'source' for 3 lines.
        # NB: Excluding points at the far edge of the geometry as those
        # are the 'source' for no lines
        n_lines = 3 * self.nx * self.ny * self.nz

        self.cell_types = np.full(n_lines, VtkCellType.LINE)
        self.cell_offsets = np.arange(0, 2 * n_lines + 2, 2, dtype=np.intc)

        self.connectivity, self.material_data = get_line_properties(
            n_lines, self.nx, self.ny, self.nz, ID
        )

        assert isinstance(self.connectivity, np.ndarray)
        assert isinstance(self.material_data, np.ndarray)

        # Write information about any PMLs, sources, receivers
        self.metadata = Metadata(self.grid, self, averaged_materials=True, materials_only=True)

        # Number of bytes of data to be written to file
        self.nbytes = (
            self.points.nbytes
            + self.cell_types.nbytes
            + self.connectivity.nbytes
            + self.cell_offsets.nbytes
            + self.material_data.nbytes
        )

    def write_vtk(self):
        """Writes geometry information to a VTKHDF file."""

        # Write the VTK file
        with VtkUnstructuredGrid(
            self.filename,
            self.points,
            self.cell_types,
            self.connectivity,
            self.cell_offsets,
        ) as f:
            f.add_cell_data("Material", self.material_data)
            self.metadata.write_to_vtkhdf(f)


class GeometryViewVoxels(GeometryView):
    """Image data for a per-cell geometry view."""

    def prep_vtk(self):
        """Prepares data for writing to VTKHDF file."""

        # Sample solid array according to geometry view spatial discretisation
        # Only create a new array if subsampling is required
        if (
            self.grid.solid.shape != (self.xf, self.yf, self.zf)
            or (self.dx, self.dy, self.dz) != (1, 1, 1)
            or (self.xs, self.ys, self.zs) != (0, 0, 0)
        ):
            # Require contiguous array
            self.material_data = np.ascontiguousarray(
                self.grid.solid[
                    self.xs : self.xf : self.dx,
                    self.ys : self.yf : self.dy,
                    self.zs : self.zf : self.dz,
                ]
            )
        else:
            # This array is contiguous by design
            self.material_data = self.grid.solid

        if isinstance(self.grid, SubGridBaseGrid):
            self.origin = np.array(
                [
                    (self.grid.i0 * self.grid.dx * self.grid.ratio),
                    (self.grid.j0 * self.grid.dy * self.grid.ratio),
                    (self.grid.k0 * self.grid.dz * self.grid.ratio),
                ]
            )
        else:
            self.origin = self.start * self.grid.dl

        self.spacing = self.step * self.grid.dl

        # Write information about any PMLs, sources, receivers
        self.metadata = Metadata(self.grid, self)

        self.nbytes = self.material_data.nbytes

    def write_vtk(self):
        """Writes geometry information to a VTKHDF file."""

        with VtkImageData(self.filename, self.size, self.origin, self.spacing) as f:
            f.add_cell_data("Material", self.material_data)
            self.metadata.write_to_vtkhdf(f)


class Metadata:
    """Comments can be strings included in the header of XML VTK file, and are
    used to hold extra (gprMax) information about the VTK data.
    """

    def __init__(
        self,
        grid: FDTDGrid,
        gv: GeometryView,
        averaged_materials: bool = False,
        materials_only: bool = False,
    ):
        self.grid = grid
        self.gv = gv
        self.averaged_materials = averaged_materials
        self.materials_only = materials_only

        self.gprmax_version = __version__
        self.dx_dy_dz = self.grid.dl
        self.nx_ny_nz = np.array([self.grid.nx, self.grid.ny, self.grid.nz], dtype=np.intc)

        self.materials = self.materials_comment()

        # Write information on PMLs, sources, and receivers
        if not self.materials_only:
            # Information on PML thickness
            if self.grid.pmls["slabs"]:
                self.pml_thickness = self.pml_gv_comment()
            else:
                self.pml_thickness = None
            srcs = (
                self.grid.hertziandipoles
                + self.grid.magneticdipoles
                + self.grid.voltagesources
                + self.grid.transmissionlines
            )
            if srcs:
                self.source_ids, self.source_positions = self.srcs_rx_gv_comment(srcs)
            else:
                self.source_ids = None
                self.source_positions = None
            if self.grid.rxs:
                self.receiver_ids, self.receiver_positions = self.srcs_rx_gv_comment(self.grid.rxs)
            else:
                self.receiver_ids = None
                self.receiver_positions = None

    def write_to_vtkhdf(self, file_handler: VtkHdfFile):
        file_handler.add_field_data(
            "gprMax_version",
            self.gprmax_version,
            dtype=h5py.string_dtype(),
        )
        file_handler.add_field_data("dx_dy_dz", self.dx_dy_dz)
        file_handler.add_field_data("nx_ny_nz", self.nx_ny_nz)

        file_handler.add_field_data(
            "material_ids",
            self.materials,
            dtype=h5py.string_dtype(),
        )

        if not self.materials_only:
            if self.pml_thickness is not None:
                file_handler.add_field_data("pml_thickness", self.pml_thickness)

            if self.source_ids is not None and self.source_positions is not None:
                file_handler.add_field_data(
                    "source_ids", self.source_ids, dtype=h5py.string_dtype()
                )
                file_handler.add_field_data("sources", self.source_positions)

            if self.receiver_ids is not None and self.receiver_positions is not None:
                file_handler.add_field_data(
                    "receiver_ids", self.receiver_ids, dtype=h5py.string_dtype()
                )
                file_handler.add_field_data("receivers", self.receiver_positions)

    def pml_gv_comment(self) -> List[int]:
        grid = self.grid

        # Only render PMLs if they are in the geometry view
        thickness: Dict[str, int] = grid.pmls["thickness"]
        gv_pml_depth = dict.fromkeys(thickness, 0)

        if self.gv.xs < thickness["x0"]:
            gv_pml_depth["x0"] = thickness["x0"] - self.gv.xs
        if self.gv.ys < thickness["y0"]:
            gv_pml_depth["y0"] = thickness["y0"] - self.gv.ys
        if thickness["z0"] - self.gv.zs > 0:
            gv_pml_depth["z0"] = thickness["z0"] - self.gv.zs
        if self.gv.xf > grid.nx - thickness["xmax"]:
            gv_pml_depth["xmax"] = self.gv.xf - (grid.nx - thickness["xmax"])
        if self.gv.yf > grid.ny - thickness["ymax"]:
            gv_pml_depth["ymax"] = self.gv.yf - (grid.ny - thickness["ymax"])
        if self.gv.zf > grid.nz - thickness["zmax"]:
            gv_pml_depth["zmax"] = self.gv.zf - (grid.nz - thickness["zmax"])

        return list(gv_pml_depth.values())

    def srcs_rx_gv_comment(
        self, srcs: Union[Sequence[Source], List[Rx]]
    ) -> Tuple[List[str], npt.NDArray[np.float32]]:
        """Used to name sources and/or receivers."""
        names: List[str] = []
        positions: npt.NDArray[np.float32] = np.empty((len(srcs), 3))
        for index, src in enumerate(srcs):
            position = src.coord * self.grid.dl
            names.append(src.ID)
            positions[index] = position

        return names, positions

    def dx_dy_dz_comment(self) -> npt.NDArray[np.float64]:
        return self.grid.dl

    def nx_ny_nz_comment(self) -> npt.NDArray[np.intc]:
        return np.array([self.grid.nx, self.grid.ny, self.grid.nz], dtype=np.intc)

    def materials_comment(self) -> List[str]:
        if not self.averaged_materials:
            return [m.ID for m in self.grid.materials if m.type != "dielectric-smoothed"]
        else:
            return [m.ID for m in self.grid.materials]


class GeometryObjects:
    """Geometry objects to be written to file."""

    def __init__(self, xs=None, ys=None, zs=None, xf=None, yf=None, zf=None, basefilename=None):
        """
        Args:
            xs, xf, ys, yf, zs, zf: ints for extent of the volume in cells.
            filename: string for filename.
        """

        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.xf = xf
        self.yf = yf
        self.zf = zf
        self.nx = self.xf - self.xs
        self.ny = self.yf - self.ys
        self.nz = self.zf - self.zs
        self.basefilename = basefilename

        # Set filenames
        parts = config.sim_config.input_file_path.with_suffix("").parts
        self.filename_hdf5 = Path(*parts[:-1], self.basefilename)
        self.filename_hdf5 = self.filename_hdf5.with_suffix(".h5")
        self.filename_materials = Path(*parts[:-1], f"{self.basefilename}_materials")
        self.filename_materials = self.filename_materials.with_suffix(".txt")

        # Sizes of arrays to write necessary to update progress bar
        self.solidsize = (
            (self.nx + 1) * (self.ny + 1) * (self.nz + 1) * np.dtype(np.uint32).itemsize
        )
        self.rigidsize = (
            18 * (self.nx + 1) * (self.ny + 1) * (self.nz + 1) * np.dtype(np.int8).itemsize
        )
        self.IDsize = (
            6 * (self.nx + 1) * (self.ny + 1) * (self.nz + 1) * np.dtype(np.uint32).itemsize
        )
        self.datawritesize = self.solidsize + self.rigidsize + self.IDsize

    def write_hdf5(self, title: str, G: FDTDGrid, pbar: tqdm):
        """Writes a geometry objects file in HDF5 format.

        Args:
            G: FDTDGrid class describing a grid in a model.
            pbar: Progress bar class instance.
        """

        ID = G.ID[:, self.xs : self.xf + 1, self.ys : self.yf + 1, self.zs : self.zf + 1]

        # Get materials present in subset of ID array
        material_ids, inverse_map = np.unique(ID, return_inverse=True)

        # Create map from material ID to 0 - number of materials
        materials_map = {material_id: index for index, material_id in enumerate(material_ids)}

        # Remap ID array to the reduced list of materials
        ID = np.array([materials_map[id] for id in material_ids])[inverse_map].reshape(ID.shape)

        data = G.solid[self.xs : self.xf, self.ys : self.yf, self.zs : self.zf].astype("int16")
        map_materials = np.vectorize(lambda id: materials_map[id])
        data = map_materials(data)

        rigidE = G.rigidE[:, self.xs : self.xf, self.ys : self.yf, self.zs : self.zf]
        rigidH = G.rigidH[:, self.xs : self.xf, self.ys : self.yf, self.zs : self.zf]

        with h5py.File(self.filename_hdf5, "w") as fdata:
            fdata.attrs["gprMax"] = __version__
            fdata.attrs["Title"] = title
            fdata.attrs["dx_dy_dz"] = (G.dx, G.dy, G.dz)

            fdata["/data"] = data
            pbar.update(self.solidsize)

            fdata["/rigidE"] = rigidE
            fdata["/rigidH"] = rigidH
            pbar.update(self.rigidsize)

            fdata["/ID"] = ID
            pbar.update(self.IDsize)

        # Write materials list to a text file
        with open(self.filename_materials, "w") as fmaterials:
            for numID in material_ids:
                self.output_material(G.materials[numID], fmaterials)

    def output_material(self, material: Material, file: TextIOWrapper):
        file.write(
            f"#material: {material.er:g} {material.se:g} "
            f"{material.mr:g} {material.sm:g} {material.ID}\n"
        )
        if hasattr(material, "poles"):
            if "debye" in material.type:
                dispersionstr = "#add_dispersion_debye: " f"{material.poles:g} "
                for pole in range(material.poles):
                    dispersionstr += f"{material.deltaer[pole]:g} " f"{material.tau[pole]:g} "
            elif "lorenz" in material.type:
                dispersionstr = f"#add_dispersion_lorenz: " f"{material.poles:g} "
                for pole in range(material.poles):
                    dispersionstr += (
                        f"{material.deltaer[pole]:g} "
                        f"{material.tau[pole]:g} "
                        f"{material.alpha[pole]:g} "
                    )
            elif "drude" in material.type:
                dispersionstr = f"#add_dispersion_drude: " f"{material.poles:g} "
                for pole in range(material.poles):
                    dispersionstr += f"{material.tau[pole]:g} " f"{material.alpha[pole]:g} "
            dispersionstr += material.ID
            file.write(dispersionstr + "\n")


class MPIGeometryObjects(GeometryObjects):
    # def __init__(self, start, stop, global_size, comm):
    # super().__init__(...)

    def write_hdf5(self, title: str, G: MPIGrid, pbar: tqdm):
        """Writes a geometry objects file in HDF5 format.

        Args:
            G: FDTDGrid class describing a grid in a model.
            pbar: Progress bar class instance.
        """

        # Get neighbours
        self.neighbours = np.full((3, 2), -1, dtype=int)
        self.neighbours[0] = self.comm.Shift(direction=0, disp=1)
        self.neighbours[1] = self.comm.Shift(direction=1, disp=1)
        self.neighbours[2] = self.comm.Shift(direction=2, disp=1)

        # Make subsection of array one larger if no positive neighbour
        slice_stop = np.where(
            self.neighbours[:, 1] == -1,
            self.stop + 1,
            self.stop,
        )
        array_slice = list(map(slice, self.start, slice_stop))

        ID = G.ID[:, array_slice[0], array_slice[1], array_slice[2]]

        # Get materials present in subset of ID
        local_material_num_ids, inverse_map = np.unique(ID, return_inverse=True)
        get_material = np.vectorize(lambda id: G.materials[id])
        local_materials = get_material(local_material_num_ids)

        # Send all materials to the coordinating rank
        materials = self.comm.gather(local_materials, root=0)

        if self.comm.rank == 0:
            # Filter out duplicate materials
            materials = np.fromiter(chain.from_iterable(materials), dtype=Material)
            _, indices = np.unique(materials, return_index=True)

            # We want to sort the materials by the order they appear,
            # not sorted alphabetically by ID (default behaviour using
            # np.unique). This ensures user defined materials are lower
            # indexed than compound materials.
            global_materials = materials[sorted(indices)]

            global_material_ids = [m.ID for m in global_materials]
        else:
            global_material_ids = None

        global_material_ids = self.comm.bcast(global_material_ids, root=0)

        # Create map from material ID (str) to global material numID
        materials_map = {
            material_id: index for index, material_id in enumerate(global_material_ids)
        }

        # Remap ID array to the global material numID
        ID = np.array([materials_map[m.ID] for m in local_materials])[inverse_map].reshape(ID.shape)

        # Other geometry arrays do not have halos, so adjustment to
        # 'stop' is not necessary
        array_slice = list(map(slice, self.start, self.stop))

        data = G.solid[array_slice[0], array_slice[1], array_slice[2]]
        map_local_materials = np.vectorize(lambda id: materials_map[G.materials[id].ID])
        data = map_local_materials(data)

        rigidE = G.rigidE[:, array_slice[0], array_slice[1], array_slice[2]]
        rigidH = G.rigidH[:, array_slice[0], array_slice[1], array_slice[2]]

        start = self.offset
        stop = start + self.size

        with h5py.File(self.filename_hdf5, "w", driver="mpio", comm=self.comm) as fdata:
            fdata.attrs["gprMax"] = __version__
            fdata.attrs["Title"] = title
            fdata.attrs["dx_dy_dz"] = (G.dx, G.dy, G.dz)

            dset = fdata.create_dataset("/data", self.global_size, dtype=data.dtype)
            dset[start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]] = data
            pbar.update(self.solidsize)

            rigid_E_dataset = fdata.create_dataset(
                "/rigidE", (12, *self.global_size), dtype=rigidE.dtype
            )
            rigid_E_dataset[:, start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]] = rigidE

            rigid_H_dataset = fdata.create_dataset(
                "/rigidH", (6, *self.global_size), dtype=rigidH.dtype
            )
            rigid_H_dataset[:, start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]] = rigidH
            pbar.update(self.rigidsize)

            stop = np.where(
                self.neighbours[:, 1] == -1,
                stop + 1,
                stop,
            )

            dset = fdata.create_dataset("/ID", (6, *(self.global_size + 1)), dtype=ID.dtype)
            dset[:, start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]] = ID
            pbar.update(self.IDsize)

        # Write materials list to a text file
        if self.comm.rank == 0:
            with open(self.filename_materials, "w") as materials_file:
                for material in global_materials:
                    self.output_material(material, materials_file)
