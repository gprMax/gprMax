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

from io import TextIOWrapper
from pathlib import Path
from typing import Generic

import h5py
import numpy as np
from mpi4py import MPI
from tqdm import tqdm

from gprMax import config
from gprMax._version import __version__
from gprMax.geometry_outputs.grid_view import GridType, GridView, MPIGridView
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.materials import Material


class GeometryObject(Generic[GridType]):
    """Geometry objects to be written to file."""

    @property
    def GRID_VIEW_TYPE(self) -> type[GridView]:
        return GridView

    def __init__(
        self, grid: GridType, xs: int, ys: int, zs: int, xf: int, yf: int, zf: int, filename: str
    ):
        """
        Args:
            xs, xf, ys, yf, zs, zf: ints for extent of the volume in cells.
            filename: string for filename.
        """
        self.grid_view = self.GRID_VIEW_TYPE(grid, xs, ys, zs, xf, yf, zf)

        # Set filenames
        parts = config.sim_config.input_file_path.with_suffix("").parts
        self.filename = Path(*parts[:-1], filename)
        self.filename_hdf5 = self.filename.with_suffix(".h5")
        self.filename_materials = Path(f"{self.filename}_materials")
        self.filename_materials = self.filename_materials.with_suffix(".txt")

        # Sizes of arrays to write necessary to update progress bar
        self.solidsize = (float)(np.prod(self.grid_view.size) * np.dtype(np.uint32).itemsize)
        self.rigidsize = (float)(18 * np.prod(self.grid_view.size) * np.dtype(np.int8).itemsize)
        self.IDsize = (float)(6 * np.prod(self.grid_view.size + 1) * np.dtype(np.uint32).itemsize)
        self.datawritesize = self.solidsize + self.rigidsize + self.IDsize

    @property
    def grid(self) -> GridType:
        return self.grid_view.grid

    def write_metadata(self, file_handler: h5py.File, title: str):
        file_handler.attrs["gprMax"] = __version__
        file_handler.attrs["Title"] = title
        file_handler.attrs["dx_dy_dz"] = (self.grid.dx, self.grid.dy, self.grid.dz)

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

    def write_hdf5(self, title: str, pbar: tqdm):
        """Writes a geometry objects file in HDF5 format.

        Args:
            G: FDTDGrid class describing a grid in a model.
            pbar: Progress bar class instance.
        """

        self.grid_view.initialise_materials()

        ID = self.grid_view.get_ID()
        data = self.grid_view.get_solid().astype(np.int16)
        rigidE = self.grid_view.get_rigidE()
        rigidH = self.grid_view.get_rigidH()

        ID = self.grid_view.map_to_view_materials(ID)
        data = self.grid_view.map_to_view_materials(data)

        with h5py.File(self.filename_hdf5, "w") as fdata:
            self.write_metadata(fdata, title)

            fdata["/data"] = data
            pbar.update(self.solidsize)

            fdata["/rigidE"] = rigidE
            fdata["/rigidH"] = rigidH
            pbar.update(self.rigidsize)

            fdata["/ID"] = ID
            pbar.update(self.IDsize)

        # Write materials list to a text file
        with open(self.filename_materials, "w") as fmaterials:
            for material in self.grid_view.materials:
                self.output_material(material, fmaterials)


class MPIGeometryObject(GeometryObject[MPIGrid]):
    @property
    def GRID_VIEW_TYPE(self) -> type[MPIGridView]:
        return MPIGridView

    def write_hdf5(self, title: str, pbar: tqdm):
        """Writes a geometry objects file in HDF5 format.

        Args:
            G: FDTDGrid class describing a grid in a model.
            pbar: Progress bar class instance.
        """
        assert isinstance(self.grid_view, self.GRID_VIEW_TYPE)

        self.grid_view.initialise_materials()

        ID = self.grid_view.get_ID()
        data = self.grid_view.get_solid().astype(np.int16)
        rigidE = self.grid_view.get_rigidE()
        rigidH = self.grid_view.get_rigidH()

        ID = self.grid_view.map_to_view_materials(ID)
        data = self.grid_view.map_to_view_materials(data)

        with h5py.File(self.filename_hdf5, "w", driver="mpio", comm=self.grid_view.comm) as fdata:
            self.write_metadata(fdata, title)

            dset_slice = self.grid_view.get_3d_output_slice()

            dset = fdata.create_dataset("/data", self.grid_view.global_size, dtype=data.dtype)
            dset[dset_slice] = data
            pbar.update(self.solidsize)

            rigid_E_dataset = fdata.create_dataset(
                "/rigidE", (12, *self.grid_view.global_size), dtype=rigidE.dtype
            )
            rigid_E_dataset[:, dset_slice[0], dset_slice[1], dset_slice[2]] = rigidE

            rigid_H_dataset = fdata.create_dataset(
                "/rigidH", (6, *self.grid_view.global_size), dtype=rigidH.dtype
            )
            rigid_H_dataset[:, dset_slice[0], dset_slice[1], dset_slice[2]] = rigidH
            pbar.update(self.rigidsize)

            dset_slice = self.grid_view.get_3d_output_slice(upper_bound_exclusive=False)

            dset = fdata.create_dataset(
                "/ID", (6, *(self.grid_view.global_size + 1)), dtype=ID.dtype
            )
            dset[:, dset_slice[0], dset_slice[1], dset_slice[2]] = ID
            pbar.update(self.IDsize)

        # Write materials list to a text file
        if self.grid_view.materials is not None:
            with open(self.filename_materials, "w") as materials_file:
                for material in self.grid_view.materials:
                    self.output_material(material, materials_file)
