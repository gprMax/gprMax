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

import json
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import h5py
import numpy as np
from tqdm import tqdm

import gprMax.config as config
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.vtkhdf_filehandlers.vtk_image_data import VtkImageData
from gprMax.vtkhdf_filehandlers.vtk_unstructured_grid import VtkUnstructuredGrid
from gprMax.vtkhdf_filehandlers.vtkhdf import VtkCellType

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
        self.size = self.stop - self.start

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
        comments = Comments(self.grid, self)
        comments.averaged_materials = True
        comments.materials_only = True
        info = comments.get_gprmax_info()
        comments = json.dumps(info)

        self.materials = comments

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
        comments = Comments(self.grid, self)
        info = comments.get_gprmax_info()
        comments = json.dumps(info)

        self.nbytes = self.material_data.nbytes

        self.materials = comments

    def write_vtk(self):
        """Writes geometry information to a VTKHDF file."""

        with VtkImageData(self.filename, self.size, self.origin, self.spacing) as f:
            f.add_cell_data("Material", self.material_data)


class Comments:
    """Comments can be strings included in the header of XML VTK file, and are
    used to hold extra (gprMax) information about the VTK data.
    """

    def __init__(self, grid, gv):
        self.grid = grid
        self.gv = gv
        self.averaged_materials = False
        self.materials_only = False

    def get_gprmax_info(self):
        """Returns gprMax specific information relating material, source,
        and receiver names to numeric identifiers.
        """

        # Comments for Paraview macro
        comments = {
            "gprMax_version": __version__,
            "dx_dy_dz": self.dx_dy_dz_comment(),
            "nx_ny_nz": self.nx_ny_nz_comment(),
            "Materials": self.materials_comment(),
        }  # Write the name and numeric ID for each material

        # Write information on PMLs, sources, and receivers
        if not self.materials_only:
            # Information on PML thickness
            if self.grid.pmls["slabs"]:
                comments["PMLthickness"] = self.pml_gv_comment()
            srcs = (
                self.grid.hertziandipoles
                + self.grid.magneticdipoles
                + self.grid.voltagesources
                + self.grid.transmissionlines
            )
            if srcs:
                comments["Sources"] = self.srcs_rx_gv_comment(srcs)
            if self.grid.rxs:
                comments["Receivers"] = self.srcs_rx_gv_comment(self.grid.rxs)

        return comments

    def pml_gv_comment(self):
        grid = self.grid

        # Only render PMLs if they are in the geometry view
        pmlstorender = dict.fromkeys(grid.pmls["thickness"], 0)

        # Casting to int required as json does not handle numpy types
        if grid.pmls["thickness"]["x0"] - self.gv.xs > 0:
            pmlstorender["x0"] = int(grid.pmls["thickness"]["x0"] - self.gv.xs)
        if grid.pmls["thickness"]["y0"] - self.gv.ys > 0:
            pmlstorender["y0"] = int(grid.pmls["thickness"]["y0"] - self.gv.ys)
        if grid.pmls["thickness"]["z0"] - self.gv.zs > 0:
            pmlstorender["z0"] = int(grid.pmls["thickness"]["z0"] - self.gv.zs)
        if self.gv.xf > grid.nx - grid.pmls["thickness"]["xmax"]:
            pmlstorender["xmax"] = int(self.gv.xf - (grid.nx - grid.pmls["thickness"]["xmax"]))
        if self.gv.yf > grid.ny - grid.pmls["thickness"]["ymax"]:
            pmlstorender["ymax"] = int(self.gv.yf - (grid.ny - grid.pmls["thickness"]["ymax"]))
        if self.gv.zf > grid.nz - grid.pmls["thickness"]["zmax"]:
            pmlstorender["zmax"] = int(self.gv.zf - (grid.nz - grid.pmls["thickness"]["zmax"]))

        return list(pmlstorender.values())

    def srcs_rx_gv_comment(self, srcs):
        """Used to name sources and/or receivers."""
        sc = []
        for src in srcs:
            p = (src.xcoord * self.grid.dx, src.ycoord * self.grid.dy, src.zcoord * self.grid.dz)
            p = list(map(float, p))

            s = {"name": src.ID, "position": p}
            sc.append(s)

        return sc

    def dx_dy_dz_comment(self):
        return list(map(float, [self.grid.dx, self.grid.dy, self.grid.dz]))

    def nx_ny_nz_comment(self):
        return list(map(int, [self.grid.nx, self.grid.ny, self.grid.nz]))

    def materials_comment(self):
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

        with h5py.File(self.filename_hdf5, "w") as fdata:
            fdata.attrs["gprMax"] = __version__
            fdata.attrs["Title"] = title
            fdata.attrs["dx_dy_dz"] = (G.dx, G.dy, G.dz)

            # Get minimum and maximum integers of materials in geometry objects volume
            minmat = np.amin(
                G.ID[:, self.xs : self.xf + 1, self.ys : self.yf + 1, self.zs : self.zf + 1]
            )
            maxmat = np.amax(
                G.ID[:, self.xs : self.xf + 1, self.ys : self.yf + 1, self.zs : self.zf + 1]
            )
            fdata["/data"] = (
                G.solid[self.xs : self.xf + 1, self.ys : self.yf + 1, self.zs : self.zf + 1].astype(
                    "int16"
                )
                - minmat
            )
            pbar.update(self.solidsize)
            fdata["/rigidE"] = G.rigidE[
                :, self.xs : self.xf + 1, self.ys : self.yf + 1, self.zs : self.zf + 1
            ]
            fdata["/rigidH"] = G.rigidH[
                :, self.xs : self.xf + 1, self.ys : self.yf + 1, self.zs : self.zf + 1
            ]
            pbar.update(self.rigidsize)
            fdata["/ID"] = (
                G.ID[:, self.xs : self.xf + 1, self.ys : self.yf + 1, self.zs : self.zf + 1]
                - minmat
            )
            pbar.update(self.IDsize)

        # Write materials list to a text file
        # This includes all materials in range whether used in volume or not
        with open(self.filename_materials, "w") as fmaterials:
            for numID in range(minmat, maxmat + 1):
                for material in G.materials:
                    if material.numID == numID:
                        fmaterials.write(
                            f"#material: {material.er:g} {material.se:g} "
                            f"{material.mr:g} {material.sm:g} {material.ID}\n"
                        )
                        if hasattr(material, "poles"):
                            if "debye" in material.type:
                                dispersionstr = "#add_dispersion_debye: " f"{material.poles:g} "
                                for pole in range(material.poles):
                                    dispersionstr += (
                                        f"{material.deltaer[pole]:g} " f"{material.tau[pole]:g} "
                                    )
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
                                    dispersionstr += (
                                        f"{material.tau[pole]:g} " f"{material.alpha[pole]:g} "
                                    )
                            dispersionstr += material.ID
                            fmaterials.write(dispersionstr + "\n")
