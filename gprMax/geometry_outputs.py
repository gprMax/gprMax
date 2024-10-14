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
from io import TextIOWrapper
from itertools import chain
from pathlib import Path

import h5py
import numpy as np
from evtk.hl import imageToVTK, linesToVTK
from evtk.vtk import VtkGroup, VtkImageData, VtkUnstructuredGrid
from mpi4py import MPI
from tqdm import tqdm

import gprMax.config as config
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.materials import Material

from ._version import __version__
from .cython.geometry_outputs import write_lines
from .subgrids.grid import SubGridBaseGrid
from .utilities.utilities import get_terminal_width

logger = logging.getLogger(__name__)


def save_geometry_views(gvs):
    """Creates and saves geometryviews.

    Args:
        gvs: list of all GeometryViews.
    """

    logger.info("")
    for i, gv in enumerate(gvs):
        gv.set_filename()
        vtk_data = gv.prep_vtk()
        pbar = tqdm(
            total=gv.nbytes,
            unit="byte",
            unit_scale=True,
            desc=f"Writing geometry view file {i + 1}/{len(gvs)}, "
            f"{gv.filename.name}{gv.vtkfiletype.ext}",
            ncols=get_terminal_width() - 1,
            file=sys.stdout,
            disable=not config.sim_config.general["progressbars"],
        )
        gv.write_vtk(vtk_data)
        pbar.update(gv.nbytes)
        pbar.close()

    logger.info("")


class GeometryView:
    """Base class for Geometry Views."""

    def __init__(self, xs, ys, zs, xf, yf, zf, dx, dy, dz, filename, grid):
        """
        Args:
            xs, xf, ys, yf, zs, zf: ints for extent of geometry view in cells.
            dx, dy, dz: ints for spatial discretisation of geometry view in cells.
            filename: string for filename.
            grid: FDTDGrid class describing a grid in a model.
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
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.filename = filename
        self.filenamebase = filename
        self.grid = grid
        self.nbytes = None

    def set_filename(self):
        """Constructs filename from user-supplied name and model run number."""
        parts = config.get_model_config().output_file_path.parts
        self.filename = Path(
            *parts[:-1], self.filenamebase + config.get_model_config().appendmodelnumber
        )


class GeometryViewLines(GeometryView):
    """Unstructured grid (.vtu) for a per-cell-edge geometry view."""

    def __init__(self, *args):
        super().__init__(*args)
        self.vtkfiletype = VtkUnstructuredGrid

    def prep_vtk(self):
        """Prepares data for writing to VTK file.

        Returns:
            vtk_data: dict of coordinates, data, and comments for VTK file.
        """

        # Sample ID array according to geometry view spatial discretisation
        # Only create a new array if subsampling is required
        if (
            self.grid.ID.shape != (self.xf, self.yf, self.zf)
            or (self.dx, self.dy, self.dz) != (1, 1, 1)
            or (self.xs, self.ys, self.zs) != (0, 0, 0)
        ):
            # Require contiguous for evtk library
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

        x, y, z, lines = write_lines(
            (self.xs * self.grid.dx),
            (self.ys * self.grid.dy),
            (self.zs * self.grid.dz),
            self.nx,
            self.ny,
            self.nz,
            (self.dx * self.grid.dx),
            (self.dy * self.grid.dy),
            (self.dz * self.grid.dz),
            ID,
        )

        # Add offset to subgrid geometry to correctly locate within main grid
        if isinstance(self.grid, SubGridBaseGrid):
            x += self.grid.i0 * self.grid.dx * self.grid.ratio
            y += self.grid.j0 * self.grid.dy * self.grid.ratio
            z += self.grid.k0 * self.grid.dz * self.grid.ratio

        # Write information about any PMLs, sources, receivers
        comments = Comments(self.grid, self)
        comments.averaged_materials = True
        comments.materials_only = True
        info = comments.get_gprmax_info()
        comments = json.dumps(info)

        # Number of bytes of data to be written to file
        offsets_size = np.arange(start=2, step=2, stop=len(x) + 1, dtype="int32").nbytes
        connect_size = len(x) * np.dtype("int32").itemsize
        cell_type_size = len(x) * np.dtype("uint8").itemsize
        self.nbytes = (
            x.nbytes
            + y.nbytes
            + z.nbytes
            + lines.nbytes
            + offsets_size
            + connect_size
            + cell_type_size
        )

        vtk_data = {"x": x, "y": y, "z": z, "data": lines, "comments": comments}

        return vtk_data

    def write_vtk(self, vtk_data):
        """Writes geometry information to a VTK file.

        Args:
            vtk_data: dict of coordinates, data, and comments for VTK file.
        """

        # Write the VTK file .vtu
        linesToVTK(
            str(self.filename),
            vtk_data["x"],
            vtk_data["y"],
            vtk_data["z"],
            cellData={"Material": vtk_data["data"]},
            comments=[vtk_data["comments"]],
        )


class GeometryViewVoxels(GeometryView):
    """Imagedata (.vti) for a per-cell geometry view."""

    def __init__(self, *args):
        super().__init__(*args)
        self.vtkfiletype = VtkImageData

    def prep_vtk(self):
        """Prepares data for writing to VTK file.

        Returns:
            vtk_data: dict of data and comments for VTK file.
        """

        # Sample solid array according to geometry view spatial discretisation
        # Only create a new array if subsampling is required
        if (
            self.grid.solid.shape != (self.xf, self.yf, self.zf)
            or (self.dx, self.dy, self.dz) != (1, 1, 1)
            or (self.xs, self.ys, self.zs) != (0, 0, 0)
        ):
            # Require contiguous for evtk library
            solid = np.ascontiguousarray(
                self.grid.solid[
                    self.xs : self.xf : self.dx,
                    self.ys : self.yf : self.dy,
                    self.zs : self.zf : self.dz,
                ]
            )
        else:
            # This array is contiguous by design
            solid = self.grid.solid

        # Write information about any PMLs, sources, receivers
        comments = Comments(self.grid, self)
        info = comments.get_gprmax_info()
        comments = json.dumps(info)

        self.nbytes = solid.nbytes

        vtk_data = {"data": solid, "comments": comments}

        return vtk_data

    def write_vtk(self, vtk_data):
        """Writes geometry information to a VTK file.

        Args:
            vtk_data: dict of data and comments for VTK file.
        """

        if isinstance(self.grid, SubGridBaseGrid):
            origin = (
                (self.grid.i0 * self.grid.dx * self.grid.ratio),
                (self.grid.j0 * self.grid.dy * self.grid.ratio),
                (self.grid.k0 * self.grid.dz * self.grid.ratio),
            )
        else:
            origin = ((self.xs * self.grid.dx), (self.ys * self.grid.dy), (self.zs * self.grid.dz))

        # Write the VTK file .vti
        imageToVTK(
            str(self.filename),
            origin=origin,
            spacing=((self.dx * self.grid.dx), (self.dy * self.grid.dy), (self.dz * self.grid.dz)),
            cellData={"Material": vtk_data["data"]},
            comments=[vtk_data["comments"]],
        )


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

        ID = G.ID[:, self.xs : self.xf + 1, self.ys : self.yf + 1, self.zs : self.zf + 1]

        # Get materials present in subset of ID array
        material_ids, inverse_map = np.unique(ID, return_inverse=True)

        # Create map from material ID to 0 - number of materials
        materials_map = {material_id: index for index, material_id in enumerate(material_ids)}

        # Remap ID array to the reduced list of materials
        ID = np.array([materials_map[id] for id in material_ids])[inverse_map].reshape(ID.shape)

        data = G.solid[self.xs : self.xf + 1, self.ys : self.yf + 1, self.zs : self.zf + 1].astype(
            "int16"
        )
        map_materials = np.vectorize(lambda id: materials_map[id])
        data = map_materials(data)

        rigidE = G.rigidE[:, self.xs : self.xf + 1, self.ys : self.yf + 1, self.zs : self.zf + 1]
        rigidH = G.rigidH[:, self.xs : self.xf + 1, self.ys : self.yf + 1, self.zs : self.zf + 1]

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
