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
from abc import abstractmethod
from pathlib import Path
from typing import Dict, Generic, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import numpy.typing as npt
from mpi4py import MPI
from tqdm import tqdm

import gprMax.config as config
from gprMax._version import __version__
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.output_controllers.grid_view import GridType, GridView, MPIGridView
from gprMax.receivers import Rx
from gprMax.sources import Source
from gprMax.utilities.utilities import get_terminal_width
from gprMax.vtkhdf_filehandlers.vtkhdf import VtkHdfFile

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


class GeometryView(Generic[GridType]):
    """Base class for Geometry Views."""

    FILE_EXTENSION = ".vtkhdf"

    @property
    def GRID_VIEW_TYPE(self) -> type[GridView]:
        return GridView

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
        grid: GridType,
    ):
        """
        Args:
            xs, xf, ys, yf, zs, zf: ints for extent of geometry view in cells.
            dx, dy, dz: ints for spatial discretisation of geometry view in cells.
            filename: string for filename.
            grid: FDTDGrid class describing a grid in a model.
        """
        self.grid_view = self.GRID_VIEW_TYPE(grid, xs, ys, zs, xf, yf, zf, dx, dy, dz)

        self.filenamebase = filename
        self.nbytes = None

        self.material_data = None
        self.materials = None

    @property
    def grid(self) -> GridType:
        return self.grid_view.grid

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


class Metadata(Generic[GridType]):
    """Comments can be strings included in the header of XML VTK file, and are
    used to hold extra (gprMax) information about the VTK data.
    """

    def __init__(
        self,
        grid_view: GridView[GridType],
        averaged_materials: bool = False,
        materials_only: bool = False,
    ):
        self.grid_view = grid_view
        self.averaged_materials = averaged_materials
        self.materials_only = materials_only

        self.gprmax_version = __version__
        self.dx_dy_dz = self.dx_dy_dz_comment()
        self.nx_ny_nz = self.nx_ny_nz_comment()

        self.materials = self.materials_comment()

        # Write information on PMLs, sources, and receivers
        if not self.materials_only:
            # Information on PML thickness
            self.pml_thickness = self.pml_gv_comment()

            sources = (
                self.grid.hertziandipoles
                + self.grid.magneticdipoles
                + self.grid.voltagesources
                + self.grid.transmissionlines
            )
            sources_comment = self.srcs_rx_gv_comment(sources)
            if sources_comment is None:
                self.source_ids = self.source_positions = None
            else:
                self.source_ids, self.source_positions = sources_comment

            receivers_comment = self.srcs_rx_gv_comment(self.grid.rxs)
            if receivers_comment is None:
                self.receiver_ids = self.receiver_positions = None
            else:
                self.receiver_ids, self.receiver_positions = receivers_comment

    @property
    def grid(self) -> GridType:
        return self.grid_view.grid

    def write_to_vtkhdf(self, file_handler: VtkHdfFile):
        file_handler.add_field_data("gprMax_version", self.gprmax_version)
        file_handler.add_field_data("dx_dy_dz", self.dx_dy_dz)
        file_handler.add_field_data("nx_ny_nz", self.nx_ny_nz)

        file_handler.add_field_data("material_ids", self.materials)

        if not self.materials_only:
            if self.pml_thickness is not None:
                file_handler.add_field_data("pml_thickness", self.pml_thickness)

            if self.source_ids is not None and self.source_positions is not None:
                file_handler.add_field_data("source_ids", self.source_ids)
                file_handler.add_field_data("sources", self.source_positions)

            if self.receiver_ids is not None and self.receiver_positions is not None:
                file_handler.add_field_data("receiver_ids", self.receiver_ids)
                file_handler.add_field_data("receivers", self.receiver_positions)

    def pml_gv_comment(self) -> Optional[npt.NDArray[np.int64]]:
        grid = self.grid

        if not grid.pmls["slabs"]:
            return None

        # Only render PMLs if they are in the geometry view
        thickness: Dict[str, int] = grid.pmls["thickness"]
        gv_pml_depth = dict.fromkeys(thickness, 0)

        if self.grid_view.xs < thickness["x0"]:
            gv_pml_depth["x0"] = thickness["x0"] - self.grid_view.xs
        if self.grid_view.ys < thickness["y0"]:
            gv_pml_depth["y0"] = thickness["y0"] - self.grid_view.ys
        if thickness["z0"] - self.grid_view.zs > 0:
            gv_pml_depth["z0"] = thickness["z0"] - self.grid_view.zs
        if self.grid_view.xf > grid.nx - thickness["xmax"]:
            gv_pml_depth["xmax"] = self.grid_view.xf - (grid.nx - thickness["xmax"])
        if self.grid_view.yf > grid.ny - thickness["ymax"]:
            gv_pml_depth["ymax"] = self.grid_view.yf - (grid.ny - thickness["ymax"])
        if self.grid_view.zf > grid.nz - thickness["zmax"]:
            gv_pml_depth["zmax"] = self.grid_view.zf - (grid.nz - thickness["zmax"])

        return np.array(list(gv_pml_depth.values()), dtype=np.int64)

    def srcs_rx_gv_comment(
        self, srcs: Union[Sequence[Source], List[Rx]]
    ) -> Optional[Tuple[List[str], npt.NDArray[np.float64]]]:
        """Used to name sources and/or receivers."""
        if not srcs:
            return None

        names: List[str] = []
        positions = np.empty((len(srcs), 3))
        for index, src in enumerate(srcs):
            position = src.coord * self.grid.dl
            names.append(src.ID)
            positions[index] = position

        return names, positions

    def dx_dy_dz_comment(self) -> npt.NDArray[np.float64]:
        return self.grid.dl

    def nx_ny_nz_comment(self) -> npt.NDArray[np.int32]:
        return self.grid.size

    def materials_comment(self) -> Optional[List[str]]:
        if hasattr(self.grid_view, "materials"):
            materials = self.grid_view.materials
        else:
            materials = self.grid.materials

        if materials is None:
            return None

        if not self.averaged_materials:
            return [m.ID for m in materials if m.type != "dielectric-smoothed"]
        else:
            return [m.ID for m in materials]


class MPIMetadata(Metadata[MPIGrid]):
    def nx_ny_nz_comment(self) -> npt.NDArray[np.int32]:
        return self.grid.global_size

    def pml_gv_comment(self) -> Optional[npt.NDArray[np.int64]]:
        gv_pml_depth = super().pml_gv_comment()

        if gv_pml_depth is None:
            gv_pml_depth = np.zeros(6, dtype=np.int64)

        assert isinstance(self.grid_view, MPIGridView)
        recv_buffer = np.empty((self.grid_view.comm.size, 6), dtype=np.int64)
        self.grid_view.comm.Allgather(gv_pml_depth, recv_buffer)

        gv_pml_depth = np.max(recv_buffer, axis=0)

        return None if all(gv_pml_depth == 0) else gv_pml_depth

    def srcs_rx_gv_comment(
        self, srcs: Union[Sequence[Source], List[Rx]]
    ) -> Optional[Tuple[List[str], npt.NDArray[np.float64]]]:
        objects: Dict[str, npt.NDArray[np.float64]] = {}
        for src in srcs:
            position = self.grid.local_to_global_coordinate(src.coord) * self.grid.dl
            objects[src.ID] = position

        assert isinstance(self.grid_view, MPIGridView)
        global_objects: List[Dict[str, npt.NDArray[np.float64]]] = self.grid_view.comm.allgather(
            objects
        )
        objects = {k: v for d in global_objects for k, v in d.items()}
        objects = dict(sorted(objects.items()))

        return (list(objects.keys()), np.array(list(objects.values()))) if objects else None
