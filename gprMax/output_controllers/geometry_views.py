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
from typing import Dict, Generic, List, Sequence, Tuple, Union

import h5py
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

import gprMax.config as config
from gprMax._version import __version__
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.output_controllers.grid_view import GridType, GridView, GridViewType
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


class GeometryView(Generic[GridViewType]):
    """Base class for Geometry Views."""

    FILE_EXTENSION = ".vtkhdf"

    def __init__(
        self,
        grid_view: GridViewType,
        filename: str,
    ):
        """
        Args:
            xs, xf, ys, yf, zs, zf: ints for extent of geometry view in cells.
            dx, dy, dz: ints for spatial discretisation of geometry view in cells.
            filename: string for filename.
            grid: FDTDGrid class describing a grid in a model.
        """
        self.grid_view = grid_view

        self.filenamebase = filename
        self.nbytes = None

        self.material_data = None
        self.materials = None

    @property
    def grid(self) -> FDTDGrid:
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
        gv: GridView[GridType],
        averaged_materials: bool = False,
        materials_only: bool = False,
    ):
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

    @property
    def grid(self) -> GridType:
        return self.gv.grid

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
