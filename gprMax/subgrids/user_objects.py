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
from copy import copy
from typing import List, Tuple, Union

import numpy as np

from gprMax.cmds_geometry.cmds_geometry import UserObjectGeometry
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.model import Model
from gprMax.subgrids.grid import SubGridBaseGrid
from gprMax.subgrids.subgrid_hsg import SubGridHSG as SubGridHSGUser
from gprMax.user_inputs import MainGridUserInput
from gprMax.user_objects.cmds_multiuse import UserObjectMulti

logger = logging.getLogger(__name__)


class SubGridBase(UserObjectMulti):
    """Allows UserObjectMulti and UserObjectGeometry to be nested in SubGrid
    type user objects.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.children_multiple: List[UserObjectMulti] = []
        self.children_geometry: List[UserObjectGeometry] = []
        self.children_multiple: List[UserObjectMulti] = []
        self.children_geometry: List[UserObjectGeometry] = []

    def add(self, node: Union[UserObjectMulti, UserObjectGeometry]):
        """Adds other user objects. Geometry and multi only."""
        if isinstance(node, UserObjectMulti):
            self.children_multiple.append(node)
        elif isinstance(node, UserObjectGeometry):
            self.children_geometry.append(node)
        else:
            logger.exception(f"{str(node)} this Object can not be added to a sub grid")
            raise ValueError

    def set_discretisation(self, sg: SubGridBaseGrid, grid: FDTDGrid):
        sg.dx = grid.dx / sg.ratio
        sg.dy = grid.dy / sg.ratio
        sg.dz = grid.dz / sg.ratio
        sg.dl = np.array([sg.dx, sg.dy, sg.dz])

    def set_main_grid_indices(
        self, sg: SubGridBaseGrid, uip: MainGridUserInput, p1: Tuple[int], p2: Tuple[int]
    ):
        """Sets subgrid indices related to main grid placement."""
        # Location of the IS
        sg.i0, sg.j0, sg.k0 = p1
        sg.i1, sg.j1, sg.k1 = p2

        sg.x1, sg.y1, sg.z1 = uip.round_to_grid(p1)
        sg.x2, sg.y2, sg.z2 = uip.round_to_grid(p2)

    def set_name(self, sg: SubGridBaseGrid):
        sg.name = self.kwargs["id"]

    def set_working_region_cells(self, sg: SubGridBaseGrid):
        """Number of cells in each dimension for the working region."""
        sg.nwx = (sg.i1 - sg.i0) * sg.ratio
        sg.nwy = (sg.j1 - sg.j0) * sg.ratio
        sg.nwz = (sg.k1 - sg.k0) * sg.ratio

    def set_total_cells(self, sg: SubGridBaseGrid):
        """Number of cells in each dimension for the whole region."""
        sg.nx = 2 * sg.n_boundary_cells_x + sg.nwx
        sg.ny = 2 * sg.n_boundary_cells_y + sg.nwy
        sg.nz = 2 * sg.n_boundary_cells_z + sg.nwz

    def set_iterations(self, sg: SubGridBaseGrid, model: Model):
        """Sets number of iterations that will take place in the subgrid."""
        sg.iterations = model.iterations * sg.ratio

    def setup(self, sg: SubGridBaseGrid, model: Model, uip: MainGridUserInput):
        """ "Common setup to both all subgrid types."""
        p1 = self.kwargs["p1"]
        p2 = self.kwargs["p2"]

        p1, p2 = uip.check_box_points(p1, p2, self.__str__())

        self.set_discretisation(sg, model.G)

        # Set temporal discretisation including any inherited time step
        # stability factor from the main grid
        sg.calculate_dt()
        if model.dt_mod:
            sg.dt = sg.dt * model.dt_mod

        # Set the indices related to the subgrids main grid placement
        self.set_main_grid_indices(sg, uip, p1, p2)

        """
        try:
            uip.check_box_points([sg.i0, sg.j0, sg.k0],
                                 [sg.i1, sg.j1, sg.k1], cmd_str)
        except CmdInputError:
            es_f = 'The subgrid should extend at least {} cells'
            es = es_f.format(sg.is_os_sep * 2)
            logger.exception(cmd_str, es)
            raise ValueError
        """

        self.set_working_region_cells(sg)
        self.set_total_cells(sg)
        self.set_iterations(sg, model)
        self.set_name(sg)

        # Copy a reference for the main grid to the sub grid
        sg.parent_grid = model.G

        # Copy a subgrid reference to self so that children.build(grid, uip)
        # can access the correct grid.
        self.subgrid = sg

        # Copy over built in materials
        sg.materials = [copy(m) for m in model.G.materials if m.type == "builtin"]

        # Don't mix and match different subgrid types
        for sg_made in model.subgrids:
            if type(sg) != type(sg_made):
                logger.exception(f"{self.__str__()} please only use one type of subgrid")
                raise ValueError

        # Reference the subgrid under the main grid to which it belongs
        model.subgrids.append(sg)


class SubGridHSG(SubGridBase):
    """Huygens Surface subgridding (HSG) user object.

    Attributes:
        p1: list of the position of the lower left corner of the Inner Surface
            (x, y, z) in the main grid.
        p2: list of the position of the upper right corner of the Inner Surface
            (x, y, z) in the main grid.
        ratio: int of the ratio of the main grid spatial step to the sub-grid
                spatial step. Must be an odd integer.
        id: string identifier for the sub-grid.
        is_os_sep: int for the number of main grid cells between the Inner
                    Surface and the Outer Surface. Defaults to 3.
        pml_separation: int for the number of sub-grid cells between the Outer
                        Surface and the PML. Defaults to ratio // 2 + 2
        subgrid_pml_thickness: int for the thickness of the PML on each of the
                                6 sides of the sub-grid. Defaults to 6.
        interpolation: string for the degree of the interpolation scheme used
                        for spatial interpolation of the fields at the Inner
                        Surface. Defaults to Linear.
        filter: boolean to turn on the 3-pole filter. Increases numerical
                stability. Defaults to True.
    """

    def __init__(
        self,
        p1=None,
        p2=None,
        ratio=3,
        id="",
        is_os_sep=3,
        pml_separation=4,
        subgrid_pml_thickness=6,
        interpolation=1,
        filter=True,
        **kwargs,
    ):
        pml_separation = ratio // 2 + 2

        # Copy over the optional parameters
        kwargs["p1"] = p1
        kwargs["p2"] = p2
        kwargs["ratio"] = ratio
        kwargs["id"] = id
        kwargs["is_os_sep"] = is_os_sep
        kwargs["pml_separation"] = pml_separation
        kwargs["subgrid_pml_thickness"] = subgrid_pml_thickness
        kwargs["interpolation"] = interpolation
        kwargs["filter"] = filter

        super().__init__(**kwargs)
        self.order = 18
        self.hash = "#subgrid_hsg"

    def build(self, model: Model, uip: MainGridUserInput) -> SubGridHSGUser:
        sg = SubGridHSGUser(**self.kwargs)
        self.setup(sg, model, uip)
        return sg
