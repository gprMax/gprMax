# Copyright (C) 2015-2019: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
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

from copy import copy

import numpy as np

from gprMax import config
from ..cmds_geometry.cmds_geometry import UserObjectGeometry
from ..cmds_multiple import UserObjectMulti
from ..cmds_multiple import Rx
from ..exceptions import CmdInputError
from .multi import ReferenceRx as ReferenceRxUser
from .subgrid_hsg import SubGridHSG as SubGridHSGUser


class SubGridBase(UserObjectMulti):
    """Class to allow UserObjectMulti and UserObjectGeometry to be nested
        in SubGrid type user objects.
    """

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.children_multiple = []
        self.children_geometry = []

    def add(self, node):
        """Function to add other user objects. Geometry and multi only."""
        if isinstance(node, UserObjectMulti):
            self.children_multiple.append(node)
        elif isinstance(node, UserObjectGeometry):
            self.children_geometry.append(node)
        else:
            raise Exception(str(node) + ' This Object can not be added to a sub grid')

    def set_discretisation(self, sg, grid):
        """Set the spatial discretisation."""
        sg.dx = grid.dx / sg.ratio
        sg.dy = grid.dy / sg.ratio
        sg.dz = grid.dz / sg.ratio
        sg.dl = np.array([sg.dx, sg.dy, sg.dz])

    def set_main_grid_indices(self, sg, grid, uip, p1, p2):
        """Set subgrid indices related to main grid placement."""
        # location of the IS
        sg.i0, sg.j0, sg.k0 = p1
        sg.i1, sg.j1, sg.k1 = p2

        sg.x1, sg.y1, sg.z1 = uip.round_to_grid(p1)
        sg.x2, sg.y2, sg.z2 = uip.round_to_grid(p2)

    def set_name(self, sg):
        sg.name = self.kwargs['id']

    def set_working_region_cells(self, sg):
        """Number of cells in each dimension for the working region."""
        sg.nwx = (sg.i1 - sg.i0) * sg.ratio
        sg.nwy = (sg.j1 - sg.j0) * sg.ratio
        sg.nwz = (sg.k1 - sg.k0) * sg.ratio

    def set_total_cells(self, sg):
        """Number of cells in each dimension for the whole region."""
        sg.nx = 2 * sg.n_boundary_cells_x + sg.nwx
        sg.ny = 2 * sg.n_boundary_cells_y + sg.nwy
        sg.nz = 2 * sg.n_boundary_cells_z + sg.nwz

    def set_iterations(self, sg, main):
        """Set number of iterations that will take place in the subgrid."""
        sg.iterations = main.iterations * sg.ratio

    def setup(self, sg, grid, uip):
        """"Common setup to both all subgrid types."""
        p1 = self.kwargs['p1']
        p2 = self.kwargs['p2']

        p1, p2 = uip.check_box_points(p1, p2, self.__str__())

        self.set_discretisation(sg, grid)

        # Set the temporal discretisation
        sg.calculate_dt()

        # set the indices related to the subgrids main grid placement
        self.set_main_grid_indices(sg, grid, uip, p1, p2)

        """
        try:
            uip.check_box_points([sg.i0, sg.j0, sg.k0],
                                 [sg.i1, sg.j1, sg.k1], cmd_str)
        except CmdInputError:
            es_f = 'The subgrid should extend at least {} cells'
            es = es_f.format(sg.is_os_sep * 2)
            raise CmdInputError(cmd_str, es)
        """

        self.set_working_region_cells(sg)
        self.set_total_cells(sg)
        self.set_iterations(sg, grid)
        self.set_name(sg)

        # Copy a reference for the main grid to the sub grid
        sg.parent_grid = grid

        sg.timewindow = grid.timewindow

        # Copy a subgrid reference to self so that children.create(grid, uip) can access
        # the correct grid
        self.subgrid = sg

        # Copy over built in materials
        sg.materials = [copy(m) for m in grid.materials if m.numID in range(0, grid.n_built_in_materials + 1)]

        # Dont mix and match different subgrids
        for sg_made in grid.subgrids:
            if type(sg) != type(sg_made):
                raise CmdInputError(self.__str__() + ' Please only use one type of subgrid')

        # Reference the sub grid under the main grid to which it belongs.
        grid.subgrids.append(sg)


class SubGridHSG(SubGridBase):
    """Huygens Surface subgridding (HSG) user object.

    :param p1: Position of the lower left corner of the Inner Surface (x, y, z) in the main grid.
    :type p1: list, non-optional
    :param p2: Position of the upper right corner of the Inner Surface (x, y, z) in the main grid.
    :type p2: list, non-optional
    :param ratio: Ratio of the main grid spatial step to the sub-grid spatial step. Must be an odd integer.
    :type ratio: int, non-optional
    :param id: Identifier for the sub-grid.
    :type id: str, non-optional
    :param is_os_sep: Number of main grid cells between the Inner Surface and the Outer Surface. Defaults to 3.
    :type is_os_sep: str, optional
    :param pml_separation: Number of sub-grid cells between the Outer Surface and the PML. Defaults to ratio // 2 + 2
    :type pml_separation: int, optional
    :param subgrid_pml_thickness: Thickness of the PML on each of the 6 sides of the sub-grid. Defaults to 6.
    :type subgrid_pml_thickness: int, optional
    :param interpolation: Degree of the interpolation scheme used for spatil interpolation of the fields at the Inner Surface. Defaults to Linear
    :type interpolation: str, optional
    :param filter: Turn on the 3-pole filter. Increases numerical stability. Defaults to True
    :type filter: bool, optional

    """
    def __init__(self,
                 p1=None,
                 p2=None,
                 ratio=3,
                 id='',
                 is_os_sep=3,
                 pml_separation=4,
                 subgrid_pml_thickness=6,
                 interpolation=1,
                 filter=True,
                 **kwargs):
        """Constructor."""

        pml_separation = ratio // 2 + 2

        # copy over the optional parameters
        kwargs['p1'] = p1
        kwargs['p2'] = p2
        kwargs['ratio'] = ratio
        kwargs['id'] = id
        kwargs['is_os_sep'] = is_os_sep
        kwargs['pml_separation'] = pml_separation
        kwargs['subgrid_pml_thickness'] = subgrid_pml_thickness
        kwargs['interpolation'] = interpolation
        kwargs['filter'] = filter

        super().__init__(**kwargs)
        self.order = 18
        self.hash = '#subgrid_hsg'

    def create(self, grid, uip):
        sg = SubGridHSGUser(grid.model_num, **self.kwargs)
        self.setup(sg, grid, uip)
        return sg


class ReferenceRx(Rx):
    """ReferenceRx User Object."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hash = '#rx_reference'
        self.constructor = ReferenceRxUser

    def create(self, grid, uip):

        r = super().create(grid, uip)

        try:
            ratio = self.kwargs['ratio']
            r.ratio = ratio
            r.offset = ratio // 2

        except KeyError:
            raise CmdInputError(f"'{self.__str__()}' has an no ratio parameter")
