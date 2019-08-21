from .hsg import SubGridHSG as SubGridHSGUser
from .multi import ReferenceRx as ReferenceRxUser
from ..exceptions import CmdInputError
from ..cmds_multiple import UserObjectMulti
from ..cmds_geometry.cmds_geometry import UserObjectGeometry
from ..cmds_multiple import Rx
import gprMax.config as config


from copy import copy

import numpy as np


class SubGridBase(UserObjectMulti):
    """Class to allow UserObjectMulti and UserObjectGeometry to be nested in SubGrid type user objects."""

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

    def check_filters(self, grid):
        """Check the filter of other grids - Only allow filters all on or filters all off."""
        if grid.subgrids:
            f = grid.subgrids[0]
            if f != self.kwargs['filter']:
                raise CmdInputError(self.__str__() + "Filters should be on or off. Set Filter on or off for all subgrids")

    def set_discretisation(self, sg, grid):
        """Set the spatial discretisation."""
        sg.dl = grid.dl / sg.ratio
        sg.dx, sg.dy, sg.dz = grid.dl

    def set_main_grid_indices(self, sg, grid, uip, p1, p2):
        """Set subgrid indices related to main grid placement."""

        # IS indices
        sg.i0, sg.j0, sg.k0 = p1
        sg.i1, sg.j1, sg.k1 = p2

        # OS indices
        sg.i_l, sg.j_l, sg.k_l = p1 - sg.is_os_sep
        sg.i_u, sg.j_u, sg.k_u = p2 + sg.is_os_sep

        # discretisted coordinates of the IS
        sg.x1, sg.y1, sg.z1 = uip.round_to_grid(p1)
        sg.x2, sg.y2, sg.z2 = uip.round_to_grid(p2)

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

    def set_name(self, sg):
        sg.name = self.kwargs['ID']

    def setup(self, sg, grid, uip):
        """"Common setup to both all subgrid types."""
        p1 = self.kwargs['p1']
        p2 = self.kwargs['p2']

        p1, p2 = uip.check_box_points(p1, p2, self.__str__())

        self.set_name(sg)

        self.check_filters(grid)

        self.set_discretisation(sg, grid)

        # Set the temporal discretisation
        sg.calculate_dt()

        # ensure stability
        sg.round_time_step()

        # set the indices related to the subgrids main grid placement
        self.set_main_grid_indices(sg, grid, uip, p1, p2)

        self.set_working_region_cells(sg)
        self.set_total_cells(sg)
        self.set_iterations(sg, grid)

        # Copy a reference for the main grid to the sub grid
        sg.parent_grid = grid

        sg.timewindow = grid.timewindow

        # Copy a subgrid reference to self so that children.create(grid, uip) can access
        # the correct grid
        self.subgrid = sg

        # Copy over built in materials
        sg.materials = [copy(m) for m in grid.materials if m.numID in range(0, grid.n_built_in_materials + 1)]
        # use same number of threads
        sg.nthreads = grid.nthreads

        # Dont mix and match different subgrids
        for sg_made in grid.subgrids:
            if type(sg) != type(sg_made):
                raise CmdInputError(self.__str__() + ' Please only use one type of subgrid')

        # Reference the sub grid under the main grid to which it belongs.
        grid.subgrids.append(sg)


class SubGridHSG(SubGridBase):
    """HSG User Object."""
    def __init__(self,
                 p1=None,
                 p2=None,
                 ratio=3,
                 ID='',
                 is_os_sep=3,
                 subgrid_pml_thickness=6,
                 interpolation=3,
                 loss_mechanism=False,
                 filter=True,
                 **kwargs):
        """Constructor."""

        # copy over the optional parameters
        kwargs['p1'] = p1
        kwargs['p2'] = p2
        kwargs['ratio'] = ratio
        kwargs['ID'] = ID
        kwargs['is_os_sep'] = is_os_sep
        kwargs['pml_separation'] = ratio // 2 + 2
        kwargs['subgrid_pml_thickness'] = subgrid_pml_thickness
        kwargs['interpolation'] = interpolation
        kwargs['filter'] = filter

        super().__init__(**kwargs)
        self.order = 18
        self.hash = '#subgrid_hsg'

    def create(self, grid, uip):
        sg = SubGridHSGUser(**self.kwargs)
        self.setup(sg, grid, uip)
        if config.general['messages']:
            print(sg)
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
            raise CmdInputError("'{}' has an no ratio parameter".format(self.__str__()))
