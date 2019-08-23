"""Class for edge command."""
from .cmds_geometry import UserObjectGeometry
from ..exceptions import CmdInputError
from ..cython.geometry_primitives import build_edge_x
from ..cython.geometry_primitives import build_edge_y
from ..cython.geometry_primitives import build_edge_z

from tqdm import tqdm


class Edge(UserObjectGeometry):
    """User class for edge command."""

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.order = 2
        self.hash = '#edge'

    def create(self, grid, uip):
        """Create edge and add it to the grid."""
        try:
            p1 = self.kwargs['p1']
            p2 = self.kwargs['p2']
            material_id = self.kwargs['material_id']
        except KeyError:
            raise CmdInputError(self.__str__() + ' requires exactly 3 parameters')

        p1, p2 = uip.check_box_points(p1, p2, self.__str__())
        xs, ys, zs = p1
        xf, yf, zf = p2

        material = next((x for x in grid.materials if x.ID == material_id), None)

        if not material:
            raise CmdInputError('Material with ID {} does not exist'.format(material_id))

        # Check for valid orientations
        # x-orientated wire
        if xs != xf:
            if ys != yf or zs != zf:
                raise CmdInputError(self.__str__() + ' the edge is not specified correctly')
            else:
                for i in range(xs, xf):
                    build_edge_x(i, ys, zs, material.numID, grid.rigidE, grid.rigidH, grid.ID)

        # y-orientated wire
        elif ys != yf:
            if xs != xf or zs != zf:
                raise CmdInputError(self.__str__() + ' the edge is not specified correctly')
            else:
                for j in range(ys, yf):
                    build_edge_y(xs, j, zs, material.numID, grid.rigidE, grid.rigidH, grid.ID)

        # z-orientated wire
        elif zs != zf:
            if xs != xf or ys != yf:
                raise CmdInputError(self.__str__() + ' the edge is not specified correctly')
            else:
                for k in range(zs, zf):
                    build_edge_z(xs, ys, k, material.numID, grid.rigidE, grid.rigidH, grid.ID)

        if config.is_messages():
            tqdm.write('Edge from {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m of material {} created.'.format(xs * grid.dx, ys * grid.dy, zs * grid.dz, xf * grid.dx, yf * grid.dy, zf * grid.dz, material_id))
