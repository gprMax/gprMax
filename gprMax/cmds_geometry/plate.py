"""Class for edge command."""
from .cmds_geometry import UserObjectGeometry
from ..exceptions import CmdInputError
from ..cython.geometry_primitives import build_face_yz
from ..cython.geometry_primitives import build_face_xz
from ..cython.geometry_primitives import build_face_xy

from tqdm import tqdm


class Plate(UserObjectGeometry):
    """User class for edge command."""

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.order = 3
        self.hash = '#plate'

    def create(self, grid, uip):

        try:
            p1 = self.kwargs['p1']
            p2 = self.kwargs['p2']

        except KeyError:
            raise CmdInputError(self.__str__() + ' 2 points must be specified')

        # isotropic
        try:
            materialsrequested = [self.kwargs['material_id']]
        except KeyError:
            # Anisotropic case
            try:
                materialsrequested = self.kwargs['material_ids']
            except KeyError:
                raise CmdInputError(self.__str__() + ' No materials have been specified')

        p1, p2 = uip.check_box_points(p1, p2, self.__str__())
        xs, ys, zs = p1
        xf, yf, zf = p2

        # Check for valid orientations
        if xs == xf:
            if ys == yf or zs == zf:
                raise CmdInputError(self.__str__() + ' the plate is not specified correctly')

        elif ys == yf:
            if xs == xf or zs == zf:
                raise CmdInputError(self.__str__() + ' the plate is not specified correctly')

        elif zs == zf:
            if xs == xf or ys == yf:
                raise CmdInputError(self.__str__() + ' the plate is not specified correctly')

        else:
            raise CmdInputError(self.__str__() + ' the plate is not specified correctly')

        # Look up requested materials in existing list of material instances
        materials = [y for x in materialsrequested for y in grid.materials if y.ID == x]

        if len(materials) != len(materialsrequested):
            notfound = [x for x in materialsrequested if x not in materials]
            raise CmdInputError(self.__str__() + ' material(s) {} do not exist'.format(notfound))

        # yz-plane plate
        if xs == xf:
            # Isotropic case
            if len(materials) == 1:
                numIDx = numIDy = numIDz = materials[0].numID

            # Uniaxial anisotropic case
            elif len(materials) == 2:
                numIDy = materials[0].numID
                numIDz = materials[1].numID

            for j in range(ys, yf):
                for k in range(zs, zf):
                    build_face_yz(xs, j, k, numIDy, numIDz, grid.rigidE, grid.rigidH, grid.ID)

        # xz-plane plate
        elif ys == yf:
            # Isotropic case
            if len(materials) == 1:
                numIDx = numIDy = numIDz = materials[0].numID

            # Uniaxial anisotropic case
            elif len(materials) == 2:
                numIDx = materials[0].numID
                numIDz = materials[1].numID

            for i in range(xs, xf):
                for k in range(zs, zf):
                    build_face_xz(i, ys, k, numIDx, numIDz, grid.rigidE, grid.rigidH, grid.ID)

        # xy-plane plate
        elif zs == zf:
            # Isotropic case
            if len(materials) == 1:
                numIDx = numIDy = numIDz = materials[0].numID

            # Uniaxial anisotropic case
            elif len(materials) == 2:
                numIDx = materials[0].numID
                numIDy = materials[1].numID

            for i in range(xs, xf):
                for j in range(ys, yf):
                    build_face_xy(i, j, zs, numIDx, numIDy, grid.rigidE, grid.rigidH, grid.ID)

        if grid.messages:
            tqdm.write('Plate from {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m of material(s) {} created.'.format(xs * grid.dx, ys * grid.dy, zs * grid.dz, xf * grid.dx, yf * grid.dy, zf * grid.dz, ', '.join(materialsrequested)))
