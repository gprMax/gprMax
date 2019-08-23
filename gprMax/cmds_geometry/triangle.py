"""Class for triangle command."""
from .cmds_geometry import UserObjectGeometry
from ..exceptions import CmdInputError
from ..materials import Material
from ..cython.geometry_primitives import build_triangle

from tqdm import tqdm
import numpy as np


class Triangle(UserObjectGeometry):
    """User class for edge command."""

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.order = 4
        self.hash = '#triangle'

    def create(self, grid, uip):

        try:
            up1 = self.kwargs['p1']
            up2 = self.kwargs['p2']
            up3 = self.kwargs['p3']
            thickness = self.kwargs['thickness']

        except KeyError:
            raise CmdInputError(self.params_str() + ' Specify 3 points and a thickness')

        # check averaging
        try:
            # go with user specified averaging
            averagetriangularprism = self.kwargs['averaging']
        except KeyError:
            # if they havent specfied - go with the grid default
            averagetriangularprism = grid.averagevolumeobjects

        # check materials have been specified
        # isotropic case
        try:
            materialsrequested = [self.kwargs['material_id']]
        except KeyError:
            # Anisotropic case
            try:
                materialsrequested = self.kwargs['material_ids']
            except KeyError:
                raise CmdInputError(self.__str__() + ' No materials have been specified')

        # Check whether points are valid against grid
        uip.check_tri_points(up1, up2, up3, object)
        # Convert points to metres
        x1, y1, z1 = uip.round_to_grid(up1)
        x2, y2, z2 = uip.round_to_grid(up2)
        x3, y3, z3 = uip.round_to_grid(up3)

        if thickness < 0:
            raise CmdInputError(self.__str__() + ' requires a positive value for thickness')

        # Check for valid orientations
        # yz-plane triangle
        if x1 == x2 and x2 == x3:
            normal = 'x'
        # xz-plane triangle
        elif y1 == y2 and y2 == y3:
            normal = 'y'
        # xy-plane triangle
        elif z1 == z2 and z2 == z3:
            normal = 'z'
        else:
            raise CmdInputError(self.__str__() + ' the triangle is not specified correctly')

        # Look up requested materials in existing list of material instances
        materials = [y for x in materialsrequested for y in grid.materials if y.ID == x]

        if len(materials) != len(materialsrequested):
            notfound = [x for x in materialsrequested if x not in materials]
            raise CmdInputError(self.__str__() + ' material(s) {} do not exist'.format(notfound))

        if thickness > 0:
            # Isotropic case
            if len(materials) == 1:
                averaging = materials[0].averagable and averagetriangularprism
                numID = numIDx = numIDy = numIDz = materials[0].numID

            # Uniaxial anisotropic case
            elif len(materials) == 3:
                averaging = False
                numIDx = materials[0].numID
                numIDy = materials[1].numID
                numIDz = materials[2].numID
                requiredID = materials[0].ID + '+' + materials[1].ID + '+' + materials[2].ID
                averagedmaterial = [x for x in grid.materials if x.ID == requiredID]
                if averagedmaterial:
                    numID = averagedmaterial.numID
                else:
                    numID = len(grid.materials)
                    m = Material(numID, requiredID)
                    m.type = 'dielectric-smoothed'
                    # Create dielectric-smoothed constituents for material
                    m.er = np.mean((materials[0].er, materials[1].er, materials[2].er), axis=0)
                    m.se = np.mean((materials[0].se, materials[1].se, materials[2].se), axis=0)
                    m.mr = np.mean((materials[0].mr, materials[1].mr, materials[2].mr), axis=0)
                    m.sm = np.mean((materials[0].mr, materials[1].mr, materials[2].mr), axis=0)

                    # Append the new material object to the materials list
                    grid.materials.append(m)
        else:
            averaging = False
            # Isotropic case
            if len(materials) == 1:
                numID = numIDx = numIDy = numIDz = materials[0].numID

            # Uniaxial anisotropic case
            elif len(materials) == 3:
                # numID requires a value but it will not be used
                numID = None
                numIDx = materials[0].numID
                numIDy = materials[1].numID
                numIDz = materials[2].numID

        build_triangle(x1, y1, z1, x2, y2, z2, x3, y3, z3, normal, thickness, grid.dx, grid.dy, grid.dz, numID, numIDx, numIDy, numIDz, averaging, grid.solid, grid.rigidE, grid.rigidH, grid.ID)

        if config.is_messages():
            if thickness > 0:
                if averaging:
                    dielectricsmoothing = 'on'
                else:
                    dielectricsmoothing = 'off'
                tqdm.write('Triangle with coordinates {:g}m {:g}m {:g}m, {:g}m {:g}m {:g}m, {:g}m {:g}m {:g}m and thickness {:g}m of material(s) {} created, dielectric smoothing is {}.'.format(x1, y1, z1, x2, y2, z2, x3, y3, z3, thickness, ', '.join(materialsrequested), dielectricsmoothing))
            else:
                tqdm.write('Triangle with coordinates {:g}m {:g}m {:g}m, {:g}m {:g}m {:g}m, {:g}m {:g}m {:g}m of material(s) {} created.'.format(x1, y1, z1, x2, y2, z2, x3, y3, z3, ', '.join(materialsrequested)))
