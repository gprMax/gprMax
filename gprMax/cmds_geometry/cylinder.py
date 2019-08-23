"""Class for cylinder command."""
from .cmds_geometry import UserObjectGeometry
from ..exceptions import CmdInputError
from ..materials import Material
from ..cython.geometry_primitives import build_cylinder

from tqdm import tqdm
import numpy as np
import gprMax.config as config


class Cylinder(UserObjectGeometry):
    """User class for edge command."""

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.order = 6
        self.hash = '#cylinder'

    def create(self, grid, uip):

        try:
            p1 = self.kwargs['p1']
            p2 = self.kwargs['p2']
            r = self.kwargs['r']

        except KeyError:
            raise CmdInputError(self.__str__() + ' Please specify 2 points and a radius')

        # check averaging
        try:
            # go with user specified averaging
            averagecylinder = self.kwargs['averaging']
        except KeyError:
            # if they havent specfied - go with the grid default
            averagecylinder = grid.averagevolumeobjects

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

        x1, y1, z1 = uip.round_to_grid(p1)
        x2, y2, z2 = uip.round_to_grid(p2)

        if r <= 0:
            raise CmdInputError(self.__str__() + ' the radius {:g} should be a positive value.'.format(r))

        # Look up requested materials in existing list of material instances
        materials = [y for x in materialsrequested for y in grid.materials if y.ID == x]

        if len(materials) != len(materialsrequested):
            notfound = [x for x in materialsrequested if x not in materials]
            raise CmdInputError(self.__str__() + ' material(s) {} do not exist'.format(notfound))

        # Isotropic case
        if len(materials) == 1:
            averaging = materials[0].averagable and averagecylinder
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

        build_cylinder(x1, y1, z1, x2, y2, z2, r, grid.dx, grid.dy, grid.dz, numID, numIDx, numIDy, numIDz, averaging, grid.solid, grid.rigidE, grid.rigidH, grid.ID)

        if config.is_messages():
            if averaging:
                dielectricsmoothing = 'on'
            else:
                dielectricsmoothing = 'off'
            tqdm.write('Cylinder with face centres {:g}m, {:g}m, {:g}m and {:g}m, {:g}m, {:g}m, with radius {:g}m, of material(s) {} created, dielectric smoothing is {}.'.format(x1, y1, z1, x2, y2, z2, r, ', '.join(materialsrequested), dielectricsmoothing))
