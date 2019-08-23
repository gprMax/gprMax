"""Class for cylinder command."""
from .cmds_geometry import UserObjectGeometry
from ..exceptions import CmdInputError
from ..materials import Material
from ..cython.geometry_primitives import build_cylindrical_sector

from tqdm import tqdm
import numpy as np


class CylindricalSector(UserObjectGeometry):
    """User class for edge command."""

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.order = 7
        self.hash = '#cylindrical_sector'

    def create(self, grid, uip):

        try:
            normal = self.kwargs['normal'].lower()
            ctr1 = self.kwargs['ctr1']
            ctr2 = self.kwargs['ctr2']
            extent1 = self.kwargs['extent1']
            extent2 = self.kwargs['extent2']
            start = self.kwargs['start']
            end = self.kwargs['end']
            r = self.kwargs['r']
            thickness = extent2 - extent1
        except KeyError:
            raise CmdInputError(self.__str__())

        # check averaging
        try:
            # go with user specified averaging
            averagecylindricalsector = self.kwargs['averaging']
        except KeyError:
            # if they havent specfied - go with the grid default
            averagecylindricalsector = grid.averagevolumeobjects

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

        sectorstartangle = 2 * np.pi * (start / 360)
        sectorangle = 2 * np.pi * (end / 360)

        if normal != 'x' and normal != 'y' and normal != 'z':
            raise CmdInputError(self.__str__() + ' the normal direction must be either x, y or z.')
        if r <= 0:
            raise CmdInputError(self.__str__() + ' the radius {:g} should be a positive value.'.format(r))
        if sectorstartangle < 0 or sectorangle <= 0:
            raise CmdInputError(self.__str__() + ' the starting angle and sector angle should be a positive values.')
        if sectorstartangle >= 2 * np.pi or sectorangle >= 2 * np.pi:
            raise CmdInputError(self.__str__() + ' the starting angle and sector angle must be less than 360 degrees.')

        # Look up requested materials in existing list of material instances
        materials = [y for x in materialsrequested for y in grid.materials if y.ID == x]

        if len(materials) != len(materialsrequested):
            notfound = [x for x in materialsrequested if x not in materials]
            raise CmdInputError(self.__str__() + ' material(s) {} do not exist'.format(notfound))

        if thickness > 0:
            # Isotropic case
            if len(materials) == 1:
                averaging = materials[0].averagable and averagecylindricalsector
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

        # yz-plane cylindrical sector
        if normal == 'x':
            level, ctr1, ctr2 = uip.round_to_grid((extent1, ctr1, ctr2))

        # xz-plane cylindrical sector
        elif normal == 'y':
            ctr1, level, ctr2 = uip.round_to_grid((ctr1, extent1, ctr2))

        # xy-plane cylindrical sector
        elif normal == 'z':
            ctr1, ctr2, level = uip.round_to_grid((ctr1, ctr2, extent1))

        build_cylindrical_sector(ctr1, ctr2, level, sectorstartangle, sectorangle, r, normal, thickness, grid.dx, grid.dy, grid.dz, numID, numIDx, numIDy, numIDz, averaging, grid.solid, grid.rigidE, grid.rigidH, grid.ID)

        if config.is_messages():
            if thickness > 0:
                if averaging:
                    dielectricsmoothing = 'on'
                else:
                    dielectricsmoothing = 'off'
                tqdm.write('Cylindrical sector with centre {:g}m, {:g}m, radius {:g}m, starting angle {:.1f} degrees, sector angle {:.1f} degrees, thickness {:g}m, of material(s) {} created, dielectric smoothing is {}.'.format(ctr1, ctr2, r, (sectorstartangle / (2 * np.pi)) * 360, (sectorangle / (2 * np.pi)) * 360, thickness, ', '.join(materialsrequested), dielectricsmoothing))
            else:
                tqdm.write('Cylindrical sector with centre {:g}m, {:g}m, radius {:g}m, starting angle {:.1f} degrees, sector angle {:.1f} degrees, of material(s) {} created.'.format(ctr1, ctr2, r, (sectorstartangle / (2 * np.pi)) * 360, (sectorangle / (2 * np.pi)) * 360, ', '.join(materialsrequested)))
