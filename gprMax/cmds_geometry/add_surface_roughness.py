"""Class for surface roughness command."""
from .cmds_geometry import UserObjectGeometry
from ..exceptions import CmdInputError
from ..fractals import FractalSurface
from ..utilities import round_value
import gprMax.config as config

from tqdm import tqdm
import numpy as np


class AddSurfaceRoughness(UserObjectGeometry):
    """User class for edge command."""

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.order = 10
        self.hash = '#add_surface_roughness'

    def create(self, grid, uip):

        try:
            p1 = self.kwargs['p1']
            p2 = self.kwargs['p2']
            frac_dim = self.kwargs['frac_dim']
            weighting = np.array(self.kwargs['weighting'], dtype=np.float64)
            limits = np.array(self.kwargs['limits'])
            fractal_box_id = self.kwargs['fractal_box_id']
        except KeyError:
            raise CmdInputError(self.__str__() + ' Incorrect parameters')

        try:
            seed = self.kwargs['seed']
        except KeyError:
            seed = None

        # grab the correct fractal volume
        volumes = [volume for volume in grid.fractalvolumes if volume.ID == fractal_box_id]
        if volumes:
            volume = volumes[0]
        else:
            raise CmdInputError(self.__str__() + ' Cant find FractalBox {}'.format(fractal_box_id))

        p1, p2 = uip.check_box_points(p1, p2, self.__str__())
        xs, ys, zs = p1
        xf, yf, zf = p2

        if frac_dim < 0:
            raise CmdInputError(self.__str__() + ' requires a positive value for the fractal dimension')
        if weighting[0] < 0:
            raise CmdInputError(self.__str__() + ' requires a positive value for the fractal weighting in the first direction of the surface')
        if weighting[1] < 0:
            raise CmdInputError(self.__str__() + ' requires a positive value for the fractal weighting in the second direction of the surface')

        # Check for valid orientations
        if xs == xf:
            if ys == yf or zs == zf:
                raise CmdInputError(self.__str__() + ' dimensions are not specified correctly')
            if xs != volume.xs and xs != volume.xf:
                raise CmdInputError(self.__str__() + ' can only be used on the external surfaces of a fractal box')
            fractalrange = (round_value(limits[0] / grid.dx), round_value(limits[1] / grid.dx))
            # xminus surface
            if xs == volume.xs:
                if fractalrange[0] < 0 or fractalrange[1] > volume.xf:
                    raise CmdInputError(self.__str__() + ' cannot apply fractal surface to fractal box as it would exceed either the upper coordinates of the fractal box or the domain in the x direction')
                requestedsurface = 'xminus'
            # xplus surface
            elif xf == volume.xf:
                if fractalrange[0] < volume.xs or fractalrange[1] > grid.nx:
                    raise CmdInputError(self.__str__() + ' cannot apply fractal surface to fractal box as it would exceed either the lower coordinates of the fractal box or the domain in the x direction')
                requestedsurface = 'xplus'

        elif ys == yf:
            if xs == xf or zs == zf:
                raise CmdInputError(self.__str__() + ' dimensions are not specified correctly')
            if ys != volume.ys and ys != volume.yf:
                raise CmdInputError(self.__str__() + ' can only be used on the external surfaces of a fractal box')
            fractalrange = (round_value(limits[0] / grid.dy), round_value(limits[1] / grid.dy))
            # yminus surface
            if ys == volume.ys:
                if fractalrange[0] < 0 or fractalrange[1] > volume.yf:
                    raise CmdInputError(self.__str__() + ' cannot apply fractal surface to fractal box as it would exceed either the upper coordinates of the fractal box or the domain in the y direction')
                requestedsurface = 'yminus'
            # yplus surface
            elif yf == volume.yf:
                if fractalrange[0] < volume.ys or fractalrange[1] > grid.ny:
                    raise CmdInputError(self.__str__() + ' cannot apply fractal surface to fractal box as it would exceed either the lower coordinates of the fractal box or the domain in the y direction')
                requestedsurface = 'yplus'

        elif zs == zf:
            if xs == xf or ys == yf:
                raise CmdInputError(self.__str__() + ' dimensions are not specified correctly')
            if zs != volume.zs and zs != volume.zf:
                raise CmdInputError(self.__str__() + ' can only be used on the external surfaces of a fractal box')
            fractalrange = (round_value(limits[0] / grid.dz), round_value(limits[1] / grid.dz))
            # zminus surface
            if zs == volume.zs:
                if fractalrange[0] < 0 or fractalrange[1] > volume.zf:
                    raise CmdInputError(self.__str__() + ' cannot apply fractal surface to fractal box as it would exceed either the upper coordinates of the fractal box or the domain in the x direction')
                requestedsurface = 'zminus'
            # zplus surface
            elif zf == volume.zf:
                if fractalrange[0] < volume.zs or fractalrange[1] > grid.nz:
                    raise CmdInputError(self.__str__() + ' cannot apply fractal surface to fractal box as it would exceed either the lower coordinates of the fractal box or the domain in the z direction')
                requestedsurface = 'zplus'

        else:
            raise CmdInputError(self.__str__() + ' dimensions are not specified correctly')

        surface = FractalSurface(xs, xf, ys, yf, zs, zf, frac_dim)
        surface.surfaceID = requestedsurface
        surface.fractalrange = fractalrange
        surface.operatingonID = volume.ID
        surface.seed = seed
        surface.weighting = weighting

        # List of existing surfaces IDs
        existingsurfaceIDs = [x.surfaceID for x in volume.fractalsurfaces]
        if surface.surfaceID in existingsurfaceIDs:
            raise CmdInputError(self.__str__() + ' has already been used on the {} surface'.format(surface.surfaceID))

        surface.generate_fractal_surface(grid)
        volume.fractalsurfaces.append(surface)

        if config.is_messages():
            tqdm.write('Fractal surface from {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m with fractal dimension {:g}, fractal weightings {:g}, {:g}, fractal seeding {}, and range {:g}m to {:g}m, added to {}.'.format(xs * grid.dx, ys * grid.dy, zs * grid.dz, xf * grid.dx, yf * grid.dy, zf * grid.dz, surface.dimension, surface.weighting[0], surface.weighting[1], surface.seed, limits[0], limits[1], surface.operatingonID))
