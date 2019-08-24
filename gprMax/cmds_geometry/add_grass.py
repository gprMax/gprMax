"""Class for add_grass command."""
from .cmds_geometry import UserObjectGeometry
from ..exceptions import CmdInputError
from ..utilities import round_value
from ..materials import Material
from ..fractals import FractalSurface
from ..fractals import Grass
import gprMax.config as config

from tqdm import tqdm
import numpy as np


class AddGrass(UserObjectGeometry):
    """Allows you to add grass with roots to a :class:`gprMax.cmds_geometry.fractal_box.FractalBox` in the model.

    :param p1: The lower left (x,y,z) coordinates of a surface on a :class:`gprMax.cmds_geometry.fractal_box.FractalBox`
    :type p1: list, non-optional
    :param p2: The lower left (x,y,z) coordinates of a surface on a :class:`gprMax.cmds_geometry.fractal_box.FractalBox`
    :type p2: list, non-optional
    :param frac_dim: is the fractal dimension which, for an orthogonal parallelepiped, should take values between zero and three.
    :type frac_dim: float, non-optional
    :param limits: Define lower and upper limits for a range over which the height of the blades of grass can vary.
    :type limits: list, non-optional
    :param n_blades: The number of blades of grass that should be applied to the surface area.
    :type n_blades: int, non-optional
    :param fractal_box_id:  An identifier for the :class:`gprMax.cmds_geometry.fractal_box.FractalBox` that the grass should be applied to
    :type fractal_box_id: list, non-optional
    """

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.order = 12
        self.hash = '#add_grass'

    def create(self, grid, uip):
        """Add Grass to fractal box."""
        try:
            p1 = self.kwargs['p1']
            p2 = self.kwargs['p2']
            fractal_box_id = self.kwargs['fractal_box_id']
            frac_dim = self.kwargs['frac_dim']
            limits = self.kwargs['limits']
            n_blades = self.kwargs['n_blades']
        except KeyError:
            raise CmdInputError(self.__str__() + ' requires at least eleven parameters')

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
        if limits[0] < 0 or limits[1] < 0:
            raise CmdInputError(self.__str__() + ' requires a positive value for the minimum and maximum heights for grass blades')

        # Check for valid orientations
        if xs == xf:
            if ys == yf or zs == zf:
                raise CmdInputError(self.__str__() + ' dimensions are not specified correctly')
            if xs != volume.xs and xs != volume.xf:
                raise CmdInputError(self.__str__() + ' must specify external surfaces on a fractal box')
            fractalrange = (round_value(limits[0] / grid.dx), round_value(limits[1] / grid.dx))
            # xminus surface
            if xs == volume.xs:
                raise CmdInputError(self.__str__() + ' grass can only be specified on surfaces in the positive axis direction')
            # xplus surface
            elif xf == volume.xf:
                if fractalrange[1] > grid.nx:
                    raise CmdInputError(self.__str__() + ' cannot apply grass to fractal box as it would exceed the domain size in the x direction')
                requestedsurface = 'xplus'

        elif ys == yf:
            if xs == xf or zs == zf:
                raise CmdInputError(self.__str__() + ' dimensions are not specified correctly')
            if ys != volume.ys and ys != volume.yf:
                raise CmdInputError(self.__str__() + ' must specify external surfaces on a fractal box')
            fractalrange = (round_value(limits[0] / grid.dy), round_value(limits[1] / grid.dy))
            # yminus surface
            if ys == volume.ys:
                raise CmdInputError(self.__str__() + ' grass can only be specified on surfaces in the positive axis direction')
            # yplus surface
            elif yf == volume.yf:
                if fractalrange[1] > grid.ny:
                    raise CmdInputError(self.__str__() + ' cannot apply grass to fractal box as it would exceed the domain size in the y direction')
                requestedsurface = 'yplus'

        elif zs == zf:
            if xs == xf or ys == yf:
                raise CmdInputError(self.__str__() + ' dimensions are not specified correctly')
            if zs != volume.zs and zs != volume.zf:
                raise CmdInputError(self.__str__() + ' must specify external surfaces on a fractal box')
            fractalrange = (round_value(limits[0] / grid.dz), round_value(limits[1] / grid.dz))
            # zminus surface
            if zs == volume.zs:
                raise CmdInputError(self.__str__() + ' grass can only be specified on surfaces in the positive axis direction')
            # zplus surface
            elif zf == volume.zf:
                if fractalrange[1] > grid.nz:
                    raise CmdInputError(self.__str__() + ' cannot apply grass to fractal box as it would exceed the domain size in the z direction')
                requestedsurface = 'zplus'

        else:
            raise CmdInputError(self.__str__() + ' dimensions are not specified correctly')

        surface = FractalSurface(xs, xf, ys, yf, zs, zf, frac_dim)
        surface.ID = 'grass'
        surface.surfaceID = requestedsurface
        surface.seed = seed

        # Set the fractal range to scale the fractal distribution between zero and one
        surface.fractalrange = (0, 1)
        surface.operatingonID = volume.ID
        surface.generate_fractal_surface(grid)
        if n_blades > surface.fractalsurface.shape[0] * surface.fractalsurface.shape[1]:
            raise CmdInputError(self.__str__() + ' the specified surface is not large enough for the number of grass blades/roots specified')

        # Scale the distribution so that the summation is equal to one, i.e. a probability distribution
        surface.fractalsurface = surface.fractalsurface / np.sum(surface.fractalsurface)

        # Set location of grass blades using probability distribution
        # Create 1D vector of probability values from the 2D surface
        probability1D = np.cumsum(np.ravel(surface.fractalsurface))

        # Create random numbers between zero and one for the number of blades of grass
        R = np.random.RandomState(surface.seed)
        A = R.random_sample(n_blades)

        # Locate the random numbers in the bins created by the 1D vector of probability values, and convert the 1D index back into a x, y index for the original surface.
        bladesindex = np.unravel_index(np.digitize(A, probability1D), (surface.fractalsurface.shape[0], surface.fractalsurface.shape[1]))

        # Set the fractal range to minimum and maximum heights of the grass blades
        surface.fractalrange = fractalrange

        # Set the fractal surface using the pre-calculated spatial distribution and a random height
        surface.fractalsurface = np.zeros((surface.fractalsurface.shape[0], surface.fractalsurface.shape[1]))
        for i in range(len(bladesindex[0])):
                surface.fractalsurface[bladesindex[0][i], bladesindex[1][i]] = R.randint(surface.fractalrange[0], surface.fractalrange[1], size=1)

        # Create grass geometry parameters
        g = Grass(n_blades)
        g.seed = surface.seed
        surface.grass.append(g)

        # Check to see if grass has been already defined as a material
        if not any(x.ID == 'grass' for x in grid.materials):
            m = Material(len(grid.materials), 'grass')
            m.averagable = False
            m.type = 'builtin, debye'
            m.er = Material.grasseri
            m.deltaer.append(Material.grassdeltaer)
            m.tau.append(Material.grasstau)
            grid.materials.append(m)
            if Material.maxpoles == 0:
                Material.maxpoles = 1

        # Check if time step for model is suitable for using grass
        grass = next((x for x in grid.materials if x.ID == 'grass'))
        testgrass = next((x for x in grass.tau if x < grid.dt), None)
        if testgrass:
            raise CmdInputError(self.__str__() + ' requires the time step for the model to be less than the relaxation time required to model grass.')

        volume.fractalsurfaces.append(surface)

        if config.is_messages():
            tqdm.write('{} blades of grass on surface from {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m with fractal dimension {:g}, fractal seeding {}, and range {:g}m to {:g}m, added to {}.'.format(n_blades, xs * grid.dx, ys * grid.dy, zs * grid.dz, xf * grid.dx, yf * grid.dy, zf * grid.dz, surface.dimension, surface.seed, limits[0], limits[1], surface.operatingonID))
