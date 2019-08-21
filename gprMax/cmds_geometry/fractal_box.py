"""Class for surface roughness command."""
from .cmds_geometry import UserObjectGeometry
from ..exceptions import CmdInputError
from ..fractals import FractalVolume
import gprMax.config as config

from tqdm import tqdm
import numpy as np


class FractalBox(UserObjectGeometry):
    """User class for edge command."""

    def __init__(self, **kwargs):
        """Constructor."""
        super().__init__(**kwargs)
        self.order = 9
        self.hash = '#fractal_box'

    def create(self, grid, uip):
        try:
            p1 = self.kwargs['p1']
            p2 = self.kwargs['p2']
            frac_dim = self.kwargs['frac_dim']
            weighting = np.array(self.kwargs['weighting'])
            n_materials = self.kwargs['n_materials']
            mixing_model_id = self.kwargs['mixing_model_id']
            ID = self.kwargs['ID']

        except KeyError:
            raise CmdInputError(self.__str__() + ' Incorrect parameters')

        try:
            seed = self.kwargs['seed']
        except KeyError:
            seed = None

        # Default is no dielectric smoothing for a fractal box
        averagefractalbox = False

        # check averaging
        try:
            # go with user specified averaging
            averagefractalbox = self.kwargs['averaging']
        except KeyError:
            # if they havent specfied - go with the grid default
            averagefractalbox = False

        p1, p2 = uip.check_box_points(p1, p2, self.__str__())
        xs, ys, zs = p1
        xf, yf, zf = p2

        if frac_dim < 0:
            raise CmdInputError(self.__str__() + ' requires a positive value for the fractal dimension')
        if weighting[0] < 0:
            raise CmdInputError(self.__str__() + ' requires a positive value for the fractal weighting in the x direction')
        if weighting[1] < 0:
            raise CmdInputError(self.__str__() + ' requires a positive value for the fractal weighting in the y direction')
        if weighting[2] < 0:
            raise CmdInputError(self.__str__() + ' requires a positive value for the fractal weighting in the z direction')
        if n_materials < 0:
            raise CmdInputError(self.__str__() + ' requires a positive value for the number of bins')

        # Find materials to use to build fractal volume, either from mixing models or normal materials
        mixingmodel = next((x for x in grid.mixingmodels if x.ID == mixing_model_id), None)
        material = next((x for x in grid.materials if x.ID == mixing_model_id), None)
        nbins = n_materials

        if mixingmodel:
            if nbins == 1:
                raise CmdInputError(self.__str__() + ' must be used with more than one material from the mixing model.')
            # Create materials from mixing model as number of bins now known from fractal_box command
            mixingmodel.calculate_debye_properties(nbins, grid)
        elif not material:
            raise CmdInputError(self.__str__() + ' mixing model or material with ID {} does not exist'.format(mixing_model_id))

        volume = FractalVolume(xs, xf, ys, yf, zs, zf, frac_dim)
        volume.ID = ID
        volume.operatingonID = mixing_model_id
        volume.nbins = nbins
        volume.seed = seed
        volume.weighting = weighting
        volume.averaging = averagefractalbox
        volume.mixingmodel = mixingmodel

        if config.general['messages']:
            if volume.averaging:
                dielectricsmoothing = 'on'
            else:
                dielectricsmoothing = 'off'
            tqdm.write('Fractal box {} from {:g}m, {:g}m, {:g}m, to {:g}m, {:g}m, {:g}m with {}, fractal dimension {:g}, fractal weightings {:g}, {:g}, {:g}, fractal seeding {}, with {} material(s) created, dielectric smoothing is {}.'.format(volume.ID, xs * grid.dx, ys * grid.dy, zs * grid.dz, xf * grid.dx, yf * grid.dy, zf * grid.dz, volume.operatingonID, volume.dimension, volume.weighting[0], volume.weighting[1], volume.weighting[2], volume.seed, volume.nbins, dielectricsmoothing))

        grid.fractalvolumes.append(volume)
