# Copyright (C) 2015-2020: The University of Edinburgh
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

import logging

import numpy as np

import gprMax.config as config
from .cmds_geometry import UserObjectGeometry
from ..exceptions import CmdInputError
from ..fractals import FractalVolume

logger = logging.getLogger(__name__)


class FractalBox(UserObjectGeometry):
    """Allows you to introduce an orthogonal parallelepiped with fractal distributed properties which are related to a mixing model or normal material into the model.

    :param p1: The lower left (x,y,z) coordinates of the parallelepiped
    :type p1: list, non-optional
    :param p2: The upper right (x,y,z) coordinates of the parallelepiped
    :type p2: list, non-optional
    :param frac_dim: The fractal dimension which, for an orthogonal parallelepiped, should take values between zero and three.
    :type frac_dim: float, non-optional
    :param weighting: Weightings in the x, y, z direction of the surface.
    :type weighting: list, non-optional
    :param n_materials: Number of materials to use for the fractal distribution (defined according to the associated mixing model). This should be set to one if using a normal material instead of a mixing model.
    :type n_materials: list, non-optional
    :param mixing_model_id: Is an identifier for the associated mixing model or material.
    :type mixing_model_id: list, non-optional
    :param id: Identifier for the fractal box itself.
    :type id: list, non-optional
    :param seed: Controls the seeding of the random number generator used to create the fractals..
    :type seed: float, non-optional
    :param averaging:  y or n, used to switch on and off dielectric smoothing.
    :type averaging: str, non-optional
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hash = '#fractal_box'

    def create(self, grid, uip):
        try:
            p1 = self.kwargs['p1']
            p2 = self.kwargs['p2']
            frac_dim = self.kwargs['frac_dim']
            weighting = np.array(self.kwargs['weighting'])
            n_materials = self.kwargs['n_materials']
            mixing_model_id = self.kwargs['mixing_model_id']
            ID = self.kwargs['id']
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

        dielectricsmoothing = 'on' if volume.averaging else 'off'
        logger.info(f'Fractal box {volume.ID} from {xs * grid.dx:g}m, {ys * grid.dy:g}m, {zs * grid.dz:g}m, to {xf * grid.dx:g}m, {yf * grid.dy:g}m, {zf * grid.dz:g}m with {volume.operatingonID}, fractal dimension {volume.dimension:g}, fractal weightings {volume.weighting[0]:g}, {volume.weighting[1]:g}, {volume.weighting[2]:g}, fractal seeding {volume.seed}, with {volume.nbins} material(s) created, dielectric smoothing is {dielectricsmoothing}.')

        grid.fractalvolumes.append(volume)
