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

import gprMax.config as config
import numpy as np
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


class UserObjectGeometry:
    """Specific Geometry object."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.hash = '#example'
        self.autotranslate = True

    def __str__(self):
        """Readable string of parameters given to object."""
        s = ''
        for _, v in self.kwargs.items():
            if isinstance(v, tuple) or isinstance(v, list):
                v = ' '.join([str(el) for el in v])
            s += str(v) + ' '

        return f'{self.hash}: {s[:-1]}'

    def create(self, grid, uip):
        """Create the object and add it to the grid."""
        pass

    def rotate(self, axis, angle):
        """Rotate geometry object.
        
        Args:
            axis (str): axis about which to perform rotation (x, y, or z)
            angle (int): angle of rotation (degrees)
        """

        orig_p1 = self.kwargs['p1']
        orig_p2 = self.kwargs['p2']
        p1 = np.array([self.kwargs['p1']])
        p2 = np.array([self.kwargs['p2']])

        # Check angle value is suitable
        if angle < 0 or angle > 360:
            logger.exception(self.__str__() + ' angle of rotation must be between 0-360 degrees')
            raise ValueError
        if angle % 90 != 0:
            logger.exception(self.__str__() + ' angle of rotation must be a multiple of 90 degrees')
            raise ValueError

        if axis != 'x' and axis != 'y' and axis != 'z':
            logger.exception(self.__str__() + ' axis of rotation must be x, y, or z')
            raise ValueError

        # Coordinates for axis of rotation (centre of object)
        offset = p1 + (p2 - p1) / 2

        # Move object to axis of rotation
        p1 -= offset
        p2 -= offset

        # Calculate rotation matrix
        r = R.from_euler(axis, angle, degrees=True)

        # Apply rotation
        p1 = r.apply(p1)
        p2 = r.apply(p2)

        # Move object back to original axis
        p1 += offset
        p2 += offset

        # Get lower left and upper right coordinates to define new object
        tmp = np.concatenate((p1, p2), axis=0)
        p1 = np.min(tmp, axis=0)
        p2 = np.max(tmp, axis=0)

        # For 2D modes check axis of rotation against mode 
        # and correct invariant coordinate
        # mode = config.get_model_config().mode
        mode = 'TMz'
        if mode == 'TMx':
            if axis == 'y' or axis =='z':
                logger.exception(self.__str__() +
                                 ' axis of rotation must be x for TMx mode models')
                raise ValueError
            p1[2] = orig_p1[0]
            p2[2] = orig_p2[0]
        elif mode == 'TMy':
            if axis == 'x' or axis == 'z':
                logger.exception(self.__str__() +
                                 ' axis of rotation must be x for TMy mode models')
                raise ValueError
            p1[2] = orig_p1[1]
            p2[2] = orig_p2[1]
        elif mode == 'TMz':
            if axis == 'x' or axis == 'y':
                logger.exception(self.__str__() +
                                 ' axis of rotation must be x for TMz mode models')
                raise ValueError
            p1[2] = orig_p1[2]
            p2[2] = orig_p2[2]

        # Write points back to original tuple
        self.kwargs['p1'] = tuple(p1)
        self.kwargs['p2'] = tuple(p2)
