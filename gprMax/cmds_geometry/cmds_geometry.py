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

    def rotate(self, axis, angle, origin=None):
        """Rotate object - specialised for each object."""
        pass

    def rotate_point(self, p, axis, angle, origin=(0, 0, 0)):
        """Rotate a point.
        
        Args:
            p (array): coordinates of point (x, y, z)
            axis (str): axis about which to perform rotation (x, y, or z)
            angle (int): angle of rotation (degrees)
            origin (tuple): point about which to perform rotation (x, y, z)

        Returns:
            p (array): coordinates of rotated point (x, y, z)
        """

        origin = np.array(origin)

        # Move point to axis of rotation
        p -= origin

        # Calculate rotation matrix
        r = R.from_euler(axis, angle, degrees=True)

        # Apply rotation
        p = r.apply(p)

        # Move object back to original axis
        p += origin

        return p
        
    def rotate_2point_object(self, pts, axis, angle, origin=None):
        """Rotate a geometry object that is defined by 2 points.
        
        Args:
            pts (array): coordinates of points of object to be rotated
            axis (str): axis about which to perform rotation (x, y, or z)
            angle (int): angle of rotation (degrees)
            origin (tuple): point about which to perform rotation (x, y, z)

        Returns:
            new_pts (array): coordinates of points of rotated object
        """

        # Use origin at centre of object if not given
        if not origin:
            origin = pts[0,:] + (pts[1,:] - pts[0,:]) / 2

        # Check angle value is suitable
        angle = int(angle)
        if angle < 0 or angle > 360:
            logger.exception(self.__str__() + ' angle of rotation must be between 0-360 degrees')
            raise ValueError
        if angle % 90 != 0:
            logger.exception(self.__str__() + ' angle of rotation must be a multiple of 90 degrees')
            raise ValueError

        # Check axis is valid
        if axis != 'x' and axis != 'y' and axis != 'z':
            logger.exception(self.__str__() + ' axis of rotation must be x, y, or z')
            raise ValueError

        # Save original points
        orig_pts = pts

        # Rotate points that define object
        pts[0, :] = self.rotate_point(pts[0, :], axis, angle, origin)
        pts[1, :] = self.rotate_point(pts[1, :], axis, angle, origin)

        # Get lower left and upper right coordinates to define new object
        new_pts = np.zeros(pts.shape)
        new_pts[0, :] = np.min(pts, axis=0)
        new_pts[1, :] = np.max(pts, axis=0)

        # Reset coordinates of invariant direction 
        # - only needed for 2D models, has no effect on 3D models.
        if axis =='x':
            new_pts[0, 0] = orig_pts[0, 0]
            new_pts[1, 0] = orig_pts[1, 0]
        elif axis == 'y':
            new_pts[0, 1] = orig_pts[0, 1]
            new_pts[1, 1] = orig_pts[1, 1]
        elif axis == 'z':
            new_pts[0, 2] = orig_pts[0, 2]
            new_pts[1, 2] = orig_pts[1, 2]

        return new_pts
