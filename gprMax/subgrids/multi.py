# Copyright (C) 2015-2022: The University of Edinburgh
#                 Authors: Craig Warren, Antonis Giannopoulos, and John Hartley
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

from ..receivers import Rx

logger = logging.getLogger(__name__)


class ReferenceRx(Rx):
    """Receiver that mimicks a receiver in coarse grid.
        We often want to compare an output in a fine reference solution with a
        the solution in the coarse grid of a subgridded solution. This receiver
        moves the output points in the fine grid such that they are in the same
        position as the coarse grid.
    """

    logger.debug('ReferenceRx has no offset member.')

    def get_field(self, str_id, field):
        """Return the field value at the equivalent coarse yee cell.

        Args:
            str_id (str): 'Ex' etc...
            field (nparray): Numpy array of grid.Ez

        Returns:
            e (float): Field value.
        """

        d = {
            'Ex': self.get_Ex_from_field,
            'Ey': self.get_Ey_from_field,
            'Ez': self.get_Ez_from_field,
            'Hx': self.get_Hx_from_field,
            'Hy': self.get_Hy_from_field,
            'Hz': self.get_Hz_from_field
        }

        e = d[str_id](field)

        return e

    def get_Ex_from_field(self, Ex):
        """Return the Ex field value from the equivalent coarse yee cell.

        Args:
            Ex (nparray): Numpy array of Ex field.

        Returns:
            e (float): Ex field value.
        """

        # offset = ratio // 2
        e = Ex[self.xcoord + self.offset, self.ycoord, self.zcoord]
        return e

    def get_Ey_from_field(self, Ey):
        """Return the Ey field value from the equivalent coarse yee cell.

        Args:
            Ey (nparray): Numpy array of Ey field.

        Returns:
            e (float): Ey field value.
        """
        e = Ey[self.xcoord, self.ycoord + self.offset, self.zcoord]
        return e

    def get_Ez_from_field(self, Ez):
        """Return the Ez field value from the equivalent coarse yee cell.

        Args:
            Ez (nparray): Numpy array of Ez field.

        Returns:
            e (float): Ez field value.
        """

        e = Ez[self.xcoord, self.ycoord, self.zcoord + self.offset]
        return e

    def get_Hx_from_field(self, Hx):
        """Return the Hx field value from the equivalent coarse yee cell.

        Args:
            Hx (nparray): Numpy array of Hx field.

        Returns:
            e (float): Hx field value.
        """
        e = Hx[self.xcoord, self.ycoord + self.offset, self.zcoord + self.offset]
        return e

    def get_Hy_from_field(self, Hy):
        """Return the Hy field value from the equivalent coarse yee cell.

        Args:
            Hy (nparray): Numpy array of Hy field.

        Returns:
            e (float): Hy field value.
        """
        e = Hy[self.xcoord + self.offset, self.ycoord, self.zcoord + self.offset]
        return e

    def get_Hz_from_field(self, Hz):
        """Return the Hz field value from the equivalent coarse yee cell.

        Args:
            Hz (nparray): Numpy array of Hz field.

        Returns:
            e (float): Hz field value.
        """
        e = Hz[self.xcoord + self.offset, self.ycoord + self.offset, self.zcoord]
        return e
