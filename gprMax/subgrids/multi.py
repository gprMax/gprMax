# Copyright (C) 2015-2022: The University of Edinburgh, United Kingdom
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
        """Field value at the equivalent coarse yee cell.

        Args:
            str_id: string of 'Ex' etc...
            field: nparray of grid.Ez

        Returns:
            e: float of field value.
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
        """Ex field value from the equivalent coarse yee cell.

        Args:
            Ex: nparray of Ex field.

        Returns:
            e: float of Ex field value.
        """

        # offset = ratio // 2
        e = Ex[self.xcoord + self.offset, self.ycoord, self.zcoord]
        return e

    def get_Ey_from_field(self, Ey):
        """Ey field value from the equivalent coarse yee cell.

        Args:
            Ey: nparray of Ey field.

        Returns:
            e: float of Ey field value.
        """
        e = Ey[self.xcoord, self.ycoord + self.offset, self.zcoord]
        return e

    def get_Ez_from_field(self, Ez):
        """Ez field value from the equivalent coarse yee cell.

        Args:
            Ez: nparray of Ez field.

        Returns:
            e: float of Ez field value.
        """

        e = Ez[self.xcoord, self.ycoord, self.zcoord + self.offset]
        return e

    def get_Hx_from_field(self, Hx):
        """Hx field value from the equivalent coarse yee cell.

        Args:
            Hx: nparray of Hx field.

        Returns:
            e: float of Hx field value.
        """
        e = Hx[self.xcoord, self.ycoord + self.offset, self.zcoord + self.offset]
        return e

    def get_Hy_from_field(self, Hy):
        """Hy field value from the equivalent coarse yee cell.

        Args:
            Hy: nparray of Hy field.

        Returns:
            e: float of Hy field value.
        """
        e = Hy[self.xcoord + self.offset, self.ycoord, self.zcoord + self.offset]
        return e

    def get_Hz_from_field(self, Hz):
        """Hz field value from the equivalent coarse yee cell.

        Args:
            Hz: nparray of Hz field.

        Returns:
            e: float of Hz field value.
        """
        e = Hz[self.xcoord + self.offset, self.ycoord + self.offset, self.zcoord]
        return e