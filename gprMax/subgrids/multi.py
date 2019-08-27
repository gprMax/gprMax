# Copyright (C) 2015-2019: The University of Edinburgh
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
from ..receivers import Rx


class ReferenceRx(Rx):
    """Receiver that micks a receiver in coarse grid."""

    """We often want to compare an output in a fine reference solution with a
    the solution in the coarse grid of a subgridded solution. This receiver
    moves the output points in the fine grid such that they are in the same
    position as the coarse grid.
    """
    def __init__(self):
        """Constructor."""
        super().__init__()

    def get_field(self, str_id, field):
        """Return the field value at the equivalent coarse yee cell.

        Parameters
        ----------
        str_id : str
            'Ex' etc...
        field : np array
            e.g. numpy array of grid.Ez

        Returns
        -------
        float
            Field value

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

        Parameters
        ----------
        Ex : 3d numpy array
            e.g. grid.Ex

        Returns
        -------
        float
            Ex field value

        """

        # offset = ratio // 2
        e = Ex[self.xcoord + self.offset, self.ycoord, self.zcoord]
        return e

    def get_Ey_from_field(self, Ey):
        """Return the Ey field value from the equivalent coarse yee cell.

        Parameters
        ----------
        Ex : 3d numpy array
            e.g. grid.Ex

        Returns
        -------
        float
            Ey field value

        """
        e = Ey[self.xcoord, self.ycoord + self.offset, self.zcoord]
        return e

    def get_Ez_from_field(self, Ez):
        """Return the Ez field value from the equivalent coarse yee cell.

        Parameters
        ----------
        Ex : 3d numpy array
            e.g. grid.Ez

        Returns
        -------
        float
            Ez field value

        """
        e = Ez[self.xcoord, self.ycoord, self.zcoord + self.offset]
        return e

    def get_Hx_from_field(self, Hx):
        """Return the Hx field value from the equivalent coarse yee cell.

        Parameters
        ----------
        Ex : 3d numpy array
            e.g. grid.Hx

        Returns
        -------
        float
            Hx field value

        """
        e = Hx[self.xcoord, self.ycoord + self.offset, self.zcoord + self.offset]
        return e

    def get_Hy_from_field(self, Hy):
        """Return the Hy field value from the equivalent coarse yee cell.

        Parameters
        ----------
        Ex : 3d numpy array
            e.g. grid.Hx

        Returns
        -------
        float
            Hy field value

        """
        e = Hy[self.xcoord + self.offset, self.ycoord, self.zcoord + self.offset]
        return e

    def get_Hz_from_field(self, Hz):
        """Return the Hz field value from the equivalent coarse yee cell.

        Parameters
        ----------
        Ex : 3d numpy array
            e.g. grid.Hx

        Returns
        -------
        float
            Hz field value

        """
        e = Hz[self.xcoord + self.offset, self.ycoord + self.offset, self.zcoord]
        return e
