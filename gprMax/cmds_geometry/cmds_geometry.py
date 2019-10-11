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

import logging

log = logging.getLogger(__name__)


class UserObjectGeometry:
    """Specific Geometry object."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # define the order of priority for calling create()
        self.order = None
        # hash command
        self.hash = '#example'

        # auto translate
        self.autotranslate = True

    def __str__(self):
        """Readble user string as per hash commands."""
        s = ''
        for k, v in self.kwargs.items():
            if isinstance(v, tuple) or isinstance(v, list):
                v = ' '.join([str(el) for el in v])
            s += str(v) + ' '

        return f'{self.hash}: {s[:-1]}'

    def params_str(self):
        """Readble string of parameters given to object."""
        return self.hash + ': ' + str(self.kwargs)

    def create(self, grid, uip):
        """Create the object and add it to the grid."""
        log.debug('This method is incomplete')
