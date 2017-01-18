# Copyright (C) 2015-2017: The University of Edinburgh
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

from collections import OrderedDict

from gprMax.grid import Ix, Iy, Iz


class Rx(object):
    """Receiver output points."""

    availableoutputs = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 'Ix', 'Iy', 'Iz']
    defaultoutputs = availableoutputs[:-3]

    def __init__(self):

        self.ID = None
        self.outputs = OrderedDict()
        self.xcoord = None
        self.ycoord = None
        self.zcoord = None
        self.xcoordorigin = None
        self.ycoordorigin = None
        self.zcoordorigin = None


def store_outputs(timestep, Ex, Ey, Ez, Hx, Hy, Hz, G):
    """Stores field component values for every receiver and transmission line.

    Args:
        timestep (int): Current iteration number.
        Ex, Ey, Ez, Hx, Hy, Hz (memory view): Current electric and magnetic field values.
        G (class): Grid class instance - holds essential parameters describing the model.
    """

    for rx in G.rxs:
        if 'Ex' in rx.outputs:
            rx.outputs['Ex'][timestep] = Ex[rx.xcoord, rx.ycoord, rx.zcoord]
        if 'Ey' in rx.outputs:
            rx.outputs['Ey'][timestep] = Ey[rx.xcoord, rx.ycoord, rx.zcoord]
        if 'Ez' in rx.outputs:
            rx.outputs['Ez'][timestep] = Ez[rx.xcoord, rx.ycoord, rx.zcoord]
        if 'Hx' in rx.outputs:
            rx.outputs['Hx'][timestep] = Hx[rx.xcoord, rx.ycoord, rx.zcoord]
        if 'Hy' in rx.outputs:
            rx.outputs['Hy'][timestep] = Hy[rx.xcoord, rx.ycoord, rx.zcoord]
        if 'Hz' in rx.outputs:
            rx.outputs['Hz'][timestep] = Hz[rx.xcoord, rx.ycoord, rx.zcoord]
        if 'Ix' in rx.outputs:
            rx.outputs['Ix'][timestep] = Ix(rx.xcoord, rx.ycoord, rx.zcoord, Hy, Hz, G)
        if 'Iy' in rx.outputs:
            rx.outputs['Iy'][timestep] = Iy(rx.xcoord, rx.ycoord, rx.zcoord, Hx, Hz, G)
        if 'Iz' in rx.outputs:
            rx.outputs['Iz'][timestep] = Iz(rx.xcoord, rx.ycoord, rx.zcoord, Hx, Hy, G)

    for tlindex, tl in enumerate(G.transmissionlines):
        tl.Vtotal[timestep] = tl.voltage[tl.antpos]
        tl.Itotal[timestep] = tl.current[tl.antpos]
