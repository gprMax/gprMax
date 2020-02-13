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
from .cmds_geometry import UserObjectGeometry
from ..exceptions import CmdInputError
from ..materials import Material
from ..utilities import round_value

log = logging.getLogger(__name__)


class AddSurfaceWater(UserObjectGeometry):
    """Allows you to add surface water to a :class:`gprMax.cmds_geometry.fractal_box.FractalBox` in the model.

    :param p1: The lower left (x,y,z) coordinates of a surface on a :class:`gprMax.cmds_geometry.fractal_box.FractalBox`
    :type p1: list, non-optional
    :param p2: The lower left (x,y,z) coordinates of a surface on a :class:`gprMax.cmds_geometry.fractal_box.FractalBox`
    :type p2: list, non-optional
    :param depth: Defines the depth of the water, which should be specified relative to the dimensions of the #fractal_box that the surface water is being applied.
    :type depth: float, non-optional
    :param fractal_box_id:  An identifier for the :class:`gprMax.cmds_geometry.fractal_box.FractalBox` that the water should be applied to
    :type fractal_box_id: list, non-optional
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = 11
        self.hash = '#add_surface_water'

    def create(self, grid, uip):
        """"Create surface water on fractal box."""
        try:
            p1 = self.kwargs['p1']
            p2 = self.kwargs['p2']
            fractal_box_id = self.kwargs['fractal_box_id']
            depth = self.kwargs['depth']
        except KeyError:
            raise CmdInputError(self.__str__() + ' requires exactly eight parameters')

        # Get the correct fractal volume
        volumes = [volume for volume in grid.fractalvolumes if volume.ID == fractal_box_id]
        if volumes:
            volume = volumes[0]
        else:
            raise CmdInputError(self.__str__() + f' cannot find FractalBox {fractal_box_id}')

        p1, p2 = uip.check_box_points(p1, p2, self.__str__())
        xs, ys, zs = p1
        xf, yf, zf = p2

        if depth <= 0:
            raise CmdInputError(self.__str__() + ' requires a positive value for the depth of water')

        # Check for valid orientations
        if xs == xf:
            if ys == yf or zs == zf:
                raise CmdInputError(self.__str__() + ' dimensions are not specified correctly')
            if xs != volume.xs and xs != volume.xf:
                raise CmdInputError(self.__str__() + ' can only be used on the external surfaces of a fractal box')
            # xminus surface
            if xs == volume.xs:
                requestedsurface = 'xminus'
            # xplus surface
            elif xf == volume.xf:
                requestedsurface = 'xplus'
            filldepthcells = round_value(depth / grid.dx)
            filldepth = filldepthcells * grid.dx

        elif ys == yf:
            if xs == xf or zs == zf:
                raise CmdInputError(self.__str__() + ' dimensions are not specified correctly')
            if ys != volume.ys and ys != volume.yf:
                raise CmdInputError(self.__str__() + ' can only be used on the external surfaces of a fractal box')
            # yminus surface
            if ys == volume.ys:
                requestedsurface = 'yminus'
            # yplus surface
            elif yf == volume.yf:
                requestedsurface = 'yplus'
            filldepthcells = round_value(depth / grid.dy)
            filldepth = filldepthcells * grid.dy

        elif zs == zf:
            if xs == xf or ys == yf:
                raise CmdInputError(self.__str__() + ' dimensions are not specified correctly')
            if zs != volume.zs and zs != volume.zf:
                raise CmdInputError(self.__str__() + ' can only be used on the external surfaces of a fractal box')
            # zminus surface
            if zs == volume.zs:
                requestedsurface = 'zminus'
            # zplus surface
            elif zf == volume.zf:
                requestedsurface = 'zplus'
            filldepthcells = round_value(depth / grid.dz)
            filldepth = filldepthcells * grid.dz

        else:
            raise CmdInputError(self.__str__() + ' dimensions are not specified correctly')

        surface = next((x for x in volume.fractalsurfaces if x.surfaceID == requestedsurface), None)
        if not surface:
            raise CmdInputError(self.__str__() + ' specified surface {} does not have a rough surface applied'.format(requestedsurface))

        surface.filldepth = filldepthcells

        # Check that requested fill depth falls within range of surface roughness
        if surface.filldepth < surface.fractalrange[0] or surface.filldepth > surface.fractalrange[1]:
            raise CmdInputError(self.__str__() + ' requires a value for the depth of water that lies with the range of the requested surface roughness')

        # Check to see if water has been already defined as a material
        if not any(x.ID == 'water' for x in grid.materials):
            m = Material(len(grid.materials), 'water')
            m.averagable = False
            m.type = 'builtin, debye'
            m.er = Material.watereri
            m.deltaer.append(Material.waterdeltaer)
            m.tau.append(Material.watertau)
            grid.materials.append(m)
            if Material.maxpoles == 0:
                Material.maxpoles = 1

        # Check if time step for model is suitable for using water
        water = next((x for x in grid.materials if x.ID == 'water'))
        testwater = next((x for x in water.tau if x < grid.dt), None)
        if testwater:
            raise CmdInputError(self.__str__() + ' requires the time step for the model to be less than the relaxation time required to model water.')

        log.info(f'Water on surface from {xs * grid.dx:g}m, {ys * grid.dy:g}m, {zs * grid.dz:g}m, to {xf * grid.dx:g}m, {yf * grid.dy:g}m, {zf * grid.dz:g}m with depth {filldepth:g}m, added to {surface.operatingonID}.')
