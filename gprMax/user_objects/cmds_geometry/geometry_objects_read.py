# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
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
from pathlib import Path

import h5py

import gprMax.config as config
from gprMax.cython.geometry_primitives import build_voxels_from_array
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.hash_cmds_file import get_user_objects
from gprMax.output_controllers.read_geometry_object import ReadGeometryObject
from gprMax.user_objects.user_objects import GeometryUserObject

logger = logging.getLogger(__name__)


class GeometryObjectsRead(GeometryUserObject):
    @property
    def hash(self):
        return "#geometry_objects_read"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, grid: FDTDGrid):
        """Creates the object and adds it to the grid."""
        try:
            p1 = self.kwargs["p1"]
            geofile = self.kwargs["geofile"]
            matfile = self.kwargs["matfile"]
        except KeyError:
            logger.exception(f"{self.__str__()} requires exactly five parameters")
            raise

        # See if material file exists at specified path and if not try input
        # file directory
        matfile = Path(matfile)

        if not matfile.exists():
            matfile = Path(config.sim_config.input_file_path.parent, matfile)

        matstr = matfile.with_suffix("").name
        numexistmaterials = len(grid.materials)

        # Read materials from file
        with open(matfile, "r") as f:
            # Read any lines that begin with a hash. Strip out any newline
            # characters and comments that must begin with double hashes.
            materials = [
                line.rstrip() + "{" + matstr + "}\n"
                for line in f
                if (line.startswith("#") and not line.startswith("##") and line.rstrip("\n"))
            ]

        # Avoid redefining default builtin materials
        pec = f"#material: 1 inf 1 0 pec{{{matstr}}}\n"
        free_space = f"#material: 1 0 1 0 free_space{{{matstr}}}\n"
        if materials[0] == pec and materials[1] == free_space:
            materials.pop(0)
            materials.pop(1)
            numexistmaterials -= 2
        elif materials[0] == pec or materials[0] == free_space:
            materials.pop(0)
            numexistmaterials -= 1

        # Build scene
        # API for multiple scenes / model runs
        scene = config.get_model_config().get_scene()
        assert scene is not None
        material_objs = get_user_objects(materials, checkessential=False)
        for material_obj in material_objs:
            scene.add(material_obj)

        # Creates the internal simulation objects
        scene.build_grid_objects(material_objs, grid)

        # Update material type
        for material in grid.materials:
            if material.numID >= numexistmaterials:
                if material.type:
                    material.type += ",\nimported"
                else:
                    material.type = "imported"

        # See if geometry object file exists at specified path and if not try
        # input file directory.
        geofile = Path(geofile)
        if not geofile.exists():
            geofile = Path(config.sim_config.input_file_path.parent, geofile)

        # Discretise the point using uip object. This has different behaviour
        # depending on the type of uip object. So we can use it for
        # the main grid, MPI grids or the subgrid.
        uip = self._create_uip(grid)
        discretised_p1 = uip.discretise_point(p1)
        p2 = uip.round_to_grid_static_point(p1)

        with ReadGeometryObject(geofile, grid, discretised_p1, numexistmaterials) as f:
            # Check spatial resolution attribute
            if not f.has_valid_discritisation():
                raise ValueError(
                    f"{self.__str__()} requires the spatial resolution "
                    "of the geometry objects file to match the spatial "
                    "resolution of the model"
                )

            if f.has_rigid_arrays() and f.has_ID_array():
                f.read_data()
                f.read_ID()
                f.read_rigidE()
                f.read_rigidH()

                logger.info(
                    f"{self.grid_name(grid)}Geometry objects from file {geofile}"
                    f" inserted at {p2[0]:g}m, {p2[1]:g}m, {p2[2]:g}m,"
                    f" with corresponding materials file"
                    f" {matfile}."
                )
            else:
                data = f.get_data()
                if data is not None:
                    averaging = False
                    build_voxels_from_array(
                        discretised_p1[0],
                        discretised_p1[1],
                        discretised_p1[2],
                        numexistmaterials,
                        averaging,
                        data,
                        grid.solid,
                        grid.rigidE,
                        grid.rigidH,
                        grid.ID,
                    )
                logger.info(
                    f"{self.grid_name(grid)}Geometry objects from file "
                    f"(voxels only){geofile} inserted at {p2[0]:g}m, "
                    f"{p2[1]:g}m, {p2[2]:g}m, with corresponding "
                    f"materials file {matfile}."
                )
