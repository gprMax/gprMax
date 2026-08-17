# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley,
#                          and Nathan Mannall
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
import numpy as np

import gprMax.config as config
from gprMax.cython.geometry_primitives import build_voxels_from_array
from gprMax.geometry_outputs.geometry_objects_read import ReadGeometryObject
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.hash_cmds_file import get_user_objects
from gprMax.material_database import (
    build_material_from_spec,
    load_material_spec,
    material_matches_spec,
)
from gprMax.user_objects.user_objects import GeometryUserObject

logger = logging.getLogger(__name__)


class GeometryObjectsRead(GeometryUserObject):
    """Allows you to insert pre-defined geometry into a model.

    The geometry is specified using integer arrays in an HDF5 file. New files
    contain ``/material_keys`` which map their compact integer indices to a
    versioned JSON material database. Legacy ``#material`` text files remain
    readable for compatibility.

    Attributes:
        p1: list of lower left (x,y,z) coordinates in the domain where
            the lower left corner of the geometry array should be
            placed.
        geofile: string path to and filename of the HDF5 file that
            contains an integer array which defines the geometry.
        material_database: database name for a new-format geometry file.
        matfile: legacy text material file (deprecated).
    """

    @property
    def hash(self):
        return "#geometry_objects_read"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._declared_tags_cache = None

    def _resolve_geofile(self) -> Path:
        geofile = Path(self.kwargs["geofile"])
        if not geofile.exists():
            geofile = Path(config.sim_config.input_file_path.parent, geofile)
        if not geofile.is_file():
            raise FileNotFoundError(f"Geometry object file '{geofile}' does not exist")
        return geofile

    def declared_geometry_tags(self) -> tuple[str, ...]:
        """Read the compact tag catalogue before destination map allocation."""

        if self._declared_tags_cache is None:
            geofile = self._resolve_geofile()
            with h5py.File(geofile, "r") as geometry:
                if "/tag_names" not in geometry:
                    self._declared_tags_cache = ()
                else:
                    raw_names = geometry["/tag_names"][:]
                    names = tuple(
                        value.decode("utf-8") if isinstance(value, bytes) else str(value)
                        for value in raw_names
                    )
                    if not names or names[0] != "untagged":
                        raise ValueError(
                            f"Geometry file '{geofile}' has an invalid /tag_names catalogue"
                        )
                    self._declared_tags_cache = names[1:]
        return self._declared_tags_cache

    def build(self, grid: FDTDGrid):
        """Creates the object and adds it to the grid."""
        try:
            p1 = self.kwargs["p1"]
            geofile = self.kwargs["geofile"]
        except KeyError:
            logger.exception(f"{self.__str__()} requires exactly five parameters")
            raise
        material_database = self.kwargs.get("material_database")
        matfile = self.kwargs.get("matfile")
        if (material_database is None) == (matfile is None):
            raise ValueError(
                f"{self.params_str()} requires exactly one of material_database or legacy matfile"
            )

        geofile = self._resolve_geofile()

        if material_database is not None:
            material_id_map, material_description = self._build_database_material_map(
                grid, geofile, material_database
            )
        else:
            material_id_map, material_description = self._build_legacy_material_map(grid, matfile)

        # Discretise the point using uip object. This has different behaviour
        # depending on the type of uip object. So we can use it for
        # the main grid, MPI grids or the subgrid.
        uip = self._create_uip(grid)
        p1 = uip.resolve_inf_point(p1, role="lower")
        discretised_p1 = uip.discretise_point(p1)
        p2 = uip.round_to_grid_static_point(p1)

        mode = config.get_model_config().mode
        invariant_axis = None
        target_invariant_size = None
        if mode.startswith("2D"):
            invariant_axis = "xyz".index(mode[-1])
            target_invariant_size = 2 if "TE" in mode else 1

            with h5py.File(geofile, "r") as check_file:
                file_invariant_size = check_file["/data"].shape[invariant_axis]
            if file_invariant_size != target_invariant_size:
                action = (
                    "broadcasting" if file_invariant_size < target_invariant_size else "reducing"
                )
                logger.info(
                    f"{self.__str__()} imported file has {file_invariant_size} cell(s) on the "
                    f"invariant axis but this model ({mode}) needs {target_invariant_size} - "
                    f"{action} automatically."
                )

        with ReadGeometryObject(
            geofile,
            grid,
            discretised_p1,
            material_id_map,
            invariant_axis=invariant_axis,
            target_invariant_size=target_invariant_size,
        ) as f:
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
                f.read_tags()

                logger.info(
                    f"{self.grid_name(grid)}Geometry objects from file {geofile}"
                    f" inserted at {p2[0]:g}m, {p2[1]:g}m, {p2[2]:g}m,"
                    f" with {material_description}."
                )
            else:
                data = f.get_data()
                if data is not None:
                    data_start = f.get_local_data_start()
                    assert data_start is not None
                    averaging = False
                    is_pec_lookup = np.array([m.is_pec for m in grid.materials], dtype=np.uint8)
                    is_averagable_lookup = np.array(
                        [m.averagable for m in grid.materials], dtype=np.uint8
                    )
                    build_voxels_from_array(
                        data_start[0],
                        data_start[1],
                        data_start[2],
                        0,
                        averaging,
                        is_pec_lookup,
                        is_averagable_lookup,
                        data,
                        grid.solid,
                        grid.rigidE,
                        grid.rigidH,
                        grid.ID,
                    )
                f.read_tags()
                logger.info(
                    f"{self.grid_name(grid)}Geometry objects from file "
                    f"(voxels only) {geofile} inserted at {p2[0]:g}m, "
                    f"{p2[1]:g}m, {p2[2]:g}m, with {material_description}."
                )

    def _build_database_material_map(self, grid, geofile, database):
        # Geometry databases are a paired artefact and therefore resolve
        # beside the HDF5 file. This remains deterministic even when an API
        # model writes outputs to a directory other than its working folder.
        search_directory = Path(geofile).parent
        with h5py.File(geofile, "r") as geometry:
            if "/material_keys" not in geometry:
                raise ValueError(
                    f"Geometry file '{geofile}' has no /material_keys dataset; "
                    "supply its legacy matfile instead"
                )
            recorded_database = geometry.attrs.get("MaterialDatabase")
            if isinstance(recorded_database, bytes):
                recorded_database = recorded_database.decode("utf-8")
            if recorded_database is not None and recorded_database != database:
                raise ValueError(
                    f"Geometry file '{geofile}' records material database "
                    f"'{recorded_database}', not '{database}'"
                )
            raw_keys = geometry["/material_keys"][:]
        keys = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in raw_keys
        ]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Geometry file '{geofile}' contains duplicate material keys")

        existing_by_id = {material.ID: material for material in grid.materials}
        material_id_map = np.empty(len(keys), dtype=np.int32)
        namespace = database
        for index, key in enumerate(keys):
            spec = load_material_spec(database, key, search_directory=search_directory)
            original_id = spec.metadata.get("original_id", key)
            if not isinstance(original_id, str) or not original_id:
                raise ValueError(f"Material '{database}:{key}' metadata original_id is invalid")
            namespaced_id = f"{original_id}{{{namespace}}}"
            original = existing_by_id.get(original_id)
            if original is not None and material_matches_spec(original, spec):
                material_id_map[index] = original.numID
                continue

            namespaced = existing_by_id.get(namespaced_id)
            if namespaced is not None:
                if not material_matches_spec(namespaced, spec):
                    raise ValueError(
                        f"Geometry material '{namespaced_id}' conflicts with a material already "
                        "defined in the model"
                    )
                material_id_map[index] = namespaced.numID
                continue

            # An unrelated model material may legitimately use the original
            # CAD/material ID with different properties. Keep both by giving
            # the imported definition its deterministic database namespace.
            created = build_material_from_spec(grid, spec, namespaced_id)
            created.type = f"{created.type},\nimported" if created.type else "imported"
            existing_by_id[namespaced_id] = created
            material_id_map[index] = created.numID
        return material_id_map, f"material database {database}"

    def _build_legacy_material_map(self, grid, matfile):
        """Build the mapping used by pre-database geometry object pairs."""

        # See if material file exists at specified path and if not try input
        # file directory
        matfile = Path(matfile)

        if not matfile.exists():
            matfile = Path(config.sim_config.input_file_path.parent, matfile)

        matstr = matfile.with_suffix("").name

        # Read materials from file. Strip out any newline characters and
        # comments that must begin with double hashes.
        with open(matfile, "r") as f:
            raw_lines = [
                line.rstrip()
                for line in f
                if (line.startswith("#") and not line.startswith("##") and line.rstrip("\n"))
            ]

        # The file's /data and /ID arrays use 0-based indices into the
        # *materials* only (in file order) - any #add_dispersion_* line
        # describes the material immediately preceding it and doesn't get
        # its own index. Group lines by material so each group can be
        # checked, by ID name, against materials already in this grid -
        # this covers the builtin pec/pmc/free_space (in whatever order or
        # subset the exporting region actually used) and anything already
        # read in from a previous #geometry_objects_read, rather than
        # assuming a fixed count/order of builtin materials.
        material_groups = []
        for line in raw_lines:
            if line.startswith("#material:"):
                material_id = line.rsplit(" ", 1)[-1]
                material_groups.append((material_id, [line]))
            else:
                material_groups[-1][1].append(line)

        existing_by_id = {m.ID: m.numID for m in grid.materials}
        material_id_map = np.empty(len(material_groups), dtype=np.int32)
        lines_to_build = []
        groups_to_build = []
        for index, (material_id, lines) in enumerate(material_groups):
            if material_id in existing_by_id:
                # Already exists in this grid (e.g. a builtin) - reuse it
                # rather than redeclaring it, which would either collide
                # with the existing material or create a needless duplicate.
                material_id_map[index] = existing_by_id[material_id]
            else:
                namespaced_id = f"{material_id}{{{matstr}}}"
                lines_to_build.extend(f"{line}{{{matstr}}}\n" for line in lines)
                groups_to_build.append((index, namespaced_id))

        # Build scene
        # API for multiple scenes / model runs
        scene = config.get_model_config().get_scene()
        assert scene is not None
        material_objs = get_user_objects(lines_to_build, checkessential=False)
        for material_obj in material_objs:
            scene.add(material_obj)

        # Creates the internal simulation objects
        scene.build_grid_objects(material_objs, grid)

        # Fill in numIDs for the materials that were actually (re)built, and
        # tag them as imported. Materials reused from the existing grid
        # (e.g. builtins) keep their original type/numID untouched.
        materials_by_id = {m.ID: m for m in grid.materials}
        for index, namespaced_id in groups_to_build:
            material = materials_by_id[namespaced_id]
            material_id_map[index] = material.numID
            material.type = f"{material.type},\nimported" if material.type else "imported"
        return material_id_map, f"legacy materials file {matfile}"
