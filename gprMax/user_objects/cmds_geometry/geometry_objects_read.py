# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

import logging
import shlex
from pathlib import Path

import h5py
import numpy as np

import gprMax.config as config
from gprMax.cython.geometry_primitives import build_voxels_from_array
from gprMax.geometry_outputs.geometry_objects_read import (
    ReadGeometryObject,
    read_geometry_tag_names,
)
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.material_database import (
    build_material_from_spec,
    load_material_spec,
    material_matches_spec,
)
from gprMax.user_objects.user_objects import GeometryUserObject

logger = logging.getLogger(__name__)


def _legacy_conversion_error(geofile, matfile="materials.txt"):
    command = shlex.join(
        [
            "python",
            "-m",
            "toolboxes.MaterialDatabase",
            "convert-geometry",
            str(geofile),
            str(matfile),
        ]
    )
    return ValueError(
        "GeometryObjectsRead requires keyed HDF5 geometry and a JSON material database. "
        "Legacy text material files are no longer accepted by the simulator. "
        f"Convert the original pair first: {command}. "
        "Use the converted HDF5 file and the printed database name in geometry_objects_read "
        "(API: material_database=...). The converter leaves the source files unchanged."
    )


class GeometryObjectsRead(GeometryUserObject):
    """Allows you to insert pre-defined geometry into a model.

    The geometry is specified using integer arrays in an HDF5 file. These files
    contain ``/material_keys`` which map their compact integer indices to a
    versioned JSON material database. Convert legacy HDF5/text pairs first with
    ``python -m toolboxes.MaterialDatabase convert-geometry``.

    Attributes:
        p1: list of lower left (x,y,z) coordinates in the domain where
            the lower left corner of the geometry array should be
            placed.
        geofile: string path to and filename of the HDF5 file that
            contains an integer array which defines the geometry.
        material_database: database name, without ``.json``, for a keyed
            geometry file. The JSON file must be beside the HDF5 file.
        averaging: optional ``"y"``/``"n"`` flag controlling interface
            averaging when a voxel-only file is reconstructed. The default
            is ``"n"``. Files containing complete ``/ID``, ``/rigidE``, and
            ``/rigidH`` arrays are authoritative and ignore this option.
    """

    @property
    def hash(self):
        return "#geometry_objects_read"

    _allowed_kwargs = frozenset({
        "averaging", "geofile", "material_database", "matfile", "p1",
    })

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs.get("matfile") is not None:
            raise _legacy_conversion_error(kwargs.get("geofile", "geometry.h5"), kwargs["matfile"])
        database = kwargs.get("material_database")
        if isinstance(database, (str, Path)) and str(database).lower().endswith(".txt"):
            raise _legacy_conversion_error(kwargs.get("geofile", "geometry.h5"), database)
        self._declared_tags_cache = None

    def _resolve_geofile(self) -> Path:
        geofile = Path(self.kwargs["geofile"])
        if not geofile.exists():
            geofile = Path(config.sim_config.input_file_path.parent, geofile)
        if not geofile.is_file():
            raise FileNotFoundError(f"Geometry object file '{geofile}' does not exist")
        return geofile

    def _resolve_averaging(self) -> bool:
        """Return the requested voxel-reconstruction averaging policy."""

        averaging = self.kwargs.get("averaging", "n")
        if isinstance(averaging, (bool, np.bool_)):
            return bool(averaging)
        if not isinstance(averaging, str) or averaging.lower() not in {"y", "n"}:
            raise ValueError(f"{self.params_str()} averaging must be 'y', 'n', True, or False")
        return averaging.lower() == "y"

    def declared_geometry_tags(self) -> tuple[str, ...]:
        """Read the compact tag catalogue before destination map allocation."""

        if self._declared_tags_cache is None:
            geofile = self._resolve_geofile()
            with h5py.File(geofile, "r") as geometry:
                self._declared_tags_cache = read_geometry_tag_names(geometry)[1:]
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
        averaging = self._resolve_averaging()
        if material_database is None:
            raise ValueError(
                f"{self.params_str()} requires material_database (a JSON database name)"
            )

        geofile = self._resolve_geofile()

        material_id_map, material_description = self._build_database_material_map(
            grid, geofile, material_database
        )

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
                if averaging:
                    logger.warning(
                        f"{self.grid_name(grid)}Geometry file {geofile} contains complete "
                        "/ID, /rigidE, and /rigidH component arrays; averaging='y' is ignored "
                        "because these arrays define the requested Yee-component model."
                    )
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
                    f" Interface averaging is {'enabled' if averaging else 'disabled'}."
                )

    def _build_database_material_map(self, grid, geofile, database):
        # Geometry databases are a paired artefact and therefore resolve
        # beside the HDF5 file. This remains deterministic even when an API
        # model writes outputs to a directory other than its working folder.
        search_directory = Path(geofile).parent
        with h5py.File(geofile, "r") as geometry:
            if "/material_keys" not in geometry:
                raise _legacy_conversion_error(geofile)
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
        specs = [
            load_material_spec(database, key, search_directory=search_directory) for key in keys
        ]
        key_indices = {key: index for index, key in enumerate(keys)}
        directional_indices = {}
        for index, spec in enumerate(specs):
            if "directional_materials" not in spec.metadata:
                continue
            references = spec.metadata["directional_materials"]
            if (
                spec.model != "constant"
                or not isinstance(references, list)
                or len(references) != 3
                or any(not isinstance(key, str) or key not in key_indices for key in references)
            ):
                raise ValueError(
                    "Invalid x/y/z directional material references in geometry database"
                )
            indices = tuple(key_indices[key] for key in references)
            if any(
                specs[item].metadata.get("directional_materials") is not None for item in indices
            ):
                raise ValueError("Nested directional material references are not supported")
            directional_indices[index] = indices
        if directional_indices:
            with h5py.File(geofile, "r") as geometry:
                if not all(name in geometry for name in ("ID", "rigidE", "rigidH")):
                    raise ValueError(
                        "Directional geometry requires complete ID, rigidE and rigidH arrays; "
                        "voxel-only scalar reconstruction would lose its anisotropy"
                    )

        # Restore scalar constituents first, even if a cropped export sorted
        # the cell record before them. Then bind tensors to the resolved local
        # objects; namespace changes cannot swap their physical definitions.
        order = [i for i in range(len(keys)) if i not in directional_indices] + list(
            directional_indices
        )
        for index in order:
            key, spec = keys[index], specs[index]
            axes = (
                tuple(grid.materials[material_id_map[item]] for item in directional_indices[index])
                if index in directional_indices
                else None
            )
            if axes is not None:
                densities = tuple(material.mass_density for material in axes)
                expected_density = (
                    densities[0] if all(value == densities[0] for value in densities) else None
                )
                if spec.mass_density != expected_density:
                    raise ValueError(
                        "Directional cell density must agree with its three constituents"
                    )
            original_id = spec.metadata.get("original_id", key)
            if not isinstance(original_id, str) or not original_id:
                raise ValueError(f"Material '{database}:{key}' metadata original_id is invalid")
            namespaced_id = f"{original_id}{{{namespace}}}"
            original = existing_by_id.get(original_id)
            if original is not None and material_matches_spec(
                original, spec, directional_materials=axes
            ):
                material_id_map[index] = original.numID
                continue

            namespaced = existing_by_id.get(namespaced_id)
            if namespaced is not None:
                if not material_matches_spec(namespaced, spec, directional_materials=axes):
                    raise ValueError(
                        f"Geometry material '{namespaced_id}' conflicts with a material already "
                        "defined in the model"
                    )
                material_id_map[index] = namespaced.numID
                continue

            # An unrelated model material may legitimately use the original
            # CAD/material ID with different properties. Keep both by giving
            # the imported definition its deterministic database namespace.
            created = build_material_from_spec(
                grid, spec, namespaced_id, directional_materials=axes
            )
            created.type = f"{created.type},\nimported" if created.type else "imported"
            existing_by_id[namespaced_id] = created
            material_id_map[index] = created.numID
        return material_id_map, f"material database {database}"
