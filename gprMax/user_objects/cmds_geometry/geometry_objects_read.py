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
from gprMax.impedance_surfaces import (
    create_impedance_marker_material,
    impedance_marker_id,
    is_reserved_impedance_id,
    private_impedance_material_id,
    restore_private_impedance_material,
    restore_surface_impedance_model,
)
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
            "gprMax.toolboxes.MaterialDatabase",
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
    ``python -m gprMax.toolboxes.MaterialDatabase convert-geometry``.

    Args:
        p1 (tuple of three floats): Lower left (x, y, z) coordinates, in metres, in
            the domain where the lower left corner of the geometry array is placed.
            Required.
        geofile (str): Path to and filename of the HDF5 file that contains an
            integer array which defines the geometry. Required.
        material_database (str): Database name, without ``.json``, for a keyed
            geometry file. The JSON file must be beside the HDF5 file. Required
            (gprMax refuses to build without it).
        averaging (str or bool): Optional flag controlling interface averaging
            when a voxel-only file is reconstructed: 'y' or 'n', or True or False.
            The default is ``"n"``. Files containing complete ``/ID``,
            ``/rigidE``, and ``/rigidH`` arrays are authoritative and ignore this
            option.
        matfile (str or None): Legacy text material file. No longer accepted: if
            given, gprMax raises an error explaining how to convert the HDF5/text
            pair with
            ``python -m gprMax.toolboxes.MaterialDatabase convert-geometry`` and
            use ``material_database`` instead. Optional (default None).
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

    def declares_surface_impedance(self) -> bool:
        """Return whether the paired geometry database exports SIBC models."""

        try:
            geofile = self._resolve_geofile()
        except FileNotFoundError:
            return False
        with h5py.File(geofile, "r") as geometry:
            return bool(geometry.attrs.get("ContainsSurfaceImpedance", 0))

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
        if not np.allclose(p1, p2, rtol=0.0, atol=1e-6 * np.min(uip.grid.dl)):
            logger.warning(
                f"{self.__str__()} p1 ({p1[0]:g}, {p1[1]:g}, {p1[2]:g}) m is not on a cell "
                f"corner; the geometry is placed at the nearest one, ({p2[0]:g}, {p2[1]:g}, "
                f"{p2[2]:g}) m"
            )

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
                file_dl = f.file_handler.attrs["dx_dy_dz"]
                file_dl = (
                    "not recorded"
                    if isinstance(file_dl, h5py.Empty)
                    else " x ".join(f"{value:.9g}" for value in np.ravel(file_dl)) + " m"
                )
                model_dl = " x ".join(f"{value:.9g}" for value in grid.dl)
                raise ValueError(
                    f"{self.__str__()} requires the spatial resolution "
                    f"of the geometry objects file ({file_dl}) to match the "
                    f"model's ({model_dl} m)"
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
        for key, spec in zip(keys, specs):
            model_metadata = spec.metadata.get("surface_impedance_model")
            private_role = spec.metadata.get("impedance_private_role")
            if model_metadata is None and private_role is None:
                continue
            if model_metadata is not None and private_role is not None:
                raise ValueError(
                    f"Geometry material '{database}:{key}' cannot be both a surface "
                    "impedance and a private coefficient row"
                )
            if "directional_materials" in spec.metadata:
                raise ValueError(
                    f"Geometry material '{database}:{key}' cannot combine surface-impedance "
                    "metadata with directional material references"
                )
        specs_by_key = dict(zip(keys, specs))
        invalid_references = "Invalid x/y/z directional material references in geometry database"

        def spec_for(key):
            if key not in specs_by_key:
                try:
                    specs_by_key[key] = load_material_spec(
                        database, key, search_directory=search_directory
                    )
                except (KeyError, ValueError, FileNotFoundError) as exc:
                    raise ValueError(f"{invalid_references}: {exc}") from exc
            return specs_by_key[key]

        def references_for(spec):
            references = spec.metadata.get("directional_materials")
            if references is None:
                return None
            if (
                spec.model != "constant"
                or not isinstance(references, list)
                or len(references) != 3
                or any(not isinstance(item, str) or not item for item in references)
            ):
                raise ValueError(invalid_references)
            return tuple(references)

        # Nested references select the same global axis, so flatten them to scalar leaves.
        tensor_leaves = {}
        resolving = set()

        def leaves_for(key):
            if key in tensor_leaves:
                return tensor_leaves[key]
            if key in resolving:
                raise ValueError(f"{invalid_references}: circular reference at '{key}'")
            resolving.add(key)
            try:
                references = references_for(spec_for(key))
                if references is None:
                    raise ValueError(f"{invalid_references}: '{key}' is not directional")
                leaves = []
                for axis, reference in enumerate(references):
                    if "directional_materials" in spec_for(reference).metadata:
                        leaves.append(leaves_for(reference)[axis])
                    else:
                        leaves.append(reference)
            finally:
                resolving.remove(key)
            tensor_leaves[key] = tuple(leaves)
            return tensor_leaves[key]

        directional_leaves = {}
        for index, spec in enumerate(specs):
            if "directional_materials" not in spec.metadata:
                continue
            references_for(spec)
            directional_leaves[index] = leaves_for(keys[index])
        if directional_leaves:
            with h5py.File(geofile, "r") as geometry:
                if not all(name in geometry for name in ("ID", "rigidE", "rigidH")):
                    raise ValueError(
                        "Directional geometry requires complete ID, rigidE and rigidH arrays; "
                        "voxel-only scalar reconstruction would lose its anisotropy"
                    )

        def restore(key, spec, axes):
            original_id = spec.metadata.get("original_id", key)
            if not isinstance(original_id, str) or not original_id:
                raise ValueError(f"Material '{database}:{key}' metadata original_id is invalid")
            model_metadata = spec.metadata.get("surface_impedance_model")
            private_role = spec.metadata.get("impedance_private_role")
            if model_metadata is not None:
                if not isinstance(model_metadata, dict):
                    raise ValueError(
                        f"Material '{database}:{key}' surface-impedance metadata is invalid"
                    )
                model_id = model_metadata.get("ID")
                if not isinstance(model_id, str) or original_id != impedance_marker_id(model_id):
                    raise ValueError(
                        f"Material '{database}:{key}' records marker ID '{original_id}' "
                        "without a matching surface-impedance model ID"
                    )
                restore_surface_impedance_model(grid, model_id, model_metadata)
                marker = create_impedance_marker_material(grid, model_id)
                existing_by_id[marker.ID] = marker
                return marker
            if private_role is not None:
                expected_id = private_impedance_material_id(private_role)
                if original_id != expected_id:
                    raise ValueError(
                        f"Material '{database}:{key}' records private role {private_role!r} "
                        f"under ID '{original_id}'"
                    )
                return restore_private_impedance_material(grid, private_role)
            if is_reserved_impedance_id(original_id):
                raise ValueError(
                    f"Material '{database}:{key}' uses the reserved surface-impedance "
                    "namespace without round-trip metadata; re-export the geometry"
                )
            namespaced_id = f"{original_id}{{{namespace}}}"
            original = existing_by_id.get(original_id)
            if original is not None and material_matches_spec(
                original, spec, directional_materials=axes
            ):
                return original

            namespaced = existing_by_id.get(namespaced_id)
            if namespaced is not None:
                if not material_matches_spec(namespaced, spec, directional_materials=axes):
                    raise ValueError(
                        f"Geometry material '{namespaced_id}' conflicts with a material already "
                        "defined in the model"
                    )
                return namespaced

            # An unrelated model material may legitimately use the original
            # CAD/material ID with different properties. Keep both by giving
            # the imported definition its deterministic database namespace.
            created = build_material_from_spec(
                grid, spec, namespaced_id, directional_materials=axes
            )
            created.type = f"{created.type},\nimported" if created.type else "imported"
            existing_by_id[namespaced_id] = created
            return created

        # Restore referenced scalar leaves before binding any tensor record.
        leaf_materials = {}
        for index in range(len(keys)):
            if index in directional_leaves:
                continue
            key = keys[index]
            leaf_materials[key] = restore(key, specs[index], None)
            material_id_map[index] = leaf_materials[key].numID
        for leaves in tensor_leaves.values():
            for key in leaves:
                if key not in leaf_materials:
                    leaf_materials[key] = restore(key, spec_for(key), None)

        for key, leaves in tensor_leaves.items():
            axes = tuple(leaf_materials[leaf] for leaf in leaves)
            densities = tuple(material.mass_density for material in axes)
            expected_density = (
                densities[0] if all(value == densities[0] for value in densities) else None
            )
            if spec_for(key).mass_density != expected_density:
                raise ValueError("Directional cell density must agree with its three constituents")

        for index, leaves in directional_leaves.items():
            axes = tuple(leaf_materials[leaf] for leaf in leaves)
            material = restore(keys[index], specs[index], axes)
            material_id_map[index] = material.numID
        return material_id_map, f"material database {database}"
