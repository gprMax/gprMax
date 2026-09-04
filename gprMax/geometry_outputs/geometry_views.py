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

from __future__ import annotations

import logging
import sys
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Generic, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

import gprMax.config as config
from gprMax._version import __version__
from gprMax.geometry_outputs.grid_view import GridType, GridView, MPIGridView
from gprMax.mode2d import mode2d_geometry
from gprMax.receivers import Rx
from gprMax.sources import Source
from gprMax.subgrids.grid import SubGridBaseGrid
from gprMax.utilities.utilities import get_terminal_width
from gprMax.vtkhdf_filehandlers.vtkhdf import VtkHdfFile

if TYPE_CHECKING:
    from gprMax.grid.mpi_grid import MPIGrid

logger = logging.getLogger(__name__)

SOURCE_GEOMETRY_SCHEMA_VERSION = 1
RECEIVER_GEOMETRY_SCHEMA_VERSION = 1


def save_geometry_views(gvs: "List[GeometryView]"):
    """Creates and saves geometryviews.

    Args:
        gvs: list of all GeometryViews.
    """

    logger.info("")
    for i, gv in enumerate(gvs):
        gv.set_filename()
        gv.prep_vtk()
        pbar = tqdm(
            total=gv.nbytes,
            unit="byte",
            unit_scale=True,
            desc=f"Writing geometry view file {i + 1}/{len(gvs)}, {gv.filename.name}",
            ncols=get_terminal_width() - 1,
            file=sys.stdout,
            disable=not config.sim_config.general["progressbars"],
        )
        gv.write_vtk()
        pbar.update(gv.nbytes)
        pbar.close()

    logger.info("")


class GeometryView(Generic[GridType]):
    """Base class for Geometry Views."""

    FILE_EXTENSION = ".vtkhdf"

    @property
    def GRID_VIEW_TYPE(self) -> type[GridView]:
        return GridView

    def __init__(
        self,
        xs: int,
        ys: int,
        zs: int,
        xf: int,
        yf: int,
        zf: int,
        dx: int,
        dy: int,
        dz: int,
        filename: str,
        grid: GridType,
    ):
        """
        Args:
            xs, xf, ys, yf, zs, zf: ints for extent of geometry view in cells.
            dx, dy, dz: ints for spatial discretisation of geometry view in cells.
            filename: string for filename.
            grid: FDTDGrid class describing a grid in a model.
        """
        self.grid_view = self.GRID_VIEW_TYPE(grid, xs, ys, zs, xf, yf, zf, dx, dy, dz)

        self.filenamebase = filename
        self.nbytes = None

        self.material_data = None
        self.materials = None

    @property
    def grid(self) -> GridType:
        return self.grid_view.grid

    def set_filename(self):
        """Construct filename from user-supplied name and model number."""
        parts = config.get_model_config().output_file_path.parts
        self.filename = Path(
            *parts[:-1], self.filenamebase + config.get_model_config().appendmodelnumber
        ).with_suffix(self.FILE_EXTENSION)

    @abstractmethod
    def prep_vtk(self):
        pass

    @abstractmethod
    def write_vtk(self):
        pass


class Metadata(Generic[GridType]):
    """Comments can be strings included in the header of XML VTK file, and are
    used to hold extra (gprMax) information about the VTK data.
    """

    def __init__(
        self,
        grid_view: GridView[GridType],
        averaged_materials: bool = False,
        materials_only: bool = False,
    ):
        self.grid_view = grid_view
        self.averaged_materials = averaged_materials
        self.materials_only = materials_only

        self.gprmax_version = __version__
        self.dx_dy_dz = self.dx_dy_dz_comment()
        self.nx_ny_nz = self.nx_ny_nz_comment()

        self.materials = self.materials_comment()

        # Write information on PMLs, sources, and receivers
        if not self.materials_only:
            # Information on PML thickness
            self.pml_thickness = self.pml_gv_comment()

            point_sources = (
                self.grid.hertziandipoles
                + self.grid.magneticdipoles
                + self.grid.voltagesources
                + self.grid.transmissionlines
                + self.grid.magneticfrillsources
                + [
                    terminal
                    for terminal in getattr(self.grid, "networkterminals", ())
                    if terminal.excited
                ]
            )
            sources_comment = self.source_points_comment(point_sources)
            if sources_comment is None:
                self.source_ids = self.source_types = self.source_positions = None
            else:
                self.source_ids, self.source_types, self.source_positions = sources_comment

            source_points, receiver_ports = self.point_geometry_entries()
            source_geometry = self.source_geometry_comment(source_points)
            if source_geometry is None:
                (
                    self.source_geometry_ids,
                    self.source_geometry_types,
                    self.source_geometry_kinds,
                    self.source_geometry_bounds,
                ) = (None, None, None, None)
            else:
                (
                    self.source_geometry_ids,
                    self.source_geometry_types,
                    self.source_geometry_kinds,
                    self.source_geometry_bounds,
                ) = source_geometry

            public_receivers = [
                receiver
                for receiver in self.grid.rxs
                if not getattr(receiver, "internal", False)
            ]
            receivers_comment = self.srcs_rx_gv_comment(public_receivers)
            if receivers_comment is None:
                self.receiver_ids = self.receiver_positions = None
            else:
                self.receiver_ids, self.receiver_positions = receivers_comment

            receiver_geometry = self.receiver_geometry_comment(
                public_receivers, receiver_ports
            )
            if receiver_geometry is None:
                (
                    self.receiver_geometry_ids,
                    self.receiver_geometry_types,
                    self.receiver_geometry_kinds,
                    self.receiver_geometry_bounds,
                ) = (None, None, None, None)
            else:
                (
                    self.receiver_geometry_ids,
                    self.receiver_geometry_types,
                    self.receiver_geometry_kinds,
                    self.receiver_geometry_bounds,
                ) = receiver_geometry

    @property
    def grid(self) -> GridType:
        return self.grid_view.grid

    def write_to_vtkhdf(self, file_handler: VtkHdfFile):
        file_handler.add_field_data("gprMax_version", self.gprmax_version)
        file_handler.add_field_data("dx_dy_dz", self.dx_dy_dz)
        file_handler.add_field_data("nx_ny_nz", self.nx_ny_nz)

        file_handler.add_field_data("material_ids", self.materials)

        if not self.materials_only:
            if self.pml_thickness is not None:
                file_handler.add_field_data("pml_thickness", self.pml_thickness)

            if self.source_ids is not None and self.source_positions is not None:
                file_handler.add_field_data("source_ids", self.source_ids)
                file_handler.add_field_data("source_types", self.source_types)
                file_handler.add_field_data("sources", self.source_positions)

            file_handler.add_field_data(
                "source_geometry_schema_version", SOURCE_GEOMETRY_SCHEMA_VERSION
            )
            if self.source_geometry_ids is not None:
                file_handler.add_field_data("source_geometry_ids", self.source_geometry_ids)
                file_handler.add_field_data("source_geometry_types", self.source_geometry_types)
                file_handler.add_field_data("source_geometry_kinds", self.source_geometry_kinds)
                file_handler.add_field_data("source_geometry_bounds", self.source_geometry_bounds)

            if self.receiver_ids is not None and self.receiver_positions is not None:
                file_handler.add_field_data("receiver_ids", self.receiver_ids)
                file_handler.add_field_data("receivers", self.receiver_positions)

            file_handler.add_field_data(
                "receiver_geometry_schema_version", RECEIVER_GEOMETRY_SCHEMA_VERSION
            )
            if self.receiver_geometry_ids is not None:
                file_handler.add_field_data("receiver_geometry_ids", self.receiver_geometry_ids)
                file_handler.add_field_data(
                    "receiver_geometry_types", self.receiver_geometry_types
                )
                file_handler.add_field_data("receiver_geometry_kinds", self.receiver_geometry_kinds)
                file_handler.add_field_data(
                    "receiver_geometry_bounds", self.receiver_geometry_bounds
                )

    def pml_gv_comment(self) -> Optional[npt.NDArray[np.int64]]:
        grid = self.grid

        if not grid.pmls["slabs"]:
            return None

        # Only render PMLs if they are in the geometry view
        thickness: Dict[str, int] = grid.pmls["thickness"]
        gv_pml_depth = dict.fromkeys(thickness, 0)

        if self.grid_view.xs < thickness["x0"]:
            gv_pml_depth["x0"] = thickness["x0"] - self.grid_view.xs
        if self.grid_view.ys < thickness["y0"]:
            gv_pml_depth["y0"] = thickness["y0"] - self.grid_view.ys
        if thickness["z0"] - self.grid_view.zs > 0:
            gv_pml_depth["z0"] = thickness["z0"] - self.grid_view.zs
        if self.grid_view.xf > grid.nx - thickness["xmax"]:
            gv_pml_depth["xmax"] = self.grid_view.xf - (grid.nx - thickness["xmax"])
        if self.grid_view.yf > grid.ny - thickness["ymax"]:
            gv_pml_depth["ymax"] = self.grid_view.yf - (grid.ny - thickness["ymax"])
        if self.grid_view.zf > grid.nz - thickness["zmax"]:
            gv_pml_depth["zmax"] = self.grid_view.zf - (grid.nz - thickness["zmax"])

        return np.array(list(gv_pml_depth.values()), dtype=np.int64)

    def srcs_rx_gv_comment(
        self, srcs: Union[Sequence[Source], List[Rx]]
    ) -> Optional[Tuple[List[str], npt.NDArray[np.float64]]]:
        """Used to name sources and/or receivers."""
        if not srcs:
            return None

        names: List[str] = []
        positions = np.empty((len(srcs), 3))
        for index, src in enumerate(srcs):
            if isinstance(self.grid, SubGridBaseGrid):
                position = self.grid.local_to_global(src.coord)
            else:
                position = src.coord * self.grid.dl
            names.append(src.ID)
            positions[index] = position

        return names, positions

    def source_points_comment(
        self, sources: Sequence[Source]
    ) -> Optional[Tuple[List[str], List[str], npt.NDArray[np.float64]]]:
        """Return IDs, runtime types, and global positions for local sources."""

        result = self.srcs_rx_gv_comment(sources)
        if result is None:
            return None
        names, positions = result
        return names, [type(source).__name__ for source in sources], positions

    @staticmethod
    def _bounds(lower, upper) -> npt.NDArray[np.float64]:
        """Return VTK bounds ordering: xmin, xmax, ymin, ymax, zmin, zmax."""

        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        return np.asarray(
            (lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]),
            dtype=np.float64,
        )

    def _physical_index_position(self, position) -> npt.NDArray[np.float64]:
        """Convert grid indices to global physical coordinates."""

        position = np.asarray(position, dtype=np.int32)
        if isinstance(self.grid, SubGridBaseGrid):
            return np.asarray(self.grid.local_to_global(position), dtype=np.float64)
        return position * self.grid.dl

    @staticmethod
    def source_has_nonzero_excitation(source) -> bool:
        """Return whether the resolved source drive deposits a non-zero field."""

        resolved_arrays = []
        for name in (
            "waveformvalues_wholedt",
            "waveformvalues_halfdt",
            "waveform_whole",
            "waveform_half",
        ):
            values = getattr(source, name, None)
            if values is not None:
                resolved_arrays.append(values)
        if resolved_arrays:
            return any(np.any(values) for values in resolved_arrays)

        if getattr(source, "study_scale", 1.0) == 0:
            return False
        waveform = getattr(source, "waveform", None)
        if waveform is not None and getattr(waveform, "amp", 1.0) == 0:
            return False
        if hasattr(source, "excited"):
            return bool(source.excited)
        return True

    @staticmethod
    def _port_identity(source, fallback_id: str, fallback_type: str, monitor=None):
        """Return a stable public port ID and type for geometry metadata."""

        output_id = getattr(monitor, "output_id", None)
        if output_id is None:
            output_id = getattr(getattr(source, "port_output", None), "output_id", None)
        if output_id is None:
            output_id = getattr(source, "port_id", None)
        return str(output_id or fallback_id), fallback_type

    def point_geometry_entries(self):
        """Split local point objects into active sources and receiving ports."""

        source_entries = []
        receiver_entries = []
        monitor_by_source = {
            id(monitor.source): monitor
            for monitor in getattr(self.grid, "port_monitors", ())
            if getattr(monitor, "source", None) is not None
        }

        for source in self.grid.hertziandipoles + self.grid.magneticdipoles:
            if self.source_has_nonzero_excitation(source):
                source_entries.append((str(source.ID), type(source).__name__, source))

        for source in self.grid.voltagesources:
            monitor = monitor_by_source.get(id(source))
            is_port = monitor is not None or getattr(source, "port_id", None) is not None
            source_id, source_type = self._port_identity(
                source,
                str(source.ID),
                "VoltageSourcePort" if is_port else type(source).__name__,
                monitor,
            )
            entry = (source_id, source_type, source)
            if self.source_has_nonzero_excitation(source):
                source_entries.append(entry)
            elif is_port:
                receiver_entries.append(entry)

        for index, source in enumerate(self.grid.transmissionlines, start=1):
            fallback_id = (
                str(source.ID)
                if getattr(self.grid, "is_distributed", False)
                else f"tl{index}"
            )
            source_id, source_type = self._port_identity(
                source, fallback_id, "TransmissionLinePort"
            )
            entry = (source_id, source_type, source)
            target = (
                source_entries
                if self.source_has_nonzero_excitation(source)
                else receiver_entries
            )
            target.append(entry)

        for index, source in enumerate(self.grid.magneticfrillsources, start=1):
            source_id, source_type = self._port_identity(
                source, f"frill{index}", "MagneticFrillPort"
            )
            entry = (source_id, source_type, source)
            target = (
                source_entries
                if self.source_has_nonzero_excitation(source)
                else receiver_entries
            )
            target.append(entry)

        for terminal in getattr(self.grid, "networkterminals", ()):
            monitor = monitor_by_source.get(id(terminal))
            is_port = monitor is not None or getattr(terminal, "output", None) is not None
            terminal_id, terminal_type = self._port_identity(
                terminal,
                str(terminal.ID),
                "RationalNetworkPort" if is_port else type(terminal).__name__,
                monitor,
            )
            entry = (terminal_id, terminal_type, terminal)
            if self.source_has_nonzero_excitation(terminal):
                source_entries.append(entry)
            elif is_port:
                receiver_entries.append(entry)

        return source_entries, receiver_entries

    def positioned_geometry_comment(self, entries):
        """Return point-geometry metadata for ID, type, coordinate entries."""

        if not entries:
            return None
        ids = []
        types = []
        bounds = []
        for object_id, object_type, obj in entries:
            if isinstance(self.grid, SubGridBaseGrid):
                position = np.asarray(self.grid.local_to_global(obj.coord), dtype=np.float64)
            else:
                position = np.asarray(obj.coord * self.grid.dl, dtype=np.float64)
            ids.append(object_id)
            types.append(object_type)
            bounds.append(self._bounds(position, position + self.grid.dl))
        return ids, types, ["point"] * len(ids), bounds

    def _append_eigenmode_geometry(self, ids, types, kinds, bounds, *, is_source):
        """Append active or passive physical modal-port apertures."""

        virtual_ports = {
            int(port_number)
            for port_number in getattr(self.grid, "virtual_waveguide_specs", {})
        }
        virtual_ports.update(
            int(guide.spec.port) for guide in getattr(self.grid, "virtual_waveguides", ())
        )
        for monitor in getattr(self.grid, "eigenmodeports", ()):
            if bool(monitor.is_source) != is_source:
                continue
            owner = monitor.owner
            lower = np.zeros(3, dtype=np.int32)
            upper = np.zeros(3, dtype=np.int32)
            lower[owner.normal_axis] = owner.global_plane_index
            upper[owner.normal_axis] = owner.global_plane_index
            lower[list(owner.transverse_axes)] = owner.global_transverse_start
            upper[list(owner.transverse_axes)] = owner.global_transverse_stop
            ids.append(str(getattr(monitor, "port_id", f"port{monitor.port_index}")))
            types.append(
                "VirtualWaveguideInterface"
                if int(monitor.port_index) in virtual_ports
                else "EigenmodePort"
            )
            kinds.append("plane")
            bounds.append(
                self._bounds(
                    self._physical_index_position(lower),
                    self._physical_index_position(upper),
                )
            )

    def source_geometry_comment(self, point_entries):
        """Build the unified point, TFSF-box, and eigenmode-plane catalogue."""

        ids: List[str] = []
        types: List[str] = []
        kinds: List[str] = []
        bounds: List[npt.NDArray[np.float64]] = []

        point_geometry = self.positioned_geometry_comment(point_entries)
        if point_geometry is not None:
            point_ids, point_types, point_kinds, point_bounds = point_geometry
            ids.extend(point_ids)
            types.extend(point_types)
            kinds.extend(point_kinds)
            bounds.extend(point_bounds)

        geometry_2d = mode2d_geometry(config.get_model_config().mode)
        for index, source in enumerate(getattr(self.grid, "discreteplanewaves", ()), start=1):
            if not self.source_has_nonzero_excitation(source):
                continue
            corners = np.asarray(source.corners, dtype=np.int32).reshape(2, 3).copy()
            if geometry_2d is not None:
                corners[:, geometry_2d.invariant_axis] = geometry_2d.live_index
            lower = self._physical_index_position(corners[0])
            upper = self._physical_index_position(corners[1])
            ids.append(f"plane_wave_{index}")
            types.append(type(source).__name__)
            kinds.append("rectangle" if geometry_2d is not None else "box")
            bounds.append(self._bounds(lower, upper))

        self._append_eigenmode_geometry(ids, types, kinds, bounds, is_source=True)

        if not ids:
            return None
        return ids, types, kinds, np.asarray(bounds, dtype=np.float64)

    def receiver_geometry_comment(self, public_receivers, receiver_ports):
        """Build typed point receivers and passive port apertures."""

        entries = [
            (str(receiver.ID), type(receiver).__name__, receiver)
            for receiver in public_receivers
        ]
        entries.extend(receiver_ports)
        point_geometry = self.positioned_geometry_comment(entries)
        if point_geometry is None:
            ids: List[str] = []
            types: List[str] = []
            kinds: List[str] = []
            bounds: List[npt.NDArray[np.float64]] = []
        else:
            point_ids, point_types, point_kinds, point_bounds = point_geometry
            ids = list(point_ids)
            types = list(point_types)
            kinds = list(point_kinds)
            bounds = list(point_bounds)

        self._append_eigenmode_geometry(ids, types, kinds, bounds, is_source=False)
        if not ids:
            return None
        return ids, types, kinds, np.asarray(bounds, dtype=np.float64)

    def dx_dy_dz_comment(self) -> npt.NDArray[np.float64]:
        return self.grid.dl

    def nx_ny_nz_comment(self) -> npt.NDArray[np.int32]:
        return self.grid.size

    def materials_comment(self) -> Optional[List[str]]:
        if hasattr(self.grid_view, "materials"):
            materials = self.grid_view.materials
        else:
            materials = self.grid.materials

        if materials is None:
            return None

        if not self.averaged_materials:
            return [m.ID for m in materials if m.type != "dielectric-smoothed"]
        else:
            return [m.ID for m in materials]


class MPIMetadata(Metadata["MPIGrid"]):
    def nx_ny_nz_comment(self) -> npt.NDArray[np.int32]:
        return self.grid.global_size

    def pml_gv_comment(self) -> Optional[npt.NDArray[np.int64]]:
        gv_pml_depth = super().pml_gv_comment()

        if gv_pml_depth is None:
            gv_pml_depth = np.zeros(6, dtype=np.int64)

        assert isinstance(self.grid_view, MPIGridView)
        recv_buffer = np.empty((self.grid_view.comm.size, 6), dtype=np.int64)
        self.grid_view.comm.Allgather(gv_pml_depth, recv_buffer)

        gv_pml_depth = np.max(recv_buffer, axis=0)

        return None if all(gv_pml_depth == 0) else gv_pml_depth

    def srcs_rx_gv_comment(
        self, srcs: Union[Sequence[Source], List[Rx]]
    ) -> Optional[Tuple[List[str], npt.NDArray[np.float64]]]:
        objects: Dict[str, npt.NDArray[np.float64]] = {}
        for src in srcs:
            position = self.grid.local_to_global_coordinate(src.coord) * self.grid.dl
            objects[src.ID] = position

        assert isinstance(self.grid_view, MPIGridView)
        global_objects: List[Dict[str, npt.NDArray[np.float64]]] = self.grid_view.comm.allgather(
            objects
        )
        objects = {k: v for d in global_objects for k, v in d.items()}
        objects = dict(sorted(objects.items()))

        return (list(objects.keys()), np.array(list(objects.values()))) if objects else None

    def source_points_comment(
        self, sources: Sequence[Source]
    ) -> Optional[Tuple[List[str], List[str], npt.NDArray[np.float64]]]:
        """Gather positioned sources once and retain their runtime types."""

        objects = {}
        for source in sources:
            position = self.grid.local_to_global_coordinate(source.coord) * self.grid.dl
            key = (str(source.ID), type(source).__name__)
            objects[key] = position

        global_objects = self.grid_view.comm.allgather(objects)
        objects = {key: value for rank_objects in global_objects for key, value in rank_objects.items()}
        if not objects:
            return None
        ordered = sorted(objects)
        return (
            [key[0] for key in ordered],
            [key[1] for key in ordered],
            np.asarray([objects[key] for key in ordered]),
        )

    def positioned_geometry_comment(self, entries):
        """Gather typed point geometry from its owning MPI rank."""

        objects = {}
        for object_id, object_type, obj in entries:
            position = self.grid.local_to_global_coordinate(obj.coord) * self.grid.dl
            objects[(str(object_id), str(object_type))] = position

        global_objects = self.grid_view.comm.allgather(objects)
        objects = {key: value for rank_objects in global_objects for key, value in rank_objects.items()}
        if not objects:
            return None
        ordered = sorted(objects)
        bounds = [
            self._bounds(objects[key], np.asarray(objects[key]) + self.grid.dl)
            for key in ordered
        ]
        return (
            [key[0] for key in ordered],
            [key[1] for key in ordered],
            ["point"] * len(ordered),
            bounds,
        )
