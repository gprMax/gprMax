# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax.

"""Sparse surface-impedance models and voxel-boundary compilation.

The first implementation deliberately supports only closed, grid-aligned
impedance volumes on the main 3-D CPU grid.  Geometry remains represented by
the normal dense Yee arrays, but conductor-interior components are assigned a
private void coefficient row and boundary tangential electric edges are owned
by this sparse subsystem.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import numpy.typing as npt

from gprMax import config
from gprMax.materials import Material

logger = logging.getLogger(__name__)

PRIVATE_IMPEDANCE_ID_PREFIX = "__impedance_"

try:
    from gprMax.cython.impedance_surface import update_impedance_surfaces as _cython_update
except ImportError:  # pragma: no cover - exercised only before an extension rebuild
    _cython_update = None


@dataclass(frozen=True)
class DiscreteSurfaceImpedance:
    """Trapezoidal runtime realization of one continuous impedance model."""

    F: npt.NDArray[np.float64]
    G: npt.NDArray[np.float64]
    L: npt.NDArray[np.float64]
    Z0: float


@dataclass(frozen=True)
class SurfaceImpedanceModel:
    """Real continuous-time realization ``Z(s) = D + C(sI-A)^-1 B``."""

    ID: str
    A: npt.ArrayLike = ()
    B: npt.ArrayLike = ()
    C: npt.ArrayLike = ()
    D: float = 0.0
    fit_fmin_hz: float = 0.0
    fit_fmax_hz: float = np.inf
    allow_active: bool = False
    preset: str | None = None
    provenance: str | None = None
    conductivity_s_per_m: float | None = None
    fit_requested_order: str | int | None = None
    fit_pole_count: int | None = None
    fit_tolerance: float | None = None
    fit_max_relative_error: float | None = None
    fit_rms_relative_error: float | None = None
    fit_method: str | None = None
    plot_fit_in_full_run: bool = False

    def __post_init__(self) -> None:
        if not self.ID or "/" in self.ID or "\x00" in self.ID:
            raise ValueError("surface-impedance ID must be a non-empty HDF5 path component")

        A = np.asarray(self.A, dtype=np.float64)
        if A.size == 0:
            A = np.empty((0, 0), dtype=np.float64)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("surface-impedance A must be a square real matrix")
        order = A.shape[0]
        B = np.asarray(self.B, dtype=np.float64).reshape(-1)
        C = np.asarray(self.C, dtype=np.float64).reshape(-1)
        if B.size != order or C.size != order:
            raise ValueError("surface-impedance B and C lengths must match A")
        D = float(self.D)
        fmin = float(self.fit_fmin_hz)
        fmax = float(self.fit_fmax_hz)
        fit_error = self.fit_max_relative_error
        fit_rms_error = self.fit_rms_relative_error
        conductivity = self.conductivity_s_per_m
        fit_tolerance = self.fit_tolerance
        fit_pole_count = self.fit_pole_count
        if not all(np.all(np.isfinite(value)) for value in (A, B, C)) or not np.isfinite(D):
            raise ValueError("surface-impedance realization coefficients must be finite")
        if fmin < 0 or not np.isfinite(fmin) or np.isnan(fmax) or fmax <= fmin:
            raise ValueError("surface-impedance fit band must satisfy 0 <= fmin < fmax")
        if order and np.any(np.real(np.linalg.eigvals(A)) >= 0):
            raise ValueError("surface-impedance A must be strictly Hurwitz")
        if not self.allow_active and D < 0:
            raise ValueError("negative surface-impedance feedthrough requires allow_active=True")
        if fit_error is not None:
            fit_error = float(fit_error)
            if not np.isfinite(fit_error) or fit_error < 0:
                raise ValueError("surface-impedance fit error must be finite and non-negative")
        if fit_rms_error is not None:
            fit_rms_error = float(fit_rms_error)
            if not np.isfinite(fit_rms_error) or fit_rms_error < 0:
                raise ValueError("surface-impedance RMS fit error must be finite and non-negative")
        if conductivity is not None:
            conductivity = float(conductivity)
            if not np.isfinite(conductivity) or conductivity <= 0:
                raise ValueError("surface-impedance conductivity must be finite and positive")
        if fit_tolerance is not None:
            fit_tolerance = float(fit_tolerance)
            if not np.isfinite(fit_tolerance) or fit_tolerance <= 0:
                raise ValueError("surface-impedance fit tolerance must be finite and positive")
        if fit_pole_count is not None:
            fit_pole_count = int(fit_pole_count)
            if fit_pole_count < 1:
                raise ValueError("surface-impedance pole count must be positive")
            if fit_pole_count != order:
                raise ValueError(
                    "surface-impedance fit pole count must match the realization order"
                )

        # The dataclass is frozen, so its array members must be frozen too.
        # Immutable byte backing detaches caller-owned buffers and also blocks
        # writes attempted through NumPy's public ``.base`` chain.
        immutable_arrays = []
        for value in (A, B, C):
            contiguous = np.array(value, dtype=np.float64, order="C", copy=True)
            immutable_arrays.append(
                np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(contiguous.shape)
            )
        object.__setattr__(self, "A", immutable_arrays[0])
        object.__setattr__(self, "B", immutable_arrays[1])
        object.__setattr__(self, "C", immutable_arrays[2])
        object.__setattr__(self, "D", D)
        object.__setattr__(self, "fit_fmin_hz", fmin)
        object.__setattr__(self, "fit_fmax_hz", fmax)
        object.__setattr__(self, "conductivity_s_per_m", conductivity)
        object.__setattr__(self, "fit_pole_count", fit_pole_count)
        object.__setattr__(self, "fit_tolerance", fit_tolerance)
        object.__setattr__(self, "fit_max_relative_error", fit_error)
        object.__setattr__(self, "fit_rms_relative_error", fit_rms_error)
        object.__setattr__(self, "plot_fit_in_full_run", bool(self.plot_fit_in_full_run))

    @property
    def order(self) -> int:
        return int(self.A.shape[0])

    @property
    def model_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.ID.encode("utf-8"))
        for value in (self.A, self.B, self.C, np.asarray((self.D,), dtype=np.float64)):
            digest.update(np.asarray(value, dtype="<f8").tobytes())
        digest.update(np.asarray((self.fit_fmin_hz, self.fit_fmax_hz), dtype="<f8").tobytes())
        digest.update((self.preset or "").encode("utf-8"))
        digest.update((self.provenance or "").encode("utf-8"))
        digest.update((self.fit_method or "").encode("utf-8"))
        digest.update(str(self.fit_requested_order or "").encode("utf-8"))
        optional_numbers = (
            np.nan if self.conductivity_s_per_m is None else self.conductivity_s_per_m,
            np.nan if self.fit_pole_count is None else self.fit_pole_count,
            np.nan if self.fit_tolerance is None else self.fit_tolerance,
            np.nan if self.fit_max_relative_error is None else self.fit_max_relative_error,
            np.nan if self.fit_rms_relative_error is None else self.fit_rms_relative_error,
            float(self.plot_fit_in_full_run),
        )
        digest.update(np.asarray(optional_numbers, dtype="<f8").tobytes())
        return digest.hexdigest()

    def impedance(self, frequencies_hz: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        frequencies = np.asarray(frequencies_hz, dtype=np.float64)
        if np.any(frequencies < 0) or not np.all(np.isfinite(frequencies)):
            raise ValueError("surface-impedance frequencies must be finite and non-negative")
        result = np.full(frequencies.shape, self.D, dtype=np.complex128)
        if self.order:
            identity = np.eye(self.order)
            for index, frequency in np.ndenumerate(frequencies):
                state = np.linalg.solve(2j * np.pi * frequency * identity - self.A, self.B)
                result[index] += self.C @ state
        return result

    def require_fit_frequency(
        self,
        frequency_hz: float,
        *,
        purpose: str,
        frequency_kind: str = "physical",
    ) -> None:
        """Reject extrapolation of a dispersive realization in a solver path."""

        frequency = float(frequency_hz)
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError(f"{purpose} {frequency_kind} frequency must be finite and positive")
        # A zero-order resistance is frequency independent even when the user
        # supplied a descriptive output band. Dynamic realizations, including
        # common-metal fits, must stay inside their declared validity band.
        if self.order and not self.fit_fmin_hz <= frequency <= self.fit_fmax_hz:
            raise ValueError(
                f"{purpose} {frequency_kind} frequency {frequency:g} Hz is outside "
                "surface-impedance "
                f"model {self.ID!r} fit band {self.fit_fmin_hz:g}--"
                f"{self.fit_fmax_hz:g} Hz; expand the fit band to include every "
                "eigenmode anchor and its bilinear-warped evaluation frequency"
            )

    def discretise(self, dt: float) -> DiscreteSurfaceImpedance:
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("surface-impedance time step must be finite and positive")
        identity = np.eye(self.order)
        left = identity - 0.5 * dt * self.A
        right = identity + 0.5 * dt * self.A
        if self.order:
            F = np.linalg.solve(left, right)
            G = np.linalg.solve(left, dt * self.B)
        else:
            F = np.empty((0, 0), dtype=np.float64)
            G = np.empty(0, dtype=np.float64)
        L = 0.5 * self.C @ (identity + F)
        Z0 = float(self.D + 0.5 * self.C @ G)
        tolerance = 256 * np.finfo(np.float64).eps * max(1.0, abs(self.D))
        if not np.isfinite(Z0) or Z0 <= tolerance:
            raise ValueError(
                f"surface-impedance model {self.ID!r} has non-positive discrete feedthrough Z0"
            )

        if not self.allow_active:
            # Trapezoidal time discretisation maps a discrete phase theta to
            # the continuous realization at
            #   f_b = tan(theta / 2) / (pi * dt).
            # Sampling only 0..physical Nyquist would therefore miss active
            # behaviour that folds into the upper part of the unit circle.
            phases = np.linspace(0, np.pi, 2049, endpoint=False)
            warped_frequencies = np.tan(0.5 * phases) / (np.pi * dt)
            values = self.impedance(warped_frequencies)
            passive_tolerance = (
                2048
                * np.finfo(np.float64).eps
                * max(
                    1.0,
                    abs(self.D),
                    float(np.max(np.abs(values), initial=0.0)),
                )
            )
            minimum = min(float(np.min(np.real(values))), self.D)
            if minimum < -passive_tolerance:
                raise ValueError(
                    f"surface-impedance model {self.ID!r} is non-passive on the discrete band "
                    f"(minimum real impedance {minimum:g} Ohm)"
                )

        return DiscreteSurfaceImpedance(
            np.ascontiguousarray(F),
            np.ascontiguousarray(G),
            np.ascontiguousarray(L),
            Z0,
        )


class _PrivateImpedanceMaterial(Material):
    """Private coefficient-table row used for held or void Yee components."""

    def __init__(self, numID: int, ID: str, role: str):
        super().__init__(numID, ID)
        self.type = f"impedance-{role}"
        self.averagable = False
        self.impedance_role = role

    def calculate_update_coeffsE(self, grid) -> None:
        hold = self.impedance_role == "surface-hold"
        self.CA = 1.0 if hold else 0.0
        self.CBx = self.CBy = self.CBz = self.srce = 0.0

    def calculate_update_coeffsH(self, grid) -> None:
        self.DA = self.DBx = self.DBy = self.DBz = self.srcm = 0.0


def is_reserved_impedance_id(ID: str) -> bool:
    """Return whether an ID belongs to the private impedance material namespace."""

    return isinstance(ID, str) and ID.startswith(PRIVATE_IMPEDANCE_ID_PREFIX)


def create_impedance_marker_material(grid, model_id: str) -> Material:
    """Return the private voxel marker associated with a surface model."""

    existing_numid = next(
        (numid for numid, value in grid.impedance_marker_models.items() if value == model_id),
        None,
    )
    if existing_numid is not None:
        return grid.materials[existing_numid]

    marker_id = f"__impedance_volume__{model_id}"
    if any(material.ID == marker_id for material in grid.materials):
        raise ValueError(
            f"material ID {marker_id!r} conflicts with a reserved internal "
            "surface-impedance material ID"
        )
    marker = Material(len(grid.materials), marker_id)
    marker.type = "impedance-volume-marker"
    marker.averagable = False
    marker.surface_impedance_id = model_id
    grid.materials.append(marker)
    grid.impedance_marker_models[marker.numID] = model_id
    return marker


def _sentinel_material(grid, role: str) -> _PrivateImpedanceMaterial:
    ID = f"__impedance_{role.replace('-', '_')}__"
    existing = next((material for material in grid.materials if material.ID == ID), None)
    if existing is not None:
        if isinstance(existing, _PrivateImpedanceMaterial) and existing.impedance_role == role:
            return existing
        raise ValueError(
            f"material ID {ID!r} conflicts with a reserved internal "
            "surface-impedance material ID"
        )
    material = _PrivateImpedanceMaterial(len(grid.materials), ID, role)
    grid.materials.append(material)
    return material


def _electric_quadrants(owner: npt.NDArray[np.int32], axis: int):
    """Return cyclic surrounding-cell owner arrays and cell offsets for one E axis."""

    nx, ny, nz = owner.shape
    if axis == 0:
        padded = np.pad(owner, ((0, 0), (1, 1), (1, 1)), constant_values=-1)
        arrays = (
            padded[:, 0 : ny + 1, 0 : nz + 1],
            padded[:, 1 : ny + 2, 0 : nz + 1],
            padded[:, 1 : ny + 2, 1 : nz + 2],
            padded[:, 0 : ny + 1, 1 : nz + 2],
        )
        offsets = ((0, -1, -1), (0, 0, -1), (0, 0, 0), (0, -1, 0))
    elif axis == 1:
        padded = np.pad(owner, ((1, 1), (0, 0), (1, 1)), constant_values=-1)
        arrays = (
            padded[0 : nx + 1, :, 0 : nz + 1],
            padded[0 : nx + 1, :, 1 : nz + 2],
            padded[1 : nx + 2, :, 1 : nz + 2],
            padded[1 : nx + 2, :, 0 : nz + 1],
        )
        offsets = ((-1, 0, -1), (-1, 0, 0), (0, 0, 0), (0, 0, -1))
    else:
        padded = np.pad(owner, ((1, 1), (1, 1), (0, 0)), constant_values=-1)
        arrays = (
            padded[0 : nx + 1, 0 : ny + 1, :],
            padded[1 : nx + 2, 0 : ny + 1, :],
            padded[1 : nx + 2, 1 : ny + 2, :],
            padded[0 : nx + 1, 1 : ny + 2, :],
        )
        offsets = ((-1, -1, 0), (0, -1, 0), (0, 0, 0), (-1, 0, 0))
    return arrays, offsets


def _face_component_count(mask: npt.NDArray[np.bool_]) -> int:
    """Count 6-connected components in a small three-dimensional Boolean mask."""

    remaining = {tuple(int(value) for value in coord) for coord in np.argwhere(mask)}
    components = 0
    while remaining:
        components += 1
        pending = [remaining.pop()]
        while pending:
            coord = pending.pop()
            for axis in range(3):
                for step in (-1, 1):
                    neighbour = list(coord)
                    neighbour[axis] += step
                    neighbour_tuple = tuple(neighbour)
                    if neighbour_tuple in remaining:
                        remaining.remove(neighbour_tuple)
                        pending.append(neighbour_tuple)
    return components


def _vertex_topology_status_table() -> npt.NDArray[np.uint8]:
    """Return a lookup table for local occupied/retained face connectivity.

    Bit zero marks a disconnected occupied set and bit one marks a
    disconnected retained set. Empty sets are valid: all-retained and
    all-occupied neighbourhoods contain no local interface.
    """

    status = np.zeros(256, dtype=np.uint8)
    coordinates = tuple(np.ndindex((2, 2, 2)))
    for pattern in range(256):
        occupied = np.zeros((2, 2, 2), dtype=np.bool_)
        for bit, coord in enumerate(coordinates):
            occupied[coord] = bool(pattern & (1 << bit))
        if _face_component_count(occupied) > 1:
            status[pattern] |= 1
        if _face_component_count(~occupied) > 1:
            status[pattern] |= 2
    return status


_VERTEX_TOPOLOGY_STATUS = _vertex_topology_status_table()


def _validate_impedance_voxel_topology(owner: npt.NDArray[np.int32]) -> None:
    """Reject locally non-manifold binary impedance-voxel configurations.

    Every non-negative owner is treated as occupied, irrespective of which
    surface-impedance model owns it. The check is deliberately local: it does
    not require separate impedance bodies to form one globally connected set.
    """

    if owner.ndim != 3:
        raise ValueError("surface-impedance voxel ownership must be three-dimensional")

    axis_names = "xyz"
    for axis in range(3):
        quadrants, _ = _electric_quadrants(owner, axis)
        occupied = tuple(value >= 0 for value in quadrants)
        diagonal = (occupied[0] & occupied[2] & ~occupied[1] & ~occupied[3]) | (
            occupied[1] & occupied[3] & ~occupied[0] & ~occupied[2]
        )
        if np.any(diagonal):
            flat_index = int(np.argmax(diagonal))
            coord = tuple(int(value) for value in np.unravel_index(flat_index, diagonal.shape))
            raise ValueError(
                "impedance-volume voxel topology is non-manifold at a Yee edge: "
                f"{axis_names[axis]}-directed edge {coord}; connect the "
                "impedance cells through a cell face, or separate them with "
                "retained cells so they do not share the edge"
            )

    if any(size < 2 for size in owner.shape):
        return

    occupied = owner >= 0
    block_shape = tuple(size - 1 for size in owner.shape)
    patterns = np.zeros(block_shape, dtype=np.uint8)
    for bit, (di, dj, dk) in enumerate(np.ndindex((2, 2, 2))):
        local = occupied[
            di : di + block_shape[0],
            dj : dj + block_shape[1],
            dk : dk + block_shape[2],
        ]
        patterns |= local.astype(np.uint8) << bit

    status = _VERTEX_TOPOLOGY_STATUS[patterns]
    if np.any(status):
        flat_index = int(np.argmax(status))
        lower = tuple(int(value) for value in np.unravel_index(flat_index, status.shape))
        vertex = tuple(value + 1 for value in lower)
        local_status = int(status[lower])
        disconnected = []
        if local_status & 1:
            disconnected.append("impedance")
        if local_status & 2:
            disconnected.append("retained")
        raise ValueError(
            "impedance-volume voxel topology is non-manifold at grid vertex "
            f"{vertex}: the {' and '.join(disconnected)} incident cells are "
            "not face-connected; reshape the geometry so both impedance and "
            "retained incident cells connect through cell faces, or separate "
            "the impedance cells with retained cells so they do not touch at "
            "an edge or vertex"
        )


def _component_valid_view(array: np.ndarray, component: int, grid):
    if component == 0:
        return array[0 : grid.nx, 0 : grid.ny + 1, 0 : grid.nz + 1]
    if component == 1:
        return array[0 : grid.nx + 1, 0 : grid.ny, 0 : grid.nz + 1]
    if component == 2:
        return array[0 : grid.nx + 1, 0 : grid.ny + 1, 0 : grid.nz]
    if component == 3:
        return array[0 : grid.nx + 1, 0 : grid.ny, 0 : grid.nz]
    if component == 4:
        return array[0 : grid.nx, 0 : grid.ny + 1, 0 : grid.nz]
    return array[0 : grid.nx, 0 : grid.ny, 0 : grid.nz + 1]


def _assign_magnetic_component_ids(grid, owner, void_numid: int) -> None:
    """Void interior H and restore interface-normal H from the retained cell."""

    shapes = (
        (grid.nx + 1, grid.ny, grid.nz),
        (grid.nx, grid.ny + 1, grid.nz),
        (grid.nx, grid.ny, grid.nz + 1),
    )
    for axis, shape in enumerate(shapes):
        padding = [(0, 0), (0, 0), (0, 0)]
        padding[axis] = (1, 1)
        padded = np.pad(owner, padding, constant_values=-1)
        low_slice = [slice(None), slice(None), slice(None)]
        high_slice = [slice(None), slice(None), slice(None)]
        low_slice[axis] = slice(0, shape[axis])
        high_slice[axis] = slice(1, shape[axis] + 1)
        low = padded[tuple(low_slice)]
        high = padded[tuple(high_slice)]
        target = grid.ID[3 + axis][tuple(slice(0, value) for value in shape)]
        both = (low >= 0) & (high >= 0)
        target[both] = void_numid
        interface = (low >= 0) ^ (high >= 0)
        for coord in np.argwhere(interface):
            cell = coord.copy()
            if high[tuple(coord)] >= 0:
                cell[axis] -= 1
            target[tuple(coord)] = int(grid.solid[tuple(cell)])


def _check_plane_wave_compatibility(
    plane_waves: Iterable, boundary_keys: set[tuple[int, int, int, int]]
) -> None:
    """Require a homogeneous DPW whose TFSF box clears the impedance boundary."""

    if not plane_waves:
        return
    boundary_coordinates = np.asarray([(i, j, k) for _, i, j, k in boundary_keys], dtype=np.int32)
    for plane_wave in plane_waves:
        # The axial DPW samples the completed grid along its propagation
        # line when it is initialised.  An opaque impedance volume on
        # that line is not a valid layered background.  Vector/angle
        # DPWs instead carry an explicit homogeneous background and are
        # safe provided the scatterer is clear of the TFSF corrections.
        if plane_wave.axial:
            raise ValueError(
                "impedance volumes require a vector/angle plane wave with an "
                "explicit homogeneous background; axial plane waves sample the geometry"
            )
        corners = np.asarray(plane_wave.corners, dtype=np.int32)
        lower, upper = corners[:3], corners[3:]
        if not (np.all(boundary_coordinates > lower) and np.all(boundary_coordinates < upper)):
            raise ValueError(
                "an impedance volume illuminated by a plane wave must lie strictly "
                "inside its TFSF box"
            )


def _check_supported_configuration(grid, boundary_keys: set[tuple[int, int, int, int]]) -> None:
    if config.get_model_config().mode != "3D":
        raise ValueError("impedance volumes currently support only 3-D models")
    if config.sim_config.general["solver"] != "cpu":
        raise ValueError("impedance volumes currently support only the CPU solver")
    from gprMax.subgrids.grid import SubGridBaseGrid

    if isinstance(grid, SubGridBaseGrid):
        raise ValueError("impedance volumes are not yet supported in subgrids")
    if getattr(grid, "is_distributed", False) is True:
        raise ValueError("impedance volumes are not yet supported with MPI domain decomposition")
    if config.sim_config.general.get("subgrid", False):
        raise ValueError("impedance volumes are not yet supported in subgridded models")
    if grid.thinwires:
        raise ValueError("impedance volumes cannot yet share a grid with thin wires")
    if grid.virtual_waveguides:
        raise ValueError(
            "impedance volumes do not yet support virtual waveguides; use direct "
            "EigenmodePort planes"
        )
    _check_plane_wave_compatibility(grid.discreteplanewaves, boundary_keys)
    if grid.symmetry_boundaries:
        raise ValueError("impedance volumes cannot yet share a grid with symmetry boundaries")

    for component, i, j, k in boundary_keys:
        if grid.within_pml(np.asarray((i, j, k), dtype=np.int32)):
            raise ValueError("impedance-volume boundary cannot intersect a PML")

    writers: Iterable = (
        list(grid.voltagesources)
        + list(grid.transmissionlines)
        + list(grid.hertziandipoles)
        + list(grid.networkterminals)
    )
    for writer in writers:
        polarisation = getattr(writer, "polarisation", "").lower()
        coord = getattr(writer, "coord", None)
        if polarisation in ("x", "y", "z") and coord is not None:
            key = ("xyz".index(polarisation), *(int(value) for value in coord))
            if key in boundary_keys:
                raise ValueError("an electric source/network terminal overlaps an impedance edge")


class ImpedanceSurfaceSystem:
    """Packed sparse boundary records and local per-port Foster state."""

    def __init__(
        self,
        *,
        edge_info,
        edge_params,
        edge_runtime,
        edge_fraction,
        h_info,
        h_weight,
        port_info,
        port_g,
        port_g_over_Z0,
        port_inv_Z0,
        port_normal,
        port_area,
        model_info,
        model_f,
        model_q,
        model_Z0,
        state_y,
        model_ids,
    ):
        self.edge_info = edge_info
        self.edge_params = edge_params
        self.edge_runtime = edge_runtime
        self.edge_fraction = edge_fraction
        self.h_info = h_info
        self.h_weight = h_weight
        self.port_info = port_info
        self.port_g = port_g
        self.port_g_over_Z0 = port_g_over_Z0
        self.port_inv_Z0 = port_inv_Z0
        self.port_normal = port_normal
        self.port_area = port_area
        self.model_info = model_info
        self.model_f = model_f
        self.model_q = model_q
        self.model_Z0 = model_Z0
        self.state_y = state_y
        self.model_ids = tuple(model_ids)

    @property
    def edge_count(self) -> int:
        return int(self.edge_info.shape[0])

    @property
    def port_count(self) -> int:
        return int(self.port_info.shape[0])

    def reset(self) -> None:
        self.state_y.fill(0)

    @staticmethod
    def _field(fields, component, i, j, k):
        return fields[component][i, j, k]

    def _update_python(self, grid) -> None:
        electric = (grid.Ex, grid.Ey, grid.Ez)
        magnetic = (grid.Hx, grid.Hy, grid.Hz)
        for edge_index, edge in enumerate(self.edge_info):
            component, i, j, k, h_start, h_count, port_start, port_count = edge
            if not 1 <= port_count <= 2:
                raise ValueError("surface-impedance boundary edges require one or two local ports")
            e_old = float(electric[component][i, j, k])
            r_h = 0.0
            for h_index in range(h_start, h_start + h_count):
                h_component, hi, hj, hk = self.h_info[h_index]
                r_h += self.h_weight[h_index] * magnetic[h_component][hi, hj, hk]
            old_e_coefficient, inverse_denominator = self.edge_runtime[edge_index]
            rhs = old_e_coefficient * e_old + r_h
            histories = []
            for port_index in range(port_start, port_start + port_count):
                model_index, state_start = self.port_info[port_index]
                state_count, _ = self.model_info[model_index]
                section = slice(state_start, state_start + state_count)
                history = float(np.sum(self.state_y[section], dtype=np.float64))
                histories.append(history)
                rhs -= self.port_g_over_Z0[port_index] * history
            e_new = rhs * inverse_denominator
            electric[component][i, j, k] = e_new
            midpoint_e = 0.5 * (e_new + e_old)
            for history, port_index in zip(
                histories,
                range(port_start, port_start + port_count),
            ):
                model_index, state_start = self.port_info[port_index]
                state_count, coefficient_start = self.model_info[model_index]
                if not state_count:
                    continue
                section = slice(state_start, state_start + state_count)
                coefficients = slice(
                    coefficient_start,
                    coefficient_start + state_count,
                )
                current = (midpoint_e - history) * self.port_inv_Z0[port_index]
                self.state_y[section] = (
                    self.model_f[coefficients] * self.state_y[section]
                    + self.model_q[coefficients] * current
                )

    def update(self, grid) -> None:
        if not self.edge_count:
            return
        if _cython_update is None:
            self._update_python(grid)
            return
        _cython_update(
            config.get_model_config().ompthreads,
            self.edge_info,
            self.edge_runtime,
            self.h_info,
            self.h_weight,
            self.port_info,
            self.port_g_over_Z0,
            self.port_inv_Z0,
            self.model_info,
            self.model_f,
            self.model_q,
            self.state_y,
            grid.Ex,
            grid.Ey,
            grid.Ez,
            grid.Hx,
            grid.Hy,
            grid.Hz,
        )


def compile_impedance_surfaces(grid) -> ImpedanceSurfaceSystem | None:
    """Compile marked impedance voxels into sparse boundary edge/port records."""

    if not grid.impedance_marker_models:
        grid.impedance_surfaces = None
        return None

    material_ids = set(np.unique(grid.solid).tolist())
    used_markers = [
        (marker_numid, ID)
        for marker_numid, ID in grid.impedance_marker_models.items()
        if marker_numid in material_ids
    ]
    model_ids = tuple(dict.fromkeys(ID for _, ID in used_markers))
    model_index = {ID: index for index, ID in enumerate(model_ids)}
    owner = np.full(grid.solid.shape, -1, dtype=np.int32)
    for marker_numid, ID in used_markers:
        owner[grid.solid == marker_numid] = model_index[ID]
    occupied = np.argwhere(owner >= 0)
    if not occupied.size:
        grid.impedance_surfaces = None
        return None
    minimum = occupied.min(axis=0)
    maximum = occupied.max(axis=0)
    if np.any(minimum == 0) or np.any(maximum == np.asarray(owner.shape) - 1):
        raise ValueError("an impedance volume must have at least one retained cell on every side")

    _validate_impedance_voxel_topology(owner)

    hold = _sentinel_material(grid, "surface-hold")
    void = _sentinel_material(grid, "volume-void")
    _assign_magnetic_component_ids(grid, owner, void.numID)

    edge_records = []
    edge_params = []
    edge_fractions = []
    h_records = []
    h_weights = []
    port_models = []
    port_g = []
    port_normals = []
    port_areas = []
    boundary_keys = set()
    dl = np.asarray((grid.dx, grid.dy, grid.dz), dtype=np.float64)
    e0 = float(config.sim_config.em_consts["e0"])
    side_b = (-1, 1, 1, -1)
    side_c = (-1, -1, 1, 1)
    across_b = (1, 0, 3, 2)
    across_c = (3, 2, 1, 0)

    for axis in range(3):
        b_axis = (axis + 1) % 3
        c_axis = (axis + 2) % 3
        quadrants, offsets = _electric_quadrants(owner, axis)
        metal = tuple(value >= 0 for value in quadrants)
        count = sum(value.astype(np.uint8) for value in metal)
        target = _component_valid_view(grid.ID[axis], axis, grid)
        target[count == 4] = void.numID

        for coord_array in np.argwhere((count > 0) & (count < 4)):
            coord = tuple(int(value) for value in coord_array)
            qowners = [int(values[coord]) for values in quadrants]
            retained = [index for index, value in enumerate(qowners) if value < 0]
            m_eps = 0.0
            m_sigma = 0.0
            local_h = {}
            local_ports = {}
            quarter_area = dl[b_axis] * dl[c_axis] / 4

            for quadrant in retained:
                cell = tuple(coord[dim] + offsets[quadrant][dim] for dim in range(3))
                material = grid.materials[int(grid.solid[cell])]
                if hasattr(material, "poles"):
                    raise ValueError(
                        "impedance-volume boundary does not yet support a dispersive retained material"
                    )
                m_eps += e0 * float(material.er) * quarter_area
                m_sigma += float(material.se) * quarter_area

                hcoord = list(coord)
                if side_b[quadrant] < 0:
                    hcoord[b_axis] -= 1
                hkey = (c_axis, *hcoord)
                local_h[hkey] = local_h.get(hkey, 0.0) + side_b[quadrant] * dl[c_axis] / 2

                hcoord = list(coord)
                if side_c[quadrant] < 0:
                    hcoord[c_axis] -= 1
                hkey = (b_axis, *hcoord)
                local_h[hkey] = local_h.get(hkey, 0.0) - side_c[quadrant] * dl[b_axis] / 2

                neighbour = across_b[quadrant]
                if qowners[neighbour] >= 0:
                    key = (b_axis, side_b[quadrant], qowners[neighbour])
                    local_ports[key] = local_ports.get(key, 0.0) + dl[c_axis] / 2
                neighbour = across_c[quadrant]
                if qowners[neighbour] >= 0:
                    key = (c_axis, side_c[quadrant], qowners[neighbour])
                    local_ports[key] = local_ports.get(key, 0.0) + dl[b_axis] / 2

            h_start = len(h_records)
            for key, weight in sorted(local_h.items()):
                h_records.append(key)
                h_weights.append(weight)
            port_start = len(port_models)
            for (normal_axis, normal_sign, surface_model), length in sorted(local_ports.items()):
                port_models.append(surface_model)
                port_g.append(-length)
                port_normals.append((normal_axis, normal_sign))
                port_areas.append(length * dl[axis])
            if not 1 <= len(local_ports) <= 2:
                raise RuntimeError(
                    "a manifold impedance boundary edge must have one or two surface ports"
                )

            edge_records.append(
                (
                    axis,
                    *coord,
                    h_start,
                    len(local_h),
                    port_start,
                    len(local_ports),
                )
            )
            edge_params.append((m_eps / grid.dt + m_sigma / 2, m_eps / grid.dt - m_sigma / 2))
            edge_fractions.append(len(retained) / 4)
            target[coord] = hold.numID
            boundary_keys.add((axis, *coord))

    _check_supported_configuration(grid, boundary_keys)

    marker_ids = np.asarray(tuple(grid.impedance_marker_models), dtype=np.uint32)
    for component in range(6):
        view = _component_valid_view(grid.ID[component], component, grid)
        if np.isin(view, marker_ids).any():
            raise RuntimeError("impedance marker material survived component compilation")

    real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
    discrete = [grid.surface_impedance_models[ID].discretise(grid.dt) for ID in model_ids]
    model_info = np.zeros((len(discrete), 2), dtype=np.int32)
    f_values = []
    q_values = []
    coefficient_offset = 0
    for index, item in enumerate(discrete):
        order = item.G.size
        diagonal_f = np.diag(item.F)
        if not np.array_equal(item.F, np.diag(diagonal_f)):
            raise ValueError(
                f"surface-impedance model {model_ids[index]!r} cannot use the local "
                "Foster runtime because its discrete state matrix is not diagonal"
            )
        model_info[index] = (order, coefficient_offset)
        f_values.extend(diagonal_f)
        # Scale x_m by L_m so the boundary history is the direct local sum of
        # y_m and every independent pole advances as y'_m=f_m*y_m+q_m*K.
        q_values.extend(item.L * item.G)
        coefficient_offset += order

    port_info = np.zeros((len(port_models), 2), dtype=np.int32)
    state_offset = 0
    for index, surface_model in enumerate(port_models):
        order = int(model_info[surface_model, 0])
        port_info[index] = (surface_model, state_offset)
        state_offset += order

    model_Z0 = np.asarray([item.Z0 for item in discrete], dtype=np.float64)
    port_inv_Z0 = 1.0 / model_Z0[np.asarray(port_models, dtype=np.intp)]
    port_g_over_Z0 = np.asarray(port_g, dtype=np.float64) * port_inv_Z0
    edge_runtime = np.empty((len(edge_records), 2), dtype=np.float64)
    for edge_index, edge in enumerate(edge_records):
        port_start = edge[6]
        port_stop = port_start + edge[7]
        metric_admittance = float(np.sum(port_g_over_Z0[port_start:port_stop]))
        a_plus, a_minus = edge_params[edge_index]
        denominator = a_plus - 0.5 * metric_admittance
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("surface-impedance local edge solve has a non-positive denominator")
        edge_runtime[edge_index] = (
            a_minus + 0.5 * metric_admittance,
            1.0 / denominator,
        )

    def packed(values):
        array = np.asarray(values, dtype=real_dtype)
        return np.ascontiguousarray(array if array.size else np.zeros(1, dtype=real_dtype))

    system = ImpedanceSurfaceSystem(
        edge_info=np.ascontiguousarray(np.asarray(edge_records, dtype=np.int32).reshape(-1, 8)),
        edge_params=np.ascontiguousarray(np.asarray(edge_params, dtype=real_dtype).reshape(-1, 2)),
        edge_runtime=np.ascontiguousarray(edge_runtime, dtype=real_dtype),
        edge_fraction=np.ascontiguousarray(np.asarray(edge_fractions, dtype=real_dtype)),
        h_info=np.ascontiguousarray(np.asarray(h_records, dtype=np.int32).reshape(-1, 4)),
        h_weight=packed(h_weights),
        port_info=np.ascontiguousarray(port_info),
        port_g=packed(port_g),
        port_g_over_Z0=packed(port_g_over_Z0),
        port_inv_Z0=packed(port_inv_Z0),
        port_normal=np.ascontiguousarray(np.asarray(port_normals, dtype=np.int8).reshape(-1, 2)),
        port_area=packed(port_areas),
        model_info=np.ascontiguousarray(model_info),
        model_f=packed(f_values),
        model_q=packed(q_values),
        model_Z0=packed(model_Z0),
        state_y=np.zeros(max(state_offset, 1), dtype=real_dtype),
        model_ids=model_ids,
    )
    grid.impedance_surfaces = system
    logger.info(
        f"Compiled {system.edge_count} impedance boundary E edges and "
        f"{system.port_count} surface-current ports [{grid.name}]."
    )
    return system
