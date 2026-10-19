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

"""Sparse surface-impedance models and voxel-boundary compilation.

Grid-aligned impedance volumes support the main 3-D and reduced 2-D CPU
grids. Geometry remains represented by
the normal dense Yee arrays, but conductor-interior components are assigned a
private void coefficient row and boundary tangential electric edges are owned
by this sparse subsystem.
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import numpy.typing as npt

from gprMax import config
from gprMax.materials import Material
from gprMax.mode2d import mode2d_geometry

logger = logging.getLogger(__name__)

PRIVATE_IMPEDANCE_ID_PREFIX = "__impedance_"
# Keep the voxel-boundary update away from the precision-sensitive CFL
# endpoint. Scene applies this through TimeStepStabilityFactor before any
# time-dependent objects or coefficients are constructed.
MAX_SIBC_TIMESTEP_FACTOR = 0.99

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
        pmc = order == 0 and np.isposinf(D)
        if not all(np.all(np.isfinite(value)) for value in (A, B, C)) or not (
            np.isfinite(D) or pmc
        ):
            raise ValueError(
                "surface-impedance realization coefficients must be finite, "
                "except positive infinite zero-order resistance (PMC)"
            )
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
    def is_pmc(self) -> bool:
        """Exact zero-admittance, opaque PMC at the main-voxel boundary."""
        return self.order == 0 and bool(np.isposinf(self.D))

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
        if self.is_pmc:
            return DiscreteSurfaceImpedance(
                np.empty((0, 0), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.inf,
            )
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


def impedance_marker_id(model_id: str) -> str:
    """Return the reserved voxel-marker ID for one surface model."""

    return f"__impedance_volume__{model_id}"


def private_impedance_material_id(role: str) -> str:
    """Return the reserved coefficient-row ID for one private impedance role."""

    return f"__impedance_{role.replace('-', '_')}__"


def create_impedance_marker_material(grid, model_id: str) -> Material:
    """Return the private voxel marker associated with a surface model."""

    existing_numid = next(
        (numid for numid, value in grid.impedance_marker_models.items() if value == model_id),
        None,
    )
    if existing_numid is not None:
        return grid.materials[existing_numid]

    marker_id = impedance_marker_id(model_id)
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
    ID = private_impedance_material_id(role)
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


_PRIVATE_IMPEDANCE_ROLES = ("surface-hold", "volume-void")


def surface_impedance_model_to_geometry_metadata(model: SurfaceImpedanceModel) -> dict[str, Any]:
    """Describe one model for the material database paired with a geometry file."""

    requested_order = model.fit_requested_order
    if requested_order is not None and not isinstance(requested_order, str):
        requested_order = int(requested_order)
    return {
        "ID": model.ID,
        "A": [[float(value) for value in row] for row in np.asarray(model.A)],
        "B": [float(value) for value in np.asarray(model.B)],
        "C": [float(value) for value in np.asarray(model.C)],
        "D": "inf" if np.isposinf(model.D) else float(model.D),
        "fit_fmin_hz": float(model.fit_fmin_hz),
        "fit_fmax_hz": "inf" if np.isposinf(model.fit_fmax_hz) else float(model.fit_fmax_hz),
        "allow_active": bool(model.allow_active),
        "preset": model.preset,
        "provenance": model.provenance,
        "conductivity_s_per_m": (
            None if model.conductivity_s_per_m is None else float(model.conductivity_s_per_m)
        ),
        "fit_requested_order": requested_order,
        "fit_pole_count": model.fit_pole_count,
        "fit_tolerance": model.fit_tolerance,
        "fit_max_relative_error": model.fit_max_relative_error,
        "fit_rms_relative_error": model.fit_rms_relative_error,
        "fit_method": model.fit_method,
        "plot_fit_in_full_run": bool(model.plot_fit_in_full_run),
    }


def _geometry_metadata_float(
    metadata: Mapping[str, Any], key: str, *, positive_infinity: bool = False
) -> float:
    value = metadata.get(key)
    if isinstance(value, str):
        if positive_infinity and value == "inf":
            return np.inf
        raise ValueError(f"Surface-impedance metadata field '{key}' is not a finite number")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Surface-impedance metadata field '{key}' is not a number")
    value = float(value)
    if not np.isfinite(value) and not (positive_infinity and np.isposinf(value)):
        raise ValueError(f"Surface-impedance metadata field '{key}' is not a finite number")
    return value


def _geometry_metadata_optional_float(
    metadata: Mapping[str, Any], key: str
) -> float | None:
    return None if metadata.get(key) is None else _geometry_metadata_float(metadata, key)


def surface_impedance_model_from_geometry_metadata(
    model_id: str, metadata: Mapping[str, Any]
) -> SurfaceImpedanceModel:
    """Rebuild one model from the metadata of an exported geometry database."""

    if not isinstance(metadata, Mapping):
        raise ValueError(f"Surface-impedance metadata for '{model_id}' must be an object")
    missing = [key for key in ("A", "B", "C", "D") if key not in metadata]
    if missing:
        raise ValueError(
            f"Surface-impedance metadata for '{model_id}' is missing {', '.join(missing)}"
        )
    try:
        A = np.asarray(metadata["A"], dtype=np.float64)
        B = np.asarray(metadata["B"], dtype=np.float64)
        C = np.asarray(metadata["C"], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Surface-impedance metadata for '{model_id}' has invalid coefficients"
        ) from error
    allow_active = metadata.get("allow_active")
    plot_fit = metadata.get("plot_fit_in_full_run")
    if not isinstance(allow_active, bool) or not isinstance(plot_fit, bool):
        raise ValueError(
            f"Surface-impedance metadata for '{model_id}' has invalid boolean flags"
        )
    for key in ("preset", "provenance", "fit_method"):
        if metadata.get(key) is not None and not isinstance(metadata[key], str):
            raise ValueError(f"Surface-impedance metadata field '{key}' must be a string")
    requested_order = metadata.get("fit_requested_order")
    if requested_order is not None and not isinstance(requested_order, (str, int)):
        raise ValueError(
            "Surface-impedance metadata field 'fit_requested_order' must be a string or integer"
        )
    return SurfaceImpedanceModel(
        ID=model_id,
        A=A,
        B=B,
        C=C,
        D=_geometry_metadata_float(metadata, "D", positive_infinity=True),
        fit_fmin_hz=_geometry_metadata_float(metadata, "fit_fmin_hz"),
        fit_fmax_hz=_geometry_metadata_float(metadata, "fit_fmax_hz", positive_infinity=True),
        allow_active=allow_active,
        preset=metadata.get("preset"),
        provenance=metadata.get("provenance"),
        conductivity_s_per_m=_geometry_metadata_optional_float(metadata, "conductivity_s_per_m"),
        fit_requested_order=requested_order,
        fit_pole_count=metadata.get("fit_pole_count"),
        fit_tolerance=_geometry_metadata_optional_float(metadata, "fit_tolerance"),
        fit_max_relative_error=_geometry_metadata_optional_float(
            metadata, "fit_max_relative_error"
        ),
        fit_rms_relative_error=_geometry_metadata_optional_float(
            metadata, "fit_rms_relative_error"
        ),
        fit_method=metadata.get("fit_method"),
        plot_fit_in_full_run=plot_fit,
    )


def restore_surface_impedance_model(
    grid, model_id: str, metadata: Mapping[str, Any]
) -> SurfaceImpedanceModel:
    """Restore one exported model, or verify an identical scene definition."""

    incoming = surface_impedance_model_from_geometry_metadata(model_id, metadata)
    existing = grid.surface_impedance_models.get(model_id)
    if existing is not None:
        if surface_impedance_model_to_geometry_metadata(existing) != dict(metadata):
            raise ValueError(
                f"Geometry surface-impedance model '{model_id}' conflicts with the model "
                "already defined in this scene"
            )
        return existing
    if any(material.ID == model_id for material in grid.materials):
        raise ValueError(
            f"Geometry surface-impedance ID '{model_id}' conflicts with an existing material ID"
        )
    grid.surface_impedance_models[model_id] = incoming
    return incoming


def restore_private_impedance_material(grid, role: str) -> _PrivateImpedanceMaterial:
    """Recreate one exported private coefficient row under its fixed role ID."""

    if role not in _PRIVATE_IMPEDANCE_ROLES:
        raise ValueError(f"Unknown private impedance material role {role!r}")
    return _sentinel_material(grid, role)


def _electric_quadrants(owner: npt.NDArray[np.int32], axis: int, *, outside: int = -1):
    """Return cyclic surrounding-cell owner arrays and cell offsets for one E axis."""

    nx, ny, nz = owner.shape
    if axis == 0:
        padded = np.pad(owner, ((0, 0), (1, 1), (1, 1)), constant_values=outside)
        arrays = (
            padded[:, 0 : ny + 1, 0 : nz + 1],
            padded[:, 1 : ny + 2, 0 : nz + 1],
            padded[:, 1 : ny + 2, 1 : nz + 2],
            padded[:, 0 : ny + 1, 1 : nz + 2],
        )
        offsets = ((0, -1, -1), (0, 0, -1), (0, 0, 0), (0, -1, 0))
    elif axis == 1:
        padded = np.pad(owner, ((1, 1), (0, 0), (1, 1)), constant_values=outside)
        arrays = (
            padded[0 : nx + 1, :, 0 : nz + 1],
            padded[0 : nx + 1, :, 1 : nz + 2],
            padded[1 : nx + 2, :, 1 : nz + 2],
            padded[1 : nx + 2, :, 0 : nz + 1],
        )
        offsets = ((-1, 0, -1), (-1, 0, 0), (0, 0, 0), (0, 0, -1))
    else:
        padded = np.pad(owner, ((1, 1), (1, 1), (0, 0)), constant_values=outside)
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


def _validate_impedance_voxel_topology(
    owner: npt.NDArray[np.int32], *, coordinate_offset=(0, 0, 0)
) -> None:
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
            coord = tuple(
                int(value) + offset
                for value, offset in zip(
                    np.unravel_index(flat_index, diagonal.shape), coordinate_offset
                )
            )
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
        vertex = tuple(value + 1 + offset for value, offset in zip(lower, coordinate_offset))
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
    """Void interior H and restore unconstrained interface H from the retained cell."""

    shapes = (
        (grid.nx + 1, grid.ny, grid.nz),
        (grid.nx, grid.ny + 1, grid.nz),
        (grid.nx, grid.ny, grid.nz + 1),
    )
    reduced = mode2d_geometry(config.get_model_config().mode)
    for axis, shape in enumerate(shapes):
        if reduced and f"H{'xyz'[axis]}" not in reduced.active_magnetic:
            continue
        padding = [(0, 0), (0, 0), (0, 0)]
        padding[axis] = (1, 1)
        # Outside the physical grid is neither retained dielectric (-1) nor
        # an impedance owner. At a symmetry cut through metal, normal H is
        # interior to the reflected body, not an exposed surface sample.
        padded = np.pad(owner, padding, constant_values=-2)
        low_slice = [slice(None), slice(None), slice(None)]
        high_slice = [slice(None), slice(None), slice(None)]
        low_slice[axis] = slice(0, shape[axis])
        high_slice[axis] = slice(1, shape[axis] + 1)
        low = padded[tuple(low_slice)]
        high = padded[tuple(high_slice)]
        target = grid.ID[3 + axis][tuple(slice(0, value) for value in shape)]
        has_metal = (low >= 0) | (high >= 0)
        if reduced:
            live = np.zeros(shape, dtype=bool)
            selection = [slice(None)] * 3
            selection[reduced.invariant_axis] = reduced.live_index
            live[tuple(selection)] = True
            has_metal &= live
        has_retained = (low == -1) | (high == -1)
        target[has_metal & ~has_retained] = void_numid
        interface = has_metal & has_retained
        for coord in np.argwhere(interface):
            if grid.materials[int(target[tuple(coord)])].is_pmc:
                # Preserve an explicit component constraint, including a
                # MagneticEdge with no corresponding PMC voxel in solid.
                continue
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
    _check_plane_wave_compatibility(grid.discreteplanewaves, boundary_keys)

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
        edge_dispersion=None,
        pole_offsets=None,
        pole_coeffs=None,
        state_p=None,
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
        # S = epsilon0 * retained_area * T is a boundary-owned bulk history.
        # Store complex values as real/imaginary pairs in the field precision,
        # so the same sparse kernel handles Debye and complex Lorentz poles.
        dtype = edge_runtime.dtype
        self.edge_dispersion = (
            np.empty((0, 2), dtype=dtype) if edge_dispersion is None else edge_dispersion
        )
        self.pole_offsets = np.empty(0, dtype=np.int32) if pole_offsets is None else pole_offsets
        self.pole_coeffs = np.empty((0, 6), dtype=dtype) if pole_coeffs is None else pole_coeffs
        self.state_p = (
            np.zeros((len(self.pole_coeffs), 2), dtype=dtype) if state_p is None else state_p
        )

    @property
    def edge_count(self) -> int:
        return int(self.edge_info.shape[0])

    @property
    def port_count(self) -> int:
        return int(self.port_info.shape[0])

    def reset(self) -> None:
        self.state_y.fill(0)
        self.state_p.fill(0)
        self._pending_pml_rhs = None
        self._pending_modal_delta = None

    def capture_modal_source(self, grid, source, iteration):
        """Collect modal forcing for the implicit solve without modifying old E.

        Native modal kernels use dense curl coefficients, which are zero on
        held SIBC rows. A source-only coefficient table supplies A/den/dn on
        those rows and zero elsewhere. The resulting solved E increment is
        applied after the sparse solve together with its ADE correction.
        """
        normal = int(source.normal_axis)
        indices = np.flatnonzero(
            (self.edge_info[:, 0] != normal) & (self.edge_info[:, normal + 1] == source.plane_index)
        )
        if not len(indices):
            return
        if not hasattr(self, "_modal_source_tables"):
            self._modal_source_tables = {}
        if normal not in self._modal_source_tables:
            ids = np.zeros(self.edge_count, dtype=grid.ID.dtype)
            coefficients = list(np.zeros_like(grid.updatecoeffsE))
            material_rows = {}
            for index, edge in enumerate(self.edge_info):
                if edge[0] == normal:
                    continue
                area = float(self.edge_fraction[index]) * float(
                    np.prod(np.delete(grid.dl, edge[0]))
                )
                coefficient = area * float(self.edge_runtime[index, 1]) / grid.dl[normal]
                if coefficient not in material_rows:
                    row = np.zeros(grid.updatecoeffsE.shape[1], dtype=grid.updatecoeffsE.dtype)
                    row[normal + 1] = coefficient
                    material_rows[coefficient] = len(coefficients)
                    coefficients.append(row)
                ids[index] = material_rows[coefficient]
            self._modal_source_tables[normal] = ids, np.ascontiguousarray(coefficients)
        edges = self.edge_info[indices, :4]
        held = []
        for component, field in enumerate((grid.Ex, grid.Ey, grid.Ez)):
            selection = np.flatnonzero(edges[:, 0] == component)
            points = tuple(edges[selection, 1:].T)
            held.append((field, points, field[points].copy(), selection))
            field[points] = 0
        id_points = tuple(edges.T)
        original_ids = grid.ID[id_points].copy()
        coefficients = grid.updatecoeffsE
        delta = np.zeros(len(indices), dtype=self.edge_runtime.dtype)
        try:
            source_ids, grid.updatecoeffsE = self._modal_source_tables[normal]
            grid.ID[id_points] = source_ids[indices]
            source.update_eigenmode_electric(iteration, grid)
            for field, points, _, selection in held:
                delta[selection] = field[points]
        finally:
            grid.ID[id_points] = original_ids
            grid.updatecoeffsE = coefficients
            for field, points, values, _ in held:
                field[points] = values
        if getattr(self, "_pending_modal_delta", None) is None:
            self._pending_modal_delta = np.zeros(self.edge_count, dtype=self.edge_runtime.dtype)
        self._pending_modal_delta[indices] += delta

    @contextmanager
    def capture_electric_pml(self, grid):
        """Collect native PML circulation without changing the physical old E.

        Electric PML kernels read only H and their own convolution histories.
        Temporarily zeroing the held boundary E therefore gives the additive
        correction directly, without subtracting nearly equal field values.
        """
        indices = getattr(self, "pml_edge_indices", ())
        if not len(indices):
            yield
            return
        edges = self.edge_info[indices, :4]
        held = []
        for component, field in enumerate((grid.Ex, grid.Ey, grid.Ez)):
            selection = np.flatnonzero(edges[:, 0] == component)
            points = tuple(edges[selection, 1:].T)
            held.append((field, points, field[points].copy(), selection))
            field[points] = 0
        correction = np.zeros(len(indices), dtype=self.edge_runtime.dtype)
        try:
            yield
            for field, points, _, selection in held:
                correction[selection] = field[points]
            self._pending_pml_rhs = correction * self.pml_edge_scale
        finally:
            for field, points, values, _ in held:
                field[points] = values

    def _electric_correction_states(self):
        """Map an additive solved E increment to its trapezoidal histories."""
        if not hasattr(self, "_correction_states"):
            edge_indices, state_indices, weights = [], [], []
            for edge_index, edge in enumerate(self.edge_info):
                for port in range(int(edge[6]), int(edge[6] + edge[7])):
                    model, start = self.port_info[port]
                    count, coefficient = self.model_info[model]
                    edge_indices.extend([edge_index] * count)
                    state_indices.extend(range(start, start + count))
                    weights.extend(
                        self.model_q[coefficient : coefficient + count]
                        * (0.5 * self.port_inv_Z0[port])
                    )
            self._correction_states = (
                np.asarray(edge_indices, dtype=np.intp),
                np.asarray(state_indices, dtype=np.intp),
                np.asarray(weights, dtype=self.edge_runtime.dtype),
            )
        return self._correction_states

    def apply_electric_correction(self, grid, edge_indices, delta_e, *, update_fields=True):
        """Add solved forcing and its exact linear Foster/polarization response.

        For a circulation forcing r, delta_e = r / denominator. Correcting
        y by q*delta_e/(2*Z0) is algebraically identical to including r in the
        implicit edge solve; treating it as a change to E_old is not.
        """
        edge_indices = np.asarray(edge_indices, dtype=np.intp)
        delta_e = np.asarray(delta_e, dtype=self.edge_runtime.dtype)
        if update_fields:
            edges = self.edge_info[edge_indices, :4]
            for component, field in enumerate((grid.Ex, grid.Ey, grid.Ez)):
                selection = edges[:, 0] == component
                field[tuple(edges[selection, 1:].T)] += delta_e[selection]
        if self.state_y.size:
            full_delta = np.zeros(self.edge_count, dtype=self.edge_runtime.dtype)
            full_delta[edge_indices] = delta_e
            rows, states, weights = self._electric_correction_states()
            self.state_y[states] += weights * full_delta[rows]
        if self.pole_coeffs.size:
            for row, increment in zip(edge_indices, delta_e):
                start, stop = self.pole_offsets[row : row + 2]
                self.state_p[start:stop] -= self.pole_coeffs[start:stop, 2:4] * increment

    def polarization_admittance(self, edge_index: int, theta: float) -> complex:
        """Exact harmonic bulk-polarization load relative to midpoint-time E.

        The real-output recurrence is S' = f*S + b*(E_old-E_new),
        Phi = Re(c*S). Both conjugate branches are needed for Lorentz poles;
        taking the real part of a complex-frequency transfer is incorrect.
        The instantaneous and Drude conductivity terms are included here,
        leaving edge_params as the physical epsilon-infinity/sigma masses.
        """
        if not self.pole_coeffs.size:
            return 0j
        start, stop = self.pole_offsets[edge_index : edge_index + 2]
        coefficients = self.pole_coeffs[start:stop].astype(np.float64)
        f = coefficients[:, 0] + 1j * coefficients[:, 1]
        b = coefficients[:, 2] + 1j * coefficients[:, 3]
        c = coefficients[:, 4] + 1j * coefficients[:, 5]
        z = np.exp(1j * theta)
        cb = c * b
        transfer = 0.5 * np.sum(cb / (z - f) + cb.conj() / (z - f.conj()))
        plus, minus = self.edge_dispersion[edge_index].astype(np.float64)
        return complex(
            1j * np.sin(theta / 2) * (plus + minus - 2 * transfer)
            + np.cos(theta / 2) * (plus - minus)
        )

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
            if self.pole_coeffs.size:
                start, stop = self.pole_offsets[edge_index : edge_index + 2]
                coefficients = self.pole_coeffs[start:stop]
                states = self.state_p[start:stop]
                rhs -= float(
                    np.sum(
                        coefficients[:, 4] * states[:, 0] - coefficients[:, 5] * states[:, 1],
                        dtype=np.float64,
                    )
                )
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
            if self.pole_coeffs.size:
                old_real = states[:, 0].copy()
                old_imag = states[:, 1].copy()
                states[:, 0] = (
                    coefficients[:, 0] * old_real
                    - coefficients[:, 1] * old_imag
                    + coefficients[:, 2] * (e_old - e_new)
                )
                states[:, 1] = (
                    coefficients[:, 1] * old_real
                    + coefficients[:, 0] * old_imag
                    + coefficients[:, 3] * (e_old - e_new)
                )
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
        # A virtual guide owns its aperture E and detached rear. Preserve
        # those completed values without changing the physical sparse rows,
        # which modal extraction and metadata still need on reused runs.
        frozen = getattr(self, "virtual_frozen_edges", None)
        held = []
        held_states = []
        if frozen is not None and len(frozen):
            for component, field in enumerate((grid.Ex, grid.Ey, grid.Ez)):
                points = frozen[frozen[:, 0] == component, 1:]
                indices = tuple(points.T)
                held.append((field, indices, field[indices].copy()))
            if getattr(self, "_frozen_state_key", None) is not frozen:
                coordinates = {tuple(edge) for edge in frozen}
                frozen_rows = [
                    index
                    for index, edge in enumerate(self.edge_info)
                    if tuple(edge[:4]) in coordinates
                ]
                rows, states, _ = self._electric_correction_states()
                self._frozen_y = states[np.isin(rows, frozen_rows)]
                self._frozen_p = np.asarray(
                    [
                        pole
                        for row in frozen_rows
                        for pole in (
                            range(*self.pole_offsets[row : row + 2])
                            if self.pole_coeffs.size
                            else ()
                        )
                    ],
                    dtype=np.intp,
                )
                self._frozen_state_key = frozen
            held_states = [
                (self.state_y, self._frozen_y, self.state_y[self._frozen_y].copy()),
                (self.state_p, self._frozen_p, self.state_p[self._frozen_p].copy()),
            ]
        try:
            self._advance(grid)
            rhs = getattr(self, "_pending_pml_rhs", None)
            if rhs is not None:
                indices = self.pml_edge_indices
                self.apply_electric_correction(grid, indices, rhs * self.edge_runtime[indices, 1])
            delta = getattr(self, "_pending_modal_delta", None)
            if delta is not None:
                indices = np.flatnonzero(delta)
                self.apply_electric_correction(grid, indices, delta[indices])
        finally:
            self._pending_pml_rhs = None
            self._pending_modal_delta = None
            for field, indices, values in held:
                field[indices] = values
            for state, indices, values in held_states:
                state[indices] = values

    def _advance(self, grid) -> None:
        if not self.edge_count:
            return
        if _cython_update is None:
            self._update_python(grid)
            return
        polarization = (
            (self.pole_offsets, self.pole_coeffs, self.state_p) if self.pole_coeffs.size else ()
        )
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
            *polarization,
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
    reduced = mode2d_geometry(config.get_model_config().mode)
    if reduced:
        section = np.take(grid.solid, 0, axis=reduced.invariant_axis)
        if not np.all(grid.solid == np.expand_dims(section, reduced.invariant_axis)):
            raise ValueError(
                "2-D impedance geometry and retained materials must be uniform "
                "across the invariant-axis storage layers; extrude volumes "
                "through the full invariant dimension using float('inf')."
            )
    minimum = occupied.min(axis=0)
    maximum = occupied.max(axis=0)
    symmetry = grid.symmetry_boundaries
    mirror_padding = []
    for axis, letter in enumerate("xyz"):
        if reduced and axis == reduced.invariant_axis:
            mirror_padding.append((1, 1))
            continue
        faces = (f"{letter}0", f"{letter}max")
        mirrored = tuple(symmetry.get(face) in ("pec", "pmc") for face in faces)
        touches = (minimum[axis] == 0, maximum[axis] == owner.shape[axis] - 1)
        continued = tuple(grid.pmls["thickness"][face] > 0 for face in faces)
        for face, contact, mirror, continuation in zip(faces, touches, mirrored, continued):
            if contact and not (mirror or continuation):
                raise ValueError(
                    "an impedance volume must have at least one retained cell on every "
                    f"non-symmetry side; it touches domain face {face}"
                )
        mirror_padding.append(
            tuple(int(mirror or continuation) for mirror, continuation in zip(mirrored, continued))
        )

    # Check connectivity in the reflected geometry too, but report coordinates
    # in the original grid. Only one image cell per declared face is needed.
    _validate_impedance_voxel_topology(owner)
    if any(any(width) for width in mirror_padding):
        _validate_impedance_voxel_topology(
            np.pad(owner, mirror_padding, mode="edge"),
            coordinate_offset=tuple(-width[0] for width in mirror_padding),
        )

    hold = _sentinel_material(grid, "surface-hold")
    void = _sentinel_material(grid, "volume-void")
    _assign_magnetic_component_ids(grid, owner, void.numID)

    edge_records = []
    edge_params = []
    edge_fractions = []
    edge_dispersion = []
    pole_offsets = [0]
    pole_coeffs = []
    material_dispersion = {}
    h_records = []
    h_weights = []
    port_models = []
    port_g = []
    port_normals = []
    port_areas = []
    boundary_keys = set()
    pec_contact_count = 0
    pec_contact_examples = []
    dl = np.asarray((grid.dx, grid.dy, grid.dz), dtype=np.float64)
    e0 = float(config.sim_config.em_consts["e0"])
    side_b = (-1, 1, 1, -1)
    side_c = (-1, -1, 1, 1)
    across_b = (1, 0, 3, 2)
    across_c = (3, 2, 1, 0)

    for axis in range(3):
        if reduced and f"E{'xyz'[axis]}" not in reduced.active_electric:
            continue
        b_axis = (axis + 1) % 3
        c_axis = (axis + 2) % 3
        quadrants, offsets = _electric_quadrants(owner, axis, outside=-2)
        metal = tuple(value >= 0 for value in quadrants)
        count = sum(value.astype(np.uint8) for value in metal)
        physical_count = sum((value != -2).astype(np.uint8) for value in quadrants)
        target = _component_valid_view(grid.ID[axis], axis, grid)
        eligible = np.ones(count.shape, dtype=bool)
        if reduced:
            eligible[:] = False
            selection = [slice(None)] * 3
            selection[reduced.invariant_axis] = reduced.live_index
            eligible[tuple(selection)] = True
        target[(count == physical_count) & eligible] = void.numID

        for coord_array in np.argwhere((count > 0) & (count < physical_count) & eligible):
            coord = tuple(int(value) for value in coord_array)
            # Keep every geometric boundary edge in compatibility checks,
            # including edges subsequently constrained by PEC.
            boundary_keys.add((axis, *coord))
            qowners = [int(values[coord]) for values in quadrants]
            # On a PMC symmetry plane the contour closes with H_t=0. Keep
            # only physical quadrants: their half area and half port lengths
            # give the same E update as odd-H images, without counting image
            # surface area or allocating duplicate surface-current states.
            # PEC symmetry E IDs are preserved by the constraint branch below.
            retained = [index for index, value in enumerate(qowners) if value == -1]
            retained_materials = {}
            for quadrant in retained:
                cell = tuple(coord[dim] + offsets[quadrant][dim] for dim in range(3))
                material = grid.materials[int(grid.solid[cell])]
                # Directional geometry now stores an anisotropic material in
                # solid (older geometry used a dielectric-smoothed average).
                # This scalar boundary compiler cannot resolve its conductor
                # axis. Preserve the guard for both material representations.
                if material.type in ("dielectric-smoothed", "anisotropic") and (
                    material.is_pec or material.is_pmc
                ):
                    raise ValueError(
                        "impedance-volume boundary does not yet support directional "
                        "PEC/PMC mixtures; use isotropic PEC/PMC volumes"
                    )
                retained_materials[quadrant] = material

            pec_quadrants = [q for q, material in retained_materials.items() if material.is_pec]
            existing_material = grid.materials[int(target[coord])]
            if pec_quadrants or existing_material.is_pec:
                # A PEC incident voxel constrains the shared tangential E
                # even when a later impedance primitive overwrote its ID.
                # Also preserve explicit component constraints (e.g. plates).
                target[coord] = (
                    existing_material.numID
                    if existing_material.is_pec
                    else retained_materials[pec_quadrants[0]].numID
                )
                if len(pec_quadrants) == 1 and len(retained) == 3:
                    opposite = (pec_quadrants[0] + 2) % 4
                    if qowners[opposite] >= 0:
                        pec_contact_count += 1
                        if len(pec_contact_examples) < 3:
                            pec_contact_examples.append(f"E{'xyz'[axis]} at {coord}")
                continue

            m_eps = 0.0
            m_sigma = 0.0
            local_h = {}
            local_ports = {}
            dispersive_areas = {}
            quarter_area = dl[b_axis] * dl[c_axis] / 4

            for quadrant in retained:
                material = retained_materials[quadrant]
                if getattr(material, "poles", 0):
                    dispersive_areas[material.numID] = (
                        dispersive_areas.get(material.numID, 0.0) + quarter_area
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
                h_component, hi, hj, hk = key
                if reduced and f"H{'xyz'[h_component]}" not in reduced.active_magnetic:
                    continue
                h_material = grid.materials[int(grid.ID[3 + h_component, hi, hj, hk])]
                # Prune only a known-zero term of the existing circulation.
                # PMC remains a retained electric material: neither area nor
                # surface-current ports change when an H sample is clamped.
                if h_material.is_pmc:
                    continue
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
                    len(h_records) - h_start,
                    port_start,
                    len(local_ports),
                )
            )
            edge_params.append((m_eps / grid.dt + m_sigma / 2, m_eps / grid.dt - m_sigma / 2))
            correction = np.zeros(2, dtype=np.float64)
            for material_id, area in sorted(dispersive_areas.items()):
                if material_id not in material_dispersion:
                    material = grid.materials[material_id]
                    # Compilation precedes the dense coefficient-table build.
                    # Reuse its constitutive calculation once per material;
                    # recalculation later is idempotent, including Drude sigma.
                    material.calculate_update_coeffsE(grid)
                    base_plus = e0 * float(material.er) / grid.dt + float(material.se) / 2
                    base_minus = e0 * float(material.er) / grid.dt - float(material.se) / 2
                    delta = np.asarray(
                        (1 / material.srce - base_plus, material.CA / material.srce - base_minus)
                    )
                    n = material.poles
                    f = material.eqt[:n]
                    b = e0 * material.zt[:n]
                    c = material.eqt2[:n]
                    coefficients = np.column_stack((f.real, f.imag, b.real, b.imag, c.real, c.imag))
                    material_dispersion[material_id] = (delta, coefficients)
                delta, coefficients = material_dispersion[material_id]
                correction += area * delta
                coefficients = coefficients.copy()
                coefficients[:, 2:4] *= area
                pole_coeffs.extend(coefficients)
            edge_dispersion.append(correction)
            pole_offsets.append(len(pole_coeffs))
            edge_fractions.append(len(retained) / 4)
            target[coord] = hold.numID

    _check_supported_configuration(grid, boundary_keys)

    marker_ids = np.asarray(tuple(grid.impedance_marker_models), dtype=np.uint32)
    for component in range(6):
        view = _component_valid_view(grid.ID[component], component, grid)
        if reduced:
            active = np.zeros(view.shape, dtype=bool)
            name = ("E" if component < 3 else "H") + "xyz"[component % 3]
            if name in (*reduced.active_electric, *reduced.active_magnetic):
                selection = [slice(None)] * 3
                selection[reduced.invariant_axis] = reduced.live_index
                active[tuple(selection)] = True
            # Marker IDs on inactive storage components must also disappear;
            # they never acquire a sparse row or a surface-current history.
            view[np.isin(view, marker_ids) & ~active] = void.numID
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
        a_plus += edge_dispersion[edge_index][0]
        a_minus += edge_dispersion[edge_index][1]
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
        edge_dispersion=(
            np.ascontiguousarray(edge_dispersion, dtype=real_dtype) if pole_coeffs else None
        ),
        pole_offsets=(np.asarray(pole_offsets, dtype=np.int32) if pole_coeffs else None),
        pole_coeffs=(np.ascontiguousarray(pole_coeffs, dtype=real_dtype) if pole_coeffs else None),
    )
    grid.impedance_surfaces = system
    if grid.pmls["slabs"]:
        from gprMax.impedance_pml import prepare_impedance_pml

        prepare_impedance_pml(grid, system)
    if pec_contact_count:
        logger.warning(
            "PEC and surface-impedance voxels meet diagonally in edge-only contact "
            f"at {pec_contact_count} Yee edge(s) [{grid.name}]. "
            f"Examples: {'; '.join(pec_contact_examples)}. "
            "The shared E components are forced to zero by PEC. "
            "Verify that these edge-only contacts are intended."
        )
    logger.info(
        f"Compiled {system.edge_count} impedance boundary E edges and "
        f"{system.port_count} surface-current ports [{grid.name}]."
    )
    return system
