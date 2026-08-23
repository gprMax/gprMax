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
    fit_max_relative_error: float | None = None

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

        # The dataclass is frozen, so its array members must be frozen too.
        # Immutable byte backing detaches caller-owned buffers and also blocks
        # writes attempted through NumPy's public ``.base`` chain.
        immutable_arrays = []
        for value in (A, B, C):
            contiguous = np.array(value, dtype=np.float64, order="C", copy=True)
            immutable_arrays.append(
                np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(
                    contiguous.shape
                )
            )
        object.__setattr__(self, "A", immutable_arrays[0])
        object.__setattr__(self, "B", immutable_arrays[1])
        object.__setattr__(self, "C", immutable_arrays[2])
        object.__setattr__(self, "D", D)
        object.__setattr__(self, "fit_fmin_hz", fmin)
        object.__setattr__(self, "fit_fmax_hz", fmax)
        object.__setattr__(self, "fit_max_relative_error", fit_error)

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
            raise ValueError(
                f"{purpose} {frequency_kind} frequency must be finite and positive"
            )
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
            passive_tolerance = 2048 * np.finfo(np.float64).eps * max(
                1.0,
                abs(self.D),
                float(np.max(np.abs(values), initial=0.0)),
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


def create_impedance_marker_material(grid, model_id: str) -> Material:
    """Return the private voxel marker associated with a surface model."""

    existing_numid = next(
        (numid for numid, value in grid.impedance_marker_models.items() if value == model_id),
        None,
    )
    if existing_numid is not None:
        return grid.materials[existing_numid]
    marker = Material(len(grid.materials), f"__impedance_volume__{model_id}")
    marker.type = "impedance-volume-marker"
    marker.averagable = False
    grid.materials.append(marker)
    grid.impedance_marker_models[marker.numID] = model_id
    return marker


def _sentinel_material(grid, role: str) -> _PrivateImpedanceMaterial:
    ID = f"__impedance_{role.replace('-', '_')}__"
    existing = next((material for material in grid.materials if material.ID == ID), None)
    if existing is not None:
        return existing
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

    shapes = ((grid.nx + 1, grid.ny, grid.nz), (grid.nx, grid.ny + 1, grid.nz), (grid.nx, grid.ny, grid.nz + 1))
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
    boundary_coordinates = np.asarray(
        [(i, j, k) for _, i, j, k in boundary_keys], dtype=np.int32
    )
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
        if not (
            np.all(boundary_coordinates > lower)
            and np.all(boundary_coordinates < upper)
        ):
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
    from gprMax.grid.mpi_grid import MPIGrid

    if isinstance(grid, SubGridBaseGrid):
        raise ValueError("impedance volumes are not yet supported in subgrids")
    if isinstance(grid, MPIGrid):
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
    """Packed sparse boundary records and per-port ADE state."""

    def __init__(
        self,
        *,
        edge_info,
        edge_params,
        edge_fraction,
        h_info,
        h_weight,
        port_info,
        port_g,
        port_normal,
        port_area,
        model_info,
        model_F,
        model_G,
        model_L,
        model_Z0,
        state,
        model_ids,
    ):
        self.edge_info = edge_info
        self.edge_params = edge_params
        self.edge_fraction = edge_fraction
        self.h_info = h_info
        self.h_weight = h_weight
        self.port_info = port_info
        self.port_g = port_g
        self.port_normal = port_normal
        self.port_area = port_area
        self.model_info = model_info
        self.model_F = model_F
        self.model_G = model_G
        self.model_L = model_L
        self.model_Z0 = model_Z0
        self.state = state
        self.state_new = np.zeros_like(state)
        self.model_ids = tuple(model_ids)

    @property
    def edge_count(self) -> int:
        return int(self.edge_info.shape[0])

    @property
    def port_count(self) -> int:
        return int(self.port_info.shape[0])

    def reset(self) -> None:
        self.state.fill(0)
        self.state_new.fill(0)

    @staticmethod
    def _field(fields, component, i, j, k):
        return fields[component][i, j, k]

    def _update_python(self, grid) -> None:
        electric = (grid.Ex, grid.Ey, grid.Ez)
        magnetic = (grid.Hx, grid.Hy, grid.Hz)
        for edge_index, edge in enumerate(self.edge_info):
            component, i, j, k, h_start, h_count, port_start, port_count = edge
            e_old = float(electric[component][i, j, k])
            r_h = 0.0
            for h_index in range(h_start, h_start + h_count):
                h_component, hi, hj, hk = self.h_info[h_index]
                r_h += self.h_weight[h_index] * magnetic[h_component][hi, hj, hk]
            a_plus, a_minus = self.edge_params[edge_index]
            denominator = a_plus
            rhs = a_minus * e_old + r_h
            for port_index in range(port_start, port_start + port_count):
                model_index, state_start = self.port_info[port_index]
                state_count, _, vector_start = self.model_info[model_index]
                section = slice(state_start, state_start + state_count)
                vector = slice(vector_start, vector_start + state_count)
                history = float(self.model_L[vector] @ self.state[section])
                g = self.port_g[port_index]
                z0 = self.model_Z0[model_index]
                denominator -= g / (2 * z0)
                rhs += g * e_old / (2 * z0) - g * history / z0
            e_new = rhs / denominator
            electric[component][i, j, k] = e_new
            for port_index in range(port_start, port_start + port_count):
                model_index, state_start = self.port_info[port_index]
                state_count, matrix_start, vector_start = self.model_info[model_index]
                section = slice(state_start, state_start + state_count)
                vector = slice(vector_start, vector_start + state_count)
                history = float(self.model_L[vector] @ self.state[section])
                current = (0.5 * (e_new + e_old) - history) / self.model_Z0[model_index]
                if state_count:
                    matrix = self.model_F[
                        matrix_start : matrix_start + state_count * state_count
                    ].reshape(state_count, state_count)
                    self.state_new[section] = (
                        matrix @ self.state[section] + self.model_G[vector] * current
                    )
        self.state, self.state_new = self.state_new, self.state

    def update(self, grid) -> None:
        if not self.edge_count:
            return
        if _cython_update is None:
            self._update_python(grid)
            return
        _cython_update(
            config.get_model_config().ompthreads,
            self.edge_info,
            self.edge_params,
            self.h_info,
            self.h_weight,
            self.port_info,
            self.port_g,
            self.model_info,
            self.model_F,
            self.model_G,
            self.model_L,
            self.model_Z0,
            self.state,
            self.state_new,
            grid.Ex,
            grid.Ey,
            grid.Ez,
            grid.Hx,
            grid.Hy,
            grid.Hz,
        )
        self.state, self.state_new = self.state_new, self.state


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
        diagonal = (metal[0] & metal[2] & ~metal[1] & ~metal[3]) | (
            metal[1] & metal[3] & ~metal[0] & ~metal[2]
        )
        if np.any(diagonal):
            raise ValueError("impedance-volume voxel topology is non-manifold at a Yee edge")

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
    model_info = np.zeros((len(discrete), 3), dtype=np.int32)
    F_values = []
    G_values = []
    L_values = []
    matrix_offset = 0
    vector_offset = 0
    for index, item in enumerate(discrete):
        order = item.G.size
        model_info[index] = (order, matrix_offset, vector_offset)
        F_values.extend(item.F.reshape(-1))
        G_values.extend(item.G)
        L_values.extend(item.L)
        matrix_offset += order * order
        vector_offset += order

    port_info = np.zeros((len(port_models), 2), dtype=np.int32)
    state_offset = 0
    for index, surface_model in enumerate(port_models):
        order = int(model_info[surface_model, 0])
        port_info[index] = (surface_model, state_offset)
        state_offset += order

    def packed(values):
        array = np.asarray(values, dtype=real_dtype)
        return np.ascontiguousarray(array if array.size else np.zeros(1, dtype=real_dtype))

    system = ImpedanceSurfaceSystem(
        edge_info=np.ascontiguousarray(np.asarray(edge_records, dtype=np.int32).reshape(-1, 8)),
        edge_params=np.ascontiguousarray(np.asarray(edge_params, dtype=real_dtype).reshape(-1, 2)),
        edge_fraction=np.ascontiguousarray(np.asarray(edge_fractions, dtype=real_dtype)),
        h_info=np.ascontiguousarray(np.asarray(h_records, dtype=np.int32).reshape(-1, 4)),
        h_weight=packed(h_weights),
        port_info=np.ascontiguousarray(port_info),
        port_g=packed(port_g),
        port_normal=np.ascontiguousarray(np.asarray(port_normals, dtype=np.int8).reshape(-1, 2)),
        port_area=packed(port_areas),
        model_info=np.ascontiguousarray(model_info),
        model_F=packed(F_values),
        model_G=packed(G_values),
        model_L=packed(L_values),
        model_Z0=packed([item.Z0 for item in discrete]),
        state=np.zeros(max(state_offset, 1), dtype=real_dtype),
        model_ids=model_ids,
    )
    grid.impedance_surfaces = system
    logger.info(
        f"Compiled {system.edge_count} impedance boundary E edges and "
        f"{system.port_count} surface-current ports [{grid.name}]."
    )
    return system
