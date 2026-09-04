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

"""Frequency-domain multi-mode eigenmode port monitors."""

import csv
import logging
from dataclasses import dataclass

import numpy as np

import gprMax.config as config
from gprMax.fdfd_eigenmode_solver.numerical_dispersion import modal_propagation

try:
    from gprMax.cython.eigenmode_dft import accumulate_eigenmode_dft
except ImportError:  # Source-tree fallback before extensions are rebuilt.

    def _accumulate_eigenmode_dft_numpy(
        nthreads,
        normal_axis,
        direction_sign,
        magnetic_side,
        u0,
        v0,
        u1,
        v1,
        plane_index,
        owned_lower,
        owned_upper,
        dt,
        measure,
        handedness,
        electric_phase,
        magnetic_phase,
        phase_step,
        conj_eu,
        conj_ev,
        conj_hu,
        conj_hv,
        electric_dft,
        magnetic_dft,
        Ex,
        Ey,
        Ez,
        Hx,
        Hy,
        Hz,
    ):
        hplane = plane_index if direction_sign * magnetic_side > 0 else plane_index - 1
        plane_owned = owned_lower[normal_axis] <= plane_index < owned_upper[normal_axis]
        if normal_axis == 0:
            sample_u0 = max(u0, owned_lower[1])
            sample_u1 = min(u1, owned_upper[1])
            sample_v0 = max(v0, owned_lower[2])
            sample_v1 = min(v1, owned_upper[2])
        elif normal_axis == 1:
            sample_u0 = max(u0, owned_lower[0])
            sample_u1 = min(u1, owned_upper[0])
            sample_v0 = max(v0, owned_lower[2])
            sample_v1 = min(v1, owned_upper[2])
        else:
            sample_u0 = max(u0, owned_lower[0])
            sample_u1 = min(u1, owned_upper[0])
            sample_v0 = max(v0, owned_lower[1])
            sample_v1 = min(v1, owned_upper[1])

        if not plane_owned:
            sample_u1 = sample_u0
            sample_v1 = sample_v0
        sample_u1 = max(sample_u0, sample_u1)
        sample_v1 = max(sample_v0, sample_v1)

        if normal_axis == 0:
            measured_eu = 0.5 * (
                Ey[plane_index, sample_u0:sample_u1, sample_v0:sample_v1]
                + Ey[plane_index, sample_u0:sample_u1, sample_v0 + 1 : sample_v1 + 1]
            )
            measured_ev = 0.5 * (
                Ez[plane_index, sample_u0:sample_u1, sample_v0:sample_v1]
                + Ez[plane_index, sample_u0 + 1 : sample_u1 + 1, sample_v0:sample_v1]
            )
            measured_hu = 0.5 * (
                Hy[hplane, sample_u0:sample_u1, sample_v0:sample_v1]
                + Hy[hplane, sample_u0 + 1 : sample_u1 + 1, sample_v0:sample_v1]
            )
            measured_hv = 0.5 * (
                Hz[hplane, sample_u0:sample_u1, sample_v0:sample_v1]
                + Hz[hplane, sample_u0:sample_u1, sample_v0 + 1 : sample_v1 + 1]
            )
        elif normal_axis == 1:
            measured_eu = 0.5 * (
                Ex[sample_u0:sample_u1, plane_index, sample_v0:sample_v1]
                + Ex[sample_u0:sample_u1, plane_index, sample_v0 + 1 : sample_v1 + 1]
            )
            measured_ev = 0.5 * (
                Ez[sample_u0:sample_u1, plane_index, sample_v0:sample_v1]
                + Ez[sample_u0 + 1 : sample_u1 + 1, plane_index, sample_v0:sample_v1]
            )
            measured_hu = 0.5 * (
                Hx[sample_u0:sample_u1, hplane, sample_v0:sample_v1]
                + Hx[sample_u0 + 1 : sample_u1 + 1, hplane, sample_v0:sample_v1]
            )
            measured_hv = 0.5 * (
                Hz[sample_u0:sample_u1, hplane, sample_v0:sample_v1]
                + Hz[sample_u0:sample_u1, hplane, sample_v0 + 1 : sample_v1 + 1]
            )
        else:
            measured_eu = 0.5 * (
                Ex[sample_u0:sample_u1, sample_v0:sample_v1, plane_index]
                + Ex[sample_u0:sample_u1, sample_v0 + 1 : sample_v1 + 1, plane_index]
            )
            measured_ev = 0.5 * (
                Ey[sample_u0:sample_u1, sample_v0:sample_v1, plane_index]
                + Ey[sample_u0 + 1 : sample_u1 + 1, sample_v0:sample_v1, plane_index]
            )
            measured_hu = 0.5 * (
                Hx[sample_u0:sample_u1, sample_v0:sample_v1, hplane]
                + Hx[sample_u0 + 1 : sample_u1 + 1, sample_v0:sample_v1, hplane]
            )
            measured_hv = 0.5 * (
                Hy[sample_u0:sample_u1, sample_v0:sample_v1, hplane]
                + Hy[sample_u0:sample_u1, sample_v0 + 1 : sample_v1 + 1, hplane]
            )
        local_u = slice(sample_u0 - u0, sample_u1 - u0)
        local_v = slice(sample_v0 - v0, sample_v1 - v0)
        local_conj_eu = conj_eu[:, :, local_u, local_v]
        local_conj_ev = conj_ev[:, :, local_u, local_v]
        local_conj_hu = conj_hu[:, :, local_u, local_v]
        local_conj_hv = conj_hv[:, :, local_u, local_v]
        factor = 0.5 * handedness * measure * dt
        electric_overlap = factor * (
            np.einsum("uv,fmuv->fm", measured_eu, local_conj_hv, optimize=True)
            - np.einsum("uv,fmuv->fm", measured_ev, local_conj_hu, optimize=True)
        )
        magnetic_overlap = (
            factor
            * direction_sign
            * (
                np.einsum("fmuv,uv->fm", local_conj_eu, measured_hv, optimize=True)
                - np.einsum("fmuv,uv->fm", local_conj_ev, measured_hu, optimize=True)
            )
        )
        electric_dft += electric_phase[:, np.newaxis] * electric_overlap
        magnetic_dft += magnetic_phase[:, np.newaxis] * magnetic_overlap
        electric_phase *= phase_step
        magnetic_phase *= phase_step

    class _FallbackDispatcher:
        def __call__(self, *args):
            return _accumulate_eigenmode_dft_numpy(*args)

        def __getitem__(self, unused):
            return _accumulate_eigenmode_dft_numpy

    accumulate_eigenmode_dft = _FallbackDispatcher()


logger = logging.getLogger(__name__)
INCIDENT_FLOOR_DB = -60.0
CONDITION_RELATIVE_ERROR_BUDGET = 1e-3
MAX_CONDITION_NUMBER = 1e10
DFT_PHASE_REANCHOR_INTERVAL = 1024
NEFF_CUTOFF_TOLERANCE = 1e-12


def _dft_phase_at_time(frequencies, time, dtype):
    """Return exp(-j omega t) with float64 transcendental argument reduction."""

    phase_frequencies = np.asarray(frequencies, dtype=np.float64)
    return np.exp(-2j * np.pi * phase_frequencies * float(time)).astype(dtype)


@dataclass(frozen=True)
class EigenmodePortResult:
    frequency: np.ndarray
    incident: np.ndarray
    outgoing: np.ndarray
    power_wave_valid: np.ndarray
    condition_number: np.ndarray
    coefficient_valid: np.ndarray


def _coefficient_result_valid(result):
    """Return the conditioned modal-coefficient mask."""
    return np.asarray(result.coefficient_valid, dtype=bool)


def _solve_conditioned_gram(
    matrix,
    right_hand_side,
    *,
    component_epsilon,
    condition_limit,
    power_coordinates,
):
    """Solve one modal Gram system and identify unambiguous coordinates.

    A well-conditioned full system uses the ordinary direct solve. If only a
    generalized-mode nullspace is unsafe, a truncated full-system SVD can
    retain coordinates that have no material projection onto that discarded
    subspace. A discarded subspace involving a physical power-wave coordinate
    invalidates the complete solve rather than silently changing the coupled
    modal system.
    """

    matrix = np.asarray(matrix, dtype=np.complex128)
    right_hand_side = np.asarray(right_hand_side, dtype=np.complex128)
    power_coordinates = np.asarray(power_coordinates, dtype=bool)
    size = right_hand_side.size
    failed = (
        np.zeros(size, dtype=np.complex128),
        np.zeros(size, dtype=bool),
        np.inf,
    )
    if (
        matrix.shape != (size, size)
        or power_coordinates.shape != (size,)
        or not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(right_hand_side))
    ):
        return failed

    try:
        left_vectors, singular_values, right_vectors_h = np.linalg.svd(matrix)
    except np.linalg.LinAlgError:
        return failed
    if singular_values.size == 0 or not np.all(np.isfinite(singular_values)):
        return failed

    singular_value_cutoff = max(
        float(component_epsilon) / CONDITION_RELATIVE_ERROR_BUDGET,
        float(singular_values[0]) / float(condition_limit),
    )
    retained = singular_values > singular_value_cutoff
    if np.all(retained):
        try:
            solution = np.linalg.solve(matrix, right_hand_side)
        except np.linalg.LinAlgError:
            return failed
        condition_number = float(singular_values[0] / singular_values[-1])
        return solution, np.ones(size, dtype=bool), condition_number

    if not np.any(retained) or not np.any(power_coordinates) or not np.any(~power_coordinates):
        return failed

    discarded_right_vectors = right_vectors_h[~retained]
    power_nullspace_projection = np.linalg.norm(
        discarded_right_vectors[:, power_coordinates],
        ord="fro",
    )
    if (
        not np.isfinite(power_nullspace_projection)
        or power_nullspace_projection > CONDITION_RELATIVE_ERROR_BUDGET
    ):
        return failed

    retained_left = left_vectors[:, retained]
    retained_right_h = right_vectors_h[retained]
    solution = retained_right_h.conj().T @ (
        (retained_left.conj().T @ right_hand_side) / singular_values[retained]
    )
    ambiguity = np.sqrt(np.sum(np.abs(discarded_right_vectors) ** 2, axis=0))
    stable = np.isfinite(ambiguity) & (ambiguity <= CONDITION_RELATIVE_ERROR_BUDGET)
    retained_singular_values = singular_values[retained]
    condition_number = float(np.max(retained_singular_values) / np.min(retained_singular_values))
    return solution, stable, condition_number


class EigenmodePortMonitor:
    """Collect one plane DFT and decompose it into generalized modal waves."""

    representation = "modal_power_waves"

    def __init__(
        self,
        *,
        owner,
        port_index,
        port_id,
        is_source,
        excitation_mode_index,
        mode_indices,
        anchor_frequencies,
        anchor_e,
        anchor_h,
        anchor_neff,
        dft_start,
        dft_stop,
        dft_points,
        anchor_mode_valid=None,
        anchor_mode_reference_valid=None,
        anchor_mode_propagating=None,
        anchor_balanced_power=None,
        mode_anchor_policies=None,
        dft_frequencies=None,
        anchor_operator_neff=None,
    ):
        self.owner = owner
        self.port_index = int(port_index)
        self.port_id = port_id
        self.is_source = bool(is_source)
        self.excitation_mode_index = (
            None if excitation_mode_index is None else int(excitation_mode_index)
        )
        self.excitation_mode_indices = (
            () if self.excitation_mode_index is None else (self.excitation_mode_index,)
        )
        self.drive_metadata = ()
        self.response_type = "passive"
        # A TF/SF source must sample H on its total-field side. A passive port
        # uses the upstream half-cell, where either side is physically
        # equivalent in the absence of a field discontinuity.
        self.magnetic_side = 1 if self.is_source else -1
        self.mode_indices = tuple(int(value) for value in mode_indices)
        self.anchor_frequencies = np.asarray(anchor_frequencies, dtype=np.float64)
        self.anchor_e = anchor_e
        self.anchor_h = anchor_h
        self.anchor_neff = np.asarray(anchor_neff, dtype=np.complex128)
        self.anchor_operator_neff = (
            None
            if anchor_operator_neff is None
            else np.asarray(anchor_operator_neff, dtype=np.complex128)
        )
        anchor_shape = (self.anchor_frequencies.size, len(self.mode_indices))
        self.anchor_mode_valid = (
            np.ones(anchor_shape, dtype=bool)
            if anchor_mode_valid is None
            else np.asarray(anchor_mode_valid, dtype=bool)
        )
        self.anchor_mode_reference_valid = (
            self.anchor_mode_valid.copy()
            if anchor_mode_reference_valid is None
            else np.asarray(anchor_mode_reference_valid, dtype=bool)
        )
        self._anchor_mode_propagating_explicit = anchor_mode_propagating is not None
        self.anchor_mode_propagating = (
            self.anchor_mode_valid.copy()
            if anchor_mode_propagating is None
            else np.asarray(anchor_mode_propagating, dtype=bool)
        )
        self.anchor_balanced_power = (
            np.ones(anchor_shape, dtype=np.float64)
            if anchor_balanced_power is None
            else np.asarray(anchor_balanced_power, dtype=np.float64)
        )
        self.mode_anchor_policies = (
            tuple("explicit" for _ in self.mode_indices)
            if mode_anchor_policies is None
            else tuple(str(value) for value in mode_anchor_policies)
        )
        self.dft_start = float(dft_start)
        self.dft_stop = float(dft_stop)
        self.dft_points = int(dft_points)
        self.dft_frequencies = (
            None if dft_frequencies is None else np.asarray(dft_frequencies, dtype=np.float64)
        )
        self.result = None
        self.s_parameters = None
        self.s_power_wave_valid = None
        self.s_coefficient_valid = None
        self.active_s_parameters = None
        self.active_s_power_wave_valid = None
        self.active_s_coefficient_valid = None
        self.active_s_driven = None

    def set_drive_metadata(self, drives):
        """Record all active modal drives represented by this physical port."""

        self.drive_metadata = tuple(
            {
                "mode": int(drive.mode_index),
                "waveform_id": str(drive.waveform.ID),
                "amplitude": float(drive.amplitude),
                "power": float(drive.power),
                "phase_deg": float(drive.phase_deg),
                "delay_s": float(drive.delay_s),
            }
            for drive in drives
        )
        self.excitation_mode_indices = tuple(metadata["mode"] for metadata in self.drive_metadata)
        self.excitation_mode_index = (
            self.excitation_mode_indices[0] if len(self.excitation_mode_indices) == 1 else None
        )
        self.is_source = bool(self.drive_metadata)
        self.response_type = "driven" if self.is_source else "passive"

    @property
    def output_id(self):
        """Public port identifier used by antenna-power associations."""

        return self.port_id if self.port_id is not None else f"port{self.port_index}"

    def _validate(self):
        if not self.mode_indices or any(value < 1 for value in self.mode_indices):
            raise ValueError("Eigenmode port mode indices must be one-based positive integers.")
        if len(set(self.mode_indices)) != len(self.mode_indices):
            raise ValueError("Eigenmode port mode indices must be unique.")
        if self.port_index < 1:
            raise ValueError("Eigenmode port indices must be one-based positive integers.")
        if self.is_source and (
            not self.excitation_mode_indices
            or any(mode not in self.mode_indices for mode in self.excitation_mode_indices)
        ):
            raise ValueError(
                "Every source excitation mode must be included in its monitored mode indices."
            )
        if not self.is_source and self.excitation_mode_indices:
            raise ValueError("A passive eigenmode port cannot have excitation modes.")
        if self.dft_points < 1:
            raise ValueError("Eigenmode port DFT points must be at least one.")
        if not np.isfinite(self.dft_start) or not np.isfinite(self.dft_stop):
            raise ValueError("Eigenmode port DFT limits must be finite.")
        if self.dft_start <= 0 or self.dft_stop < self.dft_start:
            raise ValueError("Eigenmode port DFT limits must satisfy 0 < start <= stop.")
        if self.dft_points == 1 and self.dft_stop != self.dft_start:
            raise ValueError("A one-point eigenmode DFT requires equal start and stop frequencies.")
        if self.dft_points > 1 and self.dft_stop == self.dft_start:
            raise ValueError("A multi-point eigenmode DFT requires stop greater than start.")
        if self.dft_frequencies is not None:
            if self.dft_frequencies.ndim != 1 or self.dft_frequencies.size == 0:
                raise ValueError(
                    "Eigenmode port DFT frequencies must be one-dimensional and non-empty."
                )
            if not np.all(np.isfinite(self.dft_frequencies)) or np.any(self.dft_frequencies <= 0):
                raise ValueError("Eigenmode port DFT frequencies must be finite and positive.")
            if np.any(np.diff(self.dft_frequencies) <= 0):
                raise ValueError(
                    "Eigenmode port DFT frequencies must be unique and strictly increasing."
                )
        expected_shape = (self.anchor_frequencies.size, len(self.mode_indices))
        if self.anchor_operator_neff is not None:
            if self.anchor_operator_neff.shape != expected_shape:
                raise ValueError(
                    "Eigenmode port anchor operator-index shape "
                    f"{self.anchor_operator_neff.shape} does not match {expected_shape}."
                )
            if not np.all(np.isfinite(self.anchor_operator_neff)):
                raise ValueError("Eigenmode port anchor operator indices must be finite.")
        if self.anchor_mode_valid.shape != expected_shape:
            raise ValueError(
                "Eigenmode port anchor validity shape "
                f"{self.anchor_mode_valid.shape} does not match {expected_shape}."
            )
        if self.anchor_mode_propagating.shape != expected_shape:
            raise ValueError(
                "Eigenmode port anchor propagation shape "
                f"{self.anchor_mode_propagating.shape} does not match {expected_shape}."
            )
        if self.anchor_mode_reference_valid.shape != expected_shape:
            raise ValueError(
                "Eigenmode port anchor reference-validity shape "
                f"{self.anchor_mode_reference_valid.shape} does not match {expected_shape}."
            )
        if self.anchor_balanced_power.shape != expected_shape:
            raise ValueError(
                "Eigenmode port anchor balanced-power shape "
                f"{self.anchor_balanced_power.shape} does not match {expected_shape}."
            )
        if np.any(self.anchor_mode_valid & ~self.anchor_mode_propagating):
            raise ValueError("A usable eigenmode anchor must also carry forward propagating power.")
        if np.any(self.anchor_mode_valid & ~self.anchor_mode_reference_valid):
            raise ValueError("Every usable eigenmode anchor must be a tracked reference anchor.")
        invalid_reference_scale = self.anchor_mode_reference_valid & (
            ~np.isfinite(self.anchor_balanced_power) | (self.anchor_balanced_power <= 0)
        )
        if np.any(invalid_reference_scale):
            raise ValueError(
                "Every eigenmode reference anchor requires finite positive balanced E/H power."
            )
        if len(self.mode_anchor_policies) != len(self.mode_indices):
            raise ValueError("Eigenmode port requires one anchor policy per monitored mode.")
        if np.any(~np.any(self.anchor_mode_valid, axis=0)):
            raise ValueError("Every eigenmode port mode requires at least one usable anchor.")
        for mode_position, mode_index in enumerate(self.mode_indices):
            runs = self._contiguous_true_runs(self.anchor_mode_valid[:, mode_position])
            if len(runs) > 1:
                raise ValueError(
                    f"Eigenmode port mode {mode_index} has disconnected usable "
                    "anchor ranges; interpolation across an internal anchor "
                    "gap is not valid."
                )

    @staticmethod
    def _contiguous_true_runs(mask):
        indices = np.flatnonzero(np.asarray(mask, dtype=bool))
        if indices.size == 0:
            return ()
        breaks = np.flatnonzero(np.diff(indices) > 1)
        starts = np.concatenate((indices[:1], indices[breaks + 1]))
        stops = np.concatenate((indices[breaks], indices[-1:]))
        return tuple(zip(starts, stops))

    def _propagating_frequency_mask(self, frequencies, mode_position):
        """Return bins eligible for real-power-wave interpretation."""
        frequencies = np.asarray(frequencies, dtype=np.float64)
        propagating = self.anchor_mode_propagating[:, mode_position]
        if np.all(propagating):
            # Constant-basis and guard-trimmed policies intentionally retain
            # endpoint extrapolation when every candidate anchor propagates.
            return np.ones(frequencies.shape, dtype=bool)

        usable = self.anchor_mode_valid[:, mode_position]
        anchors = self.anchor_frequencies
        tolerance = 1e-12 * max(
            float(np.max(np.abs(anchors), initial=0.0)),
            1.0,
        )
        supported = np.zeros(frequencies.shape, dtype=bool)
        intervals = []
        for start, stop in self._contiguous_true_runs(propagating):
            run_usable = np.flatnonzero(usable[start : stop + 1]) + start
            if run_usable.size == 0:
                continue
            # Propagation defines the physical support; usability only
            # selects the basis used within that support. In particular, a
            # centre-only fallback may extrapolate through the remainder of
            # the same propagating run.
            low = float(anchors[start])
            high = float(anchors[stop])
            intervals.append((start, stop, low, high))
            supported |= (frequencies >= low - tolerance) & (frequencies <= high + tolerance)

        if not intervals:
            return supported

        if not self._anchor_mode_propagating_explicit:
            # Without an explicit propagation classification, an exterior
            # validity bit may represent spectral-guard trimming rather than
            # cutoff. Extrapolate endpoints but reject gaps between runs.
            supported |= frequencies <= intervals[0][2] + tolerance
            supported |= frequencies >= intervals[-1][3] - tolerance
        else:
            # With an explicit propagation classification, extrapolate only
            # through an outer edge that is itself propagating. For example,
            # a low-frequency cutoff must not suppress the high-frequency
            # propagating endpoint beyond the final solved anchor.
            if intervals[0][0] == 0:
                supported |= frequencies <= intervals[0][2] + tolerance
            if intervals[-1][1] == propagating.size - 1:
                supported |= frequencies >= intervals[-1][3] - tolerance
        return supported

    def _nondegenerate_reference_mask(self, frequencies, mode_position):
        """Reject a DFT bin only when a solved anchor identifies exact cutoff.

        A tracked evanescent reference basis can provide generalized modal
        coefficients below cutoff, but sparse anchors cannot locate cutoff
        between solves. Only a matching raw anchor whose solved effective
        index is zero to the mode solver's branch tolerance is treated as the
        forward/backward degeneracy at exact cutoff.
        """

        frequencies = np.asarray(frequencies, dtype=np.float64)
        supported = np.ones(frequencies.shape, dtype=bool)
        anchors = self.anchor_frequencies
        neff = self.anchor_neff[:, mode_position]
        finite = np.isfinite(np.real(neff)) & np.isfinite(np.imag(neff))
        cutoff = (
            self.anchor_mode_reference_valid[:, mode_position]
            & ~self.anchor_mode_propagating[:, mode_position]
            & finite
            & (np.abs(neff) <= NEFF_CUTOFF_TOLERANCE)
        )
        for anchor_index in np.flatnonzero(cutoff):
            anchor_frequency = float(anchors[anchor_index])
            tolerance = 1e-12 * max(abs(anchor_frequency), 1.0)
            supported &= np.abs(frequencies - anchor_frequency) > tolerance
        return supported

    def prepare(self, grid):
        self._validate()
        if self.port_id is None:
            self.port_id = f"port{self.port_index}"
        if any(port.port_index == self.port_index for port in grid.eigenmodeports):
            raise ValueError(f"Eigenmode port index {self.port_index} is already in use.")
        if any(port.port_id == self.port_id for port in grid.eigenmodeports):
            raise ValueError(f"Eigenmode port ID {self.port_id!r} is already in use.")

        real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
        complex_dtype = np.dtype(config.sim_config.dtypes["complex"])
        nominal_frequency = (
            np.linspace(
                self.dft_start,
                self.dft_stop,
                self.dft_points,
                dtype=np.float64,
            )
            if self.dft_frequencies is None
            else self.dft_frequencies
        )
        self.frequency = np.asarray(nominal_frequency, dtype=real_dtype)
        if np.any(np.diff(self.frequency) <= 0):
            raise ValueError(
                f"Eigenmode port {self.port_id!r} DFT frequencies are not distinct "
                "at the configured simulation precision."
            )
        if self.frequency[-1] > 0.5 / grid.dt:
            raise ValueError(
                f"Eigenmode port {self.port_id!r} DFT stop frequency exceeds the FDTD Nyquist frequency."
            )
        if self.anchor_frequencies.size > 1 and (
            nominal_frequency[0] < self.anchor_frequencies[0]
            or nominal_frequency[-1] > self.anchor_frequencies[-1]
        ):
            logger.warning(
                f"Eigenmode port {self.port_id!r} DFT range extends outside its modal "
                "anchor range; endpoint modal profiles will be used."
            )

        nf = self.frequency.size
        nm = len(self.mode_indices)
        # Validity masks share the (mode, frequency) axes of modal coefficients.
        self.power_basis_valid = np.stack(
            tuple(
                self._propagating_frequency_mask(
                    nominal_frequency,
                    mode_position,
                )
                for mode_position in range(nm)
            )
        )
        # Bank selection remains fixed if the interpolated Yee symbol later
        # enters a spatial stop band. Such bins keep that tracked E/H branch
        # but use balanced generalized coordinates instead of power waves.
        uses_power_bank = self.power_basis_valid.copy()
        operator_interpolation = np.zeros((nm, nf), dtype=bool)
        power_weights = np.zeros(
            (nm, self.anchor_frequencies.size, nf),
            dtype=np.float64,
        )
        reference_weights = np.zeros_like(power_weights)
        reference_anchor_scale = np.zeros_like(
            self.anchor_balanced_power,
            dtype=np.float64,
        )
        for mode_position in range(nm):
            usable_anchors = np.flatnonzero(self.anchor_mode_valid[:, mode_position])
            operator_interpolation[mode_position, uses_power_bank[mode_position]] = (
                usable_anchors.size > 1
            )
            power_weights[mode_position, usable_anchors] = self.owner._linear_anchor_weights(
                self.frequency.astype(np.float64),
                self.anchor_frequencies[usable_anchors],
            )
            reference_anchors = np.flatnonzero(self.anchor_mode_reference_valid[:, mode_position])
            evanescent_reference_mask = (
                self.anchor_mode_reference_valid[:, mode_position]
                & ~self.anchor_mode_propagating[:, mode_position]
            )
            evanescent_runs = self._contiguous_true_runs(evanescent_reference_mask)
            generalized_bins = np.flatnonzero(~self.power_basis_valid[mode_position])
            generalized_frequencies = nominal_frequency[generalized_bins]
            # Beyond the solved candidate span there is no in-band
            # propagation classification to prefer an evanescent run. Retain
            # the nearest endpoint of the complete tracked reference bank.
            below_candidate_range = generalized_frequencies < self.anchor_frequencies[0]
            above_candidate_range = generalized_frequencies > self.anchor_frequencies[-1]
            exterior_bins = generalized_bins[below_candidate_range | above_candidate_range]
            operator_interpolation[mode_position, exterior_bins] = reference_anchors.size > 1
            reference_weights[
                mode_position,
                reference_anchors[0],
                generalized_bins[below_candidate_range],
            ] = 1.0
            reference_weights[
                mode_position,
                reference_anchors[-1],
                generalized_bins[above_candidate_range],
            ] = 1.0
            within_candidate_bins = generalized_bins[
                ~(below_candidate_range | above_candidate_range)
            ]
            if evanescent_runs and within_candidate_bins.size:
                run_distances = np.empty(
                    (len(evanescent_runs), within_candidate_bins.size),
                    dtype=np.float64,
                )
                within_candidate_frequencies = nominal_frequency[within_candidate_bins]
                for run_position, (start, stop) in enumerate(evanescent_runs):
                    low = self.anchor_frequencies[start]
                    high = self.anchor_frequencies[stop]
                    run_distances[run_position] = np.maximum(
                        np.maximum(low - within_candidate_frequencies, 0.0),
                        within_candidate_frequencies - high,
                    )
                selected_runs = np.argmin(run_distances, axis=0)
                for run_position, (start, stop) in enumerate(evanescent_runs):
                    bins = within_candidate_bins[selected_runs == run_position]
                    if bins.size == 0:
                        continue
                    run_anchors = np.arange(start, stop + 1)
                    operator_interpolation[mode_position, bins] = run_anchors.size > 1
                    reference_weights[mode_position][
                        np.ix_(run_anchors, bins)
                    ] = self.owner._linear_anchor_weights(
                        nominal_frequency[bins],
                        self.anchor_frequencies[run_anchors],
                    )
            elif within_candidate_bins.size:
                # A bank containing only propagating references supplies
                # its endpoint generalized basis.
                reference_weights[mode_position][
                    np.ix_(reference_anchors, within_candidate_bins)
                ] = self.owner._linear_anchor_weights(
                    nominal_frequency[within_candidate_bins],
                    self.anchor_frequencies[reference_anchors],
                )
                operator_interpolation[mode_position, within_candidate_bins] = (
                    reference_anchors.size > 1
                )
            reference_anchor_scale[reference_anchors, mode_position] = 1.0 / np.sqrt(
                self.anchor_balanced_power[reference_anchors, mode_position]
            )
        nu, nv = self.owner._transverse_cell_shape()
        shape = (nf, nm, nu, nv)
        self.eu = np.empty(shape, dtype=complex_dtype)
        self.ev = np.empty(shape, dtype=complex_dtype)
        self.hu = np.empty(shape, dtype=complex_dtype)
        self.hv = np.empty(shape, dtype=complex_dtype)
        self.neff = np.empty((nf, nm), dtype=complex_dtype)
        self.beta = np.empty((nf, nm), dtype=complex_dtype)
        self.reference_basis_valid = np.stack(
            tuple(
                self._nondegenerate_reference_mask(
                    nominal_frequency,
                    mode_position,
                )
                for mode_position in range(nm)
            )
        )
        u_axis, v_axis = self.owner.transverse_axes
        measure = (
            grid.dl[self.owner.physical_transverse_axis]
            if self.owner.invariant_axis is not None
            else grid.dl[u_axis] * grid.dl[v_axis]
        )
        if self.owner.invariant_axis is not None and self.owner.domain_polarization == "TE":
            # Both synthetic invariant cells receive half of each live-layer
            # TE field during cell averaging, halving their summed overlap.
            measure *= 2.0
        em_consts = getattr(config.sim_config, "em_consts", config.SimulationConfig.em_consts)
        impedance = float(em_consts["z0"])
        speed = float(em_consts.get("c", config.SimulationConfig.em_consts["c"]))

        for frequency_index in range(nf):
            for mode_position in range(nm):
                from_power_bank = bool(uses_power_bank[mode_position, frequency_index])
                mode_weights = (
                    power_weights[mode_position]
                    if from_power_bank
                    else reference_weights[mode_position]
                )
                weights = mode_weights[:, frequency_index]
                phase_neff = np.sum(weights * self.anchor_neff[:, mode_position])
                operator_neff = None
                if (
                    self.anchor_operator_neff is not None
                    and operator_interpolation[mode_position, frequency_index]
                ):
                    operator_neff = np.sum(
                        weights * self.anchor_operator_neff[:, mode_position]
                    )
                beta, resolved = modal_propagation(
                    float(self.frequency[frequency_index]),
                    phase_neff,
                    operator_neff=operator_neff,
                    fdtd_dt=grid.dt,
                    propagation_spacing=grid.dl[self.owner.normal_axis],
                    c=speed,
                )
                self.beta[frequency_index, mode_position] = beta
                self.neff[frequency_index, mode_position] = (
                    beta * speed / (2 * np.pi * float(self.frequency[frequency_index]))
                )
                self.power_basis_valid[mode_position, frequency_index] &= bool(resolved)
                uses_power_basis = bool(self.power_basis_valid[mode_position, frequency_index])
                anchor_scale = (
                    np.ones(self.anchor_frequencies.size, dtype=np.float64)
                    if from_power_bank
                    else reference_anchor_scale[:, mode_position]
                )
                electric = []
                magnetic = []
                for component in range(3):
                    electric.append(
                        sum(
                            mode_weights[anchor, frequency_index]
                            * anchor_scale[anchor]
                            * self.anchor_e[anchor][mode_position][component]
                            for anchor in range(self.anchor_frequencies.size)
                        )
                    )
                    magnetic.append(
                        sum(
                            mode_weights[anchor, frequency_index]
                            * anchor_scale[anchor]
                            * self.anchor_h[anchor][mode_position][component]
                            for anchor in range(self.anchor_frequencies.size)
                        )
                    )
                if uses_power_basis:
                    power = float(np.real(self.owner._modal_cross_power(electric, magnetic, grid)))
                    if not np.isfinite(power) or power <= 1e-12:
                        raise ValueError(
                            f"Eigenmode port {self.port_id!r} mode "
                            f"{self.mode_indices[mode_position]} has invalid interpolated "
                            f"power {power:g} at {self.frequency[frequency_index]:g} Hz. "
                            "Add a nearby propagating anchor, narrow the band, or inspect "
                            "the modal basis."
                        )
                    scale = 1.0 / np.sqrt(power)
                    electric = [field * scale for field in electric]
                    magnetic = [field * scale for field in magnetic]
                    eu = self.owner._average_to_transverse_cells(electric[u_axis], "eu")
                    ev = self.owner._average_to_transverse_cells(electric[v_axis], "ev")
                    hu = self.owner._average_to_transverse_cells(magnetic[u_axis], "hu")
                    hv = self.owner._average_to_transverse_cells(magnetic[v_axis], "hv")
                else:
                    eu = self.owner._average_to_transverse_cells(electric[u_axis], "eu")
                    ev = self.owner._average_to_transverse_cells(electric[v_axis], "ev")
                    hu = self.owner._average_to_transverse_cells(magnetic[u_axis], "hu")
                    hv = self.owner._average_to_transverse_cells(magnetic[v_axis], "hv")
                    balanced_power = float(
                        measure
                        * np.sum(
                            np.abs(eu) ** 2
                            + np.abs(ev) ** 2
                            + impedance**2 * (np.abs(hu) ** 2 + np.abs(hv) ** 2)
                        )
                        / (4.0 * impedance)
                    )
                    if not np.isfinite(balanced_power) or balanced_power <= 1e-300:
                        raise ValueError(
                            f"Eigenmode port {self.port_id!r} mode "
                            f"{self.mode_indices[mode_position]} has invalid interpolated "
                            f"balanced E/H power {balanced_power:g} at "
                            f"{self.frequency[frequency_index]:g} Hz. Add a nearby tracked "
                            "reference anchor, narrow the band, or inspect the modal basis."
                        )
                    scale = 1.0 / np.sqrt(balanced_power)
                    eu = eu * scale
                    ev = ev * scale
                    hu = hu * scale
                    hv = hv * scale
                self.eu[frequency_index, mode_position] = eu
                self.ev[frequency_index, mode_position] = ev
                self.hu[frequency_index, mode_position] = hu
                self.hv[frequency_index, mode_position] = hv

        self.eu = np.ascontiguousarray(self.eu)
        self.ev = np.ascontiguousarray(self.ev)
        self.hu = np.ascontiguousarray(self.hu)
        self.hv = np.ascontiguousarray(self.hv)
        self.conj_eu = np.ascontiguousarray(np.conj(self.eu))
        self.conj_ev = np.ascontiguousarray(np.conj(self.ev))
        self.conj_hu = np.ascontiguousarray(np.conj(self.hu))
        self.conj_hv = np.ascontiguousarray(np.conj(self.hv))

        handedness = self.owner._modal_basis_handedness()
        factor = 0.5 * handedness * measure
        self.electric_gram = np.empty((nf, nm, nm), dtype=complex_dtype)
        self.magnetic_gram = np.empty((nf, nm, nm), dtype=complex_dtype)
        for frequency_index in range(nf):
            for row in range(nm):
                for column in range(nm):
                    self.electric_gram[frequency_index, row, column] = factor * np.sum(
                        self.eu[frequency_index, column] * self.conj_hv[frequency_index, row]
                        - self.ev[frequency_index, column] * self.conj_hu[frequency_index, row]
                    )
                    self.magnetic_gram[frequency_index, row, column] = factor * np.sum(
                        self.conj_eu[frequency_index, row] * self.hv[frequency_index, column]
                        - self.conj_ev[frequency_index, row] * self.hu[frequency_index, column]
                    )

        # G_H = G_E^H in exact arithmetic, so their Hermitian average is the
        # discrete forward-wave power matrix W used by P = c^H W c. The
        # coefficients remain generalized modal travelling-wave coordinates:
        # an individual coefficient squared is not an additive power when W
        # is non-diagonal.
        self.power_matrix = np.ascontiguousarray(
            0.5 * (self.electric_gram + self.magnetic_gram),
            dtype=complex_dtype,
        )
        self.power_matrix = np.ascontiguousarray(
            0.5 * (self.power_matrix + np.swapaxes(np.conj(self.power_matrix), 1, 2)),
            dtype=complex_dtype,
        )
        self.power_matrix_valid = np.zeros(nf, dtype=bool)
        for frequency_index, matrix in enumerate(self.power_matrix):
            active_modes = np.flatnonzero(self.power_basis_valid[:, frequency_index])
            if active_modes.size == 0:
                continue
            active_matrix = np.asarray(
                matrix[np.ix_(active_modes, active_modes)],
                dtype=np.complex128,
            )
            if not np.all(np.isfinite(active_matrix)):
                continue
            eigenvalues = np.linalg.eigvalsh(active_matrix)
            scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
            tolerance = 64 * np.finfo(real_dtype).eps * scale
            self.power_matrix_valid[frequency_index] = not np.any(eigenvalues < -tolerance)

        self.measure = real_dtype.type(measure)
        self.handedness = int(handedness)
        self.electric_dft = np.zeros((nf, nm), dtype=complex_dtype)
        self.magnetic_dft = np.zeros((nf, nm), dtype=complex_dtype)
        self.phase_step = np.ascontiguousarray(
            _dft_phase_at_time(self.frequency, grid.dt, complex_dtype)
        )
        self.electric_phase = np.ascontiguousarray(
            _dft_phase_at_time(self.frequency, 0.0, complex_dtype)
        )
        self.magnetic_phase = np.ascontiguousarray(
            _dft_phase_at_time(self.frequency, 0.5 * grid.dt, complex_dtype)
        )
        self._next_iteration = 0

    def observe(self, grid, iteration):
        if iteration != self._next_iteration:
            raise ValueError(
                f"expected eigenmode DFT iteration {self._next_iteration}, " f"received {iteration}"
            )
        real_signature = config.sim_config.dtypes["C_float_or_double"]
        owned_lower = getattr(
            self.owner,
            "tfsf_owned_lower",
            np.zeros(3, dtype=np.int32),
        )
        owned_upper = getattr(
            self.owner,
            "tfsf_owned_upper",
            np.asarray(grid.Ex.shape, dtype=np.int32),
        )
        accumulate_eigenmode_dft[f"{real_signature}|{real_signature} complex"](
            config.get_model_config().ompthreads,
            self.owner.normal_axis,
            1 if self.owner.direction == "+" else -1,
            self.magnetic_side,
            self.owner.transverse_start[0],
            self.owner.transverse_start[1],
            self.owner.transverse_stop[0],
            self.owner.transverse_stop[1],
            self.owner.plane_index,
            owned_lower,
            owned_upper,
            grid.dt,
            self.measure,
            self.handedness,
            self.electric_phase,
            self.magnetic_phase,
            self.phase_step,
            self.conj_eu,
            self.conj_ev,
            self.conj_hu,
            self.conj_hv,
            self.electric_dft,
            self.magnetic_dft,
            grid.Ex,
            grid.Ey,
            grid.Ez,
            grid.Hx,
            grid.Hy,
            grid.Hz,
        )
        self._next_iteration += 1
        if self._next_iteration % DFT_PHASE_REANCHOR_INTERVAL == 0:
            self.electric_phase[:] = _dft_phase_at_time(
                self.frequency,
                self._next_iteration * grid.dt,
                self.electric_phase.dtype,
            )
            self.magnetic_phase[:] = _dft_phase_at_time(
                self.frequency,
                (self._next_iteration + 0.5) * grid.dt,
                self.magnetic_phase.dtype,
            )

    def reset_run_state(self, grid):
        """Clear DFT, recursive phase, and derived state for a reused run."""

        self.electric_dft.fill(0)
        self.magnetic_dft.fill(0)
        self.electric_phase[:] = _dft_phase_at_time(self.frequency, 0.0, self.electric_phase.dtype)
        self.magnetic_phase[:] = _dft_phase_at_time(
            self.frequency, 0.5 * grid.dt, self.magnetic_phase.dtype
        )
        self._next_iteration = 0
        self.result = None
        self.s_parameters = None
        self.s_power_wave_valid = None
        self.s_coefficient_valid = None
        self.active_s_parameters = None
        self.active_s_power_wave_valid = None
        self.active_s_coefficient_valid = None
        self.active_s_driven = None
        self.response_type = "driven" if self.is_source else "passive"

    def finalise(self, grid):
        nf, nm = self.electric_dft.shape
        complex_dtype = np.dtype(config.sim_config.dtypes["complex"])
        component_dtype = np.float32 if complex_dtype == np.dtype(np.complex64) else np.float64
        component_epsilon = np.finfo(component_dtype).eps
        condition_limit = min(
            MAX_CONDITION_NUMBER,
            CONDITION_RELATIVE_ERROR_BUDGET / component_epsilon,
        )
        # Generalized coefficients can remain finite below cutoff when the
        # tracked evanescent profile supplies a conditioned reference basis.
        # Their separate real-power-wave mask remains false.
        incident = np.zeros((nm, nf), dtype=complex_dtype)
        outgoing = np.zeros_like(incident)
        coefficient_valid = np.zeros((nm, nf), dtype=bool)
        condition = np.full(nf, np.inf, dtype=np.float64)
        reference_basis_valid = np.asarray(self.reference_basis_valid, dtype=bool)
        power_wave_valid = np.asarray(self.power_basis_valid, dtype=bool)
        magnetic_offset = self.magnetic_side * 0.5 * grid.dl[self.owner.normal_axis]
        beta = getattr(self, "beta", None)
        if beta is None:
            # Compatibility for downstream monitors assembled without prepare().
            em_consts = getattr(config.sim_config, "em_consts", config.SimulationConfig.em_consts)
            speed = float(em_consts.get("c", config.SimulationConfig.em_consts["c"]))
            beta = 2 * np.pi * self.frequency[:, np.newaxis] * self.neff / speed
        forward_phase = np.exp(-1j * beta * magnetic_offset)
        backward_phase = np.exp(1j * beta * magnetic_offset)

        for frequency_index in range(nf):
            active_modes = np.flatnonzero(reference_basis_valid[:, frequency_index])
            if active_modes.size == 0:
                continue
            local_power_coordinates = power_wave_valid[active_modes, frequency_index]
            electric_coeff, electric_stable, electric_condition = _solve_conditioned_gram(
                self.electric_gram[frequency_index][np.ix_(active_modes, active_modes)],
                self.electric_dft[frequency_index, active_modes],
                component_epsilon=component_epsilon,
                condition_limit=condition_limit,
                power_coordinates=local_power_coordinates,
            )
            magnetic_coeff, magnetic_stable, magnetic_condition = _solve_conditioned_gram(
                self.magnetic_gram[frequency_index][np.ix_(active_modes, active_modes)],
                self.magnetic_dft[frequency_index, active_modes],
                component_epsilon=component_epsilon,
                condition_limit=condition_limit,
                power_coordinates=local_power_coordinates,
            )
            stable = electric_stable & magnetic_stable
            denominator = (
                forward_phase[frequency_index, active_modes]
                + backward_phase[frequency_index, active_modes]
            )
            usable = (
                stable
                & np.isfinite(denominator)
                & (np.abs(denominator) > 1e-12)
                & np.isfinite(electric_coeff)
                & np.isfinite(magnetic_coeff)
            )
            a = np.zeros(active_modes.size, dtype=np.complex128)
            a[usable] = (
                magnetic_coeff[usable]
                + backward_phase[frequency_index, active_modes][usable] * electric_coeff[usable]
            ) / denominator[usable]
            b = electric_coeff - a
            local_coefficient_valid = usable & np.isfinite(a) & np.isfinite(b)
            valid_modes = active_modes[local_coefficient_valid]
            incident[valid_modes, frequency_index] = a[local_coefficient_valid]
            outgoing[valid_modes, frequency_index] = b[local_coefficient_valid]
            coefficient_valid[valid_modes, frequency_index] = True
            if np.any(local_coefficient_valid):
                condition[frequency_index] = max(
                    electric_condition,
                    magnetic_condition,
                )

        valid = (
            coefficient_valid
            & power_wave_valid
            & np.asarray(self.power_matrix_valid, dtype=bool)[np.newaxis, :]
        )

        self.result = EigenmodePortResult(
            frequency=self.frequency.copy(),
            incident=incident,
            outgoing=outgoing,
            power_wave_valid=valid,
            condition_number=condition,
            coefficient_valid=coefficient_valid,
        )
        return self.result

    def write_hdf5(self, base_group):
        group = base_group.create_group(f"eigenmode_ports/port{self.port_index}")
        group.attrs["ID"] = self.port_id
        group.attrs["IsSource"] = self.is_source
        if self.excitation_mode_index is not None:
            group.attrs["ExcitationMode"] = self.excitation_mode_index
        excitation_modes = getattr(
            self,
            "excitation_mode_indices",
            () if self.excitation_mode_index is None else (self.excitation_mode_index,),
        )
        drive_metadata = getattr(self, "drive_metadata", ())
        if excitation_modes:
            group.attrs["ExcitationModes"] = excitation_modes
        if drive_metadata:
            group.attrs["DriveAmplitudes"] = tuple(drive["amplitude"] for drive in drive_metadata)
            group.attrs["DrivePowers"] = tuple(drive["power"] for drive in drive_metadata)
            group.attrs["DrivePhasesDegrees"] = tuple(
                drive["phase_deg"] for drive in drive_metadata
            )
            group.attrs["DriveDelays"] = tuple(drive["delay_s"] for drive in drive_metadata)
            group.attrs["DriveWaveformIDs"] = tuple(
                drive["waveform_id"] for drive in drive_metadata
            )
        group.attrs["ResponseType"] = getattr(
            self,
            "response_type",
            "s_parameter_column"
            if self.s_parameters is not None
            else ("driven" if self.is_source else "passive"),
        )
        group.attrs["Direction"] = self.owner.direction
        group.attrs["Normal"] = self.owner.normal
        group.attrs["ModeIndices"] = self.mode_indices
        global_plane_index = getattr(self.owner, "global_plane_index", None)
        group.attrs["PlaneIndex"] = (
            self.owner.plane_index if global_plane_index is None else global_plane_index
        )
        group.attrs["PhaseReanchorInterval"] = DFT_PHASE_REANCHOR_INTERVAL
        group.attrs["RequestedAnchorPolicy"] = self.owner.requested_anchor_policy
        group.attrs["ResolvedAnchorPolicy"] = self.owner.resolved_anchor_policy
        resolved_anchor_union = self.anchor_frequencies[np.any(self.anchor_mode_valid, axis=1)]
        reference_anchor_union = self.anchor_frequencies[
            np.any(self.anchor_mode_reference_valid, axis=1)
        ]
        group.attrs["AnchorFrequencies"] = resolved_anchor_union
        group.attrs["ReferenceAnchorFrequencies"] = reference_anchor_union
        group.attrs["CandidateAnchorFrequencies"] = self.anchor_frequencies
        group.attrs["ModeAnchorPolicies"] = self.mode_anchor_policies
        group["frequency"] = self.result.frequency
        group["incident"] = self.result.incident
        group["outgoing"] = self.result.outgoing
        coefficient_valid = _coefficient_result_valid(self.result).astype(np.uint8)
        power_wave_valid = self.result.power_wave_valid.astype(np.uint8)
        group["coefficient_valid"] = coefficient_valid
        group["power_wave_valid"] = power_wave_valid
        group["condition_number"] = self.result.condition_number
        group["electric_cross_power_matrix"] = self.electric_gram
        group["power_matrix"] = self.power_matrix
        group["reference_basis_valid"] = self.reference_basis_valid.astype(np.uint8)
        group["power_basis_valid"] = self.power_basis_valid.astype(np.uint8)
        group["anchor_mode_valid"] = self.anchor_mode_valid.astype(np.uint8)
        group["anchor_mode_reference_valid"] = self.anchor_mode_reference_valid.astype(np.uint8)
        group["anchor_mode_propagating"] = self.anchor_mode_propagating.astype(np.uint8)
        group["anchor_balanced_power"] = self.anchor_balanced_power
        # Persist the propagation constants that define both broadband modal
        # interpolation and forward/backward de-embedding.  This makes an
        # FDTD launch reproducible and lets validation distinguish an FDFD-to-
        # FDTD mismatch from the cross-section discretisation error relative
        # to a continuum guide formula.
        group["anchor_complex_neff"] = self.anchor_neff
        if getattr(self, "anchor_operator_neff", None) is not None:
            group["anchor_operator_neff"] = self.anchor_operator_neff
        if getattr(self, "beta", None) is not None:
            group["beta"] = self.beta
            group["beta"].attrs["Units"] = "rad/m"
        group["power_matrix_valid"] = self.power_matrix_valid.astype(np.uint8)
        if self.s_parameters is not None:
            group["S"] = self.s_parameters
            group["power_wave_valid_S"] = self.s_power_wave_valid.astype(np.uint8)
            group["coefficient_valid_S"] = self.s_coefficient_valid.astype(np.uint8)
        if getattr(self, "active_s_parameters", None) is not None:
            group["active_S"] = self.active_s_parameters
            group["active_S_driven"] = self.active_s_driven.astype(np.uint8)
            group["coefficient_valid_active_S"] = self.active_s_coefficient_valid.astype(np.uint8)
            group["power_wave_valid_active_S"] = self.active_s_power_wave_valid.astype(np.uint8)


def _incident_ratio_valid(denominator, coefficient_valid, power_wave_mask):
    """Apply the incident-spectrum floor independently to both modal bases."""

    denominator = np.asarray(denominator)
    coefficient_valid = np.asarray(coefficient_valid, dtype=bool)
    power_wave_mask = np.asarray(power_wave_mask, dtype=bool)
    ratio_valid = np.zeros(denominator.shape, dtype=bool)
    for normalization_class in (power_wave_mask, ~power_wave_mask):
        candidates = coefficient_valid & normalization_class & np.isfinite(denominator)
        peak = float(np.max(np.abs(denominator[candidates]), initial=0.0))
        if peak > 0:
            ratio_valid |= candidates & (
                np.abs(denominator) >= peak * 10 ** (INCIDENT_FLOOR_DB / 20)
            )
    return ratio_valid


def _complex_csv_fields(value):
    """Return portable rectangular and polar fields for one complex coefficient."""

    magnitude = abs(value)
    if np.isfinite(magnitude) and magnitude > 0:
        magnitude_db = float(20 * np.log10(magnitude))
    elif magnitude == 0:
        magnitude_db = -np.inf
    else:
        magnitude_db = np.nan
    return (
        float(np.real(value)),
        float(np.imag(value)),
        float(magnitude),
        magnitude_db,
        float(np.angle(value, deg=True)),
        float(magnitude**2),
    )


def _write_active_sparameters(grid, sources, excitation_modes):
    """Form state-dependent active reflection coefficients for driven channels."""

    reference_frequency = np.asarray(grid.eigenmodeports[0].result.frequency)
    for port in grid.eigenmodeports:
        if not np.array_equal(port.result.frequency, reference_frequency):
            raise ValueError("All eigenmode ports must use identical DFT frequency bins.")
        port.active_s_parameters = np.full_like(
            port.result.outgoing,
            np.nan + 1j * np.nan,
        )
        port.active_s_coefficient_valid = np.zeros(port.result.outgoing.shape, dtype=bool)
        port.active_s_power_wave_valid = np.zeros(port.result.outgoing.shape, dtype=bool)
        port.active_s_driven = np.zeros(port.result.outgoing.shape, dtype=bool)

    rows = []
    for port in sources:
        port_power_wave_valid = np.asarray(port.power_basis_valid, dtype=bool)
        result_coefficient_valid = _coefficient_result_valid(port.result)
        drive_by_mode = {
            int(metadata["mode"]): metadata for metadata in getattr(port, "drive_metadata", ())
        }
        for mode_index in excitation_modes(port):
            mode_position = port.mode_indices.index(mode_index)
            port.active_s_driven[mode_position] = True
            denominator = port.result.incident[mode_position]
            power_wave_mask = port_power_wave_valid[mode_position]
            ratio_valid = _incident_ratio_valid(
                denominator,
                result_coefficient_valid[mode_position],
                power_wave_mask,
            )
            np.divide(
                port.result.outgoing[mode_position],
                denominator,
                out=port.active_s_parameters[mode_position],
                where=ratio_valid,
            )
            coefficient_valid = result_coefficient_valid[mode_position] & ratio_valid
            power_valid = (
                coefficient_valid
                & np.asarray(port.result.power_wave_valid[mode_position], dtype=bool)
                & power_wave_mask
                & np.asarray(port.power_matrix_valid, dtype=bool)
            )
            port.active_s_coefficient_valid[mode_position] = coefficient_valid
            port.active_s_power_wave_valid[mode_position] = power_valid
            port.active_s_parameters[mode_position, ~coefficient_valid] = np.nan + 1j * np.nan
            metadata = drive_by_mode.get(
                int(mode_index),
                {
                    "amplitude": np.nan,
                    "power": np.nan,
                    "phase_deg": np.nan,
                    "delay_s": np.nan,
                },
            )
            for frequency_index, frequency in enumerate(port.result.frequency):
                value = port.active_s_parameters[mode_position, frequency_index]
                rows.append(
                    (
                        float(frequency),
                        port.port_index,
                        mode_index,
                        *_complex_csv_fields(value),
                        int(coefficient_valid[frequency_index]),
                        int(power_valid[frequency_index]),
                        float(metadata["amplitude"]),
                        float(metadata["power"]),
                        float(metadata["phase_deg"]),
                        float(metadata["delay_s"]),
                    )
                )

    suffix = "" if grid.name == "main_grid" else f"_{grid.name}"
    output_path = config.get_model_config().output_file_path.with_name(
        config.get_model_config().output_file_path.name + f"{suffix}_active_sparameters.csv"
    )
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "frequency_hz",
                "port",
                "mode",
                "active_S_real",
                "active_S_imag",
                "active_S_magnitude",
                "active_S_magnitude_db",
                "active_S_phase_deg",
                "coefficient_magnitude_squared",
                "coefficient_valid",
                "power_wave_valid",
                "drive_amplitude",
                "drive_power",
                "drive_phase_deg",
                "drive_delay_s",
            )
        )
        writer.writerows(rows)
    logger.info(f"Eigenmode active S-parameter CSV written to {output_path}")
    return output_path


def finalise_eigenmode_ports(grid):
    for port in grid.eigenmodeports:
        port.finalise(grid)
    sources = [port for port in grid.eigenmodeports if port.is_source]
    if not grid.eigenmodeports:
        return None
    # Passive-only virtual guides are useful matched modal loads and probes.
    # Their incident/outgoing power waves are still meaningful and are written
    # to HDF5, but an S matrix cannot be normalised without an active source.
    if not sources:
        for port in grid.eigenmodeports:
            port.response_type = "passive"
        return None

    def excitation_modes(port):
        modes = getattr(port, "excitation_mode_indices", None)
        if modes is not None:
            return tuple(modes)
        mode = getattr(port, "excitation_mode_index", None)
        return () if mode is None else (mode,)

    drive_count = sum(len(excitation_modes(port)) for port in sources)
    # A simultaneous driven state is not an S-matrix column. Report the active
    # reflection coefficient b_i/a_i only for each deliberately driven modal
    # channel, while preserving every raw incident/outgoing wave in HDF5.
    if drive_count != 1:
        for port in grid.eigenmodeports:
            port.response_type = "driven"
        return _write_active_sparameters(grid, sources, excitation_modes)

    source = sources[0]
    for port in grid.eigenmodeports:
        port.response_type = "s_parameter_column"
    for port in grid.eigenmodeports:
        if not np.array_equal(port.result.frequency, source.result.frequency):
            raise ValueError("All eigenmode ports must use identical DFT frequency bins.")
    source_mode_position = source.mode_indices.index(source.excitation_mode_index)
    denominator = source.result.incident[source_mode_position]
    source_coefficient_result_valid = _coefficient_result_valid(source.result)[
        source_mode_position
    ] & np.isfinite(denominator)
    source_power_wave_mask = np.asarray(
        source.power_basis_valid,
        dtype=bool,
    )[source_mode_position]
    # Balanced generalized amplitudes and one-watt power-wave amplitudes use
    # different reference normalizations. Apply the incident floor within
    # each class so a large evanescent coefficient cannot suppress an
    # otherwise well-excited propagating bin (or vice versa).
    source_ratio_valid = _incident_ratio_valid(
        denominator,
        source_coefficient_result_valid,
        source_power_wave_mask,
    )
    source_power_wave_valid = (
        source_ratio_valid
        & np.asarray(source.result.power_wave_valid[source_mode_position], dtype=bool)
        & source_power_wave_mask
        & source.power_matrix_valid
    )
    for port in grid.eigenmodeports:
        port.s_parameters = np.full_like(port.result.outgoing, np.nan + 1j * np.nan)
        np.divide(
            port.result.outgoing,
            denominator[np.newaxis, :],
            out=port.s_parameters,
            where=source_ratio_valid[np.newaxis, :],
        )
        port.s_coefficient_valid = (
            _coefficient_result_valid(port.result) & source_ratio_valid[np.newaxis, :]
        )
        port.s_power_wave_valid = (
            port.s_coefficient_valid
            & np.asarray(port.result.power_wave_valid, dtype=bool)
            & np.asarray(port.power_basis_valid, dtype=bool)
            & port.power_matrix_valid[np.newaxis, :]
            & source_power_wave_valid[np.newaxis, :]
        )
        port.s_parameters[~port.s_coefficient_valid] = np.nan + 1j * np.nan

    suffix = "" if grid.name == "main_grid" else f"_{grid.name}"
    output_path = config.get_model_config().output_file_path.with_name(
        config.get_model_config().output_file_path.name + f"{suffix}_sparameters.csv"
    )
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "frequency_hz",
                "source_port",
                "source_mode",
                "destination_port",
                "destination_mode",
                "S_real",
                "S_imag",
                "S_magnitude",
                "S_magnitude_db",
                "S_phase_deg",
                "coefficient_magnitude_squared",
                "power_wave_valid",
                "coefficient_valid",
            )
        )
        for port in grid.eigenmodeports:
            for mode_position, mode_index in enumerate(port.mode_indices):
                values = port.s_parameters[mode_position]
                for frequency_index, frequency in enumerate(port.result.frequency):
                    value = values[frequency_index]
                    writer.writerow(
                        (
                            float(frequency),
                            source.port_index,
                            source.excitation_mode_index,
                            port.port_index,
                            mode_index,
                            *_complex_csv_fields(value),
                            int(port.s_power_wave_valid[mode_position, frequency_index]),
                            int(port.s_coefficient_valid[mode_position, frequency_index]),
                        )
                    )
    logger.info(f"Eigenmode S-parameter CSV written to {output_path}")
    return output_path
