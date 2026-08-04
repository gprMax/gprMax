# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
# Authors: Craig Warren, Antonis Giannopoulos, John Hartley, and Nathan Mannall
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

"""Frequency-domain multi-mode eigenmode port monitors."""

import csv
import logging
from dataclasses import dataclass

import numpy as np

import gprMax.config as config

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
        if normal_axis == 0:
            measured_eu = 0.5 * (Ey[plane_index, u0:u1, v0:v1] + Ey[plane_index, u0:u1, v0 + 1 : v1 + 1])
            measured_ev = 0.5 * (Ez[plane_index, u0:u1, v0:v1] + Ez[plane_index, u0 + 1 : u1 + 1, v0:v1])
            measured_hu = 0.5 * (Hy[hplane, u0:u1, v0:v1] + Hy[hplane, u0 + 1 : u1 + 1, v0:v1])
            measured_hv = 0.5 * (Hz[hplane, u0:u1, v0:v1] + Hz[hplane, u0:u1, v0 + 1 : v1 + 1])
        elif normal_axis == 1:
            measured_eu = 0.5 * (Ex[u0:u1, plane_index, v0:v1] + Ex[u0:u1, plane_index, v0 + 1 : v1 + 1])
            measured_ev = 0.5 * (Ez[u0:u1, plane_index, v0:v1] + Ez[u0 + 1 : u1 + 1, plane_index, v0:v1])
            measured_hu = 0.5 * (Hx[u0:u1, hplane, v0:v1] + Hx[u0 + 1 : u1 + 1, hplane, v0:v1])
            measured_hv = 0.5 * (Hz[u0:u1, hplane, v0:v1] + Hz[u0:u1, hplane, v0 + 1 : v1 + 1])
        else:
            measured_eu = 0.5 * (Ex[u0:u1, v0:v1, plane_index] + Ex[u0:u1, v0 + 1 : v1 + 1, plane_index])
            measured_ev = 0.5 * (Ey[u0:u1, v0:v1, plane_index] + Ey[u0 + 1 : u1 + 1, v0:v1, plane_index])
            measured_hu = 0.5 * (Hx[u0:u1, v0:v1, hplane] + Hx[u0 + 1 : u1 + 1, v0:v1, hplane])
            measured_hv = 0.5 * (Hy[u0:u1, v0:v1, hplane] + Hy[u0:u1, v0 + 1 : v1 + 1, hplane])
        factor = 0.5 * handedness * measure * dt
        electric_overlap = factor * (
            np.einsum("uv,fmuv->fm", measured_eu, conj_hv, optimize=True)
            - np.einsum("uv,fmuv->fm", measured_ev, conj_hu, optimize=True)
        )
        magnetic_overlap = (
            factor
            * direction_sign
            * (
                np.einsum("fmuv,uv->fm", conj_eu, measured_hv, optimize=True)
                - np.einsum("fmuv,uv->fm", conj_ev, measured_hu, optimize=True)
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


def _dft_phase_at_time(frequencies, time, dtype):
    """Return exp(-j omega t) with float64 transcendental argument reduction."""

    phase_frequencies = np.asarray(frequencies, dtype=np.float64)
    return np.exp(-2j * np.pi * phase_frequencies * float(time)).astype(dtype)


@dataclass(frozen=True)
class EigenmodePortResult:
    frequency: np.ndarray
    incident: np.ndarray
    outgoing: np.ndarray
    valid: np.ndarray
    condition_number: np.ndarray


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
    ):
        self.owner = owner
        self.port_index = int(port_index)
        self.port_id = port_id
        self.is_source = bool(is_source)
        self.excitation_mode_index = None if excitation_mode_index is None else int(excitation_mode_index)
        # A TF/SF source must sample H on its total-field side. A passive port
        # uses the upstream half-cell, where either side is physically
        # equivalent in the absence of a field discontinuity.
        self.magnetic_side = 1 if self.is_source else -1
        self.mode_indices = tuple(int(value) for value in mode_indices)
        self.anchor_frequencies = np.asarray(anchor_frequencies, dtype=np.float64)
        self.anchor_e = anchor_e
        self.anchor_h = anchor_h
        self.anchor_neff = np.asarray(anchor_neff, dtype=np.complex128)
        self.dft_start = float(dft_start)
        self.dft_stop = float(dft_stop)
        self.dft_points = int(dft_points)
        self.result = None
        self.s_parameters = None
        self.s_valid = None

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
        if self.is_source and self.excitation_mode_index not in self.mode_indices:
            raise ValueError("The source excitation mode must be included in its monitored mode indices.")
        if not self.is_source and self.excitation_mode_index is not None:
            raise ValueError("A passive eigenmode port cannot have an excitation mode.")
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
        nominal_frequency = np.linspace(
            self.dft_start,
            self.dft_stop,
            self.dft_points,
            dtype=np.float64,
        )
        self.frequency = np.asarray(nominal_frequency, dtype=real_dtype)
        if self.frequency[-1] > 0.5 / grid.dt:
            raise ValueError(f"Eigenmode port {self.port_id!r} DFT stop frequency exceeds the FDTD Nyquist frequency.")
        if self.anchor_frequencies.size > 1 and (
            nominal_frequency[0] < self.anchor_frequencies[0] or nominal_frequency[-1] > self.anchor_frequencies[-1]
        ):
            logger.warning(
                f"Eigenmode port {self.port_id!r} DFT range extends outside its modal "
                "anchor range; endpoint modal profiles will be used."
            )

        weights = self.owner._linear_anchor_weights(self.frequency.astype(np.float64), self.anchor_frequencies)
        nf = self.frequency.size
        nm = len(self.mode_indices)
        nu, nv = self.owner._transverse_cell_shape()
        shape = (nf, nm, nu, nv)
        self.eu = np.empty(shape, dtype=complex_dtype)
        self.ev = np.empty(shape, dtype=complex_dtype)
        self.hu = np.empty(shape, dtype=complex_dtype)
        self.hv = np.empty(shape, dtype=complex_dtype)
        self.neff = np.empty((nf, nm), dtype=complex_dtype)
        self.mode_power_valid = np.ones((nf, nm), dtype=bool)
        u_axis, v_axis = self.owner.transverse_axes

        for frequency_index in range(nf):
            for mode_position in range(nm):
                electric = []
                magnetic = []
                for component in range(3):
                    electric.append(
                        sum(
                            weights[anchor, frequency_index] * self.anchor_e[anchor][mode_position][component]
                            for anchor in range(self.anchor_frequencies.size)
                        )
                    )
                    magnetic.append(
                        sum(
                            weights[anchor, frequency_index] * self.anchor_h[anchor][mode_position][component]
                            for anchor in range(self.anchor_frequencies.size)
                        )
                    )
                power = float(np.real(self.owner._modal_cross_power(electric, magnetic, grid)))
                if not np.isfinite(power) or power <= 1e-12:
                    self.mode_power_valid[frequency_index, mode_position] = False
                    logger.warning(
                        f"Eigenmode port {self.port_id!r} mode {self.mode_indices[mode_position]} "
                        f"has invalid interpolated power {power:g} at "
                        f"{self.frequency[frequency_index]:g} Hz; using finite fallback normalization."
                    )
                    power = abs(power) if np.isfinite(power) and abs(power) > 1e-12 else 1.0
                scale = 1.0 / np.sqrt(power)
                electric = [field * scale for field in electric]
                magnetic = [field * scale for field in magnetic]
                self.eu[frequency_index, mode_position] = self.owner._average_to_transverse_cells(
                    electric[u_axis], "eu"
                )
                self.ev[frequency_index, mode_position] = self.owner._average_to_transverse_cells(
                    electric[v_axis], "ev"
                )
                self.hu[frequency_index, mode_position] = self.owner._average_to_transverse_cells(
                    magnetic[u_axis], "hu"
                )
                self.hv[frequency_index, mode_position] = self.owner._average_to_transverse_cells(
                    magnetic[v_axis], "hv"
                )
                self.neff[frequency_index, mode_position] = np.sum(
                    weights[:, frequency_index] * self.anchor_neff[:, mode_position]
                )

        self.eu = np.ascontiguousarray(self.eu)
        self.ev = np.ascontiguousarray(self.ev)
        self.hu = np.ascontiguousarray(self.hu)
        self.hv = np.ascontiguousarray(self.hv)
        self.conj_eu = np.ascontiguousarray(np.conj(self.eu))
        self.conj_ev = np.ascontiguousarray(np.conj(self.ev))
        self.conj_hu = np.ascontiguousarray(np.conj(self.hu))
        self.conj_hv = np.ascontiguousarray(np.conj(self.hv))

        measure = (
            grid.dl[self.owner.physical_transverse_axis]
            if self.owner.invariant_axis is not None
            else grid.dl[u_axis] * grid.dl[v_axis]
        )
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
        self.power_matrix_valid = np.all(self.mode_power_valid, axis=1)
        for frequency_index, matrix in enumerate(self.power_matrix):
            if not np.all(np.isfinite(matrix)):
                self.power_matrix_valid[frequency_index] = False
                continue
            eigenvalues = np.linalg.eigvalsh(np.asarray(matrix, dtype=np.complex128))
            scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
            tolerance = 64 * np.finfo(real_dtype).eps * scale
            if np.any(eigenvalues < -tolerance):
                self.power_matrix_valid[frequency_index] = False

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
                f"expected eigenmode DFT iteration {self._next_iteration}, "
                f"received {iteration}"
            )
        real_signature = config.sim_config.dtypes["C_float_or_double"]
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

    def finalise(self, grid):
        nf, nm = self.electric_dft.shape
        complex_dtype = np.dtype(config.sim_config.dtypes["complex"])
        component_dtype = (
            np.float32 if complex_dtype == np.dtype(np.complex64) else np.float64
        )
        condition_limit = min(
            MAX_CONDITION_NUMBER,
            CONDITION_RELATIVE_ERROR_BUDGET / np.finfo(component_dtype).eps,
        )
        incident = np.full((nm, nf), np.nan + 1j * np.nan, dtype=complex_dtype)
        outgoing = np.full_like(incident, np.nan + 1j * np.nan)
        valid = np.zeros((nm, nf), dtype=bool)
        condition = np.full(nf, np.inf, dtype=np.float64)
        magnetic_offset = self.magnetic_side * 0.5 * grid.dl[self.owner.normal_axis]
        beta = 2 * np.pi * self.frequency[:, np.newaxis] * self.neff / config.c
        forward_phase = np.exp(-1j * beta * magnetic_offset)
        backward_phase = np.exp(1j * beta * magnetic_offset)

        for frequency_index in range(nf):
            try:
                # The Gram systems are small. Solve them in complex128 even
                # when the FDTD arrays use complex64, while retaining a
                # validity limit based on the precision of the stored inputs.
                electric_gram = np.asarray(
                    self.electric_gram[frequency_index], dtype=np.complex128
                )
                magnetic_gram = np.asarray(
                    self.magnetic_gram[frequency_index], dtype=np.complex128
                )
                condition[frequency_index] = max(
                    np.linalg.cond(electric_gram),
                    np.linalg.cond(magnetic_gram),
                )
                electric_coeff = np.linalg.solve(
                    electric_gram,
                    np.asarray(self.electric_dft[frequency_index], dtype=np.complex128),
                )
                magnetic_coeff = np.linalg.solve(
                    magnetic_gram,
                    np.asarray(self.magnetic_dft[frequency_index], dtype=np.complex128),
                )
            except np.linalg.LinAlgError:
                continue
            denominator = forward_phase[frequency_index] + backward_phase[frequency_index]
            usable = np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
            a = np.full(nm, np.nan + 1j * np.nan, dtype=complex_dtype)
            a[usable] = (
                magnetic_coeff[usable] + backward_phase[frequency_index, usable] * electric_coeff[usable]
            ) / denominator[usable]
            b = electric_coeff - a
            incident[:, frequency_index] = a
            outgoing[:, frequency_index] = b
            valid[:, frequency_index] = (
                usable
                & np.isfinite(a)
                & np.isfinite(b)
                & np.isfinite(condition[frequency_index])
                & (condition[frequency_index] < condition_limit)
                & self.mode_power_valid[frequency_index]
                & self.power_matrix_valid[frequency_index]
            )

        self.result = EigenmodePortResult(
            frequency=self.frequency.copy(),
            incident=incident,
            outgoing=outgoing,
            valid=valid,
            condition_number=condition,
        )
        return self.result

    def write_hdf5(self, base_group):
        group = base_group.create_group(f"eigenmode_ports/port{self.port_index}")
        group.attrs["ID"] = self.port_id
        group.attrs["IsSource"] = self.is_source
        if self.excitation_mode_index is not None:
            group.attrs["ExcitationMode"] = self.excitation_mode_index
        group.attrs["Direction"] = self.owner.direction
        group.attrs["Normal"] = self.owner.normal
        group.attrs["ModeIndices"] = self.mode_indices
        group.attrs["PlaneIndex"] = self.owner.plane_index
        group.attrs["PhaseReanchorInterval"] = DFT_PHASE_REANCHOR_INTERVAL
        group["frequency"] = self.result.frequency
        group["incident"] = self.result.incident
        group["outgoing"] = self.result.outgoing
        group["valid"] = self.result.valid.astype(np.uint8)
        group["condition_number"] = self.result.condition_number
        group["electric_cross_power_matrix"] = self.electric_gram
        group["power_matrix"] = self.power_matrix
        group["power_normalization_valid"] = self.mode_power_valid.astype(np.uint8)
        group["power_matrix_valid"] = self.power_matrix_valid.astype(np.uint8)
        if self.s_parameters is not None:
            group["S"] = self.s_parameters
            group["valid_S"] = self.s_valid.astype(np.uint8)


def finalise_eigenmode_ports(grid):
    for port in grid.eigenmodeports:
        port.finalise(grid)
    sources = [port for port in grid.eigenmodeports if port.is_source]
    if not grid.eigenmodeports:
        return None
    if len(sources) != 1:
        raise ValueError(
            "Eigenmode S-parameters require one and only one active eigenmode source; " f"found {len(sources)}."
        )

    source = sources[0]
    for port in grid.eigenmodeports:
        if not np.array_equal(port.result.frequency, source.result.frequency):
            raise ValueError("All eigenmode ports must use identical DFT frequency bins.")
    source_mode_position = source.mode_indices.index(source.excitation_mode_index)
    denominator = source.result.incident[source_mode_position]
    source_decomposition_valid = (
        source.result.valid[source_mode_position]
        & source.mode_power_valid[:, source_mode_position]
        & source.power_matrix_valid
        & np.isfinite(denominator)
    )
    peak = float(
        np.max(np.abs(denominator[source_decomposition_valid]), initial=0.0)
    )
    source_valid = (
        source_decomposition_valid
        & (np.abs(denominator) >= peak * 10 ** (INCIDENT_FLOOR_DB / 20))
        & (peak > 0)
    )
    for port in grid.eigenmodeports:
        port.s_parameters = np.full_like(port.result.outgoing, np.nan + 1j * np.nan)
        np.divide(
            port.result.outgoing,
            denominator[np.newaxis, :],
            out=port.s_parameters,
            where=source_valid[np.newaxis, :],
        )
        port.s_valid = (
            port.result.valid
            & port.mode_power_valid.T
            & port.power_matrix_valid[np.newaxis, :]
            & source_valid[np.newaxis, :]
        )

    suffix = "" if grid.name == "main_grid" else f"_{grid.name}"
    output_path = config.get_model_config().output_file_path.with_name(
        config.get_model_config().output_file_path.name + f"{suffix}_sparameters.csv"
    )
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
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
                "valid",
            )
        )
        for port in grid.eigenmodeports:
            for mode_position, mode_index in enumerate(port.mode_indices):
                values = port.s_parameters[mode_position]
                for frequency_index, frequency in enumerate(port.result.frequency):
                    value = values[frequency_index]
                    magnitude = abs(value)
                    if np.isfinite(magnitude) and magnitude > 0:
                        magnitude_db = float(20 * np.log10(magnitude))
                    elif magnitude == 0:
                        magnitude_db = -np.inf
                    else:
                        magnitude_db = np.nan
                    writer.writerow(
                        (
                            float(frequency),
                            source.port_index,
                            source.excitation_mode_index,
                            port.port_index,
                            mode_index,
                            float(np.real(value)),
                            float(np.imag(value)),
                            float(magnitude),
                            magnitude_db,
                            float(np.angle(value, deg=True)),
                            float(magnitude**2),
                            int(port.s_valid[mode_position, frequency_index]),
                        )
                    )
    logger.info(f"Eigenmode S-parameter CSV written to {output_path}")
    return output_path
