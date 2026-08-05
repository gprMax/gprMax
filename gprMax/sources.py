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
import math
from copy import deepcopy
import numpy as np
import numpy.typing as npt

import gprMax.config as config
from gprMax.fdfd_eigenmode_solver.fdfd_1d_mode_solver import FDFD_1D_mode_solver
from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver
from gprMax.eigenmode_plotting import plot_eigenmode_excitation, plot_eigenmode_port_fields
from gprMax.waveforms import Waveform

from .cython.plane_wave import (
    calculate1DWaveformValues,
    getSource,
    updatePlaneWave_electric,
    updatePlaneWave_electric_axial,
    updatePlaneWave_electric_dispersive,
    updatePlaneWave_electric_dispersive_axial,
    updatePlaneWave_magnetic,
    updatePlaneWave_magnetic_axial,
)
from .cython.eigenmode_source import (
    update_eigenmode_magnetic as updateEigenmode_magnetic,
    update_eigenmode_electric as updateEigenmode_electric,
)
from .utilities.utilities import round_value

logger = logging.getLogger(__name__)


class Source:
    """Super-class which describes a generic source."""

    def __init__(self):
        self.ID: str
        self.polarisation = None
        self.coord = np.zeros(3, dtype=np.int32)
        self.coordorigin = np.zeros(3, dtype=np.int32)
        self.start = 0.0
        self.stop = 0.0
        self.waveformID = None
        # Waveform values for sources that need to be calculated on whole timesteps
        self.waveformvalues_wholedt = None
        # Waveform values for sources that need to be calculated on half timesteps
        self.waveformvalues_halfdt = None

    @property
    def xcoord(self) -> int:
        return self.coord[0]

    @xcoord.setter
    def xcoord(self, value: int):
        self.coord[0] = value

    @property
    def ycoord(self) -> int:
        return self.coord[1]

    @ycoord.setter
    def ycoord(self, value: int):
        self.coord[1] = value

    @property
    def zcoord(self) -> int:
        return self.coord[2]

    @zcoord.setter
    def zcoord(self, value: int):
        self.coord[2] = value

    @property
    def xcoordorigin(self) -> int:
        return self.coordorigin[0]

    @xcoordorigin.setter
    def xcoordorigin(self, value: int):
        self.coordorigin[0] = value

    @property
    def ycoordorigin(self) -> int:
        return self.coordorigin[1]

    @ycoordorigin.setter
    def ycoordorigin(self, value: int):
        self.coordorigin[1] = value

    @property
    def zcoordorigin(self) -> int:
        return self.coordorigin[2]

    @zcoordorigin.setter
    def zcoordorigin(self, value: int):
        self.coordorigin[2] = value


class EigenmodeAnchorMismatchError(ValueError):
    '''Raised when the same modal branch cannot be tracked between anchors.'''

    def __init__(
        self,
        message,
        *,
        first_frequency=None,
        second_frequency=None,
        mode_index=None,
        overlap=None,
        context=None,
    ):
        self.detail = message
        self.first_frequency = first_frequency
        self.second_frequency = second_frequency
        self.mode_index = mode_index
        self.overlap = overlap
        self.context = context
        super().__init__(
            message
            + ' This may indicate a degenerate mode or mode crossing. Use a single '
            + 'explicit frequency anchor for this port if a constant modal basis '
            + 'across the band is acceptable.'
        )


def _trim_failed_guard_anchors(frequencies, mismatch, fmin, fmax):
    """Trim an untrackable spectral guard while retaining its inner anchor."""

    first = mismatch.first_frequency
    second = mismatch.second_frequency
    if first is None or second is None:
        return None
    frequencies = tuple(float(value) for value in frequencies)
    tolerance = 1e-12 * max(abs(float(fmin)), abs(float(fmax)), 1.0)
    if second <= fmin + tolerance:
        trimmed = tuple(value for value in frequencies if value >= second - tolerance)
        return ('lower', second, trimmed) if len(trimmed) > 1 else None
    if first >= fmax - tolerance:
        trimmed = tuple(value for value in frequencies if value <= first + tolerance)
        return ('upper', first, trimmed) if len(trimmed) > 1 else None
    return None


def initialise_eigenmode_ports(grid):
    """Resolve one coordinated automatic-anchor policy for every port."""

    ports = [*grid.eigenmodesources, *grid.eigenmodereceivers]
    if not ports:
        return
    requested_policies = {
        id(port): getattr(port, 'requested_anchor_policy', port.anchor_policy)
        for port in ports
    }
    automatic_ports = [
        port for port in ports if requested_policies[id(port)] == 'auto'
    ]
    if not automatic_ports:
        for port in ports:
            port.grid_init(grid)
        return

    common_anchors = tuple(
        float(value)
        for value in (automatic_ports[0].frequencies or (automatic_ports[0].frequency,))
    )
    for port in automatic_ports[1:]:
        anchors = tuple(
            float(value) for value in (port.frequencies or (port.frequency,))
        )
        if anchors != common_anchors:
            raise ValueError(
                'Automatic eigenmode ports must start with identical anchor '
                f'frequencies; got {common_anchors} and {anchors}.'
            )

    band = grid.eigenmodeband
    candidate_anchors = common_anchors
    guard_trimmed = False
    while True:
        grid.eigenmodeports.clear()
        for port in ports:
            requested = requested_policies[id(port)]
            if requested == 'auto':
                port.frequency = candidate_anchors[0]
                port.frequencies = candidate_anchors
                # Suppress the legacy per-port fallback. The coordinator must
                # see a mismatch before choosing one policy for every auto port.
                port.anchor_policy = 'coordinated_auto'
            else:
                port.anchor_policy = requested
            port.port_monitor = None

        source = grid.eigenmodesources[0]
        source_requested = requested_policies[id(source)]
        if source_requested == 'auto':
            source.spectrum_coverage_policy = (
                'allow' if guard_trimmed else 'error'
            )

        failing_port = None
        try:
            for port in ports:
                failing_port = port
                port.grid_init(grid)
        except EigenmodeAnchorMismatchError as mismatch:
            grid.eigenmodeports.clear()
            for port in ports:
                port.anchor_policy = requested_policies[id(port)]
            if requested_policies[id(failing_port)] != 'auto':
                raise

            guard_result = _trim_failed_guard_anchors(
                candidate_anchors,
                mismatch,
                band.fmin,
                band.fmax,
            )
            if guard_result is not None:
                side, retained_frequency, trimmed = guard_result
                if trimmed != candidate_anchors:
                    detail = mismatch.detail.rstrip(' .')
                    logger.warning(
                        f'{detail}, within the {side} spectral guard '
                        f'outside the requested {band.fmin:g} to {band.fmax:g} Hz '
                        f'band. All automatic eigenmode ports will retain broadband '
                        f'tracking from {retained_frequency:g} Hz and use that '
                        f'endpoint modal profile across the trimmed significant-'
                        'spectrum tail.'
                    )
                    candidate_anchors = trimmed
                    guard_trimmed = True
                    continue

            fallback = float(failing_port.fallback_frequency)
            detail = mismatch.detail.rstrip(' .')
            logger.warning(
                f'{detail}. All automatic eigenmode ports will therefore '
                f'use the shared single modal anchor at {fallback:g} Hz. Their '
                'modal decomposition and S-parameters may be inaccurate toward '
                'frequencies far from this anchor. Inspect the failed mode for '
                'cutoff, degeneracy, or an artificial port-boundary mode.'
            )
            candidate_anchors = (fallback,)
            guard_trimmed = False
            continue

        for port in ports:
            requested = requested_policies[id(port)]
            port.anchor_policy = requested
            if requested == 'auto':
                if len(candidate_anchors) == 1:
                    port.resolved_anchor_policy = 'auto_single_fallback'
                elif guard_trimmed:
                    port.resolved_anchor_policy = 'auto_broadband_guard_trimmed'
                else:
                    port.resolved_anchor_policy = 'auto_broadband'
            else:
                port.resolved_anchor_policy = 'explicit'
        return


class EigenmodeSource(Source):
    """Holds data for an eigenmode source and prepares material slices.

    `grid_init()` runs after geometry has been converted to Yee component
    material IDs, so it is the right place to extract material data from
    `G.ID`, solve the transverse mode, and prepare modal fields for TF/SF
    injection.
    """

    FDFD_PEC_PROPERTY = np.inf + 0j
    FDFD_PMC_PROPERTY = np.inf + 0j
    COMPLEX_PROFILE_TOLERANCE = 1e-8
    ANCHOR_OVERLAP_WARNING_THRESHOLD = 0.9
    ANCHOR_OVERLAP_ERROR_THRESHOLD = 0.6

    def __init__(self, G):
        super().__init__()
        self.normal = None
        self.direction = None
        self.normal_axis = None
        self.transverse_axes = None
        self.invariant_axis = None
        self.physical_transverse_axis = None
        self.domain_polarization = None
        self.transverse_start = None
        self.transverse_stop = None
        self.mode_index = None
        self.mode_count = None
        self.mode_indices = ()
        self.frequency = None
        self.frequencies = None
        self.anchor_policy = 'explicit'
        self.requested_anchor_policy = 'explicit'
        self.resolved_anchor_policy = 'explicit'
        self.fallback_frequency = None
        self.spectral_threshold = 1e-3
        self.spectrum_coverage_policy = 'error'
        self.plane_index = None
        self.complex_eps_r_uu = None
        self.complex_eps_r_vv = None
        self.complex_eps_r_ww = None
        self.complex_mu_r_uu = None
        self.complex_mu_r_vv = None
        self.complex_mu_r_ww = None
        self.modal_e = None
        self.modal_h = None
        self.modal_e_real = None
        self.modal_h_real = None
        self.neff = None
        self.complex_neff = None
        self.mode_solver = None
        self.mode_solvers = None
        self.anchor_modal_e = None
        self.anchor_modal_h = None
        self.anchor_complex_neff = None
        self.anchor_overlaps = None
        self.broadband_e_envelopes = None
        self.broadband_h_envelopes = None
        self.broadband_modal_e_real = None
        self.broadband_modal_e_imag = None
        self.broadband_modal_h_real = None
        self.broadband_modal_h_imag = None
        self.broadband_input_waveform = None
        self.broadband_reconstructed_waveform = None
        self.broadband_waveform_error = None
        self.representative_frequency = None
        self.complex_profile_phase = None
        self.complex_profile_residual = None
        self.uses_quadrature = False
        # None selects the default policy: write modal-field plots for a
        # geometry-only build, but not for a normal simulation. True and False
        # explicitly override that policy.
        self.plot_fields = None
        # The excitation waveform/DFT figure has an independent control with
        # the same geometry-only default policy as modal-field figures.
        self.plot_waveform = None
        self.port_index = None
        self.port_id = None
        self.dft_start = None
        self.dft_stop = None
        self.dft_points = None
        self.port_monitor = None

    def grid_init(self, G):
        """Prepare source data that depends on the final built Yee grid."""
        if self.plane_index is None:
            self.plane_index = self._select_plane_index(G)
        frequencies = tuple(self.frequencies or (self.frequency,))
        if len(frequencies) > 1:
            try:
                self._solve_broadband_eigenmode(G, frequencies)
            except EigenmodeAnchorMismatchError as exc:
                if self.anchor_policy != 'auto':
                    raise
                self._fallback_to_single_anchor(G, exc)
        else:
            self.frequency = frequencies[0]
            self._extract_frequency_dependent_materials(G)
            self._solve_eigenmode(G)
            self._prepare_single_frequency_injection(G)
        self._register_port_monitor(G)

    def _fallback_to_single_anchor(self, G, mismatch):
        frequency = float(self.fallback_frequency)
        logger.warning(
            f'{mismatch} Automatic anchors for eigenmode port {self.port_index} '
            f'will therefore use a single modal anchor at {frequency:g} Hz. The '
            'modal field and S-parameters may be inaccurate toward frequencies '
            'far from this anchor.'
        )
        self.frequency = frequency
        self.frequencies = (frequency,)
        self.mode_solvers = None
        self.anchor_modal_e = None
        self.anchor_modal_h = None
        self.anchor_complex_neff = None
        self.anchor_overlaps = None
        self._extract_frequency_dependent_materials(G)
        self._solve_eigenmode(G)
        self._prepare_single_frequency_injection(G)

    def _register_port_monitor(self, G):
        """Register the automatic modal monitor owned by this source."""
        from gprMax.eigenmode_ports import EigenmodePortMonitor

        frequencies = tuple(self.frequencies or (self.frequency,))
        solvers = tuple(self.mode_solvers or (self.mode_solver,))
        mode_indices = tuple(self.mode_indices or range(1, self.mode_count + 1))
        anchor_e = []
        anchor_h = []
        anchor_neff = []
        for solver in solvers:
            frequency_e = []
            frequency_h = []
            frequency_neff = []
            for mode_index in mode_indices:
                electric, magnetic, neff = self._fields_from_solver_mode(solver, mode_index)
                frequency_e.append(electric)
                frequency_h.append(magnetic)
                frequency_neff.append(neff)
            anchor_e.append(frequency_e)
            anchor_h.append(frequency_h)
            anchor_neff.append(frequency_neff)

        if len(frequencies) > 1 and self.anchor_policy == 'auto':
            for mode_position, mode_index in enumerate(mode_indices):
                for frequency_index in range(1, len(frequencies)):
                    overlap = self._modal_overlap(
                        anchor_e[frequency_index - 1][mode_position],
                        anchor_h[frequency_index - 1][mode_position],
                        anchor_e[frequency_index][mode_position],
                        anchor_h[frequency_index][mode_position],
                    )
                    magnitude = float(abs(overlap))
                    if (
                        not np.isfinite(magnitude)
                        or magnitude < self.ANCHOR_OVERLAP_ERROR_THRESHOLD
                    ):
                        frequency = float(self.fallback_frequency)
                        logger.warning(
                            f'Automatic anchor tracking failed for eigenmode port '
                            f'{self.port_index}, mode {mode_index}, between '
                            f'{frequencies[frequency_index - 1]:g} and '
                            f'{frequencies[frequency_index]:g} Hz with overlap '
                            f'{magnitude:.6f}. This may indicate a mode crossing or '
                            f'degeneracy. The port will use a single modal anchor at '
                            f'{frequency:g} Hz. Its modal decomposition and '
                            'S-parameters may be inaccurate toward frequencies far '
                            'from this anchor.'
                        )
                        self.frequency = frequency
                        self.frequencies = (frequency,)
                        self.mode_solvers = None
                        return self.grid_init(G)

        for mode_position, mode_index in enumerate(mode_indices):
            for frequency_index in range(1, len(frequencies)):
                overlap = self._modal_overlap(
                    anchor_e[frequency_index - 1][mode_position],
                    anchor_h[frequency_index - 1][mode_position],
                    anchor_e[frequency_index][mode_position],
                    anchor_h[frequency_index][mode_position],
                )
                magnitude = float(abs(overlap))
                self._check_anchor_overlap(
                    magnitude,
                    frequencies[frequency_index - 1],
                    frequencies[frequency_index],
                    mode_index,
                    "Broadband eigenmode source monitor",
                )
                if np.isfinite(magnitude) and magnitude > 1e-300:
                    factor = np.exp(-1j * np.angle(overlap))
                    anchor_e[frequency_index][mode_position] = [
                        field * factor for field in anchor_e[frequency_index][mode_position]
                    ]
                    anchor_h[frequency_index][mode_position] = [
                        field * factor for field in anchor_h[frequency_index][mode_position]
                    ]

        monitor = EigenmodePortMonitor(
            owner=self,
            port_index=self.port_index,
            port_id=self.port_id,
            is_source=True,
            excitation_mode_index=self.mode_index,
            mode_indices=mode_indices,
            anchor_frequencies=frequencies,
            anchor_e=anchor_e,
            anchor_h=anchor_h,
            anchor_neff=np.asarray(anchor_neff, dtype=np.complex128),
            dft_start=self.dft_start,
            dft_stop=self.dft_stop,
            dft_points=self.dft_points,
        )
        monitor.prepare(G)
        self.port_monitor = monitor
        G.eigenmodeports.append(monitor)
        self._plot_eigenmode_fields()
        self._plot_eigenmode_excitation(G)

    def _store_real_modal_fields(self):
        """Store contiguous real modal arrays in the configured CPU precision."""
        dtype = config.sim_config.dtypes["float_or_double"]
        self.modal_e_real = [np.ascontiguousarray(np.real(field), dtype=dtype) for field in self.modal_e]
        self.modal_h_real = [np.ascontiguousarray(np.real(field), dtype=dtype) for field in self.modal_h]

    def _align_tangential_mode_for_real_injection(self):
        """Optimally phase-align injected fields and return their imaginary residual.

        Only components tangential to the source plane enter the TF/SF
        corrections. Normal modal components can be intrinsically in
        quadrature even when every injected component has a real profile, so
        they must not decide whether temporal quadrature is required.
        """
        impedance = float(config.sim_config.em_consts["z0"])
        total_energy = 0.0
        unconjugated_energy = 0.0j
        for axis in self.transverse_axes:
            electric = np.asarray(self.modal_e[axis], dtype=np.complex128)
            magnetic = impedance * np.asarray(self.modal_h[axis], dtype=np.complex128)
            total_energy += float(np.vdot(electric, electric).real)
            total_energy += float(np.vdot(magnetic, magnetic).real)
            unconjugated_energy += np.sum(electric * electric)
            unconjugated_energy += np.sum(magnetic * magnetic)

        if not np.isfinite(total_energy) or total_energy <= 1e-300:
            raise ValueError(
                "Cannot phase-align the eigenmode source because its tangential "
                "electric and magnetic fields have zero or invalid norm."
            )
        if not np.isfinite(unconjugated_energy):
            raise ValueError(
                "Cannot phase-align the eigenmode source because its tangential "
                "electric or magnetic fields contain non-finite values."
            )

        phase = -0.5 * np.angle(unconjugated_energy)
        phase_factor = np.exp(1j * phase)
        self.modal_e = [field * phase_factor for field in self.modal_e]
        self.modal_h = [field * phase_factor for field in self.modal_h]

        # Computing the residual from the rotated fields avoids cancellation in
        # 1 - abs(sum(field**2)) / sum(abs(field)**2) for nearly real modes.
        imaginary_energy = 0.0
        for axis in self.transverse_axes:
            imaginary_electric = np.imag(self.modal_e[axis])
            imaginary_magnetic = impedance * np.imag(self.modal_h[axis])
            imaginary_energy += float(np.vdot(imaginary_electric, imaginary_electric).real)
            imaginary_energy += float(np.vdot(imaginary_magnetic, imaginary_magnetic).real)
        residual = float(np.sqrt(imaginary_energy / total_energy))
        self.complex_profile_phase = float(phase)
        self.complex_profile_residual = residual
        return residual

    def _prepare_single_frequency_injection(self, G):
        """Choose real-only or in-phase/quadrature single-mode injection."""
        residual = self._align_tangential_mode_for_real_injection()
        self._store_real_modal_fields()
        if residual <= self.COMPLEX_PROFILE_TOLERANCE:
            logger.info(
                "Single-frequency eigenmode tangential complex-profile residual "
                f"is {residual:.3e}; using real-only injection."
            )
            return

        self.uses_quadrature = True
        self.anchor_modal_e = [[np.array(field, dtype=np.complex128, copy=True) for field in self.modal_e]]
        self.anchor_modal_h = [[np.array(field, dtype=np.complex128, copy=True) for field in self.modal_h]]
        self.anchor_complex_neff = np.asarray([complex(self.complex_neff)], dtype=np.complex128)
        self.anchor_overlaps = np.empty(0, dtype=np.float64)
        self.mode_solvers = [self.mode_solver]
        logger.info(
            "Single-frequency eigenmode tangential complex-profile residual "
            f"is {residual:.3e}, above the {self.COMPLEX_PROFILE_TOLERANCE:.3e} "
            "tolerance; using in-phase/quadrature injection."
        )
        self._prepare_broadband_time_traces(
            G,
            (self.frequency,),
            single_frequency_iq=True,
        )

    def _extract_frequency_dependent_materials(self, G):
        """Extract source-plane constitutive properties at the active frequency."""
        (
            self.complex_eps_r_uu,
            self.complex_eps_r_vv,
            self.complex_eps_r_ww,
        ) = self._extract_local_complex_property_tensors(G, electric=True)
        (
            self.complex_mu_r_uu,
            self.complex_mu_r_vv,
            self.complex_mu_r_ww,
        ) = self._extract_local_complex_property_tensors(G, electric=False)

    def _solve_broadband_eigenmode(self, G, frequencies):
        """Solve, align, validate, and synthesize a multi-frequency mode."""
        anchor_e = []
        anchor_h = []
        anchor_neff = []
        solvers = []

        for frequency in frequencies:
            self.frequency = frequency
            self._extract_frequency_dependent_materials(G)
            self._solve_eigenmode(G)
            anchor_e.append([np.array(field, dtype=np.complex128, copy=True) for field in self.modal_e])
            anchor_h.append([np.array(field, dtype=np.complex128, copy=True) for field in self.modal_h])
            anchor_neff.append(complex(self.complex_neff))
            solvers.append(self.mode_solver)

        self._validate_solver_mode_tracking(solvers, frequencies)
        overlaps = self._align_and_validate_anchors(anchor_e, anchor_h, frequencies)
        self.anchor_modal_e = anchor_e
        self.anchor_modal_h = anchor_h
        self.anchor_complex_neff = np.asarray(anchor_neff, dtype=np.complex128)
        self.anchor_overlaps = np.asarray(overlaps, dtype=np.float64)
        self.mode_solvers = solvers
        self._prepare_broadband_time_traces(G, frequencies)

        # Keep representative modal data available to diagnostics and callers.
        representative = (
            len(frequencies) // 2
            if self.representative_frequency is None
            else min(
                range(len(frequencies)),
                key=lambda index: abs(frequencies[index] - self.representative_frequency),
            )
        )
        self.frequency = frequencies[representative]
        self.modal_e = self.anchor_modal_e[representative]
        self.modal_h = self.anchor_modal_h[representative]
        self.mode_solver = self.mode_solvers[representative]
        self.complex_neff = self.anchor_complex_neff[representative]
        self.neff = float(np.real(self.complex_neff))
        dtype = config.sim_config.dtypes["float_or_double"]
        self.modal_e_real = [np.ascontiguousarray(np.real(field), dtype=dtype) for field in self.modal_e]
        self.modal_h_real = [np.ascontiguousarray(np.real(field), dtype=dtype) for field in self.modal_h]

    def _validate_solver_mode_tracking(self, solvers, frequencies):
        mode_indices = tuple(self.mode_indices or (self.mode_index,))
        for mode_index in mode_indices:
            previous_e = None
            previous_h = None
            for frequency_index, solver in enumerate(solvers):
                electric, magnetic, _ = self._fields_from_solver_mode(solver, mode_index)
                if previous_e is not None:
                    overlap = self._modal_overlap(previous_e, previous_h, electric, magnetic)
                    self._check_anchor_overlap(
                        float(abs(overlap)),
                        frequencies[frequency_index - 1],
                        frequencies[frequency_index],
                        mode_index,
                        f'Eigenmode port {self.port_index}',
                    )
                previous_e = electric
                previous_h = magnetic

    def _solve_eigenmode(self, G):
        if self.invariant_axis is not None:
            return self._solve_eigenmode_2d(G)
        return self._solve_eigenmode_3d(G)

    def _solve_eigenmode_3d(self, G):
        """Solve the local 2D eigenmode and map fields onto global components."""
        pec_u_mask, pec_v_mask, pec_w_mask = self._cell_pec_electric_component_masks(G)
        pmc_u_mask, pmc_v_mask, pmc_w_mask = self._cell_pmc_magnetic_component_masks(G)
        solver = FDFD_2D_mode_solver(
            frequency=self.frequency,
            du=G.dl[self.transverse_axes[0]],
            dv=G.dl[self.transverse_axes[1]],
            mode_index=(self.mode_count or self.mode_index) - 1,
            eps_r_uu=self.complex_eps_r_uu,
            eps_r_vv=self.complex_eps_r_vv,
            eps_r_ww=self.complex_eps_r_ww,
            mu_r_uu=self.complex_mu_r_uu,
            mu_r_vv=self.complex_mu_r_vv,
            mu_r_ww=self.complex_mu_r_ww,
            pec_u_mask=pec_u_mask,
            pec_v_mask=pec_v_mask,
            pec_w_mask=pec_w_mask,
            pmc_u_mask=pmc_u_mask,
            pmc_v_mask=pmc_v_mask,
            pmc_w_mask=pmc_w_mask,
        )
        solver.solve()

        self.mode_solver = solver
        self.modal_e, self.modal_h, self.complex_neff = self._fields_from_solver_mode(solver, self.mode_index)
        self.neff = float(np.real(self.complex_neff))
        self._validate_modal_field_shapes()
        self._store_real_modal_fields()

    def _solve_eigenmode_2d(self, G):
        """Solve a true 1D mode for a 2D TM/TE FDTD model."""
        solver_inputs = self._one_dimensional_solver_inputs(G)
        solver = FDFD_1D_mode_solver(
            frequency=self.frequency,
            dt=G.dl[self.physical_transverse_axis],
            mode_index=(self.mode_count or self.mode_index) - 1,
            polarization=self.domain_polarization,
            **solver_inputs,
        )
        solver.solve()

        self.mode_solver = solver
        self.modal_e, self.modal_h, self.complex_neff = self._fields_from_solver_mode(solver, self.mode_index)
        self.neff = float(np.real(self.complex_neff))

        self._validate_modal_field_shapes()
        self._store_real_modal_fields()

    def _fields_from_solver_mode(self, solver, public_mode_index):
        """Map one solved, one-based mode onto global Yee components."""
        mode = public_mode_index - 1
        if mode < 0 or mode >= solver.num_modes:
            raise ValueError(f"Mode index {public_mode_index} is outside the solved 1-{solver.num_modes} range.")
        if isinstance(solver, FDFD_2D_mode_solver):
            local_e = (solver.Eu[:, :, mode], solver.Ev[:, :, mode], solver.Ew[:, :, mode])
            local_h = (solver.Hu[:, :, mode], solver.Hv[:, :, mode], solver.Hw[:, :, mode])
        else:
            t_local = self.transverse_axes.index(self.physical_transverse_axis)
            a_local = self.transverse_axes.index(self.invariant_axis)
            local_e = [np.zeros(shape, dtype=np.complex128) for shape in self._expected_local_field_shapes("E")]
            local_h = [np.zeros(shape, dtype=np.complex128) for shape in self._expected_local_field_shapes("H")]
            if self.domain_polarization == "TM":
                local_e[a_local] = self._embed_1d_profile(solver.Ea[:, mode], a_local, "E")
                local_h[t_local] = self._embed_1d_profile(solver.Ht[:, mode], t_local, "H")
                local_h[2] = self._embed_1d_profile(solver.Hw[:, mode], 2, "H")
            else:
                local_e[t_local] = self._embed_1d_profile(solver.Et[:, mode], t_local, "E")
                local_e[2] = self._embed_1d_profile(solver.Ew[:, mode], 2, "E")
                local_h[a_local] = self._embed_1d_profile(solver.Ha[:, mode], a_local, "H")

        local_to_global = (*self.transverse_axes, self.normal_axis)
        electric = [None, None, None]
        magnetic = [None, None, None]
        for local_axis, global_axis in enumerate(local_to_global):
            electric[global_axis] = np.array(local_e[local_axis], dtype=np.complex128, copy=True)
            magnetic[global_axis] = np.array(local_h[local_axis], dtype=np.complex128, copy=True)
        if self._modal_basis_handedness() < 0:
            magnetic = [-field for field in magnetic]
        return electric, magnetic, complex(solver.complex_neff[mode])

    def _expected_local_field_shapes(self, field_kind):
        nu, nv = self._transverse_cell_shape()
        if field_kind == "E":
            return ((nu, nv + 1), (nu + 1, nv), (nu + 1, nv + 1))
        return ((nu + 1, nv), (nu, nv + 1), (nu, nv))

    def _sample_1d_component(self, values):
        """Collapse a local 2D Yee component onto the live invariant layer."""
        invariant_local = self.transverse_axes.index(self.invariant_axis)
        live_layer = 1 if self.domain_polarization == "TE" else 0
        # TE fields propagate on the shared interior plane at index 1. Some
        # inactive components are cell-sampled on the invariant axis and may
        # have no index 1 in TM, so clamp those unused samples to index 0.
        sample_index = min(live_layer, values.shape[invariant_local] - 1)
        return np.take(values, sample_index, axis=invariant_local).copy()

    def _one_dimensional_solver_inputs(self, G):
        """Return staggered 1D material arrays and constraint masks."""
        t_local = self.transverse_axes.index(self.physical_transverse_axis)
        a_local = self.transverse_axes.index(self.invariant_axis)
        eps = (
            self.complex_eps_r_uu,
            self.complex_eps_r_vv,
            self.complex_eps_r_ww,
        )
        mu = (
            self.complex_mu_r_uu,
            self.complex_mu_r_vv,
            self.complex_mu_r_ww,
        )
        pec = self._cell_pec_electric_component_masks(G)
        pmc = self._cell_pmc_magnetic_component_masks(G)
        return {
            "eps_r_t": self._sample_1d_component(eps[t_local]),
            "eps_r_a": self._sample_1d_component(eps[a_local]),
            "eps_r_w": self._sample_1d_component(eps[2]),
            "mu_r_t": self._sample_1d_component(mu[t_local]),
            "mu_r_a": self._sample_1d_component(mu[a_local]),
            "mu_r_w": self._sample_1d_component(mu[2]),
            "pec_t_mask": self._sample_1d_component(pec[t_local]),
            "pec_a_mask": self._sample_1d_component(pec[a_local]),
            "pec_w_mask": self._sample_1d_component(pec[2]),
            "pmc_t_mask": self._sample_1d_component(pmc[t_local]),
            "pmc_a_mask": self._sample_1d_component(pmc[a_local]),
            "pmc_w_mask": self._sample_1d_component(pmc[2]),
        }

    def _embed_1d_profile(self, profile, local_component, field_kind):
        """Expand one physical line profile into a thin 2D Yee component."""
        shape = self._expected_local_field_shapes(field_kind)[local_component]
        result = np.zeros(shape, dtype=np.complex128)
        invariant_local = self.transverse_axes.index(self.invariant_axis)
        layer = 1 if self.domain_polarization == "TE" else 0
        if invariant_local == 0:
            result[layer, :] = profile
        else:
            result[:, layer] = profile
        return result

    def _validate_modal_field_shapes(self):
        nu, nv = self._transverse_cell_shape()
        expected_e = ((nu, nv + 1), (nu + 1, nv), (nu + 1, nv + 1))
        expected_h = ((nu + 1, nv), (nu, nv + 1), (nu, nv))
        local_to_global = (self.transverse_axes[0], self.transverse_axes[1], self.normal_axis)
        for local_axis, global_axis in enumerate(local_to_global):
            actual = self.modal_e[global_axis].shape
            if actual != expected_e[local_axis]:
                raise ValueError(
                    f"Eigenmode E local component {local_axis} shape {actual} does not match {expected_e[local_axis]}."
                )
            actual = self.modal_h[global_axis].shape
            if actual != expected_h[local_axis]:
                raise ValueError(
                    f"Eigenmode H local component {local_axis} shape {actual} does not match {expected_h[local_axis]}."
                )

    def _modal_basis_handedness(self):
        basis = np.eye(3, dtype=np.int32)
        transverse_u = basis[self.transverse_axes[0]]
        transverse_v = basis[self.transverse_axes[1]]
        normal = basis[self.normal_axis]
        return int(np.dot(np.cross(transverse_u, transverse_v), normal))

    def _modal_overlap(self, first_e, first_h, second_e, second_h):
        """Return the normalized complex overlap of two modal field sets."""
        numerator = 0.0j
        first_norm = 0.0
        second_norm = 0.0
        impedance = float(config.sim_config.em_consts["z0"])
        for first, second in zip(first_e, second_e):
            numerator += np.vdot(first, second)
            first_norm += float(np.vdot(first, first).real)
            second_norm += float(np.vdot(second, second).real)
        for first, second in zip(first_h, second_h):
            first_scaled = impedance * first
            second_scaled = impedance * second
            numerator += np.vdot(first_scaled, second_scaled)
            first_norm += float(np.vdot(first_scaled, first_scaled).real)
            second_norm += float(np.vdot(second_scaled, second_scaled).real)
        denominator = np.sqrt(first_norm * second_norm)
        if not np.isfinite(denominator) or denominator <= 1e-300:
            return 0.0j
        return numerator / denominator

    @classmethod
    def _check_anchor_overlap(
        cls,
        magnitude,
        first_frequency,
        second_frequency,
        mode_index,
        context,
    ):
        """Apply the fixed warning and error limits for adjacent anchors."""
        description = (
            f"{context} anchor mode overlap between {first_frequency:g} Hz and "
            f"{second_frequency:g} Hz for mode index {mode_index} is "
            f"{magnitude:.6f}"
        )
        if not np.isfinite(magnitude) or magnitude < cls.ANCHOR_OVERLAP_ERROR_THRESHOLD:
            raise EigenmodeAnchorMismatchError(
                f"{description}, below the minimum "
                f"{cls.ANCHOR_OVERLAP_ERROR_THRESHOLD:.6f}. The broadband "
                "anchor modes cannot be tracked reliably. Use a "
                "single-frequency eigenmode solver instead.",
                first_frequency=float(first_frequency),
                second_frequency=float(second_frequency),
                mode_index=int(mode_index),
                overlap=float(magnitude),
                context=context,
            )
        if magnitude < cls.ANCHOR_OVERLAP_WARNING_THRESHOLD:
            logger.warning(
                f"{description}, below the warning threshold "
                f"{cls.ANCHOR_OVERLAP_WARNING_THRESHOLD:.6f}. The run will "
                "continue, but inspect the mode ordering, cutoff, degeneracy, "
                "and anchor spacing."
            )

    def _align_and_validate_anchors(self, anchor_e, anchor_h, frequencies):
        """Phase-align consecutive anchors after enforcing overlap limits."""
        overlaps = []
        for index in range(1, len(frequencies)):
            overlap = self._modal_overlap(
                anchor_e[index - 1],
                anchor_h[index - 1],
                anchor_e[index],
                anchor_h[index],
            )
            magnitude = float(abs(overlap))
            self._check_anchor_overlap(
                magnitude,
                frequencies[index - 1],
                frequencies[index],
                self.mode_index,
                "Broadband eigenmode source",
            )

            phase_aligned = np.isfinite(magnitude) and magnitude > 1e-300
            phase_factor = np.exp(-1j * np.angle(overlap)) if phase_aligned else 1.0 + 0.0j
            anchor_e[index] = [field * phase_factor for field in anchor_e[index]]
            anchor_h[index] = [field * phase_factor for field in anchor_h[index]]
            overlaps.append(magnitude)
            logger.info(
                f"Eigenmode anchor overlap {magnitude:.6f} between "
                f"{frequencies[index - 1]:g} Hz and {frequencies[index]:g} Hz; "
                + ("the latter anchor was phase-aligned." if phase_aligned else "its phase was left unchanged.")
            )
        return overlaps

    @staticmethod
    def _average_to_transverse_cells(field, component):
        if component in ("eu", "hv"):
            return 0.5 * (field[:, :-1] + field[:, 1:])
        if component in ("ev", "hu"):
            return 0.5 * (field[:-1, :] + field[1:, :])
        raise ValueError(f"Unknown transverse component {component!r}.")

    def _modal_cross_power(self, electric, magnetic, G):
        """Return complex power pairing for two global modal field sets."""
        if self.invariant_axis is not None:
            return self._modal_cross_power_2d(electric, magnetic, G)

        u_axis, v_axis = self.transverse_axes
        eu = self._average_to_transverse_cells(electric[u_axis], "eu")
        ev = self._average_to_transverse_cells(electric[v_axis], "ev")
        hu = self._average_to_transverse_cells(magnetic[u_axis], "hu")
        hv = self._average_to_transverse_cells(magnetic[v_axis], "hv")
        flux = eu * np.conj(hv) - ev * np.conj(hu)
        if self.invariant_axis is None:
            measure = G.dl[u_axis] * G.dl[v_axis]
        else:
            measure = G.dl[self.physical_transverse_axis]
        return 0.5 * self._modal_basis_handedness() * np.sum(flux) * measure

    def _modal_cross_power_2d(self, electric, magnetic, G):
        """Return modal power per metre from the live 2D field profiles.

        The invariant direction in a 2D grid is represented by a small
        synthetic Yee dimension. Its inactive layers must not be averaged
        into the physical profiles, particularly for TE where both power
        carrying components occupy the shared interior layer.
        """
        transverse_axis = self.physical_transverse_axis
        invariant_axis = self.invariant_axis
        invariant_local = self.transverse_axes.index(invariant_axis)
        layer = 1 if self.domain_polarization == "TE" else 0

        def live_profile(field):
            return np.take(field, layer, axis=invariant_local)

        if self.domain_polarization == "TE":
            electric_profile = live_profile(electric[transverse_axis])
            magnetic_profile = live_profile(magnetic[invariant_axis])
        else:
            electric_profile = live_profile(electric[invariant_axis])
            magnetic_profile = live_profile(magnetic[transverse_axis])
            electric_profile = 0.5 * (electric_profile[:-1] + electric_profile[1:])
            magnetic_profile = 0.5 * (magnetic_profile[:-1] + magnetic_profile[1:])

        basis = np.eye(3, dtype=np.int32)
        flux_sign = int(
            np.dot(
                np.cross(basis[transverse_axis], basis[invariant_axis]),
                basis[self.normal_axis],
            )
        )
        if self.domain_polarization == "TM":
            flux_sign *= -1
        return 0.5 * flux_sign * np.sum(electric_profile * np.conj(magnetic_profile)) * G.dl[transverse_axis]

    @staticmethod
    def _linear_anchor_weights(bin_frequencies, anchor_frequencies):
        """Return partition-of-unity weights with endpoint extrapolation.

        Frequencies between anchors use ordinary piecewise-linear weights.
        Bins below or above the anchor range retain the nearest endpoint mode.
        The spectrum-coverage check warns when those extrapolated bins are
        significant; retaining them avoids a hard spectral truncation and its
        associated time-domain ringing.
        """
        anchor_frequencies = np.asarray(anchor_frequencies, dtype=np.float64)
        weights = np.zeros((anchor_frequencies.size, bin_frequencies.size), dtype=np.float64)
        weights[0, bin_frequencies < anchor_frequencies[0]] = 1.0
        weights[-1, bin_frequencies > anchor_frequencies[-1]] = 1.0
        inside = (bin_frequencies >= anchor_frequencies[0]) & (bin_frequencies <= anchor_frequencies[-1])
        for bin_index in np.flatnonzero(inside):
            frequency = bin_frequencies[bin_index]
            if frequency == anchor_frequencies[-1]:
                weights[-1, bin_index] = 1.0
                continue
            lower = int(np.searchsorted(anchor_frequencies, frequency, side="right") - 1)
            lower = max(0, min(lower, anchor_frequencies.size - 2))
            span = anchor_frequencies[lower + 1] - anchor_frequencies[lower]
            upper_weight = (frequency - anchor_frequencies[lower]) / span
            weights[lower, bin_index] = 1.0 - upper_weight
            weights[lower + 1, bin_index] = upper_weight
        return weights

    @staticmethod
    def _magnetic_stagger_factor(omega, beta, dt, normal_spacing):
        """Return each frequency bin's own E/H time-and-space staggering."""
        return np.exp(1j * (0.5 * omega * dt + 0.5 * beta * normal_spacing))

    def _prepare_broadband_time_traces(
        self,
        G,
        frequencies,
        *,
        single_frequency_iq=False,
    ):
        """Build real temporal bases for complex, linearly interpolated modes."""
        self.uses_quadrature = True
        sample_count = int(G.iterations)
        times = np.arange(sample_count, dtype=np.float64) * G.dt
        waveform = np.asarray(
            [self.waveform.calculate_value(time - self.start, G.dt) for time in times],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(waveform)):
            logger.warning(
                "The broadband eigenmode source waveform contains non-finite samples. "
                "Replacing them with zero and continuing."
            )
            waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        padded_count = 1 << int(np.ceil(np.log2(max(2, 2 * sample_count))))
        spectrum = np.fft.rfft(waveform, n=padded_count)
        bin_frequencies = np.fft.rfftfreq(padded_count, d=G.dt)
        spectrum_magnitude = np.abs(spectrum)
        peak = float(np.max(spectrum_magnitude))
        positive = bin_frequencies > 0
        positive_magnitude = spectrum_magnitude[positive]
        if positive_magnitude.size and np.any(positive_magnitude > 0):
            peak_index = int(np.argmax(positive_magnitude))
            self.representative_frequency = float(bin_frequencies[positive][peak_index])
        else:
            self.representative_frequency = None
        if not np.isfinite(peak) or peak <= 0:
            logger.warning(
                "The broadband eigenmode source waveform has no finite spectral "
                "energy. Continuing with a zero-valued source."
            )

        significant = (
            spectrum_magnitude >= self.spectral_threshold * peak
            if np.isfinite(peak) and peak > 0
            else np.zeros_like(spectrum_magnitude, dtype=bool)
        )
        significant_indices = np.flatnonzero(significant)
        if significant_indices.size == 0:
            logger.warning(
                "The broadband eigenmode source waveform has no significant FFT "
                "bins. Continuing with endpoint mode extrapolation."
            )
            significant_low = float("nan")
            significant_high = float("nan")
        else:
            significant_low = float(bin_frequencies[significant_indices[0]])
            significant_high = float(bin_frequencies[significant_indices[-1]])
        if (
            not single_frequency_iq
            and significant_indices.size
            and (significant_low < frequencies[0] or significant_high > frequencies[-1])
        ):
            if self.spectrum_coverage_policy == 'error':
                raise ValueError(
                    'Eigenmode anchors do not cover the significant excitation '
                    f'spectrum: anchors span {frequencies[0]:g} to '
                    f'{frequencies[-1]:g} Hz, while significant bins span '
                    f'{significant_low:g} to {significant_high:g} Hz. Use '
                    'per-port anchors=\'auto\', provide wider explicit anchors, or '
                    'use EigenmodeExcitation with waveform=\'auto\' for a validated '
                    'bandpass spectrum.'
                )
            if self.spectrum_coverage_policy == 'warn':
                logger.warning(
                    "Broadband eigenmode anchor frequencies do not cover the significant waveform spectrum: "
                    f"anchors span {frequencies[0]:g} to {frequencies[-1]:g} Hz, while bins above "
                    f"{self.spectral_threshold:g} of the peak span {significant_low:g} to "
                    f"{significant_high:g} Hz. Add wider frequency anchors, narrow the waveform "
                    "bandwidth, or use the single-frequency eigenmode source. Continuing by "
                    "using the nearest endpoint mode outside the anchor range."
                )
        if significant[0] or (padded_count % 2 == 0 and significant[-1]):
            if single_frequency_iq:
                logger.warning(
                    "The single-frequency eigenmode source is using I/Q injection, but its "
                    "waveform has significant DC or Nyquist content, where a general complex "
                    "modal profile cannot be synthesized safely. Use a band-limited zero-mean "
                    "waveform. Continuing after discarding the DC and Nyquist bins."
                )
            else:
                logger.warning(
                    "The broadband eigenmode source has significant DC or Nyquist content, where a "
                    "positive-frequency propagating eigenmode cannot be synthesized safely. Use a "
                    "band-limited zero-mean waveform or the single-frequency eigenmode source. "
                    "Continuing after discarding the DC and Nyquist bins."
                )

        weights = self._linear_anchor_weights(bin_frequencies, frequencies)
        partition = np.sum(weights, axis=0)
        if not np.allclose(partition, 1.0, rtol=0.0, atol=1e-14):
            logger.warning(
                "Internal broadband eigenmode interpolation error: anchor weights "
                "do not form a partition of unity. Repairing the weights and continuing."
            )
            usable = np.isfinite(partition) & (np.abs(partition) > 1e-300)
            weights[:, usable] /= partition[usable]
            for bin_index in np.flatnonzero(~usable):
                nearest = int(np.argmin(np.abs(np.asarray(frequencies, dtype=np.float64) - bin_frequencies[bin_index])))
                weights[:, bin_index] = 0.0
                weights[nearest, bin_index] = 1.0
            partition = np.sum(weights, axis=0)
        anchor_count = len(frequencies)
        power_matrix = np.empty((anchor_count, anchor_count), dtype=np.complex128)
        for e_index in range(anchor_count):
            for h_index in range(anchor_count):
                power_matrix[e_index, h_index] = self._modal_cross_power(
                    self.anchor_modal_e[e_index],
                    self.anchor_modal_h[h_index],
                    G,
                )
        interpolated_power = np.real(np.einsum("kn,kl,ln->n", weights, power_matrix, weights, optimize=True))
        active_bins = np.sum(weights, axis=0) > 0
        invalid_power = active_bins & (~np.isfinite(interpolated_power) | (interpolated_power <= 1e-12))
        if np.any(invalid_power):
            invalid_indices = np.flatnonzero(invalid_power)
            bad_frequency = float(bin_frequencies[invalid_indices[0]])
            bad_power = float(interpolated_power[invalid_indices[0]])
            logger.warning(
                "Cannot normalize the interpolated broadband eigenmode at "
                f"{bad_frequency:g} Hz: modal power is {bad_power:g}. Add an anchor near this "
                "frequency, narrow the bandwidth, or use the single-frequency eigenmode source. "
                f"Continuing with fallback normalization for {invalid_indices.size} FFT bin(s)."
            )
            finite_nonzero = invalid_power & np.isfinite(interpolated_power) & (np.abs(interpolated_power) > 1e-12)
            interpolated_power[finite_nonzero] = np.abs(interpolated_power[finite_nonzero])
            unresolved = np.flatnonzero(invalid_power & ~finite_nonzero)
            if unresolved.size:
                nearest = np.argmax(weights[:, unresolved], axis=0)
                anchor_power = np.real(np.diag(power_matrix))[nearest]
                interpolated_power[unresolved] = np.where(
                    np.isfinite(anchor_power) & (np.abs(anchor_power) > 1e-12),
                    np.abs(anchor_power),
                    1.0,
                )

        normalization = np.zeros_like(interpolated_power)
        normalization[active_bins] = 1.0 / np.sqrt(interpolated_power[active_bins])
        omega = 2 * np.pi * bin_frequencies
        interpolated_neff = np.einsum("kn,k->n", weights, self.anchor_complex_neff, optimize=True)
        beta = omega * interpolated_neff / config.sim_config.em_consts["c"]
        normal_spacing = G.dl[self.normal_axis]
        magnetic_phase = self._magnetic_stagger_factor(omega, beta, G.dt, normal_spacing)

        electric_weights = weights * (spectrum * normalization)[np.newaxis, :]
        magnetic_weights = weights * (spectrum * normalization * magnetic_phase)[np.newaxis, :]
        # DC and Nyquist are self-conjugate FFT bins and cannot carry a
        # general complex modal coefficient.
        electric_weights[:, 0] = 0
        magnetic_weights[:, 0] = 0
        if padded_count % 2 == 0:
            electric_weights[:, -1] = 0
            magnetic_weights[:, -1] = 0

        scalar_spectrum = spectrum * partition
        scalar_spectrum[0] = 0
        if padded_count % 2 == 0:
            scalar_spectrum[-1] = 0
        reconstructed_waveform = np.fft.irfft(scalar_spectrum, n=padded_count)[:sample_count]
        waveform_peak = float(np.max(np.abs(waveform)))
        reconstruction_error = (
            float(np.max(np.abs(reconstructed_waveform - waveform)) / waveform_peak)
            if waveform_peak > 0
            else float(np.max(np.abs(reconstructed_waveform - waveform)))
        )
        self.broadband_input_waveform = waveform
        self.broadband_reconstructed_waveform = reconstructed_waveform
        self.broadband_waveform_error = reconstruction_error

        dtype = config.sim_config.dtypes["float_or_double"]
        self.broadband_e_envelopes = np.empty((anchor_count, 2, sample_count), dtype=dtype)
        self.broadband_h_envelopes = np.empty((anchor_count, 2, sample_count), dtype=dtype)
        for anchor in range(anchor_count):
            self.broadband_e_envelopes[anchor, 0] = np.fft.irfft(electric_weights[anchor], n=padded_count)[
                :sample_count
            ]
            self.broadband_e_envelopes[anchor, 1] = np.fft.irfft(1j * electric_weights[anchor], n=padded_count)[
                :sample_count
            ]
            self.broadband_h_envelopes[anchor, 0] = np.fft.irfft(magnetic_weights[anchor], n=padded_count)[
                :sample_count
            ]
            self.broadband_h_envelopes[anchor, 1] = np.fft.irfft(1j * magnetic_weights[anchor], n=padded_count)[
                :sample_count
            ]

        def split_fields(anchor_fields):
            real_fields = []
            imag_fields = []
            for fields in anchor_fields:
                real_fields.append([np.ascontiguousarray(np.real(field), dtype=dtype) for field in fields])
                imag_fields.append([np.ascontiguousarray(np.imag(field), dtype=dtype) for field in fields])
            return real_fields, imag_fields

        (
            self.broadband_modal_e_real,
            self.broadband_modal_e_imag,
        ) = split_fields(self.anchor_modal_e)
        (
            self.broadband_modal_h_real,
            self.broadband_modal_h_imag,
        ) = split_fields(self.anchor_modal_h)
        if single_frequency_iq:
            logger.info(
                "Prepared single-frequency I/Q eigenmode source with "
                f"{sample_count} time samples and significant waveform coverage "
                f"from {significant_low:g} to {significant_high:g} Hz. Scalar "
                "waveform reconstruction relative peak error is "
                f"{reconstruction_error:.3e}."
            )
        else:
            logger.info(
                f"Prepared broadband eigenmode source with {anchor_count} anchors, "
                f"{sample_count} time samples, and significant waveform coverage from "
                f"{significant_low:g} to {significant_high:g} Hz. Scalar waveform "
                f"reconstruction relative peak error is {reconstruction_error:.3e}."
            )

    def _should_plot_eigenmode_fields(self):
        """Return the explicit setting or the geometry-only default."""
        return bool(config.sim_config.geometry_only) if self.plot_fields is None else bool(self.plot_fields)

    def _should_plot_eigenmode_excitation(self):
        """Return the excitation-plot setting or the geometry-only default."""
        if self.plot_waveform is None:
            return bool(config.sim_config.geometry_only)
        return bool(self.plot_waveform)

    def _plot_eigenmode_fields(self):
        if not self._should_plot_eigenmode_fields():
            return

        input_path = config.sim_config.input_file_path
        output_dir = input_path.parent
        frequencies = tuple(self.frequencies or (self.frequency,))
        solvers = tuple(self.mode_solvers or (self.mode_solver,))
        mode_indices = tuple(self.mode_indices or range(1, self.mode_count + 1))
        for mode_index in mode_indices:
            field_path = output_dir / f"{input_path.stem}_Port{self.port_index}_Mode{mode_index}.png"
            plot_eigenmode_port_fields(
                solvers=solvers,
                frequencies=frequencies,
                mode_index=mode_index,
                port_index=self.port_index,
                output_path=field_path,
            )
            logger.info(
                f"Eigenmode port {self.port_index}, mode {mode_index} tangential "
                f"vector-field plot written to {field_path}"
            )

    def _plot_eigenmode_excitation(self, G):
        """Write the single excitation waveform and its exact port-bin DFT."""
        if not self._should_plot_eigenmode_excitation():
            return

        sample_count = int(G.iterations)
        samples = self.broadband_input_waveform
        if samples is None:
            times = np.arange(sample_count, dtype=np.float64) * G.dt
            samples = np.asarray(
                [self.waveform.calculate_value(time - self.start, G.dt) for time in times],
                dtype=np.float64,
            )
        input_path = config.sim_config.input_file_path
        output_path = input_path.parent / f"{input_path.stem}_EigenmodeExcitation.png"
        plot_eigenmode_excitation(
            samples=samples,
            dt=G.dt,
            dft_frequencies=self.port_monitor.frequency,
            band_start=self.dft_start,
            band_stop=self.dft_stop,
            port_index=self.port_index,
            waveform_id=self.waveformID,
            spectral_threshold=self.spectral_threshold,
            output_path=output_path,
        )
        logger.info(f"Eigenmode excitation waveform and DFT plot written to {output_path}")

    def _select_plane_index(self, G):
        """Choose the normal-axis plane from the propagation direction."""
        axis_names = ("x", "y", "z")
        axis_name = axis_names[self.normal_axis]
        if self.direction == "+":
            return G.pmls["thickness"][f"{axis_name}0"]
        return G.size[self.normal_axis] - G.pmls["thickness"][f"{axis_name}max"]

    def _extract_local_complex_property_tensors(self, G, electric):
        """Return local uu, vv, ww complex er or mu_r arrays on the Yee slice."""
        field_kind = "E" if electric else "H"
        component_ids = []
        local_to_global = (self.transverse_axes[0], self.transverse_axes[1], self.normal_axis)
        for local_axis, global_axis in enumerate(local_to_global):
            component = global_axis if electric else global_axis + 3
            ids = self._slice_local_component_ids(G, component, local_axis, field_kind)
            component_ids.append(ids)

        used_ids = np.unique(np.concatenate([ids.ravel() for ids in component_ids]))
        material_values = np.zeros(len(G.materials), dtype=np.complex128)
        materials_by_id = {material.numID: material for material in G.materials}
        for material_id in used_ids:
            material = materials_by_id[int(material_id)]
            material_values[material_id] = self._complex_er(material) if electric else self._complex_mur(material)

        return tuple(material_values[ids].copy() for ids in component_ids)

    def _transverse_cell_shape(self):
        u0, v0 = self.transverse_start
        u1, v1 = self.transverse_stop
        return u1 - u0, v1 - v0

    def _local_component_ranges(self, local_axis, field_kind):
        u0, v0 = self.transverse_start
        u1, v1 = self.transverse_stop
        if field_kind == "E":
            if local_axis == 0:
                return slice(u0, u1), slice(v0, v1 + 1)
            if local_axis == 1:
                return slice(u0, u1 + 1), slice(v0, v1)
            return slice(u0, u1 + 1), slice(v0, v1 + 1)

        if local_axis == 0:
            return slice(u0, u1 + 1), slice(v0, v1)
        if local_axis == 1:
            return slice(u0, u1), slice(v0, v1 + 1)
        return slice(u0, u1), slice(v0, v1)

    def _slice_local_component_ids(self, G, component, local_axis, field_kind):
        u_slice, v_slice = self._local_component_ranges(local_axis, field_kind)
        grid_slices = [slice(None), slice(None), slice(None)]
        grid_slices[self.normal_axis] = self.plane_index
        grid_slices[self.transverse_axes[0]] = u_slice
        grid_slices[self.transverse_axes[1]] = v_slice
        return G.ID[(component, *grid_slices)]

    def _cell_pec_electric_component_masks(self, G):
        """Build local Yee electric PEC masks from cell-centred PEC geometry.

        Component IDs on non-averaged PEC boxes are one-sided at Yee faces.
        These masks supplement the component-sampled material IDs so opposite
        PEC faces produce symmetric constraints in the transverse mode solve.
        """
        cell_pec_mask = self._slice_cell_pec_mask(G)
        nu, nv = self._transverse_cell_shape()
        pec_u_mask = np.zeros((nu, nv + 1), dtype=bool)
        pec_v_mask = np.zeros((nu + 1, nv), dtype=bool)
        pec_w_mask = np.zeros((nu + 1, nv + 1), dtype=bool)
        if cell_pec_mask.size == 0:
            return pec_u_mask, pec_v_mask, pec_w_mask

        cu, cv = cell_pec_mask.shape
        pec_u_mask[:cu, :cv] |= cell_pec_mask
        pec_u_mask[:cu, 1 : cv + 1] |= cell_pec_mask

        pec_v_mask[:cu, :cv] |= cell_pec_mask
        pec_v_mask[1 : cu + 1, :cv] |= cell_pec_mask

        pec_w_mask[:cu, :cv] |= cell_pec_mask
        pec_w_mask[1 : cu + 1, :cv] |= cell_pec_mask
        pec_w_mask[:cu, 1 : cv + 1] |= cell_pec_mask
        pec_w_mask[1 : cu + 1, 1 : cv + 1] |= cell_pec_mask
        return pec_u_mask, pec_v_mask, pec_w_mask

    def _slice_cell_pec_mask(self, G):
        return self._slice_cell_constraint_mask(G, electric=True)

    def _cell_pmc_magnetic_component_masks(self, G):
        """Build local Yee magnetic PMC masks from cell-centred PMC geometry.

        Component-sampled PMC material IDs constrain their exact H positions
        through the non-finite permeability tensors. These masks supplement
        those IDs so both own-axis H faces of every PMC cell are constrained.
        """
        cell_pmc_mask = self._slice_cell_pmc_mask(G)
        nu, nv = self._transverse_cell_shape()
        pmc_u_mask = np.zeros((nu + 1, nv), dtype=bool)
        pmc_v_mask = np.zeros((nu, nv + 1), dtype=bool)
        pmc_w_mask = np.zeros((nu, nv), dtype=bool)
        if cell_pmc_mask.size == 0:
            return pmc_u_mask, pmc_v_mask, pmc_w_mask

        cu, cv = cell_pmc_mask.shape
        pmc_u_mask[:cu, :cv] |= cell_pmc_mask
        pmc_u_mask[1 : cu + 1, :cv] |= cell_pmc_mask

        pmc_v_mask[:cu, :cv] |= cell_pmc_mask
        pmc_v_mask[:cu, 1 : cv + 1] |= cell_pmc_mask

        pmc_w_mask[:cu, :cv] |= cell_pmc_mask
        return pmc_u_mask, pmc_v_mask, pmc_w_mask

    def _slice_cell_pmc_mask(self, G):
        return self._slice_cell_constraint_mask(G, electric=False)

    def _slice_cell_constraint_mask(self, G, electric):
        """Return source-cross-section cells occupied by PEC or PMC."""
        u0, v0 = self.transverse_start
        u1, v1 = self.transverse_stop
        normal_indices = [
            index for index in (self.plane_index - 1, self.plane_index) if 0 <= index < G.solid.shape[self.normal_axis]
        ]
        if not normal_indices:
            return np.zeros((u1 - u0, v1 - v0), dtype=bool)

        material_is_constrained = np.zeros(len(G.materials), dtype=bool)
        for material in G.materials:
            property_value = self._complex_er(material) if electric else self._complex_mur(material)
            material_is_constrained[material.numID] = not np.isfinite(property_value)

        cell_constraint_mask = np.zeros((u1 - u0, v1 - v0), dtype=bool)
        for n in normal_indices:
            if self.normal_axis == 0:
                ids = G.solid[n, u0:u1, v0:v1]
            elif self.normal_axis == 1:
                ids = G.solid[u0:u1, n, v0:v1]
            else:
                ids = G.solid[u0:u1, v0:v1, n]
            cell_constraint_mask |= material_is_constrained[ids]
        return cell_constraint_mask

    def _complex_er(self, material):
        omega = 2 * np.pi * self.frequency
        if hasattr(material, "calculate_er") and material.__class__.__name__ != "Material":
            er = material.calculate_er(self.frequency)
        else:
            er = material.er
            if getattr(material, "se", 0) not in [0, float("inf")]:
                er = er - 1j * material.se / (omega * config.e0)
        if getattr(material, "se", 0) == float("inf"):
            er = self.FDFD_PEC_PROPERTY
        return er

    def _complex_mur(self, material):
        omega = 2 * np.pi * self.frequency
        mur = material.mr
        if getattr(material, "sm", 0) not in [0, float("inf")]:
            mur = mur - 1j * material.sm / (omega * config.m0)
        if getattr(material, "sm", 0) == float("inf"):
            mur = self.FDFD_PMC_PROPERTY
        return mur

    def update_eigenmode_magnetic(self, iteration, G):
        """Apply magnetic-field TF/SF corrections using incident modal E."""
        time = iteration * G.dt
        if not self._source_is_active(time):
            return
        if self.broadband_e_envelopes is not None:
            self._update_broadband_magnetic(iteration, G)
            return

        updateEigenmode_magnetic[config.sim_config.dtypes["C_float_or_double"]](
            config.get_model_config().ompthreads,
            self.normal_axis,
            1 if self.direction == "+" else -1,
            self.transverse_start[0],
            self.transverse_start[1],
            self.transverse_stop[0],
            self.transverse_stop[1],
            self.plane_index,
            config.sim_config.dtypes["float_or_double"](self._waveform_value(time, G)),
            self.modal_e_real[0],
            self.modal_e_real[1],
            self.modal_e_real[2],
            G.updatecoeffsH,
            G.ID,
            G.Hx,
            G.Hy,
            G.Hz,
        )

    def update_eigenmode_electric(self, iteration, G):
        """Apply electric-field TF/SF corrections using incident modal H."""
        if self.broadband_h_envelopes is not None:
            time = iteration * G.dt
            if self._source_is_active(time):
                self._update_broadband_electric(iteration, G)
            return

        time = iteration * G.dt + self._magnetic_modal_time_offset(G)
        if not self._source_is_active(time):
            return

        updateEigenmode_electric[config.sim_config.dtypes["C_float_or_double"]](
            config.get_model_config().ompthreads,
            self.normal_axis,
            1 if self.direction == "+" else -1,
            self.transverse_start[0],
            self.transverse_start[1],
            self.transverse_stop[0],
            self.transverse_stop[1],
            self.plane_index,
            config.sim_config.dtypes["float_or_double"](self._waveform_value(time, G)),
            self.modal_h_real[0],
            self.modal_h_real[1],
            self.modal_h_real[2],
            G.updatecoeffsE,
            G.ID,
            G.Ex,
            G.Ey,
            G.Ez,
        )

    def _update_broadband_magnetic(self, iteration, G):
        """Apply the broadband incident-E temporal/modal basis expansion."""
        if iteration >= self.broadband_e_envelopes.shape[2]:
            return
        dtype = config.sim_config.dtypes["float_or_double"]
        field_bases = (
            self.broadband_modal_e_real,
            self.broadband_modal_e_imag,
        )
        for anchor in range(self.broadband_e_envelopes.shape[0]):
            for quadrature, fields in enumerate(field_bases):
                envelope = self.broadband_e_envelopes[anchor, quadrature, iteration]
                if envelope == 0:
                    continue
                modal_fields = fields[anchor]
                updateEigenmode_magnetic[config.sim_config.dtypes["C_float_or_double"]](
                    config.get_model_config().ompthreads,
                    self.normal_axis,
                    1 if self.direction == "+" else -1,
                    self.transverse_start[0],
                    self.transverse_start[1],
                    self.transverse_stop[0],
                    self.transverse_stop[1],
                    self.plane_index,
                    dtype(envelope),
                    modal_fields[0],
                    modal_fields[1],
                    modal_fields[2],
                    G.updatecoeffsH,
                    G.ID,
                    G.Hx,
                    G.Hy,
                    G.Hz,
                )

    def _update_broadband_electric(self, iteration, G):
        """Apply the broadband incident-H temporal/modal basis expansion."""
        if iteration >= self.broadband_h_envelopes.shape[2]:
            return
        dtype = config.sim_config.dtypes["float_or_double"]
        field_bases = (
            self.broadband_modal_h_real,
            self.broadband_modal_h_imag,
        )
        for anchor in range(self.broadband_h_envelopes.shape[0]):
            for quadrature, fields in enumerate(field_bases):
                envelope = self.broadband_h_envelopes[anchor, quadrature, iteration]
                if envelope == 0:
                    continue
                modal_fields = fields[anchor]
                updateEigenmode_electric[config.sim_config.dtypes["C_float_or_double"]](
                    config.get_model_config().ompthreads,
                    self.normal_axis,
                    1 if self.direction == "+" else -1,
                    self.transverse_start[0],
                    self.transverse_start[1],
                    self.transverse_stop[0],
                    self.transverse_stop[1],
                    self.plane_index,
                    dtype(envelope),
                    modal_fields[0],
                    modal_fields[1],
                    modal_fields[2],
                    G.updatecoeffsE,
                    G.ID,
                    G.Ex,
                    G.Ey,
                    G.Ez,
                )

    def _source_is_active(self, time):
        return self.start <= time <= self.stop

    def _magnetic_modal_time_offset(self, G):
        """Half-step plus half-cell propagation delay for modal H sampling."""
        neff = abs(float(np.real(self.neff)))
        return 0.5 * G.dt + neff * G.dl[self.normal_axis] / (2 * config.c)

    def _waveform_value(self, time, G):
        return self.waveform.calculate_value(time - self.start, G.dt)

    def _modal_value(self, field, u, v, time, G):
        if field is None:
            return 0.0
        local_time = time - self.start
        envelope = self.waveform.calculate_value(local_time, G.dt)
        value = field[u - self.transverse_start[0], v - self.transverse_start[1]]
        return float(envelope * np.real(value))

    def _e_incident(self, component, u, v, time, G):
        return self._modal_value(self.modal_e[component], u, v, time, G)

    def _h_incident(self, component, u, v, time, G):
        direction_scale = 1.0 if self.direction == "+" else -1.0
        return direction_scale * self._modal_value(self.modal_h[component], u, v, time, G)

    def _add_h(self, G, component, i, j, k, value):
        fields = (G.Hx, G.Hy, G.Hz)
        material = G.ID[3 + component, i, j, k]
        fields[component][i, j, k] += G.updatecoeffsH[material, self.normal_axis + 1] * value

    def _add_e(self, G, component, i, j, k, value):
        fields = (G.Ex, G.Ey, G.Ez)
        material = G.ID[component, i, j, k]
        fields[component][i, j, k] += G.updatecoeffsE[material, self.normal_axis + 1] * value

    def _update_magnetic_normal_x(self, time, G):
        u0, v0 = self.transverse_start
        u1, v1 = self.transverse_stop
        i = self.plane_index

        if self.direction == "+":
            target_i = i - 1
            for j in range(u0, u1 + 1):
                for k in range(v0, v1):
                    self._add_h(G, 1, target_i, j, k, -self._e_incident(2, j, k, time, G))
            for j in range(u0, u1):
                for k in range(v0, v1 + 1):
                    self._add_h(G, 2, target_i, j, k, self._e_incident(1, j, k, time, G))
        else:
            target_i = i
            for j in range(u0, u1 + 1):
                for k in range(v0, v1):
                    self._add_h(G, 1, target_i, j, k, self._e_incident(2, j, k, time, G))
            for j in range(u0, u1):
                for k in range(v0, v1 + 1):
                    self._add_h(G, 2, target_i, j, k, -self._e_incident(1, j, k, time, G))

    def _update_electric_normal_x(self, time, G):
        u0, v0 = self.transverse_start
        u1, v1 = self.transverse_stop
        i = self.plane_index

        if self.direction == "+":
            for j in range(u0, u1 + 1):
                for k in range(v0, v1):
                    self._add_e(G, 2, i, j, k, -self._h_incident(1, j, k, time, G))
            for j in range(u0, u1):
                for k in range(v0, v1 + 1):
                    self._add_e(G, 1, i, j, k, self._h_incident(2, j, k, time, G))
        else:
            for j in range(u0, u1 + 1):
                for k in range(v0, v1):
                    self._add_e(G, 2, i, j, k, self._h_incident(1, j, k, time, G))
            for j in range(u0, u1):
                for k in range(v0, v1 + 1):
                    self._add_e(G, 1, i, j, k, -self._h_incident(2, j, k, time, G))

    def _update_magnetic_normal_y(self, time, G):
        u0, v0 = self.transverse_start
        u1, v1 = self.transverse_stop
        j = self.plane_index

        if self.direction == "+":
            target_j = j - 1
            for i in range(u0, u1 + 1):
                for k in range(v0, v1):
                    self._add_h(G, 0, i, target_j, k, self._e_incident(2, i, k, time, G))
            for i in range(u0, u1):
                for k in range(v0, v1 + 1):
                    self._add_h(G, 2, i, target_j, k, -self._e_incident(0, i, k, time, G))
        else:
            target_j = j
            for i in range(u0, u1 + 1):
                for k in range(v0, v1):
                    self._add_h(G, 0, i, target_j, k, -self._e_incident(2, i, k, time, G))
            for i in range(u0, u1):
                for k in range(v0, v1 + 1):
                    self._add_h(G, 2, i, target_j, k, self._e_incident(0, i, k, time, G))

    def _update_electric_normal_y(self, time, G):
        u0, v0 = self.transverse_start
        u1, v1 = self.transverse_stop
        j = self.plane_index

        if self.direction == "+":
            for i in range(u0, u1 + 1):
                for k in range(v0, v1):
                    self._add_e(G, 2, i, j, k, self._h_incident(0, i, k, time, G))
            for i in range(u0, u1):
                for k in range(v0, v1 + 1):
                    self._add_e(G, 0, i, j, k, -self._h_incident(2, i, k, time, G))
        else:
            for i in range(u0, u1 + 1):
                for k in range(v0, v1):
                    self._add_e(G, 2, i, j, k, -self._h_incident(0, i, k, time, G))
            for i in range(u0, u1):
                for k in range(v0, v1 + 1):
                    self._add_e(G, 0, i, j, k, self._h_incident(2, i, k, time, G))

    def _update_magnetic_normal_z(self, time, G):
        u0, v0 = self.transverse_start
        u1, v1 = self.transverse_stop
        k = self.plane_index

        if self.direction == "+":
            target_k = k - 1
            for i in range(u0, u1):
                for j in range(v0, v1 + 1):
                    self._add_h(G, 1, i, j, target_k, self._e_incident(0, i, j, time, G))
            for i in range(u0, u1 + 1):
                for j in range(v0, v1):
                    self._add_h(G, 0, i, j, target_k, -self._e_incident(1, i, j, time, G))
        else:
            target_k = k
            for i in range(u0, u1):
                for j in range(v0, v1 + 1):
                    self._add_h(G, 1, i, j, target_k, -self._e_incident(0, i, j, time, G))
            for i in range(u0, u1 + 1):
                for j in range(v0, v1):
                    self._add_h(G, 0, i, j, target_k, self._e_incident(1, i, j, time, G))

    def _update_electric_normal_z(self, time, G):
        u0, v0 = self.transverse_start
        u1, v1 = self.transverse_stop
        k = self.plane_index

        if self.direction == "+":
            for i in range(u0, u1 + 1):
                for j in range(v0, v1):
                    self._add_e(G, 1, i, j, k, -self._h_incident(0, i, j, time, G))
            for i in range(u0, u1):
                for j in range(v0, v1 + 1):
                    self._add_e(G, 0, i, j, k, self._h_incident(1, i, j, time, G))
        else:
            for i in range(u0, u1 + 1):
                for j in range(v0, v1):
                    self._add_e(G, 1, i, j, k, self._h_incident(0, i, j, time, G))
            for i in range(u0, u1):
                for j in range(v0, v1 + 1):
                    self._add_e(G, 0, i, j, k, -self._h_incident(1, i, j, time, G))


class EigenmodeReceiver(EigenmodeSource):
    """Passive multi-mode plane monitor backed by the FDFD mode solver."""

    def __init__(self, G):
        super().__init__(G)
        self.mode_indices = ()

    def grid_init(self, G):
        frequencies = tuple(self.frequencies or (self.frequency,))
        mode_indices = tuple(self.mode_indices)
        if not mode_indices:
            raise ValueError("An eigenmode receiver requires at least one mode index.")
        self.mode_index = max(mode_indices)
        anchor_e = []
        anchor_h = []
        anchor_neff = []
        solvers = []

        for frequency in frequencies:
            self.frequency = frequency
            self._extract_frequency_dependent_materials(G)
            self._solve_eigenmode(G)
            frequency_e = []
            frequency_h = []
            frequency_neff = []
            for mode_index in mode_indices:
                electric, magnetic, neff = self._fields_from_solver_mode(self.mode_solver, mode_index)
                frequency_e.append(electric)
                frequency_h.append(magnetic)
                frequency_neff.append(neff)
            anchor_e.append(frequency_e)
            anchor_h.append(frequency_h)
            anchor_neff.append(frequency_neff)
            solvers.append(self.mode_solver)

        if len(frequencies) > 1 and self.anchor_policy == 'auto':
            for mode_position, mode_index in enumerate(mode_indices):
                for frequency_index in range(1, len(frequencies)):
                    overlap = self._modal_overlap(
                        anchor_e[frequency_index - 1][mode_position],
                        anchor_h[frequency_index - 1][mode_position],
                        anchor_e[frequency_index][mode_position],
                        anchor_h[frequency_index][mode_position],
                    )
                    magnitude = float(abs(overlap))
                    if (
                        not np.isfinite(magnitude)
                        or magnitude < self.ANCHOR_OVERLAP_ERROR_THRESHOLD
                    ):
                        frequency = float(self.fallback_frequency)
                        logger.warning(
                            f'Automatic anchor tracking failed for eigenmode port '
                            f'{self.port_index}, mode {mode_index}, between '
                            f'{frequencies[frequency_index - 1]:g} and '
                            f'{frequencies[frequency_index]:g} Hz with overlap '
                            f'{magnitude:.6f}. This may indicate a mode crossing or '
                            f'degeneracy. The port will use a single modal anchor at '
                            f'{frequency:g} Hz. Its modal decomposition and '
                            'S-parameters may be inaccurate toward frequencies far '
                            'from this anchor.'
                        )
                        self.frequency = frequency
                        self.frequencies = (frequency,)
                        self.mode_solvers = None
                        return self.grid_init(G)

        for mode_position, mode_index in enumerate(mode_indices):
            for frequency_index in range(1, len(frequencies)):
                overlap = self._modal_overlap(
                    anchor_e[frequency_index - 1][mode_position],
                    anchor_h[frequency_index - 1][mode_position],
                    anchor_e[frequency_index][mode_position],
                    anchor_h[frequency_index][mode_position],
                )
                magnitude = float(abs(overlap))
                self._check_anchor_overlap(
                    magnitude,
                    frequencies[frequency_index - 1],
                    frequencies[frequency_index],
                    mode_index,
                    "Broadband eigenmode receiver",
                )
                if np.isfinite(magnitude) and magnitude > 1e-300:
                    factor = np.exp(-1j * np.angle(overlap))
                    anchor_e[frequency_index][mode_position] = [
                        field * factor for field in anchor_e[frequency_index][mode_position]
                    ]
                    anchor_h[frequency_index][mode_position] = [
                        field * factor for field in anchor_h[frequency_index][mode_position]
                    ]

        from gprMax.eigenmode_ports import EigenmodePortMonitor

        monitor = EigenmodePortMonitor(
            owner=self,
            port_index=self.port_index,
            port_id=self.port_id,
            is_source=False,
            excitation_mode_index=None,
            mode_indices=mode_indices,
            anchor_frequencies=frequencies,
            anchor_e=anchor_e,
            anchor_h=anchor_h,
            anchor_neff=np.asarray(anchor_neff, dtype=np.complex128),
            dft_start=self.dft_start,
            dft_stop=self.dft_stop,
            dft_points=self.dft_points,
        )
        monitor.prepare(G)
        self.port_monitor = monitor
        G.eigenmodeports.append(monitor)
        self.mode_solvers = solvers
        self._plot_eigenmode_fields()


class VoltageSource(Source):
    """A voltage source can be a hard source if it's resistance is zero,
    i.e. the time variation of the specified electric field component
    is prescribed. If it's resistance is non-zero it behaves as a resistive
    voltage source.
    """

    def __init__(self):
        super().__init__()
        self.resistance = None
        # Wave-reference impedance used by a coincident RxPort. For a
        # finite-resistance source it is the Thevenin resistance; for a hard
        # source it defaults to 50 Ohms unless explicitly overridden.
        self.reference_impedance = None
        # Preserved when create_material() replaces the selected electric-edge
        # material with a copy carrying the source resistance. Port outputs
        # need the pre-source values to remove the numerical Yee-gap
        # capacitance/conductance without accidentally including 1/R.
        self.background_material_numID = None
        self.background_material_ID = None
        self.background_material_type = None
        self.background_er = None
        self.background_se = None
        self.background_mr = None
        self.background_sm = None
        self.background_is_dispersive = False
        self.source_material_numID = None

    def calculate_waveform_values(self, G):
        """Calculates all waveform values for source for duration of simulation.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        # Check if a source matches existing source in terms of waveform and
        # does not have a customised start/stop time. If so, use its
        # pre-calculated waveform values, otherwise calculate them.
        src_match = False

        if self.start == 0 and self.stop == G.timewindow:
            for src in G.voltagesources:
                if src.waveformID == self.waveformID:
                    src_match = True
                    self.waveformvalues_halfdt = src.waveformvalues_halfdt
                    self.waveformvalues_wholedt = src.waveformvalues_wholedt

        if not src_match:
            waveform = next(x for x in G.waveforms if x.ID == self.waveformID)
            self.waveformvalues_halfdt = np.zeros((G.iterations + 1), dtype=config.sim_config.dtypes["float_or_double"])
            self.waveformvalues_wholedt = np.zeros(
                (G.iterations + 1), dtype=config.sim_config.dtypes["float_or_double"]
            )

            for iteration in range(G.iterations + 1):
                time = G.dt * iteration
                if time >= self.start and time <= self.stop:
                    # Set the time of the waveform evaluation to account for any
                    # delay in the start
                    time -= self.start
                    self.waveformvalues_halfdt[iteration] = waveform.calculate_value(time + 0.5 * G.dt, G.dt)
                    self.waveformvalues_wholedt[iteration] = waveform.calculate_value(time, G.dt)

    def update_electric(self, iteration, updatecoeffsE, ID, Ex, Ey, Ez, G):
        """Updates electric field values for a voltage source.

        Args:
            iteration: int of current iteration (timestep).
            updatecoeffsE: memory view of array of electric field update
                            coefficients.
            ID: memory view of array of numeric IDs corresponding to materials
                in the model.
            Ex, Ey, Ez: memory view of array of electric field values.
            G: FDTDGrid class describing a grid in a model.
        """

        if iteration * G.dt >= self.start and iteration * G.dt <= self.stop:
            i = self.xcoord
            j = self.ycoord
            k = self.zcoord
            componentID = f"E{self.polarisation}"

            if self.polarisation == "x":
                if self.resistance != 0:
                    Ex[i, j, k] -= (
                        updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4]
                        * self.waveformvalues_halfdt[iteration]
                        * (1 / (self.resistance * G.dy * G.dz))
                    )
                else:
                    Ex[i, j, k] = -1 * self.waveformvalues_wholedt[iteration] / G.dx

            elif self.polarisation == "y":
                if self.resistance != 0:
                    Ey[i, j, k] -= (
                        updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4]
                        * self.waveformvalues_halfdt[iteration]
                        * (1 / (self.resistance * G.dx * G.dz))
                    )
                else:
                    Ey[i, j, k] = -1 * self.waveformvalues_wholedt[iteration] / G.dy

            elif self.polarisation == "z":
                if self.resistance != 0:
                    Ez[i, j, k] -= (
                        updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4]
                        * self.waveformvalues_halfdt[iteration]
                        * (1 / (self.resistance * G.dx * G.dy))
                    )
                else:
                    Ez[i, j, k] = -1 * self.waveformvalues_wholedt[iteration] / G.dz

    def create_material(self, G):
        """Create a new material at the voltage source location that adds the
            voltage source conductivity to the underlying parameters.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        if self.resistance == 0:
            return
        i = self.xcoord
        j = self.ycoord
        k = self.zcoord

        componentID = f"E{self.polarisation}"
        requirednumID = G.ID[G.IDlookup[componentID], i, j, k]
        material = next(x for x in G.materials if x.numID == requirednumID)
        self.background_material_numID = int(material.numID)
        self.background_material_ID = str(material.ID)
        self.background_material_type = str(material.type)
        self.background_er = float(material.er)
        self.background_se = float(material.se)
        self.background_mr = float(material.mr)
        self.background_sm = float(material.sm)
        self.background_is_dispersive = hasattr(material, "poles")
        newmaterial = deepcopy(material)
        newmaterial.ID = f"{material.ID}+{self.ID}"
        newmaterial.numID = len(G.materials)
        newmaterial.averagable = False
        newmaterial.type += ",\nvoltage-source" if newmaterial.type else "voltage-source"

        # Add conductivity of voltage source to underlying conductivity
        if self.polarisation == "x":
            newmaterial.se += G.dx / (self.resistance * G.dy * G.dz)
        elif self.polarisation == "y":
            newmaterial.se += G.dy / (self.resistance * G.dx * G.dz)
        elif self.polarisation == "z":
            newmaterial.se += G.dz / (self.resistance * G.dx * G.dy)

        G.ID[G.IDlookup[componentID], i, j, k] = newmaterial.numID
        G.materials.append(newmaterial)
        self.source_material_numID = int(newmaterial.numID)


class HertzianDipole(Source):
    """A Hertzian dipole is an additive source (electric current density)."""

    def __init__(self):
        super().__init__()
        self.dl = 0.0

    def calculate_waveform_values(self, G):
        """Calculates all waveform values for source for duration of simulation.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        # Check if a source matches existing source in terms of waveform and
        # does not have a customised start/stop time. If so, use its
        # pre-calculated waveform values, otherwise calculate them.
        src_match = False

        if self.start == 0 and self.stop == G.timewindow:
            for src in G.hertziandipoles:
                if src.waveformID == self.waveformID:
                    src_match = True
                    self.waveformvalues_halfdt = src.waveformvalues_halfdt

        if not src_match:
            waveform = next(x for x in G.waveforms if x.ID == self.waveformID)
            self.waveformvalues_halfdt = np.zeros((G.iterations + 1), dtype=config.sim_config.dtypes["float_or_double"])

            for iteration in range(G.iterations + 1):
                time = G.dt * iteration
                if time >= self.start and time <= self.stop:
                    # Set the time of the waveform evaluation to account for any
                    # delay in the start
                    time -= self.start
                    self.waveformvalues_halfdt[iteration] = waveform.calculate_value(time + 0.5 * G.dt, G.dt)

    def update_electric(self, iteration, updatecoeffsE, ID, Ex, Ey, Ez, G):
        """Updates electric field values for a Hertzian dipole.

        Args:
            iteration: int of current iteration (timestep).
            updatecoeffsE: memory view of array of electric field update
                            coefficients.
            ID: memory view of array of numeric IDs corresponding to materials
                in the model.
            Ex, Ey, Ez: memory view of array of electric field values.
            G: FDTDGrid class describing a grid in a model.
        """

        if iteration * G.dt >= self.start and iteration * G.dt <= self.stop:
            i = self.xcoord
            j = self.ycoord
            k = self.zcoord
            componentID = f"E{self.polarisation}"
            if self.polarisation == "x":
                Ex[i, j, k] -= (
                    updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4]
                    * self.waveformvalues_halfdt[iteration]
                    * self.dl
                    * (1 / (G.dx * G.dy * G.dz))
                )

            elif self.polarisation == "y":
                Ey[i, j, k] -= (
                    updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4]
                    * self.waveformvalues_halfdt[iteration]
                    * self.dl
                    * (1 / (G.dx * G.dy * G.dz))
                )

            elif self.polarisation == "z":
                Ez[i, j, k] -= (
                    updatecoeffsE[ID[G.IDlookup[componentID], i, j, k], 4]
                    * self.waveformvalues_halfdt[iteration]
                    * self.dl
                    * (1 / (G.dx * G.dy * G.dz))
                )


class MagneticDipole(Source):
    """A magnetic dipole is an additive source (magnetic current density)."""

    def calculate_waveform_values(self, G):
        """Calculates all waveform values for source for duration of simulation.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        # Check if a source matches existing source in terms of waveform and
        # does not have a customised start/stop time. If so, use its
        # pre-calculated waveform values, otherwise calculate them.
        src_match = False

        if self.start == 0 and self.stop == G.timewindow:
            for src in G.magneticdipoles:
                if src.waveformID == self.waveformID:
                    src_match = True
                    self.waveformvalues_wholedt = src.waveformvalues_wholedt

        if not src_match:
            waveform = next(x for x in G.waveforms if x.ID == self.waveformID)
            self.waveformvalues_wholedt = np.zeros(
                (G.iterations + 1), dtype=config.sim_config.dtypes["float_or_double"]
            )

            for iteration in range(G.iterations + 1):
                time = G.dt * iteration
                if time >= self.start and time <= self.stop:
                    # Set the time of the waveform evaluation to account for any
                    # delay in the start
                    time -= self.start
                    self.waveformvalues_wholedt[iteration] = waveform.calculate_value(time, G.dt)

    def update_magnetic(self, iteration, updatecoeffsH, ID, Hx, Hy, Hz, G):
        """Updates magnetic field values for a magnetic dipole.

        Args:
            iteration: int of current iteration (timestep).
            updatecoeffsH: memory view of array of magnetic field update
                            coefficients.
            ID: memory view of array of numeric IDs corresponding to materials
                in the model.
            Hx, Hy, Hz: memory view of array of magnetic field values.
            G: FDTDGrid class describing a grid in a model.
        """

        if iteration * G.dt >= self.start and iteration * G.dt <= self.stop:
            i = self.xcoord
            j = self.ycoord
            k = self.zcoord
            componentID = f"H{self.polarisation}"

            if self.polarisation == "x":
                Hx[i, j, k] -= (
                    updatecoeffsH[ID[G.IDlookup[componentID], i, j, k], 4]
                    * self.waveformvalues_wholedt[iteration]
                    * (1 / (G.dx * G.dy * G.dz))
                )

            elif self.polarisation == "y":
                Hy[i, j, k] -= (
                    updatecoeffsH[ID[G.IDlookup[componentID], i, j, k], 4]
                    * self.waveformvalues_wholedt[iteration]
                    * (1 / (G.dx * G.dy * G.dz))
                )

            elif self.polarisation == "z":
                Hz[i, j, k] -= (
                    updatecoeffsH[ID[G.IDlookup[componentID], i, j, k], 4]
                    * self.waveformvalues_wholedt[iteration]
                    * (1 / (G.dx * G.dy * G.dz))
                )


def htod_src_arrays(sources, G, queue=None):
    """Initialise arrays on compute device for source coordinates/polarisation,
        other source information, and source waveform values.

    Args:
        sources: list of sources of one type, e.g. HertzianDipole
        G: FDTDGrid class describing a grid in a model.
        queue: pyopencl queue.

    Returns:
        srcinfo1_dev: int array of source cell coordinates and polarisation
                        information.
        srcinfo2_dev: float array of other source information, e.g. length,
                        resistance etc...
        srcwaves_dev: float array of source waveform values.
    """

    srcinfo1 = np.zeros((len(sources), 4), dtype=np.int32)
    srcinfo2 = np.zeros((len(sources)), dtype=config.sim_config.dtypes["float_or_double"])
    srcwaves = np.zeros((len(sources), G.iterations + 1), dtype=config.sim_config.dtypes["float_or_double"])
    for i, src in enumerate(sources):
        srcinfo1[i, 0] = src.xcoord
        srcinfo1[i, 1] = src.ycoord
        srcinfo1[i, 2] = src.zcoord

        if src.polarisation == "x":
            srcinfo1[i, 3] = 0
        elif src.polarisation == "y":
            srcinfo1[i, 3] = 1
        elif src.polarisation == "z":
            srcinfo1[i, 3] = 2

        if src.__class__.__name__ == "HertzianDipole":
            srcinfo2[i] = src.dl
            srcwaves[i, :] = src.waveformvalues_halfdt
        elif src.__class__.__name__ == "VoltageSource":
            if src.resistance:
                srcinfo2[i] = src.resistance
                srcwaves[i, :] = src.waveformvalues_halfdt
            else:
                srcinfo2[i] = 0
                srcwaves[i, :] = src.waveformvalues_wholedt
        elif src.__class__.__name__ == "MagneticDipole":
            srcwaves[i, :] = src.waveformvalues_wholedt

    # Copy arrays to compute device
    if config.sim_config.general["solver"] == "cuda":
        import pycuda.gpuarray as gpuarray

        srcinfo1_dev = gpuarray.to_gpu(srcinfo1)
        srcinfo2_dev = gpuarray.to_gpu(srcinfo2)
        srcwaves_dev = gpuarray.to_gpu(srcwaves)
    elif config.sim_config.general["solver"] == "opencl":
        import pyopencl.array as clarray

        srcinfo1_dev = clarray.to_device(queue, srcinfo1)
        srcinfo2_dev = clarray.to_device(queue, srcinfo2)
        srcwaves_dev = clarray.to_device(queue, srcwaves)
    elif config.sim_config.general["solver"] == "metal":
        # Metal doesn't use a queue parameter, need to get device from config
        dev = config.get_model_config().device["dev"]
        srcinfo1_dev = dev.newBufferWithBytes_length_options_(srcinfo1.tobytes(), srcinfo1.nbytes, 0)
        srcinfo2_dev = dev.newBufferWithBytes_length_options_(srcinfo2.tobytes(), srcinfo2.nbytes, 0)
        srcwaves_dev = dev.newBufferWithBytes_length_options_(srcwaves.tobytes(), srcwaves.nbytes, 0)

    return srcinfo1_dev, srcinfo2_dev, srcwaves_dev


def transmission_line_host_arrays(transmissionlines, G):
    """Pack transmission-line state into backend-neutral contiguous arrays.

    The coupled part of every transmission line is normally very short, but
    each line owns mutable voltage, current, and ABC state. Keeping this state
    in flat arrays lets one device work-item advance one complete line without
    any host interaction during the FDTD time loop.

    Args:
        transmissionlines: transmission-line sources attached to a grid.
        G: FDTDGrid containing the sources.

    Returns:
        Dictionary of NumPy arrays ready to copy to a compute device.
    """

    real = config.sim_config.dtypes["float_or_double"]
    ntl = len(transmissionlines)
    niterations = G.iterations + 1
    int32_max = np.iinfo(np.int32).max

    # Columns are x, y, z, polarisation, state offset, number of active line
    # cells, source position, antenna position, first active iteration, and
    # last active iteration. Keep this layout in sync with knl_transmission_line.py.
    info = np.zeros((ntl, 10), dtype=np.int32)
    resistance = np.zeros(ntl, dtype=real)
    abcv0 = np.zeros(ntl, dtype=real)
    abcv1 = np.zeros(ntl, dtype=real)
    waveform_whole = np.zeros((ntl, niterations), dtype=real)
    waveform_half = np.zeros((ntl, niterations), dtype=real)
    Vtotal = np.zeros((ntl, niterations), dtype=real)
    Itotal = np.zeros((ntl, niterations), dtype=real)

    nstate = sum(int(tl.nl) for tl in transmissionlines)
    if nstate > int32_max or ntl * niterations > int32_max:
        raise ValueError("Transmission-line device arrays exceed the signed 32-bit index range.")

    voltage = np.zeros(nstate, dtype=real)
    current = np.zeros(nstate, dtype=real)
    times = G.dt * np.arange(G.iterations, dtype=np.float64)
    used_ports = set()
    offset = 0

    for i, tl in enumerate(transmissionlines):
        port = (int(tl.xcoord), int(tl.ycoord), int(tl.zcoord), tl.polarisation)
        if port in used_ports:
            raise ValueError(
                "More than one transmission line is attached to the same Yee "
                f"electric-field edge at {port[:3]} with {port[3]} polarisation."
            )
        used_ports.add(port)

        if tl.nl <= tl.antpos or tl.srcpos <= 0 or tl.srcpos >= tl.nl:
            raise ValueError(f"Invalid internal transmission-line layout for {tl.ID}.")

        if tl.polarisation == "x":
            polarisation = 0
        elif tl.polarisation == "y":
            polarisation = 1
        elif tl.polarisation == "z":
            polarisation = 2
        else:
            raise ValueError(f"Invalid transmission-line polarisation {tl.polarisation!r}.")

        active = np.flatnonzero((times >= tl.start) & (times <= tl.stop))
        if active.size:
            first_active = int(active[0])
            last_active = int(active[-1])
        else:
            first_active = 1
            last_active = 0

        info[i, :] = (
            tl.xcoord,
            tl.ycoord,
            tl.zcoord,
            polarisation,
            offset,
            tl.nl,
            tl.srcpos,
            tl.antpos,
            first_active,
            last_active,
        )
        resistance[i] = tl.resistance
        abcv0[i] = tl.abcv0
        abcv1[i] = tl.abcv1
        voltage[offset : offset + tl.nl] = tl.voltage[: tl.nl]
        current[offset : offset + tl.nl] = tl.current[: tl.nl]
        waveform_whole[i, :] = tl.waveformvalues_wholedt
        waveform_half[i, :] = tl.waveformvalues_halfdt
        Vtotal[i, :] = tl.Vtotal
        Itotal[i, :] = tl.Itotal
        offset += tl.nl

    return {
        "info": info,
        "resistance": resistance,
        "abcv0": abcv0,
        "abcv1": abcv1,
        "voltage": voltage,
        "current": current,
        "waveform_whole": waveform_whole,
        "waveform_half": waveform_half,
        "Vtotal": Vtotal,
        "Itotal": Itotal,
    }


def htod_transmission_line_arrays(transmissionlines, G, queue=None):
    """Copy packed transmission-line arrays to the active compute device."""

    arrays = transmission_line_host_arrays(transmissionlines, G)
    solver = config.sim_config.general["solver"]

    if solver == "cuda":
        import pycuda.gpuarray as gpuarray

        return {name: gpuarray.to_gpu(array) for name, array in arrays.items()}
    if solver == "opencl":
        import pyopencl.array as clarray

        return {name: clarray.to_device(queue, array) for name, array in arrays.items()}
    if solver == "metal":
        dev = config.get_model_config().device["dev"]
        return {
            name: dev.newBufferWithBytes_length_options_(array.tobytes(), array.nbytes, 0)
            for name, array in arrays.items()
        }

    raise ValueError(f"Unknown device solver {solver!r} for transmission-line arrays.")


def dtoh_transmission_line_outputs(Vtotal, Itotal, G):
    """Copy device terminal histories into their transmission-line objects."""

    expected = (len(G.transmissionlines), G.iterations + 1)
    if Vtotal.shape != expected or Itotal.shape != expected:
        raise ValueError(
            "Transmission-line device output shape does not match the grid: "
            f"expected {expected}, got {Vtotal.shape} and {Itotal.shape}."
        )

    for i, tl in enumerate(G.transmissionlines):
        np.copyto(tl.Vtotal, Vtotal[i, :], casting="same_kind")
        np.copyto(tl.Itotal, Itotal[i, :], casting="same_kind")


class TransmissionLine(Source):
    """A transmission line source is a one-dimensional transmission line
    which is attached virtually to a grid cell.
    """

    def __init__(self, iterations: int, dt: float):
        """
        Args:
            iterations: number of iterations
            dt: time step of the grid
        """

        super().__init__()
        self.resistance = None
        self.iterations = iterations

        # Coefficients for ABC termination of end of the transmission line
        self.abcv0 = 0
        self.abcv1 = 0

        # Spatial step of transmission line (N.B if the magic time step is
        # used it results in instabilities for certain impedances)
        self.dl = np.sqrt(3) * config.c * dt

        # Number of cells in the transmission line (initially a long line to
        # calculate incident voltage and current); consider putting ABCs/PML at end
        self.nl = round_value(0.667 * self.iterations)

        # Cell position of the one-way injector excitation in the transmission line
        self.srcpos = 5

        # Cell position of where line connects to antenna/main grid
        self.antpos = 10

        self.voltage = np.zeros(self.nl, dtype=config.sim_config.dtypes["float_or_double"])
        self.current = np.zeros(self.nl, dtype=config.sim_config.dtypes["float_or_double"])
        self.Vinc = np.zeros(self.iterations + 1, dtype=config.sim_config.dtypes["float_or_double"])
        self.Iinc = np.zeros(self.iterations + 1, dtype=config.sim_config.dtypes["float_or_double"])
        self.Vtotal = np.zeros(self.iterations + 1, dtype=config.sim_config.dtypes["float_or_double"])
        self.Itotal = np.zeros(self.iterations + 1, dtype=config.sim_config.dtypes["float_or_double"])
        # Bound after the grid materials and time axis have been finalised.
        # Kept here rather than in the update loop because all spectral port
        # processing is post-solve.
        self.port_output = None

    def calculate_waveform_values(self, G):
        """Calculates all waveform values for source for duration of simulation.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        # Check if a source matches existing source in terms of waveform and
        # does not have a customised start/stop time. If so, use its
        # pre-calculated waveform values, otherwise calculate them.
        src_match = False

        if self.start == 0 and self.stop == G.timewindow:
            for src in G.transmissionlines:
                if src.waveformID == self.waveformID:
                    src_match = True
                    self.waveformvalues_wholedt = src.waveformvalues_wholedt
                    self.waveformvalues_halfdt = src.waveformvalues_halfdt

        if not src_match:
            waveform = next(x for x in G.waveforms if x.ID == self.waveformID)
            self.waveformvalues_wholedt = np.zeros(
                (G.iterations + 1), dtype=config.sim_config.dtypes["float_or_double"]
            )
            self.waveformvalues_halfdt = np.zeros((G.iterations + 1), dtype=config.sim_config.dtypes["float_or_double"])

            for iteration in range(G.iterations + 1):
                time = G.dt * iteration
                if time >= self.start and time <= self.stop:
                    # Set the time of the waveform evaluation to account for any
                    # delay in the start
                    time -= self.start
                    self.waveformvalues_wholedt[iteration] = waveform.calculate_value(time, G.dt)
                    self.waveformvalues_halfdt[iteration] = waveform.calculate_value(time + 0.5 * G.dt, G.dt)

    def calculate_incident_V_I(self, G):
        """Calculates the incident voltage and current with a long length
            transmission line not connected to the main grid
            from: http://dx.doi.org/10.1002/mop.10415

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        # The preliminary incident-wave calculation and the coupled FDTD run
        # use separate output histories, but they advance the same internal
        # line voltage/current vectors and ABC memories. Always start the
        # preliminary line from rest and clear any old incident histories.
        self._reset_update_state()
        self.Vinc.fill(0)
        self.Iinc.fill(0)

        for iteration in range(self.iterations):
            self.Iinc[iteration] = self.current[self.antpos]
            self.Vinc[iteration] = self.voltage[self.antpos]
            self.update_current(iteration, G)
            self.update_voltage(iteration, G)

        # Shorten number of cells in the transmission line before use with main grid
        self.nl = self.antpos + 1

        # Vinc/Iinc retain the completed incident histories, but the actual
        # coupled source must begin with zero line fields and zero ABC memory.
        self._reset_update_state()

    def _reset_update_state(self):
        """Reset mutable line and absorbing-boundary state to rest."""

        self.voltage.fill(0)
        self.current.fill(0)
        self.abcv0 = 0
        self.abcv1 = 0

    def update_abc(self, G):
        """Updates absorbing boundary condition at end of the transmission line.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        h = (config.c * G.dt - self.dl) / (config.c * G.dt + self.dl)

        self.voltage[0] = h * (self.voltage[1] - self.abcv0) + self.abcv1
        self.abcv0 = self.voltage[0]
        self.abcv1 = self.voltage[1]

    def update_voltage(self, iteration, G):
        """Updates voltage values along the transmission line.

        Args:
            iteration: int of current iteration (timestep).
            G: FDTDGrid class describing a grid in a model.
        """

        # Update all the voltage values along the line
        self.voltage[1 : self.nl] -= (
            self.resistance * (config.c * G.dt / self.dl) * (self.current[1 : self.nl] - self.current[0 : self.nl - 1])
        )

        # Update the voltage at the position of the one-way injector excitation
        self.voltage[self.srcpos] += (config.c * G.dt / self.dl) * self.waveformvalues_wholedt[iteration]

        # Update ABC before updating current
        self.update_abc(G)

    def update_current(self, iteration, G):
        """Updates current values along the transmission line.

        Args:
            iteration: int of current iteration (timestep).
            G: FDTDGrid class describing a grid in a model.
        """

        # Update all the current values along the line
        self.current[0 : self.nl - 1] -= (
            (1 / self.resistance)
            * (config.c * G.dt / self.dl)
            * (self.voltage[1 : self.nl] - self.voltage[0 : self.nl - 1])
        )

        # Update the current one cell before the position of the one-way injector excitation
        self.current[self.srcpos - 1] += (
            (1 / self.resistance) * (config.c * G.dt / self.dl) * self.waveformvalues_halfdt[iteration]
        )

    def update_electric(self, iteration, updatecoeffsE, ID, Ex, Ey, Ez, G):
        """Updates electric field value in the main grid from voltage value in
            the transmission line.

        Args:
            iteration: int of current iteration (timestep).
            updatecoeffsE: memory view of array of electric field update
                            coefficients.
            ID: memory view of array of numeric IDs corresponding to materials
                in the model.
            Ex, Ey, Ez: memory view of array of electric field values.
            G: FDTDGrid class describing a grid in a model.
        """

        if iteration * G.dt >= self.start and iteration * G.dt <= self.stop:
            i = self.xcoord
            j = self.ycoord
            k = self.zcoord

            self.update_voltage(iteration, G)

            if self.polarisation == "x":
                Ex[i, j, k] = -self.voltage[self.antpos] / G.dx

            elif self.polarisation == "y":
                Ey[i, j, k] = -self.voltage[self.antpos] / G.dy

            elif self.polarisation == "z":
                Ez[i, j, k] = -self.voltage[self.antpos] / G.dz

    # TODO: Add type information (if can avoid circular dependency)
    def update_magnetic(self, iteration, updatecoeffsH, ID, Hx, Hy, Hz, G):
        """Updates current value in transmission line from magnetic field values
            in the main grid.

        Args:
            iteration: int of current iteration (timestep).
            updatecoeffsH: memory view of array of magnetic field update
                            coefficients.
            ID: memory view of array of numeric IDs corresponding to materials
                in the model.
            Hx, Hy, Hz: memory view of array of magnetic field values.
            G: FDTDGrid class describing a grid in a model.
        """

        if iteration * G.dt >= self.start and iteration * G.dt <= self.stop:
            i = self.xcoord
            j = self.ycoord
            k = self.zcoord

            if self.polarisation == "x":
                self.current[self.antpos] = G.calculate_Ix(i, j, k)

            elif self.polarisation == "y":
                self.current[self.antpos] = G.calculate_Iy(i, j, k)

            elif self.polarisation == "z":
                self.current[self.antpos] = G.calculate_Iz(i, j, k)

            self.update_current(iteration, G)


MAGNETIC_FRILL_MAX_TERMS = 4


def magnetic_frill_source_host_arrays(magneticfrillsources, G):
    """Pack the corrected Hyun feed recurrence into contiguous arrays.

    ``MagneticFrillSource.finalise_setup()`` has already validated the attached
    thin wire and reduced its Cartesian feed stencil to at most four terms.
    Pack those terms directly instead of reconstructing geometry in a device
    kernel. This preserves the CPU path's Mäkinen ``k_H`` projection, Hyun
    ``F`` factor, anisotropic cell dimensions, and PMC image completion.

    Args:
        magneticfrillsources: magnetic-frill sources attached to a grid.
        G: FDTDGrid containing the sources.

    Returns:
        Dictionary of NumPy arrays ready to copy to a compute device.
    """

    real = config.sim_config.dtypes["float_or_double"]
    nfrill = len(magneticfrillsources)
    niterations = G.iterations + 1

    int32_max = np.iinfo(np.int32).max
    flattened_sizes = (
        nfrill * niterations,
        nfrill * MAGNETIC_FRILL_MAX_TERMS * 4,
        nfrill * MAGNETIC_FRILL_MAX_TERMS * 2,
        nfrill * 3,
    )
    if any(size > int32_max for size in flattened_sizes):
        raise ValueError("Magnetic-frill device arrays exceed the signed 32-bit index range.")

    # term_info columns are H component (0=Hx, 1=Hy, 2=Hz), x, y, z.
    # term_params columns are Ampere-loop current weight and the complete
    # magnetic-source gain. Keep these layouts in sync with
    # knl_magnetic_frill_source.py.
    term_counts = np.zeros(nfrill, dtype=np.int32)
    term_info = np.zeros((nfrill, MAGNETIC_FRILL_MAX_TERMS, 4), dtype=np.int32)
    term_params = np.zeros((nfrill, MAGNETIC_FRILL_MAX_TERMS, 2), dtype=real)
    # Per-source parameters are Z0, feed-cell self-admittance G, and the
    # current-centering theta. The previous half-step current is mutable state.
    params = np.zeros((nfrill, 3), dtype=real)
    state = np.zeros(nfrill, dtype=real)
    waveform = np.zeros((nfrill, niterations), dtype=real)
    Vinc = np.zeros((nfrill, niterations), dtype=real)
    Vtotal = np.zeros((nfrill, niterations), dtype=real)
    Itot = np.zeros((nfrill, niterations), dtype=real)

    for i, frill in enumerate(magneticfrillsources):
        nterms = len(frill._drive_terms)
        if not 1 <= nterms <= MAGNETIC_FRILL_MAX_TERMS:
            raise ValueError(
                f"{frill.ID} has {nterms} magnetic feed terms; expected between " f"1 and {MAGNETIC_FRILL_MAX_TERMS}."
            )

        term_counts[i] = nterms
        for term_index, term in enumerate(frill._drive_terms):
            component, x, y, z, current_weight, source_gain = term
            term_info[i, term_index, :] = (
                {"Hx": 0, "Hy": 1, "Hz": 2}[component],
                x,
                y,
                z,
            )
            term_params[i, term_index, :] = current_weight, source_gain

        params[i, :] = frill.Z0, frill._G_coeff, frill._theta
        state[i] = frill._previous_half_current
        waveform[i, :] = frill.waveformvalues_wholedt
        Vinc[i, :] = frill.Vinc
        Vtotal[i, :] = frill.Vtotal
        Itot[i, :] = frill.Itot

    return {
        "term_counts": term_counts,
        "term_info": term_info,
        "term_params": term_params,
        "params": params,
        "state": state,
        "waveform": waveform,
        "Vinc": Vinc,
        "Vtotal": Vtotal,
        "Itot": Itot,
    }


def htod_magnetic_frill_source_arrays(magneticfrillsources, G, queue=None):
    """Copy packed magnetic-frill-source arrays to the active compute device."""

    arrays = magnetic_frill_source_host_arrays(magneticfrillsources, G)
    solver = config.sim_config.general["solver"]

    if solver == "cuda":
        import pycuda.gpuarray as gpuarray

        return {name: gpuarray.to_gpu(array) for name, array in arrays.items()}
    if solver == "opencl":
        import pyopencl.array as clarray

        return {name: clarray.to_device(queue, array) for name, array in arrays.items()}
    if solver == "metal":
        dev = config.get_model_config().device["dev"]
        return {
            name: dev.newBufferWithBytes_length_options_(array.tobytes(), array.nbytes, 0)
            for name, array in arrays.items()
        }

    raise ValueError(f"Unknown device solver {solver!r} for magnetic-frill-source arrays.")


def dtoh_magnetic_frill_source_outputs(Vinc, Vtotal, Itot, G):
    """Copy device Vinc/Vtotal/Itot histories into their source objects."""

    expected = (len(G.magneticfrillsources), G.iterations + 1)
    solver = getattr(config.sim_config, "general", {}).get("solver")
    if solver == "metal":
        dtype = config.sim_config.dtypes["float_or_double"]
        nbytes = int(np.prod(expected)) * np.dtype(dtype).itemsize

        def _metal_to_numpy(buffer):
            if buffer.length() != nbytes:
                raise ValueError(
                    "Magnetic-frill Metal output buffer has the wrong size: "
                    f"expected {nbytes} bytes, got {buffer.length()}."
                )
            return np.frombuffer(buffer.contents().as_buffer(nbytes), dtype=dtype).reshape(expected).copy()

        Vinc, Vtotal, Itot = map(_metal_to_numpy, (Vinc, Vtotal, Itot))

    if Vinc.shape != expected or Vtotal.shape != expected or Itot.shape != expected:
        raise ValueError(
            "Magnetic-frill-source device output shape does not match the "
            f"grid: expected {expected}, got {Vinc.shape}, {Vtotal.shape}, "
            f"and {Itot.shape}."
        )

    for i, frill in enumerate(G.magneticfrillsources):
        np.copyto(frill.Vinc, Vinc[i, :], casting="same_kind")
        np.copyto(frill.Vtotal, Vtotal[i, :], casting="same_kind")
        np.copyto(frill.Itot, Itot[i, :], casting="same_kind")


class MagneticFrillSource(Source):
    """Implements a magnetic-frill (equivalent-feed) source for an antenna
    fed through a PEC ground plane by a coaxial line, following Hyun, Kim &
    Kim, "An Equivalent Feed Model for the FDTD Analysis of Antennas Driven
    Through a Ground Plane by Coaxial Lines," IEEE Trans. Antennas Propag.,
    vol 57, no 1, pp 161-167, Jan 2009 (building on Maloney, Smith & Scott
    1990 and King & Harrison's magnetic frill generator).

    Unlike TransmissionLine, this source has no explicit 1D line: the
    coax's sub-cell aperture is represented by an equivalent magnetic
    surface current, entering only the magnetic (Faraday's law) update at
    the four Yee H components immediately surrounding the feed point - the
    same four samples FDTDGrid.calculate_Ix()/calculate_Iy()/calculate_Iz()
    already read for their Ampere's-law loop current, written to here
    rather than read from. Supports x, y, or z polarisation (the antenna
    axis the source drives current along, following the same electrical
    sign convention as calculate_Ix/Iy/Iz) - each axis uses the two Yee H
    components transverse to it, exactly mirroring how
    TransmissionLine/calculate_I{x,y,z} themselves branch on polarisation.

    The characteristic impedance Z0 is supplied by the user and represents
    the physical coax dimensions and filler. The inner-conductor radius is
    inferred from a mandatory co-located ``ThinWire`` because Hyun's discrete
    feed-cell equation (8)-(9) contains the same logarithmic radius factor as
    the attached wire. Mäkinen's projected-H representation supplies k_H;
    the frill adds the corresponding F factor to its own magnetic-current
    term.

    Hyun's recommended time-average approximation (11) couples the new
    half-step loop current back into the voltage driving that same update.
    This is solved analytically each iteration using a precomputed feed-cell
    self-admittance G. The histories store Vinc, Vab, and the time-centred
    Itot at the integer electric-field time used by equations (9)-(10).
    """

    def __init__(self, iterations: int, dt: float):
        super().__init__()
        self.iterations = iterations
        self.dt = dt

        self.Z0 = None

        # Resolved once, after grid.build(), by finalise_setup(): whether
        # the two faces transverse to this source's polarisation axis (e.g.
        # x0/y0 for a z-polarised source, y0/z0 for x-polarised, z0/x0 for
        # y-polarised) are declared PMC symmetry boundaries AND the source
        # sits exactly on that boundary - in which case the retained H
        # component's contribution to Itot doubles to include its image. The
        # field deposit itself is not doubled: the retained edge still obeys
        # its local update equation. _mirror1/_mirror2 correspond to
        # the first/second transverse H component in the polarisation-
        # specific ordering used throughout finalise_setup()/update_magnetic()
        # (see the per-axis branches there). None of this is set until
        # finalise_setup() runs.
        self._mirror1 = None
        self._mirror2 = None
        self.inner_radius = None
        self._drive_terms = []
        self._G_coeff = None
        self._theta = 0.5
        self._previous_half_current = 0.0

        self.Vinc = np.zeros(self.iterations + 1, dtype=config.sim_config.dtypes["float_or_double"])
        self.Vtotal = np.zeros(self.iterations + 1, dtype=config.sim_config.dtypes["float_or_double"])
        self.Itot = np.zeros(self.iterations + 1, dtype=config.sim_config.dtypes["float_or_double"])

        # Bound after materials/update coefficients and time/frequency axes
        # are finalised - see prepare_magnetic_frill_ports() in ports.py and
        # finalise_setup() below, both called from the same post-
        # build_geometry() slot in gprMax/model.py.
        self.port_output = None

    def calculate_waveform_values(self, G):
        """Calculates all waveform values for source for duration of simulation.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """
        waveform = next(x for x in G.waveforms if x.ID == self.waveformID)
        self.waveformvalues_wholedt = np.zeros((G.iterations + 1), dtype=config.sim_config.dtypes["float_or_double"])
        for iteration in range(G.iterations + 1):
            time = G.dt * iteration
            if time >= self.start and time <= self.stop:
                time -= self.start
                self.waveformvalues_wholedt[iteration] = waveform.calculate_value(time, G.dt)

    def _validate_geometry(self, G):
        """Bind the attached thin wire and check the local PEC ground plane."""

        i, j, k = self.xcoord, self.ycoord, self.zcoord
        component = f"E{self.polarisation}"
        material_numID = int(G.ID[G.IDlookup[component], i, j, k])
        material = G.materials[material_numID]
        is_attached_wire = (
            material.type == "thin-wire"
            and getattr(material, "thin_wire_axis", None) == self.polarisation
            and getattr(material, "thin_wire_role", None) == component
        )
        if not is_attached_wire:
            raise ValueError(
                f"{self.ID} requires a co-located #thin_wire along the "
                f"{self.polarisation}-directed feed edge at {(i, j, k)}. "
                "Hyun's feed-cell equation uses the attached wire radius; "
                "an ordinary PEC edge has no unambiguous physical radius."
            )
        self.inner_radius = float(material.thin_wire_radius)

        tangential_components = {
            "x": ("Ey", "Ez"),
            "y": ("Ex", "Ez"),
            "z": ("Ex", "Ey"),
        }[self.polarisation]
        for tangential in tangential_components:
            tangential_numID = int(G.ID[G.IDlookup[tangential], i, j, k])
            tangential_material = G.materials[tangential_numID]
            if not tangential_material.is_pec:
                raise ValueError(
                    f"{self.ID} requires a PEC ground plane perpendicular to "
                    f"the feed at {(i, j, k)}; tangential component "
                    f"{tangential} has material {tangential_material.ID!r}."
                )

    def finalise_setup(self, G):
        """Resolve symmetry adjacency and validate the PEC feed point.

        Must run after grid.build() has finalised materials/update
        coefficients and grid.symmetry_boundaries - FDTDGrid.build()'s own
        command-processing pass (where this source's user-object build()
        already ran, appending it to grid.magneticfrillsources) happens
        strictly before that, so none of this data is available yet at
        that point. Called from gprMax/model.py's post-build_geometry()
        "prepare" pass, in the same slot as prepare_transmission_line_ports().

        Args:
            G: FDTDGrid class describing a grid in a model.
        """
        i, j, k = self.xcoord, self.ycoord, self.zcoord
        if G.within_pml(np.array([i, j, k], dtype=np.int32)):
            raise ValueError(
                f"{self.ID} feed point lies inside a PML - update_magnetic_pml() "
                "already overwrites this position before this source's own "
                "drive would run, silently discarding it every iteration. "
                "Move the source away from the PML region."
            )

        # The two faces transverse to the polarisation axis, in the fixed
        # order used by every branch below (matches the H-component order
        # in calculate_Ix/Iy/Iz: x-pol -> (Hy differenced along z, Hz
        # differenced along y); y-pol -> (Hx differenced along z, Hz
        # differenced along x); z-pol -> (Hy differenced along x, Hx
        # differenced along y)).
        axis_faces = {
            "x": ("z0", "y0", "zmax", "ymax", k == 0, j == 0, k == G.nz, j == G.ny),
            "y": ("z0", "x0", "zmax", "xmax", k == 0, i == 0, k == G.nz, i == G.nx),
            "z": ("x0", "y0", "xmax", "ymax", i == 0, j == 0, i == G.nx, j == G.ny),
        }
        face1, face2, maxface1, maxface2, at1, at2, atmax1, atmax2 = axis_faces[self.polarisation]
        face1_pmc = G.symmetry_boundaries.get(face1) == "pmc"
        face2_pmc = G.symmetry_boundaries.get(face2) == "pmc"

        # New required validation: a feed point accidentally placed at a
        # domain-minimum boundary with no corresponding declared PMC face
        # would otherwise silently compute zero current (calculate_Ix/Iy/Iz's
        # own unconditional domain-boundary guard fires regardless of
        # whether symmetry is declared there) - reject outright rather than
        # producing a source that silently does nothing.
        if at1 and not face1_pmc:
            raise ValueError(
                f"{self.ID} feed point sits at the domain boundary "
                f"{face1[0]}={face1[1:] or '0'} without "
                f"'#symmetry_boundary {face1} pmc' declared there - current "
                "extraction would silently be zero (a pre-existing "
                "limitation of the underlying calculate_Ix/Iy/Iz loop, "
                "which cannot distinguish 'domain edge' from 'symmetry "
                "plane'). Move the source away from the domain edge or "
                "declare the symmetry boundary."
            )
        if at2 and not face2_pmc:
            raise ValueError(
                f"{self.ID} feed point sits at the domain boundary "
                f"{face2[0]}={face2[1:] or '0'} without "
                f"'#symmetry_boundary {face2} pmc' declared there - current "
                "extraction would silently be zero. Move the source away "
                "from the domain edge or declare the symmetry boundary."
            )

        # Scope: v1 only supports symmetry-plane placement at the domain's
        # own origin corner ("0"-type faces) - the ghost-substitution
        # derivation was worked out specifically for calculate_Ix/Iy/Iz's
        # own guarded ("0"-type) case. "max"-type symmetry corners would
        # need the analogous ghost convention (interior-adjacent index,
        # flipped sign) worked out separately.
        if atmax1 and G.symmetry_boundaries.get(maxface1) == "pmc":
            raise ValueError(
                f"{self.ID} at a {maxface1}-type symmetry corner is not yet "
                "supported - v1 only supports '0'-type corners."
            )
        if atmax2 and G.symmetry_boundaries.get(maxface2) == "pmc":
            raise ValueError(
                f"{self.ID} at a {maxface2}-type symmetry corner is not yet "
                "supported - v1 only supports '0'-type corners."
            )

        self._validate_geometry(G)
        self._mirror1 = at1 and face1_pmc
        self._mirror2 = at2 and face2_pmc
        self._prepare_drive_terms(G)

    def _prepare_drive_terms(self, G):
        """Precompute Hyun's Cartesian feed stencil and self-admittance."""

        i, j, k = self.xcoord, self.ycoord, self.zcoord
        dx, dy, dz = G.dx, G.dy, G.dz
        axial_step = {"x": dx, "y": dy, "z": dz}[self.polarisation]

        # Each pair is (component, plus edge, minus edge, radial axis,
        # mirrored). An edge is (coordinates, current-loop weight,
        # magnetic-current sign). At a PMC minimum face, the missing edge is
        # its odd image: double the retained edge's loop weight, but do not
        # double its field update.
        pairs = {
            "x": (
                ("Hy", ((i, j, k), -dy, -1), ((i, j, k - 1), dy, 1), "z", self._mirror1),
                ("Hz", ((i, j, k), dz, 1), ((i, j - 1, k), -dz, -1), "y", self._mirror2),
            ),
            "y": (
                ("Hx", ((i, j, k), dx, 1), ((i, j, k - 1), -dx, -1), "z", self._mirror1),
                ("Hz", ((i, j, k), -dz, -1), ((i - 1, j, k), dz, 1), "x", self._mirror2),
            ),
            "z": (
                ("Hy", ((i, j, k), dy, 1), ((i - 1, j, k), -dy, -1), "x", self._mirror1),
                ("Hx", ((i, j, k), -dx, -1), ((i, j - 1, k), dx, 1), "y", self._mirror2),
            ),
        }[self.polarisation]

        radial_steps = {"x": dx, "y": dy, "z": dz}
        terms = []
        for component, plus, minus, radial_axis, mirrored in pairs:
            edges = (plus,) if mirrored else (plus, minus)
            for edge_index, (coordinates, current_weight, source_sign) in enumerate(edges):
                if mirrored and edge_index == 0:
                    current_weight *= 2
                x, y, z = coordinates
                material_numID = int(G.ID[G.IDlookup[component], x, y, z])
                material = G.materials[material_numID]
                correct_wire_row = (
                    material.type == "thin-wire"
                    and getattr(material, "thin_wire_axis", None) == self.polarisation
                    and getattr(material, "thin_wire_role", None) == component
                    and np.isclose(
                        material.thin_wire_radius,
                        self.inner_radius,
                        rtol=1e-12,
                        atol=0.0,
                    )
                    and getattr(material, "thin_wire_radial_axis", None) == radial_axis
                )
                if not correct_wire_row:
                    raise ValueError(
                        f"{self.ID} feed stencil component {component} at "
                        f"{coordinates} is not part of the attached thin wire."
                    )

                factor_f = float(material.thin_wire_factors["F"])
                radial_step = radial_steps[radial_axis]
                source_gain = source_sign * G.updatecoeffsH[material_numID, 4] * factor_f / (axial_step * radial_step)
                terms.append((component, x, y, z, float(current_weight), float(source_gain)))

        self._drive_terms = terms
        self._G_coeff = float(sum(term[-2] * term[-1] for term in terms))
        if not np.isfinite(self._G_coeff) or self._G_coeff <= 0:
            raise ValueError(f"{self.ID} produced invalid feed-cell self-admittance " f"G={self._G_coeff!r}.")

        # Two independently advanced feedback relations cannot safely write
        # the same H edge: their result would depend on source order on CPU
        # and would be a device write race. A coupled multiport feed would
        # require solving the shared feed-cell system as one operation.
        registry = getattr(G, "_magnetic_frill_drive_edges", {})
        drive_edges = {(term[0], term[1], term[2], term[3]) for term in terms}
        overlap = next(
            ((edge, registry[edge]) for edge in drive_edges if edge in registry and registry[edge] is not self),
            None,
        )
        if overlap is not None:
            edge, owner = overlap
            raise ValueError(
                f"{self.ID} has an overlapping magnetic feed edge {edge} "
                f"with {owner.ID}; adjacent or duplicate frill stencils "
                "require a coupled multiport formulation."
            )
        registry.update({edge: self for edge in drive_edges})
        G._magnetic_frill_drive_edges = registry

        self._previous_half_current = 0.0
        logger.info(
            f"{self.ID}: Hyun feed cell uses attached thin-wire radius "
            f"a={self.inner_radius:g}m, time-average current, and "
            f"self-admittance G={self._G_coeff:g}S."
        )

    def _calculate_Itot_frill(self, Hx, Hy, Hz):
        """Return the image-completed Ampere-loop current from stored H."""

        fields = {"Hx": Hx, "Hy": Hy, "Hz": Hz}
        return sum(weight * fields[component][x, y, z] for component, x, y, z, weight, _ in self._drive_terms)

    def update_magnetic(self, iteration, updatecoeffsH, ID, Hx, Hy, Hz, G):
        """Apply Hyun's time-average implicit feed update in closed form.

        Args:
            iteration: int of current iteration (timestep).
            updatecoeffsH: memory view of array of magnetic field update
                            coefficients.
            ID: memory view of array of numeric IDs corresponding to
                materials in the model.
            Hx, Hy, Hz: memory view of array of magnetic field values.
            G: FDTDGrid class describing a grid in a model.
        """
        self.Vinc[iteration] = 0.5 * self.waveformvalues_wholedt[iteration]
        current_bulk = self._calculate_Itot_frill(Hx, Hy, Hz)
        zeta = self._G_coeff * self.Z0
        current_new = (
            current_bulk
            + 2 * self._G_coeff * self.Vinc[iteration]
            - zeta * (1 - self._theta) * self._previous_half_current
        ) / (1 + zeta * self._theta)
        current_centred = (1 - self._theta) * self._previous_half_current + self._theta * current_new
        self.Itot[iteration] = current_centred
        V_ab = 2 * self.Vinc[iteration] - self.Z0 * current_centred
        self.Vtotal[iteration] = V_ab

        fields = {"Hx": Hx, "Hy": Hy, "Hz": Hz}
        for component, x, y, z, _, source_gain in self._drive_terms:
            fields[component][x, y, z] += source_gain * V_ab
        self._previous_half_current = current_new


class DiscretePlaneWave(Source):
    """Implements the discrete plane wave (DPW) formulation as described in
    Tan, T.; Potter, M. (2010). FDTD Discrete Planewave (FDTD-DPW)
    Formulation for a Perfectly Matched Source in TFSF Simulations., 58(8),
    0–2648. doi:10.1109/tap.2010.2050446

    Implements a PML terninated 1D DPW FDTD grid which is used to source
    a plane wave into a 2D or 3D FDTD grid using the total-field/scattered-field
    (TFSF) formulation.

    Origin of the DPW can be any corner of the FDTD grid and the
    propagation direction is defined by two angles, phi and theta. The DPW
    is defined by three integers, m_x, m_y, m_z which determine the rational
    angles corresponding to the propagation direction.

    """

    def __init__(self, G):
        """
        Args:
            m: int array stores the integer mappings, m_x, m_y, m_z which
                determine the rational angles last element stores
                max(m_x, m_y, m_z).
            directions: int array stores the directions of propagation of DPW.
            dimensions: int stores the number of dimensions in which the
                        simulation is run (2D or 3D).
            time_dimension: int stores the time length over which the simulation
                            is run.
            E_fields: double array stores the electric flieds associated with
                        1D DPW.
            H_fields: double array stores the magnetic fields associated with
                        1D DPW.
            G: FDTDGrid class describing a grid in a model.
        """

        super().__init__()
        self.m = np.zeros(3 + 1, dtype=np.int32)  # +1 to store the max(m_x, m_y, m_z)
        self.origin = np.zeros(3, dtype=np.int32)
        self.origin[0] = 0
        self.origin[1] = 0
        self.origin[2] = 0
        self.length = 0
        # self.projections = np.zeros(6, dtype=config.sim_config.dtypes["float_or_double"])
        self.projections = np.zeros(6, dtype=np.float64)  # Use float64 for better precision in projections
        self.corners = None
        self.materialID = None
        self.ds = 0
        self.speed = config.c
        self.axial = 0
        self.dispersive = False
        self.pml_cells = 20
        self.buffercells_axial = 5
        self.psi = 0.0
        self.phi = 0.0
        self.theta = 0.0
        self.max_angle_diff = 0.0
        self.actual_angles = np.zeros(2, dtype=np.float64)  # [theta, phi]
        self.angle_errors = np.zeros(2, dtype=np.float64)  # [Delta_theta, Delta_phi]
        self.total_error = 0.0

        # 2D-mode state, resolved once here (the mode is final by the time
        # any DPW user object is built - the same assumption HertzianDipole
        # relies on). mode2d uses the same encoding as CPUUpdates.__init__:
        # -1 = 3D, 0/1/2 = 2D TM invariant x/y/z, 3/4/5 = 2D TE invariant
        # x/y/z. skip_axis is the invariant axis (-1 in 3D) and is passed to
        # the Cython TFSF kernels so the pair of box faces normal to it is
        # skipped - a 2D TFSF is a rectangle of 4 edges, not a box of 6
        # faces; the perpendicular faces would write spurious corrections
        # into structurally-dead field components (and, at a TM wall,
        # out-of-bounds at index -1).
        mode = config.get_model_config().mode
        if mode.startswith("2D TM"):
            self.mode2d = "xyz".index(mode[-1])
        elif mode.startswith("2D TE"):
            self.mode2d = 3 + "xyz".index(mode[-1])
        else:
            self.mode2d = -1
        self.skip_axis = self.mode2d % 3 if self.mode2d >= 0 else -1
        self.is_TM = 0 <= self.mode2d < 3

        # Transverse sample position used by axial-mode grid_init() when
        # copying the layered material profile out of G.ID/G.solid: an
        # arbitrary interior column in 3D ((1,1,1) - the historic choice),
        # but in 2D the invariant-axis slot must be the mode's live layer -
        # 0 for TM (the only cell; index 1 rows there are wall-forced) and
        # 1 for TE (the live interior layer between the forced walls at 0
        # and 2, which for TE coincides with the 3D default).
        self.transverse_pos = [1, 1, 1]
        if self.mode2d >= 0:
            self.transverse_pos[self.skip_axis] = 0 if self.is_TM else 1

    def initializeDiscretePlaneWave(self, G):
        """Creates a DPW, assigns memory to the grids, and gets field values
            at different time and space indices.

        Args:
            psi: float for polarization angle of the incident plane wave.
            phi: float for azimuthal angle (radians) of the incident plane wave.
            Delta_phi: float for permissible error in the rational angle
                        (radians) approximation to phi.
            theta: float for polar angle (radians) of the incident plane wave.
            Delta_theta: float for permissible error in the rational angle
                            (radians) approximation to theta.
            G: FDTDGrid class describing a grid in a model.
            number: int for number of cells in the 3D FDTD simulation.
            dx: double for separation between adjacent cells in the x direction.
            dy: double for separation between adjacent cells in the y direction.
            dz: double for separation between adjacent cells in the z direction.
            dt: double for time step for the FDTD simulation.

        Returns:
            E_fields: double array for electric field for the DPW as it evolves
                        over time and space indices.
            H_fields: double array for magnetic field for the DPW as it evolves
                        over time and space indices.
            C: double array stores coefficients of the fields for the update
                equation of the electric fields.
            D: double array stores coefficients of the fields for the update
                equation of the magnetic fields.
        """

        # check for plane wave definition using angles and in this case m vector should be zero and needs to be calculated
        if self.m[0] == 0 and self.m[1] == 0 and self.m[2] == 0:
            # Find the integer mappings m_x, m_y, m_z for the DPW using partial fractions
            self.m[:3], self.actual_angles, self.angle_errors, self.total_error = self.find_dpw_integers_optimized(
                self.theta, self.phi, [G.dx, G.dy, G.dz], self.max_angle_diff
            )

        # check for axial propagation case where the user wants a plane wave normally incident using grid geometry assuming layered model at best.
        elif self.axial != 0:
            self.actual_angles[0] = self.theta
            self.actual_angles[1] = self.phi
            self.angle_errors[0] = 0.0
            self.angle_errors[1] = 0.0
            self.total_error = 0.0

        # check for plane wave definition using m vector and in this case angles should be zero and need to be calculated. There is no error in this case.
        else:
            # The physical propagation direction is the wavefront normal
            # (m_x/dx, m_y/dy, m_z/dz) - the same convention
            # find_dpw_integers_optimized() uses for its candidate selection
            # and returned angles (phys_vec = m / delta there), so vector
            # mode and angles mode report identical angles for the same m,
            # including for anisotropic cells.
            phys_vec = self.m[:3] / np.array([G.dx, G.dy, G.dz])
            phys_vec_norm = phys_vec / np.linalg.norm(phys_vec)
            self.actual_angles[0] = math.degrees(math.acos(np.clip(phys_vec_norm[2], -1.0, 1.0)))
            self.actual_angles[1] = math.degrees(math.atan2(phys_vec_norm[1], phys_vec_norm[0]))
            self.angle_errors[0] = 0.0
            self.angle_errors[1] = 0.0
            self.total_error = 0.0

        # In a 2D model the plane wave must propagate in-plane: the integer
        # mapping must have a zero component on the invariant axis. This is a
        # hard error rather than a warning because it is also a stability
        # requirement: the 1D DPW scheme is the bulk Yee scheme restricted to
        # the wavevector family (kappa*m_x, kappa*m_y, kappa*m_z); with
        # m[invariant] == 0 that family lies inside the 2D Brillouin zone,
        # for which the (larger) 2D CFL timestep is stable by construction -
        # but a nonzero invariant-axis m samples 3D wavevectors, for which
        # the 2D timestep genuinely can be unstable. (Note c*dt <= ds is NOT
        # the right stability criterion here - 1D neighbour coupling is at
        # strides m_i, not 1 - so no such runtime check is used.)
        if self.mode2d >= 0 and self.m[self.skip_axis] != 0:
            letter = "xyz"[self.skip_axis]
            logger.exception(
                f"Discrete plane wave: in {config.get_model_config().mode} mode the plane "
                f"wave must propagate in-plane: the propagation direction must have a zero "
                f"{letter}-component, but the integer mapping gave m_{letter} = "
                f"{self.m[self.skip_axis]}. (E.g. for a mode invariant in z this requires "
                f"theta = 90 degrees.)"
            )
            raise ValueError

        # Get angles in radians
        self.phi_est_rad = math.radians(self.actual_angles[1])
        self.theta_est_rad = math.radians(self.actual_angles[0])
        self.psi_rad = math.radians(self.psi)

        # The dispersive background setup below needs the source frequency
        # to evaluate the material impedance and propagation speed. Resolve
        # the waveform before that setup, rather than at the end of this
        # method after its first use.
        self.waveform = next(x for x in G.waveforms if x.ID == self.waveformID)

        # Calculate the direction cosines
        px = math.sin(self.theta_est_rad) * math.cos(self.phi_est_rad)
        py = math.sin(self.theta_est_rad) * math.sin(self.phi_est_rad)
        pz = math.cos(self.theta_est_rad)

        # Maximum of the absolute values of m_x, m_y, m_z
        self.max_m = np.max(np.abs(self.m[:3]))

        # Store the absolute value of max(m_x, m_y, m_z) in the last element of the array
        self.m[3] = self.max_m

        if self.m[0] < 0:
            self.origin[0] = G.nx + 1
        if self.m[1] < 0:
            self.origin[1] = G.ny + 1
        if self.m[2] < 0:
            self.origin[2] = G.nz + 1

        # Calculate ds that is needed for sourcing the 1D array. This is the spatial step of the 1D DPW grid.
        # For axial propagation this is simply the grid step in the direction of propagation.
        # For non-axial propagation this is calculated from the grid steps and the m vector.
        if self.m[0] == 0:
            if self.m[1] == 0:
                if self.m[2] == 0:
                    raise ValueError("DPW should not be here as not all m_i values can be zero")
                else:
                    self.ds = pz * G.dz / self.m[2]
            else:
                self.ds = py * G.dy / self.m[1]
        else:
            self.ds = px * G.dx / self.m[0]

        # get the number of 1D DPW grid PML cells from the number of 3D FDTD PML cells used for terminating the 1D grid. This is set to 20 cells by default.
        self.pml_length = (
            np.abs(self.m[0]) * self.pml_cells + np.abs(self.m[1]) * self.pml_cells + np.abs(self.m[2]) * self.pml_cells
        )
        # Set few buffer FDTD cells as extra
        buffercells = np.abs(self.m[0]) * self.max_m + np.abs(self.m[1]) * self.max_m + np.abs(self.m[2]) * self.max_m

        # Total length of the 1D grid if not axial propagation
        if self.axial == 0:
            self.length = (
                np.abs(self.m[0]) * (G.nx + 1)
                + np.abs(self.m[1]) * (G.ny + 1)
                + np.abs(self.m[2]) * (G.nz + 1)
                + self.pml_length
                + buffercells
            )
        # Total length of the 1D grid for axial propagation case where a two-sided PML is used
        else:
            buffercells = self.buffercells_axial
            self.length = (
                np.abs(self.m[0]) * (G.nx + 1)
                + np.abs(self.m[1]) * (G.ny + 1)
                + np.abs(self.m[2]) * (G.nz + 1)
                + 2 * self.pml_length
                + buffercells
            )
            self.origin_axial = self.pml_length + buffercells

        # self.length = 8000  # For testing purposes, limit length to 8000 cells

        # Setup an DPW grid ID array for accessing material IDs of the main grid for axial propagation problems only
        # Allocate memory for the 1D fields
        self.E_fields = np.zeros(
            (3, self.length),
            order="C",
            dtype=config.sim_config.dtypes["float_or_double"],
        )
        self.H_fields = np.zeros(
            (3, self.length),
            order="C",
            dtype=config.sim_config.dtypes["float_or_double"],
        )
        # Allocate memory for the 1D source fields for axial propagation case
        if self.axial != 0:
            self.E_fields_s = np.zeros(
                (3, self.length),
                order="C",
                dtype=config.sim_config.dtypes["float_or_double"],
            )
            self.H_fields_s = np.zeros(
                (3, self.length),
                order="C",
                dtype=config.sim_config.dtypes["float_or_double"],
            )

        # Allocate memory for the 1D DPW PML integrals
        # Izjxy means correcting an E_z field due to a H_x field variation in y derivative direction array position 0
        # Izjyx means correcting an E_z field due to a H_y field variation in x derivative direction array position 1
        # Izmxy means correcting an H_z field due to an E_x field variation in y derivative direction array position 2
        # Izmyx means correcting an H_z field due to an E_y field variation in x derivative direction array position 3
        self.Iz = np.zeros((4, self.pml_length), order="C", dtype=config.sim_config.dtypes["float_or_double"])

        # Allocate memory for the 1D DPW PML integrals for axial propagation case
        if self.axial != 0:
            self.Iz_s = np.zeros((4, self.pml_length), order="C", dtype=config.sim_config.dtypes["float_or_double"])
            self.Iz0 = np.zeros((4, self.pml_length), order="C", dtype=config.sim_config.dtypes["float_or_double"])

        # Iyjxz means correcting an E_y field due to a H_x field variation in z derivative direction array position 0
        # Iyjzx means correcting an E_y field due to a H_z field variation in x derivative direction array position 1
        # Iymxz means correcting an H_y field due to an E_x field variation in z derivative direction array position 2
        # Iymzx means correcting an H_y field due to an E_z field variation in x derivative direction array position 3
        self.Iy = np.zeros((4, self.pml_length), order="C", dtype=config.sim_config.dtypes["float_or_double"])

        # Allocate memory for the 1D DPW PML integrals for axial propagation case
        if self.axial != 0:
            self.Iy_s = np.zeros((4, self.pml_length), order="C", dtype=config.sim_config.dtypes["float_or_double"])
            self.Iy0 = np.zeros((4, self.pml_length), order="C", dtype=config.sim_config.dtypes["float_or_double"])

        # Ixjyz means correcting an E_x field due to a H_y field variation in z derivative direction array position 0
        # Ixjzy means correcting an E_x field due to a H_z field variation in y derivative direction array position 1
        # Ixmyz means correcting an H_x field due to an E_y field variation in z derivative direction array position 2
        # Ixmzy means correcting an H_x field due to an E_z field variation in y derivative direction array position 3
        self.Ix = np.zeros((4, self.pml_length), order="C", dtype=config.sim_config.dtypes["float_or_double"])

        # Allocate memory for the 1D DPW PML integrals for axial propagation case
        if self.axial != 0:
            self.Ix_s = np.zeros((4, self.pml_length), order="C", dtype=config.sim_config.dtypes["float_or_double"])
            self.Ix0 = np.zeros((4, self.pml_length), order="C", dtype=config.sim_config.dtypes["float_or_double"])

        # When no grid IDs are used Get the background material object with the matching ID and add it to the PlaneWave object
        if self.axial == 0:
            self.material = next((x for x in G.materials if x.ID == self.materialID), None)

            # A homogeneous DPW can use any electric dispersion supported by
            # the main grid. The real part of eps_r at the waveform centre
            # frequency sets the reference speed and impedance; the complete
            # time-domain response is evolved by the auxiliary pole state.
            if getattr(self.material, "poles", 0) > 0:
                self.dispersive = True
                material_er = np.real(self.material.calculate_er(self.waveform.freq))
                self.materialZ = math.sqrt(config.m0 * self.material.mr / (config.e0 * material_er))
                self.speed = config.c / math.sqrt(material_er * self.material.mr)
                self.max_poles = self.material.poles
            else:
                self.materialZ = math.sqrt(
                    config.m0 * self.material.mr / (config.e0 * self.material.er)
                )  # Impedance in the material
                self.speed = config.c / math.sqrt(self.material.er * self.material.mr)  # Speed in the material

            # Calculate the projections for sourcing the electric and magnetic fields
            # using double precision for better accuracy

            self.projections[0] = math.cos(self.psi_rad) * math.sin(self.phi_est_rad) - math.sin(
                self.psi_rad
            ) * math.cos(self.theta_est_rad) * math.cos(self.phi_est_rad)
            if abs(self.projections[0]) <= 1e-15:
                self.projections[0] = 0

            self.projections[1] = -math.cos(self.psi_rad) * math.cos(self.phi_est_rad) - math.sin(
                self.psi_rad
            ) * math.cos(self.theta_est_rad) * math.sin(self.phi_est_rad)
            if abs(self.projections[1]) <= 1e-15:
                self.projections[1] = 0

            self.projections[2] = math.sin(self.psi_rad) * math.sin(self.theta_est_rad)
            if abs(self.projections[2]) <= 1e-15:
                self.projections[2] = 0

            self.projections[3] = (
                math.sin(self.psi_rad) * math.sin(self.phi_est_rad)
                + math.cos(self.psi_rad) * math.cos(self.theta_est_rad) * math.cos(self.phi_est_rad)
            ) / self.materialZ
            if abs(self.projections[3]) <= 1e-15:
                self.projections[3] = 0

            self.projections[4] = (
                -math.sin(self.psi_rad) * math.cos(self.phi_est_rad)
                + math.cos(self.psi_rad) * math.cos(self.theta_est_rad) * math.sin(self.phi_est_rad)
            ) / self.materialZ
            if abs(self.projections[4]) <= 1e-15:
                self.projections[4] = 0

            self.projections[5] = (-math.cos(self.psi_rad) * math.sin(self.theta_est_rad)) / self.materialZ
            if abs(self.projections[5]) <= 1e-15:
                self.projections[5] = 0

            # Axial mode computes its projections later, in grid_init()
            # (they need the grid-sampled background material) - it runs
            # this same validation there instead.
            self._validate_2d_projections()

        if self.axial == 0:
            self._get_pml_parameters(G)

    def _validate_2d_projections(self):
        """Validates the polarisation of the plane wave against the active 2D
        mode (no-op in 3D).

        In a 2D model the structurally-dead field components must carry no
        incident-wave amplitude, or the TFSF corrections would try to excite
        components the bulk 2D kernels never update. Since the projections
        are already clamped to exactly 0 below 1e-15, this is an exact-zero
        operational check on the dead set - covering angles, vector and
        axial modes with one rule, rather than a closed-form psi condition
        (which depends on theta/phi for a general invariant axis):

        - TM (E along the invariant axis only): the two in-plane E
          projections and the invariant-axis H projection must be 0.
        - TE (E in-plane only): the invariant-axis E projection and the two
          in-plane H projections must be 0.
        """
        if self.mode2d < 0:
            return

        a = self.skip_axis
        t1, t2 = [ax for ax in (0, 1, 2) if ax != a]
        names = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

        if self.is_TM:
            dead = [t1, t2, 3 + a]
            hint = "for a mode invariant in z, TM requires psi = 90 or 270 degrees"
        else:
            dead = [a, 3 + t1, 3 + t2]
            hint = "for a mode invariant in z, TE requires psi = 0 or 180 degrees"

        nonzero = [f"{names[c]} = {self.projections[c]:g}" for c in dead if self.projections[c] != 0]
        if nonzero:
            logger.exception(
                f"Discrete plane wave: in {config.get_model_config().mode} mode the "
                f"polarisation must not excite the structurally-dead field components, "
                f"but the following incident-field projections are nonzero: "
                f"{', '.join(nonzero)}. Adjust psi ({hint})."
            )
            raise ValueError

        live = [c for c in range(6) if c not in dead]
        if all(self.projections[c] == 0 for c in live):
            logger.exception(
                "Discrete plane wave: all live-component field projections are zero - "
                "the plane wave would inject nothing. Check theta/phi/psi."
            )
            raise ValueError

    def grid_init(self, G):
        # Initialize the ID array for axial propagation problems only extending accordingly for the two PML regions
        if self.axial != 0:
            self.ID = np.zeros((6, self.length), dtype=np.uint32)  # 6 for the 6 field components

            # Copy the layered material profile out of G.ID along the
            # propagation axis, sampling at self.transverse_pos on the two
            # transverse axes - an arbitrary interior column in 3D (the
            # layered-model assumption makes the choice arbitrary), and the
            # mode's live layer on the invariant axis in 2D (see __init__).
            # Dead components in 2D sample wall-forced pec/pmc rows, which
            # is harmless AND desirable: their 1D field arrays are
            # identically zero (zero projections + m[invariant] == 0), so
            # those coefficient rows never multiply nonzero data, and a
            # zero (pec) row additionally pins them at zero against
            # roundoff.
            prop = self.axial - 1  # 0/1/2 = x/y/z
            n_prop = (G.nx, G.ny, G.nz)[prop]
            pos = list(self.transverse_pos)

            def _gid(component, prop_idx):
                pos[prop] = prop_idx
                return G.ID[(component, *pos)]

            for c in range(6):
                # Leading 1D buffer/PML region: extend the first grid cell's profile
                for idx in range(self.origin_axial + 1):
                    self.ID[c, idx] = _gid(c, 1)
                # Main grid profile
                for idx in range(self.origin_axial + 1, n_prop + self.origin_axial):
                    self.ID[c, idx] = _gid(c, idx - self.origin_axial)
                # Trailing 1D PML region: extend the last grid cell's profile
                for idx in range(n_prop + self.origin_axial, self.length):
                    self.ID[c, idx] = _gid(c, n_prop - 1)

            sampled_material_ids = set(np.unique(self.ID))
            sampled_dispersive_materials = [
                material
                for material in G.materials
                if material.numID in sampled_material_ids and getattr(material, "poles", 0) > 0
            ]

            # Get the background material near the origin (used for the
            # speed and impedance of the DPW) and near the far PML (used
            # for the PML parameters). Sampled from G.solid - the
            # cell-centred material array - rather than G.ID: solid is
            # never wall-forced by the 2D framework (the tm*/te* forcing
            # touches only ID), needs no field-component-row choice, and is
            # immune to dielectric-smoothed compound rows at layer
            # interfaces. The background at these positions is homogeneous
            # by the plane wave's own assumption, so the cell-centred value
            # is the right one. transverse_pos is valid for solid's
            # cell-centred extents too: a TM invariant axis is 1 cell
            # (index 0) and a TE one is 2 cells (index 1, the live
            # interior layer).
            pos_solid = list(self.transverse_pos)

            pos_solid[prop] = 2
            self.material = next((x for x in G.materials if x.numID == G.solid[tuple(pos_solid)]), None)

            pos_solid[prop] = n_prop - 2
            self.materialPML = next((x for x in G.materials if x.numID == G.solid[tuple(pos_solid)]), None)

            # Reference material properties for the source-side auxiliary
            # grid. All Debye, Lorentz, and Drude pole dynamics are handled
            # by the same recurrence as the main grid below.
            if getattr(self.material, "poles", 0) > 0:
                material_er = np.real(self.material.calculate_er(self.waveform.freq))
                self.materialZ = math.sqrt(config.m0 * self.material.mr / (config.e0 * material_er))
                self.speed = config.c / math.sqrt(material_er * self.material.mr)
            else:
                self.materialZ = math.sqrt(
                    config.m0 * self.material.mr / (config.e0 * self.material.er)
                )  # Impedance in the material
                self.speed = config.c / math.sqrt(self.material.er * self.material.mr)  # Speed in the material

            # Set the material ID of the PML region at origin as the same as the material next to the grid origin
            self.materialPML0 = self.material
            self.materialPML0Z = self.materialZ
            self.PML0speed = self.speed

            # Reference properties at the far-side DPW PML.
            if getattr(self.materialPML, "poles", 0) > 0:
                material_pml_er = np.real(self.materialPML.calculate_er(self.waveform.freq))
                self.materialPMLZ = math.sqrt(config.m0 * self.materialPML.mr / (config.e0 * material_pml_er))
                self.PMLspeed = config.c / math.sqrt(material_pml_er * self.materialPML.mr)
            else:
                self.materialPMLZ = math.sqrt(
                    config.m0 * self.materialPML.mr / (config.e0 * self.materialPML.er)
                )  # Impedance in the material
                self.PMLspeed = config.c / math.sqrt(self.materialPML.er * self.materialPML.mr)  # Speed in the material

            self.dispersive = bool(sampled_dispersive_materials)
            self.max_poles = max(
                (material.poles for material in sampled_dispersive_materials),
                default=0,
            )

            # Calculate the projections for sourcing the electric and magnetic fields
            # using double precision for better accuracy

            self.projections[0] = math.cos(self.psi_rad) * math.sin(self.phi_est_rad) - math.sin(
                self.psi_rad
            ) * math.cos(self.theta_est_rad) * math.cos(self.phi_est_rad)
            if abs(self.projections[0]) <= 1e-15:
                self.projections[0] = 0

            self.projections[1] = -math.cos(self.psi_rad) * math.cos(self.phi_est_rad) - math.sin(
                self.psi_rad
            ) * math.cos(self.theta_est_rad) * math.sin(self.phi_est_rad)
            if abs(self.projections[1]) <= 1e-15:
                self.projections[1] = 0

            self.projections[2] = math.sin(self.psi_rad) * math.sin(self.theta_est_rad)
            if abs(self.projections[2]) <= 1e-15:
                self.projections[2] = 0

            self.projections[3] = (
                math.sin(self.psi_rad) * math.sin(self.phi_est_rad)
                + math.cos(self.psi_rad) * math.cos(self.theta_est_rad) * math.cos(self.phi_est_rad)
            ) / self.materialZ
            if abs(self.projections[3]) <= 1e-15:
                self.projections[3] = 0

            self.projections[4] = (
                -math.sin(self.psi_rad) * math.cos(self.phi_est_rad)
                + math.cos(self.psi_rad) * math.cos(self.theta_est_rad) * math.sin(self.phi_est_rad)
            ) / self.materialZ
            if abs(self.projections[4]) <= 1e-15:
                self.projections[4] = 0

            self.projections[5] = (-math.cos(self.psi_rad) * math.sin(self.theta_est_rad)) / self.materialZ
            if abs(self.projections[5]) <= 1e-15:
                self.projections[5] = 0

            self._validate_2d_projections()

            self._get_pml_parameters(G)

            print(
                f"Discrete Plane Wave has been initialized "
                + f"with field projections (Ex, Ey, Ez, Hx, Hy, Hz) = ({self.projections[0]:.4f}, {self.projections[1]:.4f}, {self.projections[2]:.4f}, {self.projections[3]:.4f}, {self.projections[4]:.4f}, {self.projections[5]:.4f})"
                + f" , grid origin = ({self.origin[0]}, {self.origin[1]}, {self.origin[2]})"
                + f" and 1D vector length = {self.length} cells."
            )

        # Allocate the DPW auxiliary state after the model-wide dispersive
        # dtype has been resolved. Debye-only models use real storage;
        # Lorentz/Drude models use complex storage, exactly as the main grid.
        if self.dispersive:
            dispersive_dtype = config.get_model_config().materials["dispersivedtype"]
            state_shape = (self.max_poles, self.length)
            self.Px = np.zeros(state_shape, order="C", dtype=dispersive_dtype)
            self.Py = np.zeros(state_shape, order="C", dtype=dispersive_dtype)
            self.Pz = np.zeros(state_shape, order="C", dtype=dispersive_dtype)
            if self.axial != 0:
                self.Px_s = np.zeros(state_shape, order="C", dtype=dispersive_dtype)
                self.Py_s = np.zeros(state_shape, order="C", dtype=dispersive_dtype)
                self.Pz_s = np.zeros(state_shape, order="C", dtype=dispersive_dtype)

    def calculate_waveform_values(self, G, cythonize=True):
        """Calculates all waveform values for source for duration of simulation.

        Args:
            G: FDTDGrid class describing a grid in a model.
        """

        # Waveform values for sources that need to be calculated on whole timesteps
        self.waveformvalues_wholedt = np.zeros(
            (G.iterations + 1, 3, self.m[3]),
            dtype=config.sim_config.dtypes["float_or_double"],
        )

        # Waveform values for sources that need to be calculated on half timesteps
        self.waveformvalues_halfdt = np.zeros(
            (G.iterations + 1, 3, self.m[3]),
            dtype=config.sim_config.dtypes["float_or_double"],
        )

        # waveform = next(x for x in G.waveforms if x.ID == self.waveformID)
        if cythonize:
            calculate1DWaveformValues(
                self.waveformvalues_wholedt,
                self.waveformvalues_halfdt,
                G.iterations,
                self.m,
                G.dt,
                self.ds,
                self.speed,
                self.start,
                self.stop,
                self.waveform.freq,
                self.waveform.type.encode("UTF-8"),
            )
        else:
            for dimension in range(3):
                for iteration in range(G.iterations + 1):
                    for r in range(self.m[3]):
                        time1 = (
                            G.dt * (iteration + 0.5)
                            - (r + (np.abs(self.m[(dimension + 1) % 3]) + np.abs(self.m[(dimension + 2) % 3])) * 0.5)
                            * self.ds
                            / self.speed
                        )
                        if time1 >= self.start and time1 <= self.stop:
                            # Magnetic fields at half time steps
                            # Set the time of the waveform evaluation to account for any
                            # delay in the start
                            time1 -= self.start
                            self.waveformvalues_halfdt[iteration, dimension, r] = self.waveform.calculate_value(
                                time1, G.dt
                            )

                    for r in range(self.m[3]):
                        time2 = (
                            G.dt * (iteration)
                            - (r + (np.abs(self.m[(dimension)]) + np.abs(self.m[(dimension)])) * 0.5)
                            * self.ds
                            / self.speed
                        )
                        if time2 >= self.start and time2 <= self.stop:
                            # Electric fields at whole time steps
                            # Set the time of the waveform evaluation to account for any
                            # delay in the start
                            time2 -= self.start
                            self.waveformvalues_wholedt[iteration, dimension, r] = self.waveform.calculate_value(
                                time2, G.dt
                            )

    def update_plane_wave_magnetic(
        self,
        nthreads,
        updatecoeffsE,
        updatecoeffsH,
        Ex,
        Ey,
        Ez,
        Hx,
        Hy,
        Hz,
        iteration,
        G,
        cythonize=True,
        precompute=True,
    ):
        if self.axial != 0:

            updatePlaneWave_magnetic_axial(
                self.length,
                self.pml_length,
                nthreads,
                self.skip_axis,
                self.origin_axial,
                self.H_fields,
                self.E_fields,
                self.H_fields_s,
                self.E_fields_s,
                self.Ix,
                self.Iy,
                self.Iz,
                self.Ix0,
                self.Iy0,
                self.Iz0,
                self.Ix_s,
                self.Iy_s,
                self.Iz_s,
                updatecoeffsE[:, :],
                updatecoeffsH[:, :],
                self.ID,
                G.ID,
                self.pml_rex,
                self.pml_rey,
                self.pml_rez,
                self.pml_rhx,
                self.pml_rhy,
                self.pml_rhz,
                self.pml_rex0,
                self.pml_rey0,
                self.pml_rez0,
                self.pml_rhx0,
                self.pml_rhy0,
                self.pml_rhz0,
                Ex,
                Ey,
                Ez,
                Hx,
                Hy,
                Hz,
                self.projections,
                self.waveformvalues_wholedt[:, :, :],
                self.waveformvalues_halfdt[:, :, :],
                self.m,
                self.origin,
                self.corners,
                precompute,
                iteration,
                G.dt,
                G.dx,
                G.dy,
                G.dz,
                self.ds,
                self.speed,
                self.start,
                self.stop,
                self.waveform.freq,
                self.waveform.type.encode("UTF-8"),
            )

        else:

            if cythonize:

                updatePlaneWave_magnetic(
                    self.length,
                    self.pml_length,
                    nthreads,
                    self.skip_axis,
                    self.H_fields,
                    self.E_fields,
                    self.Ix,
                    self.Iy,
                    self.Iz,
                    updatecoeffsE[self.material.numID, :],
                    updatecoeffsH[self.material.numID, :],
                    self.pml_rex,
                    self.pml_rey,
                    self.pml_rez,
                    self.pml_rhx,
                    self.pml_rhy,
                    self.pml_rhz,
                    Ex,
                    Ey,
                    Ez,
                    Hx,
                    Hy,
                    Hz,
                    self.projections,
                    self.waveformvalues_wholedt[:, :, :],
                    self.waveformvalues_halfdt[:, :, :],
                    self.m,
                    self.origin,
                    self.corners,
                    precompute,
                    iteration,
                    G.dt,
                    G.dx,
                    G.dy,
                    G.dz,
                    self.ds,
                    self.speed,
                    self.start,
                    self.stop,
                    self.waveform.freq,
                    self.waveform.type.encode("UTF-8"),
                )
            else:
                self.update_magnetic_field_1D(G, iteration, precompute)
                self.apply_TFSF_conditions_magnetic(G)

    def update_plane_wave_electric(
        self,
        nthreads,
        updatecoeffsE,
        updatecoeffsH,
        Ex,
        Ey,
        Ez,
        Hx,
        Hy,
        Hz,
        iteration,
        G,
        cythonize=True,
        precompute=True,
    ):

        if self.axial != 0:
            updatePlaneWave_electric_axial(
                self.length,
                self.pml_length,
                nthreads,
                self.skip_axis,
                self.origin_axial,
                self.H_fields,
                self.E_fields,
                self.H_fields_s,
                self.E_fields_s,
                self.Ix,
                self.Iy,
                self.Iz,
                self.Ix0,
                self.Iy0,
                self.Iz0,
                self.Ix_s,
                self.Iy_s,
                self.Iz_s,
                updatecoeffsE[:, :],
                updatecoeffsH[:, :],
                self.ID,
                G.ID,
                self.pml_rex,
                self.pml_rey,
                self.pml_rez,
                self.pml_rhx,
                self.pml_rhy,
                self.pml_rhz,
                self.pml_rex0,
                self.pml_rey0,
                self.pml_rez0,
                self.pml_rhx0,
                self.pml_rhy0,
                self.pml_rhz0,
                Ex,
                Ey,
                Ez,
                Hx,
                Hy,
                Hz,
                self.projections,
                self.waveformvalues_wholedt[:, :, :],
                self.waveformvalues_halfdt[:, :, :],
                self.m,
                self.origin,
                self.corners,
                precompute,
                iteration,
                G.dt,
                G.dx,
                G.dy,
                G.dz,
                self.ds,
                self.speed,
                self.start,
                self.stop,
                self.waveform.freq,
                self.waveform.type.encode("UTF-8"),
            )

        else:

            if cythonize:
                updatePlaneWave_electric(
                    self.length,
                    self.pml_length,
                    nthreads,
                    self.skip_axis,
                    self.H_fields,
                    self.E_fields,
                    self.Ix,
                    self.Iy,
                    self.Iz,
                    updatecoeffsE[self.material.numID, :],
                    updatecoeffsH[self.material.numID, :],
                    self.pml_rex,
                    self.pml_rey,
                    self.pml_rez,
                    self.pml_rhx,
                    self.pml_rhy,
                    self.pml_rhz,
                    Ex,
                    Ey,
                    Ez,
                    Hx,
                    Hy,
                    Hz,
                    self.projections,
                    self.waveformvalues_wholedt[:, :, :],
                    self.waveformvalues_halfdt[:, :, :],
                    self.m,
                    self.origin,
                    self.corners,
                    precompute,
                    iteration,
                    G.dt,
                    G.dx,
                    G.dy,
                    G.dz,
                    self.ds,
                    self.speed,
                    self.start,
                    self.stop,
                    self.waveform.freq,
                    self.waveform.type.encode("UTF-8"),
                )
            else:
                self.update_electric_field_1D(G, iteration, precompute)
                self.apply_TFSF_conditions_electric(G)

    def update_plane_wave_electric_dispersive(
        self,
        nthreads,
        updatecoeffsE,
        updatecoeffsH,
        updatecoeffsdispersive,
        Ex,
        Ey,
        Ez,
        Hx,
        Hy,
        Hz,
        iteration,
        G,
        cythonize=True,
        precompute=True,
    ):
        if self.axial != 0:
            updatePlaneWave_electric_dispersive_axial(
                self.length,
                self.pml_length,
                nthreads,
                self.skip_axis,
                self.origin_axial,
                self.H_fields,
                self.E_fields,
                self.H_fields_s,
                self.E_fields_s,
                self.Px,
                self.Py,
                self.Pz,
                self.Px_s,
                self.Py_s,
                self.Pz_s,
                self.Ix,
                self.Iy,
                self.Iz,
                self.Ix0,
                self.Iy0,
                self.Iz0,
                self.Ix_s,
                self.Iy_s,
                self.Iz_s,
                updatecoeffsE[:, :],
                updatecoeffsH[:, :],
                updatecoeffsdispersive[:, :],
                self.ID,
                G.ID,
                self.max_poles,
                self.pml_rex,
                self.pml_rey,
                self.pml_rez,
                self.pml_rhx,
                self.pml_rhy,
                self.pml_rhz,
                self.pml_rex0,
                self.pml_rey0,
                self.pml_rez0,
                self.pml_rhx0,
                self.pml_rhy0,
                self.pml_rhz0,
                Ex,
                Ey,
                Ez,
                Hx,
                Hy,
                Hz,
                self.projections,
                self.waveformvalues_wholedt[:, :, :],
                self.waveformvalues_halfdt[:, :, :],
                self.m,
                self.origin,
                self.corners,
                precompute,
                iteration,
                G.dt,
                G.dx,
                G.dy,
                G.dz,
                self.ds,
                self.speed,
                self.start,
                self.stop,
                self.waveform.freq,
                self.waveform.type.encode("UTF-8"),
            )

        else:

            if cythonize:
                updatePlaneWave_electric_dispersive(
                    self.length,
                    self.pml_length,
                    nthreads,
                    self.skip_axis,
                    self.H_fields,
                    self.E_fields,
                    self.Px,
                    self.Py,
                    self.Pz,
                    self.Ix,
                    self.Iy,
                    self.Iz,
                    updatecoeffsE[self.material.numID, :],
                    updatecoeffsH[self.material.numID, :],
                    updatecoeffsdispersive[self.material.numID, :],
                    self.max_poles,
                    self.pml_rex,
                    self.pml_rey,
                    self.pml_rez,
                    self.pml_rhx,
                    self.pml_rhy,
                    self.pml_rhz,
                    Ex,
                    Ey,
                    Ez,
                    Hx,
                    Hy,
                    Hz,
                    self.projections,
                    self.waveformvalues_wholedt[:, :, :],
                    self.waveformvalues_halfdt[:, :, :],
                    self.m,
                    self.origin,
                    self.corners,
                    precompute,
                    iteration,
                    G.dt,
                    G.dx,
                    G.dy,
                    G.dz,
                    self.ds,
                    self.speed,
                    self.start,
                    self.stop,
                    self.waveform.freq,
                    self.waveform.type.encode("UTF-8"),
                )
            else:
                raise NotImplementedError("Cythonized version not available")

    def initialize_magnetic_fields_1D(self, G, iteration, precompute):
        if precompute:
            for dimension in range(3):
                for r in range(self.m[3]):
                    # Assign source values of magnetic field to first few gridpoints
                    self.H_fields[dimension, r] = (
                        self.projections[dimension] * self.waveformvalues_halfdt[iteration, dimension, r]
                    )
                    # self.getSource(self.real_time - (j+(self.m[(i+1)%3]+self.m[(i+2)%3])*0.5)*self.ds/config.c)#, self.waveformID, G.dt)
        else:
            for dimension in range(3):
                for r in range(self.m[3]):
                    # Assign source values of magnetic field to first few gridpoints
                    self.H_fields[dimension, r] = self.projections[dimension] * getSource(
                        (iteration + 0.5) * G.dt
                        - (r + (self.m[(dimension + 1) % 3] + self.m[(dimension + 2) % 3]) * 0.5)
                        * self.ds
                        / self.speed,
                        self.waveform.freq,
                        self.waveform.type.encode("UTF-8"),
                        G.dt,
                    )

    def initialize_electric_fields_1D(self, G, iteration, precompute):
        if precompute:
            for dimension in range(3):
                for r in range(self.m[3]):
                    # Assign source values of magnetic field to first few gridpoints
                    self.E_fields[dimension, r] = (
                        self.projections[dimension] * self.waveformvalues_wholedt[iteration + 1, dimension, r]
                    )
                    # self.getSource(self.real_time - (j+(self.m[(i+1)%3]+self.m[(i+2)%3])*0.5)*self.ds/config.c)#, self.waveformID, G.dt)
        else:
            for dimension in range(3):
                for r in range(self.m[3]):
                    # Assign source values of magnetic field to first few gridpoints
                    self.E_fields[dimension, r] = self.projections[dimension] * getSource(
                        (iteration + 1) * G.dt - (r + np.abs(self.m[dimension]) * 0.5) * self.ds / self.speed,
                        self.waveform.freq,
                        self.waveform.type.encode("UTF-8"),
                        G.dt,
                    )

    def update_magnetic_field_1D(self, G, iteration, precompute=True):
        """Updates magnetic fields for the next time step using Equation 8 of
            DOI: 10.1109/LAWP.2009.2016851

        Args:
            n: int stores spatial length of the DPW array so that each length
                grid cell is updated when updateMagneticFields() called.
            H_coefficients: double array stores coefficients of the fields in
                            the update equation for the magnetic field.
            H_fields: double array stores magnetic fields of the DPW until
                        temporal index time.
            E_fields: double array stores electric fields of the DPW until
                        temporal index time.
            time: int time index storing current axis number which would be
                    updated for the H_fields.

        Returns:
            H_fields: double array for magnetic field with the axis entry for
                        the current time added.
        """

        self.initialize_magnetic_fields_1D(G, iteration, precompute)

        for i in range(3):  # Update each component of magnetic field
            materialH = G.ID[
                3 + i,
                (self.corners[0] + self.corners[3]) // 2,
                (self.corners[1] + self.corners[4]) // 2,
                (self.corners[2] + self.corners[5]) // 2,
            ]
            # Update magnetic field at each spatial index
            for j in range(self.m[-1], self.length - self.m[-1]):
                self.H_fields[i, j] = (
                    G.updatecoeffsH[materialH, 0] * self.H_fields[i, j]
                    + G.updatecoeffsH[materialH, (i + 2) % 3 + 1]
                    * (self.E_fields[(i + 1) % 3, j + self.m[(i + 2) % 3]] - self.E_fields[(i + 1) % 3, j])
                    - G.updatecoeffsH[materialH, (i + 1) % 3 + 1]
                    * (self.E_fields[(i + 2) % 3, j + self.m[(i + 1) % 3]] - self.E_fields[(i + 2) % 3, j])
                )  # equation 8 of Tan, Potter paper

    def update_electric_field_1D(self, G, iteration, precompute=True):
        """Updates electric fields for the next time step using Equation 9 of
            DOI: 10.1109/LAWP.2009.2016851

        Args:
            n: int stores spatial length of DPW array so that each length grid
                cell is updated when updateMagneticFields() is called.
            E_coefficients: double array stores coefficients of the fields in
                            the update equation for the electric field.
            H_fields: double array stores magnetic fields of the DPW until
                        temporal index time.
            E_fields: double array stores electric fields of the DPW until
                        temporal index time.
            time: int time index storing current axis number which would be
                    updated for the E_fields.

        Returns:
            E_fields: double array for electric field with the axis entry for
                        the current time added.

        """
        self.initialize_electric_fields_1D(G, iteration, precompute)

        for i in range(3):  # Update each component of electric field
            materialE = G.ID[
                i,
                (self.corners[0] + self.corners[3]) // 2,
                (self.corners[1] + self.corners[4]) // 2,
                (self.corners[2] + self.corners[5]) // 2,
            ]
            # Update electric field at each spatial index
            for j in range(self.m[-1], self.length - self.m[-1]):
                self.E_fields[i, j] = (
                    G.updatecoeffsE[materialE, 0] * self.E_fields[i, j]
                    + G.updatecoeffsE[materialE, (i + 2) % 3 + 1]
                    * (self.H_fields[(i + 2) % 3, j] - self.H_fields[(i + 2) % 3, j - self.m[(i + 1) % 3]])
                    - G.updatecoeffsE[materialE, (i + 1) % 3 + 1]
                    * (self.H_fields[(i + 1) % 3, j] - self.H_fields[(i + 1) % 3, j - self.m[(i + 2) % 3]])
                )  # equation 9 of Tan, Potter paper

    def getField(self, i, j, k, array, m, origin, component):
        return array[component, np.dot(m[:-1], np.array([i - origin[0], j - origin[1], k - origin[2]]))]

    def apply_TFSF_conditions_magnetic(self, G):
        if self.skip_axis != 0:
            # **** constant x faces -- scattered-field nodes ****
            i = self.corners[0]
            for j in range(self.corners[1], self.corners[4] + 1):
                for k in range(self.corners[2], self.corners[5]):
                    # correct Hy at firstX-1/2 by subtracting Ez_inc
                    G.Hy[i - 1, j, k] -= G.updatecoeffsH[G.ID[4, i, j, k], 1] * self.getField(
                        i, j, k, self.E_fields, self.m, self.origin, 2
                    )

            for j in range(self.corners[1], self.corners[4]):
                for k in range(self.corners[2], self.corners[5] + 1):
                    # correct Hz at firstX-1/2 by adding Ey_inc
                    G.Hz[i - 1, j, k] += G.updatecoeffsH[G.ID[5, i, j, k], 1] * self.getField(
                        i, j, k, self.E_fields, self.m, self.origin, 1
                    )

            i = self.corners[3]
            for j in range(self.corners[1], self.corners[4] + 1):
                for k in range(self.corners[2], self.corners[5]):
                    # correct Hy at lastX+1/2 by adding Ez_inc
                    G.Hy[i, j, k] += G.updatecoeffsH[G.ID[4, i, j, k], 1] * self.getField(
                        i, j, k, self.E_fields, self.m, self.origin, 2
                    )

            for j in range(self.corners[1], self.corners[4]):
                for k in range(self.corners[2], self.corners[5] + 1):
                    # correct Hz at lastX+1/2 by subtractinging Ey_inc
                    G.Hz[i, j, k] -= G.updatecoeffsH[G.ID[5, i, j, k], 1] * self.getField(
                        i, j, k, self.E_fields, self.m, self.origin, 1
                    )

        if self.skip_axis != 1:
            # **** constant y faces -- scattered-field nodes ****
            j = self.corners[1]
            for i in range(self.corners[0], self.corners[3] + 1):
                for k in range(self.corners[2], self.corners[5]):
                    # correct Hx at firstY-1/2 by adding Ez_inc
                    G.Hx[i, j - 1, k] += G.updatecoeffsH[G.ID[3, i, j, k], 2] * self.getField(
                        i, j, k, self.E_fields, self.m, self.origin, 2
                    )

            for i in range(self.corners[0], self.corners[3]):
                for k in range(self.corners[2], self.corners[5] + 1):
                    # correct Hz at firstY-1/2 by subtracting Ex_inc
                    G.Hz[i, j - 1, k] -= G.updatecoeffsH[G.ID[5, i, j, k], 2] * self.getField(
                        i, j, k, self.E_fields, self.m, self.origin, 0
                    )

            j = self.corners[4]
            for i in range(self.corners[0], self.corners[3] + 1):
                for k in range(self.corners[2], self.corners[5]):
                    # correct Hx at lastY+1/2 by subtracting Ez_inc
                    G.Hx[i, j, k] -= G.updatecoeffsH[G.ID[3, i, j, k], 2] * self.getField(
                        i, j, k, self.E_fields, self.m, self.origin, 2
                    )

            for i in range(self.corners[0], self.corners[3]):
                for k in range(self.corners[2], self.corners[5] + 1):
                    # correct Hz at lastY-1/2 by adding Ex_inc
                    G.Hz[i, j, k] += G.updatecoeffsH[G.ID[5, i, j, k], 2] * self.getField(
                        i, j, k, self.E_fields, self.m, self.origin, 0
                    )

        if self.skip_axis != 2:
            # **** constant z faces -- scattered-field nodes ****
            k = self.corners[2]
            for i in range(self.corners[0], self.corners[3]):
                for j in range(self.corners[1], self.corners[4] + 1):
                    # correct Hy at firstZ-1/2 by adding Ex_inc
                    G.Hy[i, j, k - 1] += G.updatecoeffsH[G.ID[4, i, j, k], 3] * self.getField(
                        i, j, k, self.E_fields, self.m, self.origin, 0
                    )

            for i in range(self.corners[0], self.corners[3] + 1):
                for j in range(self.corners[1], self.corners[4]):
                    # correct Hx at firstZ-1/2 by subtracting Ey_inc
                    G.Hx[i, j, k - 1] -= G.updatecoeffsH[G.ID[3, i, j, k], 3] * self.getField(
                        i, j, k, self.E_fields, self.m, self.origin, 1
                    )

            k = self.corners[5]
            for i in range(self.corners[0], self.corners[3]):
                for j in range(self.corners[1], self.corners[4] + 1):
                    # correct Hy at firstZ-1/2 by subtracting Ex_inc
                    G.Hy[i, j, k] -= G.updatecoeffsH[G.ID[4, i, j, k], 3] * self.getField(
                        i, j, k, self.E_fields, self.m, self.origin, 0
                    )

            for i in range(self.corners[0], self.corners[3] + 1):
                for j in range(self.corners[1], self.corners[4]):
                    # correct Hx at lastZ+1/2 by adding Ey_inc
                    G.Hx[i, j, k] += G.updatecoeffsH[G.ID[3, i, j, k], 3] * self.getField(
                        i, j, k, self.E_fields, self.m, self.origin, 1
                    )

    def apply_TFSF_conditions_electric(self, G):
        if self.skip_axis != 0:
            # **** constant x faces -- total-field nodes ****/
            i = self.corners[0]
            for j in range(self.corners[1], self.corners[4] + 1):
                for k in range(self.corners[2], self.corners[5]):
                    # correct Ez at firstX face by subtracting Hy_inc
                    G.Ez[i, j, k] -= G.updatecoeffsE[G.ID[2, i, j, k], 1] * self.getField(
                        i - 1, j, k, self.H_fields, self.m, self.origin, 1
                    )

            for j in range(self.corners[1], self.corners[4]):
                for k in range(self.corners[2], self.corners[5] + 1):
                    # correct Ey at firstX face by adding Hz_inc
                    G.Ey[i, j, k] += G.updatecoeffsE[G.ID[1, i, j, k], 1] * self.getField(
                        i - 1, j, k, self.H_fields, self.m, self.origin, 2
                    )

            i = self.corners[3]
            for j in range(self.corners[1], self.corners[4] + 1):
                for k in range(self.corners[2], self.corners[5]):
                    # correct Ez at lastX face by adding Hy_inc
                    G.Ez[i, j, k] += G.updatecoeffsE[G.ID[2, i, j, k], 1] * self.getField(
                        i, j, k, self.H_fields, self.m, self.origin, 1
                    )

            i = self.corners[3]
            for j in range(self.corners[1], self.corners[4]):
                for k in range(self.corners[2], self.corners[5] + 1):
                    # correct Ey at lastX face by subtracting Hz_inc
                    G.Ey[i, j, k] -= G.updatecoeffsE[G.ID[1, i, j, k], 1] * self.getField(
                        i, j, k, self.H_fields, self.m, self.origin, 2
                    )

        if self.skip_axis != 1:
            # **** constant y faces -- total-field nodes ****/
            j = self.corners[1]
            for i in range(self.corners[0], self.corners[3] + 1):
                for k in range(self.corners[2], self.corners[5]):
                    # correct Ez at firstY face by adding Hx_inc
                    G.Ez[i, j, k] += G.updatecoeffsE[G.ID[2, i, j, k], 2] * self.getField(
                        i, j - 1, k, self.H_fields, self.m, self.origin, 0
                    )

            for i in range(self.corners[0], self.corners[3]):
                for k in range(self.corners[2], self.corners[5] + 1):
                    # correct Ex at firstY face by subtracting Hz_inc
                    G.Ex[i, j, k] -= G.updatecoeffsE[G.ID[0, i, j, k], 2] * self.getField(
                        i, j - 1, k, self.H_fields, self.m, self.origin, 2
                    )

            j = self.corners[4]
            for i in range(self.corners[0], self.corners[3] + 1):
                for k in range(self.corners[2], self.corners[5]):
                    # correct Ez at lastY face by subtracting Hx_inc
                    G.Ez[i, j, k] -= G.updatecoeffsE[G.ID[2, i, j, k], 2] * self.getField(
                        i, j, k, self.H_fields, self.m, self.origin, 0
                    )

            for i in range(self.corners[0], self.corners[3]):
                for k in range(self.corners[2], self.corners[5] + 1):
                    # correct Ex at lastY face by adding Hz_inc
                    G.Ex[i, j, k] += G.updatecoeffsE[G.ID[0, i, j, k], 2] * self.getField(
                        i, j, k, self.H_fields, self.m, self.origin, 2
                    )

        if self.skip_axis != 2:
            # **** constant z faces -- total-field nodes ****/
            k = self.corners[2]
            for i in range(self.corners[0], self.corners[3] + 1):
                for j in range(self.corners[1], self.corners[4]):
                    # correct Ey at firstZ face by subtracting Hx_inc
                    G.Ey[i, j, k] -= G.updatecoeffsE[G.ID[1, i, j, k], 3] * self.getField(
                        i, j, k - 1, self.H_fields, self.m, self.origin, 0
                    )

            for i in range(self.corners[0], self.corners[3]):
                for j in range(self.corners[1], self.corners[4] + 1):
                    # correct Ex at firstZ face by adding Hy_inc
                    G.Ex[i, j, k] += G.updatecoeffsE[G.ID[0, i, j, k], 3] * self.getField(
                        i, j, k - 1, self.H_fields, self.m, self.origin, 1
                    )

            k = self.corners[5]
            for i in range(self.corners[0], self.corners[3] + 1):
                for j in range(self.corners[1], self.corners[4]):
                    # correct Ey at lastZ face by adding Hx_inc
                    G.Ey[i, j, k] += G.updatecoeffsE[G.ID[1, i, j, k], 3] * self.getField(
                        i, j, k, self.H_fields, self.m, self.origin, 0
                    )

            for i in range(self.corners[0], self.corners[3]):
                for j in range(self.corners[1], self.corners[4] + 1):
                    # correct Ex at lastZ face by subtracting Hy_inc
                    G.Ex[i, j, k] -= G.updatecoeffsE[G.ID[0, i, j, k], 3] * self.getField(
                        i, j, k, self.H_fields, self.m, self.origin, 1
                    )

    def find_dpw_integers_optimized(self, theta_deg, phi_deg, delta_xyz, max_total_error_deg):
        """
        Finds the OPTIMAL smallest integer vector (mx, my, mz) for a DPW source
        by generating all candidates and selecting the simplest valid one.
         --- Parameters ---
        theta_deg : float
            Polar angle in degrees (0 to 180) from the +Z axis.
        phi_deg : float
            Azimuthal angle in degrees (0 to 360) from the +X axis.
        delta_xyz : list or tuple
            Grid step sizes [dx, dy, dz] in your simulation units.
        max_total_error_deg : float
            Maximum acceptable TOTAL 3D angular error in degrees.

        --- Returns ---
        m_vec : numpy.ndarray
            The optimal 1x3 integer vector [mx, my, mz]. Returns None if no solution is found.
        actual_angles_deg : tuple
            The actual (theta, phi) angles of the vector, in degrees.
        errors_deg : tuple
            The geometrically correct projected error components (d_theta, d_phi), in degrees.
        total_error_deg : float
            The final total 3D angular error of the returned vector, in degrees.
        """

        # --- Helper Function to calculate continued fraction convergents ---
        def continued_fractions(x, n_terms=15):  # Reduced to 15 to avoid flint warning
            """Computes the continued fraction convergents of a number x."""
            convergents = []
            p_prev, q_prev = 0, 1
            p_curr, q_curr = 1, 0
            xi = x
            for _ in range(n_terms):
                a = math.floor(xi)
                p_next = a * p_curr + p_prev
                q_next = a * q_curr + q_prev
                convergents.append((p_next, q_next))
                if abs(xi - a) < 1e-12:
                    return convergents
                xi = 1 / (xi - a)
                p_prev, q_prev = p_curr, q_curr
                p_curr, q_curr = p_next, q_next
            return convergents

        #  Prepare inputs for calculations converting angles from degrees (which are easy for humans) to radians
        #  required by math functions and unpacking the grid spacing vector into individual variables.
        theta_rad = math.radians(theta_deg)
        phi_rad = math.radians(phi_deg)

        #  Convert the spherical angles (theta, phi) into a standard 3D (x,y,z)
        #  vector. This vector is the "target direction" of the plane wave. It
        #  is automatically a "unit vector" (length of 1).
        u_vec_target = np.array(
            [math.sin(theta_rad) * math.cos(phi_rad), math.sin(theta_rad) * math.sin(phi_rad), math.cos(theta_rad)]
        )

        #  Snap floating-point residue on components that are analytically
        #  zero (e.g. cos(90 deg) evaluates to ~6.1e-17, not 0). Without
        #  this, the continued-fraction search below sees a tiny-but-nonzero
        #  target ratio and generates astronomically large integer
        #  candidates chasing it (overflowing on integer conversion). An
        #  exactly-zero component is also precisely what 2D in-plane
        #  propagation requires (e.g. theta = 90 for a mode invariant in z).
        u_vec_target[np.abs(u_vec_target) < 1e-12] = 0.0

        #  The algorithm works by dividing by one of the vector's components (x, y, or z).
        #  To avoid dividing by zero and to keep the math stable, we always choose
        #  the component with the LARGEST absolute value as the reference. We
        #  temporarily rearrange (permute) the vectors so this largest component
        #  is always in the 3rd position (the "z" position). Keep track of how
        #  it has been rearranged so it can be undone later.

        ref_idx = np.argmax(np.abs(u_vec_target))
        perm_order = [0, 1, 2]  # Corresponds to x, y, z
        if ref_idx != 2:
            perm_order[2], perm_order[ref_idx] = perm_order[ref_idx], perm_order[2]

        u_perm = u_vec_target[perm_order]
        d_perm = np.array(delta_xyz)[perm_order]

        #  The direction of a wave on the grid depends on both the integers (mx, my, mz)
        #  AND the grid spacing (dx,dy,dz). To find the correct integers, we must
        #  pre-compensate for the grid spacing. We create two target ratios that
        #  tell us what the ideal ratios of (mx/mz) and (my/mz) should be.
        #  We pre-compensate for the grid spacing to find the correct integer ratios.
        ratio_1 = (u_perm[0] / u_perm[2]) * (d_perm[0] / d_perm[2])
        ratio_2 = (u_perm[1] / u_perm[2]) * (d_perm[1] / d_perm[2])

        #  The target ratios are decimal numbers. We need to find simple integer
        #  fractions that as close as possible to these decimals.
        #  This is done using "continued fractions". A list of these best-guess fractions
        #  is generated using the helper function for continued fractions (called "convergents").
        convergents1 = continued_fractions(ratio_1)
        convergents2 = continued_fractions(ratio_2)

        #  Search through the lists of "best guess" fractions to find a pair
        #  that works well and meets our error tolerance. We start with the
        #  simplest fractions first and gradually move to more complex ones.
        candidates = []
        for p1, q1 in convergents1:
            for p2, q2 in convergents2:
                common_denom = math.lcm(q1, q2)
                # Guard against pathological convergents (from ratios that
                # are irrational-like or extreme): integers this large can
                # never be useful DPW mappings (the 1D vector length scales
                # with max|m|) and would overflow the C long conversion
                # below. Computed in Python ints, so this check itself
                # cannot overflow.
                if max(abs(p1 * (common_denom // q1)), abs(p2 * (common_denom // q2)), common_denom) > 10**6:
                    continue
                m_perm = np.array([p1 * (common_denom // q1), p2 * (common_denom // q2), common_denom], dtype=int)

                # Apply the crucial sign correction for the correct quadrant
                if np.sign(u_perm[2]) < 0:
                    m_perm = -m_perm

                # Undo the permutation from Step 3
                m_vec_candidate = np.zeros(3, dtype=int)
                m_vec_candidate[perm_order] = m_perm

                # Calculate the candidate's total error
                phys_vec = m_vec_candidate / np.array(delta_xyz)
                phys_vec_norm = phys_vec / (np.linalg.norm(phys_vec) or 1)
                dot_prod = np.clip(np.dot(phys_vec_norm, u_vec_target), -1.0, 1.0)
                total_error_deg = math.degrees(math.acos(dot_prod))

                # Store the candidate with its error and "size" metric. The size
                # is the largest integer component, a good measure of cost.
                size = np.max(np.abs(m_vec_candidate))
                candidates.append({"m_vec": m_vec_candidate, "error": total_error_deg, "size": size})

        # From our list, we keep only those that meet the error criteria, then
        # sort them by size to find the one with the smallest integers.
        valid_candidates = [c for c in candidates if c["error"] <= max_total_error_deg]

        if not valid_candidates:
            print("Warning: No DPW solution found within the error tolerance.")
            return None, (math.nan, math.nan), (math.nan, math.nan), math.nan

        # Sort the valid solutions by size (smallest integers first)
        valid_candidates.sort(key=lambda c: c["size"])
        best_candidate = valid_candidates[0]
        m_vec = best_candidate["m_vec"]
        total_error_deg = best_candidate["error"]
        max_m = best_candidate["size"]

        # We perform the final detailed error calculation for the winning vector.
        phys_vec = m_vec / np.array(delta_xyz)
        phys_vec_norm = phys_vec / np.linalg.norm(phys_vec)

        # Define local spherical basis vectors for error projection
        u_theta = np.array(
            [math.cos(theta_rad) * math.cos(phi_rad), math.cos(theta_rad) * math.sin(phi_rad), -math.sin(theta_rad)]
        )
        u_phi = np.array([-math.sin(phi_rad), math.cos(phi_rad), 0])

        # Project the 3D error vector onto the basis vectors
        diff_vec = phys_vec_norm - u_vec_target
        errors_deg = (math.degrees(np.dot(diff_vec, u_theta)), math.degrees(np.dot(diff_vec, u_phi)))

        # Calculate the final angles for user information
        actual_angles_deg = (
            math.degrees(math.acos(np.clip(phys_vec_norm[2], -1.0, 1.0))),
            math.degrees(math.atan2(phys_vec_norm[1], phys_vec_norm[0])),
        )

        return m_vec, actual_angles_deg, errors_deg, total_error_deg

    def _get_pml_parameters(self, G):
        """
        Calculates and sets the DPW PML parameters based on RIPML formulation.
                The forumlation can handle full CFS PML parameters but these are not needed and cannot be set by the user.
                Hence only sigma_max is calculated here based on the standard formula for PMLs. The other parameters are set to values that disable their grading
                but they can be edited here for testing purposes.

                This method uses NumPy vectorization for high performance.
        """

        if self.axial == 0:
            Z = self.materialZ

            # --- PML Configuration ---
            Order = 4
            KOrder = 1
            AOrder = 1
            # Sigma Max is calcualted in the same way to the main grid PMls. It must take into account the
            # actual physical step size of the DPW grid when calcualting the step value which is not just ds.
            sigma_max = 0.8 * (Order + 1) / (Z * np.sqrt(self.m[0] ** 2 + self.m[1] ** 2 + self.m[2] ** 2) * self.ds)
            Kappa_max = (
                1.0  # No kappa grading for DPW as it is not needed. You can change this value for testing purposes.
            )
            Alpha_max = (
                0.0  # No alpha grading for DPW as it is not needed. You can change this value for testing purposes.
            )

            # --- Create helper arrays for vectorized calculations ---
            # 'depth' array runs from 0 to PMLSize-1  (for sigma and kappa calculations)
            depth = np.arange(self.pml_length)
            # 'i_arr' array runs from PMLSize down to 0 (for alpha calculations)
            i_arr = np.arange(self.pml_length, 0, -1)

            # --- E-Field PML Parameters (Vectorized) ---
            sEx_base = (depth + self.m[0] * 0.5) / self.pml_length
            aEx_base = (i_arr + self.m[0] * 0.5) / self.pml_length
            sEx = sigma_max * np.maximum(0, sEx_base) ** Order
            kEx = 1.0 + (Kappa_max - 1.0) * sEx_base**KOrder
            aEx = Alpha_max * aEx_base**AOrder

            sEy_base = (depth + self.m[1] * 0.5) / self.pml_length
            aEy_base = (i_arr + self.m[1] * 0.5) / self.pml_length
            sEy = sigma_max * np.maximum(0, sEy_base) ** Order
            kEy = 1.0 + (Kappa_max - 1.0) * sEy_base**KOrder
            aEy = Alpha_max * aEy_base**AOrder

            sEz_base = (depth + self.m[2] * 0.5) / self.pml_length
            aEz_base = (i_arr + self.m[2] * 0.5) / self.pml_length
            sEz = sigma_max * np.maximum(0, sEz_base) ** Order
            kEz = 1.0 + (Kappa_max - 1.0) * sEz_base**KOrder
            aEz = Alpha_max * aEz_base**AOrder

            # --- H-Field PML Parameters (Vectorized) ---
            sHx_base = (depth + (self.m[1] + self.m[2]) * 0.5) / self.pml_length
            aHx_base = (i_arr + (self.m[1] + self.m[2]) * 0.5) / self.pml_length
            sHx = sigma_max * np.maximum(0, sHx_base) ** Order
            kHx = 1.0 + (Kappa_max - 1.0) * sHx_base**KOrder
            aHx = Alpha_max * aHx_base**AOrder

            sHy_base = (depth + (self.m[0] + self.m[2]) * 0.5) / self.pml_length
            aHy_base = (i_arr + (self.m[0] + self.m[2]) * 0.5) / self.pml_length
            sHy = sigma_max * np.maximum(0, sHy_base) ** Order
            kHy = 1.0 + (Kappa_max - 1.0) * sHy_base**KOrder
            aHy = Alpha_max * aHy_base**AOrder

            sHz_base = (depth + (self.m[0] + self.m[1]) * 0.5) / self.pml_length
            aHz_base = (i_arr + (self.m[0] + self.m[1]) * 0.5) / self.pml_length
            sHz = sigma_max * np.maximum(0, sHz_base) ** Order
            kHz = 1.0 + (Kappa_max - 1.0) * sHz_base**KOrder
            aHz = Alpha_max * aHz_base**AOrder

            # --- Final Update Coefficients (Vectorized) ---
            # Denominators for E and H field updates
            den_Ex = 2 * config.e0 * kEx + G.dt * kEx * aEx + G.dt * sEx
            den_Ey = 2 * config.e0 * kEy + G.dt * kEy * aEy + G.dt * sEy
            den_Ez = 2 * config.e0 * kEz + G.dt * kEz * aEz + G.dt * sEz
            den_Hx = 2 * config.e0 * kHx + G.dt * kHx * aHx + G.dt * sHx
            den_Hy = 2 * config.e0 * kHy + G.dt * kHy * aHy + G.dt * sHy
            den_Hz = 2 * config.e0 * kHz + G.dt * kHz * aHz + G.dt * sHz

            # RA Coefficients
            RAEx = (2 * config.e0 * (1 - kEx) + G.dt * aEx * (1 - kEx) - G.dt * sEx) / den_Ex
            RAEy = (2 * config.e0 * (1 - kEy) + G.dt * aEy * (1 - kEy) - G.dt * sEy) / den_Ey
            RAEz = (2 * config.e0 * (1 - kEz) + G.dt * aEz * (1 - kEz) - G.dt * sEz) / den_Ez
            RAHx = (2 * config.e0 * (1 - kHx) + G.dt * aHx * (1 - kHx) - G.dt * sHx) / den_Hx
            RAHy = (2 * config.e0 * (1 - kHy) + G.dt * aHy * (1 - kHy) - G.dt * sHy) / den_Hy
            RAHz = (2 * config.e0 * (1 - kHz) + G.dt * aHz * (1 - kHz) - G.dt * sHz) / den_Hz

            # RB Coefficients
            RBEx, RBEy, RBEz = 2 / den_Ex, 2 / den_Ey, 2 / den_Ez
            RBHx, RBHy, RBHz = 2 / den_Hx, 2 / den_Hy, 2 / den_Hz

            # RC Coefficients
            RCEx = G.dt * (kEx * aEx + sEx)
            RCEy = G.dt * (kEy * aEy + sEy)
            RCEz = G.dt * (kEz * aEz + sEz)
            RCHx = G.dt * (kHx * aHx + sHx)
            RCHy = G.dt * (kHy * aHy + sHy)
            RCHz = G.dt * (kHz * aHz + sHz)

            # RD Coefficients
            RDEx = G.dt * (aEx * (1 - kEx) - sEx)
            RDEy = G.dt * (aEy * (1 - kEy) - sEy)
            RDEz = G.dt * (aEz * (1 - kEz) - sEz)
            RDHx = G.dt * (aHx * (1 - kHx) - sHx)
            RDHy = G.dt * (aHy * (1 - kHy) - sHy)
            RDHz = G.dt * (aHz * (1 - kHz) - sHz)

            # --- Combine Coefficients into Single Matrices ---
            # Creates 2D arrays (4 rows x pml_length columns) for the PML coefficients RA row: 0, RB row: 1, RC rowe:2 and RD row: 3 for the Ex,Ey,Ez,
            # Hz, Hy, Hz components
            self.pml_rex = np.array(
                [RAEx, RBEx, RCEx, RDEx], order="C", dtype=config.sim_config.dtypes["float_or_double"]
            )
            self.pml_rey = np.array(
                [RAEy, RBEy, RCEy, RDEy], order="C", dtype=config.sim_config.dtypes["float_or_double"]
            )
            self.pml_rez = np.array(
                [RAEz, RBEz, RCEz, RDEz], order="C", dtype=config.sim_config.dtypes["float_or_double"]
            )

            self.pml_rhx = np.array(
                [RAHx, RBHx, RCHx, RDHx], order="C", dtype=config.sim_config.dtypes["float_or_double"]
            )
            self.pml_rhy = np.array(
                [RAHy, RBHy, RCHy, RDHy], order="C", dtype=config.sim_config.dtypes["float_or_double"]
            )
            self.pml_rhz = np.array(
                [RAHz, RBHz, RCHz, RDHz], order="C", dtype=config.sim_config.dtypes["float_or_double"]
            )

        else:
            Z = self.materialPMLZ

            # --- PML Configuration ---
            Order = 4
            KOrder = 1
            AOrder = 1
            # Sigma Max is calcualted in the same way to the main grid PMls. It must take into account the
            # actual physical step size of the DPW grid when calcualting the step value which is not just ds.
            sigma_max = 0.8 * (Order + 1) / (Z * np.sqrt(self.m[0] ** 2 + self.m[1] ** 2 + self.m[2] ** 2) * self.ds)
            Kappa_max = (
                1.0  # No kappa grading for DPW as it is not needed. You can change this value for testing purposes.
            )
            Alpha_max = (
                0.0  # No alpha grading for DPW as it is not needed. You can change this value for testing purposes.
            )

            # --- Create helper arrays for vectorized calculations ---
            # 'depth' array runs from 0 to PMLSize-1  (for sigma and kappa calculations)
            depth = np.arange(self.pml_length)
            # 'i_arr' array runs from PMLSize down to 0 (for alpha calculations)
            i_arr = np.arange(self.pml_length, 0, -1)

            # --- E-Field PML Parameters (Vectorized) ---
            sEx_base = (depth + self.m[0] * 0.5) / self.pml_length
            aEx_base = (i_arr + self.m[0] * 0.5) / self.pml_length
            sEx = sigma_max * np.maximum(0, sEx_base) ** Order
            kEx = 1.0 + (Kappa_max - 1.0) * sEx_base**KOrder
            aEx = Alpha_max * aEx_base**AOrder

            sEy_base = (depth + self.m[1] * 0.5) / self.pml_length
            aEy_base = (i_arr + self.m[1] * 0.5) / self.pml_length
            sEy = sigma_max * np.maximum(0, sEy_base) ** Order
            kEy = 1.0 + (Kappa_max - 1.0) * sEy_base**KOrder
            aEy = Alpha_max * aEy_base**AOrder

            sEz_base = (depth + self.m[2] * 0.5) / self.pml_length
            aEz_base = (i_arr + self.m[2] * 0.5) / self.pml_length
            sEz = sigma_max * np.maximum(0, sEz_base) ** Order
            kEz = 1.0 + (Kappa_max - 1.0) * sEz_base**KOrder
            aEz = Alpha_max * aEz_base**AOrder

            # --- H-Field PML Parameters (Vectorized) ---
            sHx_base = (depth + (self.m[1] + self.m[2]) * 0.5) / self.pml_length
            aHx_base = (i_arr + (self.m[1] + self.m[2]) * 0.5) / self.pml_length
            sHx = sigma_max * np.maximum(0, sHx_base) ** Order
            kHx = 1.0 + (Kappa_max - 1.0) * sHx_base**KOrder
            aHx = Alpha_max * aHx_base**AOrder

            sHy_base = (depth + (self.m[0] + self.m[2]) * 0.5) / self.pml_length
            aHy_base = (i_arr + (self.m[0] + self.m[2]) * 0.5) / self.pml_length
            sHy = sigma_max * np.maximum(0, sHy_base) ** Order
            kHy = 1.0 + (Kappa_max - 1.0) * sHy_base**KOrder
            aHy = Alpha_max * aHy_base**AOrder

            sHz_base = (depth + (self.m[0] + self.m[1]) * 0.5) / self.pml_length
            aHz_base = (i_arr + (self.m[0] + self.m[1]) * 0.5) / self.pml_length
            sHz = sigma_max * np.maximum(0, sHz_base) ** Order
            kHz = 1.0 + (Kappa_max - 1.0) * sHz_base**KOrder
            aHz = Alpha_max * aHz_base**AOrder

            # --- Final Update Coefficients (Vectorized) ---
            # Denominators for E and H field updates
            den_Ex = 2 * config.e0 * kEx + G.dt * kEx * aEx + G.dt * sEx
            den_Ey = 2 * config.e0 * kEy + G.dt * kEy * aEy + G.dt * sEy
            den_Ez = 2 * config.e0 * kEz + G.dt * kEz * aEz + G.dt * sEz
            den_Hx = 2 * config.e0 * kHx + G.dt * kHx * aHx + G.dt * sHx
            den_Hy = 2 * config.e0 * kHy + G.dt * kHy * aHy + G.dt * sHy
            den_Hz = 2 * config.e0 * kHz + G.dt * kHz * aHz + G.dt * sHz

            # RA Coefficients
            RAEx = (2 * config.e0 * (1 - kEx) + G.dt * aEx * (1 - kEx) - G.dt * sEx) / den_Ex
            RAEy = (2 * config.e0 * (1 - kEy) + G.dt * aEy * (1 - kEy) - G.dt * sEy) / den_Ey
            RAEz = (2 * config.e0 * (1 - kEz) + G.dt * aEz * (1 - kEz) - G.dt * sEz) / den_Ez
            RAHx = (2 * config.e0 * (1 - kHx) + G.dt * aHx * (1 - kHx) - G.dt * sHx) / den_Hx
            RAHy = (2 * config.e0 * (1 - kHy) + G.dt * aHy * (1 - kHy) - G.dt * sHy) / den_Hy
            RAHz = (2 * config.e0 * (1 - kHz) + G.dt * aHz * (1 - kHz) - G.dt * sHz) / den_Hz

            # RB Coefficients
            RBEx, RBEy, RBEz = 2 / den_Ex, 2 / den_Ey, 2 / den_Ez
            RBHx, RBHy, RBHz = 2 / den_Hx, 2 / den_Hy, 2 / den_Hz

            # RC Coefficients
            RCEx = G.dt * (kEx * aEx + sEx)
            RCEy = G.dt * (kEy * aEy + sEy)
            RCEz = G.dt * (kEz * aEz + sEz)
            RCHx = G.dt * (kHx * aHx + sHx)
            RCHy = G.dt * (kHy * aHy + sHy)
            RCHz = G.dt * (kHz * aHz + sHz)

            # RD Coefficients
            RDEx = G.dt * (aEx * (1 - kEx) - sEx)
            RDEy = G.dt * (aEy * (1 - kEy) - sEy)
            RDEz = G.dt * (aEz * (1 - kEz) - sEz)
            RDHx = G.dt * (aHx * (1 - kHx) - sHx)
            RDHy = G.dt * (aHy * (1 - kHy) - sHy)
            RDHz = G.dt * (aHz * (1 - kHz) - sHz)

            # --- Combine Coefficients into Single Matrices ---
            # Creates 2D arrays (4 rows x pml_length columns) for the PML coefficients RA row: 0, RB row: 1, RC rowe:2 and RD row: 3 for the Ex,Ey,Ez,
            # Hz, Hy, Hz components
            self.pml_rex = np.array(
                [RAEx, RBEx, RCEx, RDEx], order="C", dtype=config.sim_config.dtypes["float_or_double"]
            )
            self.pml_rey = np.array(
                [RAEy, RBEy, RCEy, RDEy], order="C", dtype=config.sim_config.dtypes["float_or_double"]
            )
            self.pml_rez = np.array(
                [RAEz, RBEz, RCEz, RDEz], order="C", dtype=config.sim_config.dtypes["float_or_double"]
            )

            self.pml_rhx = np.array(
                [RAHx, RBHx, RCHx, RDHx], order="C", dtype=config.sim_config.dtypes["float_or_double"]
            )
            self.pml_rhy = np.array(
                [RAHy, RBHy, RCHy, RDHy], order="C", dtype=config.sim_config.dtypes["float_or_double"]
            )
            self.pml_rhz = np.array(
                [RAHz, RBHz, RCHz, RDHz], order="C", dtype=config.sim_config.dtypes["float_or_double"]
            )

            # --- Repeat for PML0 ---

            Z = self.materialPML0Z

            # --- PML Configuration ---
            Order = 4
            KOrder = 1
            AOrder = 1
            # Sigma Max is calcualted in the same way to the main grid PMls. It must take into account the
            # actual physical step size of the DPW grid when calcualting the step value which is not just ds.
            sigma_max = 0.8 * (Order + 1) / (Z * np.sqrt(self.m[0] ** 2 + self.m[1] ** 2 + self.m[2] ** 2) * self.ds)
            Kappa_max = (
                1.0  # No kappa grading for DPW as it is not needed. You can change this value for testing purposes.
            )
            Alpha_max = (
                0.0  # No alpha grading for DPW as it is not needed. You can change this value for testing purposes.
            )

            # --- Create helper arrays for vectorized calculations ---
            # 'depth' array runs from 0 to PMLSize-1  (for sigma and kappa calculations)
            depth = np.arange(self.pml_length)
            # 'i_arr' array runs from PMLSize down to 0 (for alpha calculations)
            i_arr = np.arange(self.pml_length, 0, -1)

            # --- E-Field PML Parameters (Vectorized) ---
            sEx_base = (depth + self.m[0] * 0.5) / self.pml_length
            aEx_base = (i_arr + self.m[0] * 0.5) / self.pml_length
            sEx = sigma_max * np.maximum(0, sEx_base) ** Order
            kEx = 1.0 + (Kappa_max - 1.0) * sEx_base**KOrder
            aEx = Alpha_max * aEx_base**AOrder

            sEy_base = (depth + self.m[1] * 0.5) / self.pml_length
            aEy_base = (i_arr + self.m[1] * 0.5) / self.pml_length
            sEy = sigma_max * np.maximum(0, sEy_base) ** Order
            kEy = 1.0 + (Kappa_max - 1.0) * sEy_base**KOrder
            aEy = Alpha_max * aEy_base**AOrder

            sEz_base = (depth + self.m[2] * 0.5) / self.pml_length
            aEz_base = (i_arr + self.m[2] * 0.5) / self.pml_length
            sEz = sigma_max * np.maximum(0, sEz_base) ** Order
            kEz = 1.0 + (Kappa_max - 1.0) * sEz_base**KOrder
            aEz = Alpha_max * aEz_base**AOrder

            # --- H-Field PML Parameters (Vectorized) ---
            sHx_base = (depth + (self.m[1] + self.m[2]) * 0.5) / self.pml_length
            aHx_base = (i_arr + (self.m[1] + self.m[2]) * 0.5) / self.pml_length
            sHx = sigma_max * np.maximum(0, sHx_base) ** Order
            kHx = 1.0 + (Kappa_max - 1.0) * sHx_base**KOrder
            aHx = Alpha_max * aHx_base**AOrder

            sHy_base = (depth + (self.m[0] + self.m[2]) * 0.5) / self.pml_length
            aHy_base = (i_arr + (self.m[0] + self.m[2]) * 0.5) / self.pml_length
            sHy = sigma_max * np.maximum(0, sHy_base) ** Order
            kHy = 1.0 + (Kappa_max - 1.0) * sHy_base**KOrder
            aHy = Alpha_max * aHy_base**AOrder

            sHz_base = (depth + (self.m[0] + self.m[1]) * 0.5) / self.pml_length
            aHz_base = (i_arr + (self.m[0] + self.m[1]) * 0.5) / self.pml_length
            sHz = sigma_max * np.maximum(0, sHz_base) ** Order
            kHz = 1.0 + (Kappa_max - 1.0) * sHz_base**KOrder
            aHz = Alpha_max * aHz_base**AOrder

            # --- Final Update Coefficients (Vectorized) ---
            # Denominators for E and H field updates
            den_Ex = 2 * config.e0 * kEx + G.dt * kEx * aEx + G.dt * sEx
            den_Ey = 2 * config.e0 * kEy + G.dt * kEy * aEy + G.dt * sEy
            den_Ez = 2 * config.e0 * kEz + G.dt * kEz * aEz + G.dt * sEz
            den_Hx = 2 * config.e0 * kHx + G.dt * kHx * aHx + G.dt * sHx
            den_Hy = 2 * config.e0 * kHy + G.dt * kHy * aHy + G.dt * sHy
            den_Hz = 2 * config.e0 * kHz + G.dt * kHz * aHz + G.dt * sHz

            # RA Coefficients
            RAEx = (2 * config.e0 * (1 - kEx) + G.dt * aEx * (1 - kEx) - G.dt * sEx) / den_Ex
            RAEy = (2 * config.e0 * (1 - kEy) + G.dt * aEy * (1 - kEy) - G.dt * sEy) / den_Ey
            RAEz = (2 * config.e0 * (1 - kEz) + G.dt * aEz * (1 - kEz) - G.dt * sEz) / den_Ez
            RAHx = (2 * config.e0 * (1 - kHx) + G.dt * aHx * (1 - kHx) - G.dt * sHx) / den_Hx
            RAHy = (2 * config.e0 * (1 - kHy) + G.dt * aHy * (1 - kHy) - G.dt * sHy) / den_Hy
            RAHz = (2 * config.e0 * (1 - kHz) + G.dt * aHz * (1 - kHz) - G.dt * sHz) / den_Hz

            # RB Coefficients
            RBEx, RBEy, RBEz = 2 / den_Ex, 2 / den_Ey, 2 / den_Ez
            RBHx, RBHy, RBHz = 2 / den_Hx, 2 / den_Hy, 2 / den_Hz

            # RC Coefficients
            RCEx = G.dt * (kEx * aEx + sEx)
            RCEy = G.dt * (kEy * aEy + sEy)
            RCEz = G.dt * (kEz * aEz + sEz)
            RCHx = G.dt * (kHx * aHx + sHx)
            RCHy = G.dt * (kHy * aHy + sHy)
            RCHz = G.dt * (kHz * aHz + sHz)

            # RD Coefficients
            RDEx = G.dt * (aEx * (1 - kEx) - sEx)
            RDEy = G.dt * (aEy * (1 - kEy) - sEy)
            RDEz = G.dt * (aEz * (1 - kEz) - sEz)
            RDHx = G.dt * (aHx * (1 - kHx) - sHx)
            RDHy = G.dt * (aHy * (1 - kHy) - sHy)
            RDHz = G.dt * (aHz * (1 - kHz) - sHz)

            # --- Combine Coefficients into Single Matrices ---
            # Creates 2D arrays (4 rows x pml_length columns) for the PML coefficients RA row: 0, RB row: 1, RC rowe:2 and RD row: 3 for the Ex,Ey,Ez,
            # Hz, Hy, Hz components
            self.pml_rex0 = np.array(
                [RAEx, RBEx, RCEx, RDEx], order="C", dtype=config.sim_config.dtypes["float_or_double"]
            )
            self.pml_rey0 = np.array(
                [RAEy, RBEy, RCEy, RDEy], order="C", dtype=config.sim_config.dtypes["float_or_double"]
            )
            self.pml_rez0 = np.array(
                [RAEz, RBEz, RCEz, RDEz], order="C", dtype=config.sim_config.dtypes["float_or_double"]
            )

            self.pml_rhx0 = np.array(
                [RAHx, RBHx, RCHx, RDHx], order="C", dtype=config.sim_config.dtypes["float_or_double"]
            )
            self.pml_rhy0 = np.array(
                [RAHy, RBHy, RCHy, RDHy], order="C", dtype=config.sim_config.dtypes["float_or_double"]
            )
            self.pml_rhz0 = np.array(
                [RAHz, RBHz, RCHz, RDHz], order="C", dtype=config.sim_config.dtypes["float_or_double"]
            )
