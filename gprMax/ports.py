# Copyright (C) 2026: The University of Edinburgh, United Kingdom
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <http://www.gnu.org/licenses/>.

"""Frequency-domain powers for terminal and modal antenna ports.

Voltage-source ports use the known Thevenin generator voltage and the electric
field sampled on one electric Yee edge. Transmission-line sources already
carry incident and terminal voltage/current histories, so their S11 and input
impedance are calculated automatically without an additional receiver.
Eigenmode ports retain their native generalized multi-mode travelling-wave
coordinates and their generally non-diagonal power matrices.
"""

import logging
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
import numpy.typing as npt

import gprMax.config as config
from gprMax.ntff.conventions import (
    FORWARD_TRANSFORM_KERNEL,
    PHASOR_TIME_DEPENDENCE,
    engineering_dft,
)

if TYPE_CHECKING:
    from gprMax.grid.fdtd_grid import FDTDGrid
    from gprMax.model import Model
    from gprMax.receivers import Rx
    from gprMax.sources import MagneticFrillSource, TransmissionLine, VoltageSource


logger = logging.getLogger(__name__)

DEFAULT_MINIMUM_WAVELENGTH_CELLS = 10.0
MINIMUM_PHYSICAL_WAVELENGTH_CELLS = 3.0
DEFAULT_INCIDENT_FLOOR_DB = -40.0

SpectrumLimit = Union[float, Literal["nyquist"]]


@dataclass(frozen=True)
class PortPowerSpectrum:
    """Native port phasors and spectral powers at arbitrary frequencies.

    The Fourier amplitudes carry the common pulse-transform scale used by
    KSIR. Consequently the power-like quantities are intended for ratios;
    their common transform scale cancels in gain and efficiency.
    """

    port_id: str
    source_type: str
    reference_impedance: float
    frequency: npt.NDArray[np.floating]
    incident_voltage: npt.NDArray[np.complexfloating]
    terminal_voltage: npt.NDArray[np.complexfloating]
    terminal_current: npt.NDArray[np.complexfloating]
    incident_power: npt.NDArray[np.floating]
    accepted_power: npt.NDArray[np.floating]
    mesh_valid: npt.NDArray[np.bool_]
    terminal_valid: npt.NDArray[np.bool_]
    representation: str = "terminal_voltage_current"
    mode_indices: tuple[int, ...] = ()
    incident_modal_amplitudes: Optional[npt.NDArray[np.complexfloating]] = None
    outgoing_modal_amplitudes: Optional[npt.NDArray[np.complexfloating]] = None
    mode_power_matrix: Optional[npt.NDArray[np.complexfloating]] = None
    mode_cross_power_matrix: Optional[npt.NDArray[np.complexfloating]] = None
    modal_valid: Optional[npt.NDArray[np.bool_]] = None


@dataclass(frozen=True)
class PortOutputBinding:
    """A finalised port output together with the grid that sampled it."""

    output: object
    grid: "FDTDGrid"


def _port_mesh_valid(output, grid, frequencies):
    if (
        getattr(output, "spectrum_limit", None) == "nyquist"
        or getattr(output, "spectrum_limit_mode", None) == "nyquist"
    ):
        return np.ones(np.asarray(frequencies).shape, dtype=bool)
    cells, _ = minimum_wavelength_sampling(grid, frequencies)
    minimum = float(getattr(output, "minimum_wavelength_cells", DEFAULT_MINIMUM_WAVELENGTH_CELLS))
    return np.asarray(cells >= minimum, dtype=bool)


def _power_spectrum_result(
    output,
    grid,
    frequencies,
    incident_voltage,
    terminal_voltage,
    terminal_current,
    terminal_valid,
) -> PortPowerSpectrum:
    real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
    complex_dtype = np.dtype(config.sim_config.dtypes["complex"])
    frequency = np.asarray(frequencies, dtype=real_dtype)
    incident_voltage = np.asarray(incident_voltage, dtype=complex_dtype)
    terminal_voltage = np.asarray(terminal_voltage, dtype=complex_dtype)
    terminal_current = np.asarray(terminal_current, dtype=complex_dtype)
    reference_impedance = float(output.reference_impedance)
    incident_power = np.asarray(
        np.abs(incident_voltage) ** 2 / (2 * reference_impedance),
        dtype=real_dtype,
    )
    accepted_power = np.asarray(
        0.5 * np.real(terminal_voltage * np.conj(terminal_current)),
        dtype=real_dtype,
    )
    finite = (
        np.isfinite(incident_voltage)
        & np.isfinite(terminal_voltage)
        & np.isfinite(terminal_current)
        & np.isfinite(incident_power)
        & np.isfinite(accepted_power)
    )
    return PortPowerSpectrum(
        port_id=output.output_id,
        source_type=type(output.source).__name__,
        reference_impedance=reference_impedance,
        frequency=frequency,
        incident_voltage=incident_voltage,
        terminal_voltage=terminal_voltage,
        terminal_current=terminal_current,
        incident_power=incident_power,
        accepted_power=accepted_power,
        mesh_valid=_port_mesh_valid(output, grid, frequency),
        terminal_valid=np.asarray(terminal_valid, dtype=bool) & finite,
    )


def _hard_source_gap_admittance(output, frequency, dt, complex_dtype):
    """Return the Yee-gap admittance at magnetic half-step times.

    The hard-source voltage is an integer-time quantity. The Ampere-loop
    current is sampled half a step earlier, and its DFT carries that physical
    time offset. The trapezoidal conductive term therefore contributes
    ``G*cos(omega*dt/2)`` and the centred displacement difference contributes
    ``j*2*C*sin(omega*dt/2)/dt``.
    """

    half_phase = np.pi * np.asarray(frequency) * dt
    return np.asarray(
        output.background_conductance * np.cos(half_phase)
        + 1j * (2 * output.gap_capacitance / dt) * np.sin(half_phase),
        dtype=complex_dtype,
    )


def _finite_source_gap_admittance(output, frequency, dt, complex_dtype):
    """Return the background Yee-gap admittance for a resistive source.

    Nondispersive media retain the original bilinear-frequency expression.
    For a dispersive background, evaluate the material's complete complex
    permittivity at that same warped frequency. This includes its physical
    conductivity and every Debye, Lorentz, Drude, or inclusive pole without
    requiring an additional time-domain current history.
    """

    frequency = np.asarray(frequency, dtype=np.float64)
    omega_discrete = (2 / dt) * np.tan(np.pi * frequency * dt)
    nyquist = np.isclose(
        frequency * dt,
        0.5,
        rtol=0,
        atol=4 * np.finfo(np.float64).eps,
    )
    if not output.background_is_dispersive:
        admittance = np.asarray(
            output.background_conductance + 1j * omega_discrete * output.gap_capacitance,
            dtype=complex_dtype,
        )
        # The bilinear frequency is singular at the exact Nyquist bin. Keep
        # the research frequency axis, but mark this one correction undefined
        # instead of allowing its enormous floating-point approximation to
        # influence validity at independent, lower-frequency bins.
        admittance[nyquist] = np.nan + 1j * np.nan
        return admittance

    admittance = np.empty(frequency.shape, dtype=np.complex128)
    zero = frequency == 0
    admittance[zero] = output.background_conductance
    positive = ~zero
    if np.any(positive):
        effective_frequency = omega_discrete[positive] / (2 * np.pi)
        epsilon_r = np.asarray(
            output.background_material.calculate_er(effective_frequency),
            dtype=np.complex128,
        )
        admittance[positive] = (
            1j
            * omega_discrete[positive]
            * config.sim_config.em_consts["e0"]
            * epsilon_r
            * output.area
            / output.dl
        )
    admittance[nyquist] = np.nan + 1j * np.nan
    return np.asarray(admittance, dtype=complex_dtype)


def evaluate_port_power_spectrum(
    output,
    grid: "FDTDGrid",
    frequencies: npt.ArrayLike,
    *,
    window: str = "rectangular",
) -> PortPowerSpectrum:
    """Evaluate one supported port at the exact antenna-transform frequencies.

    Conventional ports use terminal voltage/current rather than S11, so an
    unexcited but coupled port remains measurable. Eigenmode ports use their
    native incident/outgoing coefficient vectors and cross-power matrices.
    Only a rectangular transform is currently accepted because delayed
    radiated and port histories would otherwise receive different window
    weights.
    """

    if window != "rectangular":
        raise ValueError("port-power evaluation currently requires a rectangular transform window")
    real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
    complex_dtype = np.dtype(config.sim_config.dtypes["complex"])
    frequency = np.asarray(frequencies, dtype=real_dtype)
    if frequency.ndim != 1 or frequency.size == 0:
        raise ValueError("port-power frequencies must be one-dimensional and non-empty")
    if not np.all(np.isfinite(frequency)) or np.any(frequency < 0):
        raise ValueError("port-power frequencies must be finite and non-negative")
    if output.result is None:
        raise RuntimeError(f"port {output.output_id!r} has not been finalised")

    from gprMax.eigenmode_ports import EigenmodePortMonitor

    if isinstance(output, EigenmodePortMonitor):
        if not np.array_equal(frequency, output.result.frequency):
            raise ValueError(
                f"eigenmode port {output.output_id!r} DFT frequencies must exactly "
                "match the antenna transform frequencies"
            )
        incident_modal = np.asarray(output.result.incident, dtype=complex_dtype)
        outgoing_modal = np.asarray(output.result.outgoing, dtype=complex_dtype)
        power_matrix = np.asarray(output.power_matrix, dtype=complex_dtype)
        cross_power_matrix = np.asarray(output.electric_gram, dtype=complex_dtype)
        if incident_modal.shape != outgoing_modal.shape:
            raise ValueError(f"eigenmode port {output.output_id!r} has inconsistent modal arrays")
        if incident_modal.shape != (len(output.mode_indices), frequency.size):
            raise ValueError(
                f"eigenmode port {output.output_id!r} has inconsistent modal dimensions"
            )
        if power_matrix.shape != (
            frequency.size,
            len(output.mode_indices),
            len(output.mode_indices),
        ):
            raise ValueError(
                f"eigenmode port {output.output_id!r} has an inconsistent power matrix"
            )
        if cross_power_matrix.shape != power_matrix.shape:
            raise ValueError(
                f"eigenmode port {output.output_id!r} has an inconsistent cross-power matrix"
            )

        accepted_power = np.zeros(frequency.shape, dtype=real_dtype)
        incident_power = np.zeros(frequency.shape, dtype=real_dtype)
        excitation_positions = np.empty(0, dtype=np.intp)
        if output.is_source:
            excitation_modes = getattr(output, "excitation_mode_indices", None)
            if excitation_modes is None:
                excitation_mode = getattr(output, "excitation_mode_index", None)
                excitation_modes = () if excitation_mode is None else (excitation_mode,)
            try:
                excitation_positions = np.asarray(
                    [output.mode_indices.index(mode) for mode in excitation_modes],
                    dtype=np.intp,
                )
            except ValueError as exc:
                raise ValueError(
                    f"eigenmode port {output.output_id!r} has a driven mode that is "
                    "not included in its monitored mode indices"
                ) from exc

        power_wave_valid_value = getattr(output, "power_wave_valid", None)
        if power_wave_valid_value is None:
            power_wave_valid_value = output.mode_power_valid
        power_wave_valid = np.asarray(power_wave_valid_value, dtype=bool)
        result_valid = np.asarray(output.result.valid, dtype=bool)
        power_matrix_valid = np.asarray(output.power_matrix_valid, dtype=bool)
        modal_valid = result_valid & power_wave_valid.T & power_matrix_valid[np.newaxis, :]
        physical_power_valid = np.zeros(frequency.shape, dtype=bool)
        for frequency_index in range(frequency.size):
            physical_modes = np.flatnonzero(power_wave_valid[frequency_index])
            if (
                physical_modes.size == 0
                or not power_matrix_valid[frequency_index]
                or not np.all(result_valid[physical_modes, frequency_index])
            ):
                continue
            if excitation_positions.size and not np.all(
                power_wave_valid[frequency_index, excitation_positions]
            ):
                continue

            incident = incident_modal[physical_modes, frequency_index]
            outgoing = outgoing_modal[physical_modes, frequency_index]
            physical_power_matrix = power_matrix[frequency_index][
                np.ix_(physical_modes, physical_modes)
            ]
            physical_cross_power_matrix = cross_power_matrix[frequency_index][
                np.ix_(physical_modes, physical_modes)
            ]
            if not (
                np.all(np.isfinite(incident))
                and np.all(np.isfinite(outgoing))
                and np.all(np.isfinite(physical_power_matrix))
                and np.all(np.isfinite(physical_cross_power_matrix))
            ):
                continue

            total_electric = incident + outgoing
            total_magnetic = incident - outgoing
            accepted_value = np.real(
                np.vdot(
                    total_magnetic,
                    physical_cross_power_matrix @ total_electric,
                )
            )
            incident_value = 0.0
            if excitation_positions.size:
                local_excitation_positions = np.asarray(
                    [
                        int(np.flatnonzero(physical_modes == position)[0])
                        for position in excitation_positions
                    ],
                    dtype=np.intp,
                )
                excitation_amplitudes = incident[local_excitation_positions]
                excitation_power_matrix = physical_power_matrix[
                    np.ix_(local_excitation_positions, local_excitation_positions)
                ]
                incident_value = np.real(
                    np.vdot(
                        excitation_amplitudes,
                        excitation_power_matrix @ excitation_amplitudes,
                    )
                )
            if not np.isfinite(accepted_value) or not np.isfinite(incident_value):
                continue

            accepted_power[frequency_index] = accepted_value
            incident_power[frequency_index] = incident_value
            physical_power_valid[frequency_index] = True

        # Generalized coefficients are retained below cutoff for modal/S
        # diagnostics, but they are not power waves and therefore do not
        # participate in the adapter-level quadratic forms or validity check.
        nan_phasor = np.full(frequency.shape, np.nan + 1j * np.nan, dtype=complex_dtype)
        return PortPowerSpectrum(
            port_id=output.output_id,
            source_type=type(output.owner).__name__,
            reference_impedance=np.nan,
            frequency=frequency,
            incident_voltage=nan_phasor,
            terminal_voltage=nan_phasor.copy(),
            terminal_current=nan_phasor.copy(),
            incident_power=incident_power,
            accepted_power=accepted_power,
            mesh_valid=_port_mesh_valid(output, grid, frequency),
            terminal_valid=physical_power_valid,
            representation="modal_power_waves",
            mode_indices=output.mode_indices,
            incident_modal_amplitudes=incident_modal,
            outgoing_modal_amplitudes=outgoing_modal,
            mode_power_matrix=power_matrix,
            mode_cross_power_matrix=cross_power_matrix,
            modal_valid=modal_valid,
        )

    if isinstance(output, VoltageSourcePortMonitor):
        result = output.result
        if output.hard_source:
            terminal_voltage = engineering_dft(
                result.total_voltage,
                frequency,
                grid.dt,
                time_offset=output.hard_voltage_time_offset,
            )
            loop_current = engineering_dft(
                output._hard_loop_current,
                frequency,
                grid.dt,
                time_offset=output.hard_current_time_offset,
            )
            gap_admittance = _hard_source_gap_admittance(
                output,
                frequency,
                grid.dt,
                complex_dtype,
            )
            terminal_current = np.asarray(
                loop_current - gap_admittance * terminal_voltage,
                dtype=complex_dtype,
            )
            incident_voltage = np.asarray(
                0.5 * (terminal_voltage + output.reference_impedance * terminal_current),
                dtype=complex_dtype,
            )
            return _power_spectrum_result(
                output,
                grid,
                frequency,
                incident_voltage,
                terminal_voltage,
                terminal_current,
                np.ones(frequency.shape, dtype=bool),
            )

        terminal_voltage = engineering_dft(
            result.total_voltage,
            frequency,
            grid.dt,
            time_offset=0.5 * grid.dt,
        )
        generator_voltage = engineering_dft(
            result.generator_voltage,
            frequency,
            grid.dt,
            time_offset=0.5 * grid.dt,
        )
        incident_voltage = np.asarray(0.5 * generator_voltage, dtype=complex_dtype)
        omega_discrete = (2 / grid.dt) * np.tan(np.pi * frequency * grid.dt)
        gap_admittance = np.asarray(
            output.background_conductance + 1j * omega_discrete * output.gap_capacitance,
            dtype=complex_dtype,
        )
        source_current = (generator_voltage - terminal_voltage) / output.reference_impedance
        terminal_current = np.asarray(
            source_current - gap_admittance * terminal_voltage,
            dtype=complex_dtype,
        )
        nyquist = np.isclose(
            frequency,
            1 / (2 * grid.dt),
            rtol=64 * np.finfo(real_dtype).eps,
            atol=0,
        )
        terminal_valid = ~(nyquist & (output.gap_capacitance != 0))
        return _power_spectrum_result(
            output,
            grid,
            frequency,
            incident_voltage,
            terminal_voltage,
            terminal_current,
            terminal_valid,
        )

    if isinstance(output, RationalNetworkPortOutput):
        result = output.result
        terminal_voltage = engineering_dft(
            result.total_voltage,
            frequency,
            grid.dt,
            time_offset=0.5 * grid.dt,
        )
        network_current = engineering_dft(
            result.network_current,
            frequency,
            grid.dt,
            time_offset=0.5 * grid.dt,
        )
        omega_discrete = (2 / grid.dt) * np.tan(np.pi * frequency * grid.dt)
        gap_admittance = np.asarray(
            output.background_conductance + 1j * omega_discrete * output.gap_capacitance,
            dtype=complex_dtype,
        )
        terminal_current = np.asarray(
            -network_current - gap_admittance * terminal_voltage,
            dtype=complex_dtype,
        )
        incident_voltage = np.asarray(
            0.5 * (terminal_voltage + output.reference_impedance * terminal_current),
            dtype=complex_dtype,
        )
        nyquist = np.isclose(
            frequency,
            1 / (2 * grid.dt),
            rtol=64 * np.finfo(real_dtype).eps,
            atol=0,
        )
        terminal_valid = ~(nyquist & (output.gap_capacitance != 0))
        return _power_spectrum_result(
            output,
            grid,
            frequency,
            incident_voltage,
            terminal_voltage,
            terminal_current,
            terminal_valid,
        )

    if isinstance(output, TransmissionLinePortOutput):
        source = output.source
        incident_voltage = engineering_dft(source.Vinc, frequency, grid.dt)
        terminal_voltage = engineering_dft(source.Vtotal, frequency, grid.dt)
        terminal_current = np.asarray(
            (2 * incident_voltage - terminal_voltage) / output.reference_impedance,
            dtype=complex_dtype,
        )
        return _power_spectrum_result(
            output,
            grid,
            frequency,
            incident_voltage,
            terminal_voltage,
            terminal_current,
            np.ones(frequency.shape, dtype=bool),
        )

    if isinstance(output, MagneticFrillPortOutput):
        source = output.source
        incident_voltage = engineering_dft(source.Vinc, frequency, grid.dt)
        terminal_voltage = engineering_dft(source.Vtotal, frequency, grid.dt)
        terminal_current = engineering_dft(source.Itot, frequency, grid.dt)
        return _power_spectrum_result(
            output,
            grid,
            frequency,
            incident_voltage,
            terminal_voltage,
            terminal_current,
            np.ones(frequency.shape, dtype=bool),
        )

    raise TypeError(f"unsupported antenna port output type {type(output).__name__}")


def modal_power_spectrum(amplitudes, power_matrix):
    """Return c^H W c for modal amplitudes arranged as (mode, frequency)."""

    values = np.einsum(
        "mf,fmn,nf->f",
        np.conj(np.asarray(amplitudes)),
        np.asarray(power_matrix),
        np.asarray(amplitudes),
        optimize=True,
    )
    return np.asarray(np.real(values))


def modal_net_power_spectrum(incident, outgoing, cross_power_matrix):
    """Return Re{(a-b)^H G_E (a+b)} at each modal-port frequency."""

    incident = np.asarray(incident)
    outgoing = np.asarray(outgoing)
    total_e_coeff = incident + outgoing
    total_h_coeff = incident - outgoing
    values = np.einsum(
        "mf,fmn,nf->f",
        np.conj(total_h_coeff),
        np.asarray(cross_power_matrix),
        total_e_coeff,
        optimize=True,
    )
    return np.asarray(np.real(values))


def port_output_registry(grid: "FDTDGrid") -> dict[str, object]:
    """Return every finalised port output using one unambiguous ID namespace."""

    outputs = list(getattr(grid, "port_monitors", ()))
    outputs.extend(
        source.port_output
        for source in getattr(grid, "transmissionlines", ())
        if getattr(source, "port_output", None) is not None
    )
    outputs.extend(getattr(grid, "eigenmodeports", ()))
    outputs.extend(
        source.port_output
        for source in getattr(grid, "magneticfrillsources", ())
        if getattr(source, "port_output", None) is not None
    )
    registry = {}
    for output in outputs:
        if output.output_id in registry:
            raise ValueError(f"antenna port ID {output.output_id!r} is ambiguous")
        registry[output.output_id] = output
    return registry


def model_port_ids(model: "Model") -> tuple[str, ...]:
    """Return physical port references across the main grid and subgrids.

    Main-grid IDs retain their public spelling. Subgrid IDs are qualified as
    ``<subgrid ID>/<local port ID>`` so per-grid automatic IDs such as ``tl1``
    and ``frill1`` remain unambiguous.
    """

    references = []
    for grid in (model.G, *model.subgrids):
        local_ids = [monitor.output_id for monitor in getattr(grid, "port_monitors", ())]
        local_ids.extend(
            f"tl{index}" for index, _ in enumerate(getattr(grid, "transmissionlines", ()), start=1)
        )
        local_ids.extend(
            f"frill{index}"
            for index, _ in enumerate(getattr(grid, "magneticfrillsources", ()), start=1)
        )
        local_ids.extend(monitor.output_id for monitor in getattr(grid, "eigenmodeports", ()))
        if len(set(local_ids)) != len(local_ids):
            location = "main grid" if grid is model.G else f"subgrid {grid.name!r}"
            raise ValueError(f"antenna port IDs are ambiguous on the {location}")
        prefix = "" if grid is model.G else f"{grid.name}/"
        references.extend(f"{prefix}{port_id}" for port_id in local_ids)
    if len(set(references)) != len(references):
        raise ValueError("antenna port references are ambiguous across model grids")
    return tuple(references)


def model_port_output_registry(model: "Model") -> dict[str, PortOutputBinding]:
    """Return finalised, grid-aware port outputs for an entire model."""

    registry = {}
    for grid in (model.G, *model.subgrids):
        prefix = "" if grid is model.G else f"{grid.name}/"
        for local_id, output in port_output_registry(grid).items():
            port_id = f"{prefix}{local_id}"
            if port_id in registry:
                raise ValueError(f"antenna port reference {port_id!r} is ambiguous")
            registry[port_id] = PortOutputBinding(output=output, grid=grid)
    return registry


def validate_spectrum_limit(value) -> SpectrumLimit:
    """Validate the public spectrum-limit tagged value.

    A finite numeric value is interpreted as cells per shortest material
    wavelength. The string ``"nyquist"`` is the explicit research override.
    """

    if isinstance(value, str):
        if value.lower() == "nyquist":
            return "nyquist"
        raise ValueError("spectrum_limit must be a finite number or 'nyquist'")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError("spectrum_limit must be a finite number or 'nyquist'")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("spectrum_limit must be a finite number or 'nyquist'")
    if value < MINIMUM_PHYSICAL_WAVELENGTH_CELLS:
        raise ValueError(
            "numeric spectrum_limit must be at least "
            f"{MINIMUM_PHYSICAL_WAVELENGTH_CELLS:g} cells per wavelength"
        )
    return value


def engineering_rfft(
    samples: npt.ArrayLike,
    dt: float,
    *,
    time_offset: float = 0.0,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.complexfloating]]:
    """Return native rFFT bins using gprMax's engineering convention."""

    real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
    complex_dtype = np.dtype(config.sim_config.dtypes["complex"])
    values = np.asarray(samples, dtype=real_dtype)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("port time samples must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("port time samples must be finite")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("port sample interval must be finite and positive")
    if not np.isfinite(time_offset):
        raise ValueError("port time offset must be finite")

    frequencies64 = np.fft.rfftfreq(values.size, d=dt)
    transformed = np.fft.rfft(values)
    phase = np.exp(-2j * np.pi * frequencies64 * time_offset)
    spectrum = np.asarray(dt * phase * transformed, dtype=complex_dtype)
    return np.asarray(frequencies64, dtype=real_dtype), spectrum


def _complex_relative_permittivity(material, frequencies):
    """Return material relative permittivity including conductivity."""

    values = np.empty(frequencies.shape, dtype=np.complex128)
    zero = frequencies == 0
    values[zero] = complex(material.er)
    positive = ~zero
    if not np.any(positive):
        return values
    if hasattr(material, "poles"):
        values[positive] = np.asarray(
            material.calculate_er(frequencies[positive]), dtype=np.complex128
        )
    else:
        omega = 2 * np.pi * frequencies[positive]
        values[positive] = material.er + material.se / (
            1j * omega * config.sim_config.em_consts["e0"]
        )
    return values


def _complex_relative_permeability(material, frequencies):
    """Return material relative permeability including magnetic loss."""

    values = np.empty(frequencies.shape, dtype=np.complex128)
    zero = frequencies == 0
    values[zero] = complex(material.mr)
    positive = ~zero
    if np.any(positive):
        omega = 2 * np.pi * frequencies[positive]
        values[positive] = material.mr + material.sm / (
            1j * omega * config.sim_config.em_consts["m0"]
        )
    return values


def minimum_wavelength_sampling(
    grid: "FDTDGrid",
    frequencies: npt.ArrayLike,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.str_]]:
    """Calculate cells per shortest material wavelength at every frequency.

    PEC, PMC, and voltage-source-generated materials are excluded. For the
    ordinary nonmagnetic/nondispersive case this reduces exactly to using the
    largest relative permittivity in the model. The complex refractive-index
    magnitude gives a conservative spatial-scale estimate for lossy and
    dispersive media.
    """

    real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
    frequencies64 = np.asarray(frequencies, dtype=np.float64)
    if frequencies64.ndim != 1 or frequencies64.size == 0:
        raise ValueError("frequencies must be a non-empty one-dimensional array")
    if np.any(frequencies64 < 0) or not np.all(np.isfinite(frequencies64)):
        raise ValueError("frequencies must be finite and non-negative")

    delta = float(max(grid.dx, grid.dy, grid.dz))
    cells = np.full(frequencies64.shape, np.inf, dtype=np.float64)
    limiting = np.full(frequencies64.shape, "", dtype=object)
    materials_found = 0

    for material in grid.materials:
        material_type = str(getattr(material, "type", "")).lower()
        if (
            "voltage-source" in material_type
            or getattr(material, "is_pec", False)
            or material.ID == "pmc"
            or not np.isfinite(material.se)
            or not np.isfinite(material.sm)
        ):
            continue

        epsilon_r = _complex_relative_permittivity(material, frequencies64)
        mu_r = _complex_relative_permeability(material, frequencies64)
        refractive_index = np.abs(np.sqrt(epsilon_r * mu_r))
        material_cells = np.full(frequencies64.shape, np.inf, dtype=np.float64)
        positive = frequencies64 > 0
        finite = positive & np.isfinite(refractive_index) & (refractive_index > 0)
        material_cells[finite] = config.sim_config.em_consts["c"] / (
            frequencies64[finite] * refractive_index[finite] * delta
        )
        update = material_cells < cells
        cells[update] = material_cells[update]
        limiting[update] = material.ID
        materials_found += 1

    if materials_found == 0:
        raise ValueError("no propagating material is available for port frequency validation")
    return np.asarray(cells, dtype=real_dtype), np.asarray(limiting, dtype=str)


def _safe_complex_divide(numerator, denominator, complex_dtype):
    """Divide where the denominator is algebraically resolvable.

    The threshold is deliberately much lower than the source-band validity
    floor. Finite but unreliable research values are retained and separately
    identified by validity masks.
    """

    numerator = np.asarray(numerator, dtype=complex_dtype)
    denominator = np.asarray(denominator, dtype=complex_dtype)
    magnitude = np.abs(denominator)
    finite_denominator = np.isfinite(denominator)
    # Every element is an independent frequency/port quotient. A global
    # maximum is unsuitable here: one very large (for example near-Nyquist)
    # denominator can otherwise invalidate ordinary O(1) denominators at all
    # other frequencies. Retain every finite, representable research value;
    # source/mesh reliability is described by separate validity masks.
    threshold = np.finfo(np.empty((), dtype=complex_dtype).real.dtype).tiny
    defined = finite_denominator & (magnitude > threshold)
    result = np.full(numerator.shape, np.nan + 1j * np.nan, dtype=complex_dtype)
    np.divide(numerator, denominator, out=result, where=defined)
    defined &= np.isfinite(result)
    return result, defined


def correct_s11_for_parallel_gap(s11_source, gap_correction, complex_dtype=None):
    """Remove a dimensionless parallel gap admittance from source-plane S11."""

    if complex_dtype is None:
        complex_dtype = np.dtype(config.sim_config.dtypes["complex"])
    else:
        complex_dtype = np.dtype(complex_dtype)
    s11_source = np.asarray(s11_source, dtype=complex_dtype)
    gap_correction = np.asarray(gap_correction, dtype=complex_dtype)
    numerator = 2 * s11_source + gap_correction * (1 + s11_source)
    denominator = 2 - gap_correction * (1 + s11_source)
    return _safe_complex_divide(numerator, denominator, complex_dtype)


def correct_smatrix_for_parallel_gaps(s_source, gap_correction, complex_dtype=None):
    """Remove the Yee-gap shunt admittances from a multiport S matrix.

    ``s_source`` uses power-wave normalisation and has shape
    ``(nfrequency, nport, nport)``. ``gap_correction`` is the dimensionless
    diagonal admittance ``Z0 * Ygap`` for every frequency and port, with
    shape ``(nfrequency, nport)``. The conversion is performed through the
    normalised admittance matrix so coupling terms are corrected consistently
    rather than treating every matrix element as an independent S11.

    Returns the corrected matrix and one validity flag per frequency.
    """

    if complex_dtype is None:
        complex_dtype = np.dtype(config.sim_config.dtypes["complex"])
    else:
        complex_dtype = np.dtype(complex_dtype)
    source = np.asarray(s_source, dtype=complex_dtype)
    correction = np.asarray(gap_correction, dtype=complex_dtype)
    if source.ndim != 3 or source.shape[1] != source.shape[2]:
        raise ValueError("s_source must have shape (nfrequency, nport, nport)")
    if correction.shape != source.shape[:2]:
        raise ValueError("gap_correction must have shape (nfrequency, nport)")

    corrected = np.full(source.shape, np.nan + 1j * np.nan, dtype=complex_dtype)
    valid = np.zeros(source.shape[0], dtype=bool)
    identity = np.eye(source.shape[1], dtype=complex_dtype)
    for index, (matrix, gaps) in enumerate(zip(source, correction)):
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(gaps)):
            continue
        try:
            # y_source = (I - S_source) (I + S_source)^-1. The transpose
            # form applies the right-hand inverse using numpy's left solve.
            y_source = np.linalg.solve(
                (identity + matrix).T,
                (identity - matrix).T,
            ).T
            y_corrected = y_source - np.diag(gaps)
            matrix_corrected = np.linalg.solve(
                identity + y_corrected,
                identity - y_corrected,
            )
        except np.linalg.LinAlgError:
            continue
        if np.all(np.isfinite(matrix_corrected)):
            corrected[index] = np.asarray(matrix_corrected, dtype=complex_dtype)
            valid[index] = True
    return corrected, valid


def impedance_from_s11(s11, reference_impedance, complex_dtype=None):
    """Calculate input impedance from S11 with a separate validity mask."""

    if complex_dtype is None:
        complex_dtype = np.dtype(config.sim_config.dtypes["complex"])
    else:
        complex_dtype = np.dtype(complex_dtype)
    s11 = np.asarray(s11, dtype=complex_dtype)
    return _safe_complex_divide(reference_impedance * (1 + s11), 1 - s11, complex_dtype)


def admittance_from_s11(s11, reference_impedance, complex_dtype=None):
    """Calculate input admittance from S11 with a separate validity mask."""

    if complex_dtype is None:
        complex_dtype = np.dtype(config.sim_config.dtypes["complex"])
    else:
        complex_dtype = np.dtype(complex_dtype)
    s11 = np.asarray(s11, dtype=complex_dtype)
    return _safe_complex_divide(1 - s11, reference_impedance * (1 + s11), complex_dtype)


@dataclass(frozen=True)
class VoltageSourcePortResult:
    """Final time- and frequency-domain result for one voltage-source port."""

    time: npt.NDArray[np.floating]
    generator_voltage: npt.NDArray[np.floating]
    total_voltage: npt.NDArray[np.floating]
    frequency: npt.NDArray[np.floating]
    incident_spectrum: npt.NDArray[np.complexfloating]
    reflected_source_spectrum: npt.NDArray[np.complexfloating]
    total_spectrum: npt.NDArray[np.complexfloating]
    gap_correction: npt.NDArray[np.complexfloating]
    s11_source: npt.NDArray[np.complexfloating]
    zin_source: npt.NDArray[np.complexfloating]
    s11: npt.NDArray[np.complexfloating]
    zin: npt.NDArray[np.complexfloating]
    yin: npt.NDArray[np.complexfloating]
    source_valid: npt.NDArray[np.bool_]
    mesh_valid: npt.NDArray[np.bool_]
    gap_correction_valid: npt.NDArray[np.bool_]
    valid_s11: npt.NDArray[np.bool_]
    valid_zin: npt.NDArray[np.bool_]
    valid_yin: npt.NDArray[np.bool_]
    incident_relative_db: npt.NDArray[np.floating]
    cells_per_minimum_wavelength: npt.NDArray[np.floating]
    tail_relative_db: float


@dataclass(frozen=True)
class RationalNetworkPortResult:
    """Final time- and frequency-domain result for a rational network terminal."""

    time: npt.NDArray[np.floating]
    generator_voltage: npt.NDArray[np.floating]
    total_voltage: npt.NDArray[np.floating]
    network_current: npt.NDArray[np.floating]
    frequency: npt.NDArray[np.floating]
    generator_spectrum: npt.NDArray[np.complexfloating]
    total_voltage_spectrum: npt.NDArray[np.complexfloating]
    network_current_spectrum: npt.NDArray[np.complexfloating]
    terminal_current_spectrum: npt.NDArray[np.complexfloating]
    incident_voltage_spectrum: npt.NDArray[np.complexfloating]
    reflected_voltage_spectrum: npt.NDArray[np.complexfloating]
    s11: npt.NDArray[np.complexfloating]
    zin: npt.NDArray[np.complexfloating]
    yin: npt.NDArray[np.complexfloating]
    source_valid: npt.NDArray[np.bool_]
    mesh_valid: npt.NDArray[np.bool_]
    gap_correction_valid: npt.NDArray[np.bool_]
    valid_s11: npt.NDArray[np.bool_]
    valid_zin: npt.NDArray[np.bool_]
    valid_yin: npt.NDArray[np.bool_]
    incident_relative_db: npt.NDArray[np.floating]
    cells_per_minimum_wavelength: npt.NDArray[np.floating]
    tail_relative_db: float


class RationalNetworkPortOutput:
    """Post-process a sparse rational-network terminal as an antenna port."""

    def __init__(
        self,
        output_id: str,
        terminal,
        reference_impedance: float,
        spectrum_limit: SpectrumLimit = DEFAULT_MINIMUM_WAVELENGTH_CELLS,
        owner=None,
    ):
        self.output_id = output_id
        self.source = terminal
        self.terminal = terminal
        self.reference_impedance = float(reference_impedance)
        self.spectrum_limit = validate_spectrum_limit(spectrum_limit)
        self.owner = owner
        self.result: Optional[RationalNetworkPortResult] = None
        self.prepared = False
        self.minimum_wavelength_cells = (
            DEFAULT_MINIMUM_WAVELENGTH_CELLS
            if self.spectrum_limit == "nyquist"
            else float(self.spectrum_limit)
        )
        self.spectrum_limit_mode = (
            "nyquist" if self.spectrum_limit == "nyquist" else "minimum_wavelength_cells"
        )
        self.incident_floor_db = DEFAULT_INCIDENT_FLOOR_DB

    def rebind_after_mpi_gather(self, grid: "FDTDGrid") -> None:
        """Reconnect the gathered output to its coordinator-side terminal."""

        terminals = [
            terminal
            for terminal in getattr(grid, "networkterminals", ())
            if terminal.ID == self.output_id
        ]
        if len(terminals) != 1:
            raise RuntimeError(
                f"NetworkPort {self.output_id!r} could not uniquely rebind its MPI "
                f"terminal ({len(terminals)} terminal(s))"
            )
        self.terminal = terminals[0]
        self.source = self.terminal
        self.terminal.output = self
        # Coordinates have been converted to global indices by the terminal
        # gather; replace the owner-rank-local position cached by prepare().
        self.source_position = np.asarray(
            self.terminal.coord * np.asarray((grid.dx, grid.dy, grid.dz)),
            dtype=np.float64,
        )

    def prepare(self, grid: "FDTDGrid") -> None:
        if not self.terminal.prepared:
            self.terminal.prepare(grid)
        if not np.isfinite(self.reference_impedance) or self.reference_impedance <= 0:
            raise ValueError(
                f"network port {self.output_id!r} requires a finite, positive reference impedance"
            )
        if grid.iterations < 2:
            raise ValueError(f"network port {self.output_id!r} requires at least two iterations")

        self.dl = float(self.terminal.dl)
        self.area = float(self.terminal.area)
        self.background_relative_permittivity = float(
            self.terminal.background_relative_permittivity
        )
        self.background_conductivity = float(self.terminal.background_conductivity)
        self.gap_capacitance = (
            config.sim_config.em_consts["e0"]
            * self.background_relative_permittivity
            * self.area
            / self.dl
        )
        self.background_conductance = self.background_conductivity * self.area / self.dl

        full_frequency = np.fft.rfftfreq(grid.iterations, d=grid.dt)
        cells, limiting_material = minimum_wavelength_sampling(grid, full_frequency)
        mesh_valid_full = cells >= self.minimum_wavelength_cells
        positive_invalid = np.flatnonzero((full_frequency > 0) & ~mesh_valid_full)
        if positive_invalid.size:
            first_invalid = int(positive_invalid[0])
            last_mesh_index = max(0, first_invalid - 1)
            limiting_index = first_invalid
        else:
            last_mesh_index = full_frequency.size - 1
            limiting_index = last_mesh_index
        self._frequency_slice = (
            slice(None) if self.spectrum_limit_mode == "nyquist" else slice(0, last_mesh_index + 1)
        )
        self._full_cells_per_wavelength = cells
        self._full_mesh_valid = mesh_valid_full
        self.nyquist_frequency = float(1 / (2 * grid.dt))
        self.mesh_frequency_limit = float(full_frequency[last_mesh_index])
        self.limiting_material = str(limiting_material[limiting_index])
        self.independent_frequency_resolution = float(1 / (grid.iterations * grid.dt))
        self.dt = float(grid.dt)
        if hasattr(grid, "local_to_global"):
            self.source_position = np.asarray(
                grid.local_to_global(self.terminal.coord), dtype=np.float64
            )
        else:
            self.source_position = np.asarray(
                self.terminal.coord * np.asarray((grid.dx, grid.dy, grid.dz)),
                dtype=np.float64,
            )
        self.prepared = True

    def reset_run_state(self) -> None:
        """Clear derived output before a reused-geometry source-study case."""

        self.result = None

    def finalise(self, grid: "FDTDGrid") -> RationalNetworkPortResult:
        if not self.prepared:
            self.prepare(grid)
        real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
        complex_dtype = np.dtype(config.sim_config.dtypes["complex"])
        total_voltage = np.asarray(
            0.5 * (self.terminal.voltage[:-1] + self.terminal.voltage[1:]),
            dtype=real_dtype,
        )
        generator_voltage = np.asarray(self.terminal.generator_voltage, dtype=real_dtype)
        network_current = np.asarray(self.terminal.network_current, dtype=real_dtype)
        time = np.asarray(
            (np.arange(grid.iterations, dtype=real_dtype) + real_dtype.type(0.5)) * grid.dt,
            dtype=real_dtype,
        )

        frequency_full, total_voltage_full = engineering_rfft(
            total_voltage, grid.dt, time_offset=0.5 * grid.dt
        )
        _, generator_full = engineering_rfft(generator_voltage, grid.dt, time_offset=0.5 * grid.dt)
        _, network_current_full = engineering_rfft(
            network_current, grid.dt, time_offset=0.5 * grid.dt
        )
        selection = self._frequency_slice
        frequency = frequency_full[selection]
        total_spectrum = total_voltage_full[selection]
        generator_spectrum = generator_full[selection]
        network_current_spectrum = network_current_full[selection]

        omega_discrete = (2 / grid.dt) * np.tan(np.pi * frequency * grid.dt)
        gap_admittance = np.asarray(
            self.background_conductance + 1j * omega_discrete * self.gap_capacitance,
            dtype=complex_dtype,
        )
        # Inetwork is positive from the FDTD gap into the external network.
        # Iterminal enters the electromagnetic structure and has the numerical
        # Yee-gap admittance removed.
        terminal_current = np.asarray(
            -network_current_spectrum - gap_admittance * total_spectrum,
            dtype=complex_dtype,
        )
        incident_voltage = np.asarray(
            0.5 * (total_spectrum + self.reference_impedance * terminal_current),
            dtype=complex_dtype,
        )
        reflected_voltage = np.asarray(
            0.5 * (total_spectrum - self.reference_impedance * terminal_current),
            dtype=complex_dtype,
        )
        s11, s11_defined = _safe_complex_divide(reflected_voltage, incident_voltage, complex_dtype)
        zin, zin_defined = _safe_complex_divide(total_spectrum, terminal_current, complex_dtype)
        yin, yin_defined = _safe_complex_divide(terminal_current, total_spectrum, complex_dtype)

        incident_magnitude = np.abs(incident_voltage)
        incident_peak = float(np.max(incident_magnitude, initial=0.0))
        incident_relative_db = np.full(frequency.shape, -np.inf, dtype=real_dtype)
        if incident_peak > 0:
            nonzero = incident_magnitude > 0
            incident_relative_db[nonzero] = np.asarray(
                20 * np.log10(incident_magnitude[nonzero] / incident_peak),
                dtype=real_dtype,
            )
        source_valid = (
            np.asarray(incident_relative_db >= self.incident_floor_db, dtype=bool)
            if self.terminal.excited
            else np.zeros(frequency.shape, dtype=bool)
        )
        mesh_valid = np.asarray(self._full_mesh_valid[selection], dtype=bool)
        cells_per_wavelength = np.asarray(
            self._full_cells_per_wavelength[selection], dtype=real_dtype
        )
        nyquist = np.isclose(
            frequency,
            1 / (2 * grid.dt),
            rtol=64 * np.finfo(real_dtype).eps,
            atol=0,
        )
        gap_correction_valid = ~(nyquist & (self.gap_capacitance != 0))
        valid_s11 = mesh_valid & gap_correction_valid & source_valid & s11_defined
        valid_zin = mesh_valid & gap_correction_valid & zin_defined
        valid_yin = mesh_valid & gap_correction_valid & yin_defined

        trace_peak = float(np.max(np.abs(total_voltage), initial=0.0))
        tail_count = max(1, total_voltage.size // 20)
        tail_peak = float(np.max(np.abs(total_voltage[-tail_count:]), initial=0.0))
        if trace_peak > 0 and tail_peak > 0:
            tail_relative_db = float(20 * np.log10(tail_peak / trace_peak))
        elif tail_peak == 0:
            tail_relative_db = float("-inf")
        else:
            tail_relative_db = float("nan")
        if np.isfinite(tail_relative_db) and tail_relative_db > -40:
            logger.warning(
                f"NetworkPort {self.output_id!r}: the final 5% of the voltage trace "
                f"reaches {tail_relative_db:.1f} dB relative to its peak; spectral "
                "leakage may be significant."
            )

        self.result = RationalNetworkPortResult(
            time=time,
            generator_voltage=generator_voltage,
            total_voltage=total_voltage,
            network_current=network_current,
            frequency=np.asarray(frequency, dtype=real_dtype),
            generator_spectrum=np.asarray(generator_spectrum, dtype=complex_dtype),
            total_voltage_spectrum=np.asarray(total_spectrum, dtype=complex_dtype),
            network_current_spectrum=np.asarray(network_current_spectrum, dtype=complex_dtype),
            terminal_current_spectrum=terminal_current,
            incident_voltage_spectrum=incident_voltage,
            reflected_voltage_spectrum=reflected_voltage,
            s11=s11,
            zin=zin,
            yin=yin,
            source_valid=source_valid,
            mesh_valid=mesh_valid,
            gap_correction_valid=np.asarray(gap_correction_valid, dtype=bool),
            valid_s11=np.asarray(valid_s11, dtype=bool),
            valid_zin=np.asarray(valid_zin, dtype=bool),
            valid_yin=np.asarray(valid_yin, dtype=bool),
            incident_relative_db=incident_relative_db,
            cells_per_minimum_wavelength=cells_per_wavelength,
            tail_relative_db=tail_relative_db,
        )
        return self.result

    def write_hdf5(self, base_group) -> None:
        if self.result is None:
            raise RuntimeError(f"network port {self.output_id!r} has not been finalised")
        result = self.result
        real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
        complex_dtype = np.dtype(config.sim_config.dtypes["complex"])
        ports_group = base_group.require_group("ports")
        group = ports_group.create_group(self.output_id)
        group.attrs["Name"] = self.output_id
        group.attrs["Position"] = np.asarray(self.source_position, dtype=real_dtype)
        group.attrs["GridPosition"] = np.asarray(self.terminal.coord, dtype=np.int32)
        group.attrs["SourceType"] = "RationalNetworkTerminal"
        group.attrs["PortMode"] = "rational_network"
        group.attrs["NetworkModelID"] = self.terminal.model.ID
        group.attrs["Polarisation"] = self.terminal.polarisation
        group.attrs["CellLength"] = self.dl
        group.attrs["ReferenceImpedance"] = self.reference_impedance
        group.attrs["WaveformID"] = self.terminal.waveformID or ""
        group.attrs["Conductance"] = self.terminal.model.conductance
        group.attrs["Capacitance"] = self.terminal.model.capacitance
        group.attrs["BackgroundMaterial"] = self.terminal.background_material_ID
        group.attrs["BackgroundRelativePermittivity"] = self.background_relative_permittivity
        group.attrs["BackgroundConductivity"] = self.background_conductivity
        group.attrs["GapCapacitance"] = self.gap_capacitance
        group.attrs["BackgroundConductance"] = self.background_conductance
        group.attrs["GapCorrection"] = "discrete_parallel_admittance"
        group.attrs["TimeSampleOffset"] = 0.5 * self.dt
        group.attrs["Window"] = "rectangular"
        group.attrs["IncidentFloorDB"] = self.incident_floor_db
        group.attrs["SpectrumLimitMode"] = self.spectrum_limit_mode
        group.attrs["MinimumWavelengthCells"] = self.minimum_wavelength_cells
        group.attrs["MeshFrequencyLimit"] = self.mesh_frequency_limit
        group.attrs["NyquistFrequency"] = self.nyquist_frequency
        group.attrs["LimitingMaterial"] = self.limiting_material
        group.attrs["IndependentFrequencyResolution"] = self.independent_frequency_resolution
        group.attrs["FrequencyRange"] = np.asarray(
            (result.frequency[0], result.frequency[-1]), dtype=real_dtype
        )
        valid_indices = np.flatnonzero(result.valid_s11)
        valid_range = (
            (result.frequency[valid_indices[0]], result.frequency[valid_indices[-1]])
            if valid_indices.size
            else (np.nan, np.nan)
        )
        group.attrs["ValidFrequencyRange"] = np.asarray(valid_range, dtype=real_dtype)
        group.attrs["TailRelativeLevelDB"] = result.tail_relative_db
        group.attrs["phasor_time_sign"] = PHASOR_TIME_DEPENDENCE
        group.attrs["forward_transform_sign"] = FORWARD_TRANSFORM_KERNEL
        group.attrs["real_dtype"] = real_dtype.name
        group.attrs["complex_dtype"] = complex_dtype.name
        group.create_dataset(
            "poles", data=np.asarray(self.terminal.model.poles, dtype=complex_dtype)
        )
        group.create_dataset(
            "residues", data=np.asarray(self.terminal.model.residues, dtype=complex_dtype)
        )
        datasets = {
            "time": result.time,
            "Vgenerator": result.generator_voltage,
            "Vtotal": result.total_voltage,
            "Inetwork": result.network_current,
            "frequency": result.frequency,
            "Vgenerator_spectrum": result.generator_spectrum,
            "Vtotal_spectrum": result.total_voltage_spectrum,
            "Inetwork_spectrum": result.network_current_spectrum,
            "Iterminal_spectrum": result.terminal_current_spectrum,
            "Vincident_spectrum": result.incident_voltage_spectrum,
            "Vreflected_spectrum": result.reflected_voltage_spectrum,
            "S11": result.s11,
            "Zin": result.zin,
            "Yin": result.yin,
            "valid_S11": result.valid_s11.astype(np.uint8),
            "valid_Zin": result.valid_zin.astype(np.uint8),
            "valid_Yin": result.valid_yin.astype(np.uint8),
            "source_valid": result.source_valid.astype(np.uint8),
            "mesh_valid": result.mesh_valid.astype(np.uint8),
            "gap_correction_valid": result.gap_correction_valid.astype(np.uint8),
            "incident_relative_dB": result.incident_relative_db,
            "cells_per_minimum_wavelength": result.cells_per_minimum_wavelength,
        }
        for name, values in datasets.items():
            group.create_dataset(name, data=values)


class VoltageSourcePortMonitor:
    """Bind terminal field receivers to one voltage source.

    Finite-resistance sources use their Thevenin voltage-wave separation.
    Zero-resistance hard sources instead use the surrounding Ampere contour,
    with explicit transform offsets for the magnetic half-step current and
    integer-time gap voltage.
    """

    def __init__(
        self,
        output_id: str,
        source: "VoltageSource",
        receiver: "Rx",
        spectrum_limit: SpectrumLimit,
        owner=None,
    ):
        self.output_id = output_id
        self.source = source
        self.receiver = receiver
        self.spectrum_limit = spectrum_limit
        self.owner = owner
        self.result: Optional[VoltageSourcePortResult] = None
        self.prepared = False

        self.minimum_wavelength_cells = (
            DEFAULT_MINIMUM_WAVELENGTH_CELLS
            if spectrum_limit == "nyquist"
            else float(spectrum_limit)
        )
        self.spectrum_limit_mode = (
            "nyquist" if spectrum_limit == "nyquist" else "minimum_wavelength_cells"
        )
        self.incident_floor_db = DEFAULT_INCIDENT_FLOOR_DB

    @property
    def component(self):
        return f"E{self.source.polarisation}"

    def reset_run_state(self) -> None:
        """Clear histories and derived results before a reused-geometry run."""

        for values in self.receiver.outputs.values():
            values.fill(0)
        self.result = None
        for name in (
            "_hard_current_time",
            "_hard_loop_current",
            "_hard_loop_current_spectrum",
            "_hard_terminal_current_spectrum",
            "_hard_source_plane_valid",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def rebind_after_mpi_gather(self, grid: "FDTDGrid") -> None:
        """Reconnect gathered monitor state to the coordinator's grid objects."""

        receivers = [
            receiver
            for receiver in grid.rxs
            if getattr(receiver, "internal", False)
            and getattr(receiver, "port_id", None) == self.output_id
        ]
        sources = [
            source
            for source in grid.voltagesources
            if source.polarisation == self.source.polarisation
            and np.array_equal(source.coord, self.source.coord)
        ]
        if len(receivers) != 1 or len(sources) != 1:
            raise RuntimeError(
                f"Voltage-source port {self.output_id!r} could not uniquely rebind its MPI "
                f"receiver/source ({len(receivers)} receiver(s), {len(sources)} source(s))"
            )
        self.receiver = receivers[0]
        self.source = sources[0]
        self.source_index = grid.voltagesources.index(self.source) + 1
        # ``gather_coord_objects`` has converted the rebound source coordinate
        # to the global MPI-grid index. ``source_position`` may previously have
        # been cached while preparing the monitor on its owning rank, where the
        # same coordinate was rank-local.
        self.source_position = np.asarray(
            self.source.coord * np.asarray((grid.dx, grid.dy, grid.dz)),
            dtype=np.float64,
        )

    def _edge_geometry(self, grid):
        if self.source.polarisation == "x":
            return float(grid.dx), float(grid.dy * grid.dz)
        if self.source.polarisation == "y":
            return float(grid.dy), float(grid.dx * grid.dz)
        return float(grid.dz), float(grid.dx * grid.dy)

    def prepare(self, grid: "FDTDGrid") -> None:
        """Validate the built Yee edge and calculate fixed port parameters."""

        if grid.iterations < 2:
            raise ValueError(
                f"Voltage-source port {self.output_id!r} requires at least two iterations"
            )
        if not np.array_equal(self.receiver.coord, self.source.coord):
            raise ValueError(
                f"Voltage-source port {self.output_id!r} receiver is no longer source-bound"
            )
        self.hard_source = self.source.resistance == 0
        if self.hard_source:
            component_id = f"E{self.source.polarisation}"
            material_num_id = int(
                grid.ID[
                    grid.IDlookup[component_id],
                    self.source.xcoord,
                    self.source.ycoord,
                    self.source.zcoord,
                ]
            )
            material = next(item for item in grid.materials if item.numID == material_num_id)
            self.source.background_material_numID = material_num_id
            self.source.background_material_ID = str(material.ID)
            self.source.background_material_type = str(material.type)
            self.source.background_er = float(material.er)
            self.source.background_se = float(material.se)
            self.source.background_mr = float(material.mr)
            self.source.background_sm = float(material.sm)
            self.source.background_is_dispersive = hasattr(material, "poles")
            self.source.source_material_numID = material_num_id
        elif getattr(self.source, "background_material_numID", None) is None:
            raise ValueError(
                f"Voltage-source port {self.output_id!r} source material was not constructed"
            )
        self.background_is_dispersive = bool(
            getattr(self.source, "background_is_dispersive", False)
        )
        self.background_material = next(
            item
            for item in grid.materials
            if item.numID == int(self.source.background_material_numID)
        )
        if self.hard_source and self.background_is_dispersive:
            raise ValueError(
                f"Voltage-source port {self.output_id!r} does not yet support a hard source "
                "on a dispersive material edge"
            )

        self.dl, self.area = self._edge_geometry(grid)
        self.reference_impedance = float(self.source.reference_impedance)
        if not np.isfinite(self.reference_impedance) or self.reference_impedance <= 0:
            raise ValueError(
                f"Voltage-source port {self.output_id!r} requires a finite, positive "
                "reference impedance"
            )
        self.background_relative_permittivity = float(self.source.background_er)
        self.background_conductivity = float(self.source.background_se)
        if not np.isfinite(self.background_conductivity):
            raise ValueError(
                f"Voltage-source port {self.output_id!r} cannot use a voltage source on a PEC edge"
            )
        self.gap_capacitance = (
            config.sim_config.em_consts["e0"]
            * self.background_relative_permittivity
            * self.area
            / self.dl
        )
        self.background_conductance = self.background_conductivity * self.area / self.dl

        source_material_id = int(self.source.source_material_numID)
        source_material = grid.materials[source_material_id]
        dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
        if not self.hard_source:
            added_conductance = (
                (float(source_material.se) - self.background_conductivity) * self.area / self.dl
            )
            tolerance = 64 * np.finfo(dtype).eps
            if not np.isclose(
                added_conductance,
                1 / self.reference_impedance,
                rtol=tolerance,
                atol=tolerance / self.reference_impedance,
            ):
                raise ValueError(
                    f"Voltage-source port {self.output_id!r} source-edge conductance is inconsistent "
                    "with its source resistance"
                )
            if not np.isfinite(grid.updatecoeffsE[source_material_id, 4]) or np.isclose(
                grid.updatecoeffsE[source_material_id, 4], 0
            ):
                raise ValueError(
                    f"Voltage-source port {self.output_id!r} source is on an inactive electric edge"
                )
        elif f"I{self.source.polarisation}" not in self.receiver.outputs:
            raise ValueError(
                f"Voltage-source port {self.output_id!r} hard source has no terminal-current samples"
            )

        nsamples = grid.iterations - 1
        self.aligned_samples = nsamples
        full_frequency = np.fft.rfftfreq(nsamples, d=grid.dt)
        cells, limiting_material = minimum_wavelength_sampling(grid, full_frequency)
        mesh_valid_full = cells >= self.minimum_wavelength_cells
        positive_invalid = np.flatnonzero((full_frequency > 0) & ~mesh_valid_full)
        if positive_invalid.size:
            first_invalid = int(positive_invalid[0])
            last_mesh_index = max(0, first_invalid - 1)
            limiting_index = first_invalid
        else:
            last_mesh_index = full_frequency.size - 1
            limiting_index = last_mesh_index

        self.nyquist_frequency = float(1 / (2 * grid.dt))
        self.mesh_frequency_limit = float(full_frequency[last_mesh_index])
        self.limiting_material = str(limiting_material[limiting_index])
        self.independent_frequency_resolution = float(1 / (nsamples * grid.dt))
        self._full_cells_per_wavelength = cells
        self._full_mesh_valid = mesh_valid_full
        if self.spectrum_limit_mode == "nyquist":
            self._frequency_slice = slice(None)
            log = logger.warning
            message = (
                f"Voltage-source port {self.output_id!r}: full native spectrum 0--"
                f"{full_frequency[-1]:g} Hz requested (Nyquist research override); "
                f"advisory lambda/{self.minimum_wavelength_cells:g} limit "
                f"{self.mesh_frequency_limit:g} Hz in material "
                f"{self.limiting_material!r}."
            )
        else:
            self._frequency_slice = slice(0, last_mesh_index + 1)
            log = logger.info
            message = (
                f"Voltage-source port {self.output_id!r}: spectrum 0--"
                f"{self.mesh_frequency_limit:g} Hz, lambda/"
                f"{self.minimum_wavelength_cells:g} mesh limit in material "
                f"{self.limiting_material!r}; native df "
                f"{self.independent_frequency_resolution:g} Hz; Nyquist "
                f"{self.nyquist_frequency:g} Hz."
            )
        log(message)
        self.set_output_context(grid)
        self.prepared = True

    def finalise(self, grid: "FDTDGrid") -> VoltageSourcePortResult:
        """Calculate voltage waves, corrected S11, Zin, and Yin."""

        if not self.prepared:
            self.prepare(grid)
        if self.hard_source:
            return self._finalise_hard_source(grid)
        if not np.array_equal(self.receiver.coord, self.source.coord):
            raise RuntimeError(
                f"Voltage-source port {self.output_id!r} receiver moved away from its source"
            )

        real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
        complex_dtype = np.dtype(config.sim_config.dtypes["complex"])
        electric = np.asarray(self.receiver.outputs[self.component], dtype=real_dtype)
        if electric.size != grid.iterations:
            raise RuntimeError(
                f"Voltage-source port {self.output_id!r} receiver history has the wrong length"
            )

        integer_voltage = np.asarray(-self.dl * electric, dtype=real_dtype)
        total_voltage = np.asarray(
            0.5 * (integer_voltage[:-1] + integer_voltage[1:]), dtype=real_dtype
        )
        generator_voltage = np.asarray(
            self.source.waveformvalues_halfdt[: total_voltage.size], dtype=real_dtype
        )
        time = np.asarray(
            (np.arange(total_voltage.size, dtype=real_dtype) + real_dtype.type(0.5)) * grid.dt,
            dtype=real_dtype,
        )

        frequency_full, total_spectrum_full = engineering_rfft(
            total_voltage, grid.dt, time_offset=0.5 * grid.dt
        )
        generator_frequency, generator_spectrum_full = engineering_rfft(
            generator_voltage, grid.dt, time_offset=0.5 * grid.dt
        )
        if not np.array_equal(frequency_full, generator_frequency):
            raise RuntimeError("port voltage transforms produced inconsistent frequencies")

        selection = self._frequency_slice
        frequency = frequency_full[selection]
        total_spectrum = total_spectrum_full[selection]
        generator_spectrum = generator_spectrum_full[selection]
        incident_spectrum = np.asarray(0.5 * generator_spectrum, dtype=complex_dtype)
        reflected_source = np.asarray(total_spectrum - incident_spectrum, dtype=complex_dtype)
        s11_source, source_defined = _safe_complex_divide(
            reflected_source, incident_spectrum, complex_dtype
        )

        incident_magnitude = np.abs(incident_spectrum)
        incident_peak = float(np.max(incident_magnitude, initial=0.0))
        incident_relative_db = np.full(frequency.shape, -np.inf, dtype=real_dtype)
        if incident_peak > 0:
            nonzero = incident_magnitude > 0
            incident_relative_db[nonzero] = np.asarray(
                20 * np.log10(incident_magnitude[nonzero] / incident_peak), dtype=real_dtype
            )
        source_valid = source_defined & (incident_relative_db >= self.incident_floor_db)

        gap_admittance = _finite_source_gap_admittance(
            self,
            frequency,
            grid.dt,
            complex_dtype,
        )
        gap_correction = np.asarray(
            self.reference_impedance * gap_admittance,
            dtype=complex_dtype,
        )
        s11, gap_correction_valid = correct_s11_for_parallel_gap(
            s11_source, gap_correction, complex_dtype
        )
        exact_nyquist_included = (
            self.aligned_samples % 2 == 0 and frequency.size == frequency_full.size
        )
        if self.gap_capacitance != 0 and exact_nyquist_included:
            gap_correction_valid[-1] = False
            s11[-1] = np.nan + 1j * np.nan

        zin_source, _ = impedance_from_s11(s11_source, self.reference_impedance, complex_dtype)
        zin, zin_defined = impedance_from_s11(s11, self.reference_impedance, complex_dtype)
        yin, yin_defined = admittance_from_s11(s11, self.reference_impedance, complex_dtype)

        mesh_valid = np.asarray(self._full_mesh_valid[selection], dtype=bool)
        cells_per_wavelength = np.asarray(
            self._full_cells_per_wavelength[selection], dtype=real_dtype
        )
        valid_s11 = mesh_valid & source_valid & gap_correction_valid
        valid_zin = valid_s11 & zin_defined
        valid_yin = valid_s11 & yin_defined

        trace_peak = float(np.max(np.abs(total_voltage), initial=0.0))
        tail_count = max(1, total_voltage.size // 20)
        tail_peak = float(np.max(np.abs(total_voltage[-tail_count:]), initial=0.0))
        if trace_peak > 0 and tail_peak > 0:
            tail_relative_db = float(20 * np.log10(tail_peak / trace_peak))
        elif tail_peak == 0:
            tail_relative_db = float("-inf")
        else:
            tail_relative_db = float("nan")
        if np.isfinite(tail_relative_db) and tail_relative_db > -40:
            logger.warning(
                f"Voltage-source port {self.output_id!r}: the final 5% of the voltage trace "
                f"reaches {tail_relative_db:.1f} dB relative to its peak; spectral "
                "leakage may be significant."
            )
        if np.isfinite(tail_relative_db) and tail_relative_db > -40:
            logger.warning(
                f"Voltage-source port {self.output_id!r}: the final 5% of the voltage trace "
                f"reaches {tail_relative_db:.1f} dB relative to its peak; spectral "
                "leakage may be significant."
            )

        self.result = VoltageSourcePortResult(
            time=time,
            generator_voltage=generator_voltage,
            total_voltage=total_voltage,
            frequency=np.asarray(frequency, dtype=real_dtype),
            incident_spectrum=incident_spectrum,
            reflected_source_spectrum=reflected_source,
            total_spectrum=np.asarray(total_spectrum, dtype=complex_dtype),
            gap_correction=gap_correction,
            s11_source=s11_source,
            zin_source=zin_source,
            s11=s11,
            zin=zin,
            yin=yin,
            source_valid=np.asarray(source_valid, dtype=bool),
            mesh_valid=mesh_valid,
            gap_correction_valid=np.asarray(gap_correction_valid, dtype=bool),
            valid_s11=np.asarray(valid_s11, dtype=bool),
            valid_zin=np.asarray(valid_zin, dtype=bool),
            valid_yin=np.asarray(valid_yin, dtype=bool),
            incident_relative_db=incident_relative_db,
            cells_per_minimum_wavelength=cells_per_wavelength,
            tail_relative_db=tail_relative_db,
        )
        return self.result

    def _hard_loop_current_history(self, real_dtype):
        """Return the Ampere-loop current at stored magnetic half steps."""

        current_component = f"I{self.source.polarisation}"
        return np.asarray(self.receiver.outputs[current_component], dtype=real_dtype)

    def _finalise_hard_source(self, grid: "FDTDGrid") -> VoltageSourcePortResult:
        """Derive a delta-gap port from prescribed voltage and loop current."""

        if not np.array_equal(self.receiver.coord, self.source.coord):
            raise RuntimeError(
                f"Voltage-source port {self.output_id!r} receiver moved away from its source"
            )

        real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
        complex_dtype = np.dtype(config.sim_config.dtypes["complex"])
        electric = np.asarray(self.receiver.outputs[self.component], dtype=real_dtype)
        loop_half = self._hard_loop_current_history(real_dtype)
        if electric.size != grid.iterations or loop_half.size != grid.iterations:
            raise RuntimeError(
                f"Voltage-source port {self.output_id!r} receiver history has the wrong length"
            )

        # At receiver-storage index m, E is at m*dt and H is at
        # (m-1/2)*dt. Drop the unexcited initial fields so each retained pair
        # is exactly V^(n+1), I_loop^(n+1/2). Their separate transform offsets
        # remove the half-step phase difference without the cosine attenuation
        # introduced by time-domain averaging.
        total_voltage = np.asarray(-self.dl * electric[1:], dtype=real_dtype)
        loop_current = np.asarray(loop_half[1:], dtype=real_dtype)
        generator_voltage = total_voltage.copy()
        self.hard_voltage_time_offset = float(grid.dt)
        self.hard_current_time_offset = float(0.5 * grid.dt)
        time = np.asarray(
            (np.arange(total_voltage.size, dtype=real_dtype) + 1) * grid.dt,
            dtype=real_dtype,
        )
        self._hard_current_time = np.asarray(
            (np.arange(loop_current.size, dtype=real_dtype) + 0.5) * grid.dt,
            dtype=real_dtype,
        )

        frequency_full, total_spectrum_full = engineering_rfft(
            total_voltage,
            grid.dt,
            time_offset=self.hard_voltage_time_offset,
        )
        _, loop_spectrum_full = engineering_rfft(
            loop_current,
            grid.dt,
            time_offset=self.hard_current_time_offset,
        )
        selection = self._frequency_slice
        frequency = frequency_full[selection]
        total_spectrum = np.asarray(total_spectrum_full[selection], dtype=complex_dtype)
        loop_spectrum = np.asarray(loop_spectrum_full[selection], dtype=complex_dtype)

        gap_admittance = _hard_source_gap_admittance(
            self,
            frequency,
            grid.dt,
            complex_dtype,
        )
        terminal_current = np.asarray(
            loop_spectrum - gap_admittance * total_spectrum, dtype=complex_dtype
        )

        incident_source = np.asarray(
            0.5 * (total_spectrum + self.reference_impedance * loop_spectrum),
            dtype=complex_dtype,
        )
        reflected_source = np.asarray(
            0.5 * (total_spectrum - self.reference_impedance * loop_spectrum),
            dtype=complex_dtype,
        )
        incident_spectrum = np.asarray(
            0.5 * (total_spectrum + self.reference_impedance * terminal_current),
            dtype=complex_dtype,
        )
        reflected_terminal = np.asarray(
            0.5 * (total_spectrum - self.reference_impedance * terminal_current),
            dtype=complex_dtype,
        )
        s11_source, source_plane_defined = _safe_complex_divide(
            reflected_source, incident_source, complex_dtype
        )
        s11, terminal_wave_defined = _safe_complex_divide(
            reflected_terminal, incident_spectrum, complex_dtype
        )
        zin_source, zin_source_defined = _safe_complex_divide(
            total_spectrum, loop_spectrum, complex_dtype
        )
        zin, zin_defined = _safe_complex_divide(total_spectrum, terminal_current, complex_dtype)
        yin, yin_defined = _safe_complex_divide(terminal_current, total_spectrum, complex_dtype)

        incident_magnitude = np.abs(incident_spectrum)
        incident_peak = float(np.max(incident_magnitude, initial=0.0))
        incident_relative_db = np.full(frequency.shape, -np.inf, dtype=real_dtype)
        if incident_peak > 0:
            nonzero = incident_magnitude > 0
            incident_relative_db[nonzero] = np.asarray(
                20 * np.log10(incident_magnitude[nonzero] / incident_peak),
                dtype=real_dtype,
            )
        source_valid = terminal_wave_defined & (incident_relative_db >= self.incident_floor_db)

        gap_correction = np.asarray(self.reference_impedance * gap_admittance, dtype=complex_dtype)
        gap_correction_valid = np.ones(frequency.shape, dtype=bool)

        mesh_valid = np.asarray(self._full_mesh_valid[selection], dtype=bool)
        cells_per_wavelength = np.asarray(
            self._full_cells_per_wavelength[selection], dtype=real_dtype
        )
        source_plane_valid = source_plane_defined & zin_source_defined
        valid_s11 = mesh_valid & source_valid & gap_correction_valid
        valid_zin = valid_s11 & zin_defined
        valid_yin = valid_s11 & yin_defined

        trace_peak = float(np.max(np.abs(total_voltage), initial=0.0))
        tail_count = max(1, total_voltage.size // 20)
        tail_peak = float(np.max(np.abs(total_voltage[-tail_count:]), initial=0.0))
        if trace_peak > 0 and tail_peak > 0:
            tail_relative_db = float(20 * np.log10(tail_peak / trace_peak))
        elif tail_peak == 0:
            tail_relative_db = float("-inf")
        else:
            tail_relative_db = float("nan")

        self._hard_loop_current = loop_current
        self._hard_loop_current_spectrum = loop_spectrum
        self._hard_terminal_current_spectrum = terminal_current
        self._hard_source_plane_valid = source_plane_valid
        self.result = VoltageSourcePortResult(
            time=time,
            generator_voltage=generator_voltage,
            total_voltage=total_voltage,
            frequency=np.asarray(frequency, dtype=real_dtype),
            incident_spectrum=incident_spectrum,
            reflected_source_spectrum=reflected_source,
            total_spectrum=total_spectrum,
            gap_correction=gap_correction,
            s11_source=s11_source,
            zin_source=zin_source,
            s11=s11,
            zin=zin,
            yin=yin,
            source_valid=np.asarray(source_valid, dtype=bool),
            mesh_valid=mesh_valid,
            gap_correction_valid=gap_correction_valid,
            valid_s11=np.asarray(valid_s11, dtype=bool),
            valid_zin=np.asarray(valid_zin, dtype=bool),
            valid_yin=np.asarray(valid_yin, dtype=bool),
            incident_relative_db=incident_relative_db,
            cells_per_minimum_wavelength=cells_per_wavelength,
            tail_relative_db=tail_relative_db,
        )
        return self.result

    def write_hdf5(self, base_group) -> None:
        """Write the finalised port result beneath ``/ports/<ID>``."""

        if self.result is None:
            raise RuntimeError(f"Voltage-source port {self.output_id!r} has not been finalised")
        result = self.result
        ports_group = base_group.require_group("ports")
        group = ports_group.create_group(self.output_id)
        real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
        complex_dtype = np.dtype(config.sim_config.dtypes["complex"])

        group.attrs["Name"] = self.output_id
        group.attrs["Position"] = np.asarray(self.source_position, dtype=real_dtype)
        group.attrs["GridPosition"] = np.asarray(self.source.coord, dtype=np.int32)
        group.attrs["SourceType"] = type(self.source).__name__
        group.attrs["PortMode"] = "hard_delta_gap" if self.hard_source else "resistive_thevenin"
        group.attrs["SourceIndex"] = self.source_index
        group.attrs["Polarisation"] = self.source.polarisation
        group.attrs["CellLength"] = self.dl
        group.attrs["ReferenceImpedance"] = self.reference_impedance
        group.attrs["ReferenceImpedanceSource"] = "voltage_source"
        group.attrs["WaveformID"] = self.source.waveformID
        group.attrs["BackgroundMaterial"] = self.source.background_material_ID
        group.attrs["BackgroundRelativePermittivity"] = self.background_relative_permittivity
        group.attrs["BackgroundConductivity"] = self.background_conductivity
        group.attrs["GapCapacitance"] = self.gap_capacitance
        group.attrs["BackgroundConductance"] = self.background_conductance
        group.attrs["GapCorrection"] = "discrete_parallel_admittance"
        group.attrs["TimeSampleOffset"] = (
            self.hard_voltage_time_offset if self.hard_source else 0.5 * self.dt
        )
        if self.hard_source:
            group.attrs["CurrentTimeSampleOffset"] = self.hard_current_time_offset
            group.attrs["CurrentTimeAlignment"] = "explicit_fft_half_step_phase"
        group.attrs["Window"] = "rectangular"
        group.attrs["IncidentFloorDB"] = self.incident_floor_db
        group.attrs["SpectrumLimitMode"] = self.spectrum_limit_mode
        group.attrs["NyquistFrequency"] = self.nyquist_frequency
        group.attrs["MinimumWavelengthCells"] = self.minimum_wavelength_cells
        group.attrs["MeshFrequencyLimit"] = self.mesh_frequency_limit
        group.attrs["LimitingMaterial"] = self.limiting_material
        group.attrs["IndependentFrequencyResolution"] = self.independent_frequency_resolution
        group.attrs["FrequencyRange"] = np.asarray(
            (result.frequency[0], result.frequency[-1]), dtype=real_dtype
        )
        valid_indices = np.flatnonzero(result.valid_s11)
        valid_range = (
            (result.frequency[valid_indices[0]], result.frequency[valid_indices[-1]])
            if valid_indices.size
            else (np.nan, np.nan)
        )
        group.attrs["ValidFrequencyRange"] = np.asarray(valid_range, dtype=real_dtype)
        group.attrs["TailRelativeLevelDB"] = result.tail_relative_db
        group.attrs["phasor_time_sign"] = PHASOR_TIME_DEPENDENCE
        group.attrs["forward_transform_sign"] = FORWARD_TRANSFORM_KERNEL
        group.attrs["real_dtype"] = real_dtype.name
        group.attrs["complex_dtype"] = complex_dtype.name

        datasets = {
            "time": result.time,
            "Vgenerator": result.generator_voltage,
            "Vtotal": result.total_voltage,
            "frequency": result.frequency,
            "Vincident_spectrum": result.incident_spectrum,
            "Vreflected_source_spectrum": result.reflected_source_spectrum,
            "Vtotal_spectrum": result.total_spectrum,
            "gap_correction_c": result.gap_correction,
            "S11_source": result.s11_source,
            "Zin_source": result.zin_source,
            "S11": result.s11,
            "Zin": result.zin,
            "Yin": result.yin,
            "valid_S11": result.valid_s11.astype(np.uint8),
            "valid_Zin": result.valid_zin.astype(np.uint8),
            "valid_Yin": result.valid_yin.astype(np.uint8),
            "source_valid": result.source_valid.astype(np.uint8),
            "mesh_valid": result.mesh_valid.astype(np.uint8),
            "gap_correction_valid": result.gap_correction_valid.astype(np.uint8),
            "incident_relative_dB": result.incident_relative_db,
            "cells_per_minimum_wavelength": result.cells_per_minimum_wavelength,
        }
        if self.hard_source:
            datasets.update(
                {
                    "time_current": self._hard_current_time,
                    "Iloop": self._hard_loop_current,
                    "Iloop_spectrum": self._hard_loop_current_spectrum,
                    "Iterminal_spectrum": self._hard_terminal_current_spectrum,
                    "Vreflected_spectrum": (result.total_spectrum - result.incident_spectrum),
                }
            )
        for name, values in datasets.items():
            group.create_dataset(name, data=values)

    def set_output_context(self, grid: "FDTDGrid") -> None:
        """Cache immutable HDF5 context after the grid is fully built."""

        self.grid_dl = np.asarray((grid.dx, grid.dy, grid.dz), dtype=np.float64)
        if hasattr(grid, "local_to_global"):
            self.source_position = np.asarray(
                grid.local_to_global(self.source.coord), dtype=np.float64
            )
        else:
            self.source_position = np.asarray(self.source.coord * self.grid_dl, dtype=np.float64)
        self.dt = float(grid.dt)
        self.source_index = grid.voltagesources.index(self.source) + 1


@dataclass(frozen=True)
class TransmissionLinePortResult:
    """Final time- and frequency-domain result for one transmission line."""

    time_voltage: npt.NDArray[np.floating]
    time_current: npt.NDArray[np.floating]
    frequency: npt.NDArray[np.floating]
    incident_voltage_spectrum: npt.NDArray[np.complexfloating]
    reflected_voltage_spectrum: npt.NDArray[np.complexfloating]
    total_voltage_spectrum: npt.NDArray[np.complexfloating]
    incident_current_spectrum: npt.NDArray[np.complexfloating]
    total_current_spectrum: npt.NDArray[np.complexfloating]
    reflected_voltage_current_spectrum: npt.NDArray[np.complexfloating]
    terminal_current_spectrum: npt.NDArray[np.complexfloating]
    s11: npt.NDArray[np.complexfloating]
    zin: npt.NDArray[np.complexfloating]
    yin: npt.NDArray[np.complexfloating]
    s11_current: npt.NDArray[np.complexfloating]
    zin_current: npt.NDArray[np.complexfloating]
    source_valid: npt.NDArray[np.bool_]
    mesh_valid: npt.NDArray[np.bool_]
    line_propagation_valid: npt.NDArray[np.bool_]
    valid_s11: npt.NDArray[np.bool_]
    valid_zin: npt.NDArray[np.bool_]
    valid_yin: npt.NDArray[np.bool_]
    valid_s11_current: npt.NDArray[np.bool_]
    valid_zin_current: npt.NDArray[np.bool_]
    incident_relative_db: npt.NDArray[np.floating]
    cells_per_minimum_wavelength: npt.NDArray[np.floating]
    tail_relative_db: float


class TransmissionLinePortOutput:
    """Calculate automatic terminal quantities from one transmission line.

    The voltage-wave result is primary. The line current is staggered by half
    a time step and half a line cell relative to the terminal voltage node.
    ``S11_current`` and ``Zin_current`` therefore use the discrete 1D-line
    dispersion relation to separate and de-embed the current waves before
    they are compared at the voltage reference plane.
    """

    def __init__(
        self,
        source: "TransmissionLine",
        source_index: int,
        spectrum_limit: SpectrumLimit = DEFAULT_MINIMUM_WAVELENGTH_CELLS,
    ):
        self.source = source
        self.source_index = int(source_index)
        self.spectrum_limit = validate_spectrum_limit(spectrum_limit)
        self.result: Optional[TransmissionLinePortResult] = None
        self.prepared = False
        self.minimum_wavelength_cells = (
            DEFAULT_MINIMUM_WAVELENGTH_CELLS
            if self.spectrum_limit == "nyquist"
            else float(self.spectrum_limit)
        )
        self.spectrum_limit_mode = (
            "nyquist" if self.spectrum_limit == "nyquist" else "minimum_wavelength_cells"
        )
        self.incident_floor_db = DEFAULT_INCIDENT_FLOOR_DB

    @property
    def output_id(self) -> str:
        return f"tl{self.source_index}"

    def prepare(self, grid: "FDTDGrid") -> None:
        """Calculate the native frequency axis and mesh-valid output band."""

        if grid.iterations < 2:
            raise ValueError(
                f"Transmission line {self.output_id!r} requires at least two iterations"
            )
        if not np.isfinite(self.source.resistance) or self.source.resistance <= 0:
            raise ValueError(f"Transmission line {self.output_id!r} has an invalid resistance")
        if not np.isfinite(self.source.dl) or self.source.dl <= 0:
            raise ValueError(f"Transmission line {self.output_id!r} has an invalid spatial step")

        nsamples = int(grid.iterations)
        full_frequency = np.fft.rfftfreq(nsamples, d=grid.dt)
        cells, limiting_material = minimum_wavelength_sampling(grid, full_frequency)
        mesh_valid_full = cells >= self.minimum_wavelength_cells
        positive_invalid = np.flatnonzero((full_frequency > 0) & ~mesh_valid_full)
        if positive_invalid.size:
            first_invalid = int(positive_invalid[0])
            last_mesh_index = max(0, first_invalid - 1)
            limiting_index = first_invalid
        else:
            last_mesh_index = full_frequency.size - 1
            limiting_index = last_mesh_index

        self.nsamples = nsamples
        self.reference_impedance = float(self.source.resistance)
        self.nyquist_frequency = float(1 / (2 * grid.dt))
        self.mesh_frequency_limit = float(full_frequency[last_mesh_index])
        self.limiting_material = str(limiting_material[limiting_index])
        self.independent_frequency_resolution = float(1 / (nsamples * grid.dt))
        self._full_cells_per_wavelength = cells
        self._full_mesh_valid = mesh_valid_full
        if self.spectrum_limit_mode == "nyquist":
            self._frequency_slice = slice(None)
            logger.warning(
                f"Transmission line {self.output_id!r}: full native spectrum 0--"
                f"{full_frequency[-1]:g} Hz requested (Nyquist research override); "
                f"advisory lambda/{self.minimum_wavelength_cells:g} limit "
                f"{self.mesh_frequency_limit:g} Hz in material "
                f"{self.limiting_material!r}."
            )
        else:
            self._frequency_slice = slice(0, last_mesh_index + 1)
            logger.info(
                f"Transmission line {self.output_id!r}: automatic S11/Zin spectrum "
                f"0--{self.mesh_frequency_limit:g} Hz, lambda/"
                f"{self.minimum_wavelength_cells:g} mesh limit in material "
                f"{self.limiting_material!r}; native df "
                f"{self.independent_frequency_resolution:g} Hz; Nyquist "
                f"{self.nyquist_frequency:g} Hz."
            )
        self.dt = float(grid.dt)
        self.prepared = True

    def _line_half_cell_phase(self, frequency, grid):
        """Return exp(j*k*dl/2) for the propagating discrete line modes."""

        real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
        complex_dtype = np.dtype(config.sim_config.dtypes["complex"])
        argument = (
            float(self.source.dl)
            / (config.sim_config.em_consts["c"] * float(grid.dt))
            * np.sin(np.pi * np.asarray(frequency, dtype=np.float64) * grid.dt)
        )
        tolerance = 32 * np.finfo(real_dtype).eps
        propagating = np.abs(argument) <= 1 + tolerance
        clipped = np.clip(argument, -1, 1)
        wavenumber_step = 2 * np.arcsin(clipped)
        phase = np.asarray(np.exp(0.5j * wavenumber_step), dtype=complex_dtype)
        return phase, np.asarray(propagating, dtype=bool)

    def finalise(self, grid: "FDTDGrid") -> TransmissionLinePortResult:
        """Calculate automatic S11, Zin, Yin, and current-based checks."""

        if not self.prepared:
            self.prepare(grid)

        real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
        complex_dtype = np.dtype(config.sim_config.dtypes["complex"])
        histories = {}
        for name in ("Vinc", "Iinc", "Vtotal", "Itotal"):
            values = np.asarray(getattr(self.source, name), dtype=real_dtype)
            if values.ndim != 1 or values.size < self.nsamples:
                raise RuntimeError(
                    f"Transmission line {self.output_id!r} {name} history has " "the wrong length"
                )
            values = values[: self.nsamples]
            if not np.all(np.isfinite(values)):
                raise RuntimeError(
                    f"Transmission line {self.output_id!r} {name} history is not finite"
                )
            histories[name] = values

        frequency_full, incident_voltage_full = engineering_rfft(histories["Vinc"], grid.dt)
        _, total_voltage_full = engineering_rfft(histories["Vtotal"], grid.dt)
        # Index n stores I at (n - 1/2) dt. Applying the physical sample-time
        # offset here preserves all N samples and the voltage FFT grid.
        _, incident_current_full = engineering_rfft(
            histories["Iinc"], grid.dt, time_offset=-0.5 * grid.dt
        )
        _, total_current_full = engineering_rfft(
            histories["Itotal"], grid.dt, time_offset=-0.5 * grid.dt
        )

        selection = self._frequency_slice
        frequency = frequency_full[selection]
        incident_voltage = incident_voltage_full[selection]
        total_voltage = total_voltage_full[selection]
        incident_current = incident_current_full[selection]
        total_current = total_current_full[selection]
        reflected_voltage = np.asarray(total_voltage - incident_voltage, dtype=complex_dtype)
        s11, source_defined = _safe_complex_divide(
            reflected_voltage, incident_voltage, complex_dtype
        )

        incident_magnitude = np.abs(incident_voltage)
        incident_peak = float(np.max(incident_magnitude, initial=0.0))
        incident_relative_db = np.full(frequency.shape, -np.inf, dtype=real_dtype)
        if incident_peak > 0:
            nonzero = incident_magnitude > 0
            incident_relative_db[nonzero] = np.asarray(
                20 * np.log10(incident_magnitude[nonzero] / incident_peak),
                dtype=real_dtype,
            )
        source_valid = source_defined & (incident_relative_db >= self.incident_floor_db)

        zin, zin_defined = impedance_from_s11(s11, self.reference_impedance, complex_dtype)
        yin, yin_defined = admittance_from_s11(s11, self.reference_impedance, complex_dtype)

        # The stored current lies half a line cell beyond the voltage node.
        # Forward and reflected currents therefore need opposite spatial
        # phase corrections; a single V/I phase multiplier is not sufficient.
        # For the discrete line phase step K,
        #   Z0 Iseg = Vinc exp(-jK/2) - Vref exp(+jK/2),
        # which gives the current-derived reflected voltage used below.
        half_cell_phase, line_propagation_valid = self._line_half_cell_phase(frequency, grid)
        reflected_from_current = np.full(frequency.shape, np.nan + 1j * np.nan, dtype=complex_dtype)
        inverse_half_cell_phase = np.conj(half_cell_phase)
        reflected_from_current[line_propagation_valid] = (
            incident_voltage[line_propagation_valid]
            * inverse_half_cell_phase[line_propagation_valid]
            - self.reference_impedance * total_current[line_propagation_valid]
        ) * inverse_half_cell_phase[line_propagation_valid]
        s11_current, s11_current_defined = _safe_complex_divide(
            reflected_from_current, incident_voltage, complex_dtype
        )
        terminal_current = np.asarray(
            (incident_voltage - reflected_from_current) / self.reference_impedance,
            dtype=complex_dtype,
        )
        terminal_voltage_current = np.asarray(
            incident_voltage + reflected_from_current, dtype=complex_dtype
        )
        zin_current, zin_current_defined = _safe_complex_divide(
            terminal_voltage_current, terminal_current, complex_dtype
        )

        mesh_valid = np.asarray(self._full_mesh_valid[selection], dtype=bool)
        cells_per_wavelength = np.asarray(
            self._full_cells_per_wavelength[selection], dtype=real_dtype
        )
        valid_s11 = mesh_valid & source_valid
        valid_zin = valid_s11 & zin_defined
        valid_yin = valid_s11 & yin_defined
        valid_s11_current = valid_s11 & line_propagation_valid & s11_current_defined
        valid_zin_current = valid_s11_current & zin_current_defined

        trace_peak = float(np.max(np.abs(histories["Vtotal"]), initial=0.0))
        tail_count = max(1, self.nsamples // 20)
        tail_peak = float(np.max(np.abs(histories["Vtotal"][-tail_count:]), initial=0.0))
        if trace_peak > 0 and tail_peak > 0:
            tail_relative_db = float(20 * np.log10(tail_peak / trace_peak))
        elif tail_peak == 0:
            tail_relative_db = float("-inf")
        else:
            tail_relative_db = float("nan")
        if np.isfinite(tail_relative_db) and tail_relative_db > -40:
            logger.warning(
                f"Transmission line {self.output_id!r}: the final 5% of the "
                f"terminal-voltage trace reaches {tail_relative_db:.1f} dB "
                "relative to its peak; spectral leakage may be significant."
            )

        time_voltage = np.asarray(
            np.arange(self.nsamples, dtype=real_dtype) * grid.dt,
            dtype=real_dtype,
        )
        time_current = np.asarray(
            (np.arange(self.nsamples, dtype=real_dtype) - real_dtype.type(0.5)) * grid.dt,
            dtype=real_dtype,
        )
        self.result = TransmissionLinePortResult(
            time_voltage=time_voltage,
            time_current=time_current,
            frequency=np.asarray(frequency, dtype=real_dtype),
            incident_voltage_spectrum=np.asarray(incident_voltage, dtype=complex_dtype),
            reflected_voltage_spectrum=reflected_voltage,
            total_voltage_spectrum=np.asarray(total_voltage, dtype=complex_dtype),
            incident_current_spectrum=np.asarray(incident_current, dtype=complex_dtype),
            total_current_spectrum=np.asarray(total_current, dtype=complex_dtype),
            reflected_voltage_current_spectrum=reflected_from_current,
            terminal_current_spectrum=terminal_current,
            s11=s11,
            zin=zin,
            yin=yin,
            s11_current=s11_current,
            zin_current=zin_current,
            source_valid=np.asarray(source_valid, dtype=bool),
            mesh_valid=mesh_valid,
            line_propagation_valid=line_propagation_valid,
            valid_s11=np.asarray(valid_s11, dtype=bool),
            valid_zin=np.asarray(valid_zin, dtype=bool),
            valid_yin=np.asarray(valid_yin, dtype=bool),
            valid_s11_current=np.asarray(valid_s11_current, dtype=bool),
            valid_zin_current=np.asarray(valid_zin_current, dtype=bool),
            incident_relative_db=incident_relative_db,
            cells_per_minimum_wavelength=cells_per_wavelength,
            tail_relative_db=tail_relative_db,
        )
        return self.result

    def write_hdf5(self, group) -> None:
        """Add derived terminal quantities to an existing ``/tls/tlN`` group."""

        if self.result is None:
            raise RuntimeError(f"Transmission line {self.output_id!r} has not been finalised")
        result = self.result
        real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
        complex_dtype = np.dtype(config.sim_config.dtypes["complex"])

        group.attrs["Polarisation"] = self.source.polarisation
        group.attrs["WaveformID"] = self.source.waveformID
        group.attrs["ReferenceImpedance"] = self.reference_impedance
        group.attrs["ZinPrimaryMethod"] = "voltage_wave_S11"
        group.attrs["CurrentCheckMethod"] = "discrete_line_wave_deembedding"
        group.attrs["TimeVoltageOffset"] = 0.0
        group.attrs["TimeCurrentOffset"] = -0.5 * self.dt
        group.attrs["Window"] = "rectangular"
        group.attrs["IncidentFloorDB"] = self.incident_floor_db
        group.attrs["SpectrumLimitMode"] = self.spectrum_limit_mode
        group.attrs["NyquistFrequency"] = self.nyquist_frequency
        group.attrs["MinimumWavelengthCells"] = self.minimum_wavelength_cells
        group.attrs["MeshFrequencyLimit"] = self.mesh_frequency_limit
        group.attrs["LimitingMaterial"] = self.limiting_material
        group.attrs["IndependentFrequencyResolution"] = self.independent_frequency_resolution
        group.attrs["FrequencyRange"] = np.asarray(
            (result.frequency[0], result.frequency[-1]), dtype=real_dtype
        )
        valid_indices = np.flatnonzero(result.valid_s11)
        valid_range = (
            (result.frequency[valid_indices[0]], result.frequency[valid_indices[-1]])
            if valid_indices.size
            else (np.nan, np.nan)
        )
        group.attrs["ValidFrequencyRange"] = np.asarray(valid_range, dtype=real_dtype)
        group.attrs["TailRelativeLevelDB"] = result.tail_relative_db
        group.attrs["phasor_time_sign"] = PHASOR_TIME_DEPENDENCE
        group.attrs["forward_transform_sign"] = FORWARD_TRANSFORM_KERNEL
        group.attrs["real_dtype"] = real_dtype.name
        group.attrs["complex_dtype"] = complex_dtype.name

        datasets = {
            "time_voltage": result.time_voltage,
            "time_current": result.time_current,
            "frequency": result.frequency,
            "Vincident_spectrum": result.incident_voltage_spectrum,
            "Vreflected_spectrum": result.reflected_voltage_spectrum,
            "Vtotal_spectrum": result.total_voltage_spectrum,
            "Iincident_spectrum": result.incident_current_spectrum,
            "Itotal_spectrum": result.total_current_spectrum,
            "Vreflected_current_spectrum": (result.reflected_voltage_current_spectrum),
            "Iterminal_current_spectrum": result.terminal_current_spectrum,
            "S11": result.s11,
            "Zin": result.zin,
            "Yin": result.yin,
            "S11_current": result.s11_current,
            "Zin_current": result.zin_current,
            "valid_S11": result.valid_s11.astype(np.uint8),
            "valid_Zin": result.valid_zin.astype(np.uint8),
            "valid_Yin": result.valid_yin.astype(np.uint8),
            "valid_S11_current": result.valid_s11_current.astype(np.uint8),
            "valid_Zin_current": result.valid_zin_current.astype(np.uint8),
            "source_valid": result.source_valid.astype(np.uint8),
            "mesh_valid": result.mesh_valid.astype(np.uint8),
            "line_propagation_valid": result.line_propagation_valid.astype(np.uint8),
            "incident_relative_dB": result.incident_relative_db,
            "cells_per_minimum_wavelength": result.cells_per_minimum_wavelength,
        }
        for name, values in datasets.items():
            group.create_dataset(name, data=values)


def prepare_transmission_line_ports(grid: "FDTDGrid") -> None:
    """Create and prepare the automatic port output for every line source."""

    for index, source in enumerate(grid.transmissionlines, start=1):
        output = getattr(source, "port_output", None)
        if output is None:
            output = TransmissionLinePortOutput(source, index)
            source.port_output = output
        else:
            output.source_index = index
        output.prepare(grid)


def finalise_transmission_line_ports(grid: "FDTDGrid") -> None:
    """Finalise all automatic transmission-line port outputs."""

    for index, source in enumerate(grid.transmissionlines, start=1):
        output = getattr(source, "port_output", None)
        if output is None:
            output = TransmissionLinePortOutput(source, index)
            source.port_output = output
        else:
            output.source_index = index
        output.finalise(grid)


@dataclass(frozen=True)
class MagneticFrillPortResult:
    """Final time- and frequency-domain result for one magnetic-frill source."""

    time: npt.NDArray[np.floating]
    frequency: npt.NDArray[np.floating]
    incident_voltage_spectrum: npt.NDArray[np.complexfloating]
    reflected_voltage_spectrum: npt.NDArray[np.complexfloating]
    total_voltage_spectrum: npt.NDArray[np.complexfloating]
    total_current_spectrum: npt.NDArray[np.complexfloating]
    s11: npt.NDArray[np.complexfloating]
    zin: npt.NDArray[np.complexfloating]
    yin: npt.NDArray[np.complexfloating]
    source_valid: npt.NDArray[np.bool_]
    mesh_valid: npt.NDArray[np.bool_]
    valid_s11: npt.NDArray[np.bool_]
    valid_zin: npt.NDArray[np.bool_]
    valid_yin: npt.NDArray[np.bool_]
    incident_relative_db: npt.NDArray[np.floating]
    cells_per_minimum_wavelength: npt.NDArray[np.floating]
    tail_relative_db: float


class MagneticFrillPortOutput:
    """Calculate automatic terminal quantities from one magnetic-frill source.

    ``Vinc`` and ``Vtotal`` (= V_ab) are at integer electric-field time.
    ``Itot`` is Hyun's average of the surrounding currents at the adjacent
    magnetic half steps, so it is centred at that same integer time. No
    additional phase correction is needed.
    """

    def __init__(
        self,
        source: "MagneticFrillSource",
        source_index: int,
        spectrum_limit: SpectrumLimit = DEFAULT_MINIMUM_WAVELENGTH_CELLS,
        owner=None,
    ):
        self.source = source
        self.source_index = int(source_index)
        self.spectrum_limit = validate_spectrum_limit(spectrum_limit)
        self.owner = owner
        self.result: Optional[MagneticFrillPortResult] = None
        self.prepared = False
        self.minimum_wavelength_cells = (
            DEFAULT_MINIMUM_WAVELENGTH_CELLS
            if self.spectrum_limit == "nyquist"
            else float(self.spectrum_limit)
        )
        self.spectrum_limit_mode = (
            "nyquist" if self.spectrum_limit == "nyquist" else "minimum_wavelength_cells"
        )
        self.incident_floor_db = DEFAULT_INCIDENT_FLOOR_DB

    @property
    def output_id(self) -> str:
        return f"frill{self.source_index}"

    def prepare(self, grid: "FDTDGrid") -> None:
        """Calculate the native frequency axis and mesh-valid output band."""

        if grid.iterations < 2:
            raise ValueError(
                f"Magnetic frill source {self.output_id!r} requires at least two iterations"
            )
        if not np.isfinite(self.source.Z0) or self.source.Z0 <= 0:
            raise ValueError(f"Magnetic frill source {self.output_id!r} has an invalid Z0")

        nsamples = int(grid.iterations)
        full_frequency = np.fft.rfftfreq(nsamples, d=grid.dt)
        cells, limiting_material = minimum_wavelength_sampling(grid, full_frequency)
        mesh_valid_full = cells >= self.minimum_wavelength_cells
        positive_invalid = np.flatnonzero((full_frequency > 0) & ~mesh_valid_full)
        if positive_invalid.size:
            first_invalid = int(positive_invalid[0])
            last_mesh_index = max(0, first_invalid - 1)
            limiting_index = first_invalid
        else:
            last_mesh_index = full_frequency.size - 1
            limiting_index = last_mesh_index

        self.nsamples = nsamples
        self.reference_impedance = float(self.source.Z0)
        self.nyquist_frequency = float(1 / (2 * grid.dt))
        self.mesh_frequency_limit = float(full_frequency[last_mesh_index])
        self.limiting_material = str(limiting_material[limiting_index])
        self.independent_frequency_resolution = float(1 / (nsamples * grid.dt))
        self._full_cells_per_wavelength = cells
        self._full_mesh_valid = mesh_valid_full
        if self.spectrum_limit_mode == "nyquist":
            self._frequency_slice = slice(None)
            logger.warning(
                f"Magnetic frill source {self.output_id!r}: full native spectrum "
                f"0--{full_frequency[-1]:g} Hz requested (Nyquist research "
                f"override); advisory lambda/{self.minimum_wavelength_cells:g} "
                f"limit {self.mesh_frequency_limit:g} Hz in material "
                f"{self.limiting_material!r}."
            )
        else:
            self._frequency_slice = slice(0, last_mesh_index + 1)
            logger.info(
                f"Magnetic frill source {self.output_id!r}: automatic S11/Zin "
                f"spectrum 0--{self.mesh_frequency_limit:g} Hz, lambda/"
                f"{self.minimum_wavelength_cells:g} mesh limit in material "
                f"{self.limiting_material!r}; native df "
                f"{self.independent_frequency_resolution:g} Hz; Nyquist "
                f"{self.nyquist_frequency:g} Hz."
            )
        self.dt = float(grid.dt)
        self.prepared = True

    def finalise(self, grid: "FDTDGrid") -> MagneticFrillPortResult:
        """Calculate automatic S11, Zin, and Yin from the source's own
        Vinc/Vtotal/Itot histories."""

        if not self.prepared:
            self.prepare(grid)

        real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
        complex_dtype = np.dtype(config.sim_config.dtypes["complex"])
        histories = {}
        for name in ("Vinc", "Vtotal", "Itot"):
            values = np.asarray(getattr(self.source, name), dtype=real_dtype)
            if values.ndim != 1 or values.size < self.nsamples:
                raise RuntimeError(
                    f"Magnetic frill source {self.output_id!r} {name} history "
                    "has the wrong length"
                )
            values = values[: self.nsamples]
            if not np.all(np.isfinite(values)):
                raise RuntimeError(
                    f"Magnetic frill source {self.output_id!r} {name} history is not finite"
                )
            histories[name] = values

        # Hyun equations (9)-(11): voltage is at integer time and Itot is
        # averaged from the adjacent magnetic half steps to the same time.
        time_offset = 0.0
        frequency_full, incident_voltage_full = engineering_rfft(
            histories["Vinc"], grid.dt, time_offset=time_offset
        )
        _, total_voltage_full = engineering_rfft(
            histories["Vtotal"], grid.dt, time_offset=time_offset
        )
        _, total_current_full = engineering_rfft(
            histories["Itot"], grid.dt, time_offset=time_offset
        )

        selection = self._frequency_slice
        frequency = frequency_full[selection]
        incident_voltage = incident_voltage_full[selection]
        total_voltage = total_voltage_full[selection]
        total_current = total_current_full[selection]
        reflected_voltage = np.asarray(total_voltage - incident_voltage, dtype=complex_dtype)
        s11, source_defined = _safe_complex_divide(
            reflected_voltage, incident_voltage, complex_dtype
        )

        incident_magnitude = np.abs(incident_voltage)
        incident_peak = float(np.max(incident_magnitude, initial=0.0))
        incident_relative_db = np.full(frequency.shape, -np.inf, dtype=real_dtype)
        if incident_peak > 0:
            nonzero = incident_magnitude > 0
            incident_relative_db[nonzero] = np.asarray(
                20 * np.log10(incident_magnitude[nonzero] / incident_peak),
                dtype=real_dtype,
            )
        source_valid = source_defined & (incident_relative_db >= self.incident_floor_db)

        zin, zin_defined = impedance_from_s11(s11, self.reference_impedance, complex_dtype)
        yin, yin_defined = admittance_from_s11(s11, self.reference_impedance, complex_dtype)

        mesh_valid = np.asarray(self._full_mesh_valid[selection], dtype=bool)
        cells_per_wavelength = np.asarray(
            self._full_cells_per_wavelength[selection], dtype=real_dtype
        )
        valid_s11 = mesh_valid & source_valid
        valid_zin = valid_s11 & zin_defined
        valid_yin = valid_s11 & yin_defined

        trace_peak = float(np.max(np.abs(histories["Vtotal"]), initial=0.0))
        tail_count = max(1, self.nsamples // 20)
        tail_peak = float(np.max(np.abs(histories["Vtotal"][-tail_count:]), initial=0.0))
        if trace_peak > 0 and tail_peak > 0:
            tail_relative_db = float(20 * np.log10(tail_peak / trace_peak))
        elif tail_peak == 0:
            tail_relative_db = float("-inf")
        else:
            tail_relative_db = float("nan")
        if np.isfinite(tail_relative_db) and tail_relative_db > -40:
            logger.warning(
                f"Magnetic frill source {self.output_id!r}: the final 5% of "
                f"the terminal-voltage trace reaches {tail_relative_db:.1f} dB "
                "relative to its peak; spectral leakage may be significant."
            )

        time = np.asarray(
            np.arange(self.nsamples, dtype=real_dtype) * grid.dt,
            dtype=real_dtype,
        )
        self.result = MagneticFrillPortResult(
            time=time,
            frequency=np.asarray(frequency, dtype=real_dtype),
            incident_voltage_spectrum=np.asarray(incident_voltage, dtype=complex_dtype),
            reflected_voltage_spectrum=reflected_voltage,
            total_voltage_spectrum=np.asarray(total_voltage, dtype=complex_dtype),
            total_current_spectrum=np.asarray(total_current, dtype=complex_dtype),
            s11=s11,
            zin=zin,
            yin=yin,
            source_valid=np.asarray(source_valid, dtype=bool),
            mesh_valid=mesh_valid,
            valid_s11=np.asarray(valid_s11, dtype=bool),
            valid_zin=np.asarray(valid_zin, dtype=bool),
            valid_yin=np.asarray(valid_yin, dtype=bool),
            incident_relative_db=incident_relative_db,
            cells_per_minimum_wavelength=cells_per_wavelength,
            tail_relative_db=tail_relative_db,
        )
        return self.result

    def write_hdf5(self, group) -> None:
        """Add derived terminal quantities to an existing ``/frills/frillN`` group."""

        if self.result is None:
            raise RuntimeError(f"Magnetic frill source {self.output_id!r} has not been finalised")
        result = self.result
        real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
        complex_dtype = np.dtype(config.sim_config.dtypes["complex"])

        group.attrs["ReferenceImpedance"] = self.reference_impedance
        group.attrs["ZinPrimaryMethod"] = "voltage_wave_S11"
        group.attrs["TimeOffset"] = 0.0
        group.attrs["Window"] = "rectangular"
        group.attrs["IncidentFloorDB"] = self.incident_floor_db
        group.attrs["SpectrumLimitMode"] = self.spectrum_limit_mode
        group.attrs["NyquistFrequency"] = self.nyquist_frequency
        group.attrs["MinimumWavelengthCells"] = self.minimum_wavelength_cells
        group.attrs["MeshFrequencyLimit"] = self.mesh_frequency_limit
        group.attrs["LimitingMaterial"] = self.limiting_material
        group.attrs["IndependentFrequencyResolution"] = self.independent_frequency_resolution
        group.attrs["FrequencyRange"] = np.asarray(
            (result.frequency[0], result.frequency[-1]), dtype=real_dtype
        )
        valid_indices = np.flatnonzero(result.valid_s11)
        valid_range = (
            (result.frequency[valid_indices[0]], result.frequency[valid_indices[-1]])
            if valid_indices.size
            else (np.nan, np.nan)
        )
        group.attrs["ValidFrequencyRange"] = np.asarray(valid_range, dtype=real_dtype)
        group.attrs["TailRelativeLevelDB"] = result.tail_relative_db
        group.attrs["phasor_time_sign"] = PHASOR_TIME_DEPENDENCE
        group.attrs["forward_transform_sign"] = FORWARD_TRANSFORM_KERNEL
        group.attrs["real_dtype"] = real_dtype.name
        group.attrs["complex_dtype"] = complex_dtype.name

        datasets = {
            "time": result.time,
            "frequency": result.frequency,
            "Vincident_spectrum": result.incident_voltage_spectrum,
            "Vreflected_spectrum": result.reflected_voltage_spectrum,
            "Vtotal_spectrum": result.total_voltage_spectrum,
            "Itotal_spectrum": result.total_current_spectrum,
            "S11": result.s11,
            "Zin": result.zin,
            "Yin": result.yin,
            "valid_S11": result.valid_s11.astype(np.uint8),
            "valid_Zin": result.valid_zin.astype(np.uint8),
            "valid_Yin": result.valid_yin.astype(np.uint8),
            "source_valid": result.source_valid.astype(np.uint8),
            "mesh_valid": result.mesh_valid.astype(np.uint8),
            "incident_relative_dB": result.incident_relative_db,
            "cells_per_minimum_wavelength": result.cells_per_minimum_wavelength,
        }
        for name, values in datasets.items():
            group.create_dataset(name, data=values)


def prepare_magnetic_frill_ports(grid: "FDTDGrid") -> None:
    """Create and prepare the automatic port output for every frill source."""

    for index, source in enumerate(grid.magneticfrillsources, start=1):
        output = getattr(source, "port_output", None)
        if output is None:
            output = MagneticFrillPortOutput(
                source,
                index,
                spectrum_limit=getattr(source, "spectrum_limit", DEFAULT_MINIMUM_WAVELENGTH_CELLS),
            )
            source.port_output = output
        else:
            output.source_index = index
        output.prepare(grid)


def finalise_magnetic_frill_ports(grid: "FDTDGrid") -> None:
    """Finalise all automatic magnetic-frill-source port outputs."""

    for index, source in enumerate(grid.magneticfrillsources, start=1):
        output = getattr(source, "port_output", None)
        if output is None:
            output = MagneticFrillPortOutput(source, index)
            source.port_output = output
        else:
            output.source_index = index
        output.finalise(grid)
