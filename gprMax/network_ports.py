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

"""Sparse rational lumped networks coupled to electric Yee edges.

The driving-point admittance of a network is represented by

    Y(s) = G + s C + sum(residue / (s - pole)).

Each placed terminal owns only its pole states and time histories. The field
coupling is a local correction to one electric edge; it does not allocate
dispersive state over the complete FDTD mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

import gprMax.config as config


def _phi1(value: complex) -> complex:
    """Return expm1(value) / value without cancellation near zero."""

    if abs(value) < 1e-4:
        return 1 + value / 2 + value**2 / 6 + value**3 / 24 + value**4 / 120
    return np.expm1(value) / value


def _phi2(value: complex) -> complex:
    """Return (expm1(value) - value) / value**2 near zero safely."""

    if abs(value) < 1e-4:
        return 0.5 + value / 6 + value**2 / 24 + value**3 / 120 + value**4 / 720
    return (np.expm1(value) - value) / value**2


def linear_interval_coefficients(
    pole: complex,
    residue: complex,
    dt: float,
    fraction: float,
) -> tuple[complex, complex, complex]:
    """Return exact state coefficients for a linearly varying voltage.

    For ``x_dot = pole*x + residue*u`` and a linear ``u`` over one FDTD
    interval, the state at ``n + fraction`` is

    ``exp_fraction*x[n] + coeff_new*u[n+1] + coeff_old*u[n]``.

    The formulation is the fractional-time extension of the exponential
    recursive-convolution treatment used for dispersive media in gprMax.
    """

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(fraction) or fraction < 0 or fraction > 1:
        raise ValueError("fraction must lie between zero and one")
    z = complex(pole) * float(dt)
    theta_z = fraction * z
    exp_fraction = np.exp(theta_z)
    coeff_new = complex(residue) * dt * fraction**2 * _phi2(theta_z)
    coeff_old = complex(residue) * dt * (fraction * _phi1(theta_z) - fraction**2 * _phi2(theta_z))
    return exp_fraction, coeff_new, coeff_old


@dataclass(frozen=True)
class RationalNetworkModel:
    """Reusable scalar driving-point admittance model."""

    ID: str
    conductance: float = 0.0
    capacitance: float = 0.0
    poles: tuple[complex, ...] = ()
    residues: tuple[complex, ...] = ()
    allow_active: bool = False

    def __post_init__(self) -> None:
        conductance = float(self.conductance)
        capacitance = float(self.capacitance)
        poles = tuple(complex(value) for value in self.poles)
        residues = tuple(complex(value) for value in self.residues)
        if not self.ID or "/" in self.ID or "\x00" in self.ID:
            raise ValueError("rational network ID must be a non-empty HDF5 path component")
        if not np.isfinite(conductance) or not np.isfinite(capacitance):
            raise ValueError("network conductance and capacitance must be finite")
        if len(poles) != len(residues):
            raise ValueError("network poles and residues must have identical lengths")
        if any(
            not (np.isfinite(value.real) and np.isfinite(value.imag)) for value in poles + residues
        ):
            raise ValueError("network poles and residues must be finite")
        if any(value.real > 0 for value in poles):
            raise ValueError("network poles must lie in the closed left half-plane")
        if not self.allow_active and (conductance < 0 or capacitance < 0):
            raise ValueError("negative conductance or capacitance requires allow_active=True")

        tolerance = 1e-12
        unmatched = list(range(len(poles)))
        while unmatched:
            index = unmatched.pop(0)
            pole = poles[index]
            residue = residues[index]
            if abs(pole.imag) <= tolerance and abs(residue.imag) <= tolerance:
                continue
            match = next(
                (
                    other
                    for other in unmatched
                    if np.isclose(poles[other], np.conj(pole), rtol=tolerance, atol=tolerance)
                    and np.isclose(
                        residues[other], np.conj(residue), rtol=tolerance, atol=tolerance
                    )
                ),
                None,
            )
            if match is None:
                raise ValueError("complex network poles and residues must occur in conjugate pairs")
            unmatched.remove(match)

        object.__setattr__(self, "conductance", conductance)
        object.__setattr__(self, "capacitance", capacitance)
        object.__setattr__(self, "poles", poles)
        object.__setattr__(self, "residues", residues)

    def admittance(self, frequency: npt.ArrayLike) -> npt.NDArray[np.complexfloating]:
        """Evaluate the continuous-time admittance at non-negative frequencies."""

        values = np.asarray(frequency, dtype=np.float64)
        if np.any(values < 0) or not np.all(np.isfinite(values)):
            raise ValueError("network frequencies must be finite and non-negative")
        s = 2j * np.pi * values
        result = np.asarray(self.conductance + s * self.capacitance, dtype=np.complex128)
        for pole, residue in zip(self.poles, self.residues):
            denominator = s - pole
            singular = denominator == 0
            term = np.full(result.shape, np.inf + 0j, dtype=np.complex128)
            np.divide(residue, denominator, out=term, where=~singular)
            result = result + term
        return result

    def validate_passivity(self, frequencies: npt.ArrayLike, tolerance: float = 0.0) -> None:
        """Reject a passive model whose sampled real admittance is negative."""

        if self.allow_active:
            return
        admittance = self.admittance(frequencies)
        if np.any(np.real(admittance) < -abs(float(tolerance))):
            minimum = float(np.min(np.real(admittance)))
            raise ValueError(
                f"network {self.ID!r} is non-passive over the checked band "
                f"(minimum real admittance {minimum:g} S)"
            )


class RationalNetworkTerminal:
    """One rational network connected to one electric Yee edge."""

    def __init__(
        self,
        terminal_id: str,
        model: RationalNetworkModel,
        coord: npt.ArrayLike,
        polarisation: str,
    ) -> None:
        self.ID = terminal_id
        self.model = model
        self.coord = np.asarray(coord, dtype=np.int32)
        self.coordorigin = self.coord.copy()
        self.polarisation = polarisation
        self.waveformID: Optional[str] = None
        self.start = 0.0
        self.stop = 0.0
        self.output = None
        self.prepared = False
        self.study_scale = 1.0

    @property
    def xcoord(self) -> int:
        return int(self.coord[0])

    @property
    def ycoord(self) -> int:
        return int(self.coord[1])

    @property
    def zcoord(self) -> int:
        return int(self.coord[2])

    @property
    def excited(self) -> bool:
        return self.waveformID is not None

    def set_excitation(
        self, waveform_id: str, start: Optional[float], stop: Optional[float]
    ) -> None:
        if self.waveformID is not None:
            raise ValueError(f"network terminal {self.ID!r} already has an excitation")
        self.waveformID = waveform_id
        self.start = 0.0 if start is None else float(start)
        self.stop = np.inf if stop is None else float(stop)

    def configure_study_excitation(self, grid, waveform_id, start, stop, scale) -> None:
        """Replace only the generator drive while retaining terminal topology."""

        if not any(waveform.ID == waveform_id for waveform in grid.waveforms):
            raise ValueError(
                f"network terminal {self.ID!r} study drive references unknown "
                f"waveform {waveform_id!r}"
            )
        start = float(start)
        stop = min(float(stop), float(grid.timewindow))
        scale = float(scale)
        if not np.isfinite(scale):
            raise ValueError(f"network terminal {self.ID!r} study scale must be finite")
        if start < 0 or stop <= start:
            raise ValueError(
                f"network terminal {self.ID!r} study drive requires "
                "0 <= start < stop <= the model time window"
            )

        self.waveformID = waveform_id
        self.start = start
        self.stop = stop
        self.study_scale = scale
        if self.prepared:
            self._prepare_waveform(grid)
            self.reset()
        if self.output is not None:
            self.output.result = None

    def _edge_geometry(self, grid) -> tuple[float, float]:
        if self.polarisation == "x":
            return float(grid.dx), float(grid.dy * grid.dz)
        if self.polarisation == "y":
            return float(grid.dy), float(grid.dx * grid.dz)
        return float(grid.dz), float(grid.dx * grid.dy)

    def _field(self, grid):
        return {"x": grid.Ex, "y": grid.Ey, "z": grid.Ez}[self.polarisation]

    def prepare(self, grid) -> None:
        """Bind final material coefficients and initialise sparse recurrence state."""

        if self.prepared:
            self.reset()
            return
        if grid.within_pml(self.coord):
            raise ValueError(f"network terminal {self.ID!r} cannot be placed inside a PML")
        component = f"E{self.polarisation}"
        component_index = grid.IDlookup[component]
        material_id = int(grid.ID[component_index, self.xcoord, self.ycoord, self.zcoord])
        material = grid.materials[material_id]
        if material.ID == "pec" or not np.isfinite(material.se):
            raise ValueError(f"network terminal {self.ID!r} is on an inactive electric edge")
        if hasattr(material, "poles"):
            raise ValueError(
                f"network terminal {self.ID!r} does not yet support a dispersive background edge"
            )
        if any(
            source.polarisation == self.polarisation and np.array_equal(source.coord, self.coord)
            for source in grid.voltagesources + grid.transmissionlines
        ):
            raise ValueError(
                f"network terminal {self.ID!r} conflicts with another terminal source on the same electric edge"
            )

        self.material_id = material_id
        self.background_material_ID = str(material.ID)
        self.background_relative_permittivity = float(material.er)
        self.background_conductivity = float(material.se)
        self.dl, self.area = self._edge_geometry(grid)
        self.source_coefficient = float(grid.updatecoeffsE[material_id, 4])
        if not np.isfinite(self.source_coefficient) or self.source_coefficient <= 0:
            raise ValueError(
                f"network terminal {self.ID!r} has an invalid electric source coefficient"
            )

        complex_dtype = np.dtype(config.sim_config.dtypes["complex"])
        self.exp_half = np.empty(len(self.model.poles), dtype=complex_dtype)
        self.coeff_half_new = np.empty(len(self.model.poles), dtype=complex_dtype)
        self.coeff_half_old = np.empty(len(self.model.poles), dtype=complex_dtype)
        self.exp_full = np.empty(len(self.model.poles), dtype=complex_dtype)
        self.coeff_full_new = np.empty(len(self.model.poles), dtype=complex_dtype)
        self.coeff_full_old = np.empty(len(self.model.poles), dtype=complex_dtype)
        for index, (pole, residue) in enumerate(zip(self.model.poles, self.model.residues)):
            (
                self.exp_half[index],
                self.coeff_half_new[index],
                self.coeff_half_old[index],
            ) = linear_interval_coefficients(pole, residue, grid.dt, 0.5)
            (
                self.exp_full[index],
                self.coeff_full_new[index],
                self.coeff_full_old[index],
            ) = linear_interval_coefficients(pole, residue, grid.dt, 1.0)

        alpha_complex = (
            self.model.conductance / 2
            + self.model.capacitance / grid.dt
            + np.sum(self.coeff_half_new, dtype=complex_dtype)
        )
        tolerance = 256 * np.finfo(np.dtype(config.sim_config.dtypes["float_or_double"])).eps
        if abs(float(np.imag(alpha_complex))) > tolerance * max(
            1.0, abs(float(np.real(alpha_complex)))
        ):
            raise ValueError(
                f"network terminal {self.ID!r} has a non-real local update coefficient"
            )
        self.alpha = float(np.real(alpha_complex))
        denominator = 1 + self.source_coefficient * self.alpha * self.dl / self.area
        if not np.isfinite(denominator) or denominator <= tolerance:
            raise ValueError(
                f"network terminal {self.ID!r} has a singular or non-passive local update denominator"
            )
        self.denominator = float(denominator)

        nyquist = 1 / (2 * grid.dt)
        checked_frequencies = np.linspace(nyquist / 2048, nyquist, 2048)
        sampled_admittance = self.model.admittance(checked_frequencies)
        passivity_tolerance = (
            256
            * np.finfo(np.float64).eps
            * max(1.0, float(np.max(np.abs(sampled_admittance), initial=0.0)))
        )
        self.model.validate_passivity(
            checked_frequencies,
            tolerance=passivity_tolerance,
        )
        self._prepare_waveform(grid)
        real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
        self.voltage = np.zeros(grid.iterations + 1, dtype=real_dtype)
        self.network_current = np.zeros(grid.iterations, dtype=real_dtype)
        self.generator_voltage = np.zeros(grid.iterations, dtype=real_dtype)
        self.states = np.zeros(len(self.model.poles), dtype=complex_dtype)
        self.prepared = True

    def _prepare_waveform(self, grid) -> None:
        real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
        self.waveform_whole = np.zeros(grid.iterations + 1, dtype=real_dtype)
        self.waveform_half = np.zeros(grid.iterations, dtype=real_dtype)
        if not self.excited:
            return
        waveform = next((item for item in grid.waveforms if item.ID == self.waveformID), None)
        if waveform is None:
            raise ValueError(
                f"network terminal {self.ID!r} has no waveform with identifier {self.waveformID!r}"
            )
        stop = min(float(self.stop), float(grid.timewindow))
        if self.start < 0 or stop <= self.start:
            raise ValueError(f"network terminal {self.ID!r} has an invalid excitation interval")
        for iteration in range(grid.iterations + 1):
            time = iteration * grid.dt
            if self.start <= time <= stop:
                self.waveform_whole[iteration] = waveform.calculate_value(
                    time - self.start, grid.dt
                )
            half_time = time + 0.5 * grid.dt
            if iteration < grid.iterations and self.start <= half_time <= stop:
                self.waveform_half[iteration] = waveform.calculate_value(
                    half_time - self.start, grid.dt
                )
        self.waveform_whole *= self.study_scale
        self.waveform_half *= self.study_scale

    def reset(self) -> None:
        """Return all dynamic network state and histories to rest."""

        if not self.prepared:
            return
        self.states.fill(0)
        self.voltage.fill(0)
        self.network_current.fill(0)
        self.generator_voltage.fill(0)

    def update(self, iteration: int, grid) -> None:
        """Apply the locally implicit edge correction and advance pole states."""

        if not self.prepared:
            raise RuntimeError(f"network terminal {self.ID!r} has not been prepared")
        from gprMax.cython.network_port import update_rational_network_terminal

        field = self._field(grid)
        current = update_rational_network_terminal(
            self.xcoord,
            self.ycoord,
            self.zcoord,
            self.dl,
            self.area,
            self.source_coefficient,
            self.denominator,
            self.alpha,
            self.model.conductance,
            self.model.capacitance,
            grid.dt,
            self.voltage[iteration],
            self.waveform_whole[iteration],
            self.waveform_whole[iteration + 1],
            self.waveform_half[iteration],
            self.exp_half,
            self.coeff_half_new,
            self.coeff_half_old,
            self.exp_full,
            self.coeff_full_new,
            self.coeff_full_old,
            self.states,
            field,
        )
        self.voltage[iteration + 1] = -self.dl * field[self.xcoord, self.ycoord, self.zcoord]
        self.network_current[iteration] = current
        self.generator_voltage[iteration] = self.waveform_half[iteration]


def rational_network_host_arrays(terminals, grid) -> dict[str, np.ndarray]:
    """Pack independent rational terminals into backend-neutral arrays.

    One device work-item owns one terminal and advances all of its pole
    states serially. Complex recurrence coefficients are split into real and
    imaginary arrays so the identical kernel body works on CUDA, OpenCL, and
    Metal without relying on backend-specific complex ABIs.
    """

    count = len(terminals)
    real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
    total_poles = sum(len(terminal.model.poles) for terminal in terminals)
    if total_poles > np.iinfo(np.int32).max:
        raise ValueError("rational-network pole state exceeds the signed 32-bit index range")
    largest_terminal_stride = max(grid.iterations + 1, 7)
    if count * largest_terminal_stride > np.iinfo(np.int32).max:
        raise ValueError("rational-network histories exceed the signed 32-bit index range")

    # x, y, z, polarisation, pole offset, pole count
    info = np.zeros((count, 6), dtype=np.int32)
    # dl, area, source coefficient, denominator, alpha, G, C
    params = np.zeros((count, 7), dtype=real_dtype)
    waveform_whole = np.zeros((count, grid.iterations + 1), dtype=real_dtype)
    waveform_half = np.zeros((count, grid.iterations), dtype=real_dtype)
    voltage = np.zeros((count, grid.iterations + 1), dtype=real_dtype)
    current = np.zeros((count, grid.iterations), dtype=real_dtype)

    # Device APIs do not consistently accept zero-byte buffers. A pole-free
    # resistor/capacitor therefore carries one unused scalar allocation.
    pole_storage = max(total_poles, 1)
    coefficient_names = (
        "exp_half",
        "coeff_half_new",
        "coeff_half_old",
        "exp_full",
        "coeff_full_new",
        "coeff_full_old",
    )
    coefficients = {
        f"{name}_{part}": np.zeros(pole_storage, dtype=real_dtype)
        for name in coefficient_names
        for part in ("real", "imag")
    }
    state_real = np.zeros(pole_storage, dtype=real_dtype)
    state_imag = np.zeros(pole_storage, dtype=real_dtype)

    offset = 0
    polarisation_index = {"x": 0, "y": 1, "z": 2}
    for index, terminal in enumerate(terminals):
        pole_count = len(terminal.model.poles)
        info[index] = (
            terminal.xcoord,
            terminal.ycoord,
            terminal.zcoord,
            polarisation_index[terminal.polarisation],
            offset,
            pole_count,
        )
        params[index] = (
            terminal.dl,
            terminal.area,
            terminal.source_coefficient,
            terminal.denominator,
            terminal.alpha,
            terminal.model.conductance,
            terminal.model.capacitance,
        )
        waveform_whole[index] = terminal.waveform_whole
        waveform_half[index] = terminal.waveform_half
        voltage[index] = terminal.voltage
        current[index] = terminal.network_current
        if pole_count:
            section = slice(offset, offset + pole_count)
            for name in coefficient_names:
                values = np.asarray(getattr(terminal, name))
                coefficients[f"{name}_real"][section] = np.real(values)
                coefficients[f"{name}_imag"][section] = np.imag(values)
            state_real[section] = np.real(terminal.states)
            state_imag[section] = np.imag(terminal.states)
        offset += pole_count

    return {
        "info": info,
        "params": params,
        "waveform_whole": waveform_whole,
        "waveform_half": waveform_half,
        "voltage": voltage,
        "current": current,
        **coefficients,
        "state_real": state_real,
        "state_imag": state_imag,
    }


def htod_rational_network_arrays(terminals, grid, queue=None):
    """Copy packed rational-network arrays to the active compute device."""

    arrays = rational_network_host_arrays(terminals, grid)
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
    raise ValueError(f"Unknown device solver {solver!r} for rational-network arrays.")


def dtoh_rational_network_outputs(voltage, current, grid) -> None:
    """Copy device terminal histories into their runtime terminal objects."""

    voltage_shape = (len(grid.networkterminals), grid.iterations + 1)
    current_shape = (len(grid.networkterminals), grid.iterations)
    if config.sim_config.general["solver"] == "metal":
        dtype = np.dtype(config.sim_config.dtypes["float_or_double"])

        def metal_array(buffer, shape):
            nbytes = int(np.prod(shape)) * dtype.itemsize
            if buffer.length() != nbytes:
                raise ValueError(
                    "Rational-network Metal output buffer has the wrong size: "
                    f"expected {nbytes} bytes, got {buffer.length()}."
                )
            return (
                np.frombuffer(buffer.contents().as_buffer(nbytes), dtype=dtype)
                .reshape(shape)
                .copy()
            )

        voltage = metal_array(voltage, voltage_shape)
        current = metal_array(current, current_shape)

    if voltage.shape != voltage_shape or current.shape != current_shape:
        raise ValueError(
            "Rational-network device output shape does not match the grid: "
            f"expected {voltage_shape} and {current_shape}, got "
            f"{voltage.shape} and {current.shape}."
        )
    for index, terminal in enumerate(grid.networkterminals):
        np.copyto(terminal.voltage, voltage[index], casting="same_kind")
        np.copyto(terminal.network_current, current[index], casting="same_kind")
        np.copyto(terminal.generator_voltage, terminal.waveform_half, casting="same_kind")
