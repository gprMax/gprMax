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

"""Source-bound port outputs.

Voltage-source ports use the known Thevenin generator voltage and the electric
field sampled on one electric Yee edge. Transmission-line sources already
carry incident and terminal voltage/current histories, so their S11 and input
impedance are calculated automatically without an additional receiver.
"""

import logging
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
import numpy.typing as npt

import gprMax.config as config
from gprMax.ntff.conventions import FORWARD_TRANSFORM_KERNEL, PHASOR_TIME_DEPENDENCE

if TYPE_CHECKING:
    from gprMax.grid.fdtd_grid import FDTDGrid
    from gprMax.receivers import Rx
    from gprMax.sources import MagneticFrillSource, TransmissionLine, VoltageSource


logger = logging.getLogger(__name__)

DEFAULT_MINIMUM_WAVELENGTH_CELLS = 10.0
MINIMUM_PHYSICAL_WAVELENGTH_CELLS = 3.0
DEFAULT_INCIDENT_FLOOR_DB = -40.0

SpectrumLimit = Union[float, Literal["nyquist"]]


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
    scale = float(np.max(magnitude[finite_denominator], initial=0.0))
    threshold = np.finfo(np.empty((), dtype=complex_dtype).real.dtype).eps * scale
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


class VoltageSourcePortMonitor:
    """Bind one internal electric-field receiver to one voltage source."""

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

    def _edge_geometry(self, grid):
        if self.source.polarisation == "x":
            return float(grid.dx), float(grid.dy * grid.dz)
        if self.source.polarisation == "y":
            return float(grid.dy), float(grid.dx * grid.dz)
        return float(grid.dz), float(grid.dx * grid.dy)

    def prepare(self, grid: "FDTDGrid") -> None:
        """Validate the built Yee edge and calculate fixed port parameters."""

        if grid.iterations < 2:
            raise ValueError(f"RxPort {self.output_id!r} requires at least two iterations")
        if not np.array_equal(self.receiver.coord, self.source.coord):
            raise ValueError(f"RxPort {self.output_id!r} receiver is no longer source-bound")
        if getattr(self.source, "background_material_numID", None) is None:
            raise ValueError(f"RxPort {self.output_id!r} source material was not constructed")
        if getattr(self.source, "background_is_dispersive", False):
            raise ValueError(
                f"RxPort {self.output_id!r} does not yet support a dispersive material "
                "on the voltage-source edge"
            )

        self.dl, self.area = self._edge_geometry(grid)
        self.reference_impedance = float(self.source.resistance)
        self.background_relative_permittivity = float(self.source.background_er)
        self.background_conductivity = float(self.source.background_se)
        if not np.isfinite(self.background_conductivity):
            raise ValueError(f"RxPort {self.output_id!r} cannot use a voltage source on a PEC edge")
        self.gap_capacitance = (
            config.sim_config.em_consts["e0"]
            * self.background_relative_permittivity
            * self.area
            / self.dl
        )
        self.background_conductance = self.background_conductivity * self.area / self.dl

        source_material_id = int(self.source.source_material_numID)
        source_material = grid.materials[source_material_id]
        added_conductance = (
            (float(source_material.se) - self.background_conductivity) * self.area / self.dl
        )
        dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
        tolerance = 64 * np.finfo(dtype).eps
        if not np.isclose(
            added_conductance,
            1 / self.reference_impedance,
            rtol=tolerance,
            atol=tolerance / self.reference_impedance,
        ):
            raise ValueError(
                f"RxPort {self.output_id!r} source-edge conductance is inconsistent "
                "with its source resistance"
            )
        if not np.isfinite(grid.updatecoeffsE[source_material_id, 4]) or np.isclose(
            grid.updatecoeffsE[source_material_id, 4], 0
        ):
            raise ValueError(f"RxPort {self.output_id!r} source is on an inactive electric edge")

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
                f"RxPort {self.output_id!r}: full native spectrum 0--"
                f"{full_frequency[-1]:g} Hz requested (Nyquist research override); "
                f"advisory lambda/{self.minimum_wavelength_cells:g} limit "
                f"{self.mesh_frequency_limit:g} Hz in material "
                f"{self.limiting_material!r}."
            )
        else:
            self._frequency_slice = slice(0, last_mesh_index + 1)
            log = logger.info
            message = (
                f"RxPort {self.output_id!r}: spectrum 0--"
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
        if not np.array_equal(self.receiver.coord, self.source.coord):
            raise RuntimeError(f"RxPort {self.output_id!r} receiver moved away from its source")

        real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
        complex_dtype = np.dtype(config.sim_config.dtypes["complex"])
        electric = np.asarray(self.receiver.outputs[self.component], dtype=real_dtype)
        if electric.size != grid.iterations:
            raise RuntimeError(f"RxPort {self.output_id!r} receiver history has the wrong length")

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

        omega_discrete = (2 / grid.dt) * np.tan(np.pi * frequency * grid.dt)
        gap_correction = np.asarray(
            self.reference_impedance
            * (self.background_conductance + 1j * omega_discrete * self.gap_capacitance),
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
                f"RxPort {self.output_id!r}: the final 5% of the voltage trace "
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

    def write_hdf5(self, base_group) -> None:
        """Write the finalised port result beneath ``/ports/<ID>``."""

        if self.result is None:
            raise RuntimeError(f"RxPort {self.output_id!r} has not been finalised")
        result = self.result
        ports_group = base_group.require_group("ports")
        group = ports_group.create_group(self.output_id)
        real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
        complex_dtype = np.dtype(config.sim_config.dtypes["complex"])

        group.attrs["Name"] = self.output_id
        group.attrs["Position"] = np.asarray(self.source.coord * self.grid_dl, dtype=real_dtype)
        group.attrs["GridPosition"] = np.asarray(self.source.coord, dtype=np.int32)
        group.attrs["SourceType"] = type(self.source).__name__
        group.attrs["SourceIndex"] = self.source_index
        group.attrs["Polarisation"] = self.source.polarisation
        group.attrs["CellLength"] = self.dl
        group.attrs["ReferenceImpedance"] = self.reference_impedance
        group.attrs["WaveformID"] = self.source.waveformID
        group.attrs["BackgroundMaterial"] = self.source.background_material_ID
        group.attrs["BackgroundRelativePermittivity"] = self.background_relative_permittivity
        group.attrs["BackgroundConductivity"] = self.background_conductivity
        group.attrs["GapCapacitance"] = self.gap_capacitance
        group.attrs["BackgroundConductance"] = self.background_conductance
        group.attrs["GapCorrection"] = "discrete_parallel_admittance"
        group.attrs["TimeSampleOffset"] = 0.5 * self.dt
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
        for name, values in datasets.items():
            group.create_dataset(name, data=values)

    def set_output_context(self, grid: "FDTDGrid") -> None:
        """Cache immutable HDF5 context after the grid is fully built."""

        self.grid_dl = np.asarray((grid.dx, grid.dy, grid.dz), dtype=np.float64)
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
class RxPortOverride:
    """Marker left on a ``MagneticFrillSource`` by ``RxPort.build()``.

    Unlike a voltage source, a magnetic-frill source's S11/Zin/Yin output is
    always on (built by ``prepare_magnetic_frill_ports`` regardless of
    whether ``#rx_port`` is present) - so ``#rx_port`` paired with this
    source type does not create a second, independent monitor. It only
    overrides the ``spectrum_limit`` of that always-on output, consumed once
    ``prepare_magnetic_frill_ports`` constructs the real
    ``MagneticFrillPortOutput`` object (which does not exist yet at the point
    ``RxPort.build()`` itself runs - see ``gprMax/model.py``'s build
    sequencing).
    """

    spectrum_limit: SpectrumLimit
    owner: object


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
            override = getattr(source, "_rx_port_override", None)
            if override is not None:
                output = MagneticFrillPortOutput(
                    source,
                    index,
                    spectrum_limit=override.spectrum_limit,
                    owner=override.owner,
                )
            else:
                output = MagneticFrillPortOutput(source, index)
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
