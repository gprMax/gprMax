# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax.

"""Frequency-domain specific absorption rate from tagged FDTD cells."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import numpy.typing as npt

import gprMax.config as config

try:
    from gprMax.cython.ntff import accumulate_sparse_dft as _accumulate_sparse_dft
except ImportError:  # pragma: no cover - source tree before Cython compilation
    _accumulate_sparse_dft = None
from gprMax.ntff.conventions import (
    FORWARD_TRANSFORM_KERNEL,
    PHASOR_TIME_DEPENDENCE,
    engineering_dft,
)
from gprMax.ntff.frequency_domain import validate_nyquist_frequencies
from gprMax.ports import (
    DEFAULT_INCIDENT_FLOOR_DB,
    DEFAULT_MINIMUM_WAVELENGTH_CELLS,
    evaluate_port_power_spectrum,
    minimum_wavelength_sampling,
    port_output_registry,
    validate_spectrum_limit,
)
from gprMax.sar_averaging import apply_spatial_average_plan, build_spatial_average_plan

if TYPE_CHECKING:
    from gprMax.grid.fdtd_grid import FDTDGrid


logger = logging.getLogger(__name__)


EDGE_OFFSETS = {
    "Ex": np.asarray(((0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1)), dtype=np.int32),
    "Ey": np.asarray(((0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)), dtype=np.int32),
    "Ez": np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)), dtype=np.int32),
}


def _pml_cell_mask(grid: "FDTDGrid") -> npt.NDArray[np.bool_]:
    """Return cells occupied by boundary or internal PML coordinate stretching."""

    mask = np.zeros((grid.nx, grid.ny, grid.nz), dtype=bool)
    thickness = grid.pmls["thickness"]
    if thickness["x0"]:
        mask[: thickness["x0"], :, :] = True
    if thickness["xmax"]:
        mask[grid.nx - thickness["xmax"] :, :, :] = True
    if thickness["y0"]:
        mask[:, : thickness["y0"], :] = True
    if thickness["ymax"]:
        mask[:, grid.ny - thickness["ymax"] :, :] = True
    if thickness["z0"]:
        mask[:, :, : thickness["z0"]] = True
    if thickness["zmax"]:
        mask[:, :, grid.nz - thickness["zmax"] :] = True
    for spec in grid.pmls["internal_specs"]:
        mask[spec.xs : spec.xf, spec.ys : spec.yf, spec.zs : spec.zf] = True
    return mask


@dataclass(frozen=True)
class SARSpec:
    """Deferred SAR request compiled after Yee material IDs are finalised."""

    output_id: str
    frequencies: tuple[float, ...]
    tags: tuple[str, ...]
    waveform_id: str
    target_amplitude: float
    spectrum_limit: float | str
    source_floor_db: float
    window: str
    averaging_masses: tuple[float, ...]
    normalisation: str
    port_id: str | None
    target_power: float | None
    owner: Any = None


@dataclass(frozen=True)
class SARResult:
    """Final frequency-domain SAR data for selected tagged cells."""

    frequency: npt.NDArray[np.floating]
    cell_indices: npt.NDArray[np.integer]
    tag_id: npt.NDArray[np.integer]
    material_id: npt.NDArray[np.integer]
    density: npt.NDArray[np.floating]
    source_spectrum: npt.NDArray[np.complexfloating]
    source_relative_db: npt.NDArray[np.floating]
    source_valid: npt.NDArray[np.bool_]
    mesh_valid: npt.NDArray[np.bool_]
    valid: npt.NDArray[np.bool_]
    cells_per_wavelength: npt.NDArray[np.floating]
    limiting_material: npt.NDArray[np.str_]
    absorbed_power_density: npt.NDArray[np.floating]
    sar: npt.NDArray[np.floating]
    spatial_averages: tuple["SARSpatialAverageResult", ...]
    normalisation_scale: npt.NDArray[np.complexfloating]
    normalising_power: npt.NDArray[np.floating]


@dataclass(frozen=True)
class SARSpatialAverageResult:
    """Mass-based spatial-average SAR sampled at selected tagged cells."""

    target_mass: float
    sar: npt.NDArray[np.floating]
    status: npt.NDArray[np.uint8]
    averaging_mass: npt.NDArray[np.floating]
    averaging_volume: npt.NDArray[np.floating]
    orientation: npt.NDArray[np.int8]
    peak_sar: npt.NDArray[np.floating]
    peak_cell: npt.NDArray[np.integer]


def _window(name: str, count: int, dtype: np.dtype) -> npt.NDArray[np.floating]:
    normalised = str(name).lower().replace("-", "_")
    if normalised in ("rectangular", "boxcar", "none"):
        return np.ones(count, dtype=dtype)
    if normalised in ("hann", "hanning"):
        return np.asarray(np.hanning(count), dtype=dtype)
    raise ValueError("SAR window must be 'rectangular' or 'hann'")


def _material_relative_permittivity(material, frequencies: npt.NDArray[np.floating]):
    values = np.empty(frequencies.shape, dtype=np.complex128)
    if hasattr(material, "poles"):
        values[:] = np.asarray(
            [material.calculate_er(float(frequency)) for frequency in frequencies],
            dtype=np.complex128,
        )
    else:
        omega = 2 * np.pi * frequencies
        values[:] = material.er + material.se / (1j * omega * config.sim_config.em_consts["e0"])
    return values


def _material_loss_conductivity(grid: "FDTDGrid", numeric_ids, frequencies):
    """Return effective electric loss conductivity for material IDs."""

    numeric_ids = np.asarray(numeric_ids, dtype=np.int64)
    material_by_id = {int(material.numID): material for material in grid.materials}
    sigma = np.empty((frequencies.size, numeric_ids.size), dtype=np.float64)
    omega = 2 * np.pi * np.asarray(frequencies, dtype=np.float64)
    epsilon_0 = float(config.sim_config.em_consts["e0"])
    for material_id in np.unique(numeric_ids):
        material = material_by_id[int(material_id)]
        epsilon_r = _material_relative_permittivity(material, frequencies)
        effective = -omega * epsilon_0 * np.imag(epsilon_r)
        tolerance = 128 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(effective))
        if np.any(effective < -tolerance):
            raise ValueError(
                f"SAR does not support active electric material {material.ID!r} with negative loss"
            )
        effective = np.maximum(effective, 0.0)
        sigma[:, numeric_ids == material_id] = effective[:, np.newaxis]
    return sigma


class SARMonitor:
    """Sparse on-the-fly electric-field DFT over selected tagged cells."""

    schema_version = 1

    def __init__(
        self,
        grid: "FDTDGrid",
        *,
        output_id: str,
        frequencies,
        tags,
        waveform_id: str,
        target_amplitude: float,
        spectrum_limit=DEFAULT_MINIMUM_WAVELENGTH_CELLS,
        source_floor_db: float = DEFAULT_INCIDENT_FLOOR_DB,
        window: str = "rectangular",
        averaging_masses=(),
        normalisation: str = "waveform",
        port_id: str | None = None,
        target_power: float | None = None,
    ) -> None:
        if grid.geometry_tag_map is None or grid.geometry_tag_registry is None:
            raise ValueError("SAR requires at least one tagged volumetric geometry object")
        self.output_id = str(output_id)
        self.frequencies = np.asarray(frequencies, dtype=np.float64)
        if self.frequencies.ndim != 1 or self.frequencies.size == 0:
            raise ValueError("SAR frequencies must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(self.frequencies)) or np.any(self.frequencies <= 0):
            raise ValueError("SAR frequencies must be finite and greater than zero")
        if np.unique(self.frequencies).size != self.frequencies.size:
            raise ValueError("SAR frequencies must not contain duplicates")
        if np.any(np.diff(self.frequencies) <= 0):
            raise ValueError("SAR frequencies must be strictly increasing")
        validate_nyquist_frequencies(self.frequencies, grid.dt)
        self.spectrum_limit = validate_spectrum_limit(spectrum_limit)
        self.minimum_wavelength_cells = (
            DEFAULT_MINIMUM_WAVELENGTH_CELLS
            if self.spectrum_limit == "nyquist"
            else float(self.spectrum_limit)
        )
        self.spectrum_limit_mode = (
            "nyquist" if self.spectrum_limit == "nyquist" else "minimum_wavelength_cells"
        )
        self.waveform_id = str(waveform_id)
        if not any(waveform.ID == self.waveform_id for waveform in grid.waveforms):
            raise ValueError(f"SAR references unknown waveform {self.waveform_id!r}")
        self.target_amplitude = float(target_amplitude)
        if not np.isfinite(self.target_amplitude) or self.target_amplitude <= 0:
            raise ValueError("SAR target_amplitude must be finite and greater than zero")
        self.source_floor_db = float(source_floor_db)
        if not np.isfinite(self.source_floor_db) or self.source_floor_db >= 0:
            raise ValueError("SAR source_floor_db must be finite and less than zero")
        self.window_name = str(window).lower().replace("-", "_")
        self.window = _window(self.window_name, grid.iterations, np.dtype(np.float64))
        masses = np.asarray(averaging_masses, dtype=np.float64)
        if masses.ndim != 1 or not np.all(np.isfinite(masses)) or np.any(masses <= 0):
            raise ValueError("SAR averaging_masses must contain finite positive masses in kg")
        if np.unique(masses).size != masses.size:
            raise ValueError("SAR averaging_masses must not contain duplicates")
        self.averaging_masses = tuple(float(value) for value in masses)
        self.normalisation = str(normalisation).lower()
        if self.normalisation not in ("waveform", "incident_power", "accepted_power"):
            raise ValueError(
                "SAR normalisation must be 'waveform', 'incident_power', or 'accepted_power'"
            )
        self.port_id = None if port_id is None else str(port_id)
        self.target_power = None if target_power is None else float(target_power)
        if self.normalisation == "waveform":
            if self.port_id is not None or self.target_power is not None:
                raise ValueError(
                    "SAR waveform normalisation does not accept port_id or target_power"
                )
        else:
            if not self.port_id:
                raise ValueError("power-normalised SAR requires port_id")
            if (
                self.target_power is None
                or not np.isfinite(self.target_power)
                or self.target_power <= 0
            ):
                raise ValueError("power-normalised SAR requires finite positive target_power")

        names = tuple(str(tag) for tag in tags)
        if not names or len(set(names)) != len(names):
            raise ValueError("SAR tags must be a non-empty collection without duplicates")
        unknown = [name for name in names if name not in grid.geometry_tag_registry.names]
        if unknown:
            raise ValueError(f"SAR references unknown geometry tag(s): {unknown}")
        self.tag_names = names
        self.tag_ids = np.asarray(
            [grid.geometry_tag_registry.id_for(name) for name in names], dtype=np.uint32
        )
        tag_data = grid.geometry_tag_map.data
        tagged_mask = np.isin(tag_data, self.tag_ids)
        pml_mask = _pml_cell_mask(grid)
        self.excluded_pml_cell_count = int(np.count_nonzero(tagged_mask & pml_mask))
        mask = tagged_mask & ~pml_mask
        self.cells = np.asarray(np.argwhere(mask), dtype=np.int32)
        if self.cells.size == 0:
            raise ValueError(
                "SAR selected geometry tags contain no physical cells outside PML regions"
            )
        self.cell_tag_ids = np.asarray(tag_data[mask], dtype=tag_data.dtype)
        self.cell_material_ids = np.asarray(grid.solid[mask], dtype=np.uint32)

        material_by_id = {int(material.numID): material for material in grid.materials}
        missing_density = sorted(
            {
                material_by_id[int(material_id)].ID
                for material_id in np.unique(self.cell_material_ids)
                if material_by_id[int(material_id)].mass_density is None
            }
        )
        if missing_density:
            raise ValueError(
                "SAR requires finite positive mass density for selected material(s): "
                + ", ".join(missing_density)
            )
        self.density = np.asarray(
            [material_by_id[int(item)].mass_density for item in self.cell_material_ids],
            dtype=np.float64,
        )

        self.edge_flat_indices = {}
        self.cell_edge_indices = {}
        for component, offsets in EDGE_OFFSETS.items():
            expanded = (self.cells[:, np.newaxis, :] + offsets[np.newaxis, :, :]).reshape(-1, 3)
            unique, inverse = np.unique(expanded, axis=0, return_inverse=True)
            self.edge_flat_indices[component] = np.asarray(
                np.ravel_multi_index(unique.T, (grid.nx + 1, grid.ny + 1, grid.nz + 1)),
                dtype=np.int64,
            )
            self.cell_edge_indices[component] = inverse.reshape(self.cells.shape[0], 4)
        self.cell_material_loss = _material_loss_conductivity(
            grid, self.cell_material_ids, self.frequencies
        )

        cells_per_wavelength, limiting_material = minimum_wavelength_sampling(
            grid, self.frequencies
        )
        self.cells_per_wavelength = np.asarray(cells_per_wavelength, dtype=np.float64)
        self.limiting_material = np.asarray(limiting_material, dtype=str)
        self.mesh_valid = self.cells_per_wavelength >= self.minimum_wavelength_cells
        if self.spectrum_limit != "nyquist" and not np.all(self.mesh_valid):
            first = int(np.flatnonzero(~self.mesh_valid)[0])
            raise ValueError(
                f"SAR frequency {self.frequencies[first]:g} Hz has only "
                f"{self.cells_per_wavelength[first]:g} cells per shortest wavelength in "
                f"material {self.limiting_material[first]!r}; requires at least "
                f"lambda/{self.minimum_wavelength_cells:g}. Use spectrum_limit='nyquist' "
                "only for explicit research output beyond the mesh-valid range."
            )

        self.real_dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
        self.complex_dtype = np.dtype(config.sim_config.dtypes["complex"])
        self.accumulators = {
            component: np.zeros((self.frequencies.size, indices.size), dtype=self.complex_dtype)
            for component, indices in self.edge_flat_indices.items()
        }
        self._phase_step = np.exp(-2j * np.pi * self.frequencies * grid.dt).astype(
            self.complex_dtype
        )
        self._phase = np.ones(self.frequencies.size, dtype=self.complex_dtype)
        self._next_iteration = 0
        self.result = None
        self.nthreads = int(config.get_model_config().ompthreads)
        self.collection_backend = (
            "cython_openmp" if _accumulate_sparse_dft is not None else "numpy_fallback"
        )
        self.grid_dt = float(grid.dt)
        self.grid_iterations = int(grid.iterations)
        self.grid_shape = (int(grid.nx), int(grid.ny), int(grid.nz))
        self.grid_spacing = (float(grid.dx), float(grid.dy), float(grid.dz))
        self.cell_volume = float(grid.dx * grid.dy * grid.dz)
        self.grid = grid
        self._source_samples = np.empty(0, dtype=self.real_dtype)
        self._spatial_average_plans = None
        self.prepare_run(grid)

    def _build_source_samples(self, grid: "FDTDGrid") -> npt.NDArray[np.floating]:
        source_groups = (
            grid.voltagesources,
            grid.hertziandipoles,
            grid.magneticdipoles,
            grid.transmissionlines,
            grid.magneticfrillsources,
            grid.discreteplanewaves,
        )
        sources = [
            source
            for group in source_groups
            for source in group
            if source.waveformID == self.waveform_id
        ]
        if not sources:
            raise ValueError(
                f"SAR waveform {self.waveform_id!r} is not associated with an active source"
            )
        active_sources = [source for source in sources if getattr(source, "study_scale", 1.0) != 0]
        if len(active_sources) != 1:
            raise ValueError(
                f"SAR waveform {self.waveform_id!r} must identify exactly one active "
                f"source; found {len(active_sources)}"
            )
        source = active_sources[0]
        windows = {(float(source.start), float(source.stop))}
        if len(windows) != 1:
            raise ValueError(
                "SAR sources sharing the normalising waveform must use the same start and stop"
            )
        start, stop = next(iter(windows))
        waveform = next(item for item in grid.waveforms if item.ID == self.waveform_id)
        source_scale = float(getattr(source, "study_scale", 1.0))
        samples = np.zeros(grid.iterations, dtype=self.real_dtype)
        for iteration in range(grid.iterations):
            time = iteration * grid.dt
            if start <= time < stop:
                samples[iteration] = source_scale * waveform.calculate_value(time - start, grid.dt)
        return samples

    def prepare_run(self, grid: "FDTDGrid") -> None:
        """Refresh source normalisation after geometry-fixed study changes."""

        self._source_samples = self._build_source_samples(grid)

    @property
    def nbytes(self) -> int:
        return int(
            sum(array.nbytes for array in self.accumulators.values())
            + sum(array.nbytes for array in self.edge_flat_indices.values())
            + sum(array.nbytes for array in self.cell_edge_indices.values())
            + self.cell_material_loss.nbytes
            + self.cells.nbytes
            + self.cell_tag_ids.nbytes
            + self.cell_material_ids.nbytes
            + self.density.nbytes
            + self.window.nbytes
            + self._source_samples.nbytes
            + sum(plan.nbytes for plan in (self._spatial_average_plans or ()))
        )

    def reset_run_state(self) -> None:
        for accumulator in self.accumulators.values():
            accumulator.fill(0)
        self._phase.fill(1)
        self._next_iteration = 0
        self.result = None

    def observe_electric(self, iteration: int, Ex, Ey, Ez) -> None:
        """Collect one CPU electric-field sample."""

        multiplier = self.device_sampling_multiplier(iteration)
        fields = {"Ex": Ex, "Ey": Ey, "Ez": Ez}
        for component, indices in self.edge_flat_indices.items():
            flat = np.ascontiguousarray(fields[component], dtype=self.real_dtype).ravel()
            if _accumulate_sparse_dft is not None:
                _accumulate_sparse_dft(
                    self.nthreads,
                    indices,
                    flat,
                    multiplier,
                    self.accumulators[component],
                )
            else:
                samples = np.take(flat, indices)
                self.accumulators[component] += multiplier[:, np.newaxis] * samples[np.newaxis, :]

    def device_sampling_multiplier(self, iteration: int):
        """Return the next DFT multiplier and advance shared sampling state."""

        if iteration != self._next_iteration:
            raise RuntimeError(
                f"SAR monitor expected iteration {self._next_iteration}, received {iteration}"
            )
        multiplier = np.asarray(
            self.grid_dt * self.window[iteration] * self._phase,
            dtype=self.complex_dtype,
        )
        self._next_iteration += 1
        self._phase *= self._phase_step
        if self._next_iteration % 1024 == 0:
            time = self._next_iteration * self.grid_dt
            self._phase = np.exp(-2j * np.pi * self.frequencies * time).astype(self.complex_dtype)
        return multiplier

    def load_device_component_dfts(self, component: str, values) -> None:
        """Load one component DFT downloaded from an accelerator."""

        if component not in self.accumulators:
            raise ValueError(f"unknown SAR electric-field component {component!r}")
        data = np.asarray(values, dtype=self.complex_dtype)
        if data.shape != self.accumulators[component].shape:
            raise ValueError(
                f"SAR {component} device DFT has shape {data.shape}, expected "
                f"{self.accumulators[component].shape}"
            )
        self.accumulators[component][...] = data

    def finalise(self) -> SARResult:
        if self._next_iteration != self.grid_iterations:
            raise RuntimeError("SAR monitor cannot be finalised before every timestep is observed")
        source_spectrum = np.asarray(
            engineering_dft(
                self._source_samples,
                self.frequencies,
                self.grid_dt,
                window=self.window,
            ),
            dtype=self.complex_dtype,
        )
        magnitude = np.abs(source_spectrum)
        peak = float(np.max(magnitude, initial=0.0))
        relative_db = np.full(self.frequencies.shape, -np.inf, dtype=self.real_dtype)
        if peak > 0:
            nonzero = magnitude > 0
            relative_db[nonzero] = np.asarray(
                20 * np.log10(magnitude[nonzero] / peak), dtype=self.real_dtype
            )
        source_valid = relative_db >= self.source_floor_db
        defined = magnitude > np.finfo(self.real_dtype).eps * peak
        valid = np.asarray(source_valid & defined & self.mesh_valid, dtype=bool)

        scale = np.full(self.frequencies.shape, np.nan + 1j * np.nan, dtype=self.complex_dtype)
        normalising_power = np.full(self.frequencies.shape, np.nan, dtype=self.real_dtype)
        if self.normalisation == "waveform":
            np.divide(
                self.target_amplitude,
                source_spectrum,
                out=scale,
                where=defined,
            )
        else:
            registry = port_output_registry(self.grid)
            if self.port_id not in registry:
                raise ValueError(f"SAR references unknown main-grid port {self.port_id!r}")
            spectrum = evaluate_port_power_spectrum(
                registry[self.port_id], self.grid, self.frequencies, window=self.window_name
            )
            normalising_power = np.asarray(
                spectrum.incident_power
                if self.normalisation == "incident_power"
                else spectrum.accepted_power,
                dtype=self.real_dtype,
            )
            power_peak = float(np.max(np.abs(normalising_power), initial=0.0))
            power_defined = (
                spectrum.terminal_valid
                & spectrum.mesh_valid
                & np.isfinite(normalising_power)
                & (normalising_power > np.finfo(self.real_dtype).eps * power_peak)
            )
            valid &= power_defined
            scale[power_defined] = np.sqrt(
                self.target_power / normalising_power[power_defined]
            ).astype(self.complex_dtype)
        absorbed = np.zeros((self.frequencies.size, self.cells.shape[0]), dtype=self.real_dtype)
        for component in EDGE_OFFSETS:
            normalised = self.accumulators[component] * scale[:, np.newaxis]
            edge_indices = self.cell_edge_indices[component]
            cell_field = np.mean(normalised[:, edge_indices], axis=2)
            absorbed += np.asarray(
                0.5 * self.cell_material_loss * np.abs(cell_field) ** 2,
                dtype=self.real_dtype,
            )
        sar = np.asarray(absorbed / self.density[np.newaxis, :], dtype=self.real_dtype)
        absorbed[~valid, :] = np.nan
        sar[~valid, :] = np.nan
        spatial_results = []
        if self.averaging_masses:
            density_volume = np.full(self.grid_shape, np.nan, dtype=np.float64)
            density_volume[tuple(self.cells.T)] = self.density
            if self._spatial_average_plans is None:
                self._spatial_average_plans = tuple(
                    build_spatial_average_plan(
                        density_volume,
                        self.grid_spacing,
                        target_mass,
                        nthreads=self.nthreads,
                    )
                    for target_mass in self.averaging_masses
                )
        for plan in getattr(self, "_spatial_average_plans", None) or ():
            averaged_sar = np.full(sar.shape, np.nan, dtype=self.real_dtype)
            status = np.zeros(sar.shape, dtype=np.uint8)
            averaging_mass = np.full(sar.shape, np.nan, dtype=self.real_dtype)
            averaging_volume = np.full(sar.shape, np.nan, dtype=self.real_dtype)
            orientation = np.zeros(sar.shape, dtype=np.int8)
            peak_sar = np.full(self.frequencies.shape, np.nan, dtype=self.real_dtype)
            peak_cell = np.full((self.frequencies.size, 3), -1, dtype=np.int32)
            for frequency_index in np.flatnonzero(valid):
                local_volume = np.zeros(self.grid_shape, dtype=np.float64)
                local_volume[tuple(self.cells.T)] = sar[frequency_index]
                averaged = apply_spatial_average_plan(plan, local_volume, density_volume)
                selection = tuple(self.cells.T)
                averaged_sar[frequency_index] = averaged.sar[selection]
                status[frequency_index] = averaged.status[selection]
                averaging_mass[frequency_index] = averaged.averaging_mass[selection]
                averaging_volume[frequency_index] = averaged.averaging_volume[selection]
                orientation[frequency_index] = averaged.orientation[selection]
                peak_sar[frequency_index] = averaged.peak_sar
                if averaged.peak_cell is not None:
                    peak_cell[frequency_index] = averaged.peak_cell
            spatial_results.append(
                SARSpatialAverageResult(
                    target_mass=plan.target_mass,
                    sar=averaged_sar,
                    status=status,
                    averaging_mass=averaging_mass,
                    averaging_volume=averaging_volume,
                    orientation=orientation,
                    peak_sar=peak_sar,
                    peak_cell=peak_cell,
                )
            )
        self.result = SARResult(
            frequency=np.asarray(self.frequencies, dtype=self.real_dtype),
            cell_indices=self.cells.copy(),
            tag_id=self.cell_tag_ids.copy(),
            material_id=self.cell_material_ids.copy(),
            density=np.asarray(self.density, dtype=self.real_dtype),
            source_spectrum=source_spectrum,
            source_relative_db=relative_db,
            source_valid=np.asarray(source_valid, dtype=bool),
            mesh_valid=np.asarray(self.mesh_valid, dtype=bool),
            valid=valid,
            cells_per_wavelength=np.asarray(self.cells_per_wavelength, dtype=self.real_dtype),
            limiting_material=self.limiting_material.copy(),
            absorbed_power_density=absorbed,
            sar=sar,
            spatial_averages=tuple(spatial_results),
            normalisation_scale=scale,
            normalising_power=normalising_power,
        )
        return self.result

    def write_hdf5(self, basegrp) -> None:
        if self.result is None:
            self.finalise()
        result = self.result
        group = basegrp.create_group(f"sar/{self.output_id}")
        group.attrs["SchemaVersion"] = self.schema_version
        group.attrs["Quantity"] = "frequency-domain specific absorption rate"
        group.attrs["Units"] = "W/kg"
        group.attrs["AbsorbedPowerDensityUnits"] = "W/m3"
        group.attrs["DensityUnits"] = "kg/m3"
        group.attrs["WaveformID"] = self.waveform_id
        group.attrs["TargetAmplitude"] = self.target_amplitude
        group.attrs["Normalisation"] = self.normalisation
        if self.normalisation == "waveform":
            group.attrs["NormalisationDefinition"] = "field_dft / waveform_dft * target_amplitude"
        else:
            group.attrs[
                "NormalisationDefinition"
            ] = f"field_dft * sqrt(target_power / {self.normalisation})"
            group.attrs["PortID"] = self.port_id
            group.attrs["TargetPower"] = self.target_power
            group.attrs["TargetPowerUnits"] = "W"
        group.attrs["PhasorAmplitude"] = "peak"
        group.attrs["PhasorTimeDependence"] = PHASOR_TIME_DEPENDENCE
        group.attrs["ForwardTransformKernel"] = FORWARD_TRANSFORM_KERNEL
        group.attrs["Window"] = self.window_name
        group.attrs["CollectionBackend"] = self.collection_backend
        group.attrs["PMLCellPolicy"] = "excluded"
        group.attrs["ExcludedPMLCellCount"] = self.excluded_pml_cell_count
        group.attrs["SourceFloorDB"] = self.source_floor_db
        group.attrs["SpectrumLimitMode"] = self.spectrum_limit_mode
        group.attrs["MinimumWavelengthCells"] = self.minimum_wavelength_cells
        group.attrs[
            "FieldCollocation"
        ] = "complex arithmetic mean of four parallel Yee edges per component"
        group.attrs["TagNames"] = np.asarray(self.tag_names, dtype="S")
        group["frequency"] = result.frequency
        group["cell_indices"] = result.cell_indices
        group["tag_id"] = result.tag_id
        group["material_id"] = result.material_id
        group["density"] = result.density
        group["source_spectrum"] = result.source_spectrum
        group["source_relative_db"] = result.source_relative_db
        group["source_valid"] = result.source_valid.astype(np.uint8)
        group["normalisation_scale"] = result.normalisation_scale
        group["normalising_power"] = result.normalising_power
        group["mesh_valid"] = result.mesh_valid.astype(np.uint8)
        group["valid"] = result.valid.astype(np.uint8)
        group["cells_per_wavelength"] = result.cells_per_wavelength
        string_dtype = h5py.string_dtype(encoding="utf-8")
        group.create_dataset(
            "limiting_material",
            data=np.asarray(result.limiting_material, dtype=object),
            dtype=string_dtype,
        )
        group["absorbed_power_density"] = result.absorbed_power_density
        group["sar"] = result.sar

        if result.spatial_averages:
            spatial_group = group.create_group("spatial_average")
            spatial_group.attrs["Algorithm"] = "IEC/IEEE 62704-1 two-step cubical averaging"
            spatial_group.attrs["DensityModel"] = "constant within each tagged FDTD cell"
            spatial_group.attrs["BackgroundPolicy"] = "cells outside selected tags are background"
            for averaged in result.spatial_averages:
                mass_grams = 1000 * averaged.target_mass
                label = f"{mass_grams:g}g"
                averaged_group = spatial_group.create_group(label)
                averaged_group.attrs["TargetMass"] = averaged.target_mass
                averaged_group.attrs["TargetMassUnits"] = "kg"
                averaged_group["sar"] = averaged.sar
                averaged_group["status"] = averaged.status
                averaged_group["averaging_mass"] = averaged.averaging_mass
                averaged_group["averaging_volume"] = averaged.averaging_volume
                averaged_group["orientation"] = averaged.orientation
                averaged_group["peak_sar"] = averaged.peak_sar
                averaged_group["peak_cell"] = averaged.peak_cell

        summaries = group.create_group("tags")
        for name, tag_id in zip(self.tag_names, self.tag_ids):
            selection = result.tag_id == tag_id
            mass = float(np.sum(result.density[selection]) * self.cell_volume)
            absorbed_power = np.asarray(
                np.nansum(result.absorbed_power_density[:, selection], axis=1) * self.cell_volume,
                dtype=self.real_dtype,
            )
            tag_group = summaries.create_group(name)
            tag_group.attrs["TagID"] = int(tag_id)
            tag_group.attrs["CellCount"] = int(np.count_nonzero(selection))
            tag_group.attrs["Mass"] = mass
            tag_group.attrs["MassUnits"] = "kg"
            tag_group["absorbed_power"] = absorbed_power
            tag_group["mass_average_sar"] = absorbed_power / mass
            peak_sar = np.full(result.frequency.shape, np.nan, dtype=self.real_dtype)
            for frequency_index in np.flatnonzero(result.valid):
                peak_sar[frequency_index] = np.max(
                    result.sar[frequency_index, selection], initial=0.0
                )
            tag_group["peak_voxel_sar"] = peak_sar


def compile_sar_outputs(grid: "FDTDGrid") -> None:
    """Compile deferred SAR requests after material IDs are available."""

    if grid.sar_monitors:
        return
    for spec in grid.sar_specs:
        monitor = SARMonitor(
            grid,
            output_id=spec.output_id,
            frequencies=spec.frequencies,
            tags=spec.tags,
            waveform_id=spec.waveform_id,
            target_amplitude=spec.target_amplitude,
            spectrum_limit=spec.spectrum_limit,
            source_floor_db=spec.source_floor_db,
            window=spec.window,
            averaging_masses=spec.averaging_masses,
            normalisation=spec.normalisation,
            port_id=spec.port_id,
            target_power=spec.target_power,
        )
        grid.sar_monitors.append(monitor)
        if spec.owner is not None:
            spec.owner._monitor = monitor
