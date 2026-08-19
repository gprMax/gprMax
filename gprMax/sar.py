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
from gprMax.mode2d import mode2d_geometry

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
    model_port_output_registry,
    port_output_registry,
    validate_spectrum_limit,
)
from gprMax.sar_averaging import apply_spatial_average_plan, build_spatial_average_plan

if TYPE_CHECKING:
    from gprMax.grid.fdtd_grid import FDTDGrid
    from gprMax.model import Model


logger = logging.getLogger(__name__)


SOURCE_NORMALISATION_METADATA = {
    "VoltageSource": ("generator_voltage", "V"),
    "HertzianDipole": ("electric_current", "A"),
    "MagneticDipole": ("magnetic_current_moment", "V m"),
    "TransmissionLine": ("generator_voltage", "V"),
    "MagneticFrillSource": ("generator_voltage", "V"),
    "RationalNetworkTerminal": ("generator_voltage", "V"),
    "DiscretePlaneWave": ("incident_electric_field", "V/m"),
    "EigenmodeSource": ("modal_excitation_coefficient", "source-defined"),
}


EDGE_OFFSETS = {
    "Ex": np.asarray(((0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1)), dtype=np.int32),
    "Ey": np.asarray(((0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)), dtype=np.int32),
    "Ez": np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)), dtype=np.int32),
}


def edge_offsets_for_mode(mode: str) -> dict[str, npt.NDArray[np.int32]]:
    """Return active electric components and their cell-centre offsets.

    The three-dimensional four-edge formula is retained.  In TE mode the
    invariant-axis offset is collapsed to zero, so the four entries contain
    two duplicated pairs and their mean is exactly the arithmetic mean of the
    two genuine tangential edges on the live plane.  TM's only electric
    component is parallel to the invariant axis and its existing four offsets
    already lie entirely in the transverse cross-section.
    """

    geometry = mode2d_geometry(mode)
    if geometry is None:
        return {component: offsets.copy() for component, offsets in EDGE_OFFSETS.items()}
    selected = {}
    for component in geometry.active_electric:
        offsets = EDGE_OFFSETS[component].copy()
        if geometry.polarisation == "TE":
            offsets[:, geometry.invariant_axis] = 0
        selected[component] = offsets
    return selected


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
        lower = np.asarray((spec.xs, spec.ys, spec.zs), dtype=np.int64)
        upper = np.asarray((spec.xf, spec.yf, spec.zf), dtype=np.int64)
        if hasattr(grid, "lower_extent"):
            lower -= np.asarray(grid.lower_extent, dtype=np.int64)
            upper -= np.asarray(grid.lower_extent, dtype=np.int64)
        lower = np.maximum(lower, 0)
        upper = np.minimum(upper, np.asarray(mask.shape, dtype=np.int64))
        if np.all(lower < upper):
            mask[
                lower[0] : upper[0],
                lower[1] : upper[1],
                lower[2] : upper[2],
            ] = True
    return mask


def _mpi_owned_edge_mask(grid, global_coordinates) -> npt.NDArray[np.bool_]:
    """Return which global Yee-edge coordinates are owned by one MPI rank."""

    global_coordinates = np.asarray(global_coordinates, dtype=np.int32)
    owned_lower = np.asarray(
        grid.lower_extent + grid.negative_halo_offset.astype(np.int32), dtype=np.int32
    )
    owned_upper = np.asarray(grid.upper_extent, dtype=np.int32)
    global_shape = np.asarray(grid.global_size, dtype=np.int32)
    owned = np.all(global_coordinates >= owned_lower, axis=1)
    for dimension in range(3):
        if owned_upper[dimension] == global_shape[dimension]:
            owned &= global_coordinates[:, dimension] <= owned_upper[dimension]
        else:
            owned &= global_coordinates[:, dimension] < owned_upper[dimension]
    return owned


@dataclass(frozen=True)
class SARSpec:
    """Deferred SAR request compiled after Yee material IDs are finalised."""

    output_id: str
    frequencies: tuple[float, ...]
    tags: tuple[str, ...]
    waveform_id: str | None
    target_amplitude: float
    spectrum_limit: float | str
    source_floor_db: float
    window: str
    averaging_masses: tuple[float, ...]
    normalisation: str
    port_id: str | None
    target_power: float | None
    target_flux: float | None
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
    incident_power: npt.NDArray[np.floating]
    incident_flux: npt.NDArray[np.floating]
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


@dataclass(frozen=True)
class SARLocalPayload:
    """Rank-local SAR data independent of source or port normalisation."""

    cell_indices: npt.NDArray[np.integer]
    tag_id: npt.NDArray[np.integer]
    material_id: npt.NDArray[np.integer]
    density: npt.NDArray[np.floating]
    absorbed_power_density: npt.NDArray[np.floating]
    excluded_pml_cell_count: int
    edge_coordinates: dict[str, npt.NDArray[np.integer]] | None = None
    edge_dft: dict[str, npt.NDArray[np.complexfloating]] | None = None


@dataclass(frozen=True)
class RadiometrySpec:
    """Deferred radiometric absorption request compiled with SAR outputs."""

    output_id: str
    frequencies: tuple[float, ...]
    tags: tuple[str, ...]
    waveform_id: str | None
    target_amplitude: float
    spectrum_limit: float | str
    source_floor_db: float
    window: str
    normalisation: str
    port_id: str | None
    target_power: float | None
    target_flux: float | None
    owner: Any = None


@dataclass(frozen=True)
class RadiometryResult:
    """Frequency-domain absorbed-power and radiometric weighting data."""

    frequency: npt.NDArray[np.floating]
    cell_indices: npt.NDArray[np.integer]
    tag_id: npt.NDArray[np.integer]
    material_id: npt.NDArray[np.integer]
    source_spectrum: npt.NDArray[np.complexfloating]
    source_relative_db: npt.NDArray[np.floating]
    source_valid: npt.NDArray[np.bool_]
    incident_power: npt.NDArray[np.floating]
    incident_flux: npt.NDArray[np.floating]
    mesh_valid: npt.NDArray[np.bool_]
    valid: npt.NDArray[np.bool_]
    cells_per_wavelength: npt.NDArray[np.floating]
    limiting_material: npt.NDArray[np.str_]
    absorbed_power_density: npt.NDArray[np.floating]
    normalised_absorption_density: npt.NDArray[np.floating]
    normalisation_scale: npt.NDArray[np.complexfloating]
    normalising_power: npt.NDArray[np.floating]


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
        waveform_id: str | None,
        target_amplitude: float,
        spectrum_limit=DEFAULT_MINIMUM_WAVELENGTH_CELLS,
        source_floor_db: float = DEFAULT_INCIDENT_FLOOR_DB,
        window: str = "rectangular",
        averaging_masses=(),
        normalisation: str = "waveform",
        port_id: str | None = None,
        target_power: float | None = None,
        target_flux: float | None = None,
        model: "Model | None" = None,
        require_density: bool = True,
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
        self.waveform_id = None if waveform_id is None else str(waveform_id)
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
        if self.normalisation not in (
            "waveform",
            "current_moment",
            "incident_flux",
            "incident_power",
            "accepted_power",
        ):
            raise ValueError(
                "SAR normalisation must be 'waveform', 'current_moment', "
                "'incident_flux', 'incident_power', or 'accepted_power'"
            )
        self.port_id = None if port_id is None else str(port_id)
        self.target_power = None if target_power is None else float(target_power)
        self.target_flux = None if target_flux is None else float(target_flux)
        if self.normalisation in ("waveform", "current_moment", "incident_flux"):
            if self.port_id is not None or self.target_power is not None:
                raise ValueError("SAR source normalisation does not accept port_id or target_power")
            if not self.waveform_id:
                raise ValueError(f"SAR {self.normalisation} normalisation requires waveform_id")
            waveform_grids = (grid,) if model is None else (model.G, *model.subgrids)
            if not any(
                waveform.ID == self.waveform_id
                for waveform_grid in waveform_grids
                for waveform in waveform_grid.waveforms
            ):
                raise ValueError(f"SAR references unknown waveform {self.waveform_id!r}")
            if (
                self.normalisation == "current_moment"
                and mode2d_geometry(config.get_model_config().mode) is not None
            ):
                raise ValueError("SAR current_moment normalisation is available only in 3-D")
            if self.normalisation == "incident_flux":
                if (
                    self.target_flux is None
                    or not np.isfinite(self.target_flux)
                    or self.target_flux <= 0
                ):
                    raise ValueError(
                        "SAR incident_flux normalisation requires finite positive target_flux"
                    )
            elif self.target_flux is not None:
                raise ValueError(
                    "SAR waveform/current_moment normalisation does not use target_flux"
                )
        else:
            if self.target_flux is not None:
                raise ValueError("power-normalised SAR does not use target_flux")
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
        self.model_mode = config.get_model_config().mode
        self.mode2d = mode2d_geometry(self.model_mode)
        self.edge_offsets = edge_offsets_for_mode(self.model_mode)
        tag_data = grid.geometry_tag_map.data
        tagged_mask = np.isin(tag_data, self.tag_ids)
        if self.mode2d is not None:
            live_plane = np.zeros(tagged_mask.shape, dtype=bool)
            live_slice = [slice(None)] * 3
            live_slice[self.mode2d.invariant_axis] = self.mode2d.live_index
            live_plane[tuple(live_slice)] = True
            tagged_mask &= live_plane
        owned_mask = np.ones(tag_data.shape, dtype=bool)
        self._mpi_distributed = hasattr(grid, "comm") and hasattr(grid, "lower_extent")
        if self._mpi_distributed:
            halo = np.asarray(grid.negative_halo_offset, dtype=np.int32)
            if halo[0]:
                owned_mask[: halo[0], :, :] = False
            if halo[1]:
                owned_mask[:, : halo[1], :] = False
            if halo[2]:
                owned_mask[:, :, : halo[2]] = False
        pml_mask = _pml_cell_mask(grid)
        self.excluded_pml_cell_count = int(np.count_nonzero(tagged_mask & pml_mask & owned_mask))
        sampling_mask = tagged_mask & ~pml_mask
        mask = sampling_mask & owned_mask
        self.cells = np.asarray(np.argwhere(mask), dtype=np.int32)
        self.sampling_cells = (
            np.asarray(np.argwhere(sampling_mask), dtype=np.int32)
            if self._mpi_distributed
            else self.cells
        )
        local_cell_count = int(self.cells.shape[0])
        global_cell_count = (
            int(grid.comm.allreduce(local_cell_count))
            if self._mpi_distributed
            else local_cell_count
        )
        if global_cell_count == 0:
            raise ValueError(
                "SAR selected geometry tags contain no physical cells outside PML regions"
            )
        self.cell_tag_ids = np.asarray(tag_data[mask], dtype=tag_data.dtype)
        self.cell_material_ids = np.asarray(grid.solid[mask], dtype=np.uint32)

        material_by_id = {int(material.numID): material for material in grid.materials}
        self.require_density = bool(require_density)
        missing_density = (
            sorted(
                {
                    material_by_id[int(material_id)].ID
                    for material_id in np.unique(self.cell_material_ids)
                    if material_by_id[int(material_id)].mass_density is None
                }
            )
            if self.require_density
            else []
        )
        if self._mpi_distributed:
            missing_density = sorted(
                {
                    name
                    for rank_names in grid.comm.allgather(tuple(missing_density))
                    for name in rank_names
                }
            )
        if missing_density:
            raise ValueError(
                "SAR requires finite positive mass density for selected material(s): "
                + ", ".join(missing_density)
            )
        if self.require_density:
            self.density = np.asarray(
                [material_by_id[int(item)].mass_density for item in self.cell_material_ids],
                dtype=np.float64,
            )
        else:
            self.density = np.empty(0, dtype=np.float64)

        self.edge_flat_indices = {}
        self.cell_edge_indices = {}
        self.edge_coordinates = {}
        for component, offsets in self.edge_offsets.items():
            expanded = (self.sampling_cells[:, np.newaxis, :] + offsets[np.newaxis, :, :]).reshape(
                -1, 3
            )
            unique, inverse = np.unique(expanded, axis=0, return_inverse=True)
            self.edge_coordinates[component] = np.asarray(unique, dtype=np.int32)
            self.edge_flat_indices[component] = np.asarray(
                np.ravel_multi_index(unique.T, (grid.nx + 1, grid.ny + 1, grid.nz + 1)),
                dtype=np.int64,
            )
            self.cell_edge_indices[component] = inverse.reshape(self.sampling_cells.shape[0], 4)
        self.cell_material_loss = _material_loss_conductivity(
            grid, self.cell_material_ids, self.frequencies
        )

        cells_per_wavelength, limiting_material = minimum_wavelength_sampling(
            grid, self.frequencies
        )
        self.cells_per_wavelength = np.asarray(cells_per_wavelength, dtype=np.float64)
        self.limiting_material = np.asarray(limiting_material, dtype=str)
        if self._mpi_distributed:
            rank_sampling = grid.comm.allgather((self.cells_per_wavelength, self.limiting_material))
            values = np.stack([item[0] for item in rank_sampling], axis=0)
            limiting_ranks = np.argmin(values, axis=0)
            self.cells_per_wavelength = values[limiting_ranks, np.arange(self.frequencies.size)]
            self.limiting_material = np.asarray(
                [
                    rank_sampling[int(rank)][1][frequency_index]
                    for frequency_index, rank in enumerate(limiting_ranks)
                ],
                dtype=str,
            )
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
        self.cell_measure = (
            self.cell_volume
            if self.mode2d is None
            else float(
                np.prod(
                    [
                        self.grid_spacing[axis]
                        for axis in range(3)
                        if axis != self.mode2d.invariant_axis
                    ]
                )
            )
        )
        if self._mpi_distributed:
            self.cell_index_frame = "main-grid"
            self.cell_index_origin = np.zeros(3, dtype=np.float64)
        elif hasattr(grid, "local_to_global"):
            self.cell_index_frame = "subgrid-local"
            self.cell_index_origin = np.asarray(grid.local_to_global((0, 0, 0)), dtype=np.float64)
        else:
            self.cell_index_frame = "main-grid"
            self.cell_index_origin = np.zeros(3, dtype=np.float64)
        self.grid = grid
        self.model = model
        self._source_samples = np.empty(0, dtype=self.real_dtype)
        self._source_dt = float(grid.dt)
        self._source_window = np.empty(0, dtype=self.real_dtype)
        self._source_type = None
        self._source_quantity = None
        self._source_units = None
        self._source_length = None
        self._source = None
        self._source_grid = None
        self._spatial_average_plans = None

    def _model_grids(self):
        if self.model is None:
            return (self.grid,)
        return (self.model.G, *self.model.subgrids)

    def _build_source_samples(self):
        bindings = []
        for source_grid in self._model_grids():
            source_groups = (
                source_grid.voltagesources,
                source_grid.hertziandipoles,
                source_grid.magneticdipoles,
                source_grid.transmissionlines,
                source_grid.magneticfrillsources,
                source_grid.networkterminals,
                source_grid.discreteplanewaves,
                source_grid.eigenmodesources,
            )
            bindings.extend(
                (source, source_grid)
                for group in source_groups
                for source in group
                if source.waveformID == self.waveform_id
            )
            bindings.extend(
                (guide.aux_source, source_grid)
                for guide in getattr(source_grid, "virtual_waveguides", ())
                if getattr(guide, "aux_source", None) is not None
                and guide.aux_source.waveformID == self.waveform_id
            )
        if not bindings:
            raise ValueError(
                f"SAR waveform {self.waveform_id!r} is not associated with an active source"
            )
        active_bindings = [
            binding for binding in bindings if getattr(binding[0], "study_scale", 1.0) != 0
        ]
        if len(active_bindings) != 1:
            raise ValueError(
                f"SAR waveform {self.waveform_id!r} must identify exactly one active "
                f"source across the model; found {len(active_bindings)}"
            )
        source, source_grid = active_bindings[0]
        self._source = source
        self._source_grid = source_grid
        if self.normalisation == "current_moment":
            if source not in source_grid.hertziandipoles:
                raise ValueError(
                    "SAR current_moment normalisation requires exactly one active "
                    "3-D Hertzian dipole using waveform_id"
                )
            source_length = float(source.dl)
            if not np.isfinite(source_length) or source_length <= 0:
                raise ValueError("SAR Hertzian-dipole source length must be finite and positive")
            self._source_length = source_length
        elif self.normalisation == "incident_flux":
            if type(source).__name__ != "DiscretePlaneWave":
                raise ValueError(
                    "SAR incident_flux normalisation requires exactly one active "
                    "discrete plane wave using waveform_id"
                )
            self._source_length = None
        else:
            self._source_length = None
        self._source_type = type(source).__name__
        self._source_quantity, self._source_units = SOURCE_NORMALISATION_METADATA.get(
            self._source_type, ("source_waveform_amplitude", "source-defined")
        )
        start, stop = float(source.start), float(source.stop)
        waveforms = [item for item in source_grid.waveforms if item.ID == self.waveform_id]
        if len(waveforms) != 1:
            raise ValueError(
                f"SAR waveform {self.waveform_id!r} is not uniquely defined on its source grid"
            )
        waveform = waveforms[0]
        source_scale = float(getattr(source, "study_scale", 1.0))

        # Prefer the exact sequence used by the source update. This preserves
        # the source's own whole/half-step convention and any Study scaling,
        # rather than reconstructing a nominal waveform on a different time
        # lattice. Plane waves have position-dependent 3-D waveform tables,
        # so their incident reference remains the undelayed scalar waveform.
        source_type = self._source_type
        prepared = None
        if source_type == "VoltageSource":
            prepared = (
                source.waveformvalues_wholedt
                if float(source.resistance) == 0
                else source.waveformvalues_halfdt
            )
        elif source_type == "HertzianDipole":
            prepared = source.waveformvalues_halfdt
        elif source_type in ("MagneticDipole", "TransmissionLine", "MagneticFrillSource"):
            prepared = source.waveformvalues_wholedt
        elif source_type == "RationalNetworkTerminal":
            prepared = getattr(source, "waveform_half", None)
        elif source_type == "EigenmodeSource":
            prepared = getattr(source, "broadband_input_waveform", None)
        if prepared is not None:
            prepared = np.asarray(prepared)
            if prepared.ndim == 1 and prepared.size >= source_grid.iterations:
                return np.asarray(
                    prepared[: source_grid.iterations], dtype=self.real_dtype
                ).copy(), float(source_grid.dt)

        samples = np.zeros(source_grid.iterations, dtype=self.real_dtype)
        half_step = source_type == "RationalNetworkTerminal"
        for iteration in range(source_grid.iterations):
            time = (iteration + 0.5 * half_step) * source_grid.dt
            if start <= time < stop:
                samples[iteration] = source_scale * waveform.calculate_value(
                    time - start, source_grid.dt
                )
        return samples, float(source_grid.dt)

    def prepare_run(self) -> None:
        """Refresh source normalisation after geometry-fixed study changes."""

        # MPI sources are partitioned between ranks. Their complete state is
        # available only on the coordinator inside gathered_output_state().
        if self.normalisation in ("incident_power", "accepted_power"):
            self._source_samples = np.empty(0, dtype=self.real_dtype)
            self._source_window = np.empty(0, dtype=self.real_dtype)
            return
        if self._mpi_distributed:
            self._source_samples = np.empty(0, dtype=self.real_dtype)
            self._source_window = np.empty(0, dtype=self.real_dtype)
            return
        self._source_samples, self._source_dt = self._build_source_samples()
        self._source_window = _window(
            self.window_name, self._source_samples.size, np.dtype(np.float64)
        )

    @property
    def nbytes(self) -> int:
        return int(
            sum(array.nbytes for array in self.accumulators.values())
            + sum(array.nbytes for array in self.edge_flat_indices.values())
            + sum(array.nbytes for array in self.cell_edge_indices.values())
            + sum(array.nbytes for array in self.edge_coordinates.values())
            + self.cell_material_loss.nbytes
            + self.cells.nbytes
            + (0 if self.sampling_cells is self.cells else self.sampling_cells.nbytes)
            + self.cell_tag_ids.nbytes
            + self.cell_material_ids.nbytes
            + self.density.nbytes
            + self.window.nbytes
            + self._source_samples.nbytes
            + self._source_window.nbytes
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

    def local_payload(self) -> SARLocalPayload:
        """Return owned-cell absorbed power before source normalisation."""

        if self._next_iteration != self.grid_iterations:
            raise RuntimeError("SAR monitor cannot be finalised before every timestep is observed")
        absorbed = np.zeros((self.frequencies.size, self.cells.shape[0]), dtype=self.real_dtype)
        edge_coordinates = None
        edge_dft = None
        if getattr(self, "_mpi_distributed", False):
            edge_coordinates = {}
            edge_dft = {}
            for component, local_coordinates in self.edge_coordinates.items():
                global_coordinates = local_coordinates + np.asarray(
                    self.grid.lower_extent, dtype=np.int32
                )
                owned = _mpi_owned_edge_mask(self.grid, global_coordinates)
                edge_coordinates[component] = global_coordinates[owned]
                edge_dft[component] = self.accumulators[component][:, owned].copy()
        else:
            for component in getattr(self, "edge_offsets", EDGE_OFFSETS):
                edge_indices = self.cell_edge_indices[component]
                cell_field = np.mean(self.accumulators[component][:, edge_indices], axis=2)
                absorbed += np.asarray(
                    0.5 * self.cell_material_loss * np.abs(cell_field) ** 2,
                    dtype=self.real_dtype,
                )

        cells = self.cells.copy()
        if getattr(self, "_mpi_distributed", False):
            cells += np.asarray(self.grid.lower_extent, dtype=np.int32)
        return SARLocalPayload(
            cell_indices=cells,
            tag_id=self.cell_tag_ids.copy(),
            material_id=self.cell_material_ids.copy(),
            density=np.asarray(self.density, dtype=self.real_dtype),
            absorbed_power_density=absorbed,
            excluded_pml_cell_count=getattr(self, "excluded_pml_cell_count", 0),
            edge_coordinates=edge_coordinates,
            edge_dft=edge_dft,
        )

    @staticmethod
    def merge_local_payloads(payloads: list[SARLocalPayload]) -> SARLocalPayload:
        """Merge rank payloads into deterministic global-cell order."""

        if not payloads:
            raise ValueError("cannot merge an empty collection of MPI SAR payloads")
        cell_indices = np.concatenate([payload.cell_indices for payload in payloads], axis=0)
        tag_id = np.concatenate([payload.tag_id for payload in payloads])
        material_id = np.concatenate([payload.material_id for payload in payloads])
        density = np.concatenate([payload.density for payload in payloads])
        absorbed = np.concatenate([payload.absorbed_power_density for payload in payloads], axis=1)
        if cell_indices.shape[0] == 0:
            raise ValueError("MPI SAR selected no physical cells outside PML regions")
        order = np.lexsort((cell_indices[:, 2], cell_indices[:, 1], cell_indices[:, 0]))
        cell_indices = cell_indices[order]
        if cell_indices.shape[0] > 1 and np.any(
            np.all(cell_indices[1:] == cell_indices[:-1], axis=1)
        ):
            raise RuntimeError("MPI SAR aggregation found duplicate owned global cells")
        sorted_density = density[order] if density.size else density
        merged_edge_coordinates = None
        merged_edge_dft = None
        if any(payload.edge_coordinates is not None for payload in payloads):
            if any(
                payload.edge_coordinates is None or payload.edge_dft is None for payload in payloads
            ):
                raise RuntimeError("MPI SAR payloads contain inconsistent edge DFT data")
            merged_edge_coordinates = {}
            merged_edge_dft = {}
            for component in EDGE_OFFSETS:
                coordinates = np.concatenate(
                    [payload.edge_coordinates[component] for payload in payloads], axis=0
                )
                dft = np.concatenate([payload.edge_dft[component] for payload in payloads], axis=1)
                edge_order = np.lexsort((coordinates[:, 2], coordinates[:, 1], coordinates[:, 0]))
                coordinates = coordinates[edge_order]
                if coordinates.shape[0] > 1 and np.any(
                    np.all(coordinates[1:] == coordinates[:-1], axis=1)
                ):
                    raise RuntimeError(
                        f"MPI SAR aggregation found duplicate owned {component} edges"
                    )
                merged_edge_coordinates[component] = coordinates
                merged_edge_dft[component] = dft[:, edge_order]
        return SARLocalPayload(
            cell_indices=cell_indices,
            tag_id=tag_id[order],
            material_id=material_id[order],
            density=sorted_density,
            absorbed_power_density=absorbed[:, order],
            excluded_pml_cell_count=sum(payload.excluded_pml_cell_count for payload in payloads),
            edge_coordinates=merged_edge_coordinates,
            edge_dft=merged_edge_dft,
        )

    def _collocate_mpi_payload(self, payload: SARLocalPayload, global_shape) -> SARLocalPayload:
        """Collocate gathered global Yee-edge DFTs at gathered cell centres."""

        if payload.edge_coordinates is None or payload.edge_dft is None:
            raise RuntimeError("MPI SAR payload does not contain electric-edge DFTs")
        field_shape = tuple(int(value) + 1 for value in global_shape)
        material_loss = _material_loss_conductivity(
            self.grid, payload.material_id, self.frequencies
        )
        absorbed = np.zeros(
            (self.frequencies.size, payload.cell_indices.shape[0]), dtype=self.real_dtype
        )
        for component, offsets in getattr(self, "edge_offsets", EDGE_OFFSETS).items():
            coordinates = payload.edge_coordinates[component]
            available = np.ravel_multi_index(coordinates.T, field_shape)
            required_coordinates = (
                payload.cell_indices[:, np.newaxis, :] + offsets[np.newaxis, :, :]
            ).reshape(-1, 3)
            required = np.ravel_multi_index(required_coordinates.T, field_shape)
            positions = np.searchsorted(available, required)
            if np.any(positions >= available.size) or np.any(available[positions] != required):
                raise RuntimeError(
                    f"MPI SAR aggregation is missing required global {component} edges"
                )
            cell_field = np.mean(
                payload.edge_dft[component][:, positions].reshape(
                    self.frequencies.size, payload.cell_indices.shape[0], 4
                ),
                axis=2,
            )
            absorbed += np.asarray(
                0.5 * material_loss * np.abs(cell_field) ** 2,
                dtype=self.real_dtype,
            )
        return SARLocalPayload(
            cell_indices=payload.cell_indices,
            tag_id=payload.tag_id,
            material_id=payload.material_id,
            density=payload.density,
            absorbed_power_density=absorbed,
            excluded_pml_cell_count=payload.excluded_pml_cell_count,
        )

    def _normalisation_data(self):
        """Return source/port spectra, validity masks, and field scaling."""

        source_spectrum = np.full(
            self.frequencies.shape, np.nan + 1j * np.nan, dtype=self.complex_dtype
        )
        relative_db = np.full(self.frequencies.shape, -np.inf, dtype=self.real_dtype)
        source_valid = np.zeros(self.frequencies.shape, dtype=bool)
        valid = np.asarray(self.mesh_valid, dtype=bool).copy()
        scale = np.full(self.frequencies.shape, np.nan + 1j * np.nan, dtype=self.complex_dtype)
        normalising_power = np.full(self.frequencies.shape, np.nan, dtype=self.real_dtype)
        incident_power = np.full(self.frequencies.shape, np.nan, dtype=self.real_dtype)
        incident_flux = np.full(self.frequencies.shape, np.nan, dtype=self.real_dtype)
        if self.normalisation in ("waveform", "current_moment", "incident_flux"):
            source_dt = getattr(self, "_source_dt", self.grid_dt)
            source_window = getattr(self, "_source_window", self.window)
            source_spectrum = np.asarray(
                engineering_dft(
                    self._source_samples,
                    self.frequencies,
                    source_dt,
                    window=source_window,
                ),
                dtype=self.complex_dtype,
            )
            magnitude = np.abs(source_spectrum)
            peak = float(np.max(magnitude, initial=0.0))
            if peak > 0:
                nonzero = magnitude > 0
                relative_db[nonzero] = np.asarray(
                    20 * np.log10(magnitude[nonzero] / peak), dtype=self.real_dtype
                )
            source_valid = relative_db >= self.source_floor_db
            defined = magnitude > np.finfo(self.real_dtype).eps * peak
            valid &= source_valid & defined
            denominator = source_spectrum
            if self.normalisation == "current_moment":
                denominator = source_spectrum * self._source_length
            if self.normalisation == "incident_flux":
                material = self._source.material
                epsilon_r = _material_relative_permittivity(material, self.frequencies)
                omega = 2 * np.pi * self.frequencies
                mu_r = material.mr + material.sm / (1j * omega * config.sim_config.em_consts["m0"])
                impedance = np.sqrt(
                    config.sim_config.em_consts["m0"]
                    * mu_r
                    / (config.sim_config.em_consts["e0"] * epsilon_r)
                )
                wave_admittance = np.real(1 / impedance)
                incident_flux = np.asarray(
                    0.5 * magnitude**2 * wave_admittance, dtype=self.real_dtype
                )
                flux_peak = float(np.max(incident_flux, initial=0.0))
                flux_defined = (
                    defined
                    & np.isfinite(incident_flux)
                    & (wave_admittance > 0)
                    & (incident_flux > np.finfo(self.real_dtype).eps * flux_peak)
                )
                valid &= flux_defined
                scale[flux_defined] = np.sqrt(
                    self.target_flux / incident_flux[flux_defined]
                ).astype(self.complex_dtype)
            else:
                np.divide(
                    self.target_amplitude,
                    denominator,
                    out=scale,
                    where=defined,
                )
        else:
            if getattr(self, "model", None) is None:
                registry = port_output_registry(self.grid)
                if self.port_id not in registry:
                    raise ValueError(f"SAR references unknown model port {self.port_id!r}")
                output = registry[self.port_id]
                port_grid = self.grid
            else:
                registry = model_port_output_registry(self.model)
                if self.port_id not in registry:
                    raise ValueError(f"SAR references unknown model port {self.port_id!r}")
                binding = registry[self.port_id]
                output = binding.output
                port_grid = binding.grid
            spectrum = evaluate_port_power_spectrum(
                output, port_grid, self.frequencies, window=self.window_name
            )
            incident_power = np.asarray(spectrum.incident_power, dtype=self.real_dtype)
            normalising_power = np.asarray(
                incident_power
                if self.normalisation == "incident_power"
                else spectrum.accepted_power,
                dtype=self.real_dtype,
            )
            positive_incident = np.isfinite(incident_power) & (incident_power > 0)
            incident_peak = float(np.max(incident_power[positive_incident], initial=0.0))
            if incident_peak > 0:
                relative_db[positive_incident] = np.asarray(
                    10 * np.log10(incident_power[positive_incident] / incident_peak),
                    dtype=self.real_dtype,
                )
            source_valid = (
                np.asarray(spectrum.terminal_valid, dtype=bool)
                & positive_incident
                & (relative_db >= self.source_floor_db)
            )
            power_peak = float(np.max(np.abs(normalising_power), initial=0.0))
            power_defined = (
                spectrum.terminal_valid
                & spectrum.mesh_valid
                & np.isfinite(normalising_power)
                & (normalising_power > np.finfo(self.real_dtype).eps * power_peak)
            )
            valid &= source_valid & power_defined
            scale[power_defined] = np.sqrt(
                self.target_power / normalising_power[power_defined]
            ).astype(self.complex_dtype)
        return (
            source_spectrum,
            relative_db,
            source_valid,
            valid,
            scale,
            normalising_power,
            incident_power,
            incident_flux,
        )

    def _finalise_payload(
        self, payload: SARLocalPayload, grid_shape: tuple[int, int, int]
    ) -> SARResult:
        (
            source_spectrum,
            relative_db,
            source_valid,
            valid,
            scale,
            normalising_power,
            incident_power,
            incident_flux,
        ) = self._normalisation_data()
        absorbed = np.asarray(
            payload.absorbed_power_density * np.abs(scale[:, np.newaxis]) ** 2,
            dtype=self.real_dtype,
        )
        sar = np.asarray(absorbed / payload.density[np.newaxis, :], dtype=self.real_dtype)
        absorbed[~valid, :] = np.nan
        sar[~valid, :] = np.nan
        spatial_results = []
        if self.averaging_masses:
            density_volume = np.full(grid_shape, np.nan, dtype=np.float64)
            density_volume[tuple(payload.cell_indices.T)] = payload.density
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
                local_volume = np.zeros(grid_shape, dtype=np.float64)
                local_volume[tuple(payload.cell_indices.T)] = sar[frequency_index]
                averaged = apply_spatial_average_plan(plan, local_volume, density_volume)
                selection = tuple(payload.cell_indices.T)
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
            cell_indices=payload.cell_indices.copy(),
            tag_id=payload.tag_id.copy(),
            material_id=payload.material_id.copy(),
            density=np.asarray(payload.density, dtype=self.real_dtype),
            source_spectrum=source_spectrum,
            source_relative_db=relative_db,
            source_valid=np.asarray(source_valid, dtype=bool),
            incident_power=incident_power,
            incident_flux=incident_flux,
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

    def finalise(self) -> SARResult:
        payload = self.local_payload()
        grid_shape = getattr(
            self,
            "grid_shape",
            tuple(int(value) for value in np.max(payload.cell_indices, axis=0) + 1),
        )
        return self._finalise_payload(payload, grid_shape)

    def finalise_mpi(self, payloads: list[SARLocalPayload], global_shape) -> SARResult:
        """Finalise gathered rank payloads on the MPI coordinator."""

        if not self._mpi_distributed:
            raise RuntimeError("finalise_mpi requires an MPI SAR monitor")
        if self.normalisation in ("waveform", "current_moment", "incident_flux"):
            self._source_samples, self._source_dt = self._build_source_samples()
            self._source_window = _window(
                self.window_name, self._source_samples.size, np.dtype(np.float64)
            )
        payload = self.merge_local_payloads(payloads)
        payload = self._collocate_mpi_payload(payload, global_shape)
        self.excluded_pml_cell_count = payload.excluded_pml_cell_count
        self.grid_shape = tuple(int(value) for value in global_shape)
        self.cell_index_frame = "main-grid"
        self.cell_index_origin = np.zeros(3, dtype=np.float64)
        return self._finalise_payload(payload, self.grid_shape)

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
        group.attrs["Dimensionality"] = 2 if self.mode2d is not None else 3
        group.attrs["ModelMode"] = self.model_mode
        if self.mode2d is not None:
            group.attrs["Polarisation"] = self.mode2d.polarisation
            group.attrs["InvariantAxis"] = self.mode2d.invariant_axis_name
            group.attrs["LiveInvariantIndex"] = self.mode2d.live_index
            group.attrs["IntegrationMeasure"] = "per_unit_invariant_length"
        else:
            group.attrs["IntegrationMeasure"] = "volume"
        if self.waveform_id is not None:
            group.attrs["WaveformID"] = self.waveform_id
        group.attrs["TargetAmplitude"] = self.target_amplitude
        group.attrs["Normalisation"] = self.normalisation
        if self.normalisation == "waveform":
            group.attrs["NormalisationDefinition"] = "field_dft / waveform_dft * target_amplitude"
            group.attrs["SourceType"] = self._source_type
            group.attrs["SourceQuantity"] = self._source_quantity
            group.attrs["TargetAmplitudeUnits"] = self._source_units
        elif self.normalisation == "current_moment":
            group.attrs[
                "NormalisationDefinition"
            ] = "field_dft / (waveform_dft * source_length) * target_current_moment"
            group.attrs["SourceType"] = self._source_type
            group.attrs["SourceLength"] = self._source_length
            group.attrs["SourceLengthUnits"] = "m"
            group.attrs["TargetAmplitudeUnits"] = "A m"
        elif self.normalisation == "incident_flux":
            group.attrs[
                "NormalisationDefinition"
            ] = "field_dft * sqrt(target_flux / incident_plane_wave_flux)"
            group.attrs["SourceType"] = self._source_type
            group.attrs["TargetFlux"] = self.target_flux
            group.attrs["TargetFluxUnits"] = "W/m2"
        else:
            group.attrs[
                "NormalisationDefinition"
            ] = f"field_dft * sqrt(target_power / {self.normalisation})"
            group.attrs["PortID"] = self.port_id
            group.attrs["TargetPower"] = self.target_power
            group.attrs["TargetPowerUnits"] = "W/m" if self.mode2d is not None else "W"
        group.attrs["PhasorAmplitude"] = "peak"
        group.attrs["PhasorTimeDependence"] = PHASOR_TIME_DEPENDENCE
        group.attrs["ForwardTransformKernel"] = FORWARD_TRANSFORM_KERNEL
        group.attrs["Window"] = self.window_name
        group.attrs["CollectionBackend"] = self.collection_backend
        group.attrs["PMLCellPolicy"] = "excluded"
        group.attrs["ExcludedPMLCellCount"] = self.excluded_pml_cell_count
        group.attrs["CellIndexFrame"] = self.cell_index_frame
        group.attrs["CellIndexOrigin"] = self.cell_index_origin
        group.attrs["CellIndexOriginUnits"] = "m"
        centre_offset = np.asarray(self.grid_spacing) / 2
        if self.mode2d is not None and self.mode2d.polarisation == "TE":
            centre_offset[self.mode2d.invariant_axis] = 0
        group.attrs["CellCentreOffset"] = centre_offset
        group.attrs["CellCentreOffsetUnits"] = "m"
        group.attrs["SourceFloorDB"] = self.source_floor_db
        group.attrs["SpectrumLimitMode"] = self.spectrum_limit_mode
        group.attrs["MinimumWavelengthCells"] = self.minimum_wavelength_cells
        group.attrs["ActiveElectricComponents"] = np.asarray(tuple(self.edge_offsets), dtype="S")
        group.attrs["FieldCollocation"] = (
            "complex arithmetic mean of four parallel Yee edges per component"
            if self.mode2d is None or self.mode2d.polarisation == "TM"
            else "complex arithmetic mean of two tangential Yee edges per active component"
        )
        group.attrs["TagNames"] = np.asarray(self.tag_names, dtype="S")
        group["frequency"] = result.frequency
        group["cell_indices"] = result.cell_indices
        group["tag_id"] = result.tag_id
        group["material_id"] = result.material_id
        group["density"] = result.density
        group["source_spectrum"] = result.source_spectrum
        group["source_relative_db"] = result.source_relative_db
        group["source_valid"] = result.source_valid.astype(np.uint8)
        group["incident_power"] = result.incident_power
        group["incident_flux"] = result.incident_flux
        group.attrs["IncidentPowerUnits"] = "W/m" if self.mode2d is not None else "W"
        group.attrs["IncidentFluxUnits"] = "W/m2"
        group.attrs["NormalisingPowerUnits"] = "W/m" if self.mode2d is not None else "W"
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
            mass = float(np.sum(result.density[selection]) * self.cell_measure)
            absorbed_power = np.asarray(
                np.nansum(result.absorbed_power_density[:, selection], axis=1) * self.cell_measure,
                dtype=self.real_dtype,
            )
            tag_group = summaries.create_group(name)
            tag_group.attrs["TagID"] = int(tag_id)
            tag_group.attrs["CellCount"] = int(np.count_nonzero(selection))
            if self.mode2d is None:
                tag_group.attrs["Mass"] = mass
                tag_group.attrs["MassUnits"] = "kg"
                tag_group["absorbed_power"] = absorbed_power
                tag_group.attrs["AbsorbedPowerUnits"] = "W"
            else:
                tag_group.attrs["MassPerLength"] = mass
                tag_group.attrs["MassPerLengthUnits"] = "kg/m"
                tag_group["absorbed_power_per_length"] = absorbed_power
                tag_group.attrs["AbsorbedPowerPerLengthUnits"] = "W/m"
            tag_group["mass_average_sar"] = absorbed_power / mass
            peak_sar = np.full(result.frequency.shape, np.nan, dtype=self.real_dtype)
            for frequency_index in np.flatnonzero(result.valid):
                peak_sar[frequency_index] = np.max(
                    result.sar[frequency_index, selection], initial=0.0
                )
            tag_group["peak_voxel_sar"] = peak_sar


class RadiometryMonitor(SARMonitor):
    """Sparse frequency-domain absorbed-power and radiometric weighting output."""

    schema_version = 1

    def __init__(self, grid: "FDTDGrid", **kwargs) -> None:
        super().__init__(
            grid,
            averaging_masses=(),
            require_density=False,
            **kwargs,
        )

    def _finalise_payload(
        self, payload: SARLocalPayload, grid_shape: tuple[int, int, int]
    ) -> RadiometryResult:
        del grid_shape
        (
            source_spectrum,
            relative_db,
            source_valid,
            valid,
            scale,
            normalising_power,
            incident_power,
            incident_flux,
        ) = self._normalisation_data()
        absorbed = np.asarray(
            payload.absorbed_power_density * np.abs(scale[:, np.newaxis]) ** 2,
            dtype=self.real_dtype,
        )
        if self.normalisation in ("incident_power", "accepted_power"):
            reference = float(self.target_power)
        elif self.normalisation == "incident_flux":
            reference = float(self.target_flux)
        else:
            reference = float(self.target_amplitude) ** 2
        normalised = np.asarray(absorbed / reference, dtype=self.real_dtype)
        absorbed[~valid, :] = np.nan
        normalised[~valid, :] = np.nan
        self.result = RadiometryResult(
            frequency=np.asarray(self.frequencies, dtype=self.real_dtype),
            cell_indices=payload.cell_indices.copy(),
            tag_id=payload.tag_id.copy(),
            material_id=payload.material_id.copy(),
            source_spectrum=source_spectrum,
            source_relative_db=relative_db,
            source_valid=np.asarray(source_valid, dtype=bool),
            incident_power=incident_power,
            incident_flux=incident_flux,
            mesh_valid=np.asarray(self.mesh_valid, dtype=bool),
            valid=valid,
            cells_per_wavelength=np.asarray(self.cells_per_wavelength, dtype=self.real_dtype),
            limiting_material=self.limiting_material.copy(),
            absorbed_power_density=absorbed,
            normalised_absorption_density=normalised,
            normalisation_scale=scale,
            normalising_power=normalising_power,
        )
        return self.result

    def write_hdf5(self, basegrp) -> None:
        if self.result is None:
            self.finalise()
        result = self.result
        group = basegrp.create_group(f"radiometry/{self.output_id}")
        group.attrs["SchemaVersion"] = self.schema_version
        group.attrs["Quantity"] = "frequency-domain radiometric absorption weighting"
        group.attrs["AbsorbedPowerDensityUnits"] = "W/m3"
        group.attrs["Dimensionality"] = 2 if self.mode2d is not None else 3
        group.attrs["ModelMode"] = self.model_mode
        if self.mode2d is not None:
            group.attrs["Polarisation"] = self.mode2d.polarisation
            group.attrs["InvariantAxis"] = self.mode2d.invariant_axis_name
            group.attrs["LiveInvariantIndex"] = self.mode2d.live_index
            group.attrs["IntegrationMeasure"] = "per_unit_invariant_length"
        else:
            group.attrs["IntegrationMeasure"] = "volume"
        if self.waveform_id is not None:
            group.attrs["WaveformID"] = self.waveform_id
        group.attrs["TargetAmplitude"] = self.target_amplitude
        group.attrs["Normalisation"] = self.normalisation
        if self.normalisation == "waveform":
            group.attrs["NormalisationDefinition"] = "field_dft / waveform_dft * target_amplitude"
            group.attrs["SourceType"] = self._source_type
            group.attrs["SourceQuantity"] = self._source_quantity
            group.attrs["TargetAmplitudeUnits"] = self._source_units
            group.attrs[
                "NormalisedAbsorptionMeaning"
            ] = "absorbed power per squared source-native amplitude"
        elif self.normalisation == "current_moment":
            group.attrs[
                "NormalisationDefinition"
            ] = "field_dft / (waveform_dft * source_length) * target_current_moment"
            group.attrs["SourceType"] = self._source_type
            group.attrs["SourceLength"] = self._source_length
            group.attrs["SourceLengthUnits"] = "m"
            group.attrs["TargetAmplitudeUnits"] = "A m"
            group.attrs[
                "NormalisedAbsorptionMeaning"
            ] = "absorbed power per squared electric-current moment"
        elif self.normalisation == "incident_flux":
            group.attrs[
                "NormalisationDefinition"
            ] = "field_dft * sqrt(target_flux / incident_plane_wave_flux)"
            group.attrs["SourceType"] = self._source_type
            group.attrs["TargetFlux"] = self.target_flux
            group.attrs["TargetFluxUnits"] = "W/m2"
            group.attrs["NormalisedAbsorptionMeaning"] = "absorption cross-section density"
        else:
            group.attrs[
                "NormalisationDefinition"
            ] = f"field_dft * sqrt(target_power / {self.normalisation})"
            group.attrs["PortID"] = self.port_id
            group.attrs["TargetPower"] = self.target_power
            group.attrs["TargetPowerUnits"] = "W/m" if self.mode2d is not None else "W"
            group.attrs["NormalisedAbsorptionMeaning"] = "absorbed-power fraction density"
        if self.normalisation in ("incident_power", "accepted_power"):
            normalised_units = "1/m2" if self.mode2d is not None else "1/m3"
            integrated_units = "1"
        elif self.normalisation == "incident_flux":
            normalised_units = "1/m"
            integrated_units = "m" if self.mode2d is not None else "m2"
        else:
            invariant = "/m" if self.mode2d is not None else ""
            normalised_units = f"W/m3/({self._source_units})2"
            integrated_units = f"W{invariant}/({self._source_units})2"
        group.attrs["NormalisedAbsorptionDensityUnits"] = normalised_units
        group.attrs["IntegratedNormalisedAbsorptionUnits"] = integrated_units
        group.attrs["PhasorAmplitude"] = "peak"
        group.attrs["PhasorTimeDependence"] = PHASOR_TIME_DEPENDENCE
        group.attrs["ForwardTransformKernel"] = FORWARD_TRANSFORM_KERNEL
        group.attrs["Window"] = self.window_name
        group.attrs["CollectionBackend"] = self.collection_backend
        group.attrs["PMLCellPolicy"] = "excluded"
        group.attrs["ExcludedPMLCellCount"] = self.excluded_pml_cell_count
        group.attrs["CellIndexFrame"] = self.cell_index_frame
        group.attrs["CellIndexOrigin"] = self.cell_index_origin
        group.attrs["CellIndexOriginUnits"] = "m"
        centre_offset = np.asarray(self.grid_spacing) / 2
        if self.mode2d is not None and self.mode2d.polarisation == "TE":
            centre_offset[self.mode2d.invariant_axis] = 0
        group.attrs["CellCentreOffset"] = centre_offset
        group.attrs["CellCentreOffsetUnits"] = "m"
        group.attrs["SourceFloorDB"] = self.source_floor_db
        group.attrs["SpectrumLimitMode"] = self.spectrum_limit_mode
        group.attrs["MinimumWavelengthCells"] = self.minimum_wavelength_cells
        group.attrs["ActiveElectricComponents"] = np.asarray(tuple(self.edge_offsets), dtype="S")
        group.attrs["TagNames"] = np.asarray(self.tag_names, dtype="S")
        group["frequency"] = result.frequency
        group["cell_indices"] = result.cell_indices
        group["tag_id"] = result.tag_id
        group["material_id"] = result.material_id
        group["source_spectrum"] = result.source_spectrum
        group["source_relative_db"] = result.source_relative_db
        group["source_valid"] = result.source_valid.astype(np.uint8)
        group["incident_power"] = result.incident_power
        group["incident_flux"] = result.incident_flux
        group.attrs["IncidentPowerUnits"] = "W/m" if self.mode2d is not None else "W"
        group.attrs["IncidentFluxUnits"] = "W/m2"
        group.attrs["NormalisingPowerUnits"] = "W/m" if self.mode2d is not None else "W"
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
        group["normalised_absorption_density"] = result.normalised_absorption_density

        summaries = group.create_group("tags")
        for name, tag_id in zip(self.tag_names, self.tag_ids):
            selection = result.tag_id == tag_id
            absorbed_power = np.asarray(
                np.nansum(result.absorbed_power_density[:, selection], axis=1) * self.cell_measure,
                dtype=self.real_dtype,
            )
            integrated = np.asarray(
                np.nansum(result.normalised_absorption_density[:, selection], axis=1)
                * self.cell_measure,
                dtype=self.real_dtype,
            )
            tag_group = summaries.create_group(name)
            tag_group.attrs["TagID"] = int(tag_id)
            tag_group.attrs["CellCount"] = int(np.count_nonzero(selection))
            if self.mode2d is None:
                tag_group["absorbed_power"] = absorbed_power
                tag_group.attrs["AbsorbedPowerUnits"] = "W"
            else:
                tag_group["absorbed_power_per_length"] = absorbed_power
                tag_group.attrs["AbsorbedPowerPerLengthUnits"] = "W/m"
            tag_group["normalised_absorption"] = integrated
            tag_group.attrs["NormalisedAbsorptionUnits"] = integrated_units


def compile_sar_outputs(grid: "FDTDGrid", *, model: "Model | None" = None) -> None:
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
            target_flux=spec.target_flux,
            model=model,
        )
        grid.sar_monitors.append(monitor)
        if spec.owner is not None:
            spec.owner._monitor = monitor
    for spec in getattr(grid, "radiometry_specs", ()):
        monitor = RadiometryMonitor(
            grid,
            output_id=spec.output_id,
            frequencies=spec.frequencies,
            tags=spec.tags,
            waveform_id=spec.waveform_id,
            target_amplitude=spec.target_amplitude,
            spectrum_limit=spec.spectrum_limit,
            source_floor_db=spec.source_floor_db,
            window=spec.window,
            normalisation=spec.normalisation,
            port_id=spec.port_id,
            target_power=spec.target_power,
            target_flux=spec.target_flux,
            model=model,
        )
        grid.sar_monitors.append(monitor)
        if spec.owner is not None:
            spec.owner._monitor = monitor
