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

"""Device-resident KSIR DFT collection for CUDA, OpenCL, and Metal."""

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import numpy.typing as npt

from gprMax.cuda_opencl.knl_ntff import (
    build_ntff_kernel_source,
    build_time_domain_ntff_kernel_source,
)

ELECTRIC_COMPONENTS = ("Ex", "Ey", "Ez")
MAGNETIC_COMPONENTS = ("Hx", "Hy", "Hz")


def _is_frequency_monitor(monitor) -> bool:
    return hasattr(monitor, "device_sampling_multiplier")


def _is_time_domain_monitor(monitor) -> bool:
    return hasattr(monitor, "load_device_component_output")


def configure_ntff_monitors(grid, *, allow_time_domain: bool = False) -> None:
    """Perform backend-independent material validation before time stepping."""

    for monitor in grid.ntff_monitors:
        if _is_time_domain_monitor(monitor) and not allow_time_domain:
            raise ValueError("advanced-time KSIR requires a time-domain-capable device collector")
        if not (_is_frequency_monitor(monitor) or _is_time_domain_monitor(monitor)):
            raise TypeError("unrecognised NTFF monitor type")
        monitor.validate_materials(grid.ID, grid.IDlookup)
        configure_background = getattr(monitor, "configure_background", None)
        if configure_background is not None:
            configure_background(grid.materials)


@dataclass
class _ComponentRecord:
    monitor: object
    component: str
    npatches: int
    nfrequencies: int
    inside_index: npt.NDArray[np.int32]
    outside_index: npt.NDArray[np.int32]
    device: Dict[str, object]

    @property
    def total(self) -> int:
        return self.npatches * self.nfrequencies

    @property
    def shape(self) -> tuple[int, int]:
        return self.nfrequencies, self.npatches


class _DeviceKSIRCollector:
    """Backend-neutral sequencing and configured-dtype reconstruction."""

    def __init__(self, updates, monitors=None, *, configure: bool = True):
        self.updates = updates
        self.grid = updates.grid
        if configure:
            configure_ntff_monitors(self.grid)
        self.monitors = list(self.grid.ntff_monitors if monitors is None else monitors)
        self.records: List[_ComponentRecord] = []
        self.incident_records = []
        limits = np.iinfo(np.int32)
        for monitor in self.monitors:
            if not _is_frequency_monitor(monitor):
                raise TypeError("frequency collector received a time-domain monitor")
            collector_dtype = getattr(self, "real_dtype", None)
            if collector_dtype is not None and monitor.real_dtype != collector_dtype:
                raise ValueError("KSIR monitor dtype does not match field dtype")
            for component, surface in monitor.surfaces.items():
                npatches = int(surface.npatches)
                nfrequencies = int(monitor.frequencies.size)
                total = npatches * nfrequencies
                if npatches > limits.max or nfrequencies > limits.max or total > limits.max:
                    raise ValueError(
                        "KSIR frequency-domain work-item count "
                        f"{total} ({npatches} surface patches * "
                        f"{nfrequencies} frequencies) exceeds device int32 "
                        "indexing"
                    )
                inside = np.concatenate([face.inside_flat_indices for face in surface.faces])
                outside = np.concatenate([face.outside_flat_indices for face in surface.faces])
                if (
                    np.any(inside < limits.min)
                    or np.any(inside > limits.max)
                    or np.any(outside < limits.min)
                    or np.any(outside > limits.max)
                ):
                    raise ValueError("KSIR device surface indices exceed int32 range")
                record = _ComponentRecord(
                    monitor=monitor,
                    component=component,
                    npatches=npatches,
                    nfrequencies=nfrequencies,
                    inside_index=inside.astype(np.int32),
                    outside_index=outside.astype(np.int32),
                    device={},
                )
                self._allocate(record)
                self.records.append(record)
            if getattr(monitor, "associated_plane_wave", None) is not None:
                plane_wave = monitor.associated_plane_wave
                if not hasattr(plane_wave, "E_fields_dev"):
                    raise ValueError(
                        "device KSIR RCS normalisation requires the associated "
                        "plane wave to have device-resident electric fields"
                    )
                reference_index = int(monitor._incident_reference_index)
                if reference_index < 0 or reference_index > limits.max:
                    raise ValueError(
                        "KSIR incident plane-wave reference index exceeds device " "int32 indexing"
                    )
                component_records = []
                indices = np.asarray([reference_index], dtype=np.int32)
                for axis, component in enumerate(ELECTRIC_COMPONENTS):
                    record = _ComponentRecord(
                        monitor=monitor,
                        component=component,
                        npatches=1,
                        nfrequencies=int(monitor.frequencies.size),
                        inside_index=indices,
                        outside_index=indices,
                        device={},
                    )
                    self._allocate(record)
                    record.device["field"] = plane_wave.E_fields_dev[axis]
                    component_records.append(record)
                self.incident_records.append((monitor, component_records))

    def _allocate(self, record: _ComponentRecord) -> None:
        raise NotImplementedError

    def _accumulate(
        self,
        record: _ComponentRecord,
        field,
        multiplier: npt.NDArray[np.complexfloating],
    ) -> None:
        raise NotImplementedError

    def _download(
        self, record: _ComponentRecord
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        raise NotImplementedError

    def _observe(self, iteration: int, components: Iterable[str]) -> None:
        selected = set(components)
        for record in self.records:
            if record.component not in selected:
                continue
            multiplier = record.monitor.device_sampling_multiplier(record.component, iteration)
            field = getattr(self.grid, f"{record.component}_dev")
            self._accumulate(record, field, multiplier)

    def observe_electric(self, iteration: int) -> None:
        self._observe(iteration, ELECTRIC_COMPONENTS)
        for monitor, records in self.incident_records:
            multiplier = monitor.device_incident_sampling_multiplier(iteration)
            for record in records:
                self._accumulate(record, record.device["field"], multiplier)

    def observe_magnetic(self, iteration: int) -> None:
        self._observe(iteration, MAGNETIC_COMPONENTS)

    def finalise(self) -> None:
        for record in self.records:
            inside_real, inside_imag, outside_real, outside_imag = self._download(record)
            dtype = record.monitor.complex_dtype
            inside = np.empty(record.shape, dtype=dtype)
            outside = np.empty(record.shape, dtype=dtype)
            inside.real = np.asarray(inside_real).reshape(record.shape)
            inside.imag = np.asarray(inside_imag).reshape(record.shape)
            outside.real = np.asarray(outside_real).reshape(record.shape)
            outside.imag = np.asarray(outside_imag).reshape(record.shape)
            record.monitor.load_device_component_dfts(record.component, inside, outside)
        for monitor, records in self.incident_records:
            incident = np.empty(
                (monitor.frequencies.size, len(ELECTRIC_COMPONENTS)),
                dtype=monitor.complex_dtype,
            )
            for axis, record in enumerate(records):
                inside_real, inside_imag, _, _ = self._download(record)
                incident[:, axis].real = np.asarray(inside_real).reshape(record.shape)[:, 0]
                incident[:, axis].imag = np.asarray(inside_imag).reshape(record.shape)[:, 0]
            monitor.load_device_incident_electric(incident)
        for monitor in self.monitors:
            monitor.finalise()


class CUDAKSIRCollector(_DeviceKSIRCollector):
    """CUDA raw-surface DFT collector."""

    def __init__(self, updates, monitors=None, *, configure: bool = True):
        c_real = updates.ntff_c_real
        source = build_ntff_kernel_source("cuda", c_real)
        module = updates.source_module(
            source, options=getattr(updates, "ntff_compiler_options", None)
        )
        self.kernel = module.get_function("accumulate_ntff")
        self.gpuarray = updates.grid.gpuarray
        self.real_dtype = np.dtype(updates.grid.Ex.dtype)
        super().__init__(updates, monitors, configure=configure)

    def _allocate(self, record: _ComponentRecord) -> None:
        record.device["inside_index"] = self.gpuarray.to_gpu(record.inside_index)
        record.device["outside_index"] = self.gpuarray.to_gpu(record.outside_index)
        record.device["multiplier_real"] = self.gpuarray.empty(record.nfrequencies, self.real_dtype)
        record.device["multiplier_imag"] = self.gpuarray.empty(record.nfrequencies, self.real_dtype)
        for name in ("inside_real", "inside_imag", "outside_real", "outside_imag"):
            record.device[name] = self.gpuarray.zeros(record.total, self.real_dtype)

    def _accumulate(self, record, field, multiplier) -> None:
        real = np.asarray(multiplier.real, dtype=self.real_dtype)
        imag = np.asarray(multiplier.imag, dtype=self.real_dtype)
        record.device["multiplier_real"].set(real)
        record.device["multiplier_imag"].set(imag)
        field_pointer = field.gpudata
        # A row view into the 2-D auxiliary plane-wave array exposes its
        # offset pointer as an integer. Passing the GPUArray view lets PyCUDA
        # obtain the correctly offset pointer from its CUDA array interface.
        if isinstance(field_pointer, (int, np.integer)):
            field_pointer = field
        self.kernel(
            np.int32(record.total),
            np.int32(record.npatches),
            record.device["inside_index"].gpudata,
            record.device["outside_index"].gpudata,
            record.device["multiplier_real"].gpudata,
            record.device["multiplier_imag"].gpudata,
            field_pointer,
            record.device["inside_real"].gpudata,
            record.device["inside_imag"].gpudata,
            record.device["outside_real"].gpudata,
            record.device["outside_imag"].gpudata,
            block=(128, 1, 1),
            grid=((record.total + 127) // 128, 1, 1),
        )

    def _download(self, record):
        return tuple(
            record.device[name].get()
            for name in ("inside_real", "inside_imag", "outside_real", "outside_imag")
        )


@dataclass
class _TimeComponentRecord:
    monitor: object
    component: str
    nbase_patches: int
    neffective_patches: int
    npoints: int
    output_length: int
    device: Dict[str, object]
    observed: int = 0

    @property
    def total(self) -> int:
        return self.neffective_patches * self.npoints


class CUDATimeDomainKSIRCollector:
    """Device-resident advanced-time KSIR collection and deposition."""

    _threads = 128

    def __init__(self, updates, monitors):
        self.updates = updates
        self.grid = updates.grid
        self.monitors = list(monitors)
        self.real_dtype = np.dtype(self.grid.Ex.dtype)
        self.real_scalar = self.real_dtype.type
        self.inverse_dt = 1 / self.grid.dt
        self.inverse_two_dt = 1 / (2 * self.grid.dt)
        self.gpuarray = self.grid.gpuarray
        source = build_time_domain_ntff_kernel_source(updates.ntff_c_real)
        module = updates.source_module(
            source, options=getattr(updates, "ntff_compiler_options", None)
        )
        self.gather_kernel = module.get_function("gather_time_domain_ntff")
        self.deposit_kernel = module.get_function("deposit_time_domain_ntff")
        self._initialise_records("cuda")

    def _initialise_records(self, backend: str) -> None:
        """Build backend-independent record metadata and device allocations."""

        self.records: List[_TimeComponentRecord] = []

        int_limit = np.iinfo(np.int32)
        for monitor in self.monitors:
            if monitor.device_backend != backend:
                raise ValueError(f"{backend} time-domain monitor is not configured for {backend}")
            if monitor.real_dtype != self.real_dtype:
                raise ValueError("KSIR monitor dtype does not match field dtype")
            if monitor.iterations > int_limit.max or monitor.output_length > int_limit.max:
                raise ValueError("KSIR time-domain extent exceeds device int32 indexing")
            for component, surface in monitor.surfaces.items():
                accumulator = monitor._accumulators[component]
                record = _TimeComponentRecord(
                    monitor=monitor,
                    component=component,
                    nbase_patches=surface.npatches,
                    neffective_patches=accumulator._source_patch_index.size,
                    npoints=monitor.points.shape[0],
                    output_length=monitor.output_length,
                    device={},
                )
                if (
                    record.total > int_limit.max
                    or record.npoints * record.output_length > int_limit.max
                ):
                    raise ValueError("KSIR time-domain arrays exceed device int32 indexing")
                self._allocate(record, accumulator)
                self.records.append(record)

    def _to_int32(self, name: str, values: npt.ArrayLike) -> npt.NDArray[np.int32]:
        array = np.asarray(values)
        limits = np.iinfo(np.int32)
        if np.any(array < limits.min) or np.any(array > limits.max):
            raise ValueError(f"{name} exceeds device int32 indexing")
        return np.ascontiguousarray(array, dtype=np.int32)

    def _allocate(self, record: _TimeComponentRecord, accumulator) -> None:
        integer_arrays = {
            "inside_index": accumulator._inside_indices,
            "outside_index": accumulator._outside_indices,
            "source_patch_index": accumulator._source_patch_index,
            "integer_delay": accumulator._integer_delay.ravel(),
            "time_origin_steps": accumulator._time_origin_steps,
        }
        for name, values in integer_arrays.items():
            record.device[name] = self.gpuarray.to_gpu(self._to_int32(name, values))

        real_arrays = {
            "normal_spacing": accumulator._normal_spacing,
            "normal_derivative_weight": accumulator._normal_derivative_weight.ravel(),
            "field_weight": accumulator._field_weight.ravel(),
            "time_derivative_weight": accumulator._time_derivative_weight.ravel(),
            "fractional_delay": accumulator._fractional_delay.ravel(),
        }
        for name, values in real_arrays.items():
            record.device[name] = self.gpuarray.to_gpu(
                np.ascontiguousarray(values, dtype=self.real_dtype)
            )

        record.device["surface"] = [
            self.gpuarray.empty(record.nbase_patches, self.real_dtype) for _ in range(3)
        ]
        record.device["normal_derivative"] = [
            self.gpuarray.empty(record.nbase_patches, self.real_dtype) for _ in range(3)
        ]
        record.device["output"] = self.gpuarray.zeros(
            record.npoints * record.output_length, self.real_dtype
        )

    def _gather(self, record: _TimeComponentRecord, iteration: int) -> None:
        slot = iteration % 3
        field = getattr(self.grid, f"{record.component}_dev")
        self.gather_kernel(
            np.int32(record.nbase_patches),
            record.device["inside_index"].gpudata,
            record.device["outside_index"].gpudata,
            record.device["normal_spacing"].gpudata,
            field.gpudata,
            record.device["surface"][slot].gpudata,
            record.device["normal_derivative"][slot].gpudata,
            block=(self._threads, 1, 1),
            grid=(
                ((record.nbase_patches + self._threads - 1) // self._threads),
                1,
                1,
            ),
        )

    def _deposit(
        self,
        record: _TimeComponentRecord,
        sample_index: int,
        derivative_samples: tuple[int, int, int],
        derivative_coefficients: tuple[float, float, float],
    ) -> None:
        surface_slot = sample_index % 3
        time_surfaces = [record.device["surface"][sample % 3] for sample in derivative_samples]
        coefficients = [self.real_scalar(value) for value in derivative_coefficients]
        self.deposit_kernel(
            np.int32(record.total),
            np.int32(record.neffective_patches),
            np.int32(record.output_length),
            np.int32(sample_index),
            record.device["surface"][surface_slot].gpudata,
            record.device["normal_derivative"][surface_slot].gpudata,
            time_surfaces[0].gpudata,
            time_surfaces[1].gpudata,
            time_surfaces[2].gpudata,
            coefficients[0],
            coefficients[1],
            coefficients[2],
            record.device["normal_derivative_weight"].gpudata,
            record.device["field_weight"].gpudata,
            record.device["time_derivative_weight"].gpudata,
            record.device["source_patch_index"].gpudata,
            record.device["integer_delay"].gpudata,
            record.device["fractional_delay"].gpudata,
            record.device["time_origin_steps"].gpudata,
            record.device["output"].gpudata,
            block=(self._threads, 1, 1),
            grid=((record.total + self._threads - 1) // self._threads, 1, 1),
        )

    def _observe(self, iteration: int, components: Iterable[str]) -> None:
        selected = set(components)
        for record in self.records:
            if record.component not in selected:
                continue
            if iteration != record.observed:
                raise ValueError(
                    f"expected device KSIR iteration {record.observed}, received {iteration}"
                )
            self._gather(record, iteration)
            record.observed += 1
            if iteration < 2:
                continue
            if iteration == 2:
                self._deposit(
                    record,
                    0,
                    (0, 1, 2),
                    (
                        -3 * self.inverse_two_dt,
                        4 * self.inverse_two_dt,
                        -self.inverse_two_dt,
                    ),
                )
            self._deposit(
                record,
                iteration - 1,
                (iteration - 2, iteration, iteration),
                (-self.inverse_two_dt, self.inverse_two_dt, 0),
            )

    def observe_electric(self, iteration: int) -> None:
        self._observe(iteration, ELECTRIC_COMPONENTS)

    def observe_magnetic(self, iteration: int) -> None:
        self._observe(iteration, MAGNETIC_COMPONENTS)

    def _deposit_endpoint(self, record: _TimeComponentRecord) -> None:
        iterations = record.monitor.iterations
        if record.observed != iterations:
            raise RuntimeError(
                f"device KSIR component {record.component} received "
                f"{record.observed} of {iterations} samples"
            )
        if iterations == 1:
            self._deposit(record, 0, (0, 0, 0), (0, 0, 0))
        elif iterations == 2:
            coefficients = (-self.inverse_dt, self.inverse_dt, 0)
            self._deposit(record, 0, (0, 1, 1), coefficients)
            self._deposit(record, 1, (0, 1, 1), coefficients)
        else:
            last = iterations - 1
            self._deposit(
                record,
                last,
                (last - 2, last - 1, last),
                (
                    self.inverse_two_dt,
                    -4 * self.inverse_two_dt,
                    3 * self.inverse_two_dt,
                ),
            )

    def finalise(self) -> None:
        for record in self.records:
            self._deposit_endpoint(record)
            output = self._download_output(record).reshape(record.npoints, record.output_length)
            record.monitor.load_device_component_output(record.component, output)
        for monitor in self.monitors:
            monitor.finalise()

    def _download_output(self, record: _TimeComponentRecord) -> npt.NDArray:
        return record.device["output"].get()


class CUDACombinedKSIRCollector:
    """Dispatch frequency-domain and advanced-time monitors on CUDA."""

    def __init__(self, updates):
        configure_ntff_monitors(updates.grid, allow_time_domain=True)
        frequency_monitors = [
            monitor for monitor in updates.grid.ntff_monitors if _is_frequency_monitor(monitor)
        ]
        time_monitors = [
            monitor for monitor in updates.grid.ntff_monitors if _is_time_domain_monitor(monitor)
        ]
        self.frequency = (
            CUDAKSIRCollector(updates, frequency_monitors, configure=False)
            if frequency_monitors
            else None
        )
        self.time_domain = (
            CUDATimeDomainKSIRCollector(updates, time_monitors) if time_monitors else None
        )

    def observe_electric(self, iteration: int) -> None:
        if self.frequency is not None:
            self.frequency.observe_electric(iteration)
        if self.time_domain is not None:
            self.time_domain.observe_electric(iteration)

    def observe_magnetic(self, iteration: int) -> None:
        if self.frequency is not None:
            self.frequency.observe_magnetic(iteration)
        if self.time_domain is not None:
            self.time_domain.observe_magnetic(iteration)

    def finalise(self) -> None:
        if self.frequency is not None:
            self.frequency.finalise()
        if self.time_domain is not None:
            self.time_domain.finalise()


class OpenCLKSIRCollector(_DeviceKSIRCollector):
    """OpenCL raw-surface DFT collector."""

    def __init__(self, updates, monitors=None, *, configure: bool = True):
        c_real = updates.ntff_c_real
        source = build_ntff_kernel_source("opencl", c_real)
        options = getattr(updates, "ntff_compiler_options", None)
        self.kernel = updates.cl.Program(updates.ctx, source).build(options=options).accumulate_ntff
        self.clarray = updates.grid.clarray
        self.queue = updates.queue
        self.real_dtype = np.dtype(updates.grid.Ex.dtype)
        super().__init__(updates, monitors, configure=configure)

    def _allocate(self, record: _ComponentRecord) -> None:
        record.device["inside_index"] = self.clarray.to_device(self.queue, record.inside_index)
        record.device["outside_index"] = self.clarray.to_device(self.queue, record.outside_index)
        record.device["multiplier_real"] = self.clarray.empty(
            self.queue, record.nfrequencies, self.real_dtype
        )
        record.device["multiplier_imag"] = self.clarray.empty(
            self.queue, record.nfrequencies, self.real_dtype
        )
        for name in ("inside_real", "inside_imag", "outside_real", "outside_imag"):
            record.device[name] = self.clarray.zeros(self.queue, record.total, self.real_dtype)

    def _accumulate(self, record, field, multiplier) -> None:
        # ``multiplier`` is complex, so its ``.real`` and ``.imag`` views
        # normally retain the interleaved complex stride. PyCUDA accepts
        # those strided views, but ``pyopencl.array.Array.set`` requires a
        # C-contiguous host array (enforced by recent PyOpenCL releases).
        # Materialise the two scalar arrays before uploading them.
        record.device["multiplier_real"].set(
            np.ascontiguousarray(multiplier.real, dtype=self.real_dtype), queue=self.queue
        )
        record.device["multiplier_imag"].set(
            np.ascontiguousarray(multiplier.imag, dtype=self.real_dtype), queue=self.queue
        )
        self.kernel(
            self.queue,
            (record.total,),
            None,
            np.int32(record.total),
            np.int32(record.npatches),
            record.device["inside_index"].data,
            record.device["outside_index"].data,
            record.device["multiplier_real"].data,
            record.device["multiplier_imag"].data,
            np.int32(field.offset // field.dtype.itemsize),
            field.base_data,
            record.device["inside_real"].data,
            record.device["inside_imag"].data,
            record.device["outside_real"].data,
            record.device["outside_imag"].data,
        )

    def _download(self, record):
        return tuple(
            record.device[name].get(queue=self.queue)
            for name in ("inside_real", "inside_imag", "outside_real", "outside_imag")
        )


class OpenCLTimeDomainKSIRCollector(CUDATimeDomainKSIRCollector):
    """Device-resident advanced-time KSIR collection on OpenCL."""

    def __init__(self, updates, monitors):
        self.updates = updates
        self.grid = updates.grid
        self.monitors = list(monitors)
        self.real_dtype = np.dtype(self.grid.Ex.dtype)
        self.real_scalar = self.real_dtype.type
        self.inverse_dt = 1 / self.grid.dt
        self.inverse_two_dt = 1 / (2 * self.grid.dt)
        self.clarray = self.grid.clarray
        self.queue = updates.queue
        source = build_time_domain_ntff_kernel_source(updates.ntff_c_real, backend="opencl")
        options = getattr(updates, "ntff_compiler_options", None)
        program = updates.cl.Program(updates.ctx, source).build(options=options)
        self.gather_kernel = program.gather_time_domain_ntff
        self.deposit_kernel = program.deposit_time_domain_ntff
        self._initialise_records("opencl")

    def _allocate(self, record: _TimeComponentRecord, accumulator) -> None:
        integer_arrays = {
            "inside_index": accumulator._inside_indices,
            "outside_index": accumulator._outside_indices,
            "source_patch_index": accumulator._source_patch_index,
            "integer_delay": accumulator._integer_delay.ravel(),
            "time_origin_steps": accumulator._time_origin_steps,
        }
        for name, values in integer_arrays.items():
            record.device[name] = self.clarray.to_device(self.queue, self._to_int32(name, values))

        real_arrays = {
            "normal_spacing": accumulator._normal_spacing,
            "normal_derivative_weight": accumulator._normal_derivative_weight.ravel(),
            "field_weight": accumulator._field_weight.ravel(),
            "time_derivative_weight": accumulator._time_derivative_weight.ravel(),
            "fractional_delay": accumulator._fractional_delay.ravel(),
        }
        for name, values in real_arrays.items():
            record.device[name] = self.clarray.to_device(
                self.queue, np.ascontiguousarray(values, dtype=self.real_dtype)
            )

        record.device["surface"] = [
            self.clarray.empty(self.queue, record.nbase_patches, self.real_dtype) for _ in range(3)
        ]
        record.device["normal_derivative"] = [
            self.clarray.empty(self.queue, record.nbase_patches, self.real_dtype) for _ in range(3)
        ]
        record.device["output"] = self.clarray.zeros(
            self.queue,
            record.npoints * record.output_length,
            self.real_dtype,
        )

    def _gather(self, record: _TimeComponentRecord, iteration: int) -> None:
        slot = iteration % 3
        field = getattr(self.grid, f"{record.component}_dev")
        self.gather_kernel(
            self.queue,
            (record.nbase_patches,),
            None,
            np.int32(record.nbase_patches),
            record.device["inside_index"].data,
            record.device["outside_index"].data,
            record.device["normal_spacing"].data,
            field.data,
            record.device["surface"][slot].data,
            record.device["normal_derivative"][slot].data,
        )

    def _deposit(
        self,
        record: _TimeComponentRecord,
        sample_index: int,
        derivative_samples: tuple[int, int, int],
        derivative_coefficients: tuple[float, float, float],
    ) -> None:
        surface_slot = sample_index % 3
        time_surfaces = [record.device["surface"][sample % 3] for sample in derivative_samples]
        coefficients = [self.real_scalar(value) for value in derivative_coefficients]
        self.deposit_kernel(
            self.queue,
            (record.npoints,),
            None,
            np.int32(record.npoints),
            np.int32(record.neffective_patches),
            np.int32(record.output_length),
            np.int32(sample_index),
            record.device["surface"][surface_slot].data,
            record.device["normal_derivative"][surface_slot].data,
            time_surfaces[0].data,
            time_surfaces[1].data,
            time_surfaces[2].data,
            coefficients[0],
            coefficients[1],
            coefficients[2],
            record.device["normal_derivative_weight"].data,
            record.device["field_weight"].data,
            record.device["time_derivative_weight"].data,
            record.device["source_patch_index"].data,
            record.device["integer_delay"].data,
            record.device["fractional_delay"].data,
            record.device["time_origin_steps"].data,
            record.device["output"].data,
        )

    def _download_output(self, record: _TimeComponentRecord) -> npt.NDArray:
        return record.device["output"].get(queue=self.queue)


class OpenCLCombinedKSIRCollector:
    """Dispatch frequency-domain and advanced-time monitors on OpenCL."""

    def __init__(self, updates):
        configure_ntff_monitors(updates.grid, allow_time_domain=True)
        frequency_monitors = [
            monitor for monitor in updates.grid.ntff_monitors if _is_frequency_monitor(monitor)
        ]
        time_monitors = [
            monitor for monitor in updates.grid.ntff_monitors if _is_time_domain_monitor(monitor)
        ]
        self.frequency = (
            OpenCLKSIRCollector(updates, frequency_monitors, configure=False)
            if frequency_monitors
            else None
        )
        self.time_domain = (
            OpenCLTimeDomainKSIRCollector(updates, time_monitors) if time_monitors else None
        )

    def observe_electric(self, iteration: int) -> None:
        if self.frequency is not None:
            self.frequency.observe_electric(iteration)
        if self.time_domain is not None:
            self.time_domain.observe_electric(iteration)

    def observe_magnetic(self, iteration: int) -> None:
        if self.frequency is not None:
            self.frequency.observe_magnetic(iteration)
        if self.time_domain is not None:
            self.time_domain.observe_magnetic(iteration)

    def finalise(self) -> None:
        if self.frequency is not None:
            self.frequency.finalise()
        if self.time_domain is not None:
            self.time_domain.finalise()


class MetalKSIRCollector(_DeviceKSIRCollector):
    """Metal raw-surface DFT collector using shared MTLBuffers."""

    def __init__(self, updates, monitors=None, *, configure: bool = True):
        c_real = updates.ntff_c_real
        source = build_ntff_kernel_source("metal", c_real)
        library, error = updates.dev.newLibraryWithSource_options_error_(source, updates.opts, None)
        if library is None:
            raise RuntimeError(f"Failed to compile Metal NTFF kernel: {error}")
        function = library.newFunctionWithName_("accumulate_ntff")
        self.pipeline = updates.dev.newComputePipelineStateWithFunction_error_(function, None)[0]
        self.dev = updates.dev
        self.queue = updates.cmdqueue
        self.metal = updates.metal
        self.storage = getattr(updates.grid, "storage", 0)
        self.real_dtype = np.dtype(updates.grid.Ex.dtype)
        super().__init__(updates, monitors, configure=configure)

    def _buffer(self, values: npt.NDArray):
        contiguous = np.ascontiguousarray(values)
        return self.dev.newBufferWithBytes_length_options_(
            contiguous, contiguous.nbytes, self.storage
        )

    def _allocate(self, record: _ComponentRecord) -> None:
        record.device["inside_index"] = self._buffer(record.inside_index)
        record.device["outside_index"] = self._buffer(record.outside_index)
        zeros = np.zeros(record.total, dtype=self.real_dtype)
        for name in ("inside_real", "inside_imag", "outside_real", "outside_imag"):
            record.device[name] = self._buffer(zeros)

    def _accumulate(self, record, field, multiplier) -> None:
        total = self._buffer(np.asarray((record.total,), dtype=np.int32))
        npatches = self._buffer(np.asarray((record.npatches,), dtype=np.int32))
        multiplier_real = self._buffer(np.asarray(multiplier.real, dtype=self.real_dtype))
        multiplier_imag = self._buffer(np.asarray(multiplier.imag, dtype=self.real_dtype))
        command = self.queue.commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(self.pipeline)
        buffers = (
            total,
            npatches,
            record.device["inside_index"],
            record.device["outside_index"],
            multiplier_real,
            multiplier_imag,
            field,
            record.device["inside_real"],
            record.device["inside_imag"],
            record.device["outside_real"],
            record.device["outside_imag"],
        )
        for index, buffer in enumerate(buffers):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            self.metal.MTLSizeMake(record.total, 1, 1),
            self.metal.MTLSizeMake(
                min(record.total, self.pipeline.maxTotalThreadsPerThreadgroup()),
                1,
                1,
            ),
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()

    def _download(self, record):
        nbytes = record.total * self.real_dtype.itemsize
        return tuple(
            np.frombuffer(
                record.device[name].contents().as_buffer(nbytes),
                dtype=self.real_dtype,
            ).copy()
            for name in ("inside_real", "inside_imag", "outside_real", "outside_imag")
        )


class MetalTimeDomainKSIRCollector(CUDATimeDomainKSIRCollector):
    """Device-resident advanced-time KSIR collection on Apple Metal."""

    def __init__(self, updates, monitors):
        self.updates = updates
        self.grid = updates.grid
        self.monitors = list(monitors)
        self.real_dtype = np.dtype(self.grid.Ex.dtype)
        self.real_scalar = self.real_dtype.type
        self.inverse_dt = 1 / self.grid.dt
        self.inverse_two_dt = 1 / (2 * self.grid.dt)
        self.dev = updates.dev
        self.queue = updates.cmdqueue
        self.metal = updates.metal
        self.storage = getattr(self.grid, "storage", 0)
        source = build_time_domain_ntff_kernel_source(updates.ntff_c_real, backend="metal")
        library, error = self.dev.newLibraryWithSource_options_error_(source, updates.opts, None)
        if library is None:
            raise RuntimeError(f"Failed to compile Metal time-domain NTFF: {error}")
        gather = library.newFunctionWithName_("gather_time_domain_ntff")
        deposit = library.newFunctionWithName_("deposit_time_domain_ntff")
        self.gather_pipeline = self.dev.newComputePipelineStateWithFunction_error_(gather, None)[0]
        self.deposit_pipeline = self.dev.newComputePipelineStateWithFunction_error_(deposit, None)[
            0
        ]
        self._initialise_records("metal")

    def _buffer(self, values: npt.NDArray):
        contiguous = np.ascontiguousarray(values)
        return self.dev.newBufferWithBytes_length_options_(
            contiguous, contiguous.nbytes, self.storage
        )

    @staticmethod
    def _set_scalar(encoder, index: int, value) -> None:
        scalar = np.asarray(value)
        encoder.setBytes_length_atIndex_(scalar.tobytes(), scalar.dtype.itemsize, index)

    def _dispatch(self, command, encoder, pipeline, work_items: int) -> None:
        encoder.dispatchThreads_threadsPerThreadgroup_(
            self.metal.MTLSizeMake(work_items, 1, 1),
            self.metal.MTLSizeMake(
                min(work_items, pipeline.maxTotalThreadsPerThreadgroup()),
                1,
                1,
            ),
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()

    def _allocate(self, record: _TimeComponentRecord, accumulator) -> None:
        integer_arrays = {
            "inside_index": accumulator._inside_indices,
            "outside_index": accumulator._outside_indices,
            "source_patch_index": accumulator._source_patch_index,
            "integer_delay": accumulator._integer_delay.ravel(),
            "time_origin_steps": accumulator._time_origin_steps,
        }
        for name, values in integer_arrays.items():
            record.device[name] = self._buffer(self._to_int32(name, values))

        real_arrays = {
            "normal_spacing": accumulator._normal_spacing,
            "normal_derivative_weight": accumulator._normal_derivative_weight.ravel(),
            "field_weight": accumulator._field_weight.ravel(),
            "time_derivative_weight": accumulator._time_derivative_weight.ravel(),
            "fractional_delay": accumulator._fractional_delay.ravel(),
        }
        for name, values in real_arrays.items():
            record.device[name] = self._buffer(np.ascontiguousarray(values, dtype=self.real_dtype))

        empty = np.empty(record.nbase_patches, dtype=self.real_dtype)
        record.device["surface"] = [self._buffer(empty) for _ in range(3)]
        record.device["normal_derivative"] = [self._buffer(empty) for _ in range(3)]
        record.device["output"] = self._buffer(
            np.zeros(
                record.npoints * record.output_length,
                dtype=self.real_dtype,
            )
        )

    def _gather(self, record: _TimeComponentRecord, iteration: int) -> None:
        slot = iteration % 3
        field = getattr(self.grid, f"{record.component}_dev")
        command = self.queue.commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(self.gather_pipeline)
        self._set_scalar(encoder, 0, np.int32(record.nbase_patches))
        buffers = (
            record.device["inside_index"],
            record.device["outside_index"],
            record.device["normal_spacing"],
            field,
            record.device["surface"][slot],
            record.device["normal_derivative"][slot],
        )
        for index, buffer in enumerate(buffers, start=1):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        self._dispatch(command, encoder, self.gather_pipeline, record.nbase_patches)

    def _deposit(
        self,
        record: _TimeComponentRecord,
        sample_index: int,
        derivative_samples: tuple[int, int, int],
        derivative_coefficients: tuple[float, float, float],
    ) -> None:
        surface_slot = sample_index % 3
        time_surfaces = [record.device["surface"][sample % 3] for sample in derivative_samples]
        command = self.queue.commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(self.deposit_pipeline)
        for index, value in enumerate(
            (
                np.int32(record.npoints),
                np.int32(record.neffective_patches),
                np.int32(record.output_length),
                np.int32(sample_index),
            )
        ):
            self._set_scalar(encoder, index, value)
        initial_buffers = (
            record.device["surface"][surface_slot],
            record.device["normal_derivative"][surface_slot],
            time_surfaces[0],
            time_surfaces[1],
            time_surfaces[2],
        )
        for index, buffer in enumerate(initial_buffers, start=4):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        for index, value in enumerate(derivative_coefficients, start=9):
            self._set_scalar(encoder, index, self.real_scalar(value))
        final_buffers = (
            record.device["normal_derivative_weight"],
            record.device["field_weight"],
            record.device["time_derivative_weight"],
            record.device["source_patch_index"],
            record.device["integer_delay"],
            record.device["fractional_delay"],
            record.device["time_origin_steps"],
            record.device["output"],
        )
        for index, buffer in enumerate(final_buffers, start=12):
            encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        self._dispatch(command, encoder, self.deposit_pipeline, record.npoints)

    def _download_output(self, record: _TimeComponentRecord) -> npt.NDArray:
        count = record.npoints * record.output_length
        nbytes = count * self.real_dtype.itemsize
        return np.frombuffer(
            record.device["output"].contents().as_buffer(nbytes),
            dtype=self.real_dtype,
        ).copy()


class MetalCombinedKSIRCollector:
    """Dispatch frequency-domain and advanced-time monitors on Metal."""

    def __init__(self, updates):
        configure_ntff_monitors(updates.grid, allow_time_domain=True)
        frequency_monitors = [
            monitor for monitor in updates.grid.ntff_monitors if _is_frequency_monitor(monitor)
        ]
        time_monitors = [
            monitor for monitor in updates.grid.ntff_monitors if _is_time_domain_monitor(monitor)
        ]
        self.frequency = (
            MetalKSIRCollector(updates, frequency_monitors, configure=False)
            if frequency_monitors
            else None
        )
        self.time_domain = (
            MetalTimeDomainKSIRCollector(updates, time_monitors) if time_monitors else None
        )

    def observe_electric(self, iteration: int) -> None:
        if self.frequency is not None:
            self.frequency.observe_electric(iteration)
        if self.time_domain is not None:
            self.time_domain.observe_electric(iteration)

    def observe_magnetic(self, iteration: int) -> None:
        if self.frequency is not None:
            self.frequency.observe_magnetic(iteration)
        if self.time_domain is not None:
            self.time_domain.observe_magnetic(iteration)

    def finalise(self) -> None:
        if self.frequency is not None:
            self.frequency.finalise()
        if self.time_domain is not None:
            self.time_domain.finalise()
