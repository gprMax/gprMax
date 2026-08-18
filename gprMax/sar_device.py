# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax.

"""Sparse device-resident electric-field DFT collection for SAR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gprMax.cuda_opencl.knl_sar import build_sar_kernel_source

COMPONENTS = ("Ex", "Ey", "Ez")


@dataclass
class _ComponentRecord:
    component: str
    indices: np.ndarray
    device: dict

    @property
    def nedges(self) -> int:
        return int(self.indices.size)


@dataclass
class _MonitorRecord:
    monitor: object
    components: tuple[_ComponentRecord, ...]
    device: dict

    @property
    def nfrequencies(self) -> int:
        return int(self.monitor.frequencies.size)


class _DeviceSARCollector:
    """Backend-neutral sequencing for sparse electric-field DFTs."""

    backend = "device"

    def __init__(self, updates):
        self.updates = updates
        self.grid = updates.grid
        self.records = []
        limits = np.iinfo(np.int32)
        for monitor in self.grid.sar_monitors:
            components = []
            for component in COMPONENTS:
                indices = np.asarray(monitor.edge_flat_indices[component])
                if indices.size > limits.max or np.any(indices < 0) or np.any(indices > limits.max):
                    raise ValueError("SAR device edge indices exceed signed int32 indexing")
                total = int(indices.size) * int(monitor.frequencies.size)
                if total > limits.max:
                    raise ValueError("SAR device work-item count exceeds signed int32 indexing")
                components.append(
                    _ComponentRecord(
                        component=component,
                        indices=np.ascontiguousarray(indices, dtype=np.int32),
                        device={},
                    )
                )
            record = _MonitorRecord(monitor=monitor, components=tuple(components), device={})
            self._allocate(record)
            self.records.append(record)
            monitor.collection_backend = f"{self.backend}_device"

    def _allocate(self, record: _MonitorRecord) -> None:
        raise NotImplementedError

    def _upload_multiplier(self, record: _MonitorRecord, multiplier) -> None:
        raise NotImplementedError

    def _accumulate(self, record: _MonitorRecord, component: _ComponentRecord, field) -> None:
        raise NotImplementedError

    def _download(self, record: _MonitorRecord, component: _ComponentRecord):
        raise NotImplementedError

    def observe_electric(self, iteration: int) -> None:
        for record in self.records:
            multiplier = record.monitor.device_sampling_multiplier(iteration)
            self._upload_multiplier(record, multiplier)
            for component in record.components:
                self._accumulate(
                    record,
                    component,
                    getattr(self.grid, f"{component.component}_dev"),
                )

    def finalise(self) -> None:
        for record in self.records:
            for component in record.components:
                real, imag = self._download(record, component)
                values = np.empty(
                    (record.nfrequencies, component.nedges),
                    dtype=record.monitor.complex_dtype,
                )
                values.real = np.asarray(real).reshape(values.shape)
                values.imag = np.asarray(imag).reshape(values.shape)
                record.monitor.load_device_component_dfts(component.component, values)


class CUDASARCollector(_DeviceSARCollector):
    """CUDA sparse SAR DFT collector."""

    backend = "cuda"
    threads = 128

    def __init__(self, updates):
        self.real_dtype = np.dtype(updates.grid.Ex.dtype)
        source = build_sar_kernel_source("cuda", updates.subs_name_args["REAL"])
        module = updates.source_module(
            source, options=getattr(updates, "ntff_compiler_options", None)
        )
        self.kernel = module.get_function("accumulate_sar")
        self.gpuarray = updates.grid.gpuarray
        super().__init__(updates)

    def _allocate(self, record):
        record.device["multiplier_real"] = self.gpuarray.empty(record.nfrequencies, self.real_dtype)
        record.device["multiplier_imag"] = self.gpuarray.empty(record.nfrequencies, self.real_dtype)
        for component in record.components:
            component.device["indices"] = self.gpuarray.to_gpu(component.indices)
            total = record.nfrequencies * component.nedges
            component.device["real"] = self.gpuarray.zeros(total, self.real_dtype)
            component.device["imag"] = self.gpuarray.zeros(total, self.real_dtype)

    def _upload_multiplier(self, record, multiplier):
        record.device["multiplier_real"].set(
            np.ascontiguousarray(multiplier.real, dtype=self.real_dtype)
        )
        record.device["multiplier_imag"].set(
            np.ascontiguousarray(multiplier.imag, dtype=self.real_dtype)
        )

    def _accumulate(self, record, component, field):
        total = record.nfrequencies * component.nedges
        self.kernel(
            np.int32(total),
            np.int32(component.nedges),
            component.device["indices"].gpudata,
            record.device["multiplier_real"].gpudata,
            record.device["multiplier_imag"].gpudata,
            field.gpudata,
            component.device["real"].gpudata,
            component.device["imag"].gpudata,
            block=(self.threads, 1, 1),
            grid=((total + self.threads - 1) // self.threads, 1, 1),
        )

    def _download(self, record, component):
        return component.device["real"].get(), component.device["imag"].get()


class OpenCLSARCollector(_DeviceSARCollector):
    """OpenCL sparse SAR DFT collector."""

    backend = "opencl"

    def __init__(self, updates):
        self.real_dtype = np.dtype(updates.grid.Ex.dtype)
        source = build_sar_kernel_source(
            "opencl", "double" if self.real_dtype.itemsize == 8 else "float"
        )
        options = getattr(updates, "ntff_compiler_options", None)
        self.kernel = updates.cl.Program(updates.ctx, source).build(options=options).accumulate_sar
        self.clarray = updates.grid.clarray
        self.queue = updates.queue
        super().__init__(updates)

    def _allocate(self, record):
        record.device["multiplier_real"] = self.clarray.empty(
            self.queue, record.nfrequencies, self.real_dtype
        )
        record.device["multiplier_imag"] = self.clarray.empty(
            self.queue, record.nfrequencies, self.real_dtype
        )
        for component in record.components:
            component.device["indices"] = self.clarray.to_device(self.queue, component.indices)
            total = record.nfrequencies * component.nedges
            component.device["real"] = self.clarray.zeros(self.queue, total, self.real_dtype)
            component.device["imag"] = self.clarray.zeros(self.queue, total, self.real_dtype)

    def _upload_multiplier(self, record, multiplier):
        record.device["multiplier_real"].set(
            np.ascontiguousarray(multiplier.real, dtype=self.real_dtype),
            queue=self.queue,
        )
        record.device["multiplier_imag"].set(
            np.ascontiguousarray(multiplier.imag, dtype=self.real_dtype),
            queue=self.queue,
        )

    def _accumulate(self, record, component, field):
        total = record.nfrequencies * component.nedges
        self.kernel(
            self.queue,
            (total,),
            None,
            np.int32(total),
            np.int32(component.nedges),
            component.device["indices"].data,
            record.device["multiplier_real"].data,
            record.device["multiplier_imag"].data,
            np.int32(field.offset // field.dtype.itemsize),
            field.base_data,
            component.device["real"].data,
            component.device["imag"].data,
        )

    def _download(self, record, component):
        return (
            component.device["real"].get(queue=self.queue),
            component.device["imag"].get(queue=self.queue),
        )


class MetalSARCollector(_DeviceSARCollector):
    """Metal sparse SAR DFT collector using shared buffers."""

    backend = "metal"

    def __init__(self, updates):
        self.real_dtype = np.dtype(updates.grid.Ex.dtype)
        source = build_sar_kernel_source("metal", updates.subs_name_args["REAL"])
        library, error = updates.dev.newLibraryWithSource_options_error_(source, updates.opts, None)
        if library is None:
            raise RuntimeError(f"Failed to compile Metal SAR kernel: {error}")
        function = library.newFunctionWithName_("accumulate_sar")
        self.pipeline = updates.dev.newComputePipelineStateWithFunction_error_(function, None)[0]
        self.dev = updates.dev
        self.queue = updates.cmdqueue
        self.metal = updates.metal
        self.storage = getattr(updates.grid, "storage", 0)
        super().__init__(updates)

    def _buffer(self, values):
        contiguous = np.ascontiguousarray(values)
        return self.dev.newBufferWithBytes_length_options_(
            contiguous, contiguous.nbytes, self.storage
        )

    def _allocate(self, record):
        record.device["multiplier_real"] = self._buffer(
            np.empty(record.nfrequencies, dtype=self.real_dtype)
        )
        record.device["multiplier_imag"] = self._buffer(
            np.empty(record.nfrequencies, dtype=self.real_dtype)
        )
        for component in record.components:
            component.device["indices"] = self._buffer(component.indices)
            zeros = np.zeros(record.nfrequencies * component.nedges, dtype=self.real_dtype)
            component.device["real"] = self._buffer(zeros)
            component.device["imag"] = self._buffer(zeros)

    def _upload_multiplier(self, record, multiplier):
        nbytes = record.nfrequencies * self.real_dtype.itemsize
        real = np.frombuffer(
            record.device["multiplier_real"].contents().as_buffer(nbytes),
            dtype=self.real_dtype,
        )
        imag = np.frombuffer(
            record.device["multiplier_imag"].contents().as_buffer(nbytes),
            dtype=self.real_dtype,
        )
        real[:] = np.asarray(multiplier.real, dtype=self.real_dtype)
        imag[:] = np.asarray(multiplier.imag, dtype=self.real_dtype)

    def _accumulate(self, record, component, field):
        total = record.nfrequencies * component.nedges
        command = self.queue.commandBuffer()
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(self.pipeline)
        scalar_buffers = (
            self._buffer(np.asarray((total,), dtype=np.int32)),
            self._buffer(np.asarray((component.nedges,), dtype=np.int32)),
        )
        buffers = (
            *scalar_buffers,
            component.device["indices"],
            record.device["multiplier_real"],
            record.device["multiplier_imag"],
            field,
            component.device["real"],
            component.device["imag"],
        )
        for index, buffer in enumerate(buffers):
            if hasattr(buffer, "buffer") and hasattr(buffer, "offset"):
                encoder.setBuffer_offset_atIndex_(buffer.buffer, buffer.offset, index)
            else:
                encoder.setBuffer_offset_atIndex_(buffer, 0, index)
        encoder.dispatchThreads_threadsPerThreadgroup_(
            self.metal.MTLSizeMake(total, 1, 1),
            self.metal.MTLSizeMake(min(total, self.pipeline.maxTotalThreadsPerThreadgroup()), 1, 1),
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()

    def _download(self, record, component):
        total = record.nfrequencies * component.nedges
        nbytes = total * self.real_dtype.itemsize
        return tuple(
            np.frombuffer(
                component.device[name].contents().as_buffer(nbytes),
                dtype=self.real_dtype,
            ).copy()
            for name in ("real", "imag")
        )
