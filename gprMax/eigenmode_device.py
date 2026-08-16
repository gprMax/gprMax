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

"""Backend-neutral packing for device-resident eigenmode operations."""

from __future__ import annotations

import numpy as np

from gprMax import config
from gprMax.eigenmode_ports import DFT_PHASE_REANCHOR_INTERVAL, _dft_phase_at_time


def _padded_components(components, nu, nv, dtype):
    result = np.zeros((3, nu + 1, nv + 1), dtype=dtype)
    for component, values in enumerate(components):
        values = np.asarray(values, dtype=dtype)
        if values.ndim != 2 or values.shape[0] > nu + 1 or values.shape[1] > nv + 1:
            raise ValueError(
                "Eigenmode source profile shape does not fit its transverse aperture: "
                f"component {component} has shape {values.shape}, maximum is "
                f"{(nu + 1, nv + 1)}."
            )
        result[component, : values.shape[0], : values.shape[1]] = values
    return result


def eigenmode_source_profiles(source):
    """Return packed E/H profile bases used by one modal source."""

    dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
    nu, nv = np.asarray(source.transverse_stop) - np.asarray(source.transverse_start)
    if source.broadband_e_envelopes is None:
        electric = [_padded_components(source.modal_e_real, nu, nv, dtype)]
        magnetic = [_padded_components(source.modal_h_real, nu, nv, dtype)]
    else:
        electric = []
        magnetic = []
        for quadrature, fields in enumerate(
            (source.broadband_modal_e_real, source.broadband_modal_e_imag)
        ):
            for anchor in range(source.broadband_e_envelopes.shape[0]):
                electric.append(_padded_components(fields[anchor], nu, nv, dtype))
        for quadrature, fields in enumerate(
            (source.broadband_modal_h_real, source.broadband_modal_h_imag)
        ):
            for anchor in range(source.broadband_h_envelopes.shape[0]):
                magnetic.append(_padded_components(fields[anchor], nu, nv, dtype))
    return np.ascontiguousarray(electric), np.ascontiguousarray(magnetic)


def eigenmode_source_envelopes(source, grid, iteration, magnetic):
    """Yield ``(basis, envelope)`` pairs active at one FDTD update."""

    envelopes = source.broadband_e_envelopes if magnetic else source.broadband_h_envelopes
    if envelopes is not None:
        if iteration >= envelopes.shape[2]:
            return
        anchor_count = envelopes.shape[0]
        for quadrature in range(2):
            for anchor in range(anchor_count):
                value = float(envelopes[anchor, quadrature, iteration])
                if value != 0:
                    yield quadrature * anchor_count + anchor, value
        return

    time = iteration * grid.dt
    if not magnetic:
        time += source._magnetic_modal_time_offset(grid)
    if source._source_is_active(time):
        yield 0, float(source._waveform_value(time, grid))


def upload_device_arrays(arrays, backend, *, queue=None, dev=None):
    """Upload named contiguous arrays using the selected backend."""

    if backend == "cuda":
        import pycuda.gpuarray as gpuarray

        return {
            name: gpuarray.to_gpu(np.ascontiguousarray(array)) for name, array in arrays.items()
        }
    if backend == "opencl":
        import pyopencl.array as clarray

        return {
            name: clarray.to_device(queue, np.ascontiguousarray(array))
            for name, array in arrays.items()
        }
    if backend == "metal":
        return {
            name: dev.newBufferWithBytes_length_options_(
                np.ascontiguousarray(array).tobytes(), array.nbytes, 0
            )
            for name, array in arrays.items()
        }
    raise ValueError(f"Unknown device backend {backend!r}.")


def prepare_device_eigenmode_source(source, backend, *, queue=None, dev=None):
    electric, magnetic = eigenmode_source_profiles(source)
    if max(electric.size, magnetic.size) > np.iinfo(np.int32).max:
        raise ValueError("Eigenmode source profiles exceed the signed 32-bit index range.")
    uploaded = upload_device_arrays(
        {"electric": electric, "magnetic": magnetic},
        backend,
        queue=queue,
        dev=dev,
    )
    source.device_electric_profiles = uploaded["electric"]
    source.device_magnetic_profiles = uploaded["magnetic"]


def prepare_device_eigenmode_monitor(monitor, backend, *, queue=None, dev=None):
    """Upload one monitor's modal bases, phases, and DFT accumulators."""

    dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
    complex_arrays = {
        "electric_phase": monitor.electric_phase,
        "magnetic_phase": monitor.magnetic_phase,
        "phase_step": monitor.phase_step,
        "conj_eu": monitor.conj_eu,
        "conj_ev": monitor.conj_ev,
        "conj_hu": monitor.conj_hu,
        "conj_hv": monitor.conj_hv,
        "electric_dft": monitor.electric_dft,
        "magnetic_dft": monitor.magnetic_dft,
    }
    if max(np.asarray(values).size for values in complex_arrays.values()) > np.iinfo(np.int32).max:
        raise ValueError("Eigenmode monitor arrays exceed the signed 32-bit index range.")
    arrays = {}
    for name, values in complex_arrays.items():
        arrays[f"{name}_real"] = np.ascontiguousarray(np.real(values), dtype=dtype)
        arrays[f"{name}_imag"] = np.ascontiguousarray(np.imag(values), dtype=dtype)
    monitor.device_arrays = upload_device_arrays(arrays, backend, queue=queue, dev=dev)
    if backend == "metal":
        monitor.device_parameters = np.zeros(
            1,
            dtype=np.dtype(
                [
                    ("NF", np.int32),
                    ("NM", np.int32),
                    ("normal_axis", np.int32),
                    ("direction_sign", np.int32),
                    ("magnetic_side", np.int32),
                    ("u0", np.int32),
                    ("v0", np.int32),
                    ("u1", np.int32),
                    ("v1", np.int32),
                    ("plane_index", np.int32),
                    ("dt", dtype),
                    ("measure", dtype),
                    ("handedness", np.int32),
                ],
                align=True,
            ),
        )
        if monitor.device_parameters.nbytes != 52:
            raise RuntimeError("Unexpected Metal eigenmode-parameter structure layout.")


def _device_array_to_host(value, backend, shape, dtype):
    if backend in ("cuda", "opencl"):
        return value.get().reshape(shape)
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    return np.frombuffer(value.contents().as_buffer(nbytes), dtype=dtype).reshape(shape).copy()


def finalise_device_eigenmode_monitors(monitors, backend):
    """Copy only completed modal DFTs back to their monitor objects."""

    dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
    for monitor in monitors:
        arrays = monitor.device_arrays
        shape = monitor.electric_dft.shape
        electric = _device_array_to_host(
            arrays["electric_dft_real"], backend, shape, dtype
        ) + 1j * _device_array_to_host(arrays["electric_dft_imag"], backend, shape, dtype)
        magnetic = _device_array_to_host(
            arrays["magnetic_dft_real"], backend, shape, dtype
        ) + 1j * _device_array_to_host(arrays["magnetic_dft_imag"], backend, shape, dtype)
        np.copyto(monitor.electric_dft, electric, casting="same_kind")
        np.copyto(monitor.magnetic_dft, magnetic, casting="same_kind")


def _replace_device_array(array, values, backend):
    values = np.ascontiguousarray(values)
    if backend in ("cuda", "opencl"):
        array.set(values)
        return
    view = np.frombuffer(array.contents().as_buffer(values.nbytes), dtype=values.dtype).reshape(
        values.shape
    )
    np.copyto(view, values)


def reanchor_device_eigenmode_monitor(monitor, grid, backend):
    """Periodically reset recursively advanced device DFT phases exactly."""

    if monitor._next_iteration % DFT_PHASE_REANCHOR_INTERVAL:
        return
    dtype = np.dtype(config.sim_config.dtypes["float_or_double"])
    electric = _dft_phase_at_time(
        monitor.frequency,
        monitor._next_iteration * grid.dt,
        monitor.electric_phase.dtype,
    )
    magnetic = _dft_phase_at_time(
        monitor.frequency,
        (monitor._next_iteration + 0.5) * grid.dt,
        monitor.magnetic_phase.dtype,
    )
    arrays = monitor.device_arrays
    _replace_device_array(
        arrays["electric_phase_real"], np.asarray(electric.real, dtype=dtype), backend
    )
    _replace_device_array(
        arrays["electric_phase_imag"], np.asarray(electric.imag, dtype=dtype), backend
    )
    _replace_device_array(
        arrays["magnetic_phase_real"], np.asarray(magnetic.real, dtype=dtype), backend
    )
    _replace_device_array(
        arrays["magnetic_phase_imag"], np.asarray(magnetic.imag, dtype=dtype), backend
    )
