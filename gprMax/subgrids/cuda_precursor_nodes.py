# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
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

"""Device-side precursor nodes for the HSG subgrid.

Subclasses the existing precursor classes rather than replacing them, so the
slice tables, field names and interpolation coordinates are inherited from the
shipped code instead of being transcribed. Only the per-timestep arithmetic is
overridden, and each override maps onto a device operation:

    gather_taps      slice extraction + FIR filter + transverse blend
    bilinear_interp  coarse main grid values -> fine subgrid resolution
    spline_rows/columns  separable higher-order FITPACK interpolation
    time_blend       previous/current main grid timestep weighting

Flat device descriptors are derived from the CPU's basic slices using scalar
offset/stride arithmetic; setup never allocates a main-grid-sized index array.
Higher-order interpolation uses precomputed separable FITPACK interpolation
matrices. Field gathering, interpolation and time blending stay on the GPU.

Collapsing the filter and the transverse blend:

    filtered H   f_1 = 0.25 u1 + 0.5 u2 + 0.25 u3
                 f_2 = 0.25 u2 + 0.5 u3 + 0.25 u4
                 f_t = c1 f_1 + c2 f_2
                     = 0.25 c1        u1
                     + (0.5 c1 + 0.25 c2) u2
                     + (0.25 c1 + 0.5 c2) u3
                     + 0.25 c2        u4

so one four-tap weighted sum covers the filtered and unfiltered classes and
both field types; the unused taps simply carry zero weight.
"""

import logging
from operator import index

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from gprMax.subgrids.precursor_nodes import (
    PrecursorNodes,
    PrecursorNodesEqualResolution,
    PrecursorNodesFiltered,
    calculate_weighting_coefficients,
)

logger = logging.getLogger(__name__)


def _descriptor(shape, slc):
    """Flat base offset, strides and extents for a numpy basic-slice tuple.

    Normalise the CPU's own basic slices just as NumPy does, without allocating
    or visiting any field cells. Offsets remain Python integers until upload.

    Args:
        shape: shape of the field array being sliced.
        slc: the slice tuple from the parent's slices table.

    Returns:
        (base, stride_a, stride_b, n_a, n_b)
    """
    if len(shape) != 3 or len(slc) != 3:
        raise ValueError(f"Expected a 3D field and basic-slice tuple, got {shape}, {slc}")
    shape = tuple(index(size) for size in shape)
    strides = (shape[1] * shape[2], shape[2], 1)
    base, axes = 0, []
    for size, stride, selection in zip(shape, strides, slc):
        if isinstance(selection, slice):
            start, stop, step = selection.indices(size)
            count = len(range(start, stop, step))
            if count == 0:
                raise ValueError(f"Empty precursor slice: {slc}")
            base += start * stride
            axes.append((count, step * stride if count > 1 else 0))
        else:
            position = index(selection)
            if position < 0:
                position += size
            if not 0 <= position < size:
                raise IndexError(f"Precursor index {selection} outside axis of size {size}")
            base += position * stride
    if len(axes) != 2:
        raise ValueError(f"Precursor slice is not 2D: {slc}")
    (n_a, stride_a), (n_b, stride_b) = axes
    return base, stride_a, stride_b, n_a, n_b


def _spline_axis_weights(source, target, order):
    """Linear map from samples to FITPACK spline values on one fixed axis.

    RectBivariateSpline is a tensor product of interpolating splines. Evaluate
    the same univariate FITPACK interpolant on an identity basis once at setup,
    including its constant extension outside the sampled interval. A local
    polynomial stencil would NOT reproduce the CPU's global spline fit.
    """
    basis = np.eye(len(source))
    weights = [InterpolatedUnivariateSpline(source, row, k=order, ext=3)(target) for row in basis]
    return np.ascontiguousarray(np.asarray(weights).T)


def _axis_weights(src, tgt):
    """Index and weight arrays for a 4-tap bilinear gather.

    The index is clamped to [0, n-2] and the weight to [0, 1]. Clamping the
    weight is what reproduces FITPACK: RectBivariateSpline with kx=ky=1 holds
    the boundary value outside the source range rather than extrapolating
    linearly. The mid=1 coordinate sets do fall outside it on every face, so
    this is the common case, not an edge case.
    """
    n = len(src)
    h = src[1] - src[0]
    p = np.clip(np.floor((tgt - src[0]) / h).astype(np.int32), 0, n - 2)
    w = np.clip((tgt - src[p]) / h, 0.0, 1.0)
    return np.ascontiguousarray(p, dtype=np.int32), np.ascontiguousarray(w)


class CUDAPrecursorMixin:
    """Per-timestep precursor arithmetic, executed on the device.

    Mixed in ahead of PrecursorNodes or PrecursorNodesFiltered; everything
    structural - slice tables, field names, interpolation coordinates - comes
    from the parent unchanged.
    """

    # Number of taps the parent's slice rows carry. The filtered classes hold
    # four index tuples for H and three for E; the plain ones hold two and
    # one. Set by the concrete subclass.
    N_TAPS_H = 2
    N_TAPS_E = 1

    def setup_device(self, kernels, gpuarray, tpb=128):
        """Allocate device buffers and precompute launch descriptors.

        Called once, after the parent constructor has built the slice tables
        and interpolation coordinates.

        Args:
            kernels: built gather/time kernels and the selected interpolation
                        kernels (bilinear or separable spline).
            gpuarray: the pycuda.gpuarray module.
            tpb: threads per block.
        """
        self.k_gather = kernels["gather_taps"]
        self.k_interp = kernels.get("bilinear_interp")
        self.k_rows = kernels.get("spline_rows")
        self.k_columns = kernels.get("spline_columns")
        self.k_blend = kernels["time_blend"]
        self.gpuarray = gpuarray
        self.tpb = tpb
        self.real = self.dtype.type

        # Main grid field arrays, resident on the device. The precursors read
        # them; the main grid's own updates write them.
        self.main_dev = {
            "Ex": self.G.Ex_dev,
            "Ey": self.G.Ey_dev,
            "Ez": self.G.Ez_dev,
            "Hx": self.G.Hx_dev,
            "Hy": self.G.Hy_dev,
            "Hz": self.G.Hz_dev,
        }
        self._field_id = {id(getattr(self.G, n)): n for n in self.main_dev}

        self.dev = {}  # name -> current-value buffer (fine)
        self.dev_0 = {}  # name -> previous main timestep (fine)
        self.dev_1 = {}  # name -> current main timestep (fine)
        self.scratch = {}  # name -> coarse gather output
        self.desc = {}  # name -> launch descriptor
        self.spline_scratch = {}

        self._prepare(self.magnetic_slices, self.N_TAPS_H)
        self._prepare(self.electric_slices, self.N_TAPS_E)

    def _prepare(self, slices, n_taps):
        """Build descriptors and buffers for one set of slice rows.

        A row is [name_1, coords, tap_0, ..., tap_{n-1}, field_array].
        """
        for obj in slices:
            name = obj[0][:-2]  # strip the trailing "_1"
            coords = obj[1]
            taps = obj[2 : 2 + n_taps]
            field = obj[-1]

            bases, sa, sb, n_a, n_b = [], None, None, None, None
            for slc in taps:
                base, s_a, s_b, na, nb = _descriptor(field.shape, slc)
                bases.append(base)
                if sa is None:
                    sa, sb, n_a, n_b = s_a, s_b, na, nb
                elif (s_a, s_b, na, nb) != (sa, sb, n_a, n_b):
                    logger.exception(f"{name}: taps disagree on stride or extent")
                    raise ValueError
            # Unused taps repeat the first base; their weight is zero
            while len(bases) < 4:
                bases.append(bases[0])

            fine = getattr(self, f"{name}_1")
            shape = (n_a, n_b) if coords is None else (len(coords[2]), len(coords[3]))
            if fine.shape != shape:
                raise ValueError(f"{name}: interpolated shape {shape} does not match precursor {fine.shape}")

            self.desc[name] = {
                "src": self._field_id[id(field)],
                "bases": bases,
                "stride_a": sa,
                "stride_b": sb,
                "n_a": n_a,
                "n_b": n_b,
                "N_a": shape[0],
                "N_b": shape[1],
            }
            d = self.desc[name]
            if coords is not None:
                x, z, x_sg, z_sg = coords
                if self.interpolation == 1:
                    p, wx = _axis_weights(x, x_sg)
                    q, wz = _axis_weights(z, z_sg)
                    d.update(
                        {
                            key: self.gpuarray.to_gpu(values)
                            for key, values in (("p", p), ("wx", wx), ("q", q), ("wz", wz))
                        }
                    )
                else:
                    d["wx"] = self.gpuarray.to_gpu(_spline_axis_weights(x, x_sg, self.interpolation))
                    d["wz"] = self.gpuarray.to_gpu(_spline_axis_weights(z, z_sg, self.interpolation))
                    self.spline_scratch[name] = self.gpuarray.zeros((shape[0], n_b), fine.dtype)

            self.dev[name] = self.gpuarray.to_gpu(np.ascontiguousarray(fine))
            self.dev_0[name] = self.gpuarray.to_gpu(np.ascontiguousarray(fine))
            self.dev_1[name] = self.gpuarray.to_gpu(np.ascontiguousarray(fine))
            # Ratio one gathers directly into the current precursor buffer:
            # there is no interpolation, filtering or extra device copy.
            self.scratch[name] = self.dev_1[name] if coords is None else self.gpuarray.zeros((n_a, n_b), fine.dtype)

    # ------------------------------------------------------------ launching
    def _grid(self, total):
        return ((total + self.tpb - 1) // self.tpb, 1, 1)

    def _gather(self, name, weights):
        """Slice, filter and transverse-blend into the coarse scratch buffer."""
        d = self.desc[name]
        w = list(weights) + [0.0] * (4 - len(weights))
        total = d["n_a"] * d["n_b"]

        self.k_gather(
            np.int32(d["n_a"]),
            np.int32(d["n_b"]),
            np.int64(d["bases"][0]),
            np.int64(d["bases"][1]),
            np.int64(d["bases"][2]),
            np.int64(d["bases"][3]),
            np.int64(d["stride_a"]),
            np.int64(d["stride_b"]),
            self.real(w[0]),
            self.real(w[1]),
            self.real(w[2]),
            self.real(w[3]),
            self.main_dev[d["src"]].gpudata,
            self.scratch[name].gpudata,
            block=(self.tpb, 1, 1),
            grid=self._grid(total),
        )

    def _interpolate(self, name):
        """Coarse scratch -> fine _1 buffer."""
        d = self.desc[name]
        total = d["N_a"] * d["N_b"]
        if self.interpolation == 0:
            return
        if self.interpolation > 1:
            intermediate = self.spline_scratch[name]
            self.k_rows(
                np.int32(d["N_a"]),
                np.int32(d["n_a"]),
                np.int32(d["n_b"]),
                d["wx"].gpudata,
                self.scratch[name].gpudata,
                intermediate.gpudata,
                block=(self.tpb, 1, 1),
                grid=self._grid(intermediate.size),
            )
            self.k_columns(
                np.int32(d["N_a"]),
                np.int32(d["N_b"]),
                np.int32(d["n_b"]),
                d["wz"].gpudata,
                intermediate.gpudata,
                self.dev_1[name].gpudata,
                block=(self.tpb, 1, 1),
                grid=self._grid(total),
            )
            return

        self.k_interp(
            np.int32(d["N_a"]),
            np.int32(d["N_b"]),
            np.int32(d["n_b"]),
            d["p"].gpudata,
            d["wx"].gpudata,
            d["q"].gpudata,
            d["wz"].gpudata,
            self.scratch[name].gpudata,
            self.dev_1[name].gpudata,
            block=(self.tpb, 1, 1),
            grid=self._grid(total),
        )

    def _blend(self, name, c1, c2):
        """Weight the previous and current main grid timesteps."""
        n = self.dev[name].size
        self.k_blend(
            np.int32(n),
            self.real(c1),
            self.real(c2),
            self.dev_0[name].gpudata,
            self.dev_1[name].gpudata,
            self.dev[name].gpudata,
            block=(self.tpb, 1, 1),
            grid=self._grid(n),
        )

    # ------------------------------------------------------------ overrides
    #
    # Same entry points as the CPU precursors, same call order. The phase
    # logic in HSGPhaseMixin cannot tell the difference.

    def update_previous_timestep_fields(self, field_names):
        for fn in field_names:
            self.dev_0[fn].set(self.dev_1[fn])

    def update_magnetic(self):
        self.update_previous_timestep_fields(self.fn_m)

        for obj in self.magnetic_slices:
            name = obj[0][:-2]
            w = self.l_weight if ("left" in name or "bottom" in name or "front" in name) else self.r_weight
            c1, c2 = calculate_weighting_coefficients(w, self.ratio)
            self._gather(name, self._h_weights(c1, c2))
            self._interpolate(name)

    def update_electric(self):
        self.update_previous_timestep_fields(self.fn_e)

        for obj in self.electric_slices:
            name = obj[0][:-2]
            self._gather(name, self._e_weights())
            self._interpolate(name)

    def weight_pre_and_current_fields(self, m, field_names):
        c1, c2 = calculate_weighting_coefficients(m, self.ratio)
        for fn in field_names:
            self._blend(fn, c1, c2)

    def calc_exact_field(self, field_names):
        for fn in field_names:
            self.dev[fn].set(self.dev_1[fn])

    # ------------------------------------------------------------ accessors
    def device_slices(self):
        """{name: (device pointer, row length)} for the IS launches."""
        return {n: (buf.gpudata, self.desc[n]["N_b"]) for n, buf in self.dev.items()}

    def download(self, name):
        """Copy one precursor slice back to the host. Debugging only."""
        return self.dev[name].get()


class CUDAPrecursorNodes(CUDAPrecursorMixin, PrecursorNodes):
    """Unfiltered precursors on the GPU.

    H: f_t = c1*u1 + c2*u2      E: f_m = u1
    """

    N_TAPS_H = 2
    N_TAPS_E = 1

    @staticmethod
    def _h_weights(c1, c2):
        return (c1, c2)

    @staticmethod
    def _e_weights():
        return (1.0,)


class CUDAPrecursorNodesEqualResolution(CUDAPrecursorMixin, PrecursorNodesEqualResolution):
    """Select exactly co-located Yee samples, without spline or FIR filtering."""

    @staticmethod
    def _h_weights(c1, c2):
        # ratio=1 gives (1, 0) on lower faces and (0, 1) on upper faces.
        return (c1, c2)

    @staticmethod
    def _e_weights():
        return (1.0,)


class CUDAPrecursorNodesFiltered(CUDAPrecursorMixin, PrecursorNodesFiltered):
    """Filtered precursors on the GPU - the default path.

    The 3-tap FIR filter and the transverse blend are folded into one
    four-tap weighted sum; see the module docstring for the derivation.
    """

    N_TAPS_H = 4
    N_TAPS_E = 3

    @staticmethod
    def _h_weights(c1, c2):
        return (0.25 * c1, 0.5 * c1 + 0.25 * c2, 0.25 * c1 + 0.5 * c2, 0.25 * c2)

    @staticmethod
    def _e_weights():
        return (0.25, 0.5, 0.25)
