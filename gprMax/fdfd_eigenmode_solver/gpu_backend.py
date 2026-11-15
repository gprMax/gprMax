# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
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
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.
"""Experimental GPU backend for FDFD eigensolves using cuPy.

This module isolates every device-specific operation behind two small
functions so the FDFD mode solvers stay importable and correct on hosts
without cuPy or an NVIDIA GPU:

``gpu_available()``
    Whether a usable cuPy installation was found.

``gpu_eigs(...)``
    Run a shift-invert Arnoldi eigensolve on the device for a plain
    sparse operator, returning host NumPy arrays.

Both functions raise ``RuntimeError`` when the GPU path cannot be used.
Callers that opt in with ``use_gpu=True`` catch that error, warn, and
transparently fall back to the established CPU scipy path, so enabling
the flag can never change numerical results, only where the matvecs run.
"""

import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    import cupyx.scipy.sparse.linalg as cp_sparse_linalg
except ImportError:
    cp = None
    cp_sparse = None
    cp_sparse_linalg = None


def gpu_available():
    """Return ``True`` when cuPy is importable and a device is usable."""
    if cp is None:
        return False
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        return False
    return True


def _device_csr(operator):
    import scipy.sparse as sp

    if not sp.issparse(operator):
        raise RuntimeError(
            "GPU eigensolve requires a plain sparse operator; "
            "matrix-free operators stay on the CPU path."
        )
    csr = operator.tocsr()
    return cp_sparse.csr_matrix(
        (csr.data, csr.indices, csr.indptr), shape=csr.shape
    )


def gpu_eigs(operator, num_modes, sigma, v0=None, ncv=None):
    """Solve an eigenvalue problem on the GPU with shift-invert Arnoldi.

    Mirrors the CPU ``scipy.sparse.linalg.eigs`` call used by the FDFD
    mode solvers. ``operator`` must be a plain scipy sparse matrix; the
    matrix is transferred once and every Arnoldi matvec runs on device.

    Raises ``RuntimeError`` for any GPU-side failure so callers can fall
    back to the CPU path.
    """
    if not gpu_available():
        raise RuntimeError("cuPy or a CUDA device is unavailable.")
    try:
        device_operator = _device_csr(operator)
        device_v0 = None if v0 is None else cp.asarray(v0)
        values, vectors = cp_sparse_linalg.eigs(
            device_operator,
            k=num_modes,
            sigma=sigma,
            which="LM",
            v0=device_v0,
            ncv=ncv,
            return_eigenvectors=True,
        )
        return cp.asnumpy(values), cp.asnumpy(vectors)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"GPU eigensolve failed: {exc}") from exc
