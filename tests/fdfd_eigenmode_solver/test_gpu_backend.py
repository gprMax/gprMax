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
"""GPU backend availability and CPU-fallback behaviour for FDFD mode solvers.

Enabling ``use_gpu=True`` must never change numerical results: when cuPy or
a CUDA device is missing, or a device solve fails, the solvers warn and fall
back to the established CPU path. These tests force every unavailable and
failing GPU state with monkeypatching so they behave identically on hosts
with and without accelerators.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse

import gprMax.config as config
import gprMax.fdfd_eigenmode_solver.gpu_backend as gpu_backend
from gprMax.fdfd_eigenmode_solver.fdfd_1d_mode_solver import FDFD_1D_mode_solver
from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver


@pytest.fixture(autouse=True)
def _solver_constants(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            em_consts={
                "e0": 8.8541878128e-12,
                "m0": 1.25663706212e-6,
                "c": 299792458.0,
                "z0": 376.73031366686166,
            }
        ),
    )


@pytest.fixture(autouse=True)
def _no_cupy(monkeypatch):
    monkeypatch.setattr(gpu_backend, "cp", None)


def _working_cupy_stub():
    return SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 1))
    )


def test_gpu_available_false_without_cupy():
    assert gpu_backend.gpu_available() is False


def test_gpu_available_false_when_device_probe_fails(monkeypatch):
    monkeypatch.setattr(gpu_backend, "cp", _working_cupy_stub())

    def raise_no_driver():
        raise RuntimeError("no CUDA driver")

    monkeypatch.setattr(
        gpu_backend.cp.cuda, "runtime", SimpleNamespace(getDeviceCount=raise_no_driver)
    )
    assert gpu_backend.gpu_available() is False


def test_gpu_available_true_with_working_stub(monkeypatch):
    monkeypatch.setattr(gpu_backend, "cp", _working_cupy_stub())
    assert gpu_backend.gpu_available() is True


def test_gpu_eigs_requires_cupy():
    operator = scipy.sparse.identity(6, format="csr", dtype=complex)
    with pytest.raises(RuntimeError, match="unavailable"):
        gpu_backend.gpu_eigs(operator, 1, -1.0)


def test_gpu_eigs_rejects_matrix_free_operators(monkeypatch):
    monkeypatch.setattr(gpu_backend, "cp", _working_cupy_stub())
    from scipy.sparse.linalg import LinearOperator

    matrix_free = LinearOperator(
        (4, 4), matvec=lambda values: values, dtype=np.complex128
    )
    with pytest.raises(RuntimeError, match="matrix-free"):
        gpu_backend.gpu_eigs(matrix_free, 1, -1.0)


def _tm_1d_solver(**kwargs):
    n = 60
    defaults = dict(
        frequency=10e9,
        dt=0.5e-3,
        mode_index=0,
        polarization="TM",
        eps_r_t=np.ones(n),
        eps_r_a=np.ones(n + 1),
        eps_r_w=np.ones(n + 1),
        mu_r_t=np.ones(n + 1),
        mu_r_a=np.ones(n),
        mu_r_w=np.ones(n),
    )
    defaults.update(kwargs)
    return FDFD_1D_mode_solver(**defaults)


def _pec_2d_solver(**kwargs):
    nu, nv = 8, 6
    epsilon = 2.0
    eu = np.full((nu, nv + 1), epsilon, dtype=complex)
    ev = np.full((nu + 1, nv), epsilon, dtype=complex)
    ew = np.full((nu + 1, nv + 1), epsilon, dtype=complex)
    eu[:, [0, -1]] = np.inf
    ev[[0, -1], :] = np.inf
    ew[:, [0, -1]] = np.inf
    ew[[0, -1], :] = np.inf
    defaults = dict(
        frequency=20e9,
        du=1e-3,
        dv=1e-3,
        mode_index=1,
        eps_r_uu=eu,
        eps_r_vv=ev,
        eps_r_ww=ew,
        mu_r_uu=np.ones_like(ev),
        mu_r_vv=np.ones_like(eu),
        mu_r_ww=np.ones((nu, nv)),
    )
    defaults.update(kwargs)
    return FDFD_2D_mode_solver(**defaults)


def test_1d_use_gpu_falls_back_to_cpu_when_cupy_missing():
    reference = _tm_1d_solver(use_gpu=False)
    reference.solve()
    solver = _tm_1d_solver(use_gpu=True)
    with pytest.warns(RuntimeWarning, match="cuPy"):
        solver.solve()
    np.testing.assert_allclose(solver.eigenvalues, reference.eigenvalues, rtol=1e-12)
    np.testing.assert_allclose(solver.modal_real_neff, reference.modal_real_neff, rtol=1e-10)
    np.testing.assert_allclose(solver.modal_Ea, reference.modal_Ea, rtol=1e-8, atol=1e-10)


def test_2d_use_gpu_falls_back_to_cpu_when_cupy_missing():
    reference = _pec_2d_solver(use_gpu=False)
    reference.solve()
    solver = _pec_2d_solver(use_gpu=True)
    with pytest.warns(RuntimeWarning, match="cuPy"):
        solver.solve()
    np.testing.assert_allclose(solver.eigenvalues, reference.eigenvalues, rtol=1e-12)
    np.testing.assert_allclose(solver.modal_real_neff, reference.modal_real_neff, rtol=1e-10)


def test_1d_gpu_failure_warns_and_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(gpu_backend, "cp", _working_cupy_stub())

    def fail(*args, **kwargs):
        raise RuntimeError("simulated device failure")

    monkeypatch.setattr(gpu_backend, "gpu_eigs", fail)
    reference = _tm_1d_solver(use_gpu=False)
    reference.solve()
    solver = _tm_1d_solver(use_gpu=True)
    with pytest.warns(RuntimeWarning, match="simulated device failure"):
        solver.solve()
    np.testing.assert_allclose(solver.eigenvalues, reference.eigenvalues, rtol=1e-12)


def test_2d_gpu_failure_warns_and_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(gpu_backend, "cp", _working_cupy_stub())

    def fail(*args, **kwargs):
        raise RuntimeError("simulated device failure")

    monkeypatch.setattr(gpu_backend, "gpu_eigs", fail)
    reference = _pec_2d_solver(use_gpu=False)
    reference.solve()
    solver = _pec_2d_solver(use_gpu=True)
    with pytest.warns(RuntimeWarning, match="simulated device failure"):
        solver.solve()
    np.testing.assert_allclose(solver.eigenvalues, reference.eigenvalues, rtol=1e-12)


def test_use_gpu_defaults_to_cpu():
    solver = _tm_1d_solver()
    assert solver.use_gpu is False
    solver2d = _pec_2d_solver()
    assert solver2d.use_gpu is False
