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

"""Shared fixtures for the fractals test suite.

The fractal machinery spans three layers:

- ``gprMax/cython/fractals_generate.pyx`` — the spectral filter. Pure
  arithmetic over numpy arrays; needs nothing from this file.
- ``gprMax/fractals/*.py`` — ``FractalSurface``, ``FractalVolume``,
  ``Grass``. These read exactly two pieces of global config:
  ``config.sim_config.dtypes["float_or_double"]`` (the output dtype) and
  ``config.get_model_config().ompthreads`` (passed straight through to
  the Cython kernel). The ``fractal_config`` fixture below supplies both.
- ``gprMax/user_objects/cmds_geometry/{fractal_box,add_*}.py`` — the
  ``build()`` dispatch layer. These additionally need a grid carrying
  ``fractalvolumes``, ``mixingmodels``, ``materials``, a time step, the
  four geometry arrays, and the two factory methods ``FDTDGrid`` exposes
  for creating fractal objects. The ``fractal_grid`` fixture supplies a
  stub with that surface, driven through the *real* ``MainGridUserInput``
  (``_create_uip`` short-circuits to it for any non-subgrid grid).

All spatial tests use a uniform discretisation of ``DL`` (1 mm), so cell
index ``i`` maps to coordinate ``i * DL``.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c, epsilon_0, mu_0

from gprMax.fractals.fractal_surface import FractalSurface
from gprMax.fractals.fractal_volume import FractalVolume
from gprMax.materials import ListMaterial

# Uniform spatial discretisation shared by the dispatch tests.
DL = 0.001

# Time step small enough to clear the Debye relaxation times of the
# built-in water (~8e-12 s) and grass (1.08e-11 s) materials, which
# AddSurfaceWater/AddGrass check against.
DT = 1.926e-12


@pytest.fixture(autouse=True)
def fractal_config(monkeypatch, request):
    """Patch ``gprMax.config`` for the fractal modules.

    Double precision output, a single OpenMP thread, and a materials dict
    that ``create_water`` / ``create_grass`` can bump ``maxpoles`` on.
    """
    if request.node.get_closest_marker("unit") is None:
        return

    from gprMax import config

    model_cfg = SimpleNamespace(
        mode="3D",
        ompthreads=1,
        materials={"maxpoles": 0},
        debye_averaging=True,
        dispersive_averaging=True,
    )
    sim_cfg = SimpleNamespace(
        general={"solver": "cpu", "precision": "double", "subgrid": False},
        dtypes={"float_or_double": np.float64},
        em_consts={"c": c, "e0": epsilon_0, "m0": mu_0},
        args=SimpleNamespace(autotranslate=False),
    )

    monkeypatch.setattr(config, "sim_config", sim_cfg)
    monkeypatch.setattr(config, "get_model_config", lambda: model_cfg)

    return SimpleNamespace(sim_config=sim_cfg, model_config=model_cfg)


def nonzero_set(arr):
    """Set of index tuples at which ``arr`` is nonzero."""
    return set(map(tuple, np.argwhere(np.asarray(arr))))


def make_material(numID, ID, averagable=True):
    """Stub with the attributes the geometry ``build()`` methods read."""
    return SimpleNamespace(
        numID=numID,
        ID=ID,
        averagable=averagable,
        er=1.0,
        se=0.0,
        mr=1.0,
        sm=0.0,
        is_pec=(ID == "pec" or False),
        is_pmc=(ID == "pmc" or False),
    )


class _StubGrid(SimpleNamespace):
    """FDTDGrid stand-in for driving ``build()`` end-to-end.

    Implements the same ``within_bounds`` contract as the real grid
    (raises ``ValueError`` carrying the axis letter) and the same two
    fractal factory methods, which on ``FDTDGrid`` are plain constructors
    (``fdtd_grid.py:201-227``).
    """

    def within_bounds(self, p):
        if p[0] < 0 or p[0] > self.nx:
            raise ValueError("x")
        if p[1] < 0 or p[1] > self.ny:
            raise ValueError("y")
        if p[2] < 0 or p[2] > self.nz:
            raise ValueError("z")
        return True

    def add_fractal_volume(self, xs, xf, ys, yf, zs, zf, frac_dim, seed):
        volume = FractalVolume(xs, xf, ys, yf, zs, zf, frac_dim, seed)
        self.fractalvolumes.append(volume)
        return volume

    def create_fractal_surface(self, xs, xf, ys, yf, zs, zf, frac_dim, seed):
        return FractalSurface(xs, xf, ys, yf, zs, zf, frac_dim, seed)


@pytest.fixture
def grid_arrays():
    """Factory for the four grid arrays with production shapes/dtypes."""

    def _make(nx=16, ny=16, nz=16):
        return SimpleNamespace(
            nx=nx,
            ny=ny,
            nz=nz,
            solid=np.zeros((nx, ny, nz), dtype=np.uint32),
            rigidE=np.zeros((12, nx, ny, nz), dtype=np.int8),
            rigidH=np.zeros((6, nx, ny, nz), dtype=np.int8),
            ID=np.zeros((6, nx + 1, ny + 1, nz + 1), dtype=np.uint32),
        )

    return _make


@pytest.fixture
def fractal_grid(grid_arrays):
    """Factory for a grid stub the fractal ``build()`` methods can run
    against: the four geometry arrays, discretisation, a time step, the
    materials list, and empty fractal-volume / mixing-model registries.
    """

    def _make(nx=16, ny=16, nz=16):
        arrays = grid_arrays(nx, ny, nz)
        return _StubGrid(
            nx=nx,
            ny=ny,
            nz=nz,
            dx=DL,
            dy=DL,
            dz=DL,
            dl=np.array([DL, DL, DL]),
            dt=DT,
            size=np.array([nx, ny, nz]),
            averagevolumeobjects=True,
            materials=[
                make_material(0, "pec", averagable=False),
                make_material(1, "free_space"),
                make_material(2, "sand"),
                make_material(3, "clay"),
                make_material(4, "gravel"),
                make_material(5, "silt"),
            ],
            mixingmodels=[],
            fractalvolumes=[],
            solid=arrays.solid,
            rigidE=arrays.rigidE,
            rigidH=arrays.rigidH,
            ID=arrays.ID,
        )

    return _make


def add_mixing_model(grid, ID="soil", materials=("sand", "clay", "gravel", "silt")):
    """Register a ``ListMaterial`` mixing model on the grid.

    ``ListMaterial`` is the simplest of the three mixing models: it does
    not synthesise new materials, it just collects existing ones by ID,
    so ``calculate_properties`` populates ``matID`` with their numIDs.
    """
    model = ListMaterial(ID, list(materials))
    grid.mixingmodels.append(model)
    return model
