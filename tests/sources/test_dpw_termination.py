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

"""Termination of the 1D auxiliary grid of discrete plane waves.

Covers option resolution (precision defaults, environment override,
per-plane-wave keyword arguments) and small end-to-end checks:

* the 'extended' (light-cone) reference is causal-exact,
* the default PML reaches the double-precision design level,
* the axial incident arrival time does not depend on the termination,
* axial TF/SF along y with Ex polarisation does not leak (regression).
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.sources import (
    DPW_TERMINATION_DEFAULTS,
    DPW_TERMINATION_ENV,
    resolve_dpw_termination,
)


class TestResolve:
    def test_precision_defaults(self):
        d = resolve_dpw_termination(precision="double", environ={})
        s = resolve_dpw_termination(precision="single", environ={})
        assert d == DPW_TERMINATION_DEFAULTS["double"] and d["pml_cells"] == 160
        assert s["pml_cells"] == 20 and s["pml_order"] == 6

    def test_env_override_and_kwarg_precedence(self):
        env = {DPW_TERMINATION_ENV: "pml,pml_cells=40,pml_order=4,pml_r0_db=-240"}
        r = resolve_dpw_termination(precision="double", environ=env)
        assert (r["pml_cells"], r["pml_order"], r["pml_r0_db"]) == (40, 4, -240)
        r = resolve_dpw_termination({"pml_cells": 80}, precision="double", environ=env)
        assert r["pml_cells"] == 80 and r["pml_order"] == 4
        r = resolve_dpw_termination(precision="double", environ={DPW_TERMINATION_ENV: "extended"})
        assert r["termination"] == "extended"

    def test_legacy_fixes_layer(self):
        r = resolve_dpw_termination({"termination": "legacy", "pml_cells": 99}, environ={})
        assert r["pml_cells"] == 20 and r["pml_order"] == 4

    @pytest.mark.parametrize(
        "opts",
        [{"termination": "mur"}, {"pml_cells": 0}, {"pml_cells": 2.5},
         {"pml_order": 0}, {"pml_r0_db": 10}, {"pml_r0_db": float("nan")}],
    )
    def test_invalid(self, opts):
        with pytest.raises(ValueError):
            resolve_dpw_termination(opts, environ={})

    def test_env_unknown_key(self):
        with pytest.raises(ValueError):
            resolve_dpw_termination(environ={DPW_TERMINATION_ENV: "cells=4"})


def _scene(wave, nt, er_bg=1.0):
    d, n, pml = 1e-3, 30, 8
    L = n * d
    sc = gprMax.Scene()
    sc.add(gprMax.OMPThreads(n=1))
    sc.add(gprMax.Discretisation(p1=(d, d, d)))
    sc.add(gprMax.Domain(p1=(L, L, L)))
    sc.add(gprMax.PMLThickness(thickness=pml))
    sc.add(gprMax.TimeWindow(iterations=nt))
    sc.add(gprMax.Waveform(wave_type="gaussiandot", amp=1, freq=6e9, id="w"))
    if er_bg != 1.0:
        sc.add(gprMax.Material(er=er_bg, se=0, mr=1, sm=0, id="bg"))
        sc.add(gprMax.Box(p1=(0, 0, 0), p2=(L, L, L), material_id="bg"))
    a, b = pml + 3, n - pml - 3
    p1, p2 = (a * d,) * 3, (b * d,) * 3
    sc.add(wave(p1, p2))
    sc.add(gprMax.Rx(p1=(n // 2 * d,) * 3))                  # total field
    sc.add(gprMax.Rx(p1=((a - 2) * d, n // 2 * d, n // 2 * d)))  # scattered field
    sc.add(gprMax.Rx(p1=(n // 2 * d, (b + 2) * d, n // 2 * d)))  # scattered field
    return sc


def _run(tmp_path, name, scene):
    out = Path(tmp_path, name)
    gprMax.run(scenes=[scene], outputfile=out, hide_progress_bars=True, log_level=40, cpu_precision="double")
    with h5py.File(out.with_suffix(".h5"), "r") as h:
        return [np.stack([np.asarray(h[f"rxs/rx{i}/{c}"]) for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")])
                for i in (1, 2, 3)]


def _db(x):
    return 20 * np.log10(max(x, 1e-300))


def _err(a, ref):
    return max(np.abs(a[0][s] - ref[0][s]).max() / np.abs(ref[0][s]).max() for s in (slice(0, 3), slice(3, 6)))


def _leak(a):
    return max(np.abs(a[i][s]).max() / np.abs(a[0][s]).max() for i in (1, 2) for s in (slice(0, 3), slice(3, 6)))


def _vector(**kw):
    return lambda p1, p2: gprMax.DiscretePlaneWaveVector(
        p1=p1, p2=p2, m_vec=(2, 1, 1), psi=30, waveform_id="w", material_id="bg", **kw)


def _axial(axis, psi, **kw):
    return lambda p1, p2: gprMax.DiscretePlaneWaveAxial(
        p1=p1, p2=p2, axis=axis, psi=psi, waveform_id="w", **kw)


def test_extended_reference_is_causal_exact(tmp_path):
    a = _run(tmp_path, "a", _scene(_vector(termination="extended"), 500, 4.0))
    b = _run(tmp_path, "b", _scene(_vector(termination="extended"), 900, 4.0))
    for x, y in zip(a, b):
        assert np.array_equal(x, y[:, : x.shape[1]])


def test_default_pml_reaches_design_level(tmp_path):
    ref = _run(tmp_path, "ref", _scene(_vector(termination="extended"), 900, 4.0))
    new = _run(tmp_path, "new", _scene(_vector(), 900, 4.0))
    old = _run(tmp_path, "old", _scene(_vector(termination="legacy"), 900, 4.0))
    assert _db(_err(new, ref)) < -220
    assert _db(_err(new, ref)) < _db(_err(old, ref)) - 40
    assert _db(_leak(new)) < -250


def test_axial_arrival_independent_of_termination(tmp_path):
    ref = _run(tmp_path, "ref", _scene(_axial("z", 30, termination="extended"), 900))
    for name, kw in (("legacy", {"termination": "legacy"}), ("pml40", {"pml_cells": 40}), ("default", {})):
        a = _run(tmp_path, name, _scene(_axial("z", 30, **kw), 900))
        assert _db(_err(a, ref)) < -100, name


@pytest.mark.parametrize("psi", [0, 30])
def test_axial_y_ex_polarisation_does_not_leak(tmp_path, psi):
    a = _run(tmp_path, f"y{psi}", _scene(_axial("y", psi), 600))
    assert _db(_leak(a)) < -250
