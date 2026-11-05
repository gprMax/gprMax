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
* the axial incident arrival time does not depend on the termination for
  layers of at least 20 cells, and a thinner layer advances it as documented,
* axial TF/SF along y with Ex polarisation does not leak (regression), in 3D
  and in 2D TE, with ``termination="legacy"``, and with a material interface
  on the upper TF/SF y face,
* a dispersive axial source with a nonzero 1D source offset matches the
  causal-exact reference.
"""

import logging
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


def _scene(wave, nt, er_bg=1.0, extra=()):
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
    for obj in extra:  # materials and geometry, added before the plane wave
        sc.add(obj)
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


def _scene_te(wave, nt):
    """2D TE model (Ex, Ey, Hz; the two-cell Yee thickness) with the same layout as _scene."""
    inf = float("inf")
    d, n, pml = 1e-3, 30, 8
    L = n * d
    sc = gprMax.Scene()
    sc.add(gprMax.DomainMode(mode="TE"))
    sc.add(gprMax.OMPThreads(n=1))
    sc.add(gprMax.Discretisation(p1=(d, d, d)))
    sc.add(gprMax.Domain(p1=(L, L, inf)))
    sc.add(gprMax.PMLThickness(thickness=(pml, pml, 0, pml, pml, 0)))
    sc.add(gprMax.TimeWindow(iterations=nt))
    sc.add(gprMax.Waveform(wave_type="gaussiandot", amp=1, freq=6e9, id="w"))
    a, b = pml + 3, n - pml - 3
    sc.add(wave((a * d, a * d, inf), (b * d, b * d, inf)))
    sc.add(gprMax.Rx(p1=(n // 2 * d, n // 2 * d, inf)))           # total field
    sc.add(gprMax.Rx(p1=(n // 2 * d, (a - 2) * d, inf)))          # scattered field, below
    sc.add(gprMax.Rx(p1=(n // 2 * d, (b + 2) * d, inf)))          # scattered field, above
    return sc


def test_axial_y_ex_two_cell_te_does_not_leak(tmp_path):
    a = _run(tmp_path, "te", _scene_te(_axial("y", 0), 600))
    assert _db(_leak(a)) < -250


def test_legacy_termination_keeps_axial_y_fix(tmp_path):
    """termination="legacy" restores the old termination only; the Ex correction stays."""
    a = _run(tmp_path, "legacy_y", _scene(_axial("y", 0, termination="legacy"), 600))
    assert _db(_leak(a)) < -250


def test_interface_on_upper_tfsf_y_face_does_not_leak(tmp_path):
    """A half-space of er = 4 starting exactly on the upper TF/SF y face (y = p2)."""
    d, n, pml = 1e-3, 30, 8
    b = n - pml - 3
    half = (
        gprMax.Material(er=4, se=0, mr=1, sm=0, id="half"),
        gprMax.Box(p1=(0, b * d, 0), p2=(n * d, n * d, n * d), material_id="half"),
    )
    a = _run(tmp_path, "iface", _scene(_axial("y", 0), 600, extra=half))
    assert _db(_leak(a)) < -250


def test_dispersive_axial_source_with_offset_matches_extended(tmp_path):
    """Debye background, double precision: the default 160-cell layer places the
    hard source 140 1D cells before its legacy position (nonzero source offset)."""
    debye = (
        gprMax.Material(er=3, se=0, mr=1, sm=0, id="debye"),
        gprMax.AddDebyeDispersion(poles=1, er_delta=(1.5,), tau=(2e-11,), material_ids=("debye",)),
        gprMax.Box(p1=(0, 0, 0), p2=(0.03, 0.03, 0.03), material_id="debye"),
    )
    ref = _run(tmp_path, "ref", _scene(_axial("z", 0, termination="extended"), 700, extra=debye))
    new = _run(tmp_path, "new", _scene(_axial("z", 0), 700, extra=debye))
    assert _db(_err(new, ref)) < -220
    assert _db(_leak(new)) < -250


class _RecordingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _lag(x, y):
    """Lag of trace x relative to trace y, in samples, from the cross-correlation peak."""
    return int(np.argmax(np.correlate(x, y, "full"))) - (len(y) - 1)


def test_thin_layer_advances_axial_arrival_and_warns(tmp_path):
    """Documented behaviour: pml_cells < 20 moves the hard source (20 - N) * sum|m|
    1D cells closer to the TF box, so the pulse arrives that many cells earlier
    and a warning is logged; layers of 20 cells or more keep the arrival."""
    # gprMax.run replaces the handlers of the 'gprMax' logger and stops
    # propagation, so listen on the module logger that emits the warning.
    handler = _RecordingHandler()
    logging.getLogger("gprMax.sources").addHandler(handler)
    try:
        ref = _run(tmp_path, "n20", _scene(_axial("z", 0, pml_cells=20), 500))
        assert not any("thinner than 20 cells" in m for m in handler.messages)
        thin = _run(tmp_path, "n10", _scene(_axial("z", 0, pml_cells=10), 500))
        assert any("thinner than 20 cells" in m for m in handler.messages)
    finally:
        logging.getLogger("gprMax.sources").removeHandler(handler)
    thick = _run(tmp_path, "n40", _scene(_axial("z", 0, pml_cells=40), 500))

    ey = 1  # an axial z plane wave with psi = 0 excites Ey
    assert _lag(thick[0][ey], ref[0][ey]) == 0
    # 10 fewer 1D cells of dx = 1 mm at the default time step dx / (c sqrt(3)):
    # the pulse arrives 10 sqrt(3) = 17.3 samples earlier.
    expected = -10 * np.sqrt(3)
    assert abs(_lag(thin[0][ey], ref[0][ey]) - expected) <= 1.0

