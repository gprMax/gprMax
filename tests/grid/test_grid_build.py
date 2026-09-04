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

"""PML construction, ``build()`` orchestration and dispersion-analysis tests.

``build()`` is the grid's assembly line. Its sub-steps are heavy — Cython Yee
cell builders, the whole PML stack, progress bars — so the orchestration tests
patch those out and assert *which* steps run under *which* configuration. That
is the part with real branching; the sub-steps themselves belong to PR 10.

``_construct_pml`` needs no patching: it is pure box arithmetic and is tested
against a real ``PML``.
"""

import numpy as np
import pytest

from gprMax.pml import CFS, PML

from .conftest import DL


@pytest.fixture
def built_grid(make_grid, make_material):
    """A grid with enough state for ``build()`` to run end to end."""
    g = make_grid(nx=8, ny=8, nz=8, dl=DL, pml_thickness=2)
    g.materials = [make_material(ID="pec", numID=0), make_material(ID="free_space", numID=1)]
    return g


@pytest.fixture
def stub_build_steps(monkeypatch):
    """Patch out ``build()``'s heavy sub-steps and record which ran.

    Returns the call-record list; each patched step appends its own name.
    """
    import gprMax.grid.fdtd_grid as fg

    calls = []
    monkeypatch.setattr(fg, "print_pml_info", lambda g: "")

    def _record(grid, name):
        monkeypatch.setattr(grid, name, lambda *a, **k: calls.append(name), raising=True)

    def _apply(grid):
        for name in (
            "_build_pmls",
            "_build_components",
            "_2d_mode_grid_update",
            "_create_voltage_source_materials",
            "_build_materials",
            "_DPW__source_grid_init",
        ):
            _record(grid, name)
        return calls

    return _apply


class TestConstructPml:
    """Each of the six slab IDs maps to a specific box within the domain.

    ``PML.__init__`` runs ``check_kappamin()``, which sums ``kappa.min`` over
    the grid's CFS list and rejects a total below one — so a grid with no CFS
    cannot construct a PML at all. ``build()`` installs the default CFS before
    calling ``_build_pmls``; the fixture below does the same.
    """

    @pytest.fixture
    def make_grid(self, make_grid):
        def _make(**kwargs):
            g = make_grid(**kwargs)
            g.pmls["cfs"] = [CFS()]
            return g

        return _make

    @pytest.mark.parametrize(
        "pml_id,direction",
        [
            ("x0", "xminus"),
            ("xmax", "xplus"),
            ("y0", "yminus"),
            ("ymax", "yplus"),
            ("z0", "zminus"),
            ("zmax", "zplus"),
        ],
    )
    def test_direction(self, make_grid, pml_id, direction):
        g = make_grid(nx=20, ny=20, nz=20)
        assert g._construct_pml(pml_id, 4).direction == direction

    def test_x0_box(self, make_grid):
        g = make_grid(nx=20, ny=21, nz=22)
        pml = g._construct_pml("x0", 4)
        assert (pml.xs, pml.xf) == (0, 4)
        assert (pml.ys, pml.yf) == (0, 21)
        assert (pml.zs, pml.zf) == (0, 22)

    def test_xmax_box_is_measured_from_the_far_face(self, make_grid):
        g = make_grid(nx=20, ny=21, nz=22)
        pml = g._construct_pml("xmax", 4)
        assert (pml.xs, pml.xf) == (16, 20)

    def test_y0_box(self, make_grid):
        g = make_grid(nx=20, ny=21, nz=22)
        pml = g._construct_pml("y0", 4)
        assert (pml.ys, pml.yf) == (0, 4)
        assert (pml.xs, pml.xf) == (0, 20)

    def test_ymax_box_is_measured_from_the_far_face(self, make_grid):
        g = make_grid(nx=20, ny=21, nz=22)
        pml = g._construct_pml("ymax", 4)
        assert (pml.ys, pml.yf) == (17, 21)

    def test_z0_box(self, make_grid):
        g = make_grid(nx=20, ny=21, nz=22)
        pml = g._construct_pml("z0", 4)
        assert (pml.zs, pml.zf) == (0, 4)

    def test_zmax_box_is_measured_from_the_far_face(self, make_grid):
        g = make_grid(nx=20, ny=21, nz=22)
        pml = g._construct_pml("zmax", 4)
        assert (pml.zs, pml.zf) == (18, 22)

    @pytest.mark.parametrize("pml_id", ["x0", "xmax", "y0", "ymax", "z0", "zmax"])
    def test_thickness_is_honoured(self, make_grid, pml_id):
        g = make_grid(nx=20, ny=20, nz=20)
        assert g._construct_pml(pml_id, 7).thickness == 7

    def test_unknown_id_raises(self, make_grid):
        g = make_grid(nx=20, ny=20, nz=20)
        with pytest.raises(ValueError, match="Unknown PML ID"):
            g._construct_pml("w0", 4)

    def test_returns_the_requested_type(self, make_grid):
        g = make_grid(nx=20, ny=20, nz=20)
        assert isinstance(g._construct_pml("x0", 4), PML)

    def test_slab_spans_the_full_transverse_extent(self, make_grid):
        """A PML slab always covers the whole face it sits on."""
        g = make_grid(nx=20, ny=21, nz=22)
        pml = g._construct_pml("x0", 4)
        assert pml.ny == 21
        assert pml.nz == 22


class TestBuildOrchestration:
    def test_runs_the_standard_steps(self, built_grid, stub_build_steps):
        calls = stub_build_steps(built_grid)
        built_grid.build()
        assert "_build_components" in calls
        assert "_2d_mode_grid_update" in calls
        assert "_build_materials" in calls

    def test_installs_a_default_cfs_when_none_given(self, built_grid, stub_build_steps):
        stub_build_steps(built_grid)
        assert built_grid.pmls["cfs"] == []
        built_grid.build()
        assert len(built_grid.pmls["cfs"]) == 1
        assert isinstance(built_grid.pmls["cfs"][0], CFS)

    def test_keeps_a_user_supplied_cfs(self, built_grid, stub_build_steps):
        stub_build_steps(built_grid)
        mine = CFS()
        built_grid.pmls["cfs"] = [mine]
        built_grid.build()
        assert built_grid.pmls["cfs"] == [mine]

    def test_builds_pmls_when_any_slab_is_non_zero(self, built_grid, stub_build_steps):
        calls = stub_build_steps(built_grid)
        built_grid.build()
        assert "_build_pmls" in calls

    def test_skips_pmls_when_all_thicknesses_are_zero(self, built_grid, stub_build_steps):
        calls = stub_build_steps(built_grid)
        built_grid.set_pml_thickness(0)
        built_grid.build()
        assert "_build_pmls" not in calls

    def test_builds_pmls_when_only_one_slab_is_non_zero(self, built_grid, stub_build_steps):
        calls = stub_build_steps(built_grid)
        built_grid.set_pml_thickness(0)
        built_grid.pmls["thickness"]["x0"] = 2
        built_grid.build()
        assert "_build_pmls" in calls

    def test_averaging_gates_component_building(self, built_grid, stub_build_steps):
        calls = stub_build_steps(built_grid)
        built_grid.averagevolumeobjects = False
        built_grid.build()
        assert "_build_components" not in calls

    def test_allocates_field_arrays(self, built_grid, stub_build_steps):
        stub_build_steps(built_grid)
        built_grid.build()
        assert built_grid.Ex.shape == (9, 9, 9)

    def test_allocates_update_coefficient_arrays(self, built_grid, stub_build_steps):
        stub_build_steps(built_grid)
        built_grid.build()
        assert built_grid.updatecoeffsE.shape == (2, 5)

    def test_skips_dispersive_arrays_without_poles(self, built_grid, stub_build_steps, grid_config):
        grid_config.model_config.materials["maxpoles"] = 0
        stub_build_steps(built_grid)
        built_grid.build()
        assert not hasattr(built_grid, "Tx")

    def test_allocates_dispersive_arrays_with_poles(
        self, built_grid, stub_build_steps, grid_config
    ):
        grid_config.model_config.materials["maxpoles"] = 2
        built_grid.maxpoles = 2
        built_grid.dispersivedtype = np.complex128
        stub_build_steps(built_grid)
        built_grid.build()
        assert built_grid.Tx.shape == (2, 9, 9, 9)
        assert built_grid.updatecoeffsdispersive.shape == (2, 6)

    def test_initialises_snapshots(self, built_grid, stub_build_steps):
        class _Snap:
            def __init__(self):
                self.initialised = False

            def initialise_snapfields(self):
                self.initialised = True

        stub_build_steps(built_grid)
        snap = _Snap()
        built_grid.snapshots = [snap]
        built_grid.build()
        assert snap.initialised


class TestDispersionAnalysisWaveformBranches:
    """The branches ``_dispersion_analysis`` reaches before any FFT."""

    def test_no_waveform(self, make_grid):
        g = make_grid(nx=8, ny=8, nz=8)
        assert g._dispersion_analysis(10)["error"] == "no waveform detected."

    def test_impulse_waveform(self, make_grid, make_waveform):
        g = make_grid(nx=8, ny=8, nz=8)
        g.waveforms = [make_waveform("impulse")]
        assert g._dispersion_analysis(10)["error"] == "impulse waveform used."

    def test_user_waveform(self, make_grid, make_waveform):
        g = make_grid(nx=8, ny=8, nz=8)
        g.waveforms = [make_waveform("user")]
        assert g._dispersion_analysis(10)["error"] == "user waveform detected."

    @pytest.mark.parametrize("wave_type", ["sine", "contsine"])
    def test_continuous_waveforms_use_four_times_the_frequency(
        self, make_grid, make_waveform, make_material, wave_type
    ):
        g = make_grid(nx=8, ny=8, nz=8)
        g.dt = 1e-12
        # A material is required: once maxfreq is populated the method looks up
        # the highest-permittivity material, and does so with a bare next().
        g.materials = [make_material(ID="free_space", numID=1, er=1.0)]
        g.waveforms = [make_waveform(wave_type, freq=1e9)]
        results = g._dispersion_analysis(10)
        # maxfreq collapses to the maximum once populated.
        assert results["maxfreq"] == 4e9

    def test_results_keys(self, make_grid):
        g = make_grid(nx=8, ny=8, nz=8)
        results = g._dispersion_analysis(10)
        assert set(results) == {"deltavp", "N", "material", "maxfreq", "error"}

    def test_no_waveform_leaves_metrics_unset(self, make_grid):
        g = make_grid(nx=8, ny=8, nz=8)
        results = g._dispersion_analysis(10)
        assert results["N"] is None
        assert results["deltavp"] is None


class TestDispersionAnalysisReporting:
    """``dispersion_analysis`` turns a results dict into a log line or a raise.

    ``_dispersion_analysis`` is stubbed so each reporting branch can be driven
    directly, without constructing a waveform whose spectrum lands in the right
    place.
    """

    def _results(self, **overrides):
        base = {
            "deltavp": None,
            "N": None,
            "material": None,
            "maxfreq": 1e9,
            "error": "",
        }
        base.update(overrides)
        return base

    def test_error_is_warned_not_raised(self, make_grid, monkeypatch, caplog):
        g = make_grid(nx=8, ny=8, nz=8)
        monkeypatch.setattr(
            g, "_dispersion_analysis", lambda it: self._results(error="no waveform detected.")
        )
        with caplog.at_level("WARNING"):
            g.dispersion_analysis(10)
        assert "not carried out" in caplog.text

    def test_undersampled_grid_raises(self, make_grid, monkeypatch, make_material):
        g = make_grid(nx=8, ny=8, nz=8)
        material = make_material(ID="soil", numID=1)
        monkeypatch.setattr(
            g,
            "_dispersion_analysis",
            lambda it: self._results(N=1, material=material),
        )
        with pytest.raises(ValueError):
            g.dispersion_analysis(10)

    def test_sampling_at_the_threshold_does_not_raise(self, make_grid, monkeypatch, make_material):
        """``mingridsampling`` is 3; ``N == 3`` is acceptable."""
        g = make_grid(nx=8, ny=8, nz=8)
        material = make_material(ID="soil", numID=1)
        monkeypatch.setattr(
            g,
            "_dispersion_analysis",
            lambda it: self._results(N=3, material=material, deltavp=0.1),
        )
        g.dispersion_analysis(10)

    def test_large_phase_error_is_warned(self, make_grid, monkeypatch, make_material, caplog):
        g = make_grid(nx=8, ny=8, nz=8)
        material = make_material(ID="soil", numID=1)
        monkeypatch.setattr(
            g,
            "_dispersion_analysis",
            lambda it: self._results(N=10, material=material, deltavp=50.0),
        )
        with caplog.at_level("WARNING"):
            g.dispersion_analysis(10)
        assert "numerical dispersion" in caplog.text

    def test_small_phase_error_is_reported_at_info(
        self, make_grid, monkeypatch, make_material, caplog
    ):
        g = make_grid(nx=8, ny=8, nz=8)
        material = make_material(ID="soil", numID=1)
        monkeypatch.setattr(
            g,
            "_dispersion_analysis",
            lambda it: self._results(N=10, material=material, deltavp=0.5),
        )
        with caplog.at_level("INFO"):
            g.dispersion_analysis(10)
        assert "phase-velocity error" in caplog.text


pytestmark = pytest.mark.unit
