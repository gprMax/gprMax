"""Unit tests for ``gprMax/sources.py``.

Conventions
-----------
* One behaviour per test; descriptive names following
  ``test_<unit>_<context>_<expected>``.
* Closed-form references where possible.
* Known bugs are pinned in dedicated tests with a clear docstring so a
  future fix that flips the assertion is obvious and intentional.

Out of scope (deferred — needs cython kernels + full FDTD grid):
    DiscretePlaneWave.initializeDiscretePlaneWave
    DiscretePlaneWave.grid_init
    DiscretePlaneWave.update_plane_wave_{magnetic,electric,electric_dispersive}
    DiscretePlaneWave.initialize_{electric,magnetic}_fields_1D
    DiscretePlaneWave.update_{electric,magnetic}_field_1D
    DiscretePlaneWave.getField, apply_TFSF_conditions_{electric,magnetic}
    DiscretePlaneWave._get_pml_parameters
"""

import math
import sys
import types
from copy import deepcopy

import numpy as np
import pytest
from scipy.constants import c

from gprMax.materials import Material
from gprMax.sources import (
    DiscretePlaneWave,
    HertzianDipole,
    MagneticDipole,
    Source,
    TransmissionLine,
    VoltageSource,
    htod_src_arrays,
)
from gprMax.waveforms import Waveform as RuntimeWaveform

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _make_id_arrays(material_id=0):
    """Build a ``(IDlookup, ID, updatecoeffsE, updatecoeffsH)`` 4-tuple.

    Material ID 0 is used everywhere; ``updatecoeffsE[0, 4] = 1.0`` and
    ``updatecoeffsH[0, 4] = 1.0`` so source-update formulas reduce to
    ``waveform / spatial_volume`` and are trivial to verify by hand.
    """
    IDlookup = {"Ex": 0, "Ey": 1, "Ez": 2, "Hx": 3, "Hy": 4, "Hz": 5}
    ID = np.zeros((6, 4, 4, 4), dtype=np.int32)
    ID[...] = material_id
    updatecoeffsE = np.zeros((1, 5))
    updatecoeffsE[0, 4] = 1.0
    updatecoeffsH = np.zeros((1, 5))
    updatecoeffsH[0, 4] = 1.0
    return IDlookup, ID, updatecoeffsE, updatecoeffsH


def _zero_fields(shape=(4, 4, 4)):
    return [np.zeros(shape) for _ in range(3)]


# ---------------------------------------------------------------------------
# Source — base class
# ---------------------------------------------------------------------------


class TestSourceBase:
    def test_init_defaults(self):
        s = Source()
        assert s.polarisation is None
        assert s.start == 0.0
        assert s.stop == 0.0
        assert s.waveformID is None
        assert s.waveformvalues_wholedt is None
        assert s.waveformvalues_halfdt is None

    def test_coord_arrays_are_zero_int32(self):
        s = Source()
        assert s.coord.dtype == np.int32
        assert s.coordorigin.dtype == np.int32
        assert np.all(s.coord == 0)
        assert np.all(s.coordorigin == 0)

    @pytest.mark.parametrize("axis, idx", [("x", 0), ("y", 1), ("z", 2)])
    def test_coord_property_round_trips(self, axis, idx):
        s = Source()
        setattr(s, f"{axis}coord", 7)
        assert s.coord[idx] == 7
        assert getattr(s, f"{axis}coord") == 7

    @pytest.mark.parametrize("axis, idx", [("x", 0), ("y", 1), ("z", 2)])
    def test_coordorigin_property_round_trips(self, axis, idx):
        s = Source()
        setattr(s, f"{axis}coordorigin", 3)
        assert s.coordorigin[idx] == 3
        assert getattr(s, f"{axis}coordorigin") == 3


# ---------------------------------------------------------------------------
# VoltageSource
# ---------------------------------------------------------------------------


def _make_voltage_source(*, polarisation, resistance, waveformID="wf"):
    src = VoltageSource()
    src.ID = "vs"
    src.polarisation = polarisation
    src.resistance = resistance
    src.waveformID = waveformID
    return src


class TestVoltageSourceInit:
    def test_inherits_source_defaults(self):
        s = VoltageSource()
        assert s.polarisation is None
        assert s.resistance is None


class TestVoltageSourceCalculateWaveformValues:
    def test_populates_both_arrays_inside_window(self, fake_grid, make_constant_waveform):
        w = make_constant_waveform(ID="wf", value=2.5)
        G = fake_grid(iterations=10, dt=1e-12, waveforms=[w])
        src = _make_voltage_source(polarisation="x", resistance=50.0)
        src.start = 0.0
        src.stop = G.timewindow

        src.calculate_waveform_values(G)

        # All in-window values equal the constant waveform value (2.5).
        # _ConstantWaveform returns 2.5 for any t >= 0.
        assert np.all(src.waveformvalues_wholedt == 2.5)
        assert np.all(src.waveformvalues_halfdt == 2.5)
        # Shape is iterations + 1 (so the solver can index past the last step).
        assert src.waveformvalues_wholedt.shape == (G.iterations + 1,)
        assert src.waveformvalues_halfdt.shape == (G.iterations + 1,)

    def test_zero_outside_window(self, fake_grid, make_constant_waveform):
        w = make_constant_waveform(value=1.0)
        G = fake_grid(iterations=10, dt=1.0, waveforms=[w])
        src = _make_voltage_source(polarisation="x", resistance=50.0)
        # Window covers only iterations 3..6 inclusive (t = 3..6).
        src.start = 3.0
        src.stop = 6.0

        src.calculate_waveform_values(G)

        # Outside window: zero. Inside: 1.0.
        for it in range(G.iterations + 1):
            t = G.dt * it
            expected = 1.0 if src.start <= t <= src.stop else 0.0
            assert src.waveformvalues_wholedt[it] == expected

    def test_reuses_precomputed_values_from_matching_source(
        self, fake_grid, make_constant_waveform
    ):
        """Per ``sources.py:131-138``: if another source in
        ``G.voltagesources`` has the same ``waveformID`` and the default
        ``start=0``/``stop=timewindow`` window, its precomputed arrays
        are reused (identity-equal numpy view).
        """
        w = make_constant_waveform(value=3.0)
        existing = _make_voltage_source(polarisation="x", resistance=50.0)
        existing.start = 0
        sentinel_half = np.full(11, 99.0)
        sentinel_whole = np.full(11, 77.0)
        existing.waveformvalues_halfdt = sentinel_half
        existing.waveformvalues_wholedt = sentinel_whole

        G = fake_grid(iterations=10, dt=1.0, waveforms=[w], voltagesources=[existing])
        existing.stop = G.timewindow

        new = _make_voltage_source(polarisation="y", resistance=10.0)
        new.start = 0
        new.stop = G.timewindow
        new.calculate_waveform_values(G)

        assert new.waveformvalues_halfdt is sentinel_half
        assert new.waveformvalues_wholedt is sentinel_whole


class TestVoltageSourceUpdateElectric:
    @pytest.mark.parametrize(
        "polarisation, field_idx, d_other",
        [("x", 0, ("dy", "dz")), ("y", 1, ("dx", "dz")), ("z", 2, ("dx", "dy"))],
    )
    def test_resistive_decrements_E_by_documented_formula(
        self, fake_grid, polarisation, field_idx, d_other
    ):
        """Per ``sources.py:179-205`` resistive case:
            E -= coeff * waveform[it] / (R * d_a * d_b)
        with ``coeff = updatecoeffsE[material, 4]``.
        """
        IDlookup, ID, updatecoeffsE, _ = _make_id_arrays()
        G = fake_grid(dt=1.0, dx=1.0, dy=1.0, dz=1.0, iterations=5, IDlookup=IDlookup, ID=ID)
        Ex, Ey, Ez = _zero_fields()

        src = _make_voltage_source(polarisation=polarisation, resistance=2.0)
        src.start = 0.0
        src.stop = G.timewindow
        src.xcoord = src.ycoord = src.zcoord = 1
        src.waveformvalues_halfdt = np.array([0.0, 4.0, 0.0, 0.0, 0.0, 0.0])
        src.waveformvalues_wholedt = np.zeros(6)

        src.update_electric(1, updatecoeffsE, ID, Ex, Ey, Ez, G)

        target = [Ex, Ey, Ez][field_idx]
        # coeff=1, waveform=4, R=2, d_a=d_b=1 → 1*4/(2*1*1) = 2.0; field decrements.
        assert target[1, 1, 1] == pytest.approx(-2.0)

    @pytest.mark.parametrize(
        "polarisation, field_idx, d_along",
        [("x", 0, "dx"), ("y", 1, "dy"), ("z", 2, "dz")],
    )
    def test_hard_source_overwrites_E_with_negated_waveform(
        self, fake_grid, polarisation, field_idx, d_along
    ):
        """Per ``sources.py:187`` hard-source case (``resistance == 0``):
        E[i,j,k] = -waveform_wholedt[it] / d_along
        """
        IDlookup, ID, updatecoeffsE, _ = _make_id_arrays()
        G = fake_grid(dt=1.0, dx=2.0, dy=2.0, dz=2.0, iterations=5, IDlookup=IDlookup, ID=ID)
        Ex, Ey, Ez = _zero_fields()

        src = _make_voltage_source(polarisation=polarisation, resistance=0)
        src.start = 0.0
        src.stop = G.timewindow
        src.xcoord = src.ycoord = src.zcoord = 1
        src.waveformvalues_halfdt = np.zeros(6)
        src.waveformvalues_wholedt = np.array([0.0, 3.0, 0.0, 0.0, 0.0, 0.0])

        # Seed the cell with a known value to confirm the assignment is a
        # *replacement*, not a decrement.
        target = [Ex, Ey, Ez][field_idx]
        target[1, 1, 1] = 999.0

        src.update_electric(1, updatecoeffsE, ID, Ex, Ey, Ez, G)

        # -1 * 3 / 2 = -1.5
        assert target[1, 1, 1] == pytest.approx(-1.5)

    def test_outside_window_is_noop(self, fake_grid):
        IDlookup, ID, updatecoeffsE, _ = _make_id_arrays()
        G = fake_grid(dt=1.0, iterations=5, IDlookup=IDlookup, ID=ID)
        Ex, Ey, Ez = _zero_fields()

        src = _make_voltage_source(polarisation="x", resistance=0)
        src.start = 2.0
        src.stop = 3.0
        src.xcoord = src.ycoord = src.zcoord = 1
        src.waveformvalues_halfdt = np.zeros(6)
        src.waveformvalues_wholedt = np.full(6, 5.0)

        src.update_electric(0, updatecoeffsE, ID, Ex, Ey, Ez, G)

        assert Ex[1, 1, 1] == 0.0


class TestVoltageSourceCreateMaterial:
    def test_resistance_zero_returns_without_modifying_grid(self, fake_grid):
        IDlookup, ID, _, _ = _make_id_arrays()
        material = Material(0, "free_space")
        G = fake_grid(IDlookup=IDlookup, ID=ID, materials=[material])

        src = _make_voltage_source(polarisation="x", resistance=0)
        src.ID = "vs"
        src.xcoord = src.ycoord = src.zcoord = 1
        src.create_material(G)

        assert len(G.materials) == 1
        assert G.materials[0] is material

    @pytest.mark.parametrize(
        "polarisation, conductivity_factor",
        [
            ("x", lambda dx, dy, dz, R: dx / (R * dy * dz)),
            ("y", lambda dx, dy, dz, R: dy / (R * dx * dz)),
            ("z", lambda dx, dy, dz, R: dz / (R * dx * dy)),
        ],
    )
    def test_resistive_appends_new_material_with_added_conductivity(
        self, fake_grid, polarisation, conductivity_factor
    ):
        IDlookup, ID, _, _ = _make_id_arrays()
        base = Material(0, "free_space")
        base.se = 0.01
        G = fake_grid(
            dx=0.001,
            dy=0.002,
            dz=0.003,
            IDlookup=IDlookup,
            ID=ID,
            materials=[base],
        )

        src = _make_voltage_source(polarisation=polarisation, resistance=50.0)
        src.ID = "vs"
        src.xcoord = src.ycoord = src.zcoord = 1
        src.create_material(G)

        assert len(G.materials) == 2
        new_mat = G.materials[1]
        assert new_mat.ID == "free_space+vs"
        assert new_mat.averagable is False
        expected_se = 0.01 + conductivity_factor(G.dx, G.dy, G.dz, src.resistance)
        assert new_mat.se == pytest.approx(expected_se)
        # The grid's ID lookup at this cell now points at the new material.
        assert G.ID[IDlookup[f"E{polarisation}"], 1, 1, 1] == new_mat.numID


# ---------------------------------------------------------------------------
# HertzianDipole
# ---------------------------------------------------------------------------


def _make_hertzian(*, polarisation, dl=1.0, waveformID="wf"):
    src = HertzianDipole()
    src.ID = "hd"
    src.polarisation = polarisation
    src.dl = dl
    src.waveformID = waveformID
    return src


class TestHertzianDipoleInit:
    def test_dl_default_is_zero(self):
        assert HertzianDipole().dl == 0.0


class TestHertzianDipoleCalculateWaveformValues:
    def test_only_halfdt_array_populated(self, fake_grid, make_constant_waveform):
        w = make_constant_waveform(value=2.0)
        G = fake_grid(iterations=5, dt=1.0, waveforms=[w])
        src = _make_hertzian(polarisation="z")
        src.start = 0.0
        src.stop = G.timewindow

        src.calculate_waveform_values(G)

        assert src.waveformvalues_halfdt is not None
        assert src.waveformvalues_wholedt is None  # never set
        assert np.all(src.waveformvalues_halfdt == 2.0)


class TestHertzianDipoleUpdateElectric:
    @pytest.mark.parametrize(
        "polarisation, field_idx",
        [("x", 0), ("y", 1), ("z", 2)],
    )
    def test_decrements_E_by_documented_formula(self, fake_grid, polarisation, field_idx):
        """Per ``sources.py:303-325``:
        E -= coeff * waveform * dl / (dx*dy*dz)
        """
        IDlookup, ID, updatecoeffsE, _ = _make_id_arrays()
        G = fake_grid(dt=1.0, dx=1.0, dy=1.0, dz=1.0, iterations=5, IDlookup=IDlookup, ID=ID)
        Ex, Ey, Ez = _zero_fields()

        src = _make_hertzian(polarisation=polarisation, dl=2.0)
        src.start = 0.0
        src.stop = G.timewindow
        src.xcoord = src.ycoord = src.zcoord = 1
        src.waveformvalues_halfdt = np.array([0.0, 3.0, 0.0, 0.0, 0.0, 0.0])

        src.update_electric(1, updatecoeffsE, ID, Ex, Ey, Ez, G)

        target = [Ex, Ey, Ez][field_idx]
        # coeff=1, waveform=3, dl=2, volume=1 → 6.0 decrement
        assert target[1, 1, 1] == pytest.approx(-6.0)


# ---------------------------------------------------------------------------
# MagneticDipole
# ---------------------------------------------------------------------------


def _make_magnetic(*, polarisation, waveformID="wf"):
    src = MagneticDipole()
    src.ID = "md"
    src.polarisation = polarisation
    src.waveformID = waveformID
    return src


class TestMagneticDipoleCalculateWaveformValues:
    def test_only_wholedt_array_populated(self, fake_grid, make_constant_waveform):
        w = make_constant_waveform(value=1.5)
        G = fake_grid(iterations=5, dt=1.0, waveforms=[w])
        src = _make_magnetic(polarisation="y")
        src.start = 0.0
        src.stop = G.timewindow

        src.calculate_waveform_values(G)

        assert src.waveformvalues_wholedt is not None
        assert src.waveformvalues_halfdt is None
        assert np.all(src.waveformvalues_wholedt == 1.5)


class TestMagneticDipoleUpdateMagnetic:
    @pytest.mark.parametrize(
        "polarisation, field_idx",
        [("x", 0), ("y", 1), ("z", 2)],
    )
    def test_decrements_H_by_documented_formula(self, fake_grid, polarisation, field_idx):
        """Per ``sources.py:382-401``:
        H -= coeff * waveform / (dx*dy*dz)
        """
        IDlookup, ID, _, updatecoeffsH = _make_id_arrays()
        G = fake_grid(dt=1.0, dx=1.0, dy=1.0, dz=1.0, iterations=5, IDlookup=IDlookup, ID=ID)
        Hx, Hy, Hz = _zero_fields()

        src = _make_magnetic(polarisation=polarisation)
        src.start = 0.0
        src.stop = G.timewindow
        src.xcoord = src.ycoord = src.zcoord = 1
        src.waveformvalues_wholedt = np.array([0.0, 7.0, 0.0, 0.0, 0.0, 0.0])

        src.update_magnetic(1, updatecoeffsH, ID, Hx, Hy, Hz, G)

        target = [Hx, Hy, Hz][field_idx]
        assert target[1, 1, 1] == pytest.approx(-7.0)


# ---------------------------------------------------------------------------
# htod_src_arrays — pinned bugs
# ---------------------------------------------------------------------------


class _FakeGpuArrayModule:
    """Stand-in for ``pycuda.gpuarray`` that returns the input numpy array
    untouched. Lets tests inspect what would have been shipped to the GPU.
    """

    @staticmethod
    def to_gpu(arr):
        return arr


class TestHtodSrcArraysCpuBug:
    """Pin the missing CPU branch in ``htod_src_arrays``.

    Source: ``sources.py:404-473``. The function only assigns the ``*_dev``
    locals inside ``cuda``/``opencl``/``metal`` branches, so on CPU the
    final ``return`` accesses unbound locals.

    When fixed (e.g. ``cpu`` branch added that returns the host numpy
    arrays), update this test to call ``htod_src_arrays`` and assert the
    returned arrays have the expected shape.
    """

    def test_cpu_solver_raises_unbound_local(self, fake_grid):
        # solver defaults to "cpu" via the autouse source_config fixture.
        G = fake_grid(iterations=5)
        with pytest.raises(UnboundLocalError):
            htod_src_arrays([], G)


class TestHtodSrcArraysVoltageSourceDeadCodeBug:
    """Pin the dead-code override at ``sources.py:448-449``.

    Even when ``resistance == 0`` (hard source), the unconditional
    overwrite after the if/else makes ``srcwaves`` end up as
    ``waveformvalues_halfdt`` instead of the intended
    ``waveformvalues_wholedt``. When the source is fixed (the two
    redundant lines deleted), this test must flip its assertion.
    """

    def test_hard_voltage_source_srcwaves_currently_uses_halfdt(self, fake_grid, monkeypatch):
        from gprMax import config

        # The autouse source_config fixture already gives us a fresh dict;
        # mutate it in place to flip the solver to "cuda" for this test.
        config.sim_config.general["solver"] = "cuda"

        # Inject a fake pycuda.gpuarray that just returns its input.
        fake_pycuda = types.ModuleType("pycuda")
        fake_gpuarray = _FakeGpuArrayModule()
        fake_pycuda.gpuarray = fake_gpuarray
        monkeypatch.setitem(sys.modules, "pycuda", fake_pycuda)
        monkeypatch.setitem(sys.modules, "pycuda.gpuarray", fake_gpuarray)

        G = fake_grid(iterations=2)

        # Hard source: resistance == 0.
        src = _make_voltage_source(polarisation="x", resistance=0)
        src.xcoord = src.ycoord = src.zcoord = 0
        # Distinguishable arrays so we can tell which one was shipped.
        src.waveformvalues_halfdt = np.array([1.0, 2.0, 3.0])
        src.waveformvalues_wholedt = np.array([10.0, 20.0, 30.0])

        _, _, srcwaves = htod_src_arrays([src], G)

        # Upstream fixed: hard source now correctly ships wholedt values.
        assert np.array_equal(srcwaves[0], src.waveformvalues_wholedt)


# ---------------------------------------------------------------------------
# TransmissionLine
# ---------------------------------------------------------------------------


class TestTransmissionLineInit:
    def test_dl_equals_sqrt3_c_dt(self):
        tl = TransmissionLine(iterations=100, dt=1e-12)
        assert tl.dl == pytest.approx(math.sqrt(3) * c * 1e-12)

    def test_nl_equals_round_two_thirds_iterations(self):
        tl = TransmissionLine(iterations=1000, dt=1e-12)
        # round_value(0.667 * 1000) → 667
        assert tl.nl == 667

    def test_voltage_and_current_arrays_sized_nl(self):
        tl = TransmissionLine(iterations=100, dt=1e-12)
        assert tl.voltage.shape == (tl.nl,)
        assert tl.current.shape == (tl.nl,)

    def test_incident_arrays_sized_iterations_plus_one(self):
        tl = TransmissionLine(iterations=100, dt=1e-12)
        assert tl.Vinc.shape == (101,)
        assert tl.Iinc.shape == (101,)
        assert tl.Vtotal.shape == (101,)
        assert tl.Itotal.shape == (101,)

    def test_default_positions(self):
        tl = TransmissionLine(iterations=100, dt=1e-12)
        assert tl.srcpos == 5
        assert tl.antpos == 10
        assert tl.abcv0 == 0 and tl.abcv1 == 0


class TestTransmissionLineCalculateWaveformValues:
    def test_populates_both_arrays(self, fake_grid, make_constant_waveform):
        w = make_constant_waveform(value=1.0)
        G = fake_grid(iterations=5, dt=1.0, waveforms=[w])
        tl = TransmissionLine(iterations=5, dt=1.0)
        tl.waveformID = "wf"
        tl.start = 0.0
        tl.stop = G.timewindow

        tl.calculate_waveform_values(G)

        assert np.all(tl.waveformvalues_wholedt == 1.0)
        assert np.all(tl.waveformvalues_halfdt == 1.0)


class TestTransmissionLineUpdateABC:
    def test_abc_coefficient_matches_closed_form(self, fake_grid):
        """``h = (c*dt - dl) / (c*dt + dl)``. Set voltage[0]=0, voltage[1]=v
        and prior abcv0/abcv1 to known values; after ``update_abc`` we have
        ``voltage[0] = h*(voltage[1] - abcv0) + abcv1``.
        """
        dt = 1e-12
        tl = TransmissionLine(iterations=20, dt=dt)
        tl.voltage[0] = 0.0
        tl.voltage[1] = 1.0
        tl.abcv0 = 0.5
        tl.abcv1 = 0.25
        G = fake_grid(dt=dt)

        h_expected = (c * dt - tl.dl) / (c * dt + tl.dl)
        expected_v0 = h_expected * (tl.voltage[1] - 0.5) + 0.25

        tl.update_abc(G)

        assert tl.voltage[0] == pytest.approx(expected_v0)
        # abcv0/abcv1 advance to the previous voltage[0]/voltage[1].
        assert tl.abcv0 == pytest.approx(expected_v0)
        assert tl.abcv1 == 1.0


class TestTransmissionLineCalculateIncidentVI:
    def test_shortens_nl_to_antpos_plus_one(self, fake_grid, make_constant_waveform):
        dt = 1e-12
        tl = TransmissionLine(iterations=50, dt=dt)
        tl.resistance = 50.0
        tl.waveformID = "wf"
        tl.start = 0.0
        w = make_constant_waveform(value=0.0)
        G = fake_grid(iterations=50, dt=dt, waveforms=[w])
        tl.stop = G.timewindow
        tl.calculate_waveform_values(G)

        tl.calculate_incident_V_I(G)

        assert tl.nl == tl.antpos + 1


class TestTransmissionLineUpdateElectric:
    @pytest.mark.parametrize(
        "polarisation, field_idx, d_along",
        [("x", 0, "dx"), ("y", 1, "dy"), ("z", 2, "dz")],
    )
    def test_sets_E_from_voltage(self, fake_grid, polarisation, field_idx, d_along):
        dt = 1e-12
        tl = TransmissionLine(iterations=20, dt=dt)
        tl.resistance = 50.0
        tl.polarisation = polarisation
        tl.xcoord = tl.ycoord = tl.zcoord = 1
        tl.start = 0.0
        tl.stop = 100.0
        # Pre-load voltage and current arrays so update_voltage runs
        # deterministically; the new E should be -voltage[antpos]/d.
        tl.voltage[tl.antpos] = 0.4
        tl.waveformvalues_wholedt = np.zeros(21)
        tl.waveformvalues_halfdt = np.zeros(21)
        IDlookup, ID, updatecoeffsE, _ = _make_id_arrays()
        G = fake_grid(dt=dt, dx=1.0, dy=1.0, dz=1.0, IDlookup=IDlookup, ID=ID)
        Ex, Ey, Ez = _zero_fields()

        tl.update_electric(0, updatecoeffsE, ID, Ex, Ey, Ez, G)

        # update_voltage runs first; with zero waveform & zero current it
        # only applies the resistance*c*dt/dl correction (zero diff → no
        # change). voltage[antpos] stays 0.4.
        target = [Ex, Ey, Ez][field_idx]
        assert target[1, 1, 1] == pytest.approx(-0.4)


class TestTransmissionLineUpdateMagnetic:
    @pytest.mark.parametrize(
        "polarisation, calc_method, expected_current",
        [
            ("x", "calculate_Ix", 0.7),
            ("y", "calculate_Iy", 0.3),
            ("z", "calculate_Iz", -0.5),
        ],
    )
    def test_pulls_current_from_grid_calculate_I(
        self, fake_grid, polarisation, calc_method, expected_current
    ):
        dt = 1e-12
        tl = TransmissionLine(iterations=20, dt=dt)
        tl.resistance = 50.0
        tl.polarisation = polarisation
        tl.xcoord = tl.ycoord = tl.zcoord = 1
        tl.start = 0.0
        tl.stop = 100.0
        tl.waveformvalues_halfdt = np.zeros(21)
        IDlookup, ID, _, updatecoeffsH = _make_id_arrays()
        G = fake_grid(dt=dt, dx=1.0, dy=1.0, dz=1.0, IDlookup=IDlookup, ID=ID)
        # Stub the calculate_I* methods on the fake grid.
        setattr(G, calc_method, lambda i, j, k: expected_current)
        Hx, Hy, Hz = _zero_fields()

        tl.update_magnetic(0, updatecoeffsH, ID, Hx, Hy, Hz, G)

        assert tl.current[tl.antpos] == pytest.approx(expected_current)


# ---------------------------------------------------------------------------
# DiscretePlaneWave
# ---------------------------------------------------------------------------


class TestDiscretePlaneWaveInit:
    def test_m_array_shape_and_dtype(self):
        dpw = DiscretePlaneWave(G=None)  # __init__ does not use G
        assert dpw.m.shape == (4,)
        assert dpw.m.dtype == np.int32

    def test_origin_array_shape_and_zero_default(self):
        dpw = DiscretePlaneWave(G=None)
        assert dpw.origin.shape == (3,)
        assert dpw.origin.dtype == np.int32
        assert np.all(dpw.origin == 0)

    def test_projections_array_is_float64_length_6(self):
        dpw = DiscretePlaneWave(G=None)
        assert dpw.projections.shape == (6,)
        assert dpw.projections.dtype == np.float64

    def test_default_scalar_fields(self):
        dpw = DiscretePlaneWave(G=None)
        assert dpw.materialID is None  # upstream made material optional
        assert dpw.pml_cells == 20
        assert dpw.buffercells_axial == 5
        assert dpw.axial == 0
        assert dpw.speed == c


class TestFindDpwIntegersOptimized:
    """``find_dpw_integers_optimized`` is pure math: given a propagation
    direction (theta, phi) and grid spacing, return the smallest integer
    vector (m_x, m_y, m_z) whose physical direction matches within the
    tolerance.

    These tests exercise the axial cases (where the answer is exact) and
    one non-axial case where the answer is non-trivial.
    """

    def _solve(self, dpw, theta_deg, phi_deg, tol_deg=0.5, ds=(1.0, 1.0, 1.0)):
        return dpw.find_dpw_integers_optimized(theta_deg, phi_deg, ds, tol_deg)

    def test_propagation_along_plus_x_axis(self):
        """theta=90°, phi=0° points along +x. m_vec should be ±(1, 0, 0)
        for a uniform grid.
        """
        dpw = DiscretePlaneWave(G=None)
        m_vec, _, _, total_err = self._solve(dpw, theta_deg=90.0, phi_deg=0.0)
        assert m_vec is not None
        # Magnitudes (1, 0, 0); sign chosen by the algorithm.
        assert np.abs(m_vec).tolist() == [1, 0, 0]
        assert total_err == pytest.approx(0.0, abs=1e-6)

    def test_propagation_along_plus_z_axis(self):
        """theta=0° points along +z."""
        dpw = DiscretePlaneWave(G=None)
        m_vec, _, _, total_err = self._solve(dpw, theta_deg=0.0, phi_deg=0.0)
        assert m_vec is not None
        assert np.abs(m_vec).tolist() == [0, 0, 1]
        assert total_err == pytest.approx(0.0, abs=1e-6)

    def test_negative_tolerance_returns_none(self):
        """A negative tolerance is unsatisfiable (errors are always >= 0),
        so the function should return ``None`` with NaN angles/errors
        instead of inventing a wrong answer. Tests the empty-candidate
        guard at ``sources.py:2095-2097``.
        """
        dpw = DiscretePlaneWave(G=None)
        m_vec, angles, errs, total_err = self._solve(
            dpw, theta_deg=37.0, phi_deg=23.0, tol_deg=-1.0
        )
        assert m_vec is None
        assert math.isnan(angles[0]) and math.isnan(angles[1])
        assert math.isnan(errs[0]) and math.isnan(errs[1])
        assert math.isnan(total_err)

    def test_oblique_uniform_grid_produces_small_integers(self):
        """45/45 oblique on a uniform grid should resolve to a small
        integer triple within a 1° tolerance.
        """
        dpw = DiscretePlaneWave(G=None)
        m_vec, _, _, total_err = self._solve(dpw, theta_deg=45.0, phi_deg=45.0, tol_deg=1.0)
        assert m_vec is not None
        assert max(abs(int(x)) for x in m_vec) < 10
        assert total_err <= 1.0


class TestDpwCalculateWaveformValuesNonCython:
    """The non-cython fallback path of
    ``DiscretePlaneWave.calculate_waveform_values`` is testable directly.
    """

    def _ready_dpw(self, *, freq=1e9):
        dpw = DiscretePlaneWave(G=None)
        # The non-cython path reads only `self.m[3]`, `self.ds`, `self.speed`,
        # `self.start`, `self.stop`, `self.waveform`, and `G.dt`/`G.iterations`.
        # Use the smallest m-vector that drives the loop bodies once.
        dpw.m = np.array([1, 1, 1, 1], dtype=np.int32)
        dpw.ds = 1.0
        dpw.speed = c
        dpw.start = 0.0
        dpw.stop = 1.0
        return dpw

    def test_array_shapes(self, fake_grid, make_constant_waveform):
        dpw = self._ready_dpw()
        dpw.waveform = make_constant_waveform(value=0.5)
        G = fake_grid(iterations=3, dt=1.0)

        dpw.calculate_waveform_values(G, cythonize=False)

        # Shape is (iterations + 1, 3, m[3])
        assert dpw.waveformvalues_wholedt.shape == (4, 3, 1)
        assert dpw.waveformvalues_halfdt.shape == (4, 3, 1)

    def test_zero_outside_window(self, fake_grid, make_constant_waveform):
        dpw = self._ready_dpw()
        # Window is t in [0, 1]; with dt=10, iteration 1 → t≈10 which is
        # outside the window, so both arrays should remain zero there.
        dpw.start = 0.0
        dpw.stop = 0.5
        dpw.waveform = make_constant_waveform(value=7.0)
        G = fake_grid(iterations=2, dt=10.0)

        dpw.calculate_waveform_values(G, cythonize=False)

        # iteration=1 → time1 ≈ 15 - offset; always > 0.5 → zero.
        assert np.all(dpw.waveformvalues_halfdt[1] == 0)
        assert np.all(dpw.waveformvalues_wholedt[1] == 0)


class TestDpwWaveformPrecomputation:
    def test_whole_step_uses_own_component_half_cell_offset(self, fake_grid):
        class TimeWaveform:
            type = "user"
            freq = 1.0

            @staticmethod
            def calculate_value(time, dt):
                return time

        dpw = DiscretePlaneWave(G=None)
        dpw.m = np.array([2, 10, 10, 10], dtype=np.int32)
        dpw.ds = 1.0
        dpw.speed = 1.0
        dpw.start = 0.0
        dpw.stop = 1e9
        dpw.waveform = TimeWaveform()

        G = fake_grid(iterations=1, dt=100.0)
        dpw.calculate_waveform_values(G, cythonize=False)

        # Ex is collocated half an x-projection cell from the auxiliary-grid
        # origin: 100 - |m_x|/2 = 99, not 98 (a full-cell offset) and not
        # 90 (the transverse Hy/Hz staggering).
        assert dpw.waveformvalues_wholedt[1, 0, 0] == pytest.approx(99.0)

    def test_cython_and_reference_histories_match_including_amplitude(self, fake_grid):
        waveform = RuntimeWaveform()
        waveform.ID = "pulse"
        waveform.type = "gaussian"
        waveform.freq = 1e9
        waveform.amp = 2.5

        dpw = DiscretePlaneWave(G=None)
        dpw.m = np.array([1, 2, 3, 3], dtype=np.int32)
        dpw.ds = 1e-4
        dpw.speed = c
        dpw.start = 0.0
        dpw.stop = 20e-12
        dpw.waveform = waveform
        grid = fake_grid(iterations=12, dt=1e-12)

        dpw.calculate_waveform_values(grid, cythonize=False)
        expected_whole = dpw.waveformvalues_wholedt.copy()
        expected_half = dpw.waveformvalues_halfdt.copy()
        dpw.calculate_waveform_values(grid, cythonize=True)

        np.testing.assert_allclose(dpw.waveformvalues_wholedt, expected_whole, rtol=1e-13)
        np.testing.assert_allclose(dpw.waveformvalues_halfdt, expected_half, rtol=1e-13)


pytestmark = pytest.mark.unit
