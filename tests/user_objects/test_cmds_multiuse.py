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

"""Tests for ``gprMax.user_objects.cmds_multiuse``.

19 repeatable user-object classes. Each class has the same contract as
singleuse (kwargs survive, order/hash properties match, ``__str__()``
round-trips), with two new wrinkles:

* Most classes take ``**kwargs`` and forward to ``super().__init__`` —
  attribute mirroring happens in ``build()``, not ``__init__``. So the
  constructor test is just "kwargs survive verbatim".
* Five classes use ``RotatableMixin``: ``VoltageSource``,
  ``HertzianDipole``, ``MagneticDipole``, ``TransmissionLine``, ``Rx``.

Four bug tripwires pin current behaviour for fixes to flip.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gprMax.materials import Material as RuntimeMaterial
from gprMax.user_objects.cmds_multiuse import (
    PMLCFS,
    AddDebyeDispersion,
    AddDrudeDispersion,
    AddLorentzDispersion,
    DiscretePlaneWaveAngles,
    DiscretePlaneWaveAxial,
    DiscretePlaneWaveVector,
    ExcitationFile,
    HertzianDipole,
    MagneticDipole,
    Material,
    MaterialCrim,
    MaterialList,
    MaterialRange,
    Rx,
    RxArray,
    SoilPeplinski,
    TransmissionLine,
    VoltageSource,
    Waveform,
)

from .conftest import make_waveform

# ---------------------------------------------------------------------------
# ExcitationFile
# ---------------------------------------------------------------------------


class TestExcitationFile:
    def test_constructor_stores_attributes_and_kwargs(self):
        e = ExcitationFile("data/wave.txt", kind="linear", fill_value="extrapolate")
        assert e.filepath == "data/wave.txt"
        assert e.kind == "linear"
        assert e.fill_value == "extrapolate"
        assert e.kwargs == {
            "filepath": "data/wave.txt",
            "kind": "linear",
            "fill_value": "extrapolate",
        }

    def test_optional_kwargs_default_to_none(self):
        e = ExcitationFile("x.txt")
        assert e.kind is None
        assert e.fill_value is None

    def test_order_and_hash(self):
        e = ExcitationFile("x.txt")
        assert e.order == 1
        assert e.hash == "#excitation_file"

    def test_build_accepts_a_single_waveform_column(self, stub_grid, tmp_path):
        path = tmp_path / "single.txt"
        path.write_text("pulse\n0\n1\n0\n", encoding="utf-8")

        ExcitationFile(path).build(stub_grid)

        assert [waveform.ID for waveform in stub_grid.waveforms] == ["pulse"]
        assert float(stub_grid.waveforms[0].userfunc(0.0)) == 0.0

    def test_build_rejects_header_data_column_mismatch(self, stub_grid, tmp_path):
        path = tmp_path / "mismatch.txt"
        path.write_text("first second\n0\n1\n", encoding="utf-8")

        with pytest.raises(ValueError, match="header declares"):
            ExcitationFile(path).build(stub_grid)

    @pytest.mark.parametrize(
        "content,error",
        [
            ("time pulse\n0 0\n0 1\n", "strictly increasing"),
            ("pulse\n0\nnan\n", "finite"),
            ("pulse\n0\n", "at least two"),
        ],
    )
    def test_build_rejects_invalid_waveform_tables(
        self, stub_grid, tmp_path, content, error
    ):
        path = tmp_path / "invalid.txt"
        path.write_text(content, encoding="utf-8")

        with pytest.raises(ValueError, match=error):
            ExcitationFile(path).build(stub_grid)


# ---------------------------------------------------------------------------
# Waveform
# ---------------------------------------------------------------------------


class TestWaveform:
    def test_constructor_stores_kwargs(self):
        w = Waveform(wave_type="gaussian", amp=1.0, freq=1e9, id="wf1")
        assert w.kwargs == {
            "wave_type": "gaussian",
            "amp": 1.0,
            "freq": 1e9,
            "id": "wf1",
        }

    def test_order_and_hash(self):
        w = Waveform()
        assert w.order == 2
        assert w.hash == "#waveform"

    def test_build_builtin_appends_waveform(self, stub_grid):
        w = Waveform(wave_type="gaussian", amp=1.0, freq=1e9, id="wf1")
        w.build(stub_grid)
        assert len(stub_grid.waveforms) == 1
        wv = stub_grid.waveforms[0]
        assert wv.ID == "wf1"
        assert wv.type == "gaussian"
        assert wv.amp == 1.0
        assert wv.freq == 1e9

    def test_build_unknown_wavetype_raises(self, stub_grid):
        w = Waveform(wave_type="notarealwave", amp=1.0, freq=1e9, id="wf1")
        with pytest.raises(ValueError):
            w.build(stub_grid)

    def test_build_missing_wavetype_raises(self, stub_grid):
        w = Waveform(amp=1.0, freq=1e9, id="wf1")
        with pytest.raises(KeyError):
            w.build(stub_grid)

    def test_build_zero_frequency_raises(self, stub_grid):
        w = Waveform(wave_type="gaussian", amp=1.0, freq=0, id="wf1")
        with pytest.raises(ValueError):
            w.build(stub_grid)

    @pytest.mark.parametrize("field,value", [("amp", np.nan), ("freq", np.inf)])
    def test_build_rejects_non_finite_builtin_parameters(self, stub_grid, field, value):
        values = {"wave_type": "gaussian", "amp": 1.0, "freq": 1e9, "id": "wf1"}
        values[field] = value
        with pytest.raises(ValueError, match="finite|greater than zero"):
            Waveform(**values).build(stub_grid)

    def test_build_duplicate_id_raises(self, stub_grid):
        stub_grid.waveforms.append(make_waveform("wf1"))
        w = Waveform(wave_type="gaussian", amp=1.0, freq=1e9, id="wf1")
        with pytest.raises(ValueError):
            w.build(stub_grid)


# ---------------------------------------------------------------------------
# VoltageSource
# ---------------------------------------------------------------------------


class TestVoltageSource:
    def test_constructor_stores_attributes_and_kwargs(self):
        v = VoltageSource(
            p1=(0.05, 0.05, 0.05),
            polarisation="x",
            resistance=50.0,
            waveform_id="wf1",
        )
        assert v.point == (0.05, 0.05, 0.05)
        assert v.polarisation == "x"
        assert v.resistance == 50.0
        assert v.waveform_id == "wf1"
        assert v.start is None
        assert v.stop is None
        assert v.kwargs["p1"] == (0.05, 0.05, 0.05)
        assert v.kwargs["resistance"] == 50.0

    def test_order_and_hash(self):
        v = VoltageSource(p1=(0, 0, 0), polarisation="x", resistance=50, waveform_id="wf1")
        assert v.order == 3
        assert v.hash == "#voltage_source"

    def test_rotatable_defaults(self):
        v = VoltageSource(p1=(0, 0, 0), polarisation="x", resistance=50, waveform_id="wf1")
        assert v.do_rotate is False
        assert v.axis == "x"
        assert v.angle == 0
        assert v.origin is None

    def test_rotate_flips_do_rotate(self):
        v = VoltageSource(p1=(0, 0, 0), polarisation="x", resistance=50, waveform_id="wf1")
        v.rotate("z", 90, origin=(0, 0, 0))
        assert v.do_rotate is True
        assert v.axis == "z"
        assert v.angle == 90

    def test_validate_rejects_unknown_polarisation(self, stub_grid):
        stub_grid.waveforms.append(make_waveform("wf1"))
        v = VoltageSource(p1=(0, 0, 0), polarisation="w", resistance=50, waveform_id="wf1")
        with pytest.raises(ValueError):
            v._validate_parameters(stub_grid)

    def test_validate_rejects_negative_resistance(self, stub_grid):
        stub_grid.waveforms.append(make_waveform("wf1"))
        v = VoltageSource(p1=(0, 0, 0), polarisation="x", resistance=-1, waveform_id="wf1")
        with pytest.raises(ValueError):
            v._validate_parameters(stub_grid)

    @pytest.mark.parametrize("resistance", [np.nan, np.inf])
    def test_validate_rejects_non_finite_resistance(self, stub_grid, resistance):
        stub_grid.waveforms.append(make_waveform("wf1"))
        v = VoltageSource(
            p1=(0, 0, 0), polarisation="x", resistance=resistance, waveform_id="wf1"
        )
        with pytest.raises(ValueError, match="finite source resistance"):
            v._validate_parameters(stub_grid)

    def test_validate_rejects_missing_waveform(self, stub_grid):
        v = VoltageSource(p1=(0, 0, 0), polarisation="x", resistance=50, waveform_id="ghost")
        with pytest.raises(ValueError):
            v._validate_parameters(stub_grid)

    def test_validate_rejects_negative_start(self, stub_grid):
        stub_grid.waveforms.append(make_waveform("wf1"))
        v = VoltageSource(
            p1=(0, 0, 0),
            polarisation="x",
            resistance=50,
            waveform_id="wf1",
            start=-1.0,
            stop=1.0,
        )
        with pytest.raises(ValueError):
            v._validate_parameters(stub_grid)

    def test_validate_rejects_zero_duration(self, stub_grid):
        stub_grid.waveforms.append(make_waveform("wf1"))
        v = VoltageSource(
            p1=(0, 0, 0),
            polarisation="x",
            resistance=50,
            waveform_id="wf1",
            start=1.0,
            stop=1.0,
        )
        with pytest.raises(ValueError):
            v._validate_parameters(stub_grid)


# ---------------------------------------------------------------------------
# HertzianDipole
# ---------------------------------------------------------------------------


class TestHertzianDipole:
    def test_constructor_stores_attributes_and_kwargs(self):
        h = HertzianDipole(p1=(0, 0, 0), polarisation="Y", waveform_id="wf1")
        # ``polarisation`` is normalised to lower-case in __init__
        assert h.polarisation == "y"
        assert h.point == (0, 0, 0)
        assert h.waveform_id == "wf1"
        assert h.kwargs["p1"] == (0, 0, 0)

    def test_order_and_hash(self):
        h = HertzianDipole(p1=(0, 0, 0), polarisation="x", waveform_id="wf1")
        assert h.order == 4
        assert h.hash == "#hertzian_dipole"

    def test_validate_rejects_missing_waveform(self, stub_grid):
        h = HertzianDipole(p1=(0, 0, 0), polarisation="x", waveform_id="ghost")
        with pytest.raises(ValueError):
            h._validate_parameters(stub_grid)


# ---------------------------------------------------------------------------
# MagneticDipole
# ---------------------------------------------------------------------------


class TestMagneticDipole:
    def test_constructor_stores_attributes_and_kwargs(self):
        m = MagneticDipole(p1=(0, 0, 0), polarisation="z", waveform_id="wf1")
        assert m.polarisation == "z"
        assert m.kwargs["waveform_id"] == "wf1"

    def test_order_and_hash(self):
        m = MagneticDipole(p1=(0, 0, 0), polarisation="x", waveform_id="wf1")
        assert m.order == 5
        assert m.hash == "#magnetic_dipole"

    def test_validate_rejects_bad_polarisation(self, stub_grid):
        stub_grid.waveforms.append(make_waveform("wf1"))
        m = MagneticDipole(p1=(0, 0, 0), polarisation="q", waveform_id="wf1")
        with pytest.raises(ValueError):
            m._validate_parameters(stub_grid)


# ---------------------------------------------------------------------------
# TransmissionLine
# ---------------------------------------------------------------------------


class TestTransmissionLine:
    def test_constructor_stores_attributes_and_kwargs(self):
        t = TransmissionLine(p1=(0, 0, 0), polarisation="x", resistance=50, waveform_id="wf1")
        assert t.resistance == 50
        assert t.kwargs["polarisation"] == "x"

    def test_order_and_hash(self):
        t = TransmissionLine(p1=(0, 0, 0), polarisation="x", resistance=50, waveform_id="wf1")
        assert t.order == 6
        assert t.hash == "#transmission_line"

    def test_validate_now_accepts_cuda_solver(self, stub_grid, user_object_config):
        """Upstream removed the CUDA-solver rejection for transmission lines."""
        user_object_config.sim_config.general["solver"] = "cuda"
        stub_grid.waveforms.append(make_waveform("wf1"))
        t = TransmissionLine(p1=(0, 0, 0), polarisation="x", resistance=50, waveform_id="wf1")
        t._validate_parameters(stub_grid)  # no longer raises

    def test_validate_rejects_zero_resistance(self, stub_grid):
        stub_grid.waveforms.append(make_waveform("wf1"))
        t = TransmissionLine(p1=(0, 0, 0), polarisation="x", resistance=0, waveform_id="wf1")
        with pytest.raises(ValueError):
            t._validate_parameters(stub_grid)

    def test_validate_rejects_resistance_above_free_space_impedance(self, stub_grid):
        stub_grid.waveforms.append(make_waveform("wf1"))
        t = TransmissionLine(p1=(0, 0, 0), polarisation="x", resistance=400.0, waveform_id="wf1")
        with pytest.raises(ValueError):
            t._validate_parameters(stub_grid)


# ---------------------------------------------------------------------------
# DiscretePlaneWave family
# ---------------------------------------------------------------------------


class TestDiscretePlaneWaveAngles:
    def test_constructor_stores_kwargs(self):
        d = DiscretePlaneWaveAngles(
            theta=30,
            phi=45,
            psi=0,
            p1=(0, 0, 0),
            p2=(0.1, 0.1, 0.1),
            waveform_id="wf1",
        )
        assert d.kwargs["theta"] == 30
        assert d.kwargs["phi"] == 45
        assert d.kwargs["waveform_id"] == "wf1"

    def test_order_and_hash(self):
        d = DiscretePlaneWaveAngles()
        assert d.order == 19
        assert d.hash == "#plane_wave_angles"


class TestDiscretePlaneWaveVector:
    def test_constructor_stores_kwargs(self):
        d = DiscretePlaneWaveVector(
            m_vec=(1, 0, 0),
            psi=0,
            p1=(0, 0, 0),
            p2=(0.1, 0.1, 0.1),
            waveform_id="wf1",
        )
        assert d.kwargs["m_vec"] == (1, 0, 0)

    def test_order_and_hash(self):
        d = DiscretePlaneWaveVector()
        assert d.order == 22
        assert d.hash == "#plane_wave_vector"


class TestDiscretePlaneWaveAxial:
    def test_constructor_stores_kwargs(self):
        d = DiscretePlaneWaveAxial(
            axis="x",
            psi=0,
            p1=(0, 0, 0),
            p2=(0.1, 0.1, 0.1),
            waveform_id="wf1",
        )
        assert d.kwargs["axis"] == "x"

    def test_order_and_hash(self):
        d = DiscretePlaneWaveAxial()
        assert d.order == 20
        assert d.hash == "#plane_wave_axial"


class TestDPWVectorMOverwriteBug:
    """Bug tripwire: ``cmds_multiuse.py:1158/1165``.

    ``DiscretePlaneWaveVector.build`` sets ``DPW.m = np.array(m_vec, …)``
    on line 1158, then immediately overwrites with ``np.zeros(...)`` on
    line 1165. The user-supplied vector is silently discarded.

    We pin the bug by checking the literal source — the two consecutive
    ``DPW.m =`` assignments mean the second wins. When fixed (drop the
    ``np.zeros`` reassignment), this tripwire should fail.
    """

    def test_build_now_reassigns_m_correctly(self):
        """Upstream partially fixed: the ``DPW.m = np.zeros`` line remains
        (allocates the array) but ``m_vec`` is now copied into it instead
        of being discarded."""
        import inspect

        src = inspect.getsource(DiscretePlaneWaveVector.build)
        assert "DPW.m[:3] = np.array(m_vec" in src


class TestDPWPrecomputeValidation:

    @pytest.mark.parametrize(
        "cls",
        [DiscretePlaneWaveAngles, DiscretePlaneWaveVector, DiscretePlaneWaveAxial],
    )
    def test_build_does_not_silently_override_precompute(self, cls):
        import inspect

        src = inspect.getsource(cls.build)
        assert src.count("precompute = True") == 1
        assert "_validate_dpw_precompute(self, precompute)" in src


# ---------------------------------------------------------------------------
# Rx
# ---------------------------------------------------------------------------


class TestRx:
    def test_constructor_stores_attributes_and_kwargs(self):
        r = Rx(p1=(0.05, 0.05, 0.05), id="rx1", outputs=["Ex", "Ey"])
        assert r.point == (0.05, 0.05, 0.05)
        assert r.id == "rx1"
        assert r.outputs == ["Ex", "Ey"]
        assert r.kwargs["p1"] == (0.05, 0.05, 0.05)

    def test_optional_kwargs_default_to_none(self):
        r = Rx(p1=(0, 0, 0))
        assert r.id is None
        assert r.outputs is None

    def test_order_and_hash(self):
        r = Rx(p1=(0, 0, 0))
        assert r.order == 7
        assert r.hash == "#rx"

    def test_rotatable_defaults(self):
        r = Rx(p1=(0, 0, 0))
        assert r.do_rotate is False


# ---------------------------------------------------------------------------
# RxArray
# ---------------------------------------------------------------------------


class TestRxArray:
    def test_constructor_stores_attributes_and_kwargs(self):
        a = RxArray(p1=(0, 0, 0), p2=(0.1, 0.1, 0.1), dl=(0.01, 0.01, 0.01))
        assert a.lower_point == (0, 0, 0)
        assert a.upper_point == (0.1, 0.1, 0.1)
        assert a.dl == (0.01, 0.01, 0.01)

    def test_order_and_hash(self):
        a = RxArray(p1=(0, 0, 0), p2=(0.1, 0.1, 0.1), dl=(0.01, 0.01, 0.01))
        assert a.order == 8
        assert a.hash == "#rx_array"


class TestRxArrayUpperBoundBug:
    """Bug tripwire: ``cmds_multiuse.py:1531-1533``.

    ``RxArray.build`` passes ``self.lower_point`` to **both** of its
    ``uip.check_src_rx_point`` calls — the second one should pass
    ``self.upper_point`` to check the upper-right corner. The upper bound
    of the array is effectively never validated against the grid.

    We pin the bug by mocking the uip and confirming both calls receive
    ``lower_point``. When fixed, the second call's positional ``point``
    argument should become ``upper_point``.
    """

    def test_both_check_src_rx_calls_currently_receive_lower_point(self, stub_grid):
        a = RxArray(p1=(0.0, 0.0, 0.0), p2=(0.1, 0.2, 0.3), dl=(0.01, 0.01, 0.01))

        uip = MagicMock()
        uip.check_src_rx_point.return_value = (True, np.array([0, 0, 0]))
        uip.discretise_static_point.return_value = np.array([10, 10, 10])
        uip.round_to_grid_static_point.return_value = (0.0, 0.0, 0.0)

        # Mock Rx so its inner build chain does not run; we just want to
        # observe the two check_src_rx_point arguments.
        with patch.object(RxArray, "_create_uip", return_value=uip), patch(
            "gprMax.user_objects.cmds_multiuse.Rx"
        ):
            try:
                a.build(stub_grid)
            except Exception:
                # We don't care if downstream blows up — only the two
                # check_src_rx_point calls matter for this tripwire.
                pass

        calls = uip.check_src_rx_point.call_args_list
        assert len(calls) >= 2
        # Both calls pass the LOWER point (the bug)
        assert calls[0].args[0] == a.lower_point
        assert calls[1].args[0] == a.lower_point


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------


class TestMaterial:
    def test_constructor_stores_kwargs(self):
        m = Material(er=2.0, se=0.0, mr=1.0, sm=0.0, id="dirt")
        assert m.kwargs == {"er": 2.0, "se": 0.0, "mr": 1.0, "sm": 0.0, "id": "dirt"}

    def test_order_and_hash(self):
        m = Material()
        assert m.order == 10
        assert m.hash == "#material"

    def test_build_appends_material(self, stub_grid):
        m = Material(er=2.0, se=0.0, mr=1.0, sm=0.0, id="dirt")
        m.build(stub_grid)
        names = [x.ID for x in stub_grid.materials]
        assert "dirt" in names

    def test_build_rejects_low_er(self, stub_grid):
        m = Material(er=0.5, se=0.0, mr=1.0, sm=0.0, id="x")
        with pytest.raises(ValueError):
            m.build(stub_grid)

    def test_build_rejects_low_mr(self, stub_grid):
        m = Material(er=1.0, se=0.0, mr=0.5, sm=0.0, id="x")
        with pytest.raises(ValueError):
            m.build(stub_grid)

    def test_build_rejects_negative_sm(self, stub_grid):
        m = Material(er=1.0, se=0.0, mr=1.0, sm=-1.0, id="x")
        with pytest.raises(ValueError):
            m.build(stub_grid)

    @pytest.mark.parametrize(
        "field,value",
        [("er", np.nan), ("er", np.inf), ("mr", np.nan), ("mr", np.inf),
         ("se", np.nan), ("sm", np.nan)],
    )
    def test_build_rejects_non_finite_non_pec_parameters(self, stub_grid, field, value):
        values = {"er": 2.0, "se": 0.0, "mr": 1.0, "sm": 0.0, "id": "bad"}
        values[field] = value
        with pytest.raises(ValueError):
            Material(**values).build(stub_grid)

    def test_build_accepts_infinite_conductivity_string(self, stub_grid):
        # ``se="inf"`` is a documented sentinel for PEC-like materials
        m = Material(er=1.0, se="inf", mr=1.0, sm=0.0, id="pec_like")
        m.build(stub_grid)
        new_mat = [x for x in stub_grid.materials if x.ID == "pec_like"][0]
        assert new_mat.se == float("inf")
        assert new_mat.averagable is False

    def test_build_rejects_duplicate_id(self, stub_grid):
        m1 = Material(er=2.0, se=0.0, mr=1.0, sm=0.0, id="dup")
        m1.build(stub_grid)
        m2 = Material(er=3.0, se=0.0, mr=1.0, sm=0.0, id="dup")
        with pytest.raises(ValueError):
            m2.build(stub_grid)

    def test_build_missing_kwarg_raises(self, stub_grid):
        m = Material(er=2.0, se=0.0, mr=1.0)  # missing sm, id
        with pytest.raises(KeyError):
            m.build(stub_grid)


# ---------------------------------------------------------------------------
# AddDebyeDispersion / AddLorentzDispersion / AddDrudeDispersion
# ---------------------------------------------------------------------------


class TestAddDebyeDispersion:
    def test_constructor_stores_kwargs(self):
        d = AddDebyeDispersion(poles=2, er_delta=(1.0, 2.0), tau=(1e-9, 2e-9), material_ids=["m1"])
        assert d.kwargs["poles"] == 2
        assert d.kwargs["material_ids"] == ["m1"]

    def test_order_and_hash(self):
        d = AddDebyeDispersion()
        assert d.order == 11
        assert d.hash == "#add_dispersion_debye"

    def test_build_negative_poles_raises(self, stub_grid):
        d = AddDebyeDispersion(poles=-1, er_delta=(1.0,), tau=(1e-9,), material_ids=["free_space"])
        with pytest.raises(ValueError):
            d.build(stub_grid)

    def test_build_unknown_material_raises(self, stub_grid):
        d = AddDebyeDispersion(poles=1, er_delta=(1.0,), tau=(1e-9,), material_ids=["ghost"])
        with pytest.raises(ValueError):
            d.build(stub_grid)

    @pytest.mark.parametrize("er_delta", (0.0, -1.0, float("inf"), float("nan")))
    def test_build_rejects_invalid_permittivity_difference(
        self, stub_grid, user_object_config, er_delta
    ):
        user_object_config.model_config.dispersive_averaging = True
        stub_grid.materials.append(RuntimeMaterial(2, "sample"))
        dispersion = AddDebyeDispersion(
            poles=1,
            er_delta=(er_delta,),
            tau=(1e-9,),
            material_ids=["sample"],
        )

        with pytest.raises(ValueError, match="finite, positive relative-permittivity"):
            dispersion.build(stub_grid)

    @pytest.mark.parametrize("tau", (0.0, -1e-9, float("inf"), float("nan")))
    def test_build_rejects_invalid_relaxation_time(self, stub_grid, user_object_config, tau):
        user_object_config.model_config.dispersive_averaging = True
        stub_grid.materials.append(RuntimeMaterial(2, "sample"))
        dispersion = AddDebyeDispersion(
            poles=1,
            er_delta=(1.0,),
            tau=(tau,),
            material_ids=["sample"],
        )

        with pytest.raises(ValueError, match="finite, positive relaxation times"):
            dispersion.build(stub_grid)


class TestAddLorentzDispersion:
    def test_constructor_stores_kwargs(self):
        d = AddLorentzDispersion(
            poles=1, er_delta=(1.0,), omega=(1e9,), delta=(1e7,), material_ids=["m1"]
        )
        assert d.kwargs["poles"] == 1

    def test_order_and_hash(self):
        d = AddLorentzDispersion()
        assert d.order == 12
        assert d.hash == "#add_dispersion_lorentz"

    def test_build_rejects_critical_pole(self, stub_grid, user_object_config):
        user_object_config.model_config.dispersive_averaging = True
        stub_grid.materials.append(RuntimeMaterial(2, "sample"))
        frequency = 10e9
        dispersion = AddLorentzDispersion(
            poles=1,
            er_delta=(1.0,),
            omega=(frequency,),
            delta=(2.0 * np.pi * frequency,),
            material_ids=["sample"],
        )

        with pytest.raises(ValueError, match=r"damping coefficient.*2 \* pi"):
            dispersion.build(stub_grid)

    def test_build_rejects_frequency_at_timestep_limit(self, stub_grid, user_object_config):
        user_object_config.model_config.dispersive_averaging = True
        stub_grid.materials.append(RuntimeMaterial(2, "sample"))
        dispersion = AddLorentzDispersion(
            poles=1,
            er_delta=(1.0,),
            omega=(1.0 / stub_grid.dt,),
            delta=(1e8,),
            material_ids=["sample"],
        )

        with pytest.raises(ValueError, match=r"frequency must be below 1 / dt"):
            dispersion.build(stub_grid)

    @pytest.mark.parametrize(("omega", "delta"), ((float("nan"), 1e8), (1e9, float("inf"))))
    def test_build_rejects_non_finite_pole(self, stub_grid, user_object_config, omega, delta):
        user_object_config.model_config.dispersive_averaging = True
        stub_grid.materials.append(RuntimeMaterial(2, "sample"))
        dispersion = AddLorentzDispersion(
            poles=1,
            er_delta=(1.0,),
            omega=(omega,),
            delta=(delta,),
            material_ids=["sample"],
        )

        with pytest.raises(ValueError, match="must be finite"):
            dispersion.build(stub_grid)


class TestAddDrudeDispersion:
    def test_constructor_stores_kwargs(self):
        d = AddDrudeDispersion(poles=1, omega=(1e9,), alpha=(1e7,), material_ids=["m1"])
        assert d.kwargs["poles"] == 1

    def test_order_and_hash(self):
        d = AddDrudeDispersion()
        assert d.order == 13
        assert d.hash == "#add_dispersion_drude"

    @pytest.mark.parametrize(("omega", "alpha"), ((float("nan"), 1e8), (1e9, float("inf"))))
    def test_build_rejects_non_finite_pole(self, stub_grid, user_object_config, omega, alpha):
        user_object_config.model_config.dispersive_averaging = True
        stub_grid.materials.append(RuntimeMaterial(2, "sample"))
        dispersion = AddDrudeDispersion(
            poles=1,
            omega=(omega,),
            alpha=(alpha,),
            material_ids=["sample"],
        )

        with pytest.raises(ValueError, match="must be finite"):
            dispersion.build(stub_grid)

    def test_build_rejects_frequency_at_timestep_limit(self, stub_grid, user_object_config):
        user_object_config.model_config.dispersive_averaging = True
        stub_grid.materials.append(RuntimeMaterial(2, "sample"))
        dispersion = AddDrudeDispersion(
            poles=1,
            omega=(1.0 / stub_grid.dt,),
            alpha=(1e8,),
            material_ids=["sample"],
        )

        with pytest.raises(ValueError, match=r"frequency must be below 1 / dt"):
            dispersion.build(stub_grid)


# ---------------------------------------------------------------------------
# SoilPeplinski
# ---------------------------------------------------------------------------


class TestSoilPeplinski:
    def test_constructor_stores_kwargs(self):
        s = SoilPeplinski(
            sand_fraction=0.5,
            clay_fraction=0.3,
            bulk_density=2.0,
            sand_density=2.66,
            water_fraction_lower=0.001,
            water_fraction_upper=0.25,
            id="soil1",
        )
        assert s.kwargs["sand_fraction"] == 0.5
        assert s.kwargs["id"] == "soil1"

    def test_order_and_hash(self):
        s = SoilPeplinski()
        assert s.order == 14
        assert s.hash == "#soil_peplinski"

    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_build_rejects_non_finite_parameters(self, stub_grid, value):
        soil = SoilPeplinski(
            sand_fraction=value,
            clay_fraction=0.3,
            bulk_density=2.0,
            sand_density=2.66,
            water_fraction_lower=0.001,
            water_fraction_upper=0.25,
            id="soil1",
        )
        with pytest.raises(ValueError, match="must all be finite"):
            soil.build(stub_grid)


# ---------------------------------------------------------------------------
# MaterialRange / MaterialList
# ---------------------------------------------------------------------------


class TestMaterialRange:
    def test_constructor_stores_kwargs(self):
        m = MaterialRange(
            er_lower=1.0,
            er_upper=2.0,
            sigma_lower=0.0,
            sigma_upper=0.1,
            mr_lower=1.0,
            mr_upper=2.0,
            ro_lower=0.0,
            ro_upper=0.1,
            id="rng",
        )
        assert m.kwargs["er_upper"] == 2.0

    def test_order_and_hash(self):
        m = MaterialRange()
        assert m.order == 15
        assert m.hash == "#material_range"

    def _make(self, **overrides):
        values = dict(
            er_lower=1.0,
            er_upper=2.0,
            sigma_lower=0.0,
            sigma_upper=0.1,
            mr_lower=1.0,
            mr_upper=2.0,
            ro_lower=0.0,
            ro_upper=0.1,
            id="rng",
        )
        values.update(overrides)
        return MaterialRange(**values)

    def test_build_rejects_negative_upper_magnetic_loss(self, stub_grid):
        with pytest.raises(ValueError):
            self._make(ro_upper=-0.1).build(stub_grid)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"er_lower": 3.0, "er_upper": 2.0},
            {"sigma_lower": 0.2, "sigma_upper": 0.1},
            {"mr_lower": 3.0, "mr_upper": 2.0},
            {"ro_lower": 0.2, "ro_upper": 0.1},
        ],
    )
    def test_build_rejects_reversed_ranges(self, stub_grid, overrides):
        with pytest.raises(ValueError, match="must not exceed"):
            self._make(**overrides).build(stub_grid)

    def test_build_rejects_nan_limit(self, stub_grid):
        with pytest.raises(ValueError, match="must all be finite"):
            self._make(sigma_upper=np.nan).build(stub_grid)


class TestMaterialList:
    def test_constructor_stores_kwargs(self):
        m = MaterialList(list_of_materials=["m1", "m2"], id="list1")
        assert m.kwargs["list_of_materials"] == ["m1", "m2"]


class TestMaterialListHash:
    def test_material_list_and_range_have_distinct_hashes(self):
        ml = MaterialList()
        mr = MaterialRange()
        assert ml.hash == "#material_list"
        assert mr.hash == "#material_range"
        assert ml.hash != mr.hash


class TestMaterialCrim:
    def test_constructor_stores_kwargs(self):
        material = MaterialCrim(
            matrix_id="sand",
            matrix_fraction=0.6,
            dispersive_id="water",
            fraction_lower=0.02,
            fraction_upper=0.35,
            f_min=1e6,
            f_max=3e9,
            a=0.5,
            id="wetsand",
        )

        assert material.kwargs["matrix_id"] == "sand"
        assert material.kwargs["fraction_upper"] == 0.35
        assert material.kwargs["id"] == "wetsand"

    def test_order_and_hash(self):
        material = MaterialCrim()
        assert material.order == 15
        assert material.hash == "#material_crim"

    @pytest.mark.parametrize(
        "overrides",
        [
            {"matrix_fraction": -0.1},
            {"fraction_lower": -0.1},
            {"fraction_lower": 0.4, "fraction_upper": 0.3},
            {"matrix_fraction": 0.8, "fraction_upper": 0.3},
            {"f_min": 0},
            {"f_min": 2e9, "f_max": 1e9},
            {"a": 0},
            {"a": np.nan},
        ],
    )
    def test_build_rejects_invalid_parameters(self, stub_grid, overrides):
        kwargs = {
            "matrix_id": "sand",
            "matrix_fraction": 0.6,
            "dispersive_id": "water",
            "fraction_lower": 0.02,
            "fraction_upper": 0.35,
            "f_min": 1e6,
            "f_max": 3e9,
            "a": 0.5,
            "id": "wetsand",
        }
        kwargs.update(overrides)

        with pytest.raises(ValueError):
            MaterialCrim(**kwargs).build(stub_grid)

    def test_build_registers_mixing_model(self, stub_grid):
        material = MaterialCrim(
            matrix_id="sand",
            matrix_fraction=0.6,
            dispersive_id="water",
            fraction_lower=0.02,
            fraction_upper=0.35,
            f_min=1e6,
            f_max=3e9,
            a=0.5,
            id="wetsand",
        )
        material.build(stub_grid)

        assert len(stub_grid.mixingmodels) == 1
        assert stub_grid.mixingmodels[0].ID == "wetsand"


# ---------------------------------------------------------------------------
# PMLCFS
# ---------------------------------------------------------------------------


class TestPMLCFS:
    def test_constructor_stores_kwargs(self):
        p = PMLCFS(
            alphascalingprofile="constant",
            alphascalingdirection="forward",
            alphamin=0.0,
            alphamax=1.0,
            kappascalingprofile="constant",
            kappascalingdirection="forward",
            kappamin=1.0,
            kappamax=1.0,
            sigmascalingprofile="quartic",
            sigmascalingdirection="forward",
            sigmamin=0.0,
            sigmamax=1.0,
        )
        assert p.kwargs["alphascalingprofile"] == "constant"

    def test_order_and_hash(self):
        p = PMLCFS()
        assert p.order == 19
        assert p.hash == "#pml_cfs"


pytestmark = pytest.mark.unit
