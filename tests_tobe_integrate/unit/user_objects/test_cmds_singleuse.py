"""Tests for ``gprMax.user_objects.cmds_singleuse``.

12 once-per-model user-object classes. For each class we check the
common contract:

* constructor stores both ``self.kwargs`` (parser path) AND mirrored
  ``self.<attr>`` attributes (``build()`` path);
* ``order`` and ``hash`` properties match the documented values;
* ``__str__()`` round-trips into a hash-command line;
* ``build(model)`` validation branches fire on bad inputs.

A handful of bug tripwires pin current behaviour for fixes to flip.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gprMax.user_objects.cmds_singleuse import (
    Discretisation,
    Domain,
    OMPThreads,
    OutputDir,
    PMLFormulation,
    PMLProps,
    PMLThickness,
    RxSteps,
    SrcSteps,
    TimeStepStabilityFactor,
    TimeWindow,
    Title,
)


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------


class TestTitle:
    def test_constructor_stores_attribute_and_kwargs(self):
        t = Title("my model")
        assert t.title == "my model"
        assert t.kwargs == {"name": "my model"}

    def test_order_and_hash(self):
        t = Title("x")
        assert t.order == 1
        assert t.hash == "#title"

    def test_str_round_trip(self):
        assert str(Title("demo")) == "#title: demo"

    def test_build_assigns_title_to_model(self, stub_model):
        Title("the title").build(stub_model)
        assert stub_model.title == "the title"


# ---------------------------------------------------------------------------
# Discretisation
# ---------------------------------------------------------------------------


class TestDiscretisation:
    def test_constructor_stores_attribute_and_kwargs(self):
        d = Discretisation((0.001, 0.002, 0.003))
        assert d.discretisation == (0.001, 0.002, 0.003)
        assert d.kwargs == {"p1": (0.001, 0.002, 0.003)}

    def test_order_and_hash(self):
        d = Discretisation((0.001, 0.001, 0.001))
        assert d.order == 2
        assert d.hash == "#dx_dy_dz"

    def test_str_round_trip(self):
        # Tuples are expanded by ``UserObject.__str__``
        assert str(Discretisation((0.001, 0.002, 0.003))) == "#dx_dy_dz: 0.001 0.002 0.003"

    def test_build_sets_model_dl(self, stub_model):
        Discretisation((0.001, 0.002, 0.003)).build(stub_model)
        np.testing.assert_array_equal(stub_model.dl, np.array([0.001, 0.002, 0.003]))


class TestDiscretisationAnyBug:
    """Bug tripwire: ``cmds_singleuse.py:89``.

    The validation reads ``if any(self.discretisation) <= 0:``. ``any()``
    returns a bool, so this is ``True <= 0`` (always False) for any
    discretisation that contains a non-zero element. Negative step sizes
    silently pass through. The intended guard is
    ``any(d <= 0 for d in self.discretisation)``.

    When fixed, this tripwire should be replaced with a ``raises`` on
    negative input.
    """

    def test_negative_step_currently_does_not_raise(self, stub_model):
        # Bug: any((-1, -1, -1)) is True, True <= 0 is False → no raise
        Discretisation((-1.0, -1.0, -1.0)).build(stub_model)

    def test_all_zero_step_still_raises(self, stub_model):
        # any((0, 0, 0)) is False, False <= 0 is True → raises
        with pytest.raises(ValueError):
            Discretisation((0.0, 0.0, 0.0)).build(stub_model)


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------


class TestDomain:
    def test_constructor_stores_attribute_and_kwargs(self):
        d = Domain((0.2, 0.3, 0.4))
        assert d.domain_size == (0.2, 0.3, 0.4)
        assert d.kwargs == {"p1": (0.2, 0.3, 0.4)}

    def test_order_and_hash(self):
        d = Domain((0.1, 0.1, 0.1))
        assert d.order == 3
        assert d.hash == "#domain"

    def test_str_round_trip(self):
        assert str(Domain((0.1, 0.2, 0.3))) == "#domain: 0.1 0.2 0.3"

    def test_build_calls_model_set_size(self, stub_model):
        # Patch _create_uip so we don't touch UserInput internals
        uip = MagicMock()
        uip.discretise_static_point.return_value = np.array([100, 100, 100])
        with patch.object(Domain, "_create_uip", return_value=uip):
            Domain((0.1, 0.1, 0.1)).build(stub_model)
        stub_model.set_size.assert_called_once()
        stub_model.G.calculate_dt.assert_called_once()

    def test_build_raises_when_a_dimension_is_zero(self, stub_model):
        uip = MagicMock()
        uip.discretise_static_point.return_value = np.array([100, 100, 100])
        stub_model.nx = 0  # Triggers "at least one cell" check
        with patch.object(Domain, "_create_uip", return_value=uip):
            with pytest.raises(ValueError):
                Domain((0.1, 0.1, 0.1)).build(stub_model)

    @pytest.mark.parametrize(
        "nx,ny,nz,expected_mode",
        [(1, 50, 50, "2D TMx"), (50, 1, 50, "2D TMy"), (50, 50, 1, "2D TMz"), (50, 50, 50, "3D")],
    )
    def test_build_sets_mode(
        self, stub_model, user_object_config, nx, ny, nz, expected_mode
    ):
        stub_model.nx, stub_model.ny, stub_model.nz = nx, ny, nz
        uip = MagicMock()
        uip.discretise_static_point.return_value = np.array([nx, ny, nz])
        with patch.object(Domain, "_create_uip", return_value=uip):
            Domain((0.1, 0.1, 0.1)).build(stub_model)
        assert user_object_config.model_config.mode == expected_mode


# ---------------------------------------------------------------------------
# TimeStepStabilityFactor
# ---------------------------------------------------------------------------


class TestTimeStepStabilityFactor:
    def test_constructor_stores_attribute_and_kwargs(self):
        t = TimeStepStabilityFactor(0.5)
        assert t.stability_factor == 0.5
        assert t.kwargs == {"f": 0.5}

    def test_order_and_hash(self):
        t = TimeStepStabilityFactor(0.5)
        assert t.order == 4
        assert t.hash == "#time_step_stability_factor"

    def test_str_round_trip(self):
        assert str(TimeStepStabilityFactor(0.5)) == "#time_step_stability_factor: 0.5"

    def test_build_applies_factor_to_model_dt(self, stub_model):
        before_dt = stub_model.dt
        TimeStepStabilityFactor(0.5).build(stub_model)
        assert stub_model.dt_mod == 0.5
        assert stub_model.dt == 0.5 * before_dt

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.01, 2.0])
    def test_build_rejects_out_of_range(self, stub_model, bad):
        with pytest.raises(ValueError):
            TimeStepStabilityFactor(bad).build(stub_model)


# ---------------------------------------------------------------------------
# TimeWindow
# ---------------------------------------------------------------------------


class TestTimeWindow:
    def test_default_constructor_both_none(self):
        t = TimeWindow()
        assert t.time is None
        assert t.iterations is None
        assert t.kwargs == {"time": None, "iterations": None}

    def test_constructor_time_only(self):
        t = TimeWindow(time=1e-9)
        assert t.time == 1e-9
        assert t.iterations is None

    def test_constructor_iterations_only(self):
        t = TimeWindow(iterations=100)
        assert t.time is None
        assert t.iterations == 100

    def test_order_and_hash(self):
        assert TimeWindow().order == 5
        assert TimeWindow().hash == "#time_window"

    def test_build_time_mode_sets_iterations(self, stub_model):
        # time=1e-9, dt=1.927e-12 → ceil(1e-9/1.927e-12)+1 = 520
        TimeWindow(time=1e-9).build(stub_model)
        assert stub_model.timewindow == 1e-9
        assert stub_model.iterations == int(np.ceil(1e-9 / stub_model.dt)) + 1

    def test_build_iterations_mode_sets_timewindow(self, stub_model):
        TimeWindow(iterations=100).build(stub_model)
        assert stub_model.iterations == 100
        assert stub_model.timewindow == 99 * stub_model.dt

    def test_build_both_none_raises(self, stub_model):
        with pytest.raises(ValueError):
            TimeWindow().build(stub_model)

    def test_build_negative_time_raises(self, stub_model):
        with pytest.raises(ValueError):
            TimeWindow(time=-1.0).build(stub_model)

    def test_build_both_set_uses_time_branch(self, stub_model):
        # Documented behaviour: ``time`` wins over ``iterations`` and
        # a warning is logged.
        TimeWindow(time=2e-9, iterations=100).build(stub_model)
        assert stub_model.timewindow == 2e-9


# ---------------------------------------------------------------------------
# OMPThreads
# ---------------------------------------------------------------------------


class TestOMPThreads:
    def test_constructor_stores_attribute_and_kwargs(self):
        t = OMPThreads(4)
        assert t.omp_threads == 4
        assert t.kwargs == {"n": 4}

    def test_order(self):
        assert OMPThreads(1).order == 6

    def test_build_sets_model_config_ompthreads(self, stub_model, user_object_config):
        # ``set_omp_threads`` is patched to avoid touching OS threads
        with patch(
            "gprMax.user_objects.cmds_singleuse.set_omp_threads", return_value=4
        ) as set_threads:
            OMPThreads(4).build(stub_model)
        set_threads.assert_called_once_with(4)
        assert user_object_config.model_config.ompthreads == 4

    def test_build_rejects_zero_threads(self, stub_model):
        with pytest.raises(ValueError):
            OMPThreads(0).build(stub_model)


class TestOMPThreadsHashMismatchBug:
    """Bug tripwire: ``cmds_singleuse.py:276``.

    ``OMPThreads.hash`` returns ``"#num_threads"`` but the rest of the
    codebase — the hash dispatcher in ``hash_cmds_singleuse.py:64``, the
    docs at ``docs/source/input_hash_cmds.rst:158``, the parser keys in
    ``hash_cmds_file.py:222`` — all use ``"#omp_threads"``. As a result
    ``str(OMPThreads(4))`` produces ``"#num_threads: 4"``, a string that
    cannot be parsed back by the dispatcher.

    When fixed (rename the property to return ``"#omp_threads"``), this
    tripwire should flip to assert the correct hash.
    """

    def test_hash_currently_diverges_from_dispatcher(self):
        assert OMPThreads(4).hash == "#num_threads"
        # Round-trip through __str__ produces a string the parser can't read
        assert str(OMPThreads(4)) == "#num_threads: 4"


# ---------------------------------------------------------------------------
# PMLFormulation
# ---------------------------------------------------------------------------


class TestPMLFormulation:
    def test_constructor_stores_attribute_and_kwargs(self):
        p = PMLFormulation("HORIPML")
        assert p.formulation == "HORIPML"
        assert p.kwargs == {"formulation": "HORIPML"}

    def test_order_and_hash(self):
        p = PMLFormulation("HORIPML")
        assert p.order == 7
        assert p.hash == "#pml_formulation"

    def test_str_round_trip(self):
        assert str(PMLFormulation("HORIPML")) == "#pml_formulation: HORIPML"

    @pytest.mark.parametrize("formulation", ["HORIPML", "MRIPML"])
    def test_build_accepts_known_formulations(self, stub_model, formulation):
        PMLFormulation(formulation).build(stub_model)
        assert stub_model.G.pmls["formulation"] == formulation

    def test_build_rejects_unknown_formulation(self, stub_model):
        with pytest.raises(ValueError):
            PMLFormulation("UNKNOWN").build(stub_model)


# ---------------------------------------------------------------------------
# PMLThickness
# ---------------------------------------------------------------------------


class TestPMLThickness:
    def test_constructor_scalar_thickness(self):
        p = PMLThickness(10)
        assert p.thickness == 10
        assert p.kwargs == {"thickness": 10}

    def test_constructor_tuple_thickness(self):
        p = PMLThickness((10, 10, 10, 10, 10, 10))
        assert p.thickness == (10, 10, 10, 10, 10, 10)

    def test_order_and_hash(self):
        p = PMLThickness(10)
        assert p.order == 7
        assert p.hash == "#pml_cells"

    def test_build_scalar_calls_set_pml_thickness(self, stub_model):
        PMLThickness(10).build(stub_model)
        stub_model.G.set_pml_thickness.assert_called_once_with(10)

    def test_build_tuple_calls_set_pml_thickness(self, stub_model):
        PMLThickness((10, 10, 10, 10, 10, 10)).build(stub_model)
        stub_model.G.set_pml_thickness.assert_called_once_with((10, 10, 10, 10, 10, 10))

    @pytest.mark.parametrize("bad_len", [2, 3, 4, 5, 7])
    def test_build_rejects_wrong_tuple_length(self, stub_model, bad_len):
        with pytest.raises(ValueError):
            PMLThickness(tuple([10] * bad_len)).build(stub_model)

    def test_build_rejects_pml_thicker_than_half_domain(self, stub_model):
        # nx=50; pml x0=30 ⇒ 2*30 ≥ 50 → fails domain check
        stub_model.G.pmls["thickness"]["x0"] = 30
        with pytest.raises(ValueError):
            PMLThickness(10).build(stub_model)


# ---------------------------------------------------------------------------
# PMLProps (deprecated)
# ---------------------------------------------------------------------------


class TestPMLProps:
    def test_formulation_only(self):
        p = PMLProps(formulation="HORIPML")
        assert isinstance(p.pml_formulation, PMLFormulation)
        assert p.pml_thickness is None

    def test_thickness_only(self):
        p = PMLProps(thickness=10)
        assert p.pml_formulation is None
        assert isinstance(p.pml_thickness, PMLThickness)

    def test_six_face_thicknesses(self):
        p = PMLProps(x0=1, y0=2, z0=3, xmax=4, ymax=5, zmax=6)
        assert isinstance(p.pml_thickness, PMLThickness)
        assert p.pml_thickness.thickness == (1, 2, 3, 4, 5, 6)

    def test_thickness_wins_over_face_kwargs(self):
        p = PMLProps(thickness=10, x0=1, y0=2, z0=3, xmax=4, ymax=5, zmax=6)
        assert p.pml_thickness.thickness == 10

    def test_no_args_raises(self):
        with pytest.raises(ValueError):
            PMLProps()

    def test_partial_face_kwargs_raises(self):
        # Missing zmax → fall-through to the empty branch → ValueError
        with pytest.raises(ValueError):
            PMLProps(x0=1, y0=2, z0=3, xmax=4, ymax=5)

    def test_order_and_hash(self):
        p = PMLProps(formulation="HORIPML")
        assert p.order == 7
        assert p.hash == "#pml_properties"


# ---------------------------------------------------------------------------
# SrcSteps / RxSteps
# ---------------------------------------------------------------------------


class TestSrcSteps:
    def test_constructor_stores_attribute_and_kwargs(self):
        s = SrcSteps((0.01, 0.02, 0.03))
        assert s.step_size == (0.01, 0.02, 0.03)
        assert s.kwargs == {"p1": (0.01, 0.02, 0.03)}

    def test_order_and_hash(self):
        s = SrcSteps((0.0, 0.0, 0.0))
        assert s.order == 8
        assert s.hash == "#src_steps"

    def test_build_writes_to_model_srcsteps(self, stub_model):
        uip = MagicMock()
        uip.discretise_static_point.return_value = np.array([10, 20, 30])
        with patch.object(SrcSteps, "_create_uip", return_value=uip):
            SrcSteps((0.01, 0.02, 0.03)).build(stub_model)
        np.testing.assert_array_equal(stub_model.srcsteps, np.array([10, 20, 30]))


class TestRxSteps:
    def test_constructor_stores_attribute_and_kwargs(self):
        r = RxSteps((0.01, 0.02, 0.03))
        assert r.step_size == (0.01, 0.02, 0.03)
        assert r.kwargs == {"p1": (0.01, 0.02, 0.03)}

    def test_order_and_hash(self):
        r = RxSteps((0.0, 0.0, 0.0))
        assert r.order == 9
        assert r.hash == "#rx_steps"

    def test_build_writes_to_model_rxsteps(self, stub_model):
        uip = MagicMock()
        uip.discretise_static_point.return_value = np.array([10, 20, 30])
        with patch.object(RxSteps, "_create_uip", return_value=uip):
            RxSteps((0.01, 0.02, 0.03)).build(stub_model)
        np.testing.assert_array_equal(stub_model.rxsteps, np.array([10, 20, 30]))


# ---------------------------------------------------------------------------
# OutputDir
# ---------------------------------------------------------------------------


class TestOutputDir:
    def test_constructor_stores_attribute_and_kwargs(self):
        o = OutputDir("results/run1")
        assert o.output_dir == "results/run1"
        assert o.kwargs == {"dir": "results/run1"}

    def test_order_and_hash(self):
        o = OutputDir("x")
        assert o.order == 10
        assert o.hash == "#output_dir"

    def test_str_round_trip(self):
        assert str(OutputDir("out")) == "#output_dir: out"

    def test_build_calls_set_output_file_path(self, stub_model, user_object_config):
        OutputDir("results/run1").build(stub_model)
        user_object_config.model_config.set_output_file_path.assert_called_once_with(
            "results/run1"
        )
