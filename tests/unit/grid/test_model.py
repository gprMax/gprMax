"""Tests for ``gprMax.model.Model``.

``Model`` is a thin owner of exactly one ``FDTDGrid``: almost every attribute
is a property forwarding to ``self.G``. That makes it cheap to test and makes
a forwarding slip invisible in normal use.

The ``dy`` / ``dz`` *getters* are defective — they read ``dl[0]`` — so no
assertion here reads them back. The setters are correct and are pinned, which
is what makes the getter defect demonstrable without asserting broken
behaviour. Write-up: ``notes/bugs/model-dy-dz-getters.md``.

``Model.__init__`` calls ``set_omp_threads``, which reads host CPU information
that has nothing to do with the class under test, so it is patched out here.
Host detection is covered by PR 11.
"""

import numpy as np
import pytest

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.model import Model

from .conftest import DL_ANISO


@pytest.fixture
def model(monkeypatch):
    """A ``Model`` with host-thread detection stubbed out."""
    import gprMax.model as model_module

    monkeypatch.setattr(model_module, "set_omp_threads", lambda n: n)
    return Model()


class TestConstruction:
    def test_creates_an_fdtd_grid_for_the_cpu_solver(self, model):
        assert isinstance(model.G, FDTDGrid)

    def test_defaults(self, model):
        assert model.title == ""
        assert model.dt_mod == 1.0
        assert model.iteration == 0

    def test_collections_start_empty(self, model):
        assert model.subgrids == []
        assert model.geometryviews == []
        assert model.geometryobjects == []

    def test_collections_are_not_shared_between_instances(self, monkeypatch):
        import gprMax.model as model_module

        monkeypatch.setattr(model_module, "set_omp_threads", lambda n: n)
        a, b = Model(), Model()
        a.subgrids.append(object())
        assert b.subgrids == []

    def test_each_model_owns_a_distinct_grid(self, monkeypatch):
        import gprMax.model as model_module

        monkeypatch.setattr(model_module, "set_omp_threads", lambda n: n)
        a, b = Model(), Model()
        assert a.G is not b.G


class TestSizeForwarding:
    """``nx``/``ny``/``nz`` forward to the owned grid."""

    @pytest.mark.parametrize("name", ["nx", "ny", "nz"])
    def test_setter_writes_through_to_grid(self, model, name):
        setattr(model, name, 12)
        assert getattr(model.G, name) == 12

    @pytest.mark.parametrize("name", ["nx", "ny", "nz"])
    def test_getter_reads_from_grid(self, model, name):
        setattr(model.G, name, 7)
        assert getattr(model, name) == 7

    @pytest.mark.parametrize("name", ["nx", "ny", "nz"])
    def test_round_trip(self, model, name):
        setattr(model, name, 15)
        assert getattr(model, name) == 15

    def test_axes_are_independent(self, model):
        model.set_size(np.array([3, 4, 5]))
        assert (model.nx, model.ny, model.nz) == (3, 4, 5)


class TestSetSize:
    def test_unpacks_all_three_axes(self, model):
        model.set_size(np.array([6, 7, 8]))
        assert list(model.G.size) == [6, 7, 8]

    def test_accepts_a_plain_sequence(self, model):
        model.set_size([2, 3, 4])
        assert list(model.G.size) == [2, 3, 4]


class TestCells:
    def test_is_the_product_of_the_three_axes(self, model):
        model.set_size(np.array([2, 3, 4]))
        assert model.cells == 24

    def test_is_uint64(self, model):
        """The count is deliberately widened: a large 3D domain overflows a
        32-bit product.
        """
        model.set_size(np.array([2, 3, 4]))
        assert model.cells.dtype == np.uint64

    def test_zero_sized_model_has_no_cells(self, model):
        assert model.cells == 0

    def test_does_not_overflow_on_a_large_domain(self, model):
        model.set_size(np.array([2000, 2000, 2000]))
        assert model.cells == 8_000_000_000


class TestDiscretisationForwarding:
    def test_dx_getter_reads_the_x_spacing(self, model):
        model.G.dl = np.array(DL_ANISO)
        assert model.dx == DL_ANISO[0]

    @pytest.mark.parametrize("axis,name", [(0, "dx"), (1, "dy"), (2, "dz")])
    def test_setters_write_the_correct_axis(self, model, axis, name):
        """All three *setters* are correct — only the getters are not."""
        setattr(model, name, 0.009)
        assert model.G.dl[axis] == 0.009

    def test_dl_forwards_the_whole_array(self, model):
        model.G.dl = np.array(DL_ANISO)
        assert list(model.dl) == list(DL_ANISO)

    def test_dl_setter_writes_through(self, model):
        model.dl = np.array(DL_ANISO)
        assert list(model.G.dl) == list(DL_ANISO)


class TestTimeForwarding:
    def test_dt_round_trips(self, model):
        model.dt = 1.5e-12
        assert model.dt == 1.5e-12
        assert model.G.dt == 1.5e-12

    def test_iterations_round_trips(self, model):
        model.iterations = 250
        assert model.iterations == 250
        assert model.G.iterations == 250

    def test_timewindow_round_trips(self, model):
        model.timewindow = 3e-9
        assert model.timewindow == 3e-9
        assert model.G.timewindow == 3e-9

    def test_dt_mod_is_model_level_not_grid_level(self, model):
        """``dt_mod`` is the one time-related attribute the model owns
        outright — subgrids inherit it during setup.
        """
        model.dt_mod = 0.5
        assert not hasattr(model.G, "dt_mod")


class TestStepForwarding:
    def test_srcsteps_round_trips(self, model):
        model.srcsteps = np.array([1, 2, 3])
        assert list(model.G.srcsteps) == [1, 2, 3]
        assert list(model.srcsteps) == [1, 2, 3]

    def test_rxsteps_round_trips(self, model):
        model.rxsteps = np.array([4, 5, 6])
        assert list(model.G.rxsteps) == [4, 5, 6]
        assert list(model.rxsteps) == [4, 5, 6]


class TestCreateGrid:
    def test_cpu_solver_gives_a_plain_fdtd_grid(self, model):
        """The cuda / opencl / metal branches need real device handles and
        are covered by PR 12.
        """
        assert type(model._create_grid()) is FDTDGrid

    def test_returns_a_new_grid_each_call(self, model):
        assert model._create_grid() is not model._create_grid()
