"""Tests for ``gprMax.user_objects.user_objects`` and ``rotatable``.

The base file defines five abstract classes that the rest of the package
builds on:

* ``UserObject``        — the common contract: ``kwargs`` storage,
                          ``__lt__``, ``__str__``, ``params_str``,
                          ``_create_uip`` dispatch.
* ``ModelUserObject``   — ``build(model)`` is abstract.
* ``GridUserObject``    — ``build(grid)`` is abstract + ``grid_name``.
* ``OutputUserObject``  — ``build(model, grid)`` is abstract + ``grid_name``.
* ``GeometryUserObject``— concrete ``order = 1``.

``RotatableMixin`` adds rotation state (``axis``, ``angle``, ``origin``,
``do_rotate``) and a public ``rotate(...)`` setter that flips
``do_rotate`` to True.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gprMax.user_objects.rotatable import RotatableMixin
from gprMax.user_objects.user_objects import (
    GeometryUserObject,
    GridUserObject,
    ModelUserObject,
    OutputUserObject,
    UserObject,
)


# ---------------------------------------------------------------------------
# UserObject — the common contract
# ---------------------------------------------------------------------------


class _ConcreteUserObject(UserObject):
    """Minimum-viable concrete subclass for testing the base contract."""

    @property
    def order(self):
        return 5

    @property
    def hash(self):
        return "#fake"

    def build(self, model):
        return model


class TestUserObjectABCEnforcement:
    def test_cannot_instantiate_userobject_directly(self):
        # ABC enforces abstract ``order`` and ``hash``
        with pytest.raises(TypeError):
            UserObject()

    def test_subclass_missing_abstracts_cannot_instantiate(self):
        class Bare(UserObject):
            pass

        with pytest.raises(TypeError):
            Bare()


class TestUserObjectKwargsAndDefaults:
    def test_init_stores_kwargs_dict_verbatim(self):
        obj = _ConcreteUserObject(a=1, b="two", c=(0.1, 0.2))
        assert obj.kwargs == {"a": 1, "b": "two", "c": (0.1, 0.2)}

    def test_autotranslate_defaults_to_true(self):
        obj = _ConcreteUserObject()
        assert obj.autotranslate is True

    def test_no_kwargs_yields_empty_kwargs_dict(self):
        obj = _ConcreteUserObject()
        assert obj.kwargs == {}


class TestUserObjectOrderingAndPrinting:
    def test_lt_sorts_by_order(self):
        class Low(_ConcreteUserObject):
            @property
            def order(self):
                return 1

        class High(_ConcreteUserObject):
            @property
            def order(self):
                return 9

        assert Low() < High()
        assert not High() < Low()
        # sorted() uses __lt__ → result is in ascending ``order`` order
        ordered = sorted([High(), Low()])
        assert [o.order for o in ordered] == [1, 9]

    def test_str_joins_scalar_kwargs_after_hash(self):
        # Scalars are space-joined after ``{hash}: ``
        obj = _ConcreteUserObject(a=1, b="two", c=3.5)
        assert str(obj) == "#fake: 1 two 3.5"

    def test_str_expands_tuple_and_list_kwargs(self):
        obj = _ConcreteUserObject(p1=(0.1, 0.2, 0.3), tag="x")
        assert str(obj) == "#fake: 0.1 0.2 0.3 x"

    def test_str_skips_none_valued_kwargs(self):
        obj = _ConcreteUserObject(a=1, b=None, c=2)
        # ``b`` is silently dropped
        assert str(obj) == "#fake: 1 2"

    def test_params_str_returns_hash_and_kwargs_repr(self):
        obj = _ConcreteUserObject(a=1, b="x")
        s = obj.params_str()
        assert s.startswith("#fake: ")
        assert "'a': 1" in s
        assert "'b': 'x'" in s


# ---------------------------------------------------------------------------
# _create_uip dispatch
# ---------------------------------------------------------------------------


class TestCreateUIPDispatch:
    """``_create_uip`` returns the right UserInput class for the grid type.

    We patch the UserInput constructors so the test does not depend on
    their internals — only the dispatch is under test.
    """

    def test_main_grid_returns_main_grid_user_input(self):
        from gprMax.user_objects import user_objects as uo_mod

        grid = MagicMock()
        obj = _ConcreteUserObject()

        with patch.object(uo_mod, "MainGridUserInput") as MainUIP:
            obj._create_uip(grid)
            MainUIP.assert_called_once_with(grid)

    def test_mpi_grid_returns_mpi_user_input(self):
        from gprMax.grid.mpi_grid import MPIGrid
        from gprMax.user_objects import user_objects as uo_mod

        grid = MagicMock(spec=MPIGrid)
        obj = _ConcreteUserObject()

        with patch.object(uo_mod, "MPIUserInput") as MPIUIP:
            obj._create_uip(grid)
            MPIUIP.assert_called_once_with(grid)

    def test_subgrid_with_autotranslate_returns_subgrid_user_input(self, user_object_config):
        from gprMax.subgrids.grid import SubGridBaseGrid
        from gprMax.user_objects import user_objects as uo_mod

        # Both global and per-object autotranslate flags must be true
        user_object_config.sim_config.args.autotranslate = True
        grid = MagicMock(spec=SubGridBaseGrid)
        obj = _ConcreteUserObject()
        obj.autotranslate = True

        with patch.object(uo_mod, "SubgridUserInput") as SubUIP:
            obj._create_uip(grid)
            SubUIP.assert_called_once_with(grid)

    def test_subgrid_without_global_autotranslate_falls_through_to_main(
        self, user_object_config
    ):
        from gprMax.subgrids.grid import SubGridBaseGrid
        from gprMax.user_objects import user_objects as uo_mod

        user_object_config.sim_config.args.autotranslate = False
        grid = MagicMock(spec=SubGridBaseGrid)
        obj = _ConcreteUserObject()

        with patch.object(uo_mod, "MainGridUserInput") as MainUIP:
            obj._create_uip(grid)
            MainUIP.assert_called_once_with(grid)

    def test_subgrid_with_object_autotranslate_disabled_falls_through(
        self, user_object_config
    ):
        # Per-object autotranslate is the override even when global is True
        from gprMax.subgrids.grid import SubGridBaseGrid
        from gprMax.user_objects import user_objects as uo_mod

        user_object_config.sim_config.args.autotranslate = True
        grid = MagicMock(spec=SubGridBaseGrid)
        obj = _ConcreteUserObject()
        obj.autotranslate = False

        with patch.object(uo_mod, "MainGridUserInput") as MainUIP:
            obj._create_uip(grid)
            MainUIP.assert_called_once_with(grid)


# ---------------------------------------------------------------------------
# ModelUserObject / GridUserObject / OutputUserObject — abstract contracts
# ---------------------------------------------------------------------------


class TestSubclassAbstractEnforcement:
    def test_modeluserobject_without_build_cannot_instantiate(self):
        class M(ModelUserObject):
            @property
            def order(self):
                return 1

            @property
            def hash(self):
                return "#m"

        with pytest.raises(TypeError):
            M()

    def test_griduserobject_without_build_cannot_instantiate(self):
        class G(GridUserObject):
            @property
            def order(self):
                return 1

            @property
            def hash(self):
                return "#g"

        with pytest.raises(TypeError):
            G()

    def test_outputuserobject_without_build_cannot_instantiate(self):
        class O(OutputUserObject):
            @property
            def order(self):
                return 1

            @property
            def hash(self):
                return "#o"

        with pytest.raises(TypeError):
            O()


class TestGridNameHelper:
    """``grid_name`` is provided by ``GridUserObject`` and ``OutputUserObject``."""

    class _G(GridUserObject):
        @property
        def order(self):
            return 1

        @property
        def hash(self):
            return "#g"

        def build(self, grid):
            return grid

    def test_grid_name_empty_for_main_grid(self):
        obj = self._G()
        # Plain stub grid is not a SubGridBaseGrid → empty string
        assert obj.grid_name(SimpleNamespace()) == ""

    def test_grid_name_brackets_subgrid_name(self):
        from gprMax.subgrids.grid import SubGridBaseGrid

        obj = self._G()
        grid = MagicMock(spec=SubGridBaseGrid)
        grid.name = "sub1"
        assert obj.grid_name(grid) == "[sub1] "


# ---------------------------------------------------------------------------
# GeometryUserObject — concrete order = 1
# ---------------------------------------------------------------------------


class TestGeometryUserObject:
    class _Geo(GeometryUserObject):
        @property
        def hash(self):
            return "#geo"

        def build(self, grid):
            return grid

    def test_geometry_order_is_one(self):
        # Geometry objects don't sort — they build in arrival order. The
        # base class enforces this by giving them all ``order = 1``.
        assert self._Geo().order == 1


# ---------------------------------------------------------------------------
# RotatableMixin
# ---------------------------------------------------------------------------


class _RotObj(RotatableMixin, _ConcreteUserObject):
    """Concrete rotatable object for mixin tests."""

    def _do_rotate(self, grid):
        # Stamp something on grid so tests can verify the hook ran
        grid.rotated = True


class TestRotatableMixinDefaults:
    def test_defaults(self):
        obj = _RotObj()
        assert obj.axis == "x"
        assert obj.angle == 0
        assert obj.origin is None
        assert obj.do_rotate is False


class TestRotatableMixinRotateSetter:
    def test_rotate_without_origin(self):
        obj = _RotObj()
        obj.rotate("y", 90)
        assert obj.axis == "y"
        assert obj.angle == 90
        assert obj.origin is None
        assert obj.do_rotate is True

    def test_rotate_with_origin(self):
        obj = _RotObj()
        obj.rotate("z", 180, origin=(0.1, 0.2, 0.3))
        assert obj.axis == "z"
        assert obj.angle == 180
        assert obj.origin == (0.1, 0.2, 0.3)
        assert obj.do_rotate is True


class TestRotatableMixinAbstractEnforcement:
    def test_without_do_rotate_implementation_cannot_instantiate(self):
        class BareRot(RotatableMixin, _ConcreteUserObject):
            pass  # _do_rotate not implemented

        with pytest.raises(TypeError):
            BareRot()
