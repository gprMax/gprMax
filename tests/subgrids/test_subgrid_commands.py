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

"""Tests for ``subgrids/user_objects.py`` — the ``#subgrid_hsg`` command.

``SubGridBase`` is the user-object side of subgridding: it collects the child
commands nested inside a subgrid block, and its ``setup()`` turns a parsed
command into a wired-up ``SubGridHSG`` grid registered on the model.

The size helpers are pure arithmetic on ``ratio`` and are asserted against
hand-computed numbers rather than against the code's own derivation.

The HSG formulation fixes ``pml_separation`` at ``ratio // 2 + 2``. The public
argument is retained for compatibility but is intentionally ignored; changing
the separation is an unsupported experimental modification to the formulation.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.subgrids.subgrid_hsg import SubGridHSG as SubGridHSGGrid
from gprMax.subgrids.user_objects import SubGridBase, SubGridHSG
from gprMax.user_objects.user_objects import GeometryUserObject, GridUserObject, OutputUserObject

from .conftest import DL


class _Geometry(GeometryUserObject):
    @property
    def order(self):
        return 1

    @property
    def hash(self):
        return "#geometry"

    def build(self, grid, uip):
        pass


class _Grid(GridUserObject):
    @property
    def order(self):
        return 2

    @property
    def hash(self):
        return "#grid"

    def build(self, grid, uip):
        pass


class _Output(OutputUserObject):
    @property
    def order(self):
        return 3

    @property
    def hash(self):
        return "#output"

    def build(self, grid, uip):
        pass


@pytest.fixture
def command():
    """A ``#subgrid_hsg`` user object over a small main-grid region."""

    def _make(**overrides):
        kwargs = {
            "p1": (0.010, 0.010, 0.010),
            "p2": (0.016, 0.016, 0.016),
            "ratio": 3,
            "id": "test_subgrid",
            "is_os_sep": 1,
            "subgrid_pml_thickness": 2,
            "interpolation": 1,
            "filter": True,
        }
        kwargs.update(overrides)
        return SubGridHSG(**kwargs)

    return _make


@pytest.fixture
def model(make_main_grid):
    """A ``Model`` stand-in with the attributes ``setup()`` reads."""
    main = make_main_grid()
    main.iterations = 100
    return SimpleNamespace(G=main, subgrids=[], dt_mod=1.0, iterations=100)


class TestCommandIdentity:
    def test_order(self, command):
        assert command().order == 18

    def test_hash(self, command):
        assert command().hash == "#subgrid_hsg"

    def test_is_not_single_use(self, command):
        """A model may contain more than one subgrid."""
        assert command().is_single_use is False

    def test_is_a_subgrid_base(self, command):
        assert isinstance(command(), SubGridBase)


class TestKwargPassThrough:
    @pytest.mark.parametrize(
        "name,value",
        [
            ("ratio", 5),
            ("id", "another"),
            ("is_os_sep", 4),
            ("subgrid_pml_thickness", 8),
            ("interpolation", 2),
            ("filter", False),
        ],
    )
    def test_arguments_reach_kwargs(self, command, name, value):
        assert command(**{name: value}).kwargs[name] == value

    def test_points_reach_kwargs(self, command):
        cmd = command()
        assert cmd.kwargs["p1"] == (0.010, 0.010, 0.010)
        assert cmd.kwargs["p2"] == (0.016, 0.016, 0.016)

    def test_pml_separation_is_fixed_by_the_formulation(self, command):
        cmd = command(ratio=5, pml_separation=99)
        assert cmd.kwargs["pml_separation"] == 5 // 2 + 2

    def test_defaults_are_applied(self):
        cmd = SubGridHSG(p1=(0, 0, 0), p2=(1, 1, 1))
        assert cmd.kwargs["ratio"] == 3
        assert cmd.kwargs["is_os_sep"] == 3
        assert cmd.kwargs["subgrid_pml_thickness"] == 6
        assert cmd.kwargs["filter"] is True


class TestChildCollection:
    def test_starts_with_no_children(self, command):
        cmd = command()
        assert cmd.children_geometry == []
        assert cmd.children_grid == []
        assert cmd.children_output == []

    def test_geometry_child_is_routed(self, command):
        cmd = command()
        child = _Geometry()
        cmd.add(child)
        assert cmd.children_geometry == [child]

    def test_grid_child_is_routed(self, command):
        cmd = command()
        child = _Grid()
        cmd.add(child)
        assert cmd.children_grid == [child]

    def test_output_child_is_routed(self, command):
        cmd = command()
        child = _Output()
        cmd.add(child)
        assert cmd.children_output == [child]

    def test_geometry_is_checked_before_grid(self, command):
        """``GeometryUserObject`` subclasses ``GridUserObject``, so the
        isinstance chain must test the more specific type first.
        """
        cmd = command()
        cmd.add(_Geometry())
        assert cmd.children_grid == []

    def test_unknown_child_raises(self, command):
        with pytest.raises(ValueError):
            command().add(object())

    def test_children_accumulate_in_order(self, command):
        cmd = command()
        first, second = _Geometry(), _Geometry()
        cmd.add(first)
        cmd.add(second)
        assert cmd.children_geometry == [first, second]


class TestSetDiscretisation:
    @pytest.mark.parametrize("ratio", [3, 5, 7])
    def test_each_axis_is_divided_by_ratio(self, command, make_subgrid, ratio):
        cmd = command(ratio=ratio)
        sg = make_subgrid(ratio=ratio)
        main = SimpleNamespace(dx=DL, dy=DL, dz=DL)
        cmd.set_discretisation(sg, main)
        assert sg.dx == pytest.approx(DL / ratio)
        assert sg.dy == pytest.approx(DL / ratio)
        assert sg.dz == pytest.approx(DL / ratio)

    def test_anisotropic_main_grid_is_preserved(self, command, make_subgrid):
        cmd = command()
        sg = make_subgrid()
        main = SimpleNamespace(dx=0.001, dy=0.002, dz=0.004)
        cmd.set_discretisation(sg, main)
        assert sg.dl == pytest.approx([0.001 / 3, 0.002 / 3, 0.004 / 3])


class TestCellCountHelpers:
    def test_working_region_scales_by_ratio(self, command, make_subgrid):
        cmd = command()
        sg = make_subgrid()
        sg.i0, sg.j0, sg.k0 = 2, 3, 4
        sg.i1, sg.j1, sg.k1 = 6, 9, 14
        cmd.set_working_region_cells(sg)
        assert (sg.nwx, sg.nwy, sg.nwz) == (12, 18, 30)

    def test_total_cells_bracket_the_working_region(self, command, make_subgrid):
        cmd = command()
        sg = make_subgrid()
        sg.nwx, sg.nwy, sg.nwz = 12, 18, 30
        cmd.set_total_cells(sg)
        n = sg.n_boundary_cells
        assert (sg.nx, sg.ny, sg.nz) == (2 * n + 12, 2 * n + 18, 2 * n + 30)

    @pytest.mark.parametrize("ratio", [3, 5, 7])
    def test_iterations_scale_by_ratio(self, command, make_subgrid, ratio):
        """The subgrid takes ``ratio`` steps per main step."""
        cmd = command(ratio=ratio)
        sg = make_subgrid(ratio=ratio)
        cmd.set_iterations(sg, SimpleNamespace(iterations=100))
        assert sg.iterations == 100 * ratio

    def test_name_comes_from_the_id_kwarg(self, command, make_subgrid):
        cmd = command(id="named")
        sg = make_subgrid()
        cmd.set_name(sg)
        assert sg.name == "named"


class TestSetMainGridIndices:
    def test_stores_the_cell_indices(self, command, make_subgrid):
        cmd = command()
        sg = make_subgrid()
        uip = SimpleNamespace(round_to_grid=lambda p: tuple(v * DL for v in p))
        cmd.set_main_grid_indices(sg, uip, (2, 3, 4), (6, 9, 14))
        assert (sg.i0, sg.j0, sg.k0) == (2, 3, 4)
        assert (sg.i1, sg.j1, sg.k1) == (6, 9, 14)

    def test_stores_the_rounded_coordinates(self, command, make_subgrid):
        cmd = command()
        sg = make_subgrid()
        uip = SimpleNamespace(round_to_grid=lambda p: tuple(v * DL for v in p))
        cmd.set_main_grid_indices(sg, uip, (2, 3, 4), (6, 9, 14))
        assert (sg.x1, sg.y1, sg.z1) == pytest.approx((0.002, 0.003, 0.004))
        assert (sg.x2, sg.y2, sg.z2) == pytest.approx((0.006, 0.009, 0.014))


class TestBuild:
    """``build()`` constructs the grid and runs the shared ``setup()``."""

    def test_returns_an_hsg_grid(self, command, model):
        assert isinstance(command().build(model), SubGridHSGGrid)

    def test_registers_the_subgrid_on_the_model(self, command, model):
        sg = command().build(model)
        assert model.subgrids == [sg]

    def test_wires_the_parent_grid(self, command, model):
        sg = command().build(model)
        assert sg.parent_grid is model.G

    def test_stores_the_subgrid_on_the_command(self, command, model):
        cmd = command()
        sg = cmd.build(model)
        assert cmd.subgrid is sg

    def test_discretisation_is_ratio_times_finer(self, command, model):
        sg = command(ratio=3).build(model)
        assert sg.dx == pytest.approx(model.G.dx / 3)

    def test_iterations_scale_by_ratio(self, command, model):
        sg = command(ratio=3).build(model)
        assert sg.iterations == model.iterations * 3

    def test_time_step_respects_the_stability_factor(self, command, model):
        """The subgrid inherits the main grid's ``dt_mod``."""
        model.dt_mod = 0.5
        sg = command().build(model)
        sg_unmodified = SubGridHSGGrid(**command().kwargs)
        sg_unmodified.dl = sg.dl
        sg_unmodified.calculate_dt()
        assert sg.dt == pytest.approx(sg_unmodified.dt * 0.5)

    def test_working_region_matches_the_requested_box(self, command, model):
        sg = command().build(model)
        # p1..p2 spans 6 main cells at DL, refined by ratio 3.
        assert sg.nwx == 18

    def test_copies_builtin_materials(self, command, model, make_material):
        builtin = make_material(ID="free_space", numID=1)
        builtin.type = "builtin"
        model.G.materials = [builtin]
        sg = command().build(model)
        assert [m.ID for m in sg.materials] == ["free_space"]

    def test_materials_are_copied_not_aliased(self, command, model, make_material):
        """The subgrid must be able to diverge from the main grid's material
        objects without mutating them.
        """
        builtin = make_material(ID="free_space", numID=1)
        builtin.type = "builtin"
        model.G.materials = [builtin]
        sg = command().build(model)
        assert sg.materials[0] is not builtin

    def test_non_builtin_materials_are_not_copied(self, command, model, make_material):
        user_material = make_material(ID="soil", numID=2)
        user_material.type = "user"
        model.G.materials = [user_material]
        sg = command().build(model)
        assert sg.materials == []

    def test_two_subgrids_of_the_same_type_are_allowed(self, command, model):
        command(id="one").build(model)
        command(id="two").build(model)
        assert len(model.subgrids) == 2

    def test_mixing_subgrid_types_raises(self, command, model):
        command(id="one").build(model)

        class _Other(SubGridHSGGrid):
            pass

        model.subgrids.append(_Other(**command().kwargs))
        with pytest.raises(ValueError):
            command(id="three").build(model)


pytestmark = pytest.mark.unit
