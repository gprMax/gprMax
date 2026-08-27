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

"""The ``Updates`` abstract base class — ``gprMax/updates/updates.py``.

``Updates`` is the contract every solver backend implements. It carries no
logic at all: eleven ``@abstractmethod`` declarations, two concrete no-ops,
and an ``__init__`` that stores the grid. There is nothing to compute, so
every test here is about the *shape* of the contract.

That is worth testing precisely because the class exists to make a mistake
impossible. If a new abstract method is added upstream and one backend does
not implement it, that backend stops being instantiable — which is the ABC
working. But if a backend is written that never inherits from ``Updates`` at
all, the ABC is silently bypassed, and ``Solver.__init__``'s ``updates:
Updates`` annotation becomes a lie. That has already happened once; see
``TestBackendConformance``.

The eleven names are asserted as a frozen set rather than individually. A test
per method would pass unchanged if a twelfth were added, which is precisely
the event worth catching.
"""

import inspect
from abc import ABC

import pytest

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.updates.cpu_updates import CPUUpdates
from gprMax.updates.updates import Updates

# The full contract, as of this suite. Adding to it is a breaking change for
# every backend; removing from it silently drops a step from the timestep.
ABSTRACT_METHODS = frozenset(
    {
        "store_outputs",
        "store_snapshots",
        "update_magnetic",
        "update_magnetic_pml",
        "update_magnetic_sources",
        "update_electric_a",
        "update_electric_pml",
        "update_electric_sources",
        "update_electric_b",
        "time_start",
        "calculate_solve_time",
    }
)

# Methods with a real (empty) body on the base class, inherited by any
# backend that does not care to override them.
CONCRETE_METHODS = frozenset({"finalise", "cleanup"})


class MinimalUpdates(Updates):
    """The smallest possible conforming backend.

    Implements all eleven abstract methods and nothing else. Used to prove
    that the eleven really are sufficient — that ``Updates`` has no hidden
    requirement beyond its declarations.
    """

    def store_outputs(self, iteration):
        return "store_outputs"

    def store_snapshots(self, iteration):
        return "store_snapshots"

    def update_magnetic(self):
        return "update_magnetic"

    def update_magnetic_pml(self):
        return "update_magnetic_pml"

    def update_magnetic_sources(self, iteration):
        return "update_magnetic_sources"

    def update_electric_a(self):
        return "update_electric_a"

    def update_electric_pml(self):
        return "update_electric_pml"

    def update_electric_sources(self, iteration):
        return "update_electric_sources"

    def update_electric_b(self):
        return "update_electric_b"

    def time_start(self):
        return "time_start"

    def calculate_solve_time(self):
        return 0.0


class TestAbstractContract:
    """The set of abstract methods, and what it means to satisfy it."""

    def test_updates_is_an_abstract_base_class(self):
        """``Updates`` derives from ``ABC``, so the machinery is active.

        Inheriting ``abstractmethod`` declarations without an ``ABCMeta``
        metaclass silently does nothing — the decorators become documentation
        and every incomplete subclass instantiates happily.
        """
        assert issubclass(Updates, ABC)
        assert isinstance(Updates, type(ABC))

    def test_abstract_method_set_is_exactly_the_eleven(self):
        """The contract is these eleven names and no others.

        A twelfth appearing here means every backend — including the three
        GPU ones this suite cannot run — needs a new implementation.
        """
        assert frozenset(Updates.__abstractmethods__) == ABSTRACT_METHODS

    def test_there_are_eleven_abstract_methods(self):
        """Count stated separately, so a rename plus an addition is caught."""
        assert len(Updates.__abstractmethods__) == 11

    def test_updates_cannot_be_instantiated(self):
        """The base class is not usable directly."""
        with pytest.raises(TypeError, match="abstract"):
            Updates(FDTDGrid())

    def test_a_complete_subclass_can_be_instantiated(self):
        """Implementing the eleven is sufficient — there is no hidden step."""
        updates = MinimalUpdates(FDTDGrid())
        assert isinstance(updates, Updates)

    @pytest.mark.parametrize("missing", sorted(ABSTRACT_METHODS))
    def test_a_subclass_missing_any_one_method_cannot_be_instantiated(self, missing):
        """Each of the eleven is individually load-bearing.

        Builds a genuine subclass of ``Updates`` implementing ten of the
        eleven — the ABC machinery computes ``__abstractmethods__`` itself —
        and asserts instantiation fails naming the one left out.
        """
        body = {name: (lambda self, *a, **k: None) for name in ABSTRACT_METHODS if name != missing}
        incomplete = type("Incomplete", (Updates,), body)

        assert incomplete.__abstractmethods__ == frozenset({missing})
        with pytest.raises(TypeError, match=missing):
            incomplete(FDTDGrid())

    @pytest.mark.parametrize("name", sorted(ABSTRACT_METHODS))
    def test_every_abstract_method_is_marked_abstract(self, name):
        """``__isabstractmethod__`` is set on each declaration."""
        method = getattr(Updates, name)
        assert getattr(method, "__isabstractmethod__", False) is True


class TestConcreteMethods:
    """``finalise`` and ``cleanup`` — the two hooks with a default."""

    @pytest.mark.parametrize("name", sorted(CONCRETE_METHODS))
    def test_hook_is_not_abstract(self, name):
        """Backends inherit these rather than being forced to write them."""
        assert name not in Updates.__abstractmethods__

    @pytest.mark.parametrize("name", sorted(CONCRETE_METHODS))
    def test_hook_returns_none(self, name):
        """Both are no-ops on the base class."""
        updates = MinimalUpdates(FDTDGrid())
        assert getattr(updates, name)() is None

    @pytest.mark.parametrize("name", sorted(CONCRETE_METHODS))
    def test_hook_takes_no_arguments_beyond_self(self, name):
        """``Solver.solve`` calls both with no arguments."""
        sig = inspect.signature(getattr(Updates, name))
        assert list(sig.parameters) == ["self"]

    def test_cpu_updates_reimplements_both_hooks_identically(self):
        """``CPUUpdates`` re-declares both as byte-identical no-ops.

        Pinned as an observation rather than a defect: the overrides are
        harmless, but they are dead code, and anyone adding behaviour to the
        base-class hooks would find it silently ignored on the CPU path.
        """
        for name in CONCRETE_METHODS:
            assert name in CPUUpdates.__dict__
            assert getattr(CPUUpdates, name) is not getattr(Updates, name)

        updates = CPUUpdates(FDTDGrid())
        assert updates.finalise() is None
        assert updates.cleanup() is None


class TestConstruction:
    """``__init__`` stores the grid and does nothing else."""

    def test_init_stores_the_grid(self):
        """``self.grid`` is the object passed in, not a copy."""
        grid = FDTDGrid()
        assert MinimalUpdates(grid).grid is grid

    def test_init_accepts_any_object(self):
        """No validation whatsoever.

        The ``GridType`` bound is a typing construct with no runtime effect,
        so a stand-in grid is accepted — which is what makes most of this
        suite possible.
        """
        sentinel = object()
        assert MinimalUpdates(sentinel).grid is sentinel

    def test_init_sets_only_the_grid_attribute(self):
        """Nothing else is initialised.

        This is why ``CPUUpdates.calculate_solve_time`` raises before
        ``time_start`` has run, and why ``update_electric_a`` raises on a
        dispersive model before ``set_dispersive_updates`` has run. Both are
        recorded in ``notes/bugs/``.
        """
        updates = MinimalUpdates(FDTDGrid())
        assert set(vars(updates)) == {"grid"}

    def test_init_signature_is_a_single_positional_grid(self):
        """Backends are constructed as ``Backend(grid)`` throughout."""
        sig = inspect.signature(Updates.__init__)
        assert list(sig.parameters) == ["self", "G"]
        assert sig.parameters["G"].default is inspect.Parameter.empty


class TestGenericParameter:
    """``Updates`` is generic over the grid type."""

    def test_updates_is_generic(self):
        """``Generic[GridType]`` is in the MRO's parameter list."""
        from typing import Generic

        assert issubclass(Updates, Generic)

    def test_grid_type_var_is_bound_to_fdtd_grid(self):
        """Backends are parameterised by a grid, not by anything else."""
        from gprMax.updates.updates import GridType

        assert GridType.__bound__ is FDTDGrid

    def test_subscripting_updates_is_accepted(self):
        """``Updates[FDTDGrid]`` is a valid base, as CPUUpdates uses it."""
        assert Updates[FDTDGrid] is not None


class TestBackendConformance:
    """Which backends actually implement the contract.

    The CPU and MPI backends do. The three accelerator backends are checked
    here for *conformance only* — none of them is executed, so no hardware is
    required.
    """

    def test_cpu_updates_is_an_updates_subclass(self):
        assert issubclass(CPUUpdates, Updates)

    def test_cpu_updates_implements_the_whole_contract(self):
        """No abstract methods remain, so it is instantiable."""
        assert not CPUUpdates.__abstractmethods__

    def test_cpu_updates_can_be_constructed(self):
        assert isinstance(CPUUpdates(FDTDGrid()), Updates)

    @pytest.mark.parametrize("name", sorted(ABSTRACT_METHODS))
    def test_cpu_updates_defines_every_contract_method(self, name):
        """Each of the eleven is a real implementation, not inherited."""
        assert name in CPUUpdates.__dict__

    def test_plane_wave_hooks_are_part_of_the_backend_contract(self):
        hooks = {"update_plane_waves_electric", "update_plane_waves_magnetic"}
        assert hooks <= set(CPUUpdates.__dict__)
        assert hooks <= set(vars(Updates))

    def test_set_dispersive_updates_is_cpu_specific(self):
        assert "set_dispersive_updates" in CPUUpdates.__dict__
        assert "set_dispersive_updates" not in vars(Updates)

    def test_metal_updates_does_not_implement_the_contract(self):
        """``MetalUpdates`` is not an ``Updates`` subclass at all.

        It is a plain class that assigns ``self.grid = G`` by hand. Because
        ``Solver.__init__`` is annotated ``updates: Updates`` and
        ``create_solver`` hands it a ``MetalUpdates``, the Metal path violates
        the solver's own type contract — and the ABC cannot catch it, because
        nothing ever inherited from the ABC.

        Written up in ``notes/bugs/metal-updates-not-an-updates-subclass.md``.
        The import is guarded because the module imports ``Metal`` lazily but
        may still fail to import on a non-Apple platform.
        """
        metal_updates = pytest.importorskip(
            "gprMax.updates.metal_updates",
            reason="metal_updates imports platform-specific machinery",
        )
        assert not issubclass(metal_updates.MetalUpdates, Updates)


pytestmark = pytest.mark.unit
