"""``set_dispersive_updates`` — ``gprMax/updates/cpu_updates.py:239``.

Twenty lines that build the *name* of a compiled kernel out of four switches
and then fetch it by string:

.. code-block:: python

    poles      = "multi" if maxpoles > 1 else "1"
    precision  = "float" if precision == "single" else "double"
    dispersion = "complex" if dispersivedtype == dtypes["complex"] else "real"

    update_f = "update_electric_dispersive_{}pole_{}_{}_{}"
    disp_a = update_f.format(poles, "A", precision, dispersion)
    disp_a_f = getattr(import_module("gprMax.cython.fields_updates_dispersive"), disp_a)

Four switches with two values each, times the A/B half, is **sixteen possible
names** — and every one must exist in a Cython module that is *generated at
build time* from ``fields_updates_dispersive_template.jinja``.

Nothing in the codebase checks that. If the template's naming scheme ever
drifts from this format string, the failure is an ``AttributeError`` at run
time, for the one combination of user settings nobody tried.
``TestEveryNameResolves`` closes that gap: it drives all sixteen against the
**real compiled module**, so a mismatch fails here rather than in a user's
simulation.

The remaining classes pin the switch logic itself, including the two silent
fallbacks — an unrecognised precision string yields ``"double"``, and a
``dispersivedtype`` that does not match yields ``"real"``.
"""

import itertools
from importlib import import_module

import numpy as np
import pytest

from gprMax.updates.cpu_updates import CPUUpdates

DISPERSIVE_MODULE = "gprMax.cython.fields_updates_dispersive"

# The exact format string the dispatcher uses. Duplicated here deliberately:
# if someone edits it in the source, these tests should notice.
NAME_TEMPLATE = "update_electric_dispersive_{}pole_{}_{}_{}"

POLES = ("1", "multi")
HALVES = ("A", "B")
PRECISIONS = ("float", "double")
DISPERSIONS = ("real", "complex")

ALL_NAMES = [
    NAME_TEMPLATE.format(poles, half, precision, dispersion)
    for poles, half, precision, dispersion in itertools.product(
        POLES, HALVES, PRECISIONS, DISPERSIONS
    )
]


@pytest.fixture
def configure(updates_config):
    """Set the four switches and return the configured stand-in config.

    ``maxpoles`` drives the pole switch, ``precision`` the precision switch,
    and the identity of ``dispersivedtype`` against ``dtypes["complex"]``
    drives the dispersion switch.
    """

    def _configure(maxpoles=1, precision="double", dispersion="real"):
        updates_config.model_config.materials["maxpoles"] = maxpoles
        updates_config.sim_config.general["precision"] = precision
        complex_dtype = updates_config.sim_config.dtypes["complex"]
        updates_config.model_config.materials["dispersivedtype"] = (
            complex_dtype if dispersion == "complex" else np.float64
        )
        return updates_config

    return _configure


class TestEveryNameResolves:
    """All sixteen constructible names exist in the compiled extension.

    This is the test that justifies the file. The kernels are generated from
    a jinja template at build time and imported by string at run time, with
    no static link between the two. These sixteen cases are the entire space
    the dispatcher can produce.
    """

    def test_there_are_exactly_sixteen_constructible_names(self):
        """Four binary switches — no more, no fewer."""
        assert len(ALL_NAMES) == 16
        assert len(set(ALL_NAMES)) == 16

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_name_exists_in_the_compiled_module(self, name):
        """Every one is a real attribute of the built extension."""
        module = import_module(DISPERSIVE_MODULE)
        assert hasattr(module, name), f"{DISPERSIVE_MODULE} has no {name!r}"

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_name_resolves_to_something_callable(self, name):
        module = import_module(DISPERSIVE_MODULE)
        assert callable(getattr(module, name))

    @pytest.mark.parametrize(
        "maxpoles,precision,dispersion",
        list(itertools.product((1, 3), ("single", "double"), ("real", "complex"))),
    )
    def test_dispatcher_binds_both_halves_for_every_combination(
        self, configure, make_wiring_grid, maxpoles, precision, dispersion
    ):
        """The eight reachable configurations, driven end to end.

        Each binds an A and a B function, so this covers all sixteen names
        through the real code path rather than by string construction.
        """
        configure(maxpoles=maxpoles, precision=precision, dispersion=dispersion)
        updates = CPUUpdates(make_wiring_grid())

        updates.set_dispersive_updates()

        assert callable(updates.dispersive_update_a)
        assert callable(updates.dispersive_update_b)

    @pytest.mark.parametrize(
        "maxpoles,precision,dispersion",
        list(itertools.product((1, 3), ("single", "double"), ("real", "complex"))),
    )
    def test_bound_functions_are_the_ones_named_by_the_switches(
        self, configure, make_wiring_grid, maxpoles, precision, dispersion
    ):
        """The bound object is the module attribute the name predicts.

        Confirms the dispatcher is not merely finding *a* function, but the
        specific one implied by the configuration.
        """
        configure(maxpoles=maxpoles, precision=precision, dispersion=dispersion)
        module = import_module(DISPERSIVE_MODULE)
        poles = "multi" if maxpoles > 1 else "1"
        prec = "float" if precision == "single" else "double"

        updates = CPUUpdates(make_wiring_grid())
        updates.set_dispersive_updates()

        expected_a = NAME_TEMPLATE.format(poles, "A", prec, dispersion)
        expected_b = NAME_TEMPLATE.format(poles, "B", prec, dispersion)
        assert updates.dispersive_update_a is getattr(module, expected_a)
        assert updates.dispersive_update_b is getattr(module, expected_b)


class TestPoleSwitch:
    """``maxpoles > 1`` selects ``multipole``, otherwise ``1pole``."""

    def test_one_pole_selects_the_single_pole_kernel(self, configure, make_wiring_grid):
        configure(maxpoles=1)
        updates = CPUUpdates(make_wiring_grid())

        updates.set_dispersive_updates()

        assert "_1pole_" in updates.dispersive_update_a.__name__

    @pytest.mark.parametrize("maxpoles", [2, 3, 5, 10])
    def test_more_than_one_pole_selects_the_multipole_kernel(
        self, configure, make_wiring_grid, maxpoles
    ):
        configure(maxpoles=maxpoles)
        updates = CPUUpdates(make_wiring_grid())

        updates.set_dispersive_updates()

        assert "_multipole_" in updates.dispersive_update_a.__name__

    def test_zero_poles_still_binds_the_single_pole_kernel(self, configure, make_wiring_grid):
        """``maxpoles == 0`` is not rejected — it selects ``1pole``.

        ``create_solver`` guards the call with ``maxpoles != 0``, so this
        never happens in production. Called directly it succeeds silently,
        binding a kernel that ``update_electric_a`` will never reach because
        that method sends ``maxpoles == 0`` to the plain update instead.
        """
        configure(maxpoles=0)
        updates = CPUUpdates(make_wiring_grid())

        updates.set_dispersive_updates()

        assert "_1pole_" in updates.dispersive_update_a.__name__

    def test_the_boundary_is_between_one_and_two(self, configure, make_wiring_grid):
        """``> 1``, so one pole is single and two are multi."""
        names = {}
        for maxpoles in (1, 2):
            configure(maxpoles=maxpoles)
            updates = CPUUpdates(make_wiring_grid())
            updates.set_dispersive_updates()
            names[maxpoles] = updates.dispersive_update_a.__name__

        assert "_1pole_" in names[1]
        assert "_multipole_" in names[2]


class TestPrecisionSwitch:
    """``precision == "single"`` selects ``float``; everything else
    selects ``double``."""

    def test_single_precision_selects_the_float_kernel(self, configure, make_wiring_grid):
        configure(precision="single")
        updates = CPUUpdates(make_wiring_grid())

        updates.set_dispersive_updates()

        assert updates.dispersive_update_a.__name__.endswith("_float_real")

    def test_double_precision_selects_the_double_kernel(self, configure, make_wiring_grid):
        configure(precision="double")
        updates = CPUUpdates(make_wiring_grid())

        updates.set_dispersive_updates()

        assert updates.dispersive_update_a.__name__.endswith("_double_real")

    @pytest.mark.parametrize("precision", ["Single", "SINGLE", "float", "", "half"])
    def test_an_unrecognised_precision_silently_selects_double(
        self, configure, make_wiring_grid, precision
    ):
        """The ternary tests equality with ``"single"`` and falls through.

        Any other string — including a capitalisation slip or the plausible
        ``"float"`` — silently produces the double-precision kernel. Since
        the grid's arrays were allocated from the same ``precision`` value
        elsewhere, this surfaces as a Cython buffer dtype mismatch rather
        than a message about the setting being wrong.

        Recorded in ``notes/bugs/config-precision-no-terminal-else.md``,
        which covers the same string's other silent failure.
        """
        configure(precision=precision)
        updates = CPUUpdates(make_wiring_grid())

        updates.set_dispersive_updates()

        assert "_double_" in updates.dispersive_update_a.__name__


class TestDispersionSwitch:
    """``dispersivedtype == dtypes["complex"]`` selects ``complex``."""

    def test_matching_complex_dtype_selects_the_complex_kernel(self, configure, make_wiring_grid):
        configure(dispersion="complex")
        updates = CPUUpdates(make_wiring_grid())

        updates.set_dispersive_updates()

        assert updates.dispersive_update_a.__name__.endswith("_complex")

    def test_a_real_dtype_selects_the_real_kernel(self, configure, make_wiring_grid):
        configure(dispersion="real")
        updates = CPUUpdates(make_wiring_grid())

        updates.set_dispersive_updates()

        assert updates.dispersive_update_a.__name__.endswith("_real")

    def test_the_comparison_is_against_the_configured_complex_dtype(
        self, updates_config, make_wiring_grid
    ):
        """Not against ``np.complexfloating`` in general.

        A single-precision run compares against ``np.complex64``; handing it
        ``np.complex128`` — a complex type, but the wrong one — takes the
        *real* branch.
        """
        updates_config.sim_config.dtypes["complex"] = np.complex64
        updates_config.model_config.materials["dispersivedtype"] = np.complex128
        updates_config.model_config.materials["maxpoles"] = 1

        updates = CPUUpdates(make_wiring_grid())
        updates.set_dispersive_updates()

        assert updates.dispersive_update_a.__name__.endswith("_real")

    def test_unset_dispersive_dtype_silently_selects_real(self, updates_config, make_wiring_grid):
        """``dispersivedtype`` defaults to ``None`` until it is derived.

        ``ModelConfig`` initialises the key to ``None``, and only
        ``set_dispersive_material_types()`` fills it in. ``None == np.complex64``
        is ``False``, so calling ``set_dispersive_updates`` first binds the
        **real** kernel for what may be a complex-pole model — a wrong answer
        with no warning, produced purely by call order.

        Written up in ``notes/bugs/dispersive-dtype-default-none.md``.
        """
        updates_config.model_config.materials["maxpoles"] = 2
        updates_config.model_config.materials["dispersivedtype"] = None

        updates = CPUUpdates(make_wiring_grid())
        updates.set_dispersive_updates()

        assert updates.dispersive_update_a.__name__.endswith("_real")


class TestBinding:
    """How the resolved functions are attached to the instance."""

    def test_both_halves_are_set_together(self, configure, make_wiring_grid):
        """Neither exists before the call; both exist after."""
        configure()
        updates = CPUUpdates(make_wiring_grid())

        assert not hasattr(updates, "dispersive_update_a")
        assert not hasattr(updates, "dispersive_update_b")

        updates.set_dispersive_updates()

        assert hasattr(updates, "dispersive_update_a")
        assert hasattr(updates, "dispersive_update_b")

    def test_the_two_halves_are_different_functions(self, configure, make_wiring_grid):
        configure()
        updates = CPUUpdates(make_wiring_grid())

        updates.set_dispersive_updates()

        assert updates.dispersive_update_a is not updates.dispersive_update_b

    def test_a_and_b_differ_only_in_the_half_marker(self, configure, make_wiring_grid):
        """The other three switches must agree between the halves."""
        configure(maxpoles=4, precision="single", dispersion="complex")
        updates = CPUUpdates(make_wiring_grid())

        updates.set_dispersive_updates()

        name_a = updates.dispersive_update_a.__name__
        name_b = updates.dispersive_update_b.__name__
        assert name_a.replace("_A_", "_B_") == name_b

    def test_binding_is_per_instance_not_per_class(self, configure, make_wiring_grid):
        """Two updaters can hold different kernels simultaneously.

        Subgrid runs do exactly this: ``create_solver`` configures the parent
        and every ``SubgridUpdater`` separately.
        """
        configure(maxpoles=1)
        first = CPUUpdates(make_wiring_grid())
        first.set_dispersive_updates()

        configure(maxpoles=5)
        second = CPUUpdates(make_wiring_grid())
        second.set_dispersive_updates()

        assert "_1pole_" in first.dispersive_update_a.__name__
        assert "_multipole_" in second.dispersive_update_a.__name__

    def test_calling_twice_rebinds_to_the_new_configuration(self, configure, make_wiring_grid):
        """The method is idempotent in effect but not cached."""
        configure(maxpoles=1)
        updates = CPUUpdates(make_wiring_grid())
        updates.set_dispersive_updates()
        first = updates.dispersive_update_a

        configure(maxpoles=9)
        updates.set_dispersive_updates()

        assert updates.dispersive_update_a is not first
        assert "_multipole_" in updates.dispersive_update_a.__name__

    def test_bound_functions_are_plain_functions_not_methods(self, configure, make_wiring_grid):
        """Assigned to the *instance*, so no ``self`` is passed.

        This is why ``update_electric_a`` calls ``self.dispersive_update_a(nx,
        ...)`` with the grid dimensions first and no leading ``self`` — an
        instance attribute holding a function is not a bound method.
        """
        configure()
        updates = CPUUpdates(make_wiring_grid())

        updates.set_dispersive_updates()

        assert "dispersive_update_a" in vars(updates)
        assert not hasattr(updates.dispersive_update_a, "__self__")


class TestModulePath:
    """The module the kernels are fetched from."""

    def test_the_module_path_is_hard_coded(self, configure, make_wiring_grid, monkeypatch):
        """Unlike the PML dispatcher, the module name is not formatted.

        Only the *function* name varies; the module is always
        ``gprMax.cython.fields_updates_dispersive``.
        """
        seen = []
        real_import = import_module

        def spy(name, *args, **kwargs):
            seen.append(name)
            return real_import(name, *args, **kwargs)

        import gprMax.updates.cpu_updates as module

        monkeypatch.setattr(module, "import_module", spy)
        configure(maxpoles=3, precision="single", dispersion="complex")

        CPUUpdates(make_wiring_grid()).set_dispersive_updates()

        assert set(seen) == {DISPERSIVE_MODULE}

    def test_the_module_is_imported_once_per_half(self, configure, make_wiring_grid, monkeypatch):
        """Two calls, one per half — the result is not reused.

        Harmless because ``import_module`` hits ``sys.modules``, but worth
        pinning so a future refactor does not assume a single import.
        """
        seen = []
        real_import = import_module

        def spy(name, *args, **kwargs):
            seen.append(name)
            return real_import(name, *args, **kwargs)

        import gprMax.updates.cpu_updates as module

        monkeypatch.setattr(module, "import_module", spy)
        configure()

        CPUUpdates(make_wiring_grid()).set_dispersive_updates()

        assert len(seen) == 2

    def test_a_missing_kernel_raises_attribute_error(
        self, configure, make_wiring_grid, monkeypatch
    ):
        """There is no ``try``/``except`` around the ``getattr``.

        If the generated module ever lost a variant, the user would see a
        bare ``AttributeError`` naming a function they have never heard of,
        rather than a message about their material settings. Simulated here
        by importing a module that has none of the sixteen names.
        """
        import types

        import gprMax.updates.cpu_updates as module

        monkeypatch.setattr(module, "import_module", lambda name: types.ModuleType("empty"))
        configure()

        with pytest.raises(AttributeError, match="update_electric_dispersive"):
            CPUUpdates(make_wiring_grid()).set_dispersive_updates()


pytestmark = pytest.mark.unit
