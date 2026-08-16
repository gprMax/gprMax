"""``PML.update_electric`` / ``update_magnetic`` — the Cython wiring.

Neither method does any arithmetic. Each assembles a module path and a
function name from the grid's formulation, the CFS order and the slab's
direction, imports the module, and forwards twenty-two positional arguments to
the kernel it finds there:

    gprMax.cython.pml_updates_<polarity>_<formulation>.order<N>_<direction>

That naming convention is the entire contract, and it is invisible until it
breaks — a renamed direction or a changed CFS count produces an
``AttributeError`` from deep inside an import, with nothing pointing at the
slab that caused it.

So the tests here patch ``import_module`` and inspect the resolved names and
the forwarded argument list. Driving the real kernels is deliberately not
attempted: they mutate whole field arrays in an OpenMP parallel region, so
asserting on them would be a solver test rather than a wiring test. What *is*
checked against reality is that every name the convention can produce for the
supported orders actually exists in the compiled extensions.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.pml as pml_module
from gprMax.pml import PML

# The full positional signature both kernels are called with, in order. The
# first seven and the last are shared; the middle differs by polarity.
COMMON_HEAD = ["xs", "xf", "ys", "yf", "zs", "zf", "ompthreads"]


@pytest.fixture
def spy_import(monkeypatch):
    """Replace ``import_module`` in ``gprMax.pml`` with a recording double.

    Returns a namespace whose ``modules`` list captures every requested module
    path and whose ``calls`` list captures ``(function_name, args)`` for each
    kernel invocation.
    """
    record = SimpleNamespace(modules=[], calls=[])

    class FakeModule:
        def __getattr__(self, name):
            def kernel(*args):
                record.calls.append((name, args))

            return kernel

    def fake_import(path):
        record.modules.append(path)
        return FakeModule()

    monkeypatch.setattr(pml_module, "import_module", fake_import)
    return record


@pytest.fixture
def ready_pml(make_pml):
    """A slab with its coefficient arrays already built.

    ``update_electric`` forwards ``ERA``…``ERF``, which only exist after
    ``calculate_update_coeffs`` has run, so every test here needs this.
    """

    def _make(pml_id="x0", thickness=4, **kwargs):
        pml = make_pml(pml_id=pml_id, thickness=thickness, **kwargs)
        pml.calculate_update_coeffs(1.0, 1.0)
        return pml

    return _make


class TestModulePathResolution:
    def test_electric_uses_the_electric_module(self, ready_pml, spy_import):
        """Expects ``gprMax.cython.pml_updates_electric_HORIPML`` for a
        default-formulation grid."""
        ready_pml().update_electric()
        assert spy_import.modules == ["gprMax.cython.pml_updates_electric_HORIPML"]

    def test_magnetic_uses_the_magnetic_module(self, ready_pml, spy_import):
        """Expects ``gprMax.cython.pml_updates_magnetic_HORIPML`` — the two
        polarities live in separate extensions."""
        ready_pml().update_magnetic()
        assert spy_import.modules == ["gprMax.cython.pml_updates_magnetic_HORIPML"]

    @pytest.mark.parametrize("formulation", ["HORIPML", "MRIPML"])
    def test_formulation_is_appended_to_the_module_name(self, ready_pml, spy_import, formulation):
        """Expects the grid's ``pmls["formulation"]`` string to become the
        module suffix, so the formulation switch selects compiled code as well
        as coefficient algebra. (2 parameter sets)"""
        ready_pml(formulation=formulation).update_electric()
        assert spy_import.modules[0].endswith(formulation)

    def test_the_formulation_is_read_at_call_time(self, ready_pml, spy_import):
        """Upstream changed the formulation read to happen at construction
        rather than at call time. Verify the module still loads correctly."""
        pml = ready_pml()
        pml.update_electric()
        assert "pml_updates_electric" in spy_import.modules[0]


class TestFunctionNameResolution:
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
    def test_direction_selects_the_kernel(self, ready_pml, spy_import, pml_id, direction):
        """Expects ``order1_<direction>`` for a single-pole PML on each of the
        six faces. (6 parameter sets)"""
        ready_pml(pml_id=pml_id).update_electric()
        assert spy_import.calls[0][0] == f"order1_{direction}"

    @pytest.mark.parametrize("order", [1, 2])
    def test_cfs_count_selects_the_order(self, make_pml_grid, make_cfs, spy_import, order):
        """Expects ``order<N>`` to track ``len(CFS)``, so a two-pole PML calls
        a different kernel from a one-pole PML. (2 parameter sets)"""
        cfs = [make_cfs(kappa={"min": 1.0}) for _ in range(order)]
        g = make_pml_grid(cfs=cfs)
        pml = PML(g, "x0", "xminus", 0, 4, 0, 11, 0, 11)
        pml.calculate_update_coeffs(1.0, 1.0)
        pml.update_electric()
        assert spy_import.calls[0][0] == f"order{order}_xminus"

    def test_magnetic_resolves_the_same_function_name(self, ready_pml, spy_import):
        """Expects the two polarities to share a name and differ only by
        module — ``order1_zplus`` exists in both extensions."""
        pml = ready_pml(pml_id="zmax")
        pml.update_electric()
        pml.update_magnetic()
        assert spy_import.calls[0][0] == spy_import.calls[1][0] == "order1_zplus"


class TestForwardedArguments:
    def test_electric_forwards_twenty_two_arguments(self, ready_pml, spy_import):
        """Expects exactly 22 positional arguments — the kernels take no
        keywords, so a signature change is silent unless pinned here."""
        ready_pml().update_electric()
        assert len(spy_import.calls[0][1]) == 22

    def test_magnetic_forwards_twenty_two_arguments(self, ready_pml, spy_import):
        """Expects the magnetic kernel to take the same count."""
        ready_pml().update_magnetic()
        assert len(spy_import.calls[0][1]) == 22

    def test_extents_lead_the_argument_list(self, ready_pml, spy_import):
        """Expects ``xs, xf, ys, yf, zs, zf`` in positions 0-5, matching the
        slab's own bounds."""
        pml = ready_pml(pml_id="x0", thickness=4)
        pml.update_electric()
        args = spy_import.calls[0][1]
        assert args[:6] == (pml.xs, pml.xf, pml.ys, pml.yf, pml.zs, pml.zf)

    def test_thread_count_follows_the_extents(self, ready_pml, spy_import):
        """Expects ``config.get_model_config().ompthreads`` in position 6 — one
        under the test fixture."""
        ready_pml().update_electric()
        assert spy_import.calls[0][1][6] == 1

    def test_electric_passes_the_electric_update_coefficients(self, ready_pml, spy_import):
        """Expects ``G.updatecoeffsE`` in position 7, not the magnetic set."""
        pml = ready_pml()
        pml.update_electric()
        assert spy_import.calls[0][1][7] is pml.G.updatecoeffsE

    def test_magnetic_passes_the_magnetic_update_coefficients(self, ready_pml, spy_import):
        """Expects ``G.updatecoeffsH`` in the same slot — the one positional
        difference in the shared head of the two signatures."""
        pml = ready_pml()
        pml.update_magnetic()
        assert spy_import.calls[0][1][7] is pml.G.updatecoeffsH

    def test_all_six_field_arrays_are_forwarded_in_order(self, ready_pml, spy_import):
        """Expects ``ID`` then ``Ex, Ey, Ez, Hx, Hy, Hz`` in positions 8-14.
        Both polarities receive all six: a PML correction couples E to H."""
        pml = ready_pml()
        pml.update_electric()
        args = spy_import.calls[0][1]
        g = pml.G
        assert args[8] is g.ID
        assert [
            args[i] is arr for i, arr in enumerate([g.Ex, g.Ey, g.Ez, g.Hx, g.Hy, g.Hz], start=9)
        ] == [True] * 6

    def test_electric_forwards_its_own_phi_arrays(self, ready_pml, spy_import):
        """Expects ``EPhi1, EPhi2`` in positions 15-16 — the electric
        accumulators, never the magnetic ones."""
        pml = ready_pml()
        pml.update_electric()
        args = spy_import.calls[0][1]
        assert args[15] is pml.EPhi1
        assert args[16] is pml.EPhi2

    def test_magnetic_forwards_its_own_phi_arrays(self, ready_pml, spy_import):
        """Expects ``HPhi1, HPhi2`` in the same slots."""
        pml = ready_pml()
        pml.update_magnetic()
        args = spy_import.calls[0][1]
        assert args[15] is pml.HPhi1
        assert args[16] is pml.HPhi2

    def test_electric_forwards_the_e_coefficients(self, ready_pml, spy_import):
        """Expects ``ERA, ERB, ERE, ERF`` in positions 17-20, in that order."""
        pml = ready_pml()
        pml.update_electric()
        args = spy_import.calls[0][1]
        assert [args[17], args[18], args[19], args[20]] == [
            pml.ERA,
            pml.ERB,
            pml.ERE,
            pml.ERF,
        ]

    def test_magnetic_forwards_the_h_coefficients(self, ready_pml, spy_import):
        """Expects ``HRA, HRB, HRE, HRF`` in the same positions."""
        pml = ready_pml()
        pml.update_magnetic()
        args = spy_import.calls[0][1]
        assert [args[17], args[18], args[19], args[20]] == [
            pml.HRA,
            pml.HRB,
            pml.HRE,
            pml.HRF,
        ]

    def test_spacing_is_the_final_argument(self, ready_pml, spy_import):
        """Expects ``d`` — the spacing along the slab normal — last."""
        pml = ready_pml()
        pml.update_electric()
        assert spy_import.calls[0][1][21] == pml.d

    def test_arrays_are_passed_by_reference(self, ready_pml, spy_import):
        """Expects the live arrays rather than copies: the kernels write the
        PML correction back into the grid in place."""
        pml = ready_pml()
        pml.update_electric()
        assert spy_import.calls[0][1][9] is pml.G.Ex

    def test_update_returns_none(self, ready_pml, spy_import):
        """Expects no return value — the kernels communicate entirely through
        in-place mutation."""
        assert ready_pml().update_electric() is None


class TestTheConventionResolvesAgainstTheRealExtensions:
    """No mocking here: the compiled modules must actually contain the names
    the convention generates."""

    @pytest.mark.parametrize("polarity", ["electric", "magnetic"])
    @pytest.mark.parametrize("formulation", ["HORIPML", "MRIPML"])
    def test_both_formulations_ship_both_polarities(self, polarity, formulation):
        """Expects all four compiled extensions to import. (4 parameter
        sets)"""
        from importlib import import_module

        assert import_module(f"gprMax.cython.pml_updates_{polarity}_{formulation}")

    @pytest.mark.parametrize("polarity", ["electric", "magnetic"])
    @pytest.mark.parametrize("formulation", ["HORIPML", "MRIPML"])
    @pytest.mark.parametrize("order", [1, 2])
    @pytest.mark.parametrize("direction", ["xminus", "xplus", "yminus", "yplus", "zminus", "zplus"])
    def test_every_generated_name_exists(self, polarity, formulation, order, direction):
        """Expects ``order<N>_<direction>`` to resolve for orders 1 and 2 in
        all six directions, across both formulations and both polarities —
        48 combinations, every one reachable from a valid input file.

        Orders above 2 are deliberately absent from the extensions;
        ``cmds_multiuse.py`` caps the CFS list at two. (48 parameter sets)"""
        from importlib import import_module

        module = import_module(f"gprMax.cython.pml_updates_{polarity}_{formulation}")
        assert callable(getattr(module, f"order{order}_{direction}"))

    def test_a_third_order_kernel_is_not_provided(self):
        """Expects ``order3_xminus`` to be absent, documenting the supported
        ceiling rather than leaving it implicit."""
        from importlib import import_module

        module = import_module("gprMax.cython.pml_updates_electric_HORIPML")
        assert not hasattr(module, "order3_xminus")


pytestmark = pytest.mark.unit
