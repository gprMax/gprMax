"""Tests for ``subgrid_hsg.py`` — the Inner/Outer Surface field stitching.

The four update methods wrap OpenMP Cython kernels
(``cython/fields_updates_hsg.pyx``) that inject the main grid's fields into
the subgrid across the Inner Surface, and push the subgrid's result back out
across the Outer Surface.

These tests assert **locality and effect**, never physical correctness:

- zero in gives zero out (no spurious injection);
- a single excited precursor cell changes exactly the cell it should;
- the six faces are independent;
- the OS updates touch the main grid only near the Outer Surface.

Verifying the stitched field is numerically *right* needs an analytic
reference solution and belongs to the integration suite.

The update-coefficient arrays are filled with ones by the fixture. They must
be non-zero for any of this to be observable — the kernels multiply the
incoming value by a coefficient looked up through the ``ID`` array.
"""

import numpy as np
import pytest

from .conftest import nonzero_set


@pytest.fixture
def coupled(coupled_grids):
    return coupled_grids()


class TestCoupledGridsFixture:
    """The IS/OS tests are only meaningful if the fixture is self-consistent,
    so it is asserted directly before anything else relies on it.
    """

    def test_subgrid_knows_its_parent(self, coupled):
        assert coupled.sub.parent_grid is coupled.main

    def test_inner_surface_sits_inside_the_main_grid(self, coupled):
        sub, main = coupled.sub, coupled.main
        assert 0 < sub.i0 < sub.i1 < main.nx
        assert 0 < sub.j0 < sub.j1 < main.ny
        assert 0 < sub.k0 < sub.k1 < main.nz

    def test_outer_surface_also_fits_inside_the_main_grid(self, coupled):
        """The OS sits ``is_os_sep`` main cells outside the IS."""
        sub, main = coupled.sub, coupled.main
        assert sub.i0 - sub.is_os_sep >= 0
        assert sub.i1 + sub.is_os_sep <= main.nx

    def test_precursor_slices_stay_inside_the_main_grid(self, coupled):
        """The magnetic slices reach one main cell below the IS."""
        assert coupled.sub.i0 - 1 >= 0

    def test_working_region_scales_by_ratio(self, coupled):
        sub = coupled.sub
        assert sub.nwx == (sub.i1 - sub.i0) * sub.ratio

    def test_total_size_brackets_the_working_region(self, coupled):
        sub = coupled.sub
        assert sub.nx == 2 * sub.n_boundary_cells_x + sub.nwx

    def test_subgrid_time_step_is_finer(self, coupled):
        assert coupled.sub.dt == pytest.approx(coupled.main.dt / coupled.sub.ratio)

    def test_both_grids_have_usable_update_coefficients(self, coupled):
        assert np.all(coupled.sub.updatecoeffsE == 1.0)
        assert np.all(coupled.main.updatecoeffsH == 1.0)

    def test_field_arrays_start_at_zero(self, coupled):
        assert not np.any(coupled.sub.Hy)
        assert not np.any(coupled.main.Ex)


class TestNoSpuriousInjection:
    """With every precursor zero, the IS updates must leave the subgrid
    untouched. A non-zero result would mean the Huygens surface is radiating
    on its own.
    """

    def test_magnetic_is_with_zero_precursors(self, coupled):
        coupled.precursors.calc_exact_magnetic_in_time()
        coupled.precursors.calc_exact_electric_in_time()
        before = coupled.sub.Hy.copy()
        coupled.sub.update_magnetic_is(coupled.precursors)
        assert np.array_equal(coupled.sub.Hy, before)

    def test_electric_is_with_zero_precursors(self, coupled):
        coupled.precursors.calc_exact_magnetic_in_time()
        coupled.precursors.calc_exact_electric_in_time()
        before = coupled.sub.Ex.copy()
        coupled.sub.update_electric_is(coupled.precursors)
        assert np.array_equal(coupled.sub.Ex, before)

    def test_electric_os_with_zero_subgrid_fields(self, coupled):
        before = coupled.main.Ex.copy()
        coupled.sub.update_electric_os(coupled.main)
        assert np.array_equal(coupled.main.Ex, before)

    def test_magnetic_os_with_zero_subgrid_fields(self, coupled):
        before = coupled.main.Hx.copy()
        coupled.sub.update_magnetic_os(coupled.main)
        assert np.array_equal(coupled.main.Hx, before)


class TestInnerSurfaceLocality:
    """One excited precursor cell must change exactly one subgrid cell.

    The mapping is ``precursor[a, b]`` -> subgrid cell offset by
    ``n_boundary_cells`` on the two in-face axes, on the layer just inside the
    corresponding face.
    """

    @pytest.fixture
    def ready(self, coupled):
        coupled.precursors.calc_exact_magnetic_in_time()
        coupled.precursors.calc_exact_electric_in_time()
        return coupled

    def test_bottom_face_changes_exactly_one_cell(self, ready):
        sub, p = ready.sub, ready.precursors
        p.ex_bottom[5, 5] = 1.0
        before = sub.Hy.copy()
        sub.update_magnetic_is(p)
        changed = nonzero_set(sub.Hy - before)
        n = sub.n_boundary_cells
        assert changed == {(n + 5, n + 5, n - 1)}

    def test_top_face_changes_exactly_one_cell(self, ready):
        sub, p = ready.sub, ready.precursors
        p.ex_top[5, 5] = 1.0
        before = sub.Hy.copy()
        sub.update_magnetic_is(p)
        changed = nonzero_set(sub.Hy - before)
        n = sub.n_boundary_cells
        assert changed == {(n + 5, n + 5, n + sub.nwz)}

    def test_bottom_and_top_are_independent(self, ready):
        """Exciting one face must not disturb the opposite one."""
        sub, p = ready.sub, ready.precursors
        p.ex_bottom[5, 5] = 1.0
        before = sub.Hy.copy()
        sub.update_magnetic_is(p)
        changed = nonzero_set(sub.Hy - before)
        top_layer = sub.n_boundary_cells + sub.nwz
        assert all(k != top_layer for _, _, k in changed)

    @pytest.mark.parametrize("a,b", [(0, 0), (3, 7), (17, 17)])
    def test_mapping_holds_across_the_face(self, ready, a, b):
        sub, p = ready.sub, ready.precursors
        p.ex_bottom[a, b] = 1.0
        before = sub.Hy.copy()
        sub.update_magnetic_is(p)
        n = sub.n_boundary_cells
        assert nonzero_set(sub.Hy - before) == {(n + a, n + b, n - 1)}

    def test_two_excited_cells_change_two_cells(self, ready):
        sub, p = ready.sub, ready.precursors
        p.ex_bottom[2, 2] = 1.0
        p.ex_bottom[9, 4] = 1.0
        before = sub.Hy.copy()
        sub.update_magnetic_is(p)
        n = sub.n_boundary_cells
        assert nonzero_set(sub.Hy - before) == {
            (n + 2, n + 2, n - 1),
            (n + 9, n + 4, n - 1),
        }

    def test_effect_is_linear_in_the_precursor_value(self, ready):
        sub, p = ready.sub, ready.precursors
        n = sub.n_boundary_cells

        p.ex_bottom[5, 5] = 1.0
        sub.update_magnetic_is(p)
        single = sub.Hy[n + 5, n + 5, n - 1]

        sub.Hy[:] = 0.0
        p.ex_bottom[5, 5] = 2.0
        sub.update_magnetic_is(p)
        assert sub.Hy[n + 5, n + 5, n - 1] == pytest.approx(2 * single)

    def test_electric_is_changes_cells_in_the_subgrid(self, ready):
        """The electric counterpart is driven by the magnetic precursors."""
        sub, p = ready.sub, ready.precursors
        p.hz_front[:] = 1.0
        before = sub.Ex.copy()
        sub.update_electric_is(p)
        assert np.any(sub.Ex != before)

    def test_electric_is_stays_within_the_working_region(self, ready):
        sub, p = ready.sub, ready.precursors
        p.hz_front[:] = 1.0
        before = sub.Ex.copy()
        sub.update_electric_is(p)
        changed = np.argwhere(sub.Ex != before)
        lo = sub.n_boundary_cells - 1
        hi = sub.n_boundary_cells + sub.nwx + 1
        assert changed.min() >= lo
        assert changed.max() <= hi


class TestOuterSurfaceLocality:
    """The OS updates write into the *main* grid, and only near the Outer
    Surface — everything beyond it must be untouched.
    """

    def os_bounds(self, sub):
        return sub.i0 - sub.is_os_sep, sub.i1 + sub.is_os_sep

    def test_electric_os_writes_into_the_main_grid(self, coupled):
        coupled.sub.Hx[:] = 1.0
        coupled.sub.Hy[:] = 1.0
        coupled.sub.Hz[:] = 1.0
        before = coupled.main.Ex.copy()
        coupled.sub.update_electric_os(coupled.main)
        assert np.any(coupled.main.Ex != before)

    @pytest.mark.parametrize("component", ["Ex", "Ey", "Ez"])
    def test_electric_os_stays_within_the_outer_surface(self, coupled, component):
        sub, main = coupled.sub, coupled.main
        sub.Hx[:] = 1.0
        sub.Hy[:] = 1.0
        sub.Hz[:] = 1.0
        before = getattr(main, component).copy()
        sub.update_electric_os(main)
        changed = np.argwhere(getattr(main, component) != before)
        lo, hi = self.os_bounds(sub)
        assert changed.min() >= lo
        assert changed.max() <= hi

    @pytest.mark.parametrize("component", ["Hx", "Hy", "Hz"])
    def test_magnetic_os_stays_within_the_outer_surface(self, coupled, component):
        """Magnetic nodes sit half a cell back, so they reach one cell
        further on the low side.
        """
        sub, main = coupled.sub, coupled.main
        sub.Ex[:] = 1.0
        sub.Ey[:] = 1.0
        sub.Ez[:] = 1.0
        before = getattr(main, component).copy()
        sub.update_magnetic_os(main)
        changed = np.argwhere(getattr(main, component) != before)
        lo, hi = self.os_bounds(sub)
        assert changed.min() >= lo - 1
        assert changed.max() <= hi

    def test_far_field_is_untouched(self, coupled):
        """A cell in the far corner of the domain must never move."""
        sub, main = coupled.sub, coupled.main
        sub.Hx[:] = 1.0
        sub.Hy[:] = 1.0
        sub.Hz[:] = 1.0
        sub.update_electric_os(main)
        assert main.Ex[0, 0, 0] == 0.0
        assert main.Ex[main.nx - 1, main.ny - 1, main.nz - 1] == 0.0

    def test_electric_os_does_not_disturb_the_magnetic_field(self, coupled):
        sub, main = coupled.sub, coupled.main
        sub.Hx[:] = 1.0
        sub.Hy[:] = 1.0
        sub.Hz[:] = 1.0
        before = main.Hx.copy()
        sub.update_electric_os(main)
        assert np.array_equal(main.Hx, before)

    def test_magnetic_os_does_not_disturb_the_electric_field(self, coupled):
        sub, main = coupled.sub, coupled.main
        sub.Ex[:] = 1.0
        sub.Ey[:] = 1.0
        sub.Ez[:] = 1.0
        before = main.Ex.copy()
        sub.update_magnetic_os(main)
        assert np.array_equal(main.Ex, before)

    def test_os_effect_is_linear(self, coupled):
        sub, main = coupled.sub, coupled.main
        sub.Hz[:] = 1.0
        sub.update_electric_os(main)
        single = main.Ex.copy()

        main.Ex[:] = 0.0
        sub.Hz[:] = 2.0
        sub.update_electric_os(main)
        assert np.allclose(main.Ex, 2 * single)


class TestRatioVariations:
    """The kernels must work for every supported refinement factor."""

    @pytest.mark.parametrize("ratio", [3, 5, 7])
    def test_inner_surface_mapping_holds(self, coupled_grids, ratio):
        c = coupled_grids(ratio=ratio)
        c.precursors.calc_exact_magnetic_in_time()
        c.precursors.calc_exact_electric_in_time()
        sub, p = c.sub, c.precursors
        p.ex_bottom[4, 4] = 1.0
        before = sub.Hy.copy()
        sub.update_magnetic_is(p)
        n = sub.n_boundary_cells
        assert nonzero_set(sub.Hy - before) == {(n + 4, n + 4, n - 1)}

    @pytest.mark.parametrize("ratio", [3, 5, 7])
    def test_working_region_scales(self, coupled_grids, ratio):
        c = coupled_grids(ratio=ratio)
        assert c.sub.nwx == (c.sub.i1 - c.sub.i0) * ratio

    @pytest.mark.parametrize("ratio", [3, 5, 7])
    def test_outer_surface_stays_local(self, coupled_grids, ratio):
        c = coupled_grids(ratio=ratio)
        c.sub.Hz[:] = 1.0
        before = c.main.Ex.copy()
        c.sub.update_electric_os(c.main)
        changed = np.argwhere(c.main.Ex != before)
        assert changed.min() >= c.sub.i0 - c.sub.is_os_sep
        assert changed.max() <= c.sub.i1 + c.sub.is_os_sep


class TestPrintInfo:
    """``print_info`` logs; it returns ``None``."""

    def test_returns_none(self, coupled, caplog):
        with caplog.at_level("DEBUG"):
            assert coupled.sub.print_info() is None

    def test_reports_the_ratio(self, coupled, caplog):
        with caplog.at_level("INFO"):
            coupled.sub.print_info()
        assert f"1:{coupled.sub.ratio}" in caplog.text

    def test_names_the_grid(self, coupled, caplog):
        with caplog.at_level("INFO"):
            coupled.sub.print_info()
        assert coupled.sub.name in caplog.text

    def test_reports_the_working_region_cell_count(self, coupled, caplog):
        with caplog.at_level("INFO"):
            coupled.sub.print_info()
        total = coupled.sub.nwx * coupled.sub.nwy * coupled.sub.nwz
        assert str(total) in caplog.text

    def test_reports_the_time_step(self, coupled, caplog):
        with caplog.at_level("INFO"):
            coupled.sub.print_info()
        assert "Time step" in caplog.text


pytestmark = pytest.mark.unit
