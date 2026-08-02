"""Tests for ``SubGridBaseGrid`` — the subgrid's constructor arithmetic.

Every dimension of a Huygens subgrid derives from ``ratio``. This module pins
that derivation against hand-computed values, so a refactor that changes any
of the size formulas has to change these numbers deliberately.

``ratio`` must be odd: only then does a fine cell centre coincide with the
coarse cell centre, letting the two lattices share sample points instead of
disagreeing by half a cell everywhere.
"""

import numpy as np
import pytest

from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.subgrids.grid import SubGridBaseGrid
from gprMax.subgrids.subgrid_hsg import SubGridHSG as SubGridHSGGrid


class TestRatioValidation:
    @pytest.mark.parametrize("ratio", [1, 3, 5, 7, 9])
    def test_odd_ratios_are_accepted(self, subgrid_kwargs, ratio):
        sg = SubGridHSGGrid(**{**subgrid_kwargs, "ratio": ratio})
        assert sg.ratio == ratio

    @pytest.mark.parametrize("ratio", [2, 4, 6, 8])
    def test_even_ratios_are_rejected(self, subgrid_kwargs, ratio):
        with pytest.raises(ValueError):
            SubGridHSGGrid(**{**subgrid_kwargs, "ratio": ratio})

    def test_zero_ratio_is_rejected(self, subgrid_kwargs):
        with pytest.raises(ValueError):
            SubGridHSGGrid(**{**subgrid_kwargs, "ratio": 0})


class TestRequiredKwargs:
    @pytest.mark.parametrize(
        "missing",
        [
            "ratio",
            "id",
            "filter",
            "is_os_sep",
            "pml_separation",
            "subgrid_pml_thickness",
            "interpolation",
        ],
    )
    def test_each_kwarg_is_required(self, subgrid_kwargs, missing):
        kwargs = {k: v for k, v in subgrid_kwargs.items() if k != missing}
        with pytest.raises(KeyError):
            SubGridHSGGrid(**kwargs)

    def test_all_kwargs_present_constructs(self, subgrid_kwargs):
        assert SubGridHSGGrid(**subgrid_kwargs) is not None


class TestSizeDerivation:
    def test_s_is_os_sep_scales_with_ratio(self, subgrid_kwargs):
        sg = SubGridHSGGrid(**{**subgrid_kwargs, "is_os_sep": 4, "ratio": 3})
        assert sg.s_is_os_sep == 12

    @pytest.mark.parametrize("ratio", [1, 3, 5, 7])
    def test_s_is_os_sep_for_various_ratios(self, subgrid_kwargs, ratio):
        sg = SubGridHSGGrid(**{**subgrid_kwargs, "is_os_sep": 2, "ratio": ratio})
        assert sg.s_is_os_sep == 2 * ratio

    def test_n_boundary_cells_sums_the_three_gaps(self, subgrid_kwargs):
        kwargs = {
            **subgrid_kwargs,
            "ratio": 3,
            "is_os_sep": 2,
            "pml_separation": 4,
            "subgrid_pml_thickness": 5,
        }
        sg = SubGridHSGGrid(**kwargs)
        # s_is_os_sep (6) + pml_separation (4) + pml thickness (5)
        assert sg.n_boundary_cells == 15

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_per_axis_boundary_cells_match_the_scalar(self, subgrid_kwargs, axis):
        """All six PML thicknesses come from one kwarg, so the three per-axis
        counts are equal on construction.
        """
        sg = SubGridHSGGrid(**subgrid_kwargs)
        assert getattr(sg, f"n_boundary_cells_{axis}") == sg.n_boundary_cells

    def test_boundary_cells_grow_with_pml_thickness(self, subgrid_kwargs):
        thin = SubGridHSGGrid(**{**subgrid_kwargs, "subgrid_pml_thickness": 2})
        thick = SubGridHSGGrid(**{**subgrid_kwargs, "subgrid_pml_thickness": 10})
        assert thick.n_boundary_cells - thin.n_boundary_cells == 8


class TestPmlThickness:
    def test_all_six_faces_take_the_single_kwarg(self, subgrid_kwargs):
        sg = SubGridHSGGrid(**{**subgrid_kwargs, "subgrid_pml_thickness": 4})
        assert all(v == 4 for v in sg.pmls["thickness"].values())

    def test_overrides_the_fdtd_grid_default_of_ten(self, subgrid_kwargs):
        sg = SubGridHSGGrid(**{**subgrid_kwargs, "subgrid_pml_thickness": 6})
        assert sg.pmls["thickness"]["x0"] == 6

    def test_zero_thickness_is_allowed(self, subgrid_kwargs):
        sg = SubGridHSGGrid(**{**subgrid_kwargs, "subgrid_pml_thickness": 0})
        assert all(v == 0 for v in sg.pmls["thickness"].values())


class TestConstructorState:
    def test_name_comes_from_the_id_kwarg(self, subgrid_kwargs):
        sg = SubGridHSGGrid(**{**subgrid_kwargs, "id": "my_subgrid"})
        assert sg.name == "my_subgrid"

    def test_name_overrides_the_fdtd_grid_default(self, subgrid_kwargs):
        sg = SubGridHSGGrid(**subgrid_kwargs)
        assert sg.name != "main_grid"

    def test_iterations_start_at_zero(self, subgrid_kwargs):
        assert SubGridHSGGrid(**subgrid_kwargs).iterations == 0

    def test_filter_flag_is_stored(self, subgrid_kwargs):
        assert SubGridHSGGrid(**{**subgrid_kwargs, "filter": False}).filter is False

    def test_interpolation_is_stored(self, subgrid_kwargs):
        assert SubGridHSGGrid(**{**subgrid_kwargs, "interpolation": 2}).interpolation == 2

    def test_is_os_sep_is_stored(self, subgrid_kwargs):
        assert SubGridHSGGrid(**{**subgrid_kwargs, "is_os_sep": 5}).is_os_sep == 5


class TestAbstractBase:
    def test_base_class_cannot_be_instantiated(self, subgrid_kwargs):
        with pytest.raises(TypeError):
            SubGridBaseGrid(**subgrid_kwargs)

    def test_subclass_missing_an_abstract_method_cannot_be_instantiated(
        self, subgrid_kwargs
    ):
        class Incomplete(SubGridBaseGrid):
            def update_magnetic_is(self, precursors):
                pass

            def update_electric_is(self, precursors):
                pass

            def update_electric_os(self, main_grid):
                pass

            def update_magnetic_os(self, main_grid):
                pass

            # print_info deliberately not implemented

        with pytest.raises(TypeError):
            Incomplete(**subgrid_kwargs)

    def test_complete_subclass_can_be_instantiated(self, subgrid_kwargs):
        class Complete(SubGridBaseGrid):
            def update_magnetic_is(self, precursors):
                pass

            def update_electric_is(self, precursors):
                pass

            def update_electric_os(self, main_grid):
                pass

            def update_magnetic_os(self, main_grid):
                pass

            def print_info(self):
                pass

        assert Complete(**subgrid_kwargs) is not None

    def test_hsg_implements_the_whole_interface(self, subgrid_kwargs):
        sg = SubGridHSGGrid(**subgrid_kwargs)
        for name in (
            "update_magnetic_is",
            "update_electric_is",
            "update_electric_os",
            "update_magnetic_os",
            "print_info",
        ):
            assert callable(getattr(sg, name))


class TestInheritsFdtdGrid:
    """A subgrid *is* an ``FDTDGrid``, so the whole PR-9 grid surface applies."""

    def test_is_an_fdtd_grid(self, subgrid_kwargs):
        assert isinstance(SubGridHSGGrid(**subgrid_kwargs), FDTDGrid)

    def test_size_properties_work(self, make_subgrid):
        sg = make_subgrid()
        assert (sg.nx, sg.ny, sg.nz) == (32, 32, 32)

    def test_discretisation_is_ratio_times_finer(self, make_subgrid):
        from .conftest import DL

        sg = make_subgrid()
        assert sg.dx == pytest.approx(DL / 3)

    def test_within_bounds_contract_is_inherited(self, make_subgrid):
        sg = make_subgrid()
        assert sg.within_bounds(np.array([0, 0, 0])) is True
        with pytest.raises(ValueError, match="x"):
            sg.within_bounds(np.array([-1, 0, 0]))

    def test_array_initialisers_are_inherited(self, make_subgrid):
        sg = make_subgrid(arrays=True)
        assert sg.solid.shape == (32, 32, 32)
        assert sg.Ex.shape == (33, 33, 33)

    def test_calculate_dt_uses_the_finer_spacing(self, make_subgrid):
        """A ratio-3 subgrid has a time step three times smaller."""
        from scipy.constants import c

        from .conftest import DL

        sg = make_subgrid()
        sg.calculate_dt()
        assert sg.dt == pytest.approx((DL / 3) / (c * np.sqrt(3)))
