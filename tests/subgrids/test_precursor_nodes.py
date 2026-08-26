"""Tests for ``precursor_nodes.py`` — the translator between grid rates.

The main grid produces field values once per main time step; the subgrid needs
them ``ratio`` times per main step. The precursor nodes hold the main grid's
fields at the previous (``_0``) and current (``_1``) main tick and interpolate
between them for each intermediate subgrid tick.

Two interpolations are involved:

- **in time**, a linear blend of ``_0`` and ``_1`` whose weights come from
  ``calculate_weighting_coefficients``;
- **in space**, a ``RectBivariateSpline`` across each Huygens face, because
  the face carries ``ratio`` times more subgrid cells than main cells.
"""

import numpy as np
import pytest

from gprMax.subgrids.precursor_nodes import (
    PrecursorNodes,
    PrecursorNodesFiltered,
    calculate_weighting_coefficients,
)


class TestWeightingCoefficients:
    """``c1 = (x - x1)/x``, ``c2 = x1/x`` — a linear partition of unity."""

    def test_at_the_start_all_weight_is_on_the_previous_value(self):
        assert calculate_weighting_coefficients(0, 3) == (1.0, 0.0)

    def test_at_the_end_all_weight_is_on_the_current_value(self):
        assert calculate_weighting_coefficients(3, 3) == (0.0, 1.0)

    def test_midpoint_splits_evenly(self):
        c1, c2 = calculate_weighting_coefficients(2, 4)
        assert (c1, c2) == (0.5, 0.5)

    @pytest.mark.parametrize("ratio", [1, 3, 5, 7])
    @pytest.mark.parametrize("m", [0, 1, 2])
    def test_weights_sum_to_one(self, ratio, m):
        """A partition of unity: any other sum would scale the field."""
        c1, c2 = calculate_weighting_coefficients(m, ratio)
        assert c1 + c2 == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "m,ratio,expected", [(1, 3, (2 / 3, 1 / 3)), (2, 3, (1 / 3, 2 / 3)), (1, 5, (4 / 5, 1 / 5))]
    )
    def test_intermediate_values(self, m, ratio, expected):
        assert calculate_weighting_coefficients(m, ratio) == pytest.approx(expected)

    def test_is_monotone_in_m(self):
        """Advancing through the sub-steps moves weight steadily from old to
        new; a non-monotone blend would make the source jitter.
        """
        weights = [calculate_weighting_coefficients(m, 5)[1] for m in range(6)]
        assert weights == sorted(weights)


class TestPrecursorConstruction:
    def test_copies_the_scaling_parameters_from_the_subgrid(self, coupled_grids):
        c = coupled_grids()
        assert c.precursors.ratio == c.sub.ratio
        assert c.precursors.nwx == c.sub.nwx
        assert c.precursors.interpolation == c.sub.interpolation

    def test_copies_the_is_indices(self, coupled_grids):
        c = coupled_grids()
        assert (c.precursors.i0, c.precursors.j0, c.precursors.k0) == (
            c.sub.i0,
            c.sub.j0,
            c.sub.k0,
        )
        assert (c.precursors.i1, c.precursors.j1, c.precursors.k1) == (
            c.sub.i1,
            c.sub.j1,
            c.sub.k1,
        )

    def test_holds_references_to_the_main_grid_fields(self, coupled_grids):
        """Not copies — the precursors must see the main grid's live arrays."""
        c = coupled_grids()
        assert c.precursors.Ex is c.main.Ex
        assert c.precursors.Hz is c.main.Hz

    def test_half_sub_cell_offset(self, coupled_grids):
        c = coupled_grids()
        assert c.precursors.d == pytest.approx(1 / (2 * c.sub.ratio))

    def test_left_and_right_weights_partition_the_ratio(self, coupled_grids):
        c = coupled_grids()
        assert c.precursors.l_weight + c.precursors.r_weight == c.sub.ratio

    def test_left_weight_is_the_floor_of_half(self, coupled_grids):
        c = coupled_grids()
        assert c.precursors.l_weight == c.sub.ratio // 2


class TestFieldArrayShapes:
    """Shapes follow the Yee stagger: a component is offset by half a cell
    along its own direction and not along the others, so it has one fewer
    sample there and one more on each transverse axis.
    """

    def test_front_face_electric_shapes(self, coupled_grids):
        c = coupled_grids()
        p, sub = c.precursors, c.sub
        assert p.ex_front_1.shape == (sub.nwx, sub.nwz + 1)
        assert p.ez_front_1.shape == (sub.nwx + 1, sub.nwz)

    def test_left_face_electric_shapes(self, coupled_grids):
        c = coupled_grids()
        p, sub = c.precursors, c.sub
        assert p.ey_left_1.shape == (sub.nwy, sub.nwz + 1)
        assert p.ez_left_1.shape == (sub.nwy + 1, sub.nwz)

    def test_bottom_face_electric_shapes(self, coupled_grids):
        c = coupled_grids()
        p, sub = c.precursors, c.sub
        assert p.ex_bottom_1.shape == (sub.nwx, sub.nwy + 1)
        assert p.ey_bottom_1.shape == (sub.nwx + 1, sub.nwy)

    def test_magnetic_shapes_are_the_transverse_swap(self, coupled_grids):
        """``hx_front`` is shaped like ``ez_front``, and ``hz_front`` like
        ``ex_front`` — H sits on the faces E threads through.
        """
        c = coupled_grids()
        p = c.precursors
        assert p.hx_front_1.shape == p.ez_front_1.shape
        assert p.hz_front_1.shape == p.ex_front_1.shape

    def test_opposite_faces_have_equal_shapes(self, coupled_grids):
        c = coupled_grids()
        p = c.precursors
        assert p.ex_front_1.shape == p.ex_back_1.shape
        assert p.ey_left_1.shape == p.ey_right_1.shape
        assert p.ex_bottom_1.shape == p.ex_top_1.shape

    def test_all_arrays_start_at_zero(self, coupled_grids):
        c = coupled_grids()
        p = c.precursors
        for name in p.fn_e + p.fn_m:
            assert np.all(getattr(p, f"{name}_0") == 0)
            assert np.all(getattr(p, f"{name}_1") == 0)

    def test_previous_and_current_pages_are_distinct_arrays(self, coupled_grids):
        """If ``_0`` aliased ``_1`` the time interpolation would collapse to
        the current value and the subgrid would see a stepped source.
        """
        c = coupled_grids()
        p = c.precursors
        for name in p.fn_e + p.fn_m:
            assert getattr(p, f"{name}_0") is not getattr(p, f"{name}_1")

    def test_faces_are_distinct_arrays(self, coupled_grids):
        c = coupled_grids()
        p = c.precursors
        p.ex_front_1[0, 0] = 1.0
        assert p.ex_back_1[0, 0] == 0.0


class TestFieldNameTables:
    def test_twelve_electric_names(self, coupled_grids):
        assert len(coupled_grids().precursors.fn_e) == 12

    def test_twelve_magnetic_names(self, coupled_grids):
        assert len(coupled_grids().precursors.fn_m) == 12

    def test_names_are_unique(self, coupled_grids):
        p = coupled_grids().precursors
        assert len(set(p.fn_e)) == 12
        assert len(set(p.fn_m)) == 12

    def test_every_name_resolves_to_both_pages(self, coupled_grids):
        p = coupled_grids().precursors
        for name in p.fn_e + p.fn_m:
            assert hasattr(p, f"{name}_0")
            assert hasattr(p, f"{name}_1")

    def test_electric_and_magnetic_names_do_not_overlap(self, coupled_grids):
        p = coupled_grids().precursors
        assert not set(p.fn_e) & set(p.fn_m)

    def test_all_six_faces_appear_in_each_table(self, coupled_grids):
        p = coupled_grids().precursors
        for table in (p.fn_e, p.fn_m):
            for face in ("front", "back", "left", "right", "top", "bottom"):
                assert any(face in name for name in table)


class TestTimeInterpolation:
    @pytest.fixture
    def loaded(self, coupled_grids):
        """Precursors with known constants on both time pages."""
        c = coupled_grids()
        p = c.precursors
        for name in p.fn_e + p.fn_m:
            getattr(p, f"{name}_0")[:] = 10.0
            getattr(p, f"{name}_1")[:] = 20.0
        return p

    def test_start_of_step_gives_the_previous_value(self, loaded):
        loaded.interpolate_electric_in_time(0)
        assert np.all(loaded.ex_front == 10.0)

    def test_end_of_step_gives_the_current_value(self, loaded):
        loaded.interpolate_electric_in_time(loaded.ratio)
        assert np.all(loaded.ex_front == 20.0)

    def test_intermediate_step_blends(self, loaded):
        loaded.interpolate_electric_in_time(1)  # ratio 3 -> 2/3 old, 1/3 new
        assert np.all(loaded.ex_front == pytest.approx(10 * 2 / 3 + 20 * 1 / 3))

    def test_electric_interpolation_leaves_magnetic_alone(self, loaded):
        loaded.interpolate_electric_in_time(1)
        assert not hasattr(loaded, "hx_front")

    def test_magnetic_interpolation_covers_every_magnetic_name(self, loaded):
        loaded.interpolate_magnetic_in_time(1)
        for name in loaded.fn_m:
            assert hasattr(loaded, name)

    def test_electric_interpolation_covers_every_electric_name(self, loaded):
        loaded.interpolate_electric_in_time(1)
        for name in loaded.fn_e:
            assert hasattr(loaded, name)

    def test_exact_electric_takes_the_current_page(self, loaded):
        loaded.calc_exact_electric_in_time()
        assert np.all(loaded.ex_front == 20.0)

    def test_exact_magnetic_takes_the_current_page(self, loaded):
        loaded.calc_exact_magnetic_in_time()
        assert np.all(loaded.hx_front == 20.0)

    def test_exact_field_is_a_copy_not_a_view(self, loaded):
        """Writing through the working field must not corrupt the stored
        current-step page.
        """
        loaded.calc_exact_electric_in_time()
        loaded.ex_front[0, 0] = -1.0
        assert loaded.ex_front_1[0, 0] == 20.0

    def test_interpolated_field_is_not_a_view(self, loaded):
        loaded.interpolate_electric_in_time(1)
        loaded.ex_front[0, 0] = -1.0
        assert loaded.ex_front_0[0, 0] == 10.0
        assert loaded.ex_front_1[0, 0] == 20.0


class TestPageRotation:
    def test_current_page_is_copied_into_previous(self, coupled_grids):
        c = coupled_grids()
        p = c.precursors
        p.ex_front_1[:] = 7.0
        p.update_previous_timestep_fields(p.fn_e)
        assert np.all(p.ex_front_0 == 7.0)

    def test_rotation_copies_by_value(self, coupled_grids):
        c = coupled_grids()
        p = c.precursors
        p.ex_front_1[:] = 7.0
        p.update_previous_timestep_fields(p.fn_e)
        p.ex_front_1[:] = 9.0
        assert np.all(p.ex_front_0 == 7.0)

    def test_rotation_covers_every_named_field(self, coupled_grids):
        c = coupled_grids()
        p = c.precursors
        for name in p.fn_m:
            getattr(p, f"{name}_1")[:] = 3.0
        p.update_previous_timestep_fields(p.fn_m)
        for name in p.fn_m:
            assert np.all(getattr(p, f"{name}_0") == 3.0)


class TestInterpolatedCoords:
    def test_mid_branch_offsets_the_first_axis(self, coupled_grids):
        p = coupled_grids().precursors
        field = np.zeros((4, 5))
        x, z, x_sg, z_sg = p.create_interpolated_coords(True, field)
        assert x[0] == 0.5
        assert z[0] == 0.0

    def test_non_mid_branch_offsets_the_second_axis(self, coupled_grids):
        p = coupled_grids().precursors
        field = np.zeros((4, 5))
        x, z, x_sg, z_sg = p.create_interpolated_coords(False, field)
        assert x[0] == 0.0
        assert z[0] == 0.5

    def test_mid_branch_output_lengths(self, coupled_grids):
        p = coupled_grids().precursors
        n_x, n_y, ratio = 4, 5, p.ratio
        _, _, x_sg, z_sg = p.create_interpolated_coords(True, np.zeros((n_x, n_y)))
        assert len(x_sg) == n_x * ratio
        assert len(z_sg) == (n_y - 1) * ratio + 1

    def test_non_mid_branch_output_lengths(self, coupled_grids):
        p = coupled_grids().precursors
        n_x, n_y, ratio = 4, 5, p.ratio
        _, _, x_sg, z_sg = p.create_interpolated_coords(False, np.zeros((n_x, n_y)))
        assert len(x_sg) == (n_x - 1) * ratio + 1
        assert len(z_sg) == n_y * ratio

    def test_sample_coords_match_the_field_shape(self, coupled_grids):
        p = coupled_grids().precursors
        x, z, _, _ = p.create_interpolated_coords(True, np.zeros((4, 5)))
        assert (len(x), len(z)) == (4, 5)


class TestSpatialInterpolation:
    def test_output_shape_follows_the_requested_coords(self, coupled_grids):
        p = coupled_grids().precursors
        field = np.zeros((4, 5))
        coords = p.create_interpolated_coords(True, field)
        assert p.interpolate_to_sub_grid(field, coords).shape == (
            len(coords[2]),
            len(coords[3]),
        )

    def test_constant_field_interpolates_to_the_same_constant(self, coupled_grids):
        p = coupled_grids().precursors
        field = np.full((4, 5), 2.5)
        coords = p.create_interpolated_coords(True, field)
        assert np.allclose(p.interpolate_to_sub_grid(field, coords), 2.5)

    def test_linear_field_is_reproduced_by_linear_interpolation(self, coupled_grids):
        """With ``interpolation=1`` the spline is linear, so a linear ramp must
        come back exactly — the strongest available correctness check that
        needs no reference implementation.
        """
        p = coupled_grids()
        p = p.precursors
        n_x, n_y = 4, 5
        x = np.arange(n_x, dtype=float)
        field = np.repeat(x[:, None], n_y, axis=1)  # varies only along axis 0
        coords = p.create_interpolated_coords(False, field)
        out = p.interpolate_to_sub_grid(field, coords)
        expected = np.repeat(coords[2][:, None], len(coords[3]), axis=1)
        assert np.allclose(out, expected)

    def test_zero_field_interpolates_to_zero(self, coupled_grids):
        p = coupled_grids().precursors
        field = np.zeros((4, 5))
        coords = p.create_interpolated_coords(True, field)
        assert np.allclose(p.interpolate_to_sub_grid(field, coords), 0.0)


class TestSliceTables:
    @pytest.mark.parametrize("filtered", [False, True])
    def test_twelve_slices_per_field_type(self, coupled_grids, filtered):
        p = coupled_grids(filtered=filtered).precursors
        assert len(p.magnetic_slices) == 12
        assert len(p.electric_slices) == 12

    @pytest.mark.parametrize("filtered", [False, True])
    def test_every_slice_targets_a_real_attribute(self, coupled_grids, filtered):
        p = coupled_grids(filtered=filtered).precursors
        for obj in p.magnetic_slices + p.electric_slices:
            assert hasattr(p, obj[0])

    @pytest.mark.parametrize("filtered", [False, True])
    def test_slice_targets_are_current_step_pages(self, coupled_grids, filtered):
        """The slice tables fill the ``_1`` pages; the bare names come later,
        from the time interpolation.
        """
        p = coupled_grids(filtered=filtered).precursors
        for obj in p.magnetic_slices + p.electric_slices:
            assert obj[0].endswith("_1")

    @pytest.mark.parametrize("filtered", [False, True])
    def test_coords_are_resolved_at_construction(self, coupled_grids, filtered):
        """``obj[1]`` starts as a boolean ``mid`` flag and is replaced in place
        by the coordinate 4-tuple at the end of each slice-table builder.
        """
        p = coupled_grids(filtered=filtered).precursors
        for obj in p.magnetic_slices + p.electric_slices:
            assert len(obj[1]) == 4

    def test_magnetic_slices_carry_two_index_tuples(self, coupled_grids):
        """H is averaged across the IS, so it needs samples either side."""
        p = coupled_grids().precursors
        for obj in p.magnetic_slices:
            assert len(obj) == 5

    def test_electric_slices_carry_one_index_tuple(self, coupled_grids):
        p = coupled_grids().precursors
        for obj in p.electric_slices:
            assert len(obj) == 4


class TestUpdateFromMainGrid:
    def test_update_electric_runs_and_fills_the_current_page(self, coupled_grids):
        c = coupled_grids()
        c.main.Ex[:] = 3.0
        c.main.Ey[:] = 3.0
        c.main.Ez[:] = 3.0
        c.precursors.update_electric()
        assert np.allclose(c.precursors.ex_front_1, 3.0)

    def test_update_electric_rotates_the_pages_first(self, coupled_grids):
        c = coupled_grids()
        c.precursors.ex_front_1[:] = 5.0
        c.precursors.update_electric()
        assert np.all(c.precursors.ex_front_0 == 5.0)

    def test_update_magnetic_runs_and_fills_the_current_page(self, coupled_grids):
        c = coupled_grids()
        c.main.Hx[:] = 4.0
        c.main.Hy[:] = 4.0
        c.main.Hz[:] = 4.0
        c.precursors.update_magnetic()
        assert np.allclose(c.precursors.hx_front_1, 4.0)

    def test_refining_ratio_rejects_bypassed_magnetic_interpolation(
        self, coupled_grids, monkeypatch
    ):
        c = coupled_grids(ratio=3)
        monkeypatch.setattr(c.precursors, "interpolate_to_sub_grid", lambda field, coords: field)

        with pytest.raises(RuntimeError, match="did not refine"):
            c.precursors.update_magnetic()

    def test_unity_ratio_allows_identity_magnetic_interpolation(self, coupled_grids):
        c = coupled_grids(ratio=1)
        c.main.Hx[:] = 2.0
        c.main.Hy[:] = 2.0
        c.main.Hz[:] = 2.0

        c.precursors.update_magnetic()

        assert np.allclose(c.precursors.hx_front_1, 2.0)

    def test_zero_main_grid_gives_zero_precursors(self, coupled_grids):
        c = coupled_grids()
        c.precursors.update_electric()
        assert np.allclose(c.precursors.ex_front_1, 0.0)

    @pytest.mark.parametrize("filtered", [False, True])
    def test_both_precursor_types_update(self, coupled_grids, filtered):
        c = coupled_grids(filtered=filtered)
        c.main.Ex[:] = 1.0
        c.main.Ey[:] = 1.0
        c.main.Ez[:] = 1.0
        c.precursors.update_electric()
        c.precursors.update_magnetic()
        assert c.precursors.ex_front_1.shape == (c.sub.nwx, c.sub.nwz + 1)


class TestPrecursorTypes:
    def test_unfiltered_type(self, coupled_grids):
        assert isinstance(coupled_grids(filtered=False).precursors, PrecursorNodes)

    def test_filtered_type(self, coupled_grids):
        assert isinstance(coupled_grids(filtered=True).precursors, PrecursorNodesFiltered)

    def test_both_share_the_same_field_shapes(self, coupled_grids):
        plain = coupled_grids(filtered=False).precursors
        filtered = coupled_grids(filtered=True).precursors
        for name in plain.fn_e + plain.fn_m:
            assert getattr(plain, f"{name}_1").shape == getattr(filtered, f"{name}_1").shape


pytestmark = pytest.mark.unit
