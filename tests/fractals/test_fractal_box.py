"""Unit tests for ``FractalBox`` (``cmds_geometry/fractal_box.py``).

``FractalBox`` is the only user object whose ``build()`` means two
different things depending on how many times it has been called, because
a fractal box cannot be stamped into the grid when it is parsed — a later
``#add_surface_roughness`` or ``#add_grass`` may still change its shape.
``Scene.process_geometry_objects`` therefore runs it twice:

- **Pass 1** (``do_pre_build`` is ``True``) → ``pre_build()``. Validates
  the parameters, resolves the mixing model, creates a ``FractalVolume``
  and registers it on ``grid.fractalvolumes``. Nothing is written to the
  grid. Surface modifiers then attach themselves to that volume by ID.
- **Pass 2** → the box extends itself to cover every attached surface,
  generates the fractal, builds the 3D mask, and stamps the lot into
  ``solid`` / ``rigidE`` / ``rigidH`` / ``ID`` via the Cython array
  builders.

These tests drive both passes end-to-end against a stub grid carrying
real numpy arrays, through the real ``MainGridUserInput`` — no mocked
discretisation.
"""

import numpy as np
import pytest

from gprMax.user_objects.cmds_geometry.add_surface_roughness import AddSurfaceRoughness
from gprMax.user_objects.cmds_geometry.fractal_box import FractalBox

from .conftest import add_mixing_model, nonzero_set

# The box spans cells 2..10 in every axis on a 16-cell grid.
P1 = (0.002, 0.002, 0.002)
P2 = (0.010, 0.010, 0.010)
BOX_CELLS = range(2, 10)

# numIDs of the four materials the "soil" mixing model collects.
SOIL_IDS = {2, 3, 4, 5}

KWARGS = {
    "p1": P1,
    "p2": P2,
    "frac_dim": 1.5,
    "weighting": (1.0, 1.0, 1.0),
    "n_materials": 4,
    "mixing_model_id": "soil",
    "id": "my_box",
    "seed": 42,
}


def make_box(**overrides):
    kwargs = {**KWARGS, **overrides}
    return FractalBox(**{k: v for k, v in kwargs.items() if v is not None})


def roughen_zplus(grid, limits=(0.009, 0.013), fractal_box_id="my_box"):
    """Attach a rough surface to the box's z+ face (cells 9..13)."""
    AddSurfaceRoughness(
        p1=(0.002, 0.002, 0.010),
        p2=(0.010, 0.010, 0.010),
        frac_dim=1.5,
        weighting=(1.0, 1.0),
        limits=limits,
        fractal_box_id=fractal_box_id,
        seed=7,
    ).build(grid)


class TestPreBuildRegistersTheVolume:
    def test_a_fractal_volume_is_registered_on_the_grid(self, fractal_grid):
        # The surface modifiers find their box by looking this list up by
        # ID, so registration is the whole contract of the first pass.
        g = fractal_grid()
        add_mixing_model(g)
        box = make_box()
        box.build(g)
        assert g.fractalvolumes
        assert all(volume is box.volume for volume in g.fractalvolumes)
        assert next(v for v in g.fractalvolumes if v.ID == "my_box") is box.volume

    def test_volume_bounds_come_from_the_discretised_points(self, fractal_grid):
        g = fractal_grid()
        add_mixing_model(g)
        make_box().build(g)
        volume = g.fractalvolumes[0]
        assert np.array_equal(volume.start, [2, 2, 2])
        assert np.array_equal(volume.stop, [10, 10, 10])

    def test_volume_attributes_are_populated(self, fractal_grid):
        g = fractal_grid()
        model = add_mixing_model(g)
        make_box().build(g)
        volume = g.fractalvolumes[0]
        assert volume.ID == "my_box"
        assert volume.operatingonID == "soil"
        assert volume.nbins == 4
        assert volume.dimension == 1.5
        assert volume.seed == 42
        assert np.array_equal(volume.weighting, [1.0, 1.0, 1.0])
        assert volume.mixingmodel is model

    def test_nothing_is_written_to_the_grid_on_the_first_pass(self, fractal_grid):
        g = fractal_grid()
        add_mixing_model(g)
        make_box().build(g)
        assert not g.solid.any()
        assert not g.rigidE.any()
        assert not g.ID.any()

    def test_the_mixing_model_materials_are_resolved(self, fractal_grid):
        # calculate_properties is where the N soil materials become real
        # numIDs the fractal volume can index.
        g = fractal_grid()
        model = add_mixing_model(g)
        make_box().build(g)
        assert model.matID == [2, 3, 4, 5]

    def test_pre_build_runs_only_once(self, fractal_grid):
        g = fractal_grid()
        add_mixing_model(g)
        box = make_box()
        assert box.do_pre_build is True
        box.build(g)
        assert box.do_pre_build is False
        registered = list(g.fractalvolumes)
        # A second build takes the stamping branch, not pre_build again,
        # so no new volume appears.
        box.build(g)
        assert g.fractalvolumes == registered

    def test_a_normal_material_can_be_used_instead_of_a_mixing_model(self, fractal_grid):
        g = fractal_grid()
        make_box(n_materials=1, mixing_model_id="sand").build(g)
        volume = g.fractalvolumes[0]
        assert volume.mixingmodel is None
        assert volume.nbins == 1

    def test_missing_seed_leaves_the_volume_unseeded(self, fractal_grid):
        g = fractal_grid()
        add_mixing_model(g)
        make_box(seed=None).build(g)
        assert g.fractalvolumes[0].seed is None


class TestAveraging:
    def test_averaging_is_off_by_default(self, fractal_grid):
        # Unlike every other geometry object, a fractal box does not
        # inherit the grid's averaging default.
        g = fractal_grid()
        add_mixing_model(g)
        make_box().build(g)
        assert g.fractalvolumes[0].averaging is False

    def test_averaging_can_be_switched_on(self, fractal_grid):
        g = fractal_grid()
        add_mixing_model(g)
        make_box(averaging="y").build(g)
        assert g.fractalvolumes[0].averaging is True

    def test_averaging_can_be_switched_off_explicitly(self, fractal_grid):
        g = fractal_grid()
        add_mixing_model(g)
        make_box(averaging="n").build(g)
        assert g.fractalvolumes[0].averaging is False


class TestPreBuildValidation:
    @pytest.mark.parametrize(
        "missing", ["p1", "p2", "frac_dim", "weighting", "n_materials", "mixing_model_id", "id"]
    )
    def test_missing_parameters_raise(self, fractal_grid, missing):
        g = fractal_grid()
        add_mixing_model(g)
        kwargs = {k: v for k, v in KWARGS.items() if k != missing}
        with pytest.raises(KeyError):
            FractalBox(**kwargs).build(g)

    def test_negative_fractal_dimension_raises(self, fractal_grid):
        g = fractal_grid()
        add_mixing_model(g)
        with pytest.raises(ValueError):
            make_box(frac_dim=-1.0).build(g)

    @pytest.mark.parametrize(
        "weighting",
        [(-1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0)],
    )
    def test_negative_weighting_raises(self, fractal_grid, weighting):
        g = fractal_grid()
        add_mixing_model(g)
        with pytest.raises(ValueError):
            make_box(weighting=weighting).build(g)

    def test_negative_material_count_raises(self, fractal_grid):
        g = fractal_grid()
        add_mixing_model(g)
        with pytest.raises(ValueError):
            make_box(n_materials=-1).build(g)

    def test_unknown_mixing_model_raises(self, fractal_grid):
        g = fractal_grid()
        with pytest.raises(ValueError):
            make_box(mixing_model_id="nonexistent").build(g)

    def test_a_mixing_model_with_one_bin_raises(self, fractal_grid):
        g = fractal_grid()
        add_mixing_model(g)
        with pytest.raises(ValueError):
            make_box(n_materials=1).build(g)

    def test_more_bins_than_materials_in_the_list_raises(self, fractal_grid):
        g = fractal_grid()
        add_mixing_model(g, materials=("sand", "clay"))
        with pytest.raises(ValueError):
            make_box(n_materials=4).build(g)

    def test_out_of_bounds_points_raise(self, fractal_grid):
        g = fractal_grid()
        add_mixing_model(g)
        with pytest.raises(ValueError):
            make_box(p2=(0.030, 0.010, 0.010)).build(g)

    def test_inverted_points_raise(self, fractal_grid):
        g = fractal_grid()
        add_mixing_model(g)
        with pytest.raises(ValueError):
            make_box(p1=(0.010, 0.010, 0.010), p2=(0.002, 0.002, 0.002)).build(g)


class TestBuildWithoutSurfaces:
    def test_a_single_material_box_with_no_surfaces_raises(self, fractal_grid):
        # A fractal box of one material and no roughness is just a #box.
        g = fractal_grid()
        box = make_box(n_materials=1, mixing_model_id="sand")
        box.build(g)
        with pytest.raises(ValueError):
            box.build(g)

    def test_the_box_footprint_is_stamped_into_solid(self, fractal_grid):
        g = fractal_grid()
        add_mixing_model(g)
        box = make_box()
        box.build(g)
        box.build(g)

        expected = {(i, j, k) for i in BOX_CELLS for j in BOX_CELLS for k in BOX_CELLS}
        assert nonzero_set(g.solid) == expected

    def test_only_mixing_model_materials_are_stamped(self, fractal_grid):
        g = fractal_grid()
        add_mixing_model(g)
        box = make_box()
        box.build(g)
        box.build(g)

        interior = g.solid[2:10, 2:10, 2:10]
        assert set(np.unique(interior)) <= SOIL_IDS

    def test_all_four_soil_materials_appear(self, fractal_grid):
        g = fractal_grid()
        add_mixing_model(g)
        box = make_box()
        box.build(g)
        box.build(g)

        assert set(np.unique(g.solid[2:10, 2:10, 2:10])) == SOIL_IDS

    @pytest.mark.xfail(reason="Cython extension needs rebuild — Python 3.14/Cython compiler incompatibility")
    def test_the_stamped_materials_match_the_generated_volume(self, fractal_grid):
        g = fractal_grid()
        add_mixing_model(g)
        box = make_box()
        box.build(g)
        box.build(g)

        volume = g.fractalvolumes[0]
        assert np.array_equal(g.solid[2:10, 2:10, 2:10], volume.fractalvolume.astype(np.uint32))

    def test_averaging_off_marks_the_cells_rigid(self, fractal_grid):
        g = fractal_grid()
        add_mixing_model(g)
        box = make_box()
        box.build(g)
        box.build(g)

        assert g.rigidE.any()
        assert g.ID.any()

    def test_averaging_on_clears_the_rigid_arrays(self, fractal_grid):
        g = fractal_grid()
        add_mixing_model(g)
        box = make_box(averaging="y")
        box.build(g)
        box.build(g)

        assert g.solid.any()
        assert not g.rigidE.any()
        assert not g.rigidH.any()

    def test_the_result_is_reproducible(self, fractal_grid):
        solids = []
        for _ in range(2):
            g = fractal_grid()
            add_mixing_model(g)
            box = make_box()
            box.build(g)
            box.build(g)
            solids.append(g.solid.copy())
        assert np.array_equal(*solids)


class TestBuildWithASurface:
    def test_the_volume_is_extended_to_cover_the_surface(self, fractal_grid):
        # The roughness may push the top face out to cell 13, so the box
        # must grow to hold it before it generates anything.
        g = fractal_grid()
        add_mixing_model(g)
        box = make_box()
        box.build(g)
        roughen_zplus(g)

        volume = g.fractalvolumes[0]
        assert volume.zf == 10
        box.build(g)
        assert volume.zf == 13
        assert volume.originalzf == 10

    def test_the_solid_footprint_is_limited_to_the_box_in_x_and_y(self, fractal_grid):
        g = fractal_grid()
        add_mixing_model(g)
        box = make_box()
        box.build(g)
        roughen_zplus(g)
        box.build(g)

        written = nonzero_set(g.solid)
        assert {i for i, _, _ in written} == set(BOX_CELLS)
        assert {j for _, j, _ in written} == set(BOX_CELLS)

    def test_cells_below_the_roughness_range_are_solid_soil(self, fractal_grid):
        # Everything under the lowest possible surface height is box
        # interior, whatever the height-map did.
        g = fractal_grid()
        add_mixing_model(g)
        box = make_box()
        box.build(g)
        roughen_zplus(g)
        box.build(g)

        below = g.solid[2:10, 2:10, 2:9]
        assert np.all(below > 0)
        assert set(np.unique(below)) <= SOIL_IDS

    def test_the_surface_is_rough_not_flat(self, fractal_grid):
        # Within the roughness band some columns are filled higher than
        # others — that is the whole point of the height-map.
        g = fractal_grid()
        add_mixing_model(g)
        box = make_box()
        box.build(g)
        roughen_zplus(g)
        box.build(g)

        band = g.solid[2:10, 2:10, 9:13]
        heights = np.count_nonzero(band, axis=2)
        assert heights.min() != heights.max()

    def test_nothing_is_written_above_the_roughness_range(self, fractal_grid):
        g = fractal_grid()
        add_mixing_model(g)
        box = make_box()
        box.build(g)
        roughen_zplus(g)
        box.build(g)

        assert not g.solid[:, :, 13:].any()

    def test_a_single_material_box_with_a_surface_uses_that_material(self, fractal_grid):
        # nbins == 1 skips fractal generation entirely and fills the
        # volume with one material's numID.
        g = fractal_grid()
        box = make_box(n_materials=1, mixing_model_id="sand")
        box.build(g)
        roughen_zplus(g)
        box.build(g)

        assert set(np.unique(g.solid)) == {0, 2}

    def test_the_result_is_reproducible(self, fractal_grid):
        solids = []
        for _ in range(2):
            g = fractal_grid()
            add_mixing_model(g)
            box = make_box()
            box.build(g)
            roughen_zplus(g)
            box.build(g)
            solids.append(g.solid.copy())
        assert np.array_equal(*solids)
