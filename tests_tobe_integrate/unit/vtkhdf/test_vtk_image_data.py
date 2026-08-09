"""``VtkImageData`` — the voxel writer behind every geometry view and snapshot.

An ImageData file describes a regular grid implicitly: an origin, a spacing
and an extent are enough to place every cell, so no coordinates are stored.
That makes it compact — a 200³ geometry view is one integer array rather than
eight million points — and it makes the four root attributes load-bearing. Get
``Spacing`` wrong and the model renders at the wrong physical size with no
error; get ``WholeExtent`` wrong and VTK reads past the data.

Three things here are specific to this class and covered nowhere else:

**Shape padding.** VTKHDF ImageData is always three-dimensional. A 1-D or 2-D
shape is padded with ones, so a 2-D gprMax model writes a file one cell deep
rather than failing. The padding also decides what ``add_cell_data`` will
accept afterwards, since the stored ``self.shape`` is the padded one.

**The extent convention.** ``WholeExtent`` is six numbers — a
``[min, max]`` pair per axis — and the maxima are *cell* counts, so a shape of
``(2, 3, 4)`` gives ``[0, 2, 0, 3, 0, 4]``. Points are one more than cells in
each direction, which is why ``add_point_data`` demands ``shape + 1``.

**The dimension checks.** ``add_cell_data`` and ``add_point_data`` each raise
before writing anything if the array does not match, which is the difference
between a clear error and a file VTK cannot open.

PR 10's outputs suite already pins the *values* these attributes take for a
real geometry view. This file tests the writer's own contract: the defaults,
the validation, and the padding — none of which a round-trip through
``write_vtk`` can reach.
"""

import h5py
import numpy as np
import pytest

from gprMax.vtkhdf_filehandlers.vtk_image_data import VtkImageData

from .conftest import IMAGE_DATA_TYPE


class TestConstruction:
    """What a newly created file contains before anything is added."""

    def test_the_type_attribute_says_image_data(self, make_image_data, read_h5):
        with make_image_data() as handler:
            path = handler.filename

        attrs, _ = read_h5(path)

        assert attrs["VTKHDF/Type"] == IMAGE_DATA_TYPE

    def test_the_file_is_always_opened_for_writing(self, make_image_data):
        """The constructor hard-codes mode ``"w"`` — there is no read path.

        A ``VtkImageData`` always truncates; it is a writer, not a reader.
        """
        with make_image_data() as handler:
            assert handler.file_handler.mode == "r+"

    def test_the_shape_is_stored(self, make_image_data):
        with make_image_data(shape=(2, 3, 4)) as handler:
            assert list(handler.shape) == [2, 3, 4]

    def test_the_four_attributes_are_written(self, make_image_data, read_h5):
        with make_image_data() as handler:
            path = handler.filename

        attrs, _ = read_h5(path)

        assert sorted(attrs) == [
            "VTKHDF/Direction",
            "VTKHDF/Origin",
            "VTKHDF/Spacing",
            "VTKHDF/Type",
            "VTKHDF/Version",
            "VTKHDF/WholeExtent",
        ]

    def test_no_datasets_are_written(self, make_image_data, read_h5):
        """Geometry is implicit; only the attributes describe the grid."""
        with make_image_data() as handler:
            path = handler.filename

        _, datasets = read_h5(path)

        assert datasets == {}

    def test_the_dimension_count_is_three(self):
        assert VtkImageData.DIMENSIONS == 3


class TestWholeExtent:
    """Six numbers: a ``[min, max]`` pair per axis, in cells."""

    def test_it_is_derived_from_the_shape(self, make_image_data):
        with make_image_data(shape=(2, 3, 4)) as handler:
            assert list(handler.whole_extent) == [0, 2, 0, 3, 0, 4]

    def test_the_minima_are_zero(self, make_image_data):
        """gprMax models always start at the origin."""
        with make_image_data(shape=(5, 6, 7)) as handler:
            assert list(handler.whole_extent[::2]) == [0, 0, 0]

    def test_the_maxima_are_the_cell_counts(self, make_image_data):
        with make_image_data(shape=(5, 6, 7)) as handler:
            assert list(handler.whole_extent[1::2]) == [5, 6, 7]

    def test_it_is_integral(self, make_image_data):
        """A float extent makes VTK compute fractional indices."""
        with make_image_data() as handler:
            assert np.issubdtype(handler.whole_extent.dtype, np.integer)

    def test_it_persists_to_disk(self, make_image_data, read_h5):
        with make_image_data(shape=(2, 3, 4)) as handler:
            path = handler.filename

        attrs, _ = read_h5(path)

        assert list(attrs["VTKHDF/WholeExtent"]) == [0, 2, 0, 3, 0, 4]


class TestShapePadding:
    """1-D and 2-D models become 3-D files, padded with ones."""

    def test_a_two_dimensional_shape_gains_a_third(self, make_image_data):
        """A 2-D gprMax model is one cell deep, not an error."""
        with make_image_data(shape=(2, 3)) as handler:
            assert list(handler.shape) == [2, 3, 1]

    def test_a_one_dimensional_shape_gains_two(self, make_image_data):
        with make_image_data(shape=(5,)) as handler:
            assert list(handler.shape) == [5, 1, 1]

    def test_the_padded_extent_is_still_six_numbers(self, make_image_data):
        with make_image_data(shape=(2, 3)) as handler:
            assert list(handler.whole_extent) == [0, 2, 0, 3, 0, 1]

    def test_an_empty_shape_raises(self, make_image_data):
        with pytest.raises(ValueError, match="must not be empty"):
            make_image_data(shape=())

    def test_a_four_dimensional_shape_raises(self, make_image_data):
        """There is no fourth spatial axis to pad away."""
        with pytest.raises(ValueError, match="more than 3 dimensions"):
            make_image_data(shape=(1, 2, 3, 4))

    def test_the_padded_shape_is_what_cell_data_must_match(
        self, make_image_data
    ):
        """The padding is not cosmetic: it changes the validation downstream."""
        with make_image_data(shape=(2, 3)) as handler:
            with pytest.raises(ValueError):
                handler.add_cell_data("Material", np.zeros((2, 3)))


class TestOrigin:
    """Where the grid's first corner sits in physical space."""

    def test_it_defaults_to_the_coordinate_origin(self, make_image_data):
        with make_image_data() as handler:
            assert list(handler.origin) == [0.0, 0.0, 0.0]

    def test_a_supplied_origin_is_used(self, make_image_data):
        origin = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        with make_image_data(origin=origin) as handler:
            assert list(handler.origin) == [1.0, 2.0, 3.0]

    def test_it_can_be_changed_afterwards(self, make_image_data):
        """Snapshots of a subgrid share a writer but not an origin."""
        with make_image_data() as handler:
            handler.set_origin(np.array([9.0, 9.0, 9.0], dtype=np.float32))

            assert list(handler.origin) == [9.0, 9.0, 9.0]

    def test_a_two_element_origin_raises(self, make_image_data):
        with make_image_data() as handler:
            with pytest.raises(ValueError, match="must have 3 dimensions"):
                handler.set_origin(np.array([1.0, 2.0], dtype=np.float32))

    def test_a_four_element_origin_raises(self, make_image_data):
        with make_image_data() as handler:
            with pytest.raises(ValueError, match="must have 3 dimensions"):
                handler.set_origin(np.zeros(4, dtype=np.float32))

    def test_it_persists_to_disk(self, make_image_data, read_h5):
        with make_image_data(
            origin=np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ) as handler:
            path = handler.filename

        attrs, _ = read_h5(path)

        assert list(attrs["VTKHDF/Origin"]) == [1.0, 2.0, 3.0]


class TestSpacing:
    """Cell size — the attribute that sets the model's physical scale."""

    def test_it_defaults_to_unit_cells(self, make_image_data):
        with make_image_data() as handler:
            assert list(handler.spacing) == [1.0, 1.0, 1.0]

    def test_a_supplied_spacing_is_used(self, make_image_data):
        """gprMax passes the grid discretisation here — usually millimetres."""
        spacing = np.array([0.002, 0.002, 0.002], dtype=np.float32)

        with make_image_data(spacing=spacing) as handler:
            assert handler.spacing == pytest.approx([0.002, 0.002, 0.002])

    def test_anisotropic_spacing_is_kept_per_axis(self, make_image_data):
        spacing = np.array([0.001, 0.002, 0.004], dtype=np.float32)

        with make_image_data(spacing=spacing) as handler:
            assert handler.spacing == pytest.approx([0.001, 0.002, 0.004])

    def test_it_can_be_changed_afterwards(self, make_image_data):
        with make_image_data() as handler:
            handler.set_spacing(np.array([0.5, 0.5, 0.5], dtype=np.float32))

            assert handler.spacing == pytest.approx([0.5, 0.5, 0.5])

    def test_a_two_element_spacing_raises(self, make_image_data):
        with make_image_data() as handler:
            with pytest.raises(ValueError, match="must have 3 dimensions"):
                handler.set_spacing(np.array([1.0, 1.0], dtype=np.float32))

    def test_it_persists_to_disk(self, make_image_data, read_h5):
        with make_image_data(
            spacing=np.array([0.002, 0.002, 0.002], dtype=np.float32)
        ) as handler:
            path = handler.filename

        attrs, _ = read_h5(path)

        assert attrs["VTKHDF/Spacing"] == pytest.approx([0.002] * 3)


class TestDirection:
    """The nine-element basis; the identity unless a model is rotated."""

    def test_it_defaults_to_the_identity(self, make_image_data):
        with make_image_data() as handler:
            assert list(handler.direction) == [1, 0, 0, 0, 1, 0, 0, 0, 1]

    def test_a_flat_array_is_accepted(self, make_image_data):
        direction = np.array(
            [0, 1, 0, 1, 0, 0, 0, 0, 1], dtype=np.float32
        )

        with make_image_data(direction=direction) as handler:
            assert list(handler.direction) == [0, 1, 0, 1, 0, 0, 0, 0, 1]

    def test_a_nested_array_is_flattened(self, make_image_data):
        """The docstring promises the two forms are equivalent."""
        direction = np.array(
            [[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32
        )

        with make_image_data(direction=direction) as handler:
            assert list(handler.direction) == [0, 1, 0, 1, 0, 0, 0, 0, 1]

    def test_it_is_always_stored_flat(self, make_image_data, read_h5):
        with make_image_data() as handler:
            path = handler.filename

        attrs, _ = read_h5(path)

        assert attrs["VTKHDF/Direction"].shape == (9,)

    def test_it_can_be_changed_afterwards(self, make_image_data):
        with make_image_data() as handler:
            handler.set_direction(np.zeros((3, 3), dtype=np.float32))

            assert list(handler.direction) == [0] * 9

    def test_a_three_element_direction_raises(self, make_image_data):
        with make_image_data() as handler:
            with pytest.raises(ValueError, match="must contain 9 elements"):
                handler.set_direction(np.ones(3, dtype=np.float32))

    def test_a_four_by_four_direction_raises(self, make_image_data):
        with make_image_data() as handler:
            with pytest.raises(ValueError, match="must contain 9 elements"):
                handler.set_direction(np.ones((4, 4), dtype=np.float32))


class TestAddCellData:
    """One value per cell — materials, and every snapshot field."""

    def test_a_matching_array_is_written(self, make_image_data, read_h5):
        with make_image_data(shape=(2, 3, 4)) as handler:
            handler.add_cell_data("Material", np.ones((2, 3, 4)))
            path = handler.filename

        _, datasets = read_h5(path)

        assert "VTKHDF/CellData/Material" in datasets

    def test_it_is_stored_in_zyx_order(self, make_image_data, tmp_path):
        """The shape on disk is reversed, as for every spatial dataset."""
        with make_image_data(shape=(2, 3, 4)) as handler:
            handler.add_cell_data("Material", np.ones((2, 3, 4)))

        with h5py.File(tmp_path / "image.vtkhdf", "r") as f:
            assert f["VTKHDF/CellData/Material"].shape == (4, 3, 2)

    def test_the_values_survive_the_round_trip(self, make_image_data, tmp_path):
        data = np.arange(24).reshape((2, 3, 4))

        with make_image_data(shape=(2, 3, 4)) as handler:
            handler.add_cell_data("Material", data)

        with h5py.File(tmp_path / "image.vtkhdf", "r") as f:
            stored = f["VTKHDF/CellData/Material"][()]

        assert np.array_equal(stored.transpose(), data)

    def test_a_mismatched_shape_raises(self, make_image_data):
        with make_image_data(shape=(2, 3, 4)) as handler:
            with pytest.raises(ValueError, match="must match the dimensions"):
                handler.add_cell_data("Material", np.ones((2, 3, 5)))

    def test_the_error_shows_both_shapes(self, make_image_data):
        with make_image_data(shape=(2, 3, 4)) as handler:
            with pytest.raises(ValueError, match=r"\(2, 3, 5\)"):
                handler.add_cell_data("Material", np.ones((2, 3, 5)))

    def test_a_partial_write_bypasses_the_shape_check(self, make_image_data):
        """With an offset, the array is a slab and need not match."""
        with make_image_data(shape=(2, 3, 4)) as handler:
            handler.add_cell_data(
                "Material", np.ones((1, 3, 4)), offset=np.array([0, 0, 0])
            )

            assert handler._get_dataset("VTKHDF/CellData/Material").shape == (
                4,
                3,
                2,
            )

    def test_the_slab_lands_at_the_offset(self, make_image_data, tmp_path):
        """The MPI case: each rank writes its own x-slab."""
        with make_image_data(shape=(2, 3, 4)) as handler:
            handler.add_cell_data(
                "Material", np.full((1, 3, 4), 7.0), offset=np.array([1, 0, 0])
            )

        with h5py.File(tmp_path / "image.vtkhdf", "r") as f:
            stored = f["VTKHDF/CellData/Material"][()]

        assert np.all(stored[:, :, 1] == 7.0) and np.all(stored[:, :, 0] == 0.0)

    def test_several_cell_arrays_coexist(self, make_image_data, read_h5):
        """A geometry view holds materials; a snapshot holds six fields."""
        with make_image_data(shape=(2, 3, 4)) as handler:
            handler.add_cell_data("Material", np.zeros((2, 3, 4)))
            handler.add_cell_data("Sources", np.zeros((2, 3, 4)))
            path = handler.filename

        _, datasets = read_h5(path)

        assert sorted(datasets) == [
            "VTKHDF/CellData/Material",
            "VTKHDF/CellData/Sources",
        ]

    def test_the_dtype_is_preserved(self, make_image_data, read_h5):
        """Material IDs are integers; storing them as floats doubles the file."""
        with make_image_data(shape=(2, 3, 4)) as handler:
            handler.add_cell_data(
                "Material", np.zeros((2, 3, 4), dtype=np.int32)
            )
            path = handler.filename

        _, datasets = read_h5(path)

        assert datasets["VTKHDF/CellData/Material"].dtype == np.int32


class TestAddPointData:
    """One value per *point* — one more than the cells in each direction."""

    def test_an_array_one_larger_in_each_dimension_is_accepted(
        self, make_image_data, read_h5
    ):
        with make_image_data(shape=(2, 3, 4)) as handler:
            handler.add_point_data("Field", np.ones((3, 4, 5)))
            path = handler.filename

        _, datasets = read_h5(path)

        assert "VTKHDF/PointData/Field" in datasets

    def test_the_cell_shape_is_rejected(self, make_image_data):
        """The off-by-one that would otherwise be silent in a viewer."""
        with make_image_data(shape=(2, 3, 4)) as handler:
            with pytest.raises(ValueError, match="one larger in each dimension"):
                handler.add_point_data("Field", np.ones((2, 3, 4)))

    def test_the_error_shows_the_expected_shape(self, make_image_data):
        with make_image_data(shape=(2, 3, 4)) as handler:
            with pytest.raises(ValueError, match=r"\[3 4 5\]"):
                handler.add_point_data("Field", np.ones((2, 3, 4)))

    def test_it_is_stored_in_zyx_order(self, make_image_data, tmp_path):
        with make_image_data(shape=(2, 3, 4)) as handler:
            handler.add_point_data("Field", np.ones((3, 4, 5)))

        with h5py.File(tmp_path / "image.vtkhdf", "r") as f:
            assert f["VTKHDF/PointData/Field"].shape == (5, 4, 3)

    def test_a_partial_write_bypasses_the_shape_check(self, make_image_data):
        with make_image_data(shape=(2, 3, 4)) as handler:
            handler.add_point_data(
                "Field", np.ones((1, 4, 5)), offset=np.array([0, 0, 0])
            )

            assert handler._get_dataset("VTKHDF/PointData/Field").shape == (
                5,
                4,
                3,
            )

    def test_point_and_cell_data_coexist(self, make_image_data, read_h5):
        with make_image_data(shape=(2, 3, 4)) as handler:
            handler.add_cell_data("Material", np.zeros((2, 3, 4)))
            handler.add_point_data("Field", np.zeros((3, 4, 5)))
            path = handler.filename

        _, datasets = read_h5(path)

        assert sorted(datasets) == [
            "VTKHDF/CellData/Material",
            "VTKHDF/PointData/Field",
        ]
