"""``_write_dataset`` — everything that actually puts bytes in the file.

One method, about a hundred lines, and every dataset in every VTKHDF file
gprMax writes goes through it. It does four separable jobs, and the tests
below are grouped by them:

**Coercion.** Anything array-like is accepted — a list, a scalar, a NumPy
array — and a scalar is expanded to shape ``(1,)`` so HDF5 has something to
store. This is why ``NumberOfCells`` can be written as a plain ``int``.

**String conversion.** HDF5 has no UTF-32, so NumPy's ``'U'`` dtype cannot be
stored, and VTKHDF only reads *variable-length ASCII*. Any string data is
therefore rewritten to that one representation, and the two ways of asking for
something else — a fixed length, or UTF-8 — each produce a warning explaining
what happened instead. Three of the four warning paths only trigger when the
caller passes an explicit ``dtype``, which is why the tests do so.

**The transpose.** VTKHDF stores arrays in ZYX order; gprMax works in XYZ.
``_write_dataset`` therefore calls ``data.transpose()`` — which reverses *all*
axes — and flips ``shape`` and ``offset`` to match. This is the single most
consequential line in the package for anyone reading the output: get it wrong
and the model loads mirrored, with no error anywhere. ``xyz_data_ordering``
defaults to ``True``, and **no public method exposes it** — only
``add_field_data`` sets it to ``False``, internally. Its consequences for 1-D
and 2-D data are pinned here because they are not obvious: a 1-D transpose is
a no-op, and a 2-D one swaps rows and columns.

**Partial writes.** ``shape`` plus ``offset`` place a block inside a larger
dataset, which is how MPI ranks each write their own slab. Three separate
validation errors guard it, and each is asserted on its message, because the
message is the only thing a user has to work with when a rank's arithmetic is
wrong.

Everything is verified by reopening the file. Where a test asserts on layout,
it reads the raw HDF5 array — *not* transposed back — because the on-disk
order is exactly the thing being pinned.
"""

import h5py
import numpy as np
import pytest


class TestScalarsAndSequences:
    """Coercion of anything array-like into something HDF5 can store."""

    def test_a_list_is_accepted(self, make_vtkhdf_file, read_h5):
        with make_vtkhdf_file() as handler:
            handler._write_dataset("VTKHDF/Numbers", [1, 2, 3])
            path = handler.filename

        _, datasets = read_h5(path)

        assert list(datasets["VTKHDF/Numbers"]) == [1, 2, 3]

    def test_a_scalar_becomes_a_one_element_dataset(self, make_vtkhdf_file, read_h5):
        """``NumberOfCells`` is written this way — a bare Python ``int``."""
        with make_vtkhdf_file() as handler:
            handler._write_dataset("VTKHDF/Count", 7)
            path = handler.filename

        _, datasets = read_h5(path)

        assert datasets["VTKHDF/Count"].shape == (1,)

    def test_the_scalar_value_survives(self, make_vtkhdf_file, read_h5):
        with make_vtkhdf_file() as handler:
            handler._write_dataset("VTKHDF/Count", 7)
            path = handler.filename

        _, datasets = read_h5(path)

        assert datasets["VTKHDF/Count"][0] == 7

    def test_an_array_is_written_unchanged(self, make_vtkhdf_file, read_h5):
        with make_vtkhdf_file() as handler:
            handler._write_dataset("VTKHDF/Numbers", np.arange(5))
            path = handler.filename

        _, datasets = read_h5(path)

        assert list(datasets["VTKHDF/Numbers"]) == [0, 1, 2, 3, 4]

    def test_the_dtype_is_deduced_from_the_data(self, make_vtkhdf_file, read_h5):
        with make_vtkhdf_file() as handler:
            handler._write_dataset("VTKHDF/Numbers", np.arange(3, dtype=np.int32))
            path = handler.filename

        _, datasets = read_h5(path)

        assert datasets["VTKHDF/Numbers"].dtype == np.int32

    def test_an_explicit_dtype_overrides_the_data(self, make_vtkhdf_file, read_h5):
        with make_vtkhdf_file() as handler:
            handler._write_dataset("VTKHDF/Numbers", [1, 2, 3], dtype=np.float64)
            path = handler.filename

        _, datasets = read_h5(path)

        assert datasets["VTKHDF/Numbers"].dtype == np.float64

    def test_floating_point_data_keeps_its_precision(self, make_vtkhdf_file, read_h5):
        """Snapshots are ``float32``; promoting them would double file size."""
        with make_vtkhdf_file() as handler:
            handler._write_dataset("VTKHDF/Field", np.ones(4, dtype=np.float32))
            path = handler.filename

        _, datasets = read_h5(path)

        assert datasets["VTKHDF/Field"].dtype == np.float32

    def test_an_empty_array_is_written(self, make_vtkhdf_file, read_h5):
        """A model with no cells of a given type is legitimate."""
        with make_vtkhdf_file() as handler:
            handler._write_dataset("VTKHDF/Nothing", np.array([]))
            path = handler.filename

        _, datasets = read_h5(path)

        assert datasets["VTKHDF/Nothing"].shape == (0,)

    def test_a_duplicate_dataset_raises(self, make_vtkhdf_file):
        """``create_dataset`` will not overwrite; the second write fails.

        Relevant to any caller that names two datasets alike — geometry views
        and snapshots share a file in some configurations.
        """
        with make_vtkhdf_file() as handler:
            handler._write_dataset("VTKHDF/Numbers", [1])

            with pytest.raises(ValueError):
                handler._write_dataset("VTKHDF/Numbers", [2])

    def test_a_nested_path_creates_intermediate_groups(self, make_vtkhdf_file, read_h5):
        """``CellData/Material`` needs a ``CellData`` group that nothing made."""
        with make_vtkhdf_file() as handler:
            handler._write_dataset("VTKHDF/CellData/Material", [1, 2])
            path = handler.filename

        _, datasets = read_h5(path)

        assert "VTKHDF/CellData/Material" in datasets


class TestTheTranspose:
    """ZYX on disk, XYZ in memory — reversal of *all* axes."""

    def test_a_three_dimensional_array_is_reversed(self, make_vtkhdf_file, tmp_path):
        """The shape on disk is the reverse of the shape in memory.

        Asserted on the raw HDF5 dataset, because that is what VTK reads.
        """
        with make_vtkhdf_file() as handler:
            handler._write_dataset("VTKHDF/Field", np.zeros((2, 3, 4)))

        with h5py.File(tmp_path / "model.vtkhdf", "r") as f:
            assert f["VTKHDF/Field"].shape == (4, 3, 2)

    def test_the_values_land_where_the_reversal_says(self, make_vtkhdf_file, tmp_path):
        """Shape alone would pass for a wrong permutation; values pin it."""
        data = np.arange(24).reshape((2, 3, 4))

        with make_vtkhdf_file() as handler:
            handler._write_dataset("VTKHDF/Field", data)

        with h5py.File(tmp_path / "model.vtkhdf", "r") as f:
            stored = f["VTKHDF/Field"][()]

        assert stored[3, 2, 1] == data[1, 2, 3]

    def test_the_round_trip_recovers_the_original(self, make_vtkhdf_file, tmp_path):
        """A reader that transposes back gets exactly what was written."""
        data = np.arange(24).reshape((2, 3, 4))

        with make_vtkhdf_file() as handler:
            handler._write_dataset("VTKHDF/Field", data)

        with h5py.File(tmp_path / "model.vtkhdf", "r") as f:
            stored = f["VTKHDF/Field"][()]

        assert np.array_equal(stored.transpose(), data)

    def test_a_one_dimensional_array_is_unaffected(self, make_vtkhdf_file, read_h5):
        """Transposing a vector is a no-op, so scalars and IDs are safe."""
        with make_vtkhdf_file() as handler:
            handler._write_dataset("VTKHDF/Ids", np.arange(5))
            path = handler.filename

        _, datasets = read_h5(path)

        assert list(datasets["VTKHDF/Ids"]) == [0, 1, 2, 3, 4]

    def test_a_two_dimensional_array_is_swapped(self, make_vtkhdf_file, tmp_path):
        """An ``(N, 3)`` array of vectors becomes ``(3, N)`` on disk.

        VTK expects ``(nTuples, nComponents)``, so any caller writing vector
        data through the default path gets it stored the wrong way round.
        This is why ``Points`` is written with ``xyz_data_ordering=False``.
        """
        with make_vtkhdf_file() as handler:
            handler._write_dataset("VTKHDF/Vectors", np.zeros((5, 3)))

        with h5py.File(tmp_path / "model.vtkhdf", "r") as f:
            assert f["VTKHDF/Vectors"].shape == (3, 5)

    def test_it_can_be_switched_off(self, make_vtkhdf_file, tmp_path):
        """The escape hatch ``Points`` and field data use."""
        with make_vtkhdf_file() as handler:
            handler._write_dataset("VTKHDF/Vectors", np.zeros((5, 3)), xyz_data_ordering=False)

        with h5py.File(tmp_path / "model.vtkhdf", "r") as f:
            assert f["VTKHDF/Vectors"].shape == (5, 3)

    def test_it_is_on_by_default(self, make_vtkhdf_file, tmp_path):
        """Stated explicitly: a caller who says nothing gets the transpose."""
        with make_vtkhdf_file() as handler:
            handler._write_dataset("VTKHDF/Field", np.zeros((2, 4)))

        with h5py.File(tmp_path / "model.vtkhdf", "r") as f:
            assert f["VTKHDF/Field"].shape == (4, 2)

    def test_no_public_method_exposes_the_switch(self):
        """The transpose cannot be turned off from outside the package.

        ``add_point_data``, ``add_cell_data`` and the ``VtkImageData`` /
        ``VtkUnstructuredGrid`` constructors take no such argument, so every
        externally written dataset is transposed. Worth pinning: it means the
        2-D swap above is not an edge case a caller can avoid.
        """
        import inspect

        from gprMax.vtkhdf_filehandlers.vtk_image_data import VtkImageData
        from gprMax.vtkhdf_filehandlers.vtk_unstructured_grid import VtkUnstructuredGrid

        public = [
            VtkImageData.add_point_data,
            VtkImageData.add_cell_data,
            VtkUnstructuredGrid.add_point_data,
            VtkUnstructuredGrid.add_cell_data,
        ]

        assert not any(
            "xyz_data_ordering" in inspect.signature(method).parameters for method in public
        )


class TestStringData:
    """VTKHDF reads variable-length ASCII, and nothing else."""

    def test_a_list_of_strings_is_written(self, make_vtkhdf_file, read_h5):
        with make_vtkhdf_file() as handler:
            handler.add_field_data("Names", ["alpha", "beta"])
            path = handler.filename

        _, datasets = read_h5(path)

        assert [value.decode() for value in datasets["VTKHDF/FieldData/Names"]] == [
            "alpha",
            "beta",
        ]

    def test_strings_are_stored_as_variable_length(self, make_vtkhdf_file, tmp_path):
        """Fixed-length padding would be read by VTK as trailing nulls."""
        with make_vtkhdf_file() as handler:
            handler.add_field_data("Names", ["a", "much longer name"])

        with h5py.File(tmp_path / "model.vtkhdf", "r") as f:
            info = h5py.check_string_dtype(f["VTKHDF/FieldData/Names"].dtype)

        assert info.length is None

    def test_strings_are_stored_as_ascii(self, make_vtkhdf_file, tmp_path):
        with make_vtkhdf_file() as handler:
            handler.add_field_data("Names", ["alpha"])

        with h5py.File(tmp_path / "model.vtkhdf", "r") as f:
            info = h5py.check_string_dtype(f["VTKHDF/FieldData/Names"].dtype)

        assert info.encoding == "ascii"

    def test_a_single_string_becomes_a_one_element_dataset(self, make_vtkhdf_file, read_h5):
        with make_vtkhdf_file() as handler:
            handler.add_field_data("Title", "a model")
            path = handler.filename

        _, datasets = read_h5(path)

        assert datasets["VTKHDF/FieldData/Title"][0].decode() == "a model"

    def test_an_explicit_unicode_dtype_warns(self, make_vtkhdf_file, caplog):
        """HDF5 has no UTF-32; the conversion is silent otherwise."""
        with make_vtkhdf_file() as handler:
            handler.add_field_data("Names", ["alpha"], dtype="U5")

        assert "UTF-32" in caplog.text

    def test_an_explicit_unicode_dtype_still_writes_ascii(self, make_vtkhdf_file, tmp_path):
        with make_vtkhdf_file() as handler:
            handler.add_field_data("Names", ["alpha"], dtype="U5")

        with h5py.File(tmp_path / "model.vtkhdf", "r") as f:
            info = h5py.check_string_dtype(f["VTKHDF/FieldData/Names"].dtype)

        assert info.encoding == "ascii"

    def test_a_utf8_string_dtype_warns(self, make_vtkhdf_file, caplog):
        with make_vtkhdf_file() as handler:
            handler.add_field_data("Names", ["alpha"], dtype=h5py.string_dtype("utf-8"))

        assert "utf-8 encoding is not supported" in caplog.text

    def test_a_fixed_length_string_dtype_warns(self, make_vtkhdf_file, caplog):
        """Serial I/O converts to variable length and says so."""
        with make_vtkhdf_file() as handler:
            handler.add_field_data("Names", ["alpha"], dtype=h5py.string_dtype("ascii", 10))

        assert "Fixed length strings are not supported" in caplog.text

    def test_a_fixed_length_string_dtype_is_converted(self, make_vtkhdf_file, tmp_path):
        with make_vtkhdf_file() as handler:
            handler.add_field_data("Names", ["alpha"], dtype=h5py.string_dtype("ascii", 10))

        with h5py.File(tmp_path / "model.vtkhdf", "r") as f:
            info = h5py.check_string_dtype(f["VTKHDF/FieldData/Names"].dtype)

        assert info.length is None

    def test_a_bytes_array_is_also_converted(self, make_vtkhdf_file, tmp_path):
        """``'S'`` dtype takes the same path as ``'U'``."""
        with make_vtkhdf_file() as handler:
            handler.add_field_data("Names", np.array([b"alpha", b"beta"]))

        with h5py.File(tmp_path / "model.vtkhdf", "r") as f:
            info = h5py.check_string_dtype(f["VTKHDF/FieldData/Names"].dtype)

        assert info is not None and info.length is None

    def test_a_plain_string_array_does_not_warn(self, make_vtkhdf_file, caplog):
        """The warnings only fire for an explicitly requested dtype.

        Passing a ``'U'`` array without naming a dtype converts silently —
        the common case, and arguably the one most worth reporting.
        """
        with make_vtkhdf_file() as handler:
            handler.add_field_data("Names", np.array(["alpha", "beta"]))

        assert "UTF-32" not in caplog.text


class TestPartialWrites:
    """``shape`` plus ``offset`` — one rank's slab inside the whole dataset."""

    def test_the_full_dataset_is_created(self, make_vtkhdf_file, read_h5):
        with make_vtkhdf_file() as handler:
            handler._write_dataset(
                "VTKHDF/Numbers",
                np.array([1, 2]),
                shape=np.array([5]),
                offset=np.array([0]),
            )
            path = handler.filename

        _, datasets = read_h5(path)

        assert datasets["VTKHDF/Numbers"].shape == (5,)

    def test_the_data_lands_at_the_offset(self, make_vtkhdf_file, read_h5):
        with make_vtkhdf_file() as handler:
            handler._write_dataset(
                "VTKHDF/Numbers",
                np.array([7, 8]),
                shape=np.array([5]),
                offset=np.array([2]),
            )
            path = handler.filename

        _, datasets = read_h5(path)

        assert list(datasets["VTKHDF/Numbers"]) == [0, 0, 7, 8, 0]

    def test_the_rest_of_the_dataset_is_zero(self, make_vtkhdf_file, read_h5):
        """Ranks that never write leave zeros, not uninitialised memory."""
        with make_vtkhdf_file() as handler:
            handler._write_dataset(
                "VTKHDF/Numbers",
                np.array([7]),
                shape=np.array([4]),
                offset=np.array([0]),
            )
            path = handler.filename

        _, datasets = read_h5(path)

        assert list(datasets["VTKHDF/Numbers"][1:]) == [0, 0, 0]

    def test_a_shape_equal_to_the_data_needs_no_offset(self, make_vtkhdf_file, read_h5):
        """The single-rank case: shape is given but describes the whole thing."""
        with make_vtkhdf_file() as handler:
            handler._write_dataset("VTKHDF/Numbers", np.array([1, 2, 3]), shape=np.array([3]))
            path = handler.filename

        _, datasets = read_h5(path)

        assert list(datasets["VTKHDF/Numbers"]) == [1, 2, 3]

    def test_a_larger_shape_without_an_offset_raises(self, make_vtkhdf_file):
        with make_vtkhdf_file() as handler:
            with pytest.raises(ValueError, match="Offset must not be None"):
                handler._write_dataset("VTKHDF/Numbers", np.array([1, 2]), shape=np.array([5]))

    def test_a_shape_of_the_wrong_rank_raises(self, make_vtkhdf_file):
        with make_vtkhdf_file() as handler:
            with pytest.raises(ValueError, match="same number of dimensions"):
                handler._write_dataset(
                    "VTKHDF/Numbers",
                    np.array([1, 2]),
                    shape=np.array([5, 5]),
                    offset=np.array([0, 0]),
                )

    def test_an_offset_of_the_wrong_rank_raises(self, make_vtkhdf_file):
        with make_vtkhdf_file() as handler:
            with pytest.raises(ValueError, match="offset must have the same"):
                handler._write_dataset(
                    "VTKHDF/Numbers",
                    np.array([1, 2]),
                    shape=np.array([5]),
                    offset=np.array([0, 0]),
                )

    def test_data_that_overruns_the_dataset_raises(self, make_vtkhdf_file):
        """The MPI failure mode: a rank computed its offset wrongly."""
        with make_vtkhdf_file() as handler:
            with pytest.raises(ValueError, match="does not fit within the bounds"):
                handler._write_dataset(
                    "VTKHDF/Numbers",
                    np.array([1, 2]),
                    shape=np.array([3]),
                    offset=np.array([2]),
                )

    def test_the_overrun_message_shows_the_arithmetic(self, make_vtkhdf_file):
        """``[2] + (2,) = [4] > [3]`` — enough to debug without a rerun."""
        with make_vtkhdf_file() as handler:
            with pytest.raises(ValueError, match=r"\[4\] > \[3\]"):
                handler._write_dataset(
                    "VTKHDF/Numbers",
                    np.array([1, 2]),
                    shape=np.array([3]),
                    offset=np.array([2]),
                )

    def test_a_two_dimensional_partial_write_places_correctly(self, make_vtkhdf_file, tmp_path):
        """Both ``shape`` and ``offset`` are flipped alongside the data."""
        with make_vtkhdf_file() as handler:
            handler._write_dataset(
                "VTKHDF/Block",
                np.ones((1, 2)),
                shape=np.array([3, 2]),
                offset=np.array([1, 0]),
            )

        with h5py.File(tmp_path / "model.vtkhdf", "r") as f:
            stored = f["VTKHDF/Block"][()]

        assert stored.shape == (2, 3) and np.array_equal(stored[:, 1], np.ones(2))

    def test_the_offset_is_flipped_with_the_data(self, make_vtkhdf_file, tmp_path):
        """Stated as its own test: the flip must apply to all three, or the
        block lands in the wrong corner of a correctly shaped dataset.
        """
        with make_vtkhdf_file() as handler:
            handler._write_dataset(
                "VTKHDF/Block",
                np.array([[9.0]]),
                shape=np.array([2, 3]),
                offset=np.array([0, 2]),
            )

        with h5py.File(tmp_path / "model.vtkhdf", "r") as f:
            stored = f["VTKHDF/Block"][()]

        assert stored[2, 0] == 9.0

    def test_a_partial_write_without_the_transpose_uses_the_given_order(
        self, make_vtkhdf_file, tmp_path
    ):
        with make_vtkhdf_file() as handler:
            handler._write_dataset(
                "VTKHDF/Block",
                np.array([[9.0]]),
                shape=np.array([2, 3]),
                offset=np.array([0, 2]),
                xyz_data_ordering=False,
            )

        with h5py.File(tmp_path / "model.vtkhdf", "r") as f:
            stored = f["VTKHDF/Block"][()]

        assert stored.shape == (2, 3) and stored[0, 2] == 9.0


class TestCreateDataset:
    """``_create_dataset`` — reserve space without writing anything.

    Used when some MPI ranks have no data but must still take part in the
    collective creation call.
    """

    def test_a_dataset_of_the_requested_shape_is_created(self, make_vtkhdf_file, read_h5):
        with make_vtkhdf_file() as handler:
            handler.add_field_data("Empty", None, shape=(4,), dtype=np.int32)
            path = handler.filename

        _, datasets = read_h5(path)

        assert datasets["VTKHDF/FieldData/Empty"].shape == (4,)

    def test_it_is_filled_with_zeros(self, make_vtkhdf_file, read_h5):
        with make_vtkhdf_file() as handler:
            handler.add_field_data("Empty", None, shape=(3,), dtype=np.int32)
            path = handler.filename

        _, datasets = read_h5(path)

        assert list(datasets["VTKHDF/FieldData/Empty"]) == [0, 0, 0]

    def test_the_requested_dtype_is_used(self, make_vtkhdf_file, read_h5):
        with make_vtkhdf_file() as handler:
            handler.add_field_data("Empty", None, shape=(2,), dtype=np.float64)
            path = handler.filename

        _, datasets = read_h5(path)

        assert datasets["VTKHDF/FieldData/Empty"].dtype == np.float64

    def test_a_unicode_dtype_warns(self, make_vtkhdf_file, caplog):
        with make_vtkhdf_file() as handler:
            handler.add_field_data("Empty", None, shape=(2,), dtype="U5")

        assert "UTF-32" in caplog.text

    def test_a_unicode_dtype_becomes_variable_length_ascii(self, make_vtkhdf_file, tmp_path):
        with make_vtkhdf_file() as handler:
            handler.add_field_data("Empty", None, shape=(2,), dtype="U5")

        with h5py.File(tmp_path / "model.vtkhdf", "r") as f:
            info = h5py.check_string_dtype(f["VTKHDF/FieldData/Empty"].dtype)

        assert info.encoding == "ascii" and info.length is None

    def test_a_utf8_dtype_warns(self, make_vtkhdf_file, caplog):
        with make_vtkhdf_file() as handler:
            handler.add_field_data("Empty", None, shape=(2,), dtype=h5py.string_dtype("utf-8"))

        assert "utf-8 encoding is not supported" in caplog.text

    def test_no_data_and_no_shape_raises(self, make_vtkhdf_file):
        with make_vtkhdf_file() as handler:
            with pytest.raises(ValueError, match="shape and dtype must be provided"):
                handler.add_field_data("Empty", None)

    def test_no_data_and_no_dtype_raises(self, make_vtkhdf_file):
        with make_vtkhdf_file() as handler:
            with pytest.raises(ValueError, match="shape and dtype must be provided"):
                handler.add_field_data("Empty", None, shape=(2,))


class TestFieldData:
    """``add_field_data`` — the only public writer on the base class."""

    def test_it_writes_under_the_field_data_group(self, make_vtkhdf_file, read_h5):
        with make_vtkhdf_file() as handler:
            handler.add_field_data("Title", [1])
            path = handler.filename

        _, datasets = read_h5(path)

        assert "VTKHDF/FieldData/Title" in datasets

    def test_it_does_not_transpose(self, make_vtkhdf_file, tmp_path):
        """Field data is metadata, not a spatial array — ZYX is meaningless.

        The one place ``xyz_data_ordering=False`` is passed from inside the
        package.
        """
        with make_vtkhdf_file() as handler:
            handler.add_field_data("Table", np.zeros((2, 5)))

        with h5py.File(tmp_path / "model.vtkhdf", "r") as f:
            assert f["VTKHDF/FieldData/Table"].shape == (2, 5)

    def test_a_scalar_is_accepted(self, make_vtkhdf_file, read_h5):
        with make_vtkhdf_file() as handler:
            handler.add_field_data("Time", 1.5)
            path = handler.filename

        _, datasets = read_h5(path)

        assert datasets["VTKHDF/FieldData/Time"][0] == 1.5

    def test_an_explicit_dtype_is_honoured(self, make_vtkhdf_file, read_h5):
        with make_vtkhdf_file() as handler:
            handler.add_field_data("Steps", [1, 2], dtype=np.int64)
            path = handler.filename

        _, datasets = read_h5(path)

        assert datasets["VTKHDF/FieldData/Steps"].dtype == np.int64

    def test_several_fields_coexist(self, make_vtkhdf_file, read_h5):
        with make_vtkhdf_file() as handler:
            handler.add_field_data("Time", 1.5)
            handler.add_field_data("Iteration", 42)
            path = handler.filename

        _, datasets = read_h5(path)

        assert sorted(datasets) == [
            "VTKHDF/FieldData/Iteration",
            "VTKHDF/FieldData/Time",
        ]

    def test_a_partial_field_write_is_supported(self, make_vtkhdf_file, read_h5):
        with make_vtkhdf_file() as handler:
            handler.add_field_data(
                "Steps", np.array([9]), shape=np.array([3]), offset=np.array([1])
            )
            path = handler.filename

        _, datasets = read_h5(path)

        assert list(datasets["VTKHDF/FieldData/Steps"]) == [0, 9, 0]


pytestmark = pytest.mark.unit
