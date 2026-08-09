"""``VtkHdfFile`` — opening, naming, marking and closing a VTKHDF file.

Before any data is written, four things have to be right or VTK will refuse
the file, or worse, misread it:

* the **file name** must end in ``.vtkhdf``;
* there must be a group called ``VTKHDF`` at the root;
* that group must carry a ``Version`` attribute VTK recognises;
* and a ``Type`` attribute naming the dataset class, stored as *fixed-length
  ASCII* — not a Python string, not UTF-8.

The last of those is the subtle one. ``h5py`` will happily write a variable
length UTF-8 string for a Python ``str``, and the file will open in ``h5py``
and look correct in an HDF5 viewer, while VTK reads nothing. The constructor
therefore encodes the type and hands ``h5py`` an explicit fixed-length ASCII
dtype. That is asserted here by dtype, not by value.

The extension rewriting deserves its own attention. ``Path.with_suffix``
replaces everything after the *last* dot, so a name that contains a version
number or a dimension loses part of itself — ``model_1.5`` becomes
``model_1.vtkhdf``. The warning fires for the extension it replaced, not for
the characters it dropped. Both are pinned below.
"""

import h5py
import numpy as np
import pytest

from gprMax.vtkhdf_filehandlers.vtkhdf import VtkFileType, VtkHdfFile

from .conftest import IMAGE_DATA_TYPE, VERSION


class TestClassConstants:
    """The format constants, which are also the compatibility contract."""

    def test_the_version_is_two_two(self):
        """VTKHDF 2.2 — the version whose layout this writer implements."""
        assert VtkHdfFile.VERSION == VERSION

    def test_the_extension_is_vtkhdf(self):
        assert VtkHdfFile.FILE_EXTENSION == ".vtkhdf"

    def test_the_root_group_is_named_vtkhdf(self):
        """VTK looks for this exact group; nothing else in the file matters."""
        assert VtkHdfFile.ROOT_GROUP == "VTKHDF"

    def test_the_two_required_attributes_are_named(self):
        assert (VtkHdfFile.VERSION_ATTR, VtkHdfFile.TYPE_ATTR) == (
            "Version",
            "Type",
        )

    def test_the_file_type_enum_has_two_members(self):
        """ImageData for voxels, UnstructuredGrid for lines."""
        assert [member.value for member in VtkFileType] == [
            "ImageData",
            "UnstructuredGrid",
        ]

    def test_the_file_type_enum_members_are_strings(self):
        """``VtkFileType`` subclasses ``str``, so ``.encode`` works directly.

        The constructor relies on that: ``vtk_file_type.encode("ascii")``.
        """
        assert isinstance(VtkFileType.IMAGE_DATA, str)


class TestFileNaming:
    """The extension is rewritten, not merely checked."""

    def test_a_correct_extension_is_kept(self, make_vtkhdf_file):
        with make_vtkhdf_file("model.vtkhdf") as handler:
            assert handler.filename.name == "model.vtkhdf"

    def test_a_wrong_extension_is_replaced(self, make_vtkhdf_file):
        with make_vtkhdf_file("model.h5") as handler:
            assert handler.filename.name == "model.vtkhdf"

    def test_a_missing_extension_is_added(self, make_vtkhdf_file):
        with make_vtkhdf_file("model") as handler:
            assert handler.filename.name == "model.vtkhdf"

    def test_a_wrong_extension_warns(self, make_vtkhdf_file, caplog):
        with make_vtkhdf_file("model.h5"):
            pass

        assert "Invalid file extension" in caplog.text

    def test_the_offending_extension_is_named_in_the_warning(
        self, make_vtkhdf_file, caplog
    ):
        with make_vtkhdf_file("model.h5"):
            pass

        assert "'.h5'" in caplog.text

    def test_a_missing_extension_does_not_warn(self, make_vtkhdf_file, caplog):
        """Nothing was replaced, so there is nothing to report."""
        with make_vtkhdf_file("model"):
            pass

        assert "Invalid file extension" not in caplog.text

    def test_a_correct_extension_does_not_warn(self, make_vtkhdf_file, caplog):
        with make_vtkhdf_file("model.vtkhdf"):
            pass

        assert "Invalid file extension" not in caplog.text

    def test_the_directory_is_preserved(self, make_vtkhdf_file, tmp_path):
        with make_vtkhdf_file("model.h5") as handler:
            assert handler.filename.parent == tmp_path

    def test_the_file_is_written_where_the_name_says(
        self, make_vtkhdf_file, tmp_path
    ):
        with make_vtkhdf_file("model.h5"):
            pass

        assert (tmp_path / "model.vtkhdf").is_file()

    def test_a_dot_in_the_name_truncates_it(self, make_vtkhdf_file):
        """``with_suffix`` replaces everything after the last dot.

        A snapshot called ``model_1.5`` — a perfectly ordinary name for a
        1.5 ns time point — is written as ``model_1.vtkhdf``, and the next
        snapshot at ``model_1.7`` overwrites it. The warning that fires says
        the extension ``'.5'`` was invalid, which is technically true and
        entirely unhelpful. Pinned as the current behaviour; written up in
        ``notes/bugs/vtkhdf-filename-suffix-truncation.md``.
        """
        with make_vtkhdf_file("model_1.5") as handler:
            assert handler.filename.name == "model_1.vtkhdf"

    def test_the_filename_is_stored_as_a_path(self, make_vtkhdf_file):
        """Callers read ``handler.filename`` to log where output went."""
        with make_vtkhdf_file("model.vtkhdf") as handler:
            assert isinstance(handler.filename, type(handler.filename.parent))


class TestRootGroup:
    """The ``VTKHDF`` group, and the two attributes VTK requires on it."""

    def test_the_root_group_exists_on_disk(self, make_vtkhdf_file, tmp_path):
        with make_vtkhdf_file():
            pass

        with h5py.File(tmp_path / "model.vtkhdf", "r") as f:
            assert "VTKHDF" in f

    def test_the_root_group_is_a_group(self, make_vtkhdf_file, tmp_path):
        with make_vtkhdf_file():
            pass

        with h5py.File(tmp_path / "model.vtkhdf", "r") as f:
            assert isinstance(f["VTKHDF"], h5py.Group)

    def test_the_version_attribute_is_written(self, make_vtkhdf_file, read_h5):
        with make_vtkhdf_file() as handler:
            path = handler.filename

        attrs, _ = read_h5(path)

        assert list(attrs["VTKHDF/Version"]) == VERSION

    def test_the_version_attribute_is_integral(self, make_vtkhdf_file, read_h5):
        """VTK parses it as a pair of integers, not a string."""
        with make_vtkhdf_file() as handler:
            path = handler.filename

        attrs, _ = read_h5(path)

        assert np.issubdtype(attrs["VTKHDF/Version"].dtype, np.integer)

    def test_the_type_attribute_is_written(self, make_vtkhdf_file, read_h5):
        with make_vtkhdf_file(vtk_file_type=VtkFileType.IMAGE_DATA) as handler:
            path = handler.filename

        attrs, _ = read_h5(path)

        assert attrs["VTKHDF/Type"] == IMAGE_DATA_TYPE

    def test_the_type_attribute_is_bytes_not_a_string(
        self, make_vtkhdf_file, read_h5
    ):
        """The whole reason for the explicit dtype.

        A Python ``str`` would be stored as variable-length UTF-8, which VTK
        does not read.
        """
        with make_vtkhdf_file() as handler:
            path = handler.filename

        attrs, _ = read_h5(path)

        assert isinstance(attrs["VTKHDF/Type"], bytes)

    def test_the_type_attribute_is_fixed_length_ascii(
        self, make_vtkhdf_file, tmp_path
    ):
        """Length exactly ``len("ImageData")`` — 9 bytes, no terminator."""
        with make_vtkhdf_file():
            pass

        with h5py.File(tmp_path / "model.vtkhdf", "r") as f:
            dtype = f["VTKHDF"].attrs.get_id("Type").dtype

        assert dtype == np.dtype("S9")

    def test_the_unstructured_type_gets_its_own_length(
        self, make_vtkhdf_file, tmp_path
    ):
        """16 bytes for ``UnstructuredGrid`` — the length is per-value."""
        with make_vtkhdf_file(vtk_file_type=VtkFileType.UNSTRUCTURED_GRID):
            pass

        with h5py.File(tmp_path / "model.vtkhdf", "r") as f:
            dtype = f["VTKHDF"].attrs.get_id("Type").dtype

        assert dtype == np.dtype("S16")

    def test_exactly_two_attributes_are_written(self, make_vtkhdf_file, read_h5):
        """A third would be non-standard and might confuse a reader."""
        with make_vtkhdf_file() as handler:
            path = handler.filename

        attrs, _ = read_h5(path)

        assert sorted(attrs) == ["VTKHDF/Type", "VTKHDF/Version"]

    def test_no_datasets_are_written_by_the_base_constructor(
        self, make_vtkhdf_file, read_h5
    ):
        with make_vtkhdf_file() as handler:
            path = handler.filename

        _, datasets = read_h5(path)

        assert datasets == {}


class TestExistingFiles:
    """Reopening — the attributes are checked rather than rewritten."""

    def test_an_existing_file_is_truncated_in_write_mode(
        self, make_vtkhdf_file, vtkhdf_path, read_h5
    ):
        with make_vtkhdf_file() as handler:
            handler._write_root_dataset("Leftover", np.array([1, 2, 3]))

        with make_vtkhdf_file():
            pass

        _, datasets = read_h5(vtkhdf_path())

        assert datasets == {}

    def test_a_matching_version_does_not_warn(
        self, make_vtkhdf_file, caplog
    ):
        with make_vtkhdf_file():
            pass
        caplog.clear()

        with make_vtkhdf_file(mode="r+"):
            pass

        assert "mismatch" not in caplog.text

    def test_a_mismatched_version_warns(
        self, make_vtkhdf_file, vtkhdf_path, caplog
    ):
        """A file written by a future gprMax should not be silently trusted."""
        with make_vtkhdf_file() as handler:
            handler._set_root_attribute("Version", [9, 9])
        caplog.clear()

        with make_vtkhdf_file(mode="r+"):
            pass

        assert "mismatch" in caplog.text

    def test_a_readonly_file_missing_the_attributes_warns(
        self, make_vtkhdf_file, vtkhdf_path, caplog
    ):
        """Read-only mode cannot repair the file, so it reports instead."""
        with h5py.File(vtkhdf_path("bare.vtkhdf"), "w") as f:
            f.create_group("VTKHDF")
        caplog.clear()

        with make_vtkhdf_file("bare.vtkhdf", mode="r"):
            pass

        assert "not found" in caplog.text

    def test_a_readonly_file_does_not_gain_the_attributes(
        self, make_vtkhdf_file, vtkhdf_path, read_h5
    ):
        with h5py.File(vtkhdf_path("bare.vtkhdf"), "w") as f:
            f.create_group("VTKHDF")

        with make_vtkhdf_file("bare.vtkhdf", mode="r"):
            pass

        attrs, _ = read_h5(vtkhdf_path("bare.vtkhdf"))

        assert attrs == {}

    def test_a_readwrite_file_gains_the_attributes(
        self, make_vtkhdf_file, vtkhdf_path, read_h5
    ):
        """Mode ``r+`` is the one branch that repairs a file in place."""
        with h5py.File(vtkhdf_path("bare.vtkhdf"), "w") as f:
            f.create_group("VTKHDF")

        with make_vtkhdf_file("bare.vtkhdf", mode="r+"):
            pass

        attrs, _ = read_h5(vtkhdf_path("bare.vtkhdf"))

        assert sorted(attrs) == ["VTKHDF/Type", "VTKHDF/Version"]

    def test_a_missing_file_in_read_mode_raises(self, make_vtkhdf_file):
        with pytest.raises(OSError):
            make_vtkhdf_file("absent.vtkhdf", mode="r")

    def test_the_root_group_is_created_if_absent(
        self, make_vtkhdf_file, vtkhdf_path
    ):
        """``require_group``, so an empty HDF5 file is upgraded rather than
        rejected.
        """
        with h5py.File(vtkhdf_path("empty.vtkhdf"), "w"):
            pass

        with make_vtkhdf_file("empty.vtkhdf", mode="r+") as handler:
            assert handler.root_group.name == "/VTKHDF"


class TestTheContextManager:
    """``with VtkHdfFile(...) as f:`` — the intended way to use it."""

    def test_entering_returns_the_handler(self, make_vtkhdf_file):
        handler = make_vtkhdf_file()

        with handler as entered:
            assert entered is handler

    def test_exiting_closes_the_file(self, make_vtkhdf_file):
        with make_vtkhdf_file() as handler:
            pass

        assert not handler.file_handler

    def test_it_closes_even_when_the_body_raises(self, make_vtkhdf_file):
        """Otherwise a failed export would leave a locked, truncated file."""
        handler = make_vtkhdf_file()

        with pytest.raises(RuntimeError):
            with handler:
                raise RuntimeError("export failed")

        assert not handler.file_handler

    def test_it_does_not_suppress_exceptions(self, make_vtkhdf_file):
        """``__exit__`` returns ``None``, so the error still propagates."""
        with pytest.raises(RuntimeError):
            with make_vtkhdf_file():
                raise RuntimeError("must escape")

    def test_the_data_is_readable_after_exit(
        self, make_vtkhdf_file, vtkhdf_path, read_h5
    ):
        """The point of closing: buffers are flushed."""
        with make_vtkhdf_file() as handler:
            handler._write_root_dataset("Numbers", np.array([1, 2, 3]))

        _, datasets = read_h5(vtkhdf_path())

        assert list(datasets["VTKHDF/Numbers"]) == [1, 2, 3]


class TestClose:
    """``close`` — also callable directly, and more than once."""

    def test_it_closes_the_handle(self, make_vtkhdf_file):
        handler = make_vtkhdf_file()

        handler.close()

        assert not handler.file_handler

    def test_calling_it_twice_is_harmless(self, make_vtkhdf_file):
        """h5py treats closing a closed file as a no-op, so the ``__exit__``
        of a handler that was already closed cannot raise.
        """
        handler = make_vtkhdf_file()

        handler.close()
        handler.close()

        assert not handler.file_handler

    def test_closing_inside_a_context_is_harmless(self, make_vtkhdf_file):
        with make_vtkhdf_file() as handler:
            handler.close()

        assert not handler.file_handler

    def test_writing_after_close_raises(self, make_vtkhdf_file):
        handler = make_vtkhdf_file()
        handler.close()

        with pytest.raises(Exception):
            handler._write_root_dataset("Late", np.array([1]))


class TestDatasetPaths:
    """``_build_dataset_path`` and ``_get_dataset`` — the plumbing beneath."""

    def test_a_single_component_hangs_off_the_root(self, make_vtkhdf_file):
        with make_vtkhdf_file() as handler:
            assert handler._build_dataset_path("Points") == "VTKHDF/Points"

    def test_several_components_are_joined(self, make_vtkhdf_file):
        with make_vtkhdf_file() as handler:
            assert (
                handler._build_dataset_path("CellData", "Material")
                == "VTKHDF/CellData/Material"
            )

    def test_no_components_gives_the_root_group(self, make_vtkhdf_file):
        with make_vtkhdf_file() as handler:
            assert handler._build_dataset_path() == "VTKHDF"

    def test_an_existing_dataset_is_returned(self, make_vtkhdf_file):
        with make_vtkhdf_file() as handler:
            handler._write_root_dataset("Numbers", np.array([1, 2, 3]))

            assert handler._get_root_dataset("Numbers").shape == (3,)

    def test_a_missing_dataset_raises_a_key_error(self, make_vtkhdf_file):
        with make_vtkhdf_file() as handler:
            with pytest.raises(KeyError):
                handler._get_dataset("VTKHDF/Absent")

    def test_the_missing_path_message_is_never_the_one_reported(
        self, make_vtkhdf_file
    ):
        """``h5py`` returns ``None`` for an absent path, not ``"default"``.

        So the ``cls == "default"`` branch — and its clear "Path does not
        exist" message — is unreachable, and a simple typo in a dataset name
        surfaces as ``Dataset not found. Found 'None' instead``, which reads
        like a type problem rather than a missing key. Pinned as the current
        behaviour; written up in
        ``notes/bugs/vtkhdf-unreachable-missing-path-branch.md``.
        """
        with make_vtkhdf_file() as handler:
            with pytest.raises(KeyError, match="Found 'None' instead"):
                handler._get_dataset("VTKHDF/Absent")

    def test_a_path_pointing_at_a_group_raises(self, make_vtkhdf_file):
        """Asking for the root group as a dataset is a caller error.

        The message names what was found, which is the difference between a
        typo and a structural mistake.
        """
        with make_vtkhdf_file() as handler:
            with pytest.raises(KeyError, match="Dataset not found"):
                handler._get_dataset("VTKHDF")


class TestRootAttributes:
    """``_set_root_attribute`` / ``_get_root_attribute`` / ``_has_...``."""

    def test_an_attribute_can_be_set_and_read_back(self, make_vtkhdf_file):
        with make_vtkhdf_file() as handler:
            handler._set_root_attribute("Spacing", [0.1, 0.2, 0.3])

            assert list(handler._get_root_attribute("Spacing")) == [0.1, 0.2, 0.3]

    def test_setting_an_attribute_twice_replaces_it(self, make_vtkhdf_file):
        with make_vtkhdf_file() as handler:
            handler._set_root_attribute("Spacing", [1.0])
            handler._set_root_attribute("Spacing", [2.0])

            assert list(handler._get_root_attribute("Spacing")) == [2.0]

    def test_an_explicit_dtype_is_honoured(self, make_vtkhdf_file):
        """``WholeExtent`` must be integral or VTK reads garbage extents."""
        with make_vtkhdf_file() as handler:
            handler._set_root_attribute("Extent", [0, 1, 2], dtype=np.int32)

            assert handler._get_root_attribute("Extent").dtype == np.int32

    def test_a_present_attribute_is_reported_present(self, make_vtkhdf_file):
        with make_vtkhdf_file() as handler:
            assert handler._has_root_attribute("Version") is True

    def test_an_absent_attribute_is_reported_absent(self, make_vtkhdf_file):
        with make_vtkhdf_file() as handler:
            assert handler._has_root_attribute("Nothing") is False

    def test_reading_an_absent_attribute_raises(self, make_vtkhdf_file):
        with make_vtkhdf_file() as handler:
            with pytest.raises(KeyError):
                handler._get_root_attribute("Nothing")

    def test_the_error_names_the_group(self, make_vtkhdf_file):
        """So a user can tell a missing attribute from a missing group."""
        with make_vtkhdf_file() as handler:
            with pytest.raises(KeyError, match="VTKHDF"):
                handler._get_root_attribute("Nothing")

    def test_attributes_persist_to_disk(self, make_vtkhdf_file, read_h5):
        with make_vtkhdf_file() as handler:
            handler._set_root_attribute("Spacing", [0.1, 0.2, 0.3])
            path = handler.filename

        attrs, _ = read_h5(path)

        assert list(attrs["VTKHDF/Spacing"]) == [0.1, 0.2, 0.3]


class TestCellTypes:
    """``VtkCellType`` — the numeric codes VTK defines, mirrored here."""

    def test_the_line_type_is_three(self):
        """The only one gprMax writes: geometry views in line mode."""
        from gprMax.vtkhdf_filehandlers.vtkhdf import VtkCellType

        assert VtkCellType.LINE == 3

    def test_the_voxel_type_is_eleven(self):
        from gprMax.vtkhdf_filehandlers.vtkhdf import VtkCellType

        assert VtkCellType.VOXEL == 11

    def test_the_values_are_the_vtk_ones(self):
        """Copied from ``vtkCellType.h``; a drift here is a silent corruption
        of every file written, since VTK trusts the number.
        """
        from gprMax.vtkhdf_filehandlers.vtkhdf import VtkCellType

        assert [member.value for member in VtkCellType] == list(range(17))

    def test_the_members_are_unsigned_bytes(self):
        """The ``Types`` dataset is written as ``uint8``."""
        from gprMax.vtkhdf_filehandlers.vtkhdf import VtkCellType

        assert isinstance(VtkCellType.LINE, np.uint8)

    def test_a_cell_type_can_be_used_as_a_number(self):
        from gprMax.vtkhdf_filehandlers.vtkhdf import VtkCellType

        assert int(VtkCellType.TETRA) + 1 == 11
