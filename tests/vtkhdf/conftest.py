"""Shared fixtures for the VTKHDF file-handler suite.

These three classes are the last stage of every geometry and snapshot export:
whatever gprMax computed, this is the code that turns it into bytes another
program will read. VTK is that other program, and it is unforgiving — a
dataset written with the wrong dimension order does not fail to load, it loads
as a model that is silently mirrored.

**How this suite differs from PR 10's.** ``tests/unit/outputs/`` already
drives these writers, but always from the outside: it builds a geometry view,
calls ``write_vtk``, and asserts on the resulting file. That pins the *current
on-disk layout*, which is what a regression test should do. It says nothing
about the writer's own contract — what happens with a bad file extension, a
partial write, a 1-D shape, a duplicate dataset name, a string field. Those
paths have never been executed by anything. This suite covers them, and
deliberately does not re-assert what PR 10 already pins.

**Everything is verified by reopening the file.** A writer test that only
inspects the writer's own attributes proves nothing; ``h5py`` buffers, and a
dataset can be created and never flushed. Every assertion here goes through
``read_h5`` or a fresh ``h5py.File``, after the writer has been closed.

**Nothing writes outside ``tmp_path``.** The constructors truncate on open —
mode ``"w"`` — so a stray relative filename would destroy a real file in the
working directory. Every fixture takes ``tmp_path`` and returns an absolute
path.

One inherited constraint is worth stating: ``VtkHdfFile.__init__`` opens the
file *before* it validates anything, so a constructor that raises still leaves
a truncated file behind and an open handle. Tests that provoke a validation
error therefore cannot assume the file is absent afterwards.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from gprMax.vtkhdf_filehandlers.vtkhdf import VtkCellType

# The ``Type`` attribute string for each concrete handler, as it appears on
# disk. VTK matches on these exactly.
IMAGE_DATA_TYPE = b"ImageData"
UNSTRUCTURED_GRID_TYPE = b"UnstructuredGrid"

# The VTKHDF format version this code targets.
VERSION = [2, 2]


@pytest.fixture
def vtkhdf_path(tmp_path):
    """An absolute path with the correct extension, inside ``tmp_path``."""

    def _path(name="model.vtkhdf"):
        return tmp_path / name

    return _path


@pytest.fixture
def read_h5():
    """Reopen a written file and hand back its attributes and datasets.

    Returns ``(attrs, datasets)`` as plain dicts so assertions read as data
    comparisons rather than h5py mechanics; group members are flattened to
    ``"parent/child"`` keys. The same helper PR 10's outputs suite uses, so
    the two sets of round-trip assertions are directly comparable.
    """

    def _read(path):
        attrs = {}
        datasets = {}

        def visit(name, obj):
            for key, value in obj.attrs.items():
                attrs[f"{name}/{key}"] = value
            if isinstance(obj, h5py.Dataset):
                datasets[name] = obj[()]

        with h5py.File(Path(path), "r") as f:
            attrs.update(dict(f.attrs))
            f.visititems(visit)
        return attrs, datasets

    return _read


@pytest.fixture
def make_vtkhdf_file(vtkhdf_path):
    """Factory for a bare ``VtkHdfFile``, opened for writing.

    The base class carries no ``@abstractmethod``, so it can be instantiated
    directly — which is what lets the dataset-writing machinery be tested
    without dragging in either concrete subclass's validation rules.

    Files are *not* closed automatically: several tests assert on ``close``
    itself. Those that do not use the returned handle as a context manager.
    """
    from gprMax.vtkhdf_filehandlers.vtkhdf import VtkFileType, VtkHdfFile

    def _make(name="model.vtkhdf", vtk_file_type=VtkFileType.IMAGE_DATA, mode="w"):
        return VtkHdfFile(vtkhdf_path(name), vtk_file_type, mode)

    return _make


@pytest.fixture
def make_image_data(vtkhdf_path):
    """Factory for a minimal ``VtkImageData``.

    The default shape is deliberately anisotropic — ``(2, 3, 4)`` — because a
    cubic default would hide every transposition bug this suite exists to
    catch.
    """
    from gprMax.vtkhdf_filehandlers.vtk_image_data import VtkImageData

    def _make(name="image.vtkhdf", shape=(2, 3, 4), **kwargs):
        return VtkImageData(
            vtkhdf_path(name), np.array(shape, dtype=np.int32), **kwargs
        )

    return _make


@pytest.fixture
def make_unstructured_grid(vtkhdf_path):
    """Factory for a minimal ``VtkUnstructuredGrid``: two points, one line.

    That is the smallest valid grid — and it is also the shape gprMax actually
    writes, since a line-mode geometry view is a collection of two-point
    edges.
    """
    from gprMax.vtkhdf_filehandlers.vtk_unstructured_grid import (
        VtkUnstructuredGrid,
    )

    def _make(
        name="grid.vtkhdf",
        points=None,
        cell_types=None,
        connectivity=None,
        cell_offsets=None,
    ):
        if points is None:
            points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        if cell_types is None:
            cell_types = np.array([VtkCellType.LINE], dtype=np.uint8)
        if connectivity is None:
            connectivity = np.array([0, 1], dtype=np.int32)
        if cell_offsets is None:
            cell_offsets = np.array([0, 2], dtype=np.int32)
        return VtkUnstructuredGrid(
            vtkhdf_path(name), points, cell_types, connectivity, cell_offsets
        )

    return _make


@pytest.fixture
def line_grid_arrays():
    """The four arrays describing a three-point, two-line grid.

    Returned as a dict so a test can override one and leave the rest
    consistent — most validation tests are exactly that.
    """
    return {
        "points": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        "cell_types": np.array(
            [VtkCellType.LINE, VtkCellType.LINE], dtype=np.uint8
        ),
        "connectivity": np.array([0, 1, 1, 2], dtype=np.int32),
        "cell_offsets": np.array([0, 2, 4], dtype=np.int32),
    }


@pytest.fixture(autouse=True)
def _reset_gprmax_logger():
    """Reset gprMax logger so caplog works after upstream tests pollute it."""
    import logging
    logger = logging.getLogger("gprMax")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    yield
    logger.handlers.clear()
