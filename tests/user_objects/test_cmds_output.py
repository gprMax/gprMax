"""Tests for ``gprMax.user_objects.cmds_output``.

Three output user-object classes: ``Snapshot``, ``GeometryView``,
``GeometryObjectsWrite``. Each writes data out at the end of a
simulation. ``build(model, grid)`` validates inputs then calls a
``model.add_*`` factory.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gprMax.user_objects.cmds_output import GeometryObjectsWrite, GeometryView, Snapshot

# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def _make(self, **overrides):
        defaults = dict(
            p1=(0.0, 0.0, 0.0),
            p2=(0.05, 0.05, 0.05),
            dl=(0.001, 0.001, 0.001),
            filename="snap1",
            iterations=10,
        )
        defaults.update(overrides)
        return Snapshot(**defaults)

    def test_constructor_stores_attributes(self):
        s = self._make()
        assert s.lower_bound == (0.0, 0.0, 0.0)
        assert s.upper_bound == (0.05, 0.05, 0.05)
        assert s.dl == (0.001, 0.001, 0.001)
        assert s.filename == "snap1"
        assert s.iterations == 10
        assert s.time is None
        assert s.outputs is None

    def test_constructor_stores_kwargs(self):
        s = self._make(time=1e-9, fileext=".h5", outputs=["Ex"])
        assert s.kwargs["filename"] == "snap1"
        assert s.kwargs["fileext"] == ".h5"
        assert s.kwargs["outputs"] == ["Ex"]
        assert s.kwargs["iterations"] == 10
        assert s.kwargs["time"] == 1e-9

    def test_order_and_hash(self):
        s = self._make()
        assert s.order == 9
        assert s.hash == "#snapshot"


class TestSnapshotCalculateUpperBound:
    """``_calculate_upper_bound`` is the pure-math helper used to align
    the upper bound of a snapshot to the requested step size.
    """

    def test_returns_start_plus_step_times_ceil_size_over_step(self):
        s = Snapshot(
            p1=(0, 0, 0),
            p2=(0.05, 0.05, 0.05),
            dl=(0.001, 0.001, 0.001),
            filename="x",
            iterations=10,
        )
        start = np.array([0, 0, 0])
        step = np.array([3, 3, 3])
        size = np.array([10, 10, 10])
        # ceil(10/3) = 4 → 0 + 3*4 = 12
        out = s._calculate_upper_bound(start, step, size)
        np.testing.assert_array_equal(out, np.array([12, 12, 12]))

    def test_aligned_size_returns_size(self):
        s = Snapshot(
            p1=(0, 0, 0),
            p2=(0.05, 0.05, 0.05),
            dl=(0.001, 0.001, 0.001),
            filename="x",
            iterations=10,
        )
        # size is exact multiple of step → ceil is exact
        out = s._calculate_upper_bound(
            np.array([0, 0, 0]), np.array([2, 2, 2]), np.array([10, 10, 10])
        )
        np.testing.assert_array_equal(out, np.array([10, 10, 10]))


class TestSnapshotBuildValidation:
    def _patch_uip(self, lower=None, upper=None, dl=None):
        """Return a MagicMock-style uip with bounded responses."""
        uip = MagicMock()
        lower = np.array([0, 0, 0]) if lower is None else lower
        upper = np.array([10, 10, 10]) if upper is None else upper
        dl = np.array([1, 1, 1]) if dl is None else dl
        uip.check_output_object_bounds.return_value = (lower, upper)
        uip.discretise_static_point.return_value = dl
        uip.round_to_grid_static_point.return_value = (0.0, 0.0, 0.0)
        return uip

    def test_build_accepts_subgrid(self, stub_model):
        from gprMax.subgrids.grid import SubGridBaseGrid

        s = Snapshot(
            p1=(0, 0, 0),
            p2=(0.05, 0.05, 0.05),
            dl=(0.001, 0.001, 0.001),
            filename="x",
            iterations=10,
        )
        subgrid = MagicMock(spec=SubGridBaseGrid)
        subgrid.iterations = 100
        subgrid.dt = 1e-12
        subgrid.dl = np.ones(3)
        stub_model.add_snapshot.return_value = None
        with patch.object(Snapshot, "_create_uip", return_value=self._patch_uip()):
            s.build(stub_model, subgrid)
        stub_model.add_snapshot.assert_called_once()

    def test_build_rejects_iterations_above_grid(self, stub_model, stub_grid):
        # grid.iterations = 100 (stub default); iterations=200 → too big
        s = Snapshot(
            p1=(0, 0, 0),
            p2=(0.05, 0.05, 0.05),
            dl=(0.001, 0.001, 0.001),
            filename="x",
            iterations=200,
        )
        with patch.object(Snapshot, "_create_uip", return_value=self._patch_uip()):
            with pytest.raises(ValueError):
                s.build(stub_model, stub_grid)

    def test_build_rejects_zero_iterations(self, stub_model, stub_grid):
        s = Snapshot(
            p1=(0, 0, 0),
            p2=(0.05, 0.05, 0.05),
            dl=(0.001, 0.001, 0.001),
            filename="x",
            iterations=0,
        )
        with patch.object(Snapshot, "_create_uip", return_value=self._patch_uip()):
            with pytest.raises(ValueError):
                s.build(stub_model, stub_grid)

    def test_build_rejects_negative_time(self, stub_model, stub_grid):
        s = Snapshot(
            p1=(0, 0, 0),
            p2=(0.05, 0.05, 0.05),
            dl=(0.001, 0.001, 0.001),
            filename="x",
            time=-1.0,
        )
        with patch.object(Snapshot, "_create_uip", return_value=self._patch_uip()):
            with pytest.raises(ValueError):
                s.build(stub_model, stub_grid)

    def test_build_rejects_missing_iterations_and_time(self, stub_model, stub_grid):
        s = Snapshot(
            p1=(0, 0, 0),
            p2=(0.05, 0.05, 0.05),
            dl=(0.001, 0.001, 0.001),
            filename="x",
        )
        with patch.object(Snapshot, "_create_uip", return_value=self._patch_uip()):
            with pytest.raises(ValueError):
                s.build(stub_model, stub_grid)

    def test_build_rejects_invalid_fileext(self, stub_model, stub_grid):
        s = Snapshot(
            p1=(0, 0, 0),
            p2=(0.05, 0.05, 0.05),
            dl=(0.001, 0.001, 0.001),
            filename="x",
            iterations=10,
            fileext=".bogus",
        )
        with patch.object(Snapshot, "_create_uip", return_value=self._patch_uip()):
            with pytest.raises(ValueError):
                s.build(stub_model, stub_grid)

    def test_build_rejects_invalid_output(self, stub_model, stub_grid):
        s = Snapshot(
            p1=(0, 0, 0),
            p2=(0.05, 0.05, 0.05),
            dl=(0.001, 0.001, 0.001),
            filename="x",
            iterations=10,
            outputs=["Notreal"],
        )
        with patch.object(Snapshot, "_create_uip", return_value=self._patch_uip()):
            with pytest.raises(ValueError):
                s.build(stub_model, stub_grid)

    def test_build_defaults_fileext_to_vtkhdf(self, stub_model, stub_grid):
        s = Snapshot(
            p1=(0, 0, 0),
            p2=(0.05, 0.05, 0.05),
            dl=(0.001, 0.001, 0.001),
            filename="x",
            iterations=10,
        )
        # The logging branch reads ``snapshot.time`` and ``snapshot.filename``
        # — give the stub a real ``SimpleNamespace`` so f-string formatting
        # works.
        stub_model.add_snapshot.return_value = SimpleNamespace(time=10, filename="x")
        with patch.object(Snapshot, "_create_uip", return_value=self._patch_uip()):
            s.build(stub_model, stub_grid)
        # ``self.file_extension`` is set during build()
        assert s.file_extension == ".vtkhdf"
        stub_model.add_snapshot.assert_called_once()


# ---------------------------------------------------------------------------
# GeometryView
# ---------------------------------------------------------------------------


class TestGeometryView:
    def _make(self, **overrides):
        defaults = dict(
            p1=(0.0, 0.0, 0.0),
            p2=(0.05, 0.05, 0.05),
            dl=(0.001, 0.001, 0.001),
            output_type="n",
            filename="geo1",
        )
        defaults.update(overrides)
        return GeometryView(**defaults)

    def test_constructor_stores_attributes_and_kwargs(self):
        g = self._make()
        assert g.lower_bound == (0.0, 0.0, 0.0)
        assert g.upper_bound == (0.05, 0.05, 0.05)
        assert g.output_type == "n"
        assert g.filename == "geo1"
        assert g.kwargs["output_type"] == "n"

    def test_order_and_hash(self):
        g = self._make()
        assert g.order == 17
        assert g.hash == "#geometry_view"

    def _patch_uip(self, dl=None):
        uip = MagicMock()
        uip.check_output_object_bounds.return_value = (np.array([0, 0, 0]), np.array([50, 50, 50]))
        uip.discretise_static_point.return_value = np.array([1, 1, 1]) if dl is None else dl
        uip.round_to_grid_static_point.return_value = (0.0, 0.0, 0.0)
        return uip

    def test_build_rejects_unknown_type(self, stub_model, stub_grid):
        g = self._make(output_type="x")
        with patch.object(GeometryView, "_create_uip", return_value=self._patch_uip()):
            with pytest.raises(ValueError):
                g.build(stub_model, stub_grid)

    def test_build_fine_requires_dl_equals_one(self, stub_model, stub_grid):
        # output_type='f' requires every dl component == 1
        g = self._make(output_type="f")
        with patch.object(
            GeometryView,
            "_create_uip",
            return_value=self._patch_uip(dl=np.array([2, 2, 2])),
        ):
            with pytest.raises(ValueError):
                g.build(stub_model, stub_grid)

    def test_build_normal_calls_add_voxels(self, stub_model, stub_grid):
        g = self._make(output_type="n")
        with patch.object(GeometryView, "_create_uip", return_value=self._patch_uip()):
            g.build(stub_model, stub_grid)
        stub_model.add_geometry_view_voxels.assert_called_once()
        stub_model.add_geometry_view_lines.assert_not_called()

    def test_build_fine_calls_add_lines(self, stub_model, stub_grid):
        g = self._make(output_type="f")
        with patch.object(GeometryView, "_create_uip", return_value=self._patch_uip()):
            g.build(stub_model, stub_grid)
        stub_model.add_geometry_view_lines.assert_called_once()
        stub_model.add_geometry_view_voxels.assert_not_called()

    def test_build_rejects_negative_dl(self, stub_model, stub_grid):
        g = self._make()
        with patch.object(
            GeometryView,
            "_create_uip",
            return_value=self._patch_uip(dl=np.array([-1, 1, 1])),
        ):
            with pytest.raises(ValueError):
                g.build(stub_model, stub_grid)


# ---------------------------------------------------------------------------
# GeometryObjectsWrite
# ---------------------------------------------------------------------------


class TestGeometryObjectsWrite:
    def test_constructor_stores_attributes_and_kwargs(self):
        g = GeometryObjectsWrite(p1=(0, 0, 0), p2=(0.05, 0.05, 0.05), filename="objs")
        assert g.lower_bound == (0, 0, 0)
        assert g.upper_bound == (0.05, 0.05, 0.05)
        assert g.basefilename == "objs"
        assert g.kwargs["filename"] == "objs"

    def test_order_and_hash(self):
        g = GeometryObjectsWrite(p1=(0, 0, 0), p2=(0.05, 0.05, 0.05), filename="objs")
        assert g.order == 18
        assert g.hash == "#geometry_objects_write"

    def test_build_rejects_subgrid(self, stub_model):
        from gprMax.subgrids.grid import SubGridBaseGrid

        g = GeometryObjectsWrite(p1=(0, 0, 0), p2=(0.05, 0.05, 0.05), filename="objs")
        subgrid = MagicMock(spec=SubGridBaseGrid)
        with pytest.raises(ValueError):
            g.build(stub_model, subgrid)

    def test_build_calls_add_geometry_object(self, stub_model, stub_grid):
        g = GeometryObjectsWrite(p1=(0, 0, 0), p2=(0.05, 0.05, 0.05), filename="objs")
        uip = MagicMock()
        uip.check_output_object_bounds.return_value = (np.array([0, 0, 0]), np.array([50, 50, 50]))
        uip.round_to_grid_static_point.return_value = (0.0, 0.0, 0.0)
        with patch.object(GeometryObjectsWrite, "_create_uip", return_value=uip):
            g.build(stub_model, stub_grid)
        stub_model.add_geometry_object.assert_called_once()


pytestmark = pytest.mark.unit
