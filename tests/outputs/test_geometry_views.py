"""``GeometryView`` and ``Metadata`` — the self-describing part of an export.

A geometry view exports what a model is *made of*. The pixels are the easy
part; the reason the files are usable in ParaView without a separate legend is
``Metadata``, which attaches version, discretisation, domain size, material
names, PML depths, and every source and receiver position as VTKHDF field data.

Three things here repay attention.

**``pml_gv_comment`` reports the PML depth visible *in this view*, not the
grid's PML thickness.** A view of the model's interior sees none of it and gets
zeros; a view overlapping a slab reports how far in it reaches. So the answer
depends on the view's bounds as well as the grid's settings, and the six faces
are computed by six separate comparisons.

**Empty means absent, not zero.** A model with no sources writes no
``source_ids`` field rather than an empty one, because ``srcs_rx_gv_comment``
returns ``None`` and ``write_to_vtkhdf`` skips on ``None``. Same for receivers
and for PMLs that are switched off.

**``materials_comment`` prefers the view's material list if there is one.**
``GeometryViewLines`` and ``GeometryObject`` call ``initialise_materials``
first, so their metadata carries the view's remapped list;
``GeometryViewVoxels`` does not, so its metadata falls back to the grid's whole
unfiltered list. The two exporters therefore describe their materials
differently — see ``test_geometry_view_voxels.py``.
"""

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gprMax._version import __version__
from gprMax.geometry_outputs.geometry_views import (
    GeometryView,
    Metadata,
    MPIMetadata,
    save_geometry_views,
)

from .conftest import DL, DL_ANISO


@pytest.fixture
def pml_grid(make_view_grid):
    """A grid whose PML slabs are switched on, with a known thickness."""

    def _make(thickness=4, nx=20, ny=20, nz=20, **kwargs):
        g = make_view_grid(nx=nx, ny=ny, nz=nz, **kwargs)
        g.set_pml_thickness(thickness)
        g.pmls["slabs"] = ["a-slab"]
        return g

    return _make


@pytest.fixture
def make_metadata(make_grid_view):
    """Factory for a ``Metadata`` over a view of a real grid."""

    def _make(grid=None, view=None, **kwargs):
        gv = view if view is not None else make_grid_view(grid=grid)
        return Metadata(gv, **kwargs)

    return _make


class TestGeometryViewConstruction:
    def test_is_abstract(self, make_view_grid):
        """Expects ``prep_vtk`` and ``write_vtk`` to be abstract, so the base
        class cannot be used directly as an exporter."""
        assert GeometryView.prep_vtk.__isabstractmethod__
        assert GeometryView.write_vtk.__isabstractmethod__

    def test_builds_a_grid_view_from_the_extents(self, make_view_grid):
        """Expects the nine coordinate arguments handed to a ``GridView``."""
        view = _ConcreteView(1, 2, 3, 5, 6, 7, 1, 1, 1, "geo", make_view_grid())
        assert view.grid_view.start.tolist() == [1, 2, 3]

    def test_stores_the_filename_base(self, make_view_grid):
        """Expects the user's name kept separately from the resolved path —
        ``set_filename`` combines it with the model number later."""
        view = _ConcreteView(0, 0, 0, 4, 4, 4, 1, 1, 1, "geo", make_view_grid())
        assert view.filenamebase == "geo"

    def test_starts_with_no_prepared_data(self, make_view_grid):
        """Expects ``nbytes``, ``material_data`` and ``materials`` all unset
        until ``prep_vtk`` runs."""
        view = _ConcreteView(0, 0, 0, 4, 4, 4, 1, 1, 1, "geo", make_view_grid())
        assert (view.nbytes, view.material_data, view.materials) == (None, None, None)

    def test_grid_is_reached_through_the_view(self, make_view_grid):
        """Expects ``view.grid`` to forward to ``grid_view.grid``."""
        g = make_view_grid()
        assert _ConcreteView(0, 0, 0, 4, 4, 4, 1, 1, 1, "geo", g).grid is g

    def test_the_file_extension_is_vtkhdf(self):
        """Expects ``.vtkhdf`` for every geometry view — unlike snapshots,
        there is no HDF5 alternative."""
        assert GeometryView.FILE_EXTENSION == ".vtkhdf"


class TestSetFilename:
    def test_uses_the_output_directory(self, make_view_grid, outputs_config):
        """Expects the file to land beside the model's output file, not in the
        working directory."""
        view = _ConcreteView(0, 0, 0, 4, 4, 4, 1, 1, 1, "geo", make_view_grid())
        view.set_filename()
        assert view.filename.parent == outputs_config.model_config.output_file_path.parent

    def test_uses_the_user_supplied_base_name(self, make_view_grid):
        """Expects the stem to come from ``filenamebase`` rather than from the
        model name."""
        view = _ConcreteView(0, 0, 0, 4, 4, 4, 1, 1, 1, "antenna", make_view_grid())
        view.set_filename()
        assert view.filename.stem == "antenna"

    def test_appends_the_model_number(self, make_view_grid, outputs_config):
        """Expects the per-model suffix, so a B-scan's many models do not
        overwrite each other's geometry files."""
        outputs_config.model_config.appendmodelnumber = "3"
        view = _ConcreteView(0, 0, 0, 4, 4, 4, 1, 1, 1, "geo", make_view_grid())
        view.set_filename()
        assert view.filename.stem == "geo3"

    def test_applies_the_vtkhdf_extension(self, make_view_grid):
        """Expects ``.vtkhdf`` whatever the base name."""
        view = _ConcreteView(0, 0, 0, 4, 4, 4, 1, 1, 1, "geo", make_view_grid())
        view.set_filename()
        assert view.filename.suffix == ".vtkhdf"

    def test_replaces_an_existing_suffix(self, make_view_grid):
        """Expects ``with_suffix`` semantics — a base name containing a dot is
        truncated at it, as for snapshots."""
        view = _ConcreteView(0, 0, 0, 4, 4, 4, 1, 1, 1, "geo.old", make_view_grid())
        view.set_filename()
        assert view.filename.name == "geo.vtkhdf"


class TestMetadataBasics:
    def test_records_the_gprmax_version(self, make_metadata):
        """Expects the writing version, so a file can be traced to its build."""
        assert make_metadata().gprmax_version == __version__

    def test_records_the_grid_discretisation(self, make_metadata, make_view_grid):
        """Expects ``grid.dl`` — the *grid's* spacing, not the view's stride."""
        g = make_view_grid(dl=DL_ANISO)
        assert make_metadata(grid=g).dx_dy_dz == pytest.approx(list(DL_ANISO))

    def test_records_the_whole_domain_size(self, make_metadata, make_grid_view, make_view_grid):
        """Expects ``grid.size``, so the metadata locates a partial view within
        the full model."""
        g = make_view_grid(nx=20, ny=20, nz=20)
        view = make_grid_view(grid=g, start=(2, 2, 2), stop=(6, 6, 6))
        assert make_metadata(view=view).nx_ny_nz.tolist() == [20, 20, 20]

    def test_materials_only_skips_the_extra_sections(self, make_metadata):
        """Expects PML, source and receiver information not to be computed at
        all when ``materials_only`` is set — ``GeometryViewLines`` uses this."""
        meta = make_metadata(materials_only=True)
        assert not hasattr(meta, "pml_thickness")
        assert not hasattr(meta, "source_ids")

    def test_the_full_form_computes_them(self, make_metadata):
        """Expects the three extra attributes present by default."""
        meta = make_metadata()
        assert hasattr(meta, "pml_thickness")
        assert hasattr(meta, "source_ids")
        assert hasattr(meta, "receiver_ids")

    def test_grid_is_reached_through_the_view(self, make_metadata, make_view_grid):
        """Expects the ``grid`` property to forward to ``grid_view.grid``."""
        g = make_view_grid()
        assert make_metadata(grid=g).grid is g


class TestMaterialsComment:
    def test_falls_back_to_the_grids_material_list(self, make_metadata, make_view_grid):
        """Expects the grid's whole list when the view has not called
        ``initialise_materials`` — the ``GeometryViewVoxels`` situation."""
        g = make_view_grid(materials=3)
        assert len(make_metadata(grid=g).materials) == 3

    def test_prefers_the_views_material_list(self, make_metadata, make_grid_view, make_view_grid):
        """Expects the view's filtered list once it exists — the
        ``GeometryViewLines`` and ``GeometryObject`` situation."""
        g = make_view_grid(nx=4, ny=4, nz=4, materials=3)
        g.ID[...] = 1
        view = make_grid_view(grid=g, start=(0, 0, 0), stop=(4, 4, 4))
        view.initialise_materials(filter_materials=True)
        assert len(make_metadata(view=view).materials) == 1

    def test_reports_material_names(self, make_metadata, make_view_grid):
        """Expects the user-facing ``#material`` identifiers, not numeric
        IDs — the whole point of the table."""
        g = make_view_grid(materials=2)
        assert make_metadata(grid=g).materials == ["pec", "free_space"]

    def test_smoothed_materials_are_hidden_by_default(self, make_metadata, make_view_grid):
        """Expects the automatically generated dielectric-smoothing materials
        to be omitted: they are an implementation detail of averaging, not
        something the user defined."""
        g = make_view_grid(materials=3)
        g.materials[1].type = "dielectric-smoothed"
        assert len(make_metadata(grid=g).materials) == 2

    def test_averaged_materials_includes_them(self, make_metadata, make_view_grid):
        """Expects the full list when the caller asks for averaged materials."""
        g = make_view_grid(materials=3)
        g.materials[1].type = "dielectric-smoothed"
        assert len(make_metadata(grid=g, averaged_materials=True).materials) == 3

    def test_a_none_material_list_is_passed_through(self, make_metadata, make_grid_view):
        """Expects ``None`` rather than a crash — a non-coordinating MPI rank
        ends up with no material list at all."""
        view = make_grid_view()
        view.materials = None
        assert make_metadata(view=view).materials is None


class TestPmlComment:
    def test_returns_none_when_no_slabs_were_built(self, make_metadata, make_view_grid):
        """Expects ``None`` for a model with PMLs switched off, so the field is
        omitted from the file entirely."""
        g = make_view_grid()
        g.pmls["slabs"] = []
        assert make_metadata(grid=g).pml_thickness is None

    def test_reports_six_depths(self, make_metadata, pml_grid):
        """Expects one entry per face, in the ``pmls["thickness"]`` key
        order."""
        assert len(make_metadata(grid=pml_grid()).pml_thickness) == 6

    def test_a_full_domain_view_sees_the_whole_pml(self, make_metadata, pml_grid, make_grid_view):
        """Expects every face to report the grid's own thickness when the view
        covers the whole domain."""
        g = pml_grid(thickness=4, nx=20, ny=20, nz=20)
        view = make_grid_view(grid=g, start=(0, 0, 0), stop=(20, 20, 20))
        assert make_metadata(view=view).pml_thickness.tolist() == [4] * 6

    def test_an_interior_view_sees_none_of_it(self, make_metadata, pml_grid, make_grid_view):
        """Expects all zeros for a view entirely inside the absorbing shell —
        there is no PML to draw."""
        g = pml_grid(thickness=4, nx=20, ny=20, nz=20)
        view = make_grid_view(grid=g, start=(6, 6, 6), stop=(14, 14, 14))
        assert make_metadata(view=view).pml_thickness.tolist() == [0] * 6

    def test_a_partial_overlap_reports_the_visible_depth(
        self, make_metadata, pml_grid, make_grid_view
    ):
        """Expects ``thickness - xs``: a view starting one cell into a 4-cell
        PML shows three cells of it."""
        g = pml_grid(thickness=4, nx=20, ny=20, nz=20)
        view = make_grid_view(grid=g, start=(1, 1, 1), stop=(14, 14, 14))
        assert make_metadata(view=view).pml_thickness.tolist()[:3] == [3, 3, 3]

    def test_the_high_faces_are_measured_from_the_far_edge(
        self, make_metadata, pml_grid, make_grid_view
    ):
        """Expects ``xf - (nx - thickness)`` for the max faces, so a view
        reaching one cell past the PML's inner edge reports one."""
        g = pml_grid(thickness=4, nx=20, ny=20, nz=20)
        view = make_grid_view(grid=g, start=(0, 0, 0), stop=(17, 17, 17))
        assert make_metadata(view=view).pml_thickness.tolist()[3:] == [1, 1, 1]

    def test_the_six_faces_are_independent(self, make_metadata, pml_grid, make_grid_view):
        """Expects a view clipped on one axis only to report a depth on that
        axis and zero on the others."""
        g = pml_grid(thickness=4, nx=20, ny=20, nz=20)
        view = make_grid_view(grid=g, start=(0, 6, 6), stop=(14, 14, 14))
        assert make_metadata(view=view).pml_thickness.tolist() == [4, 0, 0, 0, 0, 0]

    def test_the_result_is_int64(self, make_metadata, pml_grid):
        """Expects an integer array, since these are cell counts."""
        assert make_metadata(grid=pml_grid()).pml_thickness.dtype == np.int64


class TestSourceAndReceiverComment:
    def test_returns_none_for_an_empty_list(self, make_metadata):
        """Expects ``None`` rather than empty arrays, so nothing is written."""
        assert make_metadata().source_ids is None
        assert make_metadata().receiver_ids is None

    def test_records_receiver_names(self, make_metadata, make_view_grid, make_rx):
        """Expects the user's ``#rx`` labels."""
        g = make_view_grid()
        g.rxs = [make_rx(ID="antenna-A"), make_rx(ID="antenna-B")]
        assert make_metadata(grid=g).receiver_ids == ["antenna-A", "antenna-B"]

    def test_records_receiver_positions_in_metres(self, make_metadata, make_view_grid, make_rx):
        """Expects ``coord * grid.dl``, so ParaView places the marker at the
        physical location."""
        g = make_view_grid(dl=DL_ANISO)
        g.rxs = [make_rx(ID="a", position=(1, 2, 3))]
        assert make_metadata(grid=g).receiver_positions[0] == pytest.approx(
            [1 * DL_ANISO[0], 2 * DL_ANISO[1], 3 * DL_ANISO[2]]
        )

    def test_positions_are_one_row_per_object(self, make_metadata, make_view_grid, make_rx):
        """Expects an ``(n, 3)`` array."""
        g = make_view_grid()
        g.rxs = [make_rx(ID="a"), make_rx(ID="b"), make_rx(ID="c")]
        assert make_metadata(grid=g).receiver_positions.shape == (3, 3)

    def test_all_four_source_types_are_combined(self, make_metadata, make_view_grid):
        """Expects dipoles, voltage sources and transmission lines in one
        list — the file makes no distinction between them."""
        g = make_view_grid()
        g.hertziandipoles = [_FakeSource("hd")]
        g.magneticdipoles = [_FakeSource("md")]
        g.voltagesources = [_FakeSource("vs")]
        g.transmissionlines = [_FakeSource("tl")]
        assert make_metadata(grid=g).source_ids == ["hd", "md", "vs", "tl"]

    def test_sources_and_receivers_are_kept_separate(self, make_metadata, make_view_grid, make_rx):
        """Expects two independent groups, so ParaView can style them
        differently."""
        g = make_view_grid()
        g.hertziandipoles = [_FakeSource("src")]
        g.rxs = [make_rx(ID="rx")]
        meta = make_metadata(grid=g)
        assert meta.source_ids == ["src"]
        assert meta.receiver_ids == ["rx"]


class TestWriteToVtkhdf:
    def test_always_writes_the_four_core_fields(self, make_metadata):
        """Expects version, spacing, size and material list unconditionally."""
        handler = _RecordingHandler()
        make_metadata().write_to_vtkhdf(handler)
        assert {"gprMax_version", "dx_dy_dz", "nx_ny_nz", "material_ids"} <= set(handler.fields)

    def test_omits_the_pml_field_when_there_is_none(self, make_metadata):
        """Expects ``pml_thickness`` absent rather than zeroed."""
        handler = _RecordingHandler()
        make_metadata().write_to_vtkhdf(handler)
        assert "pml_thickness" not in handler.fields

    def test_writes_the_pml_field_when_present(self, make_metadata, pml_grid):
        """Expects the six depths written when slabs were built."""
        handler = _RecordingHandler()
        make_metadata(grid=pml_grid()).write_to_vtkhdf(handler)
        assert "pml_thickness" in handler.fields

    def test_omits_source_fields_when_there_are_none(self, make_metadata):
        """Expects both ``source_ids`` and ``sources`` absent — they are
        written as a pair or not at all."""
        handler = _RecordingHandler()
        make_metadata().write_to_vtkhdf(handler)
        assert "source_ids" not in handler.fields
        assert "sources" not in handler.fields

    def test_writes_source_fields_as_a_pair(self, make_metadata, make_view_grid):
        """Expects names and positions together, so neither is meaningful
        alone."""
        g = make_view_grid()
        g.hertziandipoles = [_FakeSource("src")]
        handler = _RecordingHandler()
        make_metadata(grid=g).write_to_vtkhdf(handler)
        assert {"source_ids", "sources"} <= set(handler.fields)

    def test_writes_receiver_fields_as_a_pair(self, make_metadata, make_view_grid, make_rx):
        """Expects the receiver equivalent."""
        g = make_view_grid()
        g.rxs = [make_rx(ID="a")]
        handler = _RecordingHandler()
        make_metadata(grid=g).write_to_vtkhdf(handler)
        assert {"receiver_ids", "receivers"} <= set(handler.fields)

    def test_materials_only_writes_nothing_extra(
        self, make_metadata, pml_grid, make_view_grid, make_rx
    ):
        """Expects exactly the four core fields even when the grid has PMLs,
        sources and receivers to report."""
        g = pml_grid()
        g.rxs = [make_rx(ID="a")]
        g.hertziandipoles = [_FakeSource("s")]
        handler = _RecordingHandler()
        make_metadata(grid=g, materials_only=True).write_to_vtkhdf(handler)
        assert set(handler.fields) == {"gprMax_version", "dx_dy_dz", "nx_ny_nz", "material_ids"}


class TestMpiMetadata:
    @pytest.fixture
    def mpi_view(self, make_mpi_grid, make_materials):
        from gprMax.geometry_outputs.grid_view import MPIGridView

        grid = make_mpi_grid(
            size=(8, 8, 8),
            negative_halo_offset=(0, 0, 0),
            arrays={"ID": np.ones((6, 9, 9, 9), dtype=np.uint32)},
        )
        grid.materials = make_materials(2)
        grid.pmls = {
            "slabs": [],
            "thickness": dict.fromkeys(["x0", "y0", "z0", "xmax", "ymax", "zmax"], 0),
        }
        grid.nx, grid.ny, grid.nz = 8, 8, 8
        grid.rxs = []
        grid.hertziandipoles = []
        grid.magneticdipoles = []
        grid.voltagesources = []
        grid.transmissionlines = []
        return MPIGridView(grid, 0, 0, 0, 8, 8, 8)

    def test_extends_the_serial_metadata(self):
        """Expects only the three rank-dependent methods to be overridden."""
        assert issubclass(MPIMetadata, Metadata)
        overrides = {n for n in MPIMetadata.__dict__ if not n.startswith("__")}
        assert overrides == {"nx_ny_nz_comment", "pml_gv_comment", "srcs_rx_gv_comment"}

    def test_domain_size_is_the_global_one(self, mpi_view):
        """Expects ``grid.global_size`` rather than this rank's local size, so
        every rank writes the same value."""
        assert MPIMetadata(mpi_view).nx_ny_nz.tolist() == [8, 8, 8]

    def test_pml_depths_are_reduced_across_ranks(self, mpi_view):
        """Expects an ``Allgather`` followed by an elementwise maximum: a rank
        that sees no PML must not veto a rank that does.

        With one rank the maximum is that rank's own value, and an all-zero
        result still collapses to ``None``."""
        assert MPIMetadata(mpi_view).pml_thickness is None

    def test_a_visible_pml_survives_the_reduction(self, mpi_view):
        """Expects a non-zero depth to be reported after the reduction."""
        mpi_view.grid.pmls["slabs"] = ["slab"]
        mpi_view.grid.pmls["thickness"]["x0"] = 3
        assert MPIMetadata(mpi_view).pml_thickness.tolist()[0] == 3

    def test_sources_are_gathered_and_sorted_by_name(self, mpi_view):
        """Expects ``allgather`` of a name-to-position dict, then a sort — so
        every rank writes the same order regardless of who owns what."""
        mpi_view.grid.hertziandipoles = [_FakeSource("zulu"), _FakeSource("alpha")]
        meta = MPIMetadata(mpi_view)
        assert meta.source_ids == ["alpha", "zulu"]

    def test_positions_are_converted_to_global_coordinates(self, mpi_view):
        """Expects ``local_to_global_coordinate`` applied before scaling, so a
        rank's local index maps to the right place in the whole model."""
        mpi_view.grid.hertziandipoles = [_FakeSource("a", coord=(1, 1, 1))]
        meta = MPIMetadata(mpi_view)
        assert meta.source_positions[0] == pytest.approx([101 * DL] * 3)

    def test_an_empty_list_still_gives_none(self, mpi_view):
        """Expects ``None`` when no rank contributed anything, matching the
        serial behaviour so the field is omitted."""
        assert MPIMetadata(mpi_view).source_ids is None


class TestSaveGeometryViews:
    def test_prepares_and_writes_each_view(self):
        """Expects ``set_filename``, ``prep_vtk`` then ``write_vtk``, in that
        order — the filename must exist before the writer opens it."""
        view = _SpyView()
        save_geometry_views([view])
        assert view.calls == ["set_filename", "prep_vtk", "write_vtk"]

    def test_handles_several_views(self):
        """Expects every view in the list to be written."""
        views = [_SpyView(), _SpyView(), _SpyView()]
        save_geometry_views(views)
        assert all(v.calls[-1] == "write_vtk" for v in views)

    def test_an_empty_list_is_a_no_op(self):
        """Expects no error for a model with no geometry views."""
        assert save_geometry_views([]) is None

    def test_logs_blank_spacer_lines(self, caplog):
        """Expects two ``info`` records framing the progress bars — the only
        output this orchestrator produces."""
        with caplog.at_level(logging.INFO, logger="gprMax.geometry_outputs.geometry_views"):
            save_geometry_views([_SpyView()])
        assert len(caplog.records) == 2


# --- Test doubles ------------------------------------------------------------


class _ConcreteView(GeometryView):
    """Minimal concrete subclass, so the abstract base can be constructed."""

    def prep_vtk(self):
        pass

    def write_vtk(self):
        pass


class _FakeSource:
    def __init__(self, ID, coord=(1, 1, 1)):
        self.ID = ID
        self.coord = np.array(coord, dtype=np.int32)


class _RecordingHandler:
    """Captures the field data a ``Metadata`` writes."""

    def __init__(self):
        self.fields = {}

    def add_field_data(self, name, data, **kwargs):
        self.fields[name] = data


class _SpyView:
    """Records the call order ``save_geometry_views`` drives."""

    nbytes = 16
    filename = Path("spy.vtkhdf")

    def __init__(self):
        self.calls = []

    def set_filename(self):
        self.calls.append("set_filename")

    def prep_vtk(self):
        self.calls.append("prep_vtk")

    def write_vtk(self):
        self.calls.append("write_vtk")


pytestmark = pytest.mark.unit
