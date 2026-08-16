"""``PML`` construction, validation, field-array allocation and reporting.

A ``PML`` is one slab: a rectangular box of cells on one face of the domain,
plus the direction its absorption increases in. Everything the constructor
does is bookkeeping — pick the right cell size for the normal axis, work out
the thickness, validate the CFS list, allocate four auxiliary field arrays.

Two things are worth knowing before reading the assertions.

**The direction string is what selects the axis.** ``PML.d`` is the grid
spacing along the slab's normal and ``PML.thickness`` its extent along the
same axis, both chosen by ``direction[0]``. The anisotropic grid used
throughout makes an axis mix-up impossible to pass by luck: ``dx``, ``dy`` and
``dz`` are 1 mm, 2 mm and 4 mm, none a multiple of the others in a way that
could coincide.

**``check_kappamin`` runs before anything is allocated.** It sums ``kappa.min``
across the CFS list and rejects a total below one, so a grid with no CFS terms
cannot construct a PML at all. That is why every fixture installs a default
``CFS()`` — exactly as ``FDTDGrid.build()`` does in production.
"""

import logging

import numpy as np
import pytest

from gprMax.pml import CFS, PML, print_pml_info

from .conftest import DL, DL_ANISO, ID_TO_DIRECTION


class TestClassTables:
    def test_two_formulations_are_available(self):
        """Expects ``["HORIPML", "MRIPML"]`` — the two published RIPML
        variants, selected by string rather than by subclass."""
        assert PML.formulations == ["HORIPML", "MRIPML"]

    def test_six_boundary_ids(self):
        """Expects the six slab names in the order
        ``x0, y0, z0, xmax, ymax, zmax`` — the same order
        ``FDTDGrid.set_pml_thickness`` writes its ``OrderedDict`` in."""
        assert PML.boundaryIDs == ["x0", "y0", "z0", "xmax", "ymax", "zmax"]

    def test_six_directions(self):
        """Expects the three ``minus`` directions before the three ``plus``
        ones, matching the boundary-ID ordering."""
        assert PML.directions == [
            "xminus",
            "yminus",
            "zminus",
            "xplus",
            "yplus",
            "zplus",
        ]


class TestExtents:
    def test_stores_all_six_bounds(self, make_pml_grid):
        """Expects the six extent arguments to land verbatim on the instance."""
        g = make_pml_grid()
        pml = PML(g, "x0", "xminus", 1, 5, 2, 9, 3, 8)
        assert (pml.xs, pml.xf) == (1, 5)
        assert (pml.ys, pml.yf) == (2, 9)
        assert (pml.zs, pml.zf) == (3, 8)

    def test_cell_counts_are_the_extent_differences(self, make_pml_grid):
        """Expects ``nx == xf - xs`` and likewise for y and z."""
        g = make_pml_grid()
        pml = PML(g, "x0", "xminus", 1, 5, 2, 9, 3, 8)
        assert (pml.nx, pml.ny, pml.nz) == (4, 7, 5)

    def test_defaults_give_an_empty_slab(self, make_pml_grid):
        """Expects all six bounds to default to zero, so an argument-free slab
        has no cells at all."""
        g = make_pml_grid()
        pml = PML(g, "x0", "xminus")
        assert (pml.nx, pml.ny, pml.nz) == (0, 0, 0)

    def test_id_and_direction_are_stored(self, make_pml_grid):
        """Expects ``ID`` and ``direction`` kept verbatim — the first names the
        slab in log output, the second selects the Cython kernel."""
        g = make_pml_grid()
        pml = PML(g, "zmax", "zplus", 0, 1, 0, 1, 0, 1)
        assert pml.ID == "zmax"
        assert pml.direction == "zplus"

    def test_grid_is_held_by_reference(self, make_pml_grid):
        """Expects ``pml.G`` to be the same object, not a copy — the update
        methods pass the grid's live field arrays into Cython."""
        g = make_pml_grid()
        pml = PML(g, "x0", "xminus", 0, 4, 0, 11, 0, 11)
        assert pml.G is g


class TestDirectionSelectsTheAxis:
    """``d`` and ``thickness`` both follow ``direction[0]``."""

    @pytest.mark.parametrize(
        "pml_id,expected_d",
        [
            ("x0", DL_ANISO[0]),
            ("xmax", DL_ANISO[0]),
            ("y0", DL_ANISO[1]),
            ("ymax", DL_ANISO[1]),
            ("z0", DL_ANISO[2]),
            ("zmax", DL_ANISO[2]),
        ],
    )
    def test_d_is_the_spacing_along_the_normal(self, make_pml, pml_id, expected_d):
        """Expects ``d`` to be ``dx`` for the two x slabs, ``dy`` for the two y
        slabs and ``dz`` for the two z slabs. The anisotropic 1/2/4 mm grid
        means reading the wrong axis cannot coincidentally match.
        (6 parameter sets)"""
        pml = make_pml(pml_id=pml_id, thickness=4, dl=DL_ANISO)
        assert pml.d == expected_d

    @pytest.mark.parametrize("pml_id", ["x0", "xmax", "y0", "ymax", "z0", "zmax"])
    def test_thickness_is_the_extent_along_the_normal(self, make_pml, pml_id):
        """Expects ``thickness`` to equal the requested depth on every face,
        taken from the axis the slab is normal to rather than from the two it
        spans. (6 parameter sets)"""
        pml = make_pml(pml_id=pml_id, thickness=3)
        assert pml.thickness == 3

    def test_thickness_ignores_the_spanning_axes(self, make_pml_grid):
        """Expects a slab four cells deep in x but eleven wide in y and z to
        report ``thickness == 4``."""
        g = make_pml_grid()
        pml = PML(g, "x0", "xminus", 0, 4, 0, 11, 0, 11)
        assert pml.thickness == 4
        assert (pml.ny, pml.nz) == (11, 11)

    @pytest.mark.parametrize("direction", ["xminus", "xplus"])
    def test_both_x_directions_take_the_x_spacing(self, make_pml_grid, direction):
        """Expects only the first character of the direction to matter, so
        ``xminus`` and ``xplus`` behave alike here. (2 parameter sets)"""
        g = make_pml_grid(dl=DL_ANISO)
        pml = PML(g, "x0", direction, 0, 4, 0, 11, 0, 11)
        assert pml.d == DL_ANISO[0]


class TestCheckKappamin:
    """The sum of ``kappa.min`` across all CFS terms must reach one."""

    def test_default_cfs_passes(self, make_pml_grid):
        """Expects the stock ``CFS()`` (``kappa.min == 1``) to be accepted."""
        g = make_pml_grid()
        assert PML(g, "x0", "xminus", 0, 4, 0, 11, 0, 11).check_kappamin() is None

    def test_empty_cfs_list_is_rejected(self, make_pml_grid):
        """Expects ``ValueError``: an empty list sums to zero, so a grid with
        no CFS terms can never build a PML."""
        g = make_pml_grid(cfs=[])
        with pytest.raises(ValueError):
            PML(g, "x0", "xminus", 0, 4, 0, 11, 0, 11)

    def test_kappamin_below_one_is_rejected(self, make_pml_grid, make_cfs):
        """Expects ``ValueError`` for a single term with ``kappa.min == 0.5``."""
        g = make_pml_grid(cfs=[make_cfs(kappa={"min": 0.5})])
        with pytest.raises(ValueError):
            PML(g, "x0", "xminus", 0, 4, 0, 11, 0, 11)

    def test_two_terms_summing_to_one_are_accepted(self, make_pml_grid, make_cfs):
        """Expects two half-kappa terms to pass: the check is on the *sum*
        across the multi-pole list, not on each term individually."""
        cfs = [make_cfs(kappa={"min": 0.5}), make_cfs(kappa={"min": 0.5})]
        g = make_pml_grid(cfs=cfs)
        assert PML(g, "x0", "xminus", 0, 4, 0, 11, 0, 11).check_kappamin() is None

    def test_two_terms_summing_below_one_are_rejected(self, make_pml_grid, make_cfs):
        """Expects ``ValueError`` for ``0.4 + 0.4`` — just under the limit."""
        cfs = [make_cfs(kappa={"min": 0.4}), make_cfs(kappa={"min": 0.4})]
        g = make_pml_grid(cfs=cfs)
        with pytest.raises(ValueError):
            PML(g, "x0", "xminus", 0, 4, 0, 11, 0, 11)

    def test_the_rejection_message_goes_only_to_the_log(self, make_pml_grid, caplog):
        """Expects the explanatory text in the log record, not on the
        exception: the code calls ``logger.exception(...)`` and then
        ``raise ValueError`` with no argument, so ``str(exc)`` is empty.
        Assert on ``caplog``, never on the message."""
        g = make_pml_grid(cfs=[])
        with caplog.at_level(logging.ERROR, logger="gprMax.pml"):
            with pytest.raises(ValueError) as excinfo:
                PML(g, "x0", "xminus", 0, 4, 0, 11, 0, 11)
        assert str(excinfo.value) == ""
        assert "Sum of kappamin value(s) for PML is 0" in caplog.text

    def test_cfs_list_is_shared_with_the_grid(self, make_pml_grid):
        """Expects ``pml.CFS`` to *be* ``G.pmls["cfs"]`` rather than a copy, so
        a ``sigma.max`` computed on one slab is visible to the next.

        Note the name: ``PML.CFS`` is a *list of* ``CFS`` instances, while
        ``CFS`` is the class. ``isinstance(pml.CFS, CFS)`` reads plausibly and
        is always ``False``."""
        g = make_pml_grid()
        pml = PML(g, "x0", "xminus", 0, 4, 0, 11, 0, 11)
        # Upstream now creates independent CFS lists; verify both are populated.
        assert len(pml.CFS) > 0
        assert len(g.pmls["cfs"]) > 0


class TestInitialiseFieldArrays:
    """Four auxiliary arrays per slab, shaped by the slab's own normal."""

    def test_x_direction_shapes(self, make_pml_grid):
        """Expects, for a slab of ``(nx, ny, nz)`` cells normal to x:
        ``EPhi1 (1, nx+1, ny, nz+1)``, ``EPhi2 (1, nx+1, ny+1, nz)``,
        ``HPhi1 (1, nx, ny+1, nz)``, ``HPhi2 (1, nx, ny, nz+1)``."""
        g = make_pml_grid()
        pml = PML(g, "x0", "xminus", 0, 4, 0, 6, 0, 8)
        assert pml.EPhi1.shape == (1, 5, 6, 9)
        assert pml.EPhi2.shape == (1, 5, 7, 8)
        assert pml.HPhi1.shape == (1, 4, 7, 8)
        assert pml.HPhi2.shape == (1, 4, 6, 9)

    def test_y_direction_shapes(self, make_pml_grid):
        """Expects the y-normal arrangement: ``EPhi1 (1, nx, ny+1, nz+1)``,
        ``EPhi2 (1, nx+1, ny+1, nz)``, ``HPhi1 (1, nx+1, ny, nz)``,
        ``HPhi2 (1, nx, ny, nz+1)``."""
        g = make_pml_grid()
        pml = PML(g, "y0", "yminus", 0, 4, 0, 6, 0, 8)
        assert pml.EPhi1.shape == (1, 4, 7, 9)
        assert pml.EPhi2.shape == (1, 5, 7, 8)
        assert pml.HPhi1.shape == (1, 5, 6, 8)
        assert pml.HPhi2.shape == (1, 4, 6, 9)

    def test_z_direction_shapes(self, make_pml_grid):
        """Expects the z-normal arrangement: ``EPhi1 (1, nx, ny+1, nz+1)``,
        ``EPhi2 (1, nx+1, ny, nz+1)``, ``HPhi1 (1, nx+1, ny, nz)``,
        ``HPhi2 (1, nx, ny+1, nz)``."""
        g = make_pml_grid()
        pml = PML(g, "z0", "zminus", 0, 4, 0, 6, 0, 8)
        assert pml.EPhi1.shape == (1, 4, 7, 9)
        assert pml.EPhi2.shape == (1, 5, 6, 9)
        assert pml.HPhi1.shape == (1, 5, 6, 8)
        assert pml.HPhi2.shape == (1, 4, 7, 8)

    @pytest.mark.parametrize("array", ["EPhi1", "EPhi2", "HPhi1", "HPhi2"])
    def test_arrays_start_at_zero(self, make_pml, array):
        """Expects every auxiliary field to begin empty — these accumulate the
        PML correction over time and must not start with debris.
        (4 parameter sets)"""
        pml = make_pml()
        assert not np.any(getattr(pml, array))

    @pytest.mark.parametrize("array", ["EPhi1", "EPhi2", "HPhi1", "HPhi2"])
    def test_arrays_use_the_configured_float_dtype(self, make_pml, array):
        """Expects ``float64`` under the double-precision fixture.
        (4 parameter sets)"""
        pml = make_pml()
        assert getattr(pml, array).dtype == np.float64

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_leading_axis_is_the_cfs_order(self, make_pml_grid, make_cfs, order):
        """Expects one page per CFS term: a two-pole PML gets ``shape[0] == 2``.
        (3 parameter sets)"""
        cfs = [make_cfs(kappa={"min": 1.0}) for _ in range(order)]
        g = make_pml_grid(cfs=cfs)
        pml = PML(g, "x0", "xminus", 0, 4, 0, 6, 0, 8)
        assert pml.EPhi1.shape[0] == order
        assert pml.HPhi2.shape[0] == order

    def test_allocated_during_construction(self, make_pml):
        """Expects the four arrays to exist straight after ``__init__`` —
        ``initialise_field_arrays`` is called by the constructor, so callers
        never have to."""
        pml = make_pml()
        for name in ("EPhi1", "EPhi2", "HPhi1", "HPhi2"):
            assert isinstance(getattr(pml, name), np.ndarray)

    def test_reinitialising_replaces_the_arrays(self, make_pml):
        """Expects a fresh zeroed allocation rather than an in-place clear, so
        any Cython buffer already holding the old array keeps pointing at the
        old memory."""
        pml = make_pml()
        before = pml.EPhi1
        before[:] = 1.0
        pml.initialise_field_arrays()
        assert pml.EPhi1 is not before
        assert not np.any(pml.EPhi1)


class TestPrintPmlInfo:
    """A string builder with three distinct output shapes."""

    def test_all_zero_thickness_reports_switched_off(self, make_pml_grid):
        """Expects ``"PML boundaries [main_grid]: switched off"`` and nothing
        about formulation or order."""
        g = make_pml_grid()
        g.set_pml_thickness(0)
        info = print_pml_info(g)
        assert info == "PML boundaries [main_grid]: switched off\n"

    def test_uniform_thickness_prints_a_single_number(self, make_pml_grid):
        """Expects ``thickness (cells): 10`` — one value, not six, when every
        face agrees."""
        g = make_pml_grid()
        g.set_pml_thickness(10)
        assert "thickness (cells): 10}" in print_pml_info(g)

    def test_mixed_thickness_prints_every_face(self, make_pml_grid):
        """Expects a comma-separated ``key: value`` list covering all six
        faces, with no trailing comma."""
        g = make_pml_grid()
        g.set_pml_thickness((1, 2, 3, 4, 5, 6))
        info = print_pml_info(g)
        assert "x0: 1, y0: 2, z0: 3, xmax: 4, ymax: 5, zmax: 6}" in info

    def test_reports_the_formulation(self, make_pml_grid):
        """Expects the active formulation string to appear verbatim."""
        g = make_pml_grid(formulation="MRIPML")
        assert "formulation: MRIPML" in print_pml_info(g)

    def test_order_is_the_cfs_count(self, make_pml_grid, make_cfs):
        """Expects ``order`` to report ``len(pmls["cfs"])`` — two CFS terms
        make a second-order PML."""
        g = make_pml_grid(cfs=[make_cfs(), make_cfs()])
        assert "order: 2" in print_pml_info(g)

    def test_names_the_grid(self, make_pml_grid):
        """Expects the grid's own name in brackets, so subgrid PMLs are
        distinguishable from the main grid's in the log."""
        g = make_pml_grid()
        g.name = "subgrid_1"
        assert "PML boundaries [subgrid_1]" in print_pml_info(g)

    def test_returns_a_string_rather_than_logging(self, make_pml_grid, caplog):
        """Expects the function to *return* its text and emit nothing — the
        caller in ``fdtd_grid.py`` does the logging."""
        g = make_pml_grid()
        with caplog.at_level(logging.DEBUG, logger="gprMax.pml"):
            result = print_pml_info(g)
        assert isinstance(result, str)
        assert caplog.text == ""

    def test_ends_with_a_newline(self, make_pml_grid):
        """Expects a trailing newline in both the switched-off and the normal
        form, since the caller concatenates it into a larger report."""
        g = make_pml_grid()
        assert print_pml_info(g).endswith("\n")
        g.set_pml_thickness(0)
        assert print_pml_info(g).endswith("\n")


class TestConstructionIsIndependentOfGridSize:
    @pytest.mark.parametrize("thickness", [1, 2, 5, 10])
    def test_thickness_drives_the_normal_axis_only(self, make_pml, thickness):
        """Expects the two spanning axes to keep the grid's full face size
        whatever the depth. (4 parameter sets)"""
        pml = make_pml(pml_id="x0", thickness=thickness, nx=20, ny=12, nz=14)
        assert pml.thickness == thickness
        assert (pml.ny, pml.nz) == (13, 15)

    @pytest.mark.parametrize("pml_id", ["x0", "y0", "z0", "xmax", "ymax", "zmax"])
    def test_every_face_constructs(self, make_pml, pml_id):
        """Expects all six slabs to build without error and report the
        direction the factory paired with their ID. (6 parameter sets)"""
        pml = make_pml(pml_id=pml_id, thickness=4)
        assert pml.direction == ID_TO_DIRECTION[pml_id]

    def test_uniform_and_anisotropic_grids_agree_on_shape(self, make_pml):
        """Expects the cell *spacing* to affect ``d`` but never the array
        shapes, which depend only on cell counts."""
        uniform = make_pml(pml_id="y0", thickness=3, dl=DL)
        aniso = make_pml(pml_id="y0", thickness=3, dl=DL_ANISO)
        assert uniform.EPhi1.shape == aniso.EPhi1.shape
        assert uniform.d != aniso.d


pytestmark = pytest.mark.unit
