"""Eigenmode ports in models reduced by a PMC symmetry plane.

A rectangular guide carrying TE10 is symmetric about the mid-plane across its
wide axis, which is a magnetic wall. Each test compares a model cut there by
a PMC symmetry boundary with the model it stands for.
"""

from unittest.mock import patch

import numpy as np
import pytest

import gprMax
from gprMax.grid.fdtd_grid import FDTDGrid

pytestmark = pytest.mark.integration

DL = 1e-3
FREQUENCY = 10e9
NY, NZ = 18, 28
PLANE = 14


def _guide(wall, *, half_face=None, window_in_wall=2, tracking="legacy"):
    """A 20 x 10 mm guide along z with 3 mm walls, or the half beside ``half_face``.

    The full guide spans x = 4..24 cells. The half keeps the cells beside the
    PMC plane at its mid-plane, on domain face ``x0`` or ``xmax``.
    """
    if half_face is None:
        nx, walls, interior = 28, (1, 27), (4, 24)
        window = (interior[0] - window_in_wall, interior[1] + window_in_wall)
    elif half_face == "x0":
        nx, walls, interior = 14, (0, 13), (0, 10)
        window = (0, interior[1] + window_in_wall)
    else:
        nx, walls, interior = 14, (1, 14), (4, 14)
        window = (interior[0] - window_in_wall, 14)

    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=(nx * DL, NY * DL, NZ * DL)),
        gprMax.Discretisation(p1=(DL, DL, DL)),
        gprMax.TimeWindow(iterations=200),
        gprMax.OMPThreads(1),
        gprMax.PMLThickness(thickness=(0, 0, 8, 0, 0, 8)),
        gprMax.Waveform(wave_type="contsine", amp=1, freq=FREQUENCY, id="pulse"),
    ):
        scene.add(obj)
    if wall != "pec":
        scene.add(gprMax.SurfaceImpedance(id=wall, resistance=20.0))
    if half_face is not None:
        scene.add(gprMax.SymmetryBoundary(face=half_face, type="pmc"))
    scene.add(gprMax.Box(
        p1=(walls[0] * DL, 1 * DL, 0), p2=(walls[1] * DL, 17 * DL, NZ * DL), material_id=wall
    ))
    scene.add(gprMax.Box(
        p1=(interior[0] * DL, 4 * DL, 0), p2=(interior[1] * DL, 14 * DL, NZ * DL),
        material_id="free_space",
    ))
    scene.add(gprMax.EigenmodeBand(id="band", fmin=FREQUENCY, fmax=FREQUENCY, points=1))
    scene.add(gprMax.EigenmodePort(
        port=1,
        p1=(window[0] * DL, (4 - window_in_wall) * DL, PLANE * DL),
        p2=(window[1] * DL, (14 + window_in_wall) * DL, PLANE * DL),
        direction="+", modes=(1,), anchors=(FREQUENCY,), plot_fields=False, tracking=tracking,
    ))
    scene.add(gprMax.EigenmodeExcitation(port=1, mode=1, waveform="pulse", plot_waveform=False))
    return scene


def _port_source(scene, path):
    grids = []
    original = FDTDGrid.build

    def capture(grid):
        original(grid)
        grids.append(grid)

    with patch.object(FDTDGrid, "build", capture):
        gprMax.run(
            scenes=[scene], outputfile=path, cpu_precision="double",
            hide_progress_bars=True, log_level=40,
        )
    return grids[0].eigenmodesources[0]


@pytest.mark.parametrize("half_face", ("x0", "xmax"))
def test_impedance_wall_half_guide_matches_full_guide(tmp_path, half_face):
    """Surface-impedance rows on the PMC cut keep their equations."""
    full = _port_source(_guide("wall"), tmp_path / "full")
    half = _port_source(_guide("wall", half_face=half_face), tmp_path / "half")

    # The top and bottom walls each cross the symmetry plane at one E_w sample.
    edge = 0 if half_face == "x0" else half.mode_solver.Nu
    on_cut = [
        row for row in half.fdfd_surface_boundary.rows
        if row.electric_axis == 2 and row.electric_index[0] == edge
    ]
    assert len(on_cut) == 2
    assert 2 * half.surface_impedance_fdfd_edges == full.surface_impedance_fdfd_edges + 2

    full_neff = complex(full.mode_solver.modal_complex_neff)
    half_neff = complex(half.mode_solver.modal_complex_neff)
    assert full_neff.imag < -1e-2  # the walls are lossy
    assert half_neff.real == pytest.approx(full_neff.real, rel=1e-10)
    assert half_neff.imag == pytest.approx(full_neff.imag, rel=1e-10)


def test_closed_half_guide_is_verified_like_the_full_guide(tmp_path):
    """A PMC symmetry edge counts as a physical wall for automatic verification."""
    options = dict(window_in_wall=0, tracking="auto")
    full = _port_source(_guide("pec", **options), tmp_path / "full")
    half = _port_source(_guide("pec", half_face="x0", **options), tmp_path / "half")
    full_record, half_record = full.anchor_quality_diagnostics[0], half.anchor_quality_diagnostics[0]

    assert half.mode_solver.pmc_edges == ("u_min",)
    assert full_record["physical_enclosure"] and half_record["physical_enclosure"]
    assert "insufficient_non_pml_exterior_geometry" not in half_record["reasons"]
    for key in ("confinement", "artifact", "reasons"):
        assert half_record[key] == full_record[key]
    assert [check["kind"] for check in half_record["verification"]] == ["mesh"]
    assert complex(half.mode_solver.modal_complex_neff) == pytest.approx(
        complex(full.mode_solver.modal_complex_neff), rel=1e-10
    )


def _virtual_guide_scene(original, *, pmc, shielded):
    """The mixed PEC/impedance validation guide with its port window on face x0.

    A PEC block fills x = 0..3 cells, so the window edge on that face is
    covered by a physical wall unless the block is opened.
    """

    def scene(**kwargs):
        result = original(**kwargs)
        if pmc:
            result.add(gprMax.SymmetryBoundary(face="x0", type="pmc"))
        for obj in result.grid_objects:
            if isinstance(obj, gprMax.EigenmodePort):
                plane = obj.kwargs["p1"][2]
                obj.kwargs.update(p1=(0, 2 * DL, plane), p2=(14 * DL, 14 * DL, plane))
        if not shielded:
            result.add(gprMax.Box(
                p1=(0, 4 * DL, 0), p2=(3 * DL, 12 * DL, 72 * DL), material_id="free_space"
            ))
        return result

    return scene


def test_virtual_waveguide_accepts_pmc_edge_covered_by_pec(tmp_path, monkeypatch):
    """A window edge on a PMC face behind a physical PEC wall is a PEC edge."""
    from testing.validation.impedance_surface import virtual_waveguide as validation

    original = validation.guide_scene
    options = dict(virtual=True, active=True, resistance=1000.0, steps=300)

    monkeypatch.setattr(
        validation, "guide_scene", _virtual_guide_scene(original, pmc=False, shielded=True)
    )
    control, control_grid = validation.run_guide(tmp_path / "pec", **options)
    monkeypatch.setattr(
        validation, "guide_scene", _virtual_guide_scene(original, pmc=True, shielded=True)
    )
    actual, grid = validation.run_guide(tmp_path / "pmc", **options)

    guide = grid.virtual_waveguides[0]
    assert guide.port.pmc_symmetry_edges(grid) == ("u_min",)
    scale = np.max(np.abs(control))
    assert scale > 0
    assert np.max(np.abs(actual - control)) / scale < 1e-12
    assert guide.port.mode_solver.modal_complex_neff == pytest.approx(
        control_grid.virtual_waveguides[0].port.mode_solver.modal_complex_neff, rel=1e-12
    )


def test_virtual_waveguide_rejects_aperture_cut_by_pmc(tmp_path, monkeypatch):
    from testing.validation.impedance_surface import virtual_waveguide as validation

    monkeypatch.setattr(
        validation,
        "guide_scene",
        _virtual_guide_scene(validation.guide_scene, pmc=True, shielded=False),
    )
    with pytest.raises(ValueError, match="aperture is cut by a PMC symmetry boundary"):
        validation.run_guide(
            tmp_path / "cut", virtual=True, active=True, resistance=1000.0, steps=300
        )
