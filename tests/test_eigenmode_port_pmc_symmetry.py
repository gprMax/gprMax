"""Eigenmode ports in models reduced by a PMC symmetry plane.

A rectangular guide carrying TE10 is symmetric about the mid-plane across its
wide axis, which is a magnetic wall. Each test compares a model cut there by
a PMC symmetry boundary with the model it stands for.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import gprMax
from gprMax.grid.fdtd_grid import FDTDGrid
from gprMax.virtual_waveguide import VirtualWaveguide

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


def _virtual_guide_scene(original, *, pmc, shielded, face="x0", shield_material="pec"):
    """Put a mixed PEC/impedance guide's window edge inside an opaque wall.

    The left/right PEC block can instead contain excluded SIBC interior.
    Opening the block exposes an active aperture to the symmetry plane.
    """

    def scene(**kwargs):
        result = original(**kwargs)
        # SIBC contact requires an explicitly declared symmetry face, including
        # the PEC control. Keep PEC padding at the other transverse faces.
        if pmc or shield_material != "pec":
            result.add(gprMax.SymmetryBoundary(face=face, type="pmc" if pmc else "pec"))
        window = (0, 14) if face == "x0" else (2, 16)
        wall = (0, 3) if face == "x0" else (13, 16)
        for obj in result.grid_objects:
            if isinstance(obj, gprMax.EigenmodePort):
                plane = obj.kwargs["p1"][2]
                obj.kwargs.update(
                    p1=(window[0] * DL, 2 * DL, plane),
                    p2=(window[1] * DL, 14 * DL, plane),
                )
        if shield_material != "pec":
            result.add(gprMax.Box(
                p1=(wall[0] * DL, DL, 0), p2=(wall[1] * DL, 15 * DL, 72 * DL),
                material_id=shield_material,
            ))
        if not shielded:
            result.add(gprMax.Box(
                p1=(wall[0] * DL, 4 * DL, 0), p2=(wall[1] * DL, 12 * DL, 72 * DL),
                material_id="free_space",
            ))
        return result

    return scene


@pytest.mark.parametrize("face", ("x0", "xmax"))
@pytest.mark.parametrize("shield_material", ("pec", "wall"))
def test_virtual_waveguide_accepts_inactive_pmc_edge(tmp_path, monkeypatch, face, shield_material):
    """PEC constraints and excluded SIBC interiors have no live aperture E."""
    from testing.validation.impedance_surface import virtual_waveguide as validation

    original = validation.guide_scene
    options = dict(virtual=True, active=True, resistance=1000.0, steps=300)
    geometry = dict(face=face, shield_material=shield_material, shielded=True)

    monkeypatch.setattr(
        validation, "guide_scene", _virtual_guide_scene(original, pmc=False, **geometry)
    )
    control, control_grid = validation.run_guide(tmp_path / "pec", **options)
    monkeypatch.setattr(
        validation, "guide_scene", _virtual_guide_scene(original, pmc=True, **geometry)
    )
    actual, grid = validation.run_guide(tmp_path / "pmc", **options)

    guide = grid.virtual_waveguides[0]
    assert guide.port.pmc_symmetry_edges(grid) == ("u_min" if face == "x0" else "u_max",)
    if shield_material == "wall":
        ids = guide._component_cross_section()
        side = 0 if face == "x0" else -1
        for component in (1, 2):
            assert all(
                getattr(grid.materials[int(value)], "impedance_role", None) == "volume-void"
                for value in ids[component, side, 1:-1]
            )
    scale = np.max(np.abs(control))
    assert scale > 0
    assert np.max(np.abs(actual - control)) / scale < 1e-12
    assert guide.port.mode_solver.modal_complex_neff == pytest.approx(
        control_grid.virtual_waveguides[0].port.mode_solver.modal_complex_neff, rel=1e-12
    )


@pytest.mark.parametrize("face", ("x0", "xmax"))
@pytest.mark.parametrize("shield_material", ("pec", "wall"))
def test_virtual_waveguide_rejects_aperture_cut_by_pmc(tmp_path, monkeypatch, face, shield_material):
    from testing.validation.impedance_surface import virtual_waveguide as validation

    monkeypatch.setattr(
        validation,
        "guide_scene",
        _virtual_guide_scene(
            validation.guide_scene, pmc=True, shielded=False, face=face, shield_material=shield_material,
        ),
    )
    with pytest.raises(ValueError, match="aperture is cut by a PMC symmetry boundary"):
        validation.run_guide(
            tmp_path / "cut", virtual=True, active=True, resistance=1000.0, steps=300
        )


@pytest.mark.parametrize("normal", range(3))
@pytest.mark.parametrize("local_axis,high", ((0, False), (0, True), (1, False), (1, True)))
@pytest.mark.parametrize("longitudinal", (False, True))
@pytest.mark.parametrize("active_id", (2, 3), ids=("surface-hold", "dielectric"))
def test_one_live_sample_on_pmc_edge_is_rejected(normal, local_axis, high, longitudinal, active_id):
    """In particular, a held SIBC surface sample is not an excluded interior."""
    guide = VirtualWaveguide.__new__(VirtualWaveguide)
    guide.mpi = False
    guide.normal_axis = normal
    guide.transverse_axes = tuple(axis for axis in range(3) if axis != normal)
    guide.u0 = guide.v0 = 0
    guide.u1 = guide.nu = 4
    guide.v1 = guide.nv = 3
    guide.port = SimpleNamespace(invariant_axis=None)
    axis = guide.transverse_axes[local_axis]
    size = np.full(3, 6)
    size[list(guide.transverse_axes)] = (guide.nu, guide.nv)
    guide.main_grid = SimpleNamespace(
        size=size,
        symmetry_boundaries={"xyz"[axis] + ("max" if high else "0"): "pmc"},
        materials=[
            SimpleNamespace(numID=0, is_pec=True),
            SimpleNamespace(numID=1, is_pec=False, impedance_role="volume-void"),
            SimpleNamespace(numID=2, is_pec=False, impedance_role="surface-hold"),
            SimpleNamespace(numID=3, is_pec=False),
        ],
    )
    ids = np.ones((6, guide.nu + 1, guide.nv + 1), dtype=np.uint32)
    with patch.object(guide, "_component_cross_section", return_value=ids):
        guide._validate_symmetry()  # all samples excluded; no active aperture
        component = normal if longitudinal else guide.transverse_axes[1 - local_axis]
        point = [1, 1]
        point[local_axis] = (guide.nu, guide.nv)[local_axis] if high else 0
        ids[component, point[0], point[1]] = active_id
        with pytest.raises(ValueError, match="aperture is cut by a PMC symmetry boundary"):
            guide._validate_symmetry()
