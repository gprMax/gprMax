# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

"""PEC precedence and compatibility with the existing PMC volume stencil."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import gprMax
import gprMax.config as config
import gprMax.impedance_surfaces as implementation


DL = 0.001
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("suppress_sibc_fit_plots")]
EDGE = np.array((4, 4, 4))
# Cyclic quadrants in the two transverse directions of each E component.
OFFSETS = (
    ((0, -1, -1), (0, 0, -1), (0, 0, 0), (0, -1, 0)),
    ((-1, 0, -1), (-1, 0, 0), (0, 0, 0), (0, 0, -1)),
    ((-1, -1, 0), (0, -1, 0), (0, 0, 0), (-1, 0, 0)),
)


def _scene(*, dynamic=False, iterations=2):
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=(0.008, 0.008, 0.008)),
        gprMax.Discretisation(p1=(DL, DL, DL)),
        gprMax.TimeWindow(iterations=iterations),
        gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(1),
        gprMax.Material(er=2, se=float("inf"), mr=1, sm=0, id="custom_pec"),
        gprMax.Material(er=3, se=0.02, mr=1, sm=float("inf"), id="custom_pmc"),
    ):
        scene.add(obj)
    if dynamic:
        scene.add(gprMax.SurfaceImpedance(
            id="wall", preset="copper", fit_frequency_range=(8e9, 12e9), fit_order=4,
        ))
    else:
        scene.add(gprMax.SurfaceImpedance(id="wall", resistance=50))
    return scene


def _cell(scene, coordinate, material):
    coordinate = np.asarray(coordinate)
    kwargs = {"material_id": material} if isinstance(material, str) else {"material_ids": material}
    scene.add(gprMax.Box(p1=tuple(coordinate * DL), p2=tuple((coordinate + 1) * DL), **kwargs))


def _build(scene, tmp_path, monkeypatch, *, solve=False, precision="double"):
    captured = {}
    original = implementation.compile_impedance_surfaces

    def capture(grid):
        captured["grid"] = grid
        captured["system"] = original(grid)
        return captured["system"]

    monkeypatch.setattr(implementation, "compile_impedance_surfaces", capture)
    gprMax.run(
        scenes=[scene], outputfile=tmp_path / "contact", geometry_only=not solve,
        hide_progress_bars=True, cpu_precision=precision,
    )
    return captured["grid"], captured["system"]


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("quadrant", range(4))
@pytest.mark.parametrize("pec_first", (False, True))
@pytest.mark.parametrize("material", ("pec", "custom_pec"))
def test_pec_quadrant_overrides_sibc_in_both_geometry_orders(
    axis, quadrant, pec_first, material, tmp_path, monkeypatch,
):
    scene = _scene()
    # Alternate face-sharing and diagonal contacts as the placement rotates.
    sibc_quadrant = (quadrant + (1 if quadrant % 2 else 2)) % 4
    cells = [(quadrant, material), (sibc_quadrant, "wall")]
    if not pec_first:
        cells.reverse()
    for q, mat in cells:
        _cell(scene, EDGE + OFFSETS[axis][q], mat)
    grid, system = _build(scene, tmp_path, monkeypatch)
    assert grid.materials[int(grid.ID[(axis, *EDGE)])].is_pec
    assert not np.any(np.all(system.edge_info[:, :4] == (axis, *EDGE), axis=1))
    assert np.isfinite(system.edge_runtime).all()


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("diagonal", (False, True))
def test_only_edge_only_pec_sibc_contact_warns(axis, diagonal, tmp_path, monkeypatch, capsys):
    scene = _scene()
    _cell(scene, EDGE + OFFSETS[axis][0], "wall")
    _cell(scene, EDGE + OFFSETS[axis][2 if diagonal else 1], "pec")
    _build(scene, tmp_path, monkeypatch)
    output = capsys.readouterr().out
    assert output.count("PEC and surface-impedance voxels meet diagonally") == int(diagonal)
    if diagonal:
        assert "at 1 Yee edge(s)" in output
        assert f"E{'xyz'[axis]} at (4, 4, 4)" in output
        assert "forced to zero by PEC" in output


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("pmc_quadrant", (1, 2, 3))
@pytest.mark.parametrize("material", ("pmc", "custom_pmc"))
def test_pruned_pmc_circulation_matches_full_contour_and_foster_history(
    axis, pmc_quadrant, material, tmp_path, monkeypatch,
):
    scene = _scene(dynamic=True)
    _cell(scene, EDGE + OFFSETS[axis][0], "wall")
    _cell(scene, EDGE + OFFSETS[axis][pmc_quadrant], material)
    grid, system = _build(scene, tmp_path, monkeypatch)
    index = np.flatnonzero(np.all(system.edge_info[:, :4] == (axis, *EDGE), axis=1))[0]
    edge = system.edge_info[index].copy()
    assert edge[5] == 2  # Two of the four contour H samples are constrained.
    assert edge[7] == 2  # The two impedance ports are unchanged.
    assert system.edge_fraction[index] == 0.75
    quarter_area = DL**2 / 4
    epsilon_sum = 5 if material == "custom_pmc" else 3
    sigma_sum = 0.02 if material == "custom_pmc" else 0
    mass = config.sim_config.em_consts["e0"] * epsilon_sum * quarter_area / grid.dt
    loss = sigma_sum * quarter_area / 2
    np.testing.assert_allclose(system.edge_params[index], (mass + loss, mass - loss), rtol=1e-14)
    np.testing.assert_array_equal(system.port_g[edge[6]:edge[6] + 2], (-DL / 2, -DL / 2))
    assert all(not grid.materials[int(grid.ID[(3 + h[0], *h[1:])])].is_pmc for h in system.h_info)

    # Independent full contour for a lower-left excluded SIBC quarter:
    # Hy(bottom)/2 - Hy(top) + Hz(right) - Hz(left)/2 (cyclic axes).
    b, c = (axis + 1) % 3, (axis + 2) % 3
    low_b, low_c = EDGE.copy(), EDGE.copy()
    low_b[b] -= 1
    low_c[c] -= 1
    full_h = np.array(((b, *low_c), (b, *EDGE), (c, *EDGE), (c, *low_b)), dtype=np.int32)
    full_weights = np.array((DL / 2, -DL, DL, -DL / 2))

    # Keep only this edge in each runtime, preserving its own port/state offsets.
    system.edge_info = system.edge_info[index:index + 1].copy()
    system.edge_runtime = system.edge_runtime[index:index + 1].copy()
    reference = deepcopy(system)
    reference.edge_info[0, 4:6] = (0, 4)
    reference.h_info = full_h
    reference.h_weight = full_weights
    rng = np.random.default_rng(12)
    for component, name in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")):
        field = getattr(grid, name)
        field[:] = rng.normal(size=field.shape)
        if component >= 3:
            pmc_ids = [m.numID for m in grid.materials if m.is_pmc]
            field[np.isin(grid.ID[component], pmc_ids)] = 0
    system.state_y[:] = rng.normal(scale=1e-4, size=system.state_y.shape)
    reference.state_y[:] = system.state_y
    # Only fields are needed by the two update implementations.
    reference_grid = SimpleNamespace(**{
        name: getattr(grid, name).copy() for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    })
    for _ in range(3):
        reference._update_python(reference_grid)
        system.update(grid)
        np.testing.assert_allclose(
            getattr(grid, "E" + "xyz"[axis]), getattr(reference_grid, "E" + "xyz"[axis]),
            rtol=1e-13, atol=1e-13,
        )
        np.testing.assert_allclose(system.state_y, reference.state_y, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("enclosure", ("pec", "pmc"))
def test_fully_enclosed_sibc_supports_empty_edge_or_h_records(enclosure, tmp_path, monkeypatch):
    scene = _scene(dynamic=True)
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.008, 0.008, 0.008), material_id=enclosure))
    _cell(scene, EDGE, "wall")
    grid, system = _build(scene, tmp_path, monkeypatch)
    assert system.h_info.shape == (0, 4)
    if enclosure == "pec":
        assert system.edge_count == system.port_count == 0
    else:
        assert system.edge_count == 12
        assert np.all(system.edge_info[:, 5] == 0)
        grid.Ex.fill(0.25)
        grid.Ey.fill(0.25)
        grid.Ez.fill(0.25)
    system.update(grid)
    assert np.isfinite(system.state_y).all()
    assert all(np.isfinite(getattr(grid, name)).all() for name in ("Ex", "Ey", "Ez"))


@pytest.mark.parametrize("precision", ("single", "double"))
def test_pec_pmc_and_dynamic_sibc_run_together(precision, tmp_path, monkeypatch):
    scene = _scene(dynamic=True, iterations=160)
    for lo, hi, material in (
        ((3, 3, 3), (5, 5, 5), "wall"),
        ((3, 2, 3), (5, 3, 5), "pec"),
        ((3, 5, 3), (5, 6, 5), "pmc"),
    ):
        scene.add(gprMax.Box(p1=tuple(np.array(lo) * DL), p2=tuple(np.array(hi) * DL), material_id=material))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    scene.add(gprMax.HertzianDipole((0.002, 0.004, 0.004), "z", "pulse"))
    grid, system = _build(scene, tmp_path, monkeypatch, solve=True, precision=precision)
    for component, name in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")):
        field = getattr(grid, name)
        assert np.isfinite(field).all()
        ids = [m.numID for m in grid.materials if (m.is_pec if component < 3 else m.is_pmc)]
        np.testing.assert_array_equal(field[np.isin(grid.ID[component], ids)], 0)
    assert np.isfinite(system.state_y).all()
    assert np.any(system.state_y != 0)


@pytest.mark.parametrize("material", ("pec", "pmc"))
def test_directional_conductor_average_is_not_interpreted_as_isotropic(material, tmp_path, monkeypatch):
    scene = _scene()
    _cell(scene, EDGE + OFFSETS[0][0], "wall")
    _cell(scene, EDGE + OFFSETS[0][1], (material, "free_space", "free_space"))
    with pytest.raises(ValueError, match="directional PEC/PMC mixtures"):
        _build(scene, tmp_path, monkeypatch)


def test_diagonal_contact_warning_is_aggregated_and_examples_are_bounded(tmp_path, monkeypatch, capsys):
    scene = _scene()
    scene.add(gprMax.Box(p1=(0.002, 0.002, 0.002), p2=(0.006, 0.003, 0.003), material_id="wall"))
    scene.add(gprMax.Box(p1=(0.002, 0.003, 0.003), p2=(0.006, 0.004, 0.004), material_id="pec"))
    _build(scene, tmp_path, monkeypatch)
    output = capsys.readouterr().out
    assert output.count("PEC and surface-impedance voxels meet diagonally") == 1
    assert "at 4 Yee edge(s)" in output
    assert "Ex at (2, 3, 3); Ex at (3, 3, 3); Ex at (4, 3, 3)" in output
    assert "Ex at (5, 3, 3)" not in output


def test_existing_component_pec_constraint_is_preserved_without_pec_voxel(tmp_path, monkeypatch):
    scene = _scene()
    _cell(scene, EDGE + OFFSETS[0][0], "wall")
    scene.add(gprMax.Plate(p1=(0.004, 0.004, 0.003), p2=(0.005, 0.004, 0.004), material_id="pec"))
    grid, system = _build(scene, tmp_path, monkeypatch)
    assert grid.materials[int(grid.ID[(0, *EDGE)])].is_pec
    assert not np.any(np.all(system.edge_info[:, :4] == (0, *EDGE), axis=1))


def test_existing_component_pmc_constraint_is_preserved_without_pmc_voxel(tmp_path, monkeypatch):
    scene = _scene()
    _cell(scene, EDGE + OFFSETS[0][0], "wall")
    scene.add(gprMax.MagneticEdge(
        p1=(0.004, 0.004, 0.003), p2=(0.004, 0.005, 0.003), material_id="pmc",
    ))
    grid, system = _build(scene, tmp_path, monkeypatch)
    assert grid.materials[int(grid.ID[4, 4, 4, 3])].is_pmc
    assert not np.any(np.all(system.h_info == (1, 4, 4, 3), axis=1))
