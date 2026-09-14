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

"""Receiver arrays must stay within their snapped, inclusive grid bounds."""

from itertools import product

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.model import Model

pytestmark = pytest.mark.integration


@pytest.fixture
def built_model(monkeypatch):
    captured = {}
    original = Model.build

    def capture(model):
        result = original(model)
        captured["grid"] = model.G
        captured["subgrids"] = list(model.subgrids)
        return result

    monkeypatch.setattr(Model, "build", capture)
    return captured


def _scene(dl, domain):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=dl))
    scene.add(gprMax.Domain(p1=domain))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.TimeWindow(iterations=1))
    return scene


def _build(scene, built_model, tmp_path, **kwargs):
    gprMax.run(
        scenes=[scene],
        geometry_only=True,
        outputfile=tmp_path / "array",
        hide_progress_bars=True,
        **kwargs,
    )
    return built_model["grid"]


@pytest.mark.parametrize("spacing", (0.001, 0.002, 0.0025, 0.005))
@pytest.mark.parametrize("zero_step", (True, False))
def test_documented_line_has_no_extra_rows(built_model, tmp_path, spacing, zero_step):
    scene = _scene((spacing,) * 3, (0.15, 0.12, 0.10))
    transverse_step = 0 if zero_step else spacing
    scene.add(
        gprMax.RxArray(
            p1=(0.02, 0.10, 0.05),
            p2=(0.12, 0.10, 0.05),
            dl=(0.01, transverse_step, transverse_step),
        )
    )
    grid = _build(scene, built_model, tmp_path)
    expected = [(x / 100, 0.10, 0.05) for x in range(2, 13)]
    np.testing.assert_allclose([rx.coord * grid.dl for rx in grid.rxs], expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("boundary_axis", range(3))
def test_array_on_far_domain_face_does_not_step_outside(built_model, tmp_path, boundary_axis):
    domain = np.full(3, 0.08)
    domain[boundary_axis] = 0.07
    lower = np.full(3, 0.03)
    lower[boundary_axis] = 0.07
    upper = lower.copy()
    varying_axis = (boundary_axis + 1) % 3
    upper[varying_axis] = 0.07
    step = np.zeros(3)
    step[varying_axis] = 0.01
    scene = _scene((0.002,) * 3, tuple(domain))
    scene.add(gprMax.RxArray(p1=tuple(lower), p2=tuple(upper), dl=tuple(step)))
    grid = _build(scene, built_model, tmp_path)
    expected = np.tile(np.rint(lower / grid.dl).astype(int), (5, 1))
    expected[:, varying_axis] = (15, 20, 25, 30, 35)
    np.testing.assert_array_equal([rx.coord for rx in grid.rxs], expected)


@pytest.mark.parametrize(
    "dl,lower,upper,step,expected_axes",
    [
        pytest.param(
            (0.001,) * 3,
            (0.011,) * 3,
            (0.011,) * 3,
            (0,) * 3,
            ([11], [11], [11]),
            id="singleton",
        ),
        pytest.param(
            (0.001,) * 3,
            (0.011,) * 3,
            (0.013,) * 3,
            (0,) * 3,
            ([11, 12, 13],) * 3,
            id="zero-step-means-one-cell",
        ),
        pytest.param(
            (0.001,) * 3,
            (0.002,) * 3,
            (0.010,) * 3,
            (0.003,) * 3,
            ([2, 5, 8],) * 3,
            id="non-divisible-volume",
        ),
        pytest.param(
            (0.001,) * 3,
            (0.011,) * 3,
            (0.013,) * 3,
            (0.003,) * 3,
            ([11],) * 3,
            id="step-larger-than-extent",
        ),
        pytest.param(
            (0.001, 0.002, 0.005),
            (0.0106, 0.0211, 0.051),
            (0.0164, 0.0291, 0.059),
            (0.0016, 0.0051, 0.014),
            ([11, 13, 15], [11, 14], [10]),
            id="anisotropic-snapped-bounds-and-steps",
        ),
    ],
)
def test_cartesian_array_uses_bounded_integer_steps(built_model, tmp_path, dl, lower, upper, step, expected_axes):
    scene = _scene(dl, (0.03, 0.04, 0.08))
    scene.add(gprMax.RxArray(p1=lower, p2=upper, dl=step))
    grid = _build(scene, built_model, tmp_path)
    # Assert every coordinate in x/y/z nesting order, not just the count.
    np.testing.assert_array_equal([rx.coord for rx in grid.rxs], list(product(*expected_axes)))


@pytest.mark.parametrize("family", ("TM", "TE"))
@pytest.mark.parametrize("invariant_axis", range(3))
def test_reduced_array_preserves_single_live_layer(built_model, tmp_path, family, invariant_axis):
    domain = np.full(3, 0.08)
    domain[invariant_axis] = float("inf")
    lower = np.full(3, 0.03)
    lower[invariant_axis] = float("inf")
    upper = lower.copy()
    varying_axis = (invariant_axis + 1) % 3
    upper[varying_axis] = 0.07
    step = np.zeros(3)
    step[varying_axis] = 0.01
    scene = _scene((0.001,) * 3, tuple(domain))
    scene.add(gprMax.DomainMode(mode=family))
    scene.add(gprMax.RxArray(p1=tuple(lower), p2=tuple(upper), dl=tuple(step)))
    grid = _build(scene, built_model, tmp_path)
    expected = np.full((5, 3), 30)
    expected[:, invariant_axis] = 1 if family == "TE" else 0
    expected[:, varying_axis] = (30, 40, 50, 60, 70)
    np.testing.assert_array_equal([rx.coord for rx in grid.rxs], expected)


@pytest.mark.parametrize("family", ("TM", "TE"))
@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("sign", (-1, 1))
def test_symbolic_array_is_reusable_in_independent_builds(built_model, tmp_path, family, axis, sign):
    """A new full build resolves inf afresh; no running mesh is changed."""
    domain = [0.04] * 3
    lower, upper, step = [0.02] * 3, [0.02] * 3, [0.0] * 3
    domain[axis] = lower[axis] = upper[axis] = sign * float("inf")
    varying = (axis + 1) % 3
    lower[varying], upper[varying], step[varying] = 0.022, 0.026, 0.002
    kwargs = dict(p1=tuple(lower), p2=tuple(upper), dl=tuple(step))
    array = gprMax.RxArray(**kwargs)
    for spacing, mode in ((0.002, family), (0.001, family), (0.001, "TE"), (0.001, "TM")):
        coordinates = []
        for declaration in (array, gprMax.RxArray(**kwargs)):
            scene = _scene((spacing,) * 3, tuple(domain))
            scene.add(gprMax.DomainMode(mode=mode))
            scene.add(declaration)
            grid = _build(scene, built_model, tmp_path)
            coordinates.append(np.array([rx.coord for rx in grid.rxs]))
            assert declaration.lower_point == kwargs["p1"]
            assert declaration.upper_point == kwargs["p2"]
            assert declaration.kwargs == kwargs
        np.testing.assert_array_equal(*coordinates)
        assert len(coordinates[0]) == 3
        np.testing.assert_array_equal(coordinates[0][:, axis], 1 if mode == "TE" else 0)


def test_reused_array_reresolves_transverse_domain_extent(built_model, tmp_path):
    array = gprMax.RxArray(
        p1=(float("inf"), 0.02, float("inf")), p2=(float("inf"), 0.02, float("inf")), dl=(0.002, 0, 0)
    )
    for extent in (0.04, 0.06):
        scene = _scene((0.002,) * 3, (extent, 0.04, float("inf")))
        scene.add(gprMax.DomainMode(mode="TE"))
        scene.add(array)
        grid = _build(scene, built_model, tmp_path)
        np.testing.assert_array_equal([rx.coord[0] for rx in grid.rxs], np.arange(round(extent / 0.002) + 1))


@pytest.mark.parametrize(
    "backend", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu), pytest.param("opencl", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize("axis", range(3))
def test_reused_te_array_records_same_fields_as_fresh_array(tmp_path, request, backend, axis):
    options = {}
    if backend != "cpu":
        options["gpu" if backend == "cuda" else "opencl"] = [
            request.getfixturevalue("gpu_device" if backend == "cuda" else "opencl_device")
        ]
    varying = (axis + 1) % 3
    source, lower, upper, domain = ([0.02] * 3 for _ in range(4))
    domain[:] = [0.04] * 3
    for point in (source, lower, upper, domain):
        point[axis] = float("inf")
    lower[varying], upper[varying] = 0.022, 0.026
    step = [0.0] * 3
    step[varying] = 0.002
    array_kwargs = dict(p1=tuple(lower), p2=tuple(upper), dl=tuple(step))
    array = gprMax.RxArray(**array_kwargs)
    for name, spacing, declaration in (
        ("coarse", 0.002, array),
        ("reused", 0.001, array),
        ("fresh", 0.001, gprMax.RxArray(**array_kwargs)),
    ):
        scene = gprMax.Scene()
        for obj in (
            gprMax.Discretisation((spacing,) * 3),
            gprMax.Domain(tuple(domain)),
            gprMax.DomainMode("TE"),
            gprMax.TimeWindow(iterations=32),
            gprMax.OMPThreads(1),
            gprMax.PMLThickness(0),
            gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"),
            gprMax.HertzianDipole(p1=tuple(source), polarisation="xyz"[varying], waveform_id="pulse"),
            declaration,
        ):
            scene.add(obj)
        gprMax.run(scenes=[scene], outputfile=tmp_path / name, hide_progress_bars=True, **options)
    with h5py.File(tmp_path / "reused.h5") as reused, h5py.File(tmp_path / "fresh.h5") as fresh:
        assert len(reused["rxs"]) == len(fresh["rxs"]) == 3
        for name in reused["rxs"]:
            np.testing.assert_array_equal(
                reused[f"rxs/{name}"].attrs["Position"], fresh[f"rxs/{name}"].attrs["Position"]
            )
            for component in reused[f"rxs/{name}"]:
                np.testing.assert_array_equal(
                    reused[f"rxs/{name}/{component}"][...], fresh[f"rxs/{name}/{component}"][...]
                )
            assert np.any(reused[f"rxs/{name}/E{'xyz'[varying]}"][...])


@pytest.mark.parametrize("autotranslate", (True, False))
def test_off_origin_subgrid_uses_fine_grid_and_correct_coordinate_frame(built_model, tmp_path, autotranslate):
    scene = _scene((0.003,) * 3, (0.09,) * 3)
    fine = gprMax.SubGridHSG(p1=(0.03,) * 3, p2=(0.06,) * 3, ratio=3, id="fine")
    fine.add(gprMax.RxArray(p1=(0.041, 0.05, 0.05), p2=(0.048, 0.05, 0.05), dl=(0.002, 0, 0)))
    scene.add(fine)
    _build(scene, built_model, tmp_path, subgrid=True, autotranslate=autotranslate)
    grid = built_model["subgrids"][0]
    expected = np.array([(x, 50, 50) for x in (41, 43, 45, 47)])
    if autotranslate:
        expected += np.array((grid.n_boundary_cells_x, grid.n_boundary_cells_y, grid.n_boundary_cells_z)) - 30
    np.testing.assert_array_equal([rx.coord for rx in grid.rxs], expected)
    np.testing.assert_allclose(grid.dl, (0.001,) * 3, rtol=0, atol=1e-15)
    assert [rx.build_index for rx in grid.rxs] == list(range(4))


@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("on_edge", (False, True))
def test_hash_array_writes_only_requested_receivers(built_model, tmp_path, precision, on_edge):
    if on_edge:
        domain = "0.160 0.070 0.002"
        array = "0.030 0.070 0 0.140 0.070 0 0.010 0 0"
        expected = [(i * 0.002, 0.070, 0) for i in range(15, 71, 5)]
    else:
        domain = "0.240 0.060 0.002"
        array = "0.014 0.030 0 0.214 0.030 0 0.020 0.002 0.002"
        expected = [(i * 0.002, 0.030, 0) for i in range(7, 108, 10)]
    path = tmp_path / "array.in"
    path.write_text(
        f"#domain: {domain}\n"
        "#dx_dy_dz: 0.002 0.002 0.002\n"
        "#time_window: 2e-10\n"
        "#pml_cells: 0\n"
        "#num_threads: 1\n"
        "#waveform: ricker 1 3e9 w\n"
        "#hertzian_dipole: z 0.020 0.030 0 w\n"
        f"#rx_array: {array}\n"
    )
    gprMax.run(inputfile=path, cpu_precision=precision, hide_progress_bars=True)
    grid = built_model["grid"]
    np.testing.assert_allclose([rx.coord * grid.dl for rx in grid.rxs], expected, rtol=0, atol=1e-15)
    with h5py.File(path.with_suffix(".h5"), "r") as output:
        positions = [output[f"rxs/rx{i+1}"].attrs["Position"] for i in range(len(expected))]
        np.testing.assert_allclose(positions, expected, rtol=0, atol=1e-15)
