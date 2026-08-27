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

"""Surface-impedance model, geometry, kernel, and integration tests."""

import logging
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.hash_cmds_file import get_user_objects
from gprMax.impedance_surfaces import (
    ImpedanceSurfaceSystem,
    SurfaceImpedanceModel,
    _check_plane_wave_compatibility,
)
from gprMax.user_objects.cmds_geometry.box import Box
from gprMax.user_objects.cmds_geometry.sphere import Sphere
from gprMax.user_objects.cmds_multiuse import SurfaceImpedance


@pytest.fixture(autouse=True)
def restore_package_logging():
    """Do not leak ``gprMax.run``'s application logger into later tests."""

    yield
    logger = logging.getLogger("gprMax")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def test_plane_wave_guard_rejects_sources_that_sample_opaque_boundaries():
    boundary = {(2, 10, 12, 14), (1, 15, 17, 19)}
    vector_wave = SimpleNamespace(
        axial=0,
        corners=np.asarray((5, 6, 7, 20, 21, 22)),
    )
    _check_plane_wave_compatibility([vector_wave], boundary)

    touching = SimpleNamespace(
        axial=0,
        corners=np.asarray((10, 6, 7, 20, 21, 22)),
    )
    with pytest.raises(ValueError, match="strictly inside its TFSF box"):
        _check_plane_wave_compatibility([touching], boundary)

    axial = SimpleNamespace(axial=1, corners=vector_wave.corners)
    with pytest.raises(ValueError, match="axial plane waves sample the geometry"):
        _check_plane_wave_compatibility([axial], boundary)


def test_trapezoidal_surface_transfer_matches_bilinear_warp():
    model = SurfaceImpedanceModel(
        "passive_first_order",
        A=((-2.0e9,),),
        B=(1.0e9,),
        C=(20.0,),
        D=30.0,
    )
    dt = 20e-12
    discrete = model.discretise(dt)
    frequency = 2.2e9
    theta = 2 * np.pi * frequency * dt
    z = np.exp(1j * theta)
    calculated = discrete.Z0 + discrete.L @ np.linalg.solve(
        z * np.eye(model.order) - discrete.F,
        discrete.G,
    )
    warped_s = 2j / dt * np.tan(theta / 2)
    expected = model.D + model.C @ np.linalg.solve(
        warped_s * np.eye(model.order) - model.A,
        model.B,
    )
    np.testing.assert_allclose(calculated, expected, rtol=2e-14, atol=2e-14)


def test_discrete_passivity_check_covers_bilinear_warp_to_infinite_frequency():
    model = SurfaceImpedanceModel(
        "high_frequency_active_band",
        A=((-1.0e3, 0.0), (0.0, -1.0e6)),
        B=(1.0, 1.0),
        C=(1.0e5, -2.0e6),
        D=1.0,
    )
    dt = 0.1
    # A naive continuous check ending at physical Nyquist sees no problem.
    physical_band = np.linspace(0, 1 / (2 * dt), 2049)
    assert np.min(model.impedance(physical_band).real) > 0
    # The trapezoidal map reaches the negative-real high-frequency band on
    # the discrete unit circle and must reject it.
    with pytest.raises(ValueError, match="non-passive on the discrete band"):
        model.discretise(dt)


def test_hash_commands_use_surface_impedance_as_box_material():
    objects = get_user_objects(
        [
            "#surface_impedance: copper_test resistance 0.02\n",
            "#box: 0.01 0.02 0.03 0.04 0.05 0.06 copper_test n\n",
        ],
        checkessential=False,
    )
    assert [type(item) for item in objects] == [SurfaceImpedance, Box]
    assert objects[0].D == pytest.approx(0.02)
    assert objects[1].kwargs["material_id"] == "copper_test"


def test_hash_command_uses_surface_impedance_as_tagged_sphere_material():
    objects = get_user_objects(
        [
            "#surface_impedance: copper_test resistance 0.02\n",
            "#sphere: 0.05 0.05 0.05 0.01 copper_test n metal_body\n",
        ],
        checkessential=False,
    )
    assert isinstance(objects[-1], Sphere)
    assert objects[-1].kwargs["material_id"] == "copper_test"
    assert objects[-1].kwargs["tag"] == "metal_body"


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_cython_sparse_kernel_advances_ade_state_and_edge(dtype):
    from gprMax.cython.impedance_surface import update_impedance_surfaces

    discrete = SurfaceImpedanceModel(
        "first_order",
        A=((-2.0,),),
        B=(1.5,),
        C=(4.0,),
        D=3.0,
    ).discretise(0.1)
    edge_info = np.asarray(((0, 1, 1, 1, 0, 1, 0, 1),), dtype=np.int32)
    edge_params = np.asarray(((2.0, 1.0),), dtype=dtype)
    h_info = np.asarray(((2, 1, 0, 1),), dtype=np.int32)
    h_weight = np.asarray((0.5,), dtype=dtype)
    port_info = np.asarray(((0, 0),), dtype=np.int32)
    port_g = np.asarray((-0.25,), dtype=dtype)
    port_inv_Z0 = np.asarray((1 / discrete.Z0,), dtype=dtype)
    port_g_over_Z0 = port_g * port_inv_Z0
    edge_runtime = np.asarray(
        (
            (
                edge_params[0, 1] + 0.5 * port_g_over_Z0[0],
                1 / (edge_params[0, 0] - 0.5 * port_g_over_Z0[0]),
            ),
        ),
        dtype=dtype,
    )
    model_info = np.asarray(((1, 0),), dtype=np.int32)
    state_old = np.asarray((0.7,), dtype=dtype)
    state_y = np.ascontiguousarray(discrete.L * state_old, dtype=dtype)
    model_f = np.array(np.diag(discrete.F), dtype=dtype, order="C", copy=True)
    model_q = np.array(discrete.L * discrete.G, dtype=dtype, order="C", copy=True)
    fields = [np.zeros((3, 3, 3), dtype=dtype) for _ in range(6)]
    fields[0][1, 1, 1] = 0.4
    fields[5][1, 0, 1] = 4.0

    history = float(discrete.L @ state_old)
    e_old = 0.4
    r_h = 2.0
    denominator = 2.0 - port_g[0] / (2 * discrete.Z0)
    rhs = 1.0 * e_old + r_h + port_g[0] * e_old / (2 * discrete.Z0)
    rhs -= port_g[0] * history / discrete.Z0
    expected_e = rhs / denominator
    expected_k = (0.5 * (expected_e + e_old) - history) / discrete.Z0
    expected_state = discrete.F @ state_old + discrete.G * expected_k

    update_impedance_surfaces(
        1,
        edge_info,
        edge_runtime,
        h_info,
        h_weight,
        port_info,
        port_g_over_Z0,
        port_inv_Z0,
        model_info,
        model_f,
        model_q,
        state_y,
        *fields,
    )
    tolerance = 2e-6 if dtype is np.float32 else 2e-15
    assert fields[0][1, 1, 1] == pytest.approx(expected_e, rel=tolerance, abs=tolerance)
    np.testing.assert_allclose(
        state_y,
        discrete.L * expected_state,
        rtol=tolerance,
        atol=tolerance,
    )


def _two_port_local_runtime(dtype=np.float64):
    """Return a mixed-order corner-edge system and its dense reference data."""

    dt = 0.1
    models = (
        SurfaceImpedanceModel(
            "two_pole",
            A=((-2.0, 0.0), (0.0, -5.0)),
            B=(1.5, 0.75),
            C=(0.8, 0.4),
            D=3.0,
        ).discretise(dt),
        SurfaceImpedanceModel(
            "one_pole",
            A=((-3.0,),),
            B=(0.9,),
            C=(0.7,),
            D=4.0,
        ).discretise(dt),
    )
    x_old = (
        np.asarray((0.7, -0.2), dtype=np.float64),
        np.asarray((-0.4,), dtype=np.float64),
    )
    edge_info = np.asarray(((0, 1, 1, 1, 0, 1, 0, 2),), dtype=np.int32)
    edge_params = np.asarray(((2.0, 1.0),), dtype=dtype)
    h_info = np.asarray(((2, 1, 0, 1),), dtype=np.int32)
    h_weight = np.asarray((0.5,), dtype=dtype)
    port_info = np.asarray(((0, 0), (1, 2)), dtype=np.int32)
    port_g = np.asarray((-0.25, -0.4), dtype=dtype)
    model_info = np.asarray(((2, 0), (1, 2)), dtype=np.int32)
    model_f = np.array(
        np.concatenate(tuple(np.diag(model.F) for model in models)),
        dtype=dtype,
        order="C",
        copy=True,
    )
    model_q = np.array(
        np.concatenate(tuple(model.L * model.G for model in models)),
        dtype=dtype,
        order="C",
        copy=True,
    )
    model_Z0 = np.asarray(tuple(model.Z0 for model in models), dtype=dtype)
    port_inv_Z0 = np.ascontiguousarray(1 / model_Z0, dtype=dtype)
    port_g_over_Z0 = np.ascontiguousarray(port_g * port_inv_Z0, dtype=dtype)
    metric_admittance = float(np.sum(port_g_over_Z0, dtype=np.float64))
    edge_runtime = np.asarray(
        (
            (
                edge_params[0, 1] + 0.5 * metric_admittance,
                1 / (edge_params[0, 0] - 0.5 * metric_admittance),
            ),
        ),
        dtype=dtype,
    )
    state_y = np.ascontiguousarray(
        np.concatenate(tuple(model.L * state for model, state in zip(models, x_old))),
        dtype=dtype,
    )
    system = ImpedanceSurfaceSystem(
        edge_info=edge_info,
        edge_params=edge_params,
        edge_runtime=edge_runtime,
        edge_fraction=np.asarray((0.75,), dtype=dtype),
        h_info=h_info,
        h_weight=h_weight,
        port_info=port_info,
        port_g=port_g,
        port_g_over_Z0=port_g_over_Z0,
        port_inv_Z0=port_inv_Z0,
        port_normal=np.asarray(((1, -1), (2, 1)), dtype=np.int8),
        port_area=np.asarray((0.25, 0.4), dtype=dtype),
        model_info=model_info,
        model_f=model_f,
        model_q=model_q,
        model_Z0=model_Z0,
        state_y=state_y,
        model_ids=("two_pole", "one_pole"),
    )
    fields = [np.zeros((3, 3, 3), dtype=dtype) for _ in range(6)]
    fields[0][1, 1, 1] = 0.4
    fields[5][1, 0, 1] = 4.0
    grid = SimpleNamespace(
        Ex=fields[0],
        Ey=fields[1],
        Ez=fields[2],
        Hx=fields[3],
        Hy=fields[4],
        Hz=fields[5],
    )
    return system, grid, models, x_old


def test_python_and_cython_local_foster_updates_match_dense_two_port_reference():
    from gprMax.cython.impedance_surface import update_impedance_surfaces

    python_system, python_grid, models, x_old = _two_port_local_runtime()
    cython_system, cython_grid, _, _ = _two_port_local_runtime()

    e_old = 0.4
    r_h = 2.0
    histories = tuple(float(model.L @ state) for model, state in zip(models, x_old))
    denominator = 2.0 - sum(g / (2 * model.Z0) for g, model in zip(python_system.port_g, models))
    rhs = (
        1.0 * e_old
        + r_h
        + sum(
            g * e_old / (2 * model.Z0) - g * history / model.Z0
            for g, model, history in zip(python_system.port_g, models, histories)
        )
    )
    expected_e = rhs / denominator
    expected_y = []
    for model, state, history in zip(models, x_old, histories):
        current = (0.5 * (expected_e + e_old) - history) / model.Z0
        expected_y.extend(model.L * (model.F @ state + model.G * current))

    python_system._update_python(python_grid)
    update_impedance_surfaces(
        1,
        cython_system.edge_info,
        cython_system.edge_runtime,
        cython_system.h_info,
        cython_system.h_weight,
        cython_system.port_info,
        cython_system.port_g_over_Z0,
        cython_system.port_inv_Z0,
        cython_system.model_info,
        cython_system.model_f,
        cython_system.model_q,
        cython_system.state_y,
        cython_grid.Ex,
        cython_grid.Ey,
        cython_grid.Ez,
        cython_grid.Hx,
        cython_grid.Hy,
        cython_grid.Hz,
    )

    assert python_grid.Ex[1, 1, 1] == pytest.approx(expected_e, rel=2e-15, abs=2e-15)
    assert cython_grid.Ex[1, 1, 1] == pytest.approx(expected_e, rel=2e-15, abs=2e-15)
    np.testing.assert_allclose(python_system.state_y, expected_y, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(cython_system.state_y, expected_y, rtol=2e-15, atol=2e-15)


def _run_parallel_corner_edges(nthreads, edge_count=128):
    """Advance independent mixed-order corner edges with one OpenMP setting."""

    from gprMax.cython.impedance_surface import update_impedance_surfaces

    template, _, _, _ = _two_port_local_runtime()
    coordinates = np.arange(1, edge_count + 1, dtype=np.int32)
    edge_info = np.tile(template.edge_info, (edge_count, 1))
    edge_info[:, 1] = coordinates
    edge_info[:, 4] = np.arange(edge_count, dtype=np.int32)
    edge_info[:, 6] = 2 * np.arange(edge_count, dtype=np.int32)
    h_info = np.tile(template.h_info, (edge_count, 1))
    h_info[:, 1] = coordinates
    h_weight = np.tile(template.h_weight, edge_count)
    edge_runtime = np.tile(template.edge_runtime, (edge_count, 1))
    port_info = np.tile(template.port_info, (edge_count, 1))
    state_starts = 3 * np.arange(edge_count, dtype=np.int32)
    port_info[0::2, 1] = state_starts
    port_info[1::2, 1] = state_starts + 2
    port_g_over_Z0 = np.tile(template.port_g_over_Z0, edge_count)
    port_inv_Z0 = np.tile(template.port_inv_Z0, edge_count)
    state_y = np.tile(template.state_y, edge_count)
    fields = [np.zeros((edge_count + 2, 3, 3), dtype=np.float64) for _ in range(6)]
    fields[0][coordinates, 1, 1] = np.linspace(0.2, 0.6, edge_count)
    fields[5][coordinates, 0, 1] = np.linspace(3.0, 5.0, edge_count)

    update_impedance_surfaces(
        nthreads,
        edge_info,
        edge_runtime,
        h_info,
        h_weight,
        port_info,
        port_g_over_Z0,
        port_inv_Z0,
        template.model_info,
        template.model_f,
        template.model_q,
        state_y,
        *fields,
    )
    return fields[0], state_y


def test_local_foster_kernel_is_thread_count_independent():
    serial_e, serial_state = _run_parallel_corner_edges(1)
    parallel_e, parallel_state = _run_parallel_corner_edges(4)

    np.testing.assert_array_equal(parallel_e, serial_e)
    np.testing.assert_array_equal(parallel_state, serial_state)


def test_local_foster_reset_clears_only_in_place_y_state():
    system, _, _, _ = _two_port_local_runtime()

    assert np.any(system.state_y)
    system.reset()

    np.testing.assert_array_equal(system.state_y, 0)
    assert not hasattr(system, "state")
    assert not hasattr(system, "state_new")


def _box_scene(*, iterations=2, source=False, inside_receiver=False, dynamic=False):
    dl = 0.001
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.03, 0.03, 0.03)))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(iterations=iterations))
    scene.add(gprMax.PMLThickness(thickness=0 if not source else 3))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.SurfaceImpedance(id="unused", resistance=1.0))
    if dynamic:
        scene.add(
            gprMax.SurfaceImpedance(
                id="wall",
                preset="copper",
                fit_frequency_range=(8e9, 12e9),
                fit_order=4,
            )
        )
    else:
        scene.add(gprMax.SurfaceImpedance(id="wall", resistance=50.0))
    scene.add(
        gprMax.Box(
            p1=(0.012, 0.012, 0.012),
            p2=(0.018, 0.018, 0.018),
            material_id="wall",
            averaging="n",
        )
    )
    if source:
        scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
        scene.add(gprMax.HertzianDipole((0.008, 0.015, 0.015), "z", "pulse"))
    if inside_receiver:
        scene.add(gprMax.Rx((0.015, 0.015, 0.015), id="inside"))
    return scene


@pytest.mark.integration
def test_closed_six_cell_box_compiles_flat_and_convex_edges(tmp_path, monkeypatch):
    import gprMax.impedance_surfaces as implementation

    captured = {}
    original = implementation.compile_impedance_surfaces

    def capture(grid):
        system = original(grid)
        captured["grid"] = grid
        captured["system"] = system
        return system

    monkeypatch.setattr(implementation, "compile_impedance_surfaces", capture)
    gprMax.run(
        scenes=[_box_scene()],
        outputfile=tmp_path / "geometry",
        geometry_only=True,
        hide_progress_bars=True,
        cpu_precision="double",
    )
    grid = captured["grid"]
    system = captured["system"]
    assert system.edge_count == 432
    assert system.port_count == 504
    assert np.count_nonzero(system.edge_fraction == 0.5) == 360
    assert np.count_nonzero(system.edge_fraction == 0.75) == 72
    assert np.count_nonzero(system.edge_info[:, 7] == 1) == 360
    assert np.count_nonzero(system.edge_info[:, 7] == 2) == 72
    assert system.model_ids == ("wall",)
    assert system.model_info.shape == (1, 2)
    assert system.model_info[0, 0] == 0
    np.testing.assert_array_equal(system.model_f, 0)
    np.testing.assert_array_equal(system.model_q, 0)
    np.testing.assert_array_equal(system.state_y, 0)

    # A uniform exterior H=e_t x n produces positive sheet current along
    # e_t. Check both tangential directions on all six face orientations.
    magnetic = (grid.Hx, grid.Hy, grid.Hz)
    for normal_axis in range(3):
        for normal_sign in (-1, 1):
            for electric_axis in range(3):
                if electric_axis == normal_axis:
                    continue
                candidates = []
                for index, edge in enumerate(system.edge_info):
                    if edge[0] != electric_axis or edge[7] != 1:
                        continue
                    port_index = edge[6]
                    if tuple(system.port_normal[port_index]) == (normal_axis, normal_sign):
                        candidates.append(index)
                assert candidates
                edge = system.edge_info[candidates[0]]
                for field in magnetic:
                    field.fill(0)
                electric_direction = np.eye(3, dtype=np.int8)[electric_axis]
                normal = normal_sign * np.eye(3, dtype=np.int8)[normal_axis]
                exterior_h = np.cross(electric_direction, normal)
                h_component = int(np.flatnonzero(exterior_h)[0])
                magnetic[h_component].fill(exterior_h[h_component])
                r_h = sum(
                    system.h_weight[h_index]
                    * magnetic[system.h_info[h_index, 0]][tuple(system.h_info[h_index, 1:])]
                    for h_index in range(edge[4], edge[4] + edge[5])
                )
                assert r_h == pytest.approx(-system.port_g[edge[6]])

    # One x-directed edge on the minimum-y flat face. Only its exterior Hz
    # sample is excited, so the integral-form scalar result is analytic.
    matches = np.flatnonzero(
        (system.edge_info[:, 0] == 0)
        & (system.edge_info[:, 1] == 13)
        & (system.edge_info[:, 2] == 12)
        & (system.edge_info[:, 3] == 15)
    )
    assert matches.size == 1
    edge_index = int(matches[0])
    edge = system.edge_info[edge_index]
    grid.Hx.fill(0)
    grid.Hy.fill(0)
    grid.Hz.fill(0)
    r_h = 0.0
    for h_index in range(edge[4], edge[4] + edge[5]):
        component, i, j, k = system.h_info[h_index]
        if component == 2 and j == 11:
            grid.Hz[i, j, k] = 1.0
        r_h += system.h_weight[h_index] * (grid.Hx, grid.Hy, grid.Hz)[component][i, j, k]
    e_old = grid.Ex[13, 12, 15]
    g = system.port_g[edge[6]]
    expected = (system.edge_params[edge_index, 1] * e_old + r_h + g * e_old / 100) / (
        system.edge_params[edge_index, 0] - g / 100
    )
    system.update(grid)
    assert grid.Ex[13, 12, 15] == pytest.approx(expected, rel=2e-13, abs=1e-13)


@pytest.mark.integration
def test_dynamic_surface_runtime_storage_is_linear_in_poles(tmp_path, monkeypatch):
    import gprMax.impedance_surfaces as implementation

    captured = {}
    original = implementation.compile_impedance_surfaces

    def capture(grid):
        system = original(grid)
        captured["grid"] = grid
        captured["system"] = system
        return system

    monkeypatch.setattr(implementation, "compile_impedance_surfaces", capture)
    gprMax.run(
        scenes=[_box_scene(dynamic=True)],
        outputfile=tmp_path / "dynamic_geometry",
        geometry_only=True,
        hide_progress_bars=True,
        cpu_precision="double",
    )
    grid = captured["grid"]
    system = captured["system"]
    order = int(system.model_info[0, 0])
    port_state_count = int(np.sum(system.model_info[system.port_info[:, 0], 0], dtype=np.int64))
    discrete = grid.surface_impedance_models["wall"].discretise(grid.dt)

    assert order > 0
    assert system.model_info.shape == (1, 2)
    assert system.model_f.shape == (order,)
    assert system.model_q.shape == (order,)
    assert system.model_Z0.shape == (1,)
    assert system.state_y.shape == (port_state_count,)
    assert system.edge_runtime.shape == (system.edge_count, 2)
    assert system.port_g_over_Z0.shape == (system.port_count,)
    assert system.port_inv_Z0.shape == (system.port_count,)
    np.testing.assert_allclose(system.model_f, np.diag(discrete.F), rtol=0, atol=0)
    np.testing.assert_allclose(system.model_q, discrete.L * discrete.G, rtol=0, atol=0)
    np.testing.assert_allclose(system.model_Z0, discrete.Z0, rtol=0, atol=0)
    assert not hasattr(system, "model_F")
    assert not hasattr(system, "model_G")
    assert not hasattr(system, "model_L")
    assert not hasattr(system, "state_new")


@pytest.mark.integration
def test_closed_box_keeps_all_interior_fields_zero(tmp_path):
    output = tmp_path / "interior"
    gprMax.run(
        scenes=[_box_scene(iterations=80, source=True, inside_receiver=True)],
        outputfile=output,
        hide_progress_bars=True,
        cpu_precision="double",
    )
    with h5py.File(output.with_suffix(".h5"), "r") as data:
        receiver = data["rxs/rx1"]
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            np.testing.assert_array_equal(receiver[component][...], 0)
