"""Surface-impedance model, geometry, kernel, and integration tests."""

import h5py
import logging
import numpy as np
import pytest

import gprMax
from gprMax.hash_cmds_file import get_user_objects
from gprMax.impedance_surfaces import SurfaceImpedanceModel
from gprMax.user_objects.cmds_geometry.impedance_box import ImpedanceBox
from gprMax.user_objects.cmds_geometry.impedance_volume import ImpedanceVolume
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


def test_hash_commands_create_constant_model_and_closed_box():
    objects = get_user_objects(
        [
            "#surface_impedance: copper_test 0.02\n",
            "#impedance_box: 0.01 0.02 0.03 0.04 0.05 0.06 copper_test\n",
        ],
        checkessential=False,
    )
    assert [type(item) for item in objects] == [SurfaceImpedance, ImpedanceBox]
    assert objects[0].D == pytest.approx(0.02)
    assert objects[1].surface_impedance_id == "copper_test"


def test_hash_command_creates_tagged_impedance_volume_in_scene_order():
    objects = get_user_objects(
        [
            "#surface_impedance: copper_test 0.02\n",
            "#sphere: 0.05 0.05 0.05 0.01 free_space n metal_body\n",
            "#impedance_volume: metal_body copper_test\n",
        ],
        checkessential=False,
    )
    assert isinstance(objects[-1], ImpedanceVolume)
    assert objects[-1].geometry_tag == "metal_body"
    assert objects[-1].surface_impedance_id == "copper_test"


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
    model_info = np.asarray(((1, 0, 0),), dtype=np.int32)
    state_old = np.asarray((0.7,), dtype=dtype)
    state_new = np.zeros_like(state_old)
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
        edge_params,
        h_info,
        h_weight,
        port_info,
        port_g,
        model_info,
        np.ascontiguousarray(discrete.F.reshape(-1), dtype=dtype),
        np.ascontiguousarray(discrete.G, dtype=dtype),
        np.ascontiguousarray(discrete.L, dtype=dtype),
        np.asarray((discrete.Z0,), dtype=dtype),
        state_old,
        state_new,
        *fields,
    )
    tolerance = 2e-6 if dtype is np.float32 else 2e-15
    assert fields[0][1, 1, 1] == pytest.approx(expected_e, rel=tolerance, abs=tolerance)
    np.testing.assert_allclose(state_new, expected_state, rtol=tolerance, atol=tolerance)


def _box_scene(*, iterations=2, source=False, inside_receiver=False):
    dl = 0.001
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.03, 0.03, 0.03)))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(iterations=iterations))
    scene.add(gprMax.PMLThickness(thickness=0 if not source else 3))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.SurfaceImpedance(id="unused", D=0.0))
    scene.add(gprMax.SurfaceImpedance(id="wall", resistance=50.0))
    scene.add(
        gprMax.ImpedanceBox(
            p1=(0.012, 0.012, 0.012),
            p2=(0.018, 0.018, 0.018),
            surface_impedance_id="wall",
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
