"""Tests for the sub-cell ``#thin_wire`` geometry command."""

from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

import gprMax
import gprMax.model as model_mod

try:
    import pycuda.driver as _cuda_driver

    _cuda_driver.init()
    HAS_CUDA = _cuda_driver.Device.count() > 0
except Exception:
    HAS_CUDA = False


def _capture_grid(monkeypatch):
    captured = {}
    original_build = model_mod.Model.build

    def patched_build(self):
        original_build(self)
        captured["grid"] = self.G

    monkeypatch.setattr(model_mod.Model, "build", patched_build)
    return captured


def _scene(dl=(1e-3, 1.5e-3, 2e-3)):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=dl))
    scene.add(gprMax.Domain(p1=(0.012, 0.012, 0.012)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=1e-12))
    return scene


@pytest.mark.parametrize(
    "axis,p1,p2,e_component,h_components",
    [
        (
            "x",
            (0.003, 0.003, 0.004),
            (0.006, 0.003, 0.004),
            "Ex",
            ("Hy", "Hz"),
        ),
        (
            "y",
            (0.003, 0.003, 0.004),
            (0.003, 0.006, 0.004),
            "Ey",
            ("Hx", "Hz"),
        ),
        (
            "z",
            (0.003, 0.003, 0.004),
            (0.003, 0.003, 0.008),
            "Ez",
            ("Hx", "Hy"),
        ),
    ],
)
def test_directional_ids_and_coefficients(
    monkeypatch,
    tmp_path,
    axis,
    p1,
    p2,
    e_component,
    h_components,
):
    scene = _scene()
    radius = 0.1e-3
    scene.add(gprMax.ThinWire(p1=p1, p2=p2, radius=radius))

    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / f"wire_{axis}",
        hide_progress_bars=True,
    )
    grid = captured["grid"]
    wire = grid.thinwires[0]
    i, j, k = next(wire.cells())

    material_e = grid.materials[int(grid.ID[grid.IDlookup[e_component], i, j, k])]
    assert material_e.is_pec
    assert material_e.type == "thin-wire"
    assert material_e.thin_wire_axis == axis
    assert material_e.thin_wire_radius == radius
    assert material_e.thin_wire_role == e_component
    assert material_e.thin_wire_radial_axis is None

    targets = list(grid._thin_wire_h_targets(axis, i, j, k))
    assert {target[0] for target in targets} == set(h_components)
    assert len(targets) == 4
    free_space = next(material for material in grid.materials if material.ID == "free_space")
    base_coefficients = grid.updatecoeffsH[free_space.numID]
    for component, x, y, z in targets:
        material_h = grid.materials[int(grid.ID[grid.IDlookup[component], x, y, z])]
        assert material_h.type == "thin-wire"
        assert material_h.thin_wire_axis == axis
        assert material_h.thin_wire_role == component

        radial_axis = grid._thin_wire_h_radial_axis(axis, component)
        h_axis = component[1].lower()
        radial_step = grid.dl["xyz".index(radial_axis)]
        h_step = grid.dl["xyz".index(h_axis)]
        factor_f = 2 / np.log(radial_step / radius)
        factor_kh = (radial_step / h_step) * np.arctan(h_step / radial_step)
        factor_ke = 1 / factor_kh
        wire_coefficients = grid.updatecoeffsH[material_h.numID]

        assert material_h.thin_wire_radial_axis == radial_axis
        assert material_h.thin_wire_h_is_projected
        assert material_h.thin_wire_background_numID == free_space.numID
        np.testing.assert_allclose(material_h.thin_wire_factors["F"], factor_f)
        np.testing.assert_allclose(material_h.thin_wire_factors["kH"], factor_kh)
        np.testing.assert_allclose(material_h.thin_wire_factors["kE"], factor_ke)
        np.testing.assert_allclose(material_h.thin_wire_factors["F_kH"], factor_f * factor_kh)

        np.testing.assert_allclose(wire_coefficients[0], base_coefficients[0])
        np.testing.assert_allclose(wire_coefficients[4], base_coefficients[4] * factor_kh)
        for column, direction in zip((1, 2, 3), "xyz"):
            expected_factor = factor_f * factor_kh if direction == radial_axis else 1.0
            np.testing.assert_allclose(
                wire_coefficients[column], base_coefficients[column] * expected_factor
            )

    own_h = f"H{axis}"
    assert grid.ID[grid.IDlookup[own_h], i, j, k] == free_space.numID


@pytest.mark.parametrize(
    "radial_step,h_step,radius",
    [(1e-3, 1e-3, 0.1e-3), (1e-3, 2e-3, 0.1e-3)],
)
def test_projected_h_update_is_algebraically_makinen_equivalent(radial_step, h_step, radius):
    """The stored-H update is equations (8)-(14), not a new approximation."""

    factor_f = 2 / np.log(radial_step / radius)
    factor_kh = (radial_step / h_step) * np.arctan(h_step / radial_step)
    factor_ke = 1 / factor_kh
    db_wire = 0.37
    db_radial = -0.61
    radial_e_difference = 1.23
    axial_e_difference = -0.48

    literal_then_projected = factor_kh * (
        factor_ke * db_wire * radial_e_difference + factor_f * db_radial * axial_e_difference
    )
    stored_field_update = (
        db_wire * radial_e_difference + factor_f * factor_kh * db_radial * axial_e_difference
    )
    np.testing.assert_allclose(literal_then_projected, stored_field_update)


def test_square_cell_makinen_factors():
    factor_kh = np.arctan(1.0)
    factor_ke = 1 / factor_kh
    np.testing.assert_allclose(factor_kh, np.pi / 4)
    np.testing.assert_allclose(factor_ke, 4 / np.pi)


def test_colocated_magnetic_frill_deposits_projected_h(monkeypatch, tmp_path):
    """The frill applies F to the projected-H source coefficient."""

    dl = 1e-3
    scene = _scene(dl=(dl, dl, dl))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.012, 0.012, dl), material_id="pec"))
    scene.add(
        gprMax.ThinWire(
            p1=(0.006, 0.006, 0),
            p2=(0.006, 0.006, 0.004),
            radius=0.1e-3,
        )
    )
    scene.add(
        gprMax.MagneticFrillSource(
            p1=(0.006, 0.006, 0),
            polarisation="z",
            zcoax=50,
            waveform_id="w",
        )
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "thin_wire_frill",
        hide_progress_bars=True,
    )
    grid = captured["grid"]
    frill = grid.magneticfrillsources[0]
    i, j, k = frill.xcoord, frill.ycoord, frill.zcoord
    frill.waveformvalues_wholedt[0] = 2

    expected_gains = []
    current_weights = []
    for component, x, y, z, sign, denominator, current_weight in (
        ("Hy", i, j, k, 1, grid.dx * grid.dz, grid.dy),
        ("Hy", i - 1, j, k, -1, grid.dx * grid.dz, -grid.dy),
        ("Hx", i, j, k, -1, grid.dy * grid.dz, -grid.dx),
        ("Hx", i, j - 1, k, 1, grid.dy * grid.dz, grid.dx),
    ):
        material_id = int(grid.ID[grid.IDlookup[component], x, y, z])
        material = grid.materials[material_id]
        assert material.thin_wire_h_is_projected
        np.testing.assert_allclose(material.thin_wire_factors["kH"], np.pi / 4)
        gain = (
            sign
            * grid.updatecoeffsH[material_id, 4]
            * material.thin_wire_factors["F"]
            / denominator
        )
        expected_gains.append((component, x, y, z, gain))
        current_weights.append(current_weight)

    feed_self_admittance = sum(
        weight * term[-1] for weight, term in zip(current_weights, expected_gains)
    )
    expected_vtotal = 2 / (1 + 0.5 * 50 * feed_self_admittance)

    frill.update_magnetic(0, grid.updatecoeffsH, grid.ID, grid.Hx, grid.Hy, grid.Hz, grid)

    assert frill.Vinc[0] == 1
    np.testing.assert_allclose(frill._G_coeff, feed_self_admittance)
    np.testing.assert_allclose(frill.Vtotal[0], expected_vtotal)
    np.testing.assert_allclose(frill.Vtotal[0], 2 * frill.Vinc[0] - 50 * frill.Itot[0])
    for component, x, y, z, gain in expected_gains:
        np.testing.assert_allclose(getattr(grid, component)[x, y, z], gain * expected_vtotal)

    # Stopping the source gates only the incident waveform. The coaxial
    # terminal relation must remain connected and respond to antenna current.
    for component in ("Hx", "Hy", "Hz"):
        getattr(grid, component).fill(0)
    component, x, y, z, weight, _ = frill._drive_terms[0]
    getattr(grid, component)[x, y, z] = 1 / weight
    frill._previous_half_current = 0
    frill.waveformvalues_wholedt[1] = 0
    frill.stop = 0
    frill.update_magnetic(1, grid.updatecoeffsH, grid.ID, grid.Hx, grid.Hy, grid.Hz, grid)
    expected_current = 0.5 / (1 + 0.5 * frill._G_coeff * frill.Z0)
    np.testing.assert_allclose(frill.Itot[1], expected_current)
    np.testing.assert_allclose(frill.Vtotal[1], -frill.Z0 * expected_current)
    assert frill.Vtotal[1] != 0


def test_inherits_resolved_magnetic_background(monkeypatch, tmp_path):
    scene = _scene(dl=(1e-3, 1e-3, 1e-3))
    scene.add(gprMax.Material(er=3, se=0, mr=4, sm=0, id="magnetic_host"))
    scene.add(
        gprMax.Box(
            p1=(0, 0, 0),
            p2=(0.012, 0.012, 0.012),
            material_id="magnetic_host",
            averaging="n",
        )
    )
    scene.add(
        gprMax.ThinWire(
            p1=(0.004, 0.004, 0.003),
            p2=(0.004, 0.004, 0.007),
            radius=0.1e-3,
        )
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "wire_magnetic_host",
        hide_progress_bars=True,
    )
    grid = captured["grid"]
    derived = [material for material in grid.materials if material.type == "thin-wire"]
    assert derived
    assert all(material.mr == 4 for material in derived)
    assert all(material.sm == 0 for material in derived)


def test_transverse_boundary_requires_pmc(monkeypatch, tmp_path):
    scene = _scene(dl=(1e-3, 1e-3, 1e-3))
    scene.add(
        gprMax.ThinWire(
            p1=(0, 0.004, 0.003),
            p2=(0, 0.004, 0.007),
            radius=0.1e-3,
        )
    )
    with pytest.raises(ValueError, match="PMC symmetry boundary"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "wire_no_symmetry",
            hide_progress_bars=True,
        )


def test_transverse_pmc_boundary_uses_only_active_h_components(monkeypatch, tmp_path):
    scene = _scene(dl=(1e-3, 1e-3, 1e-3))
    scene.add(gprMax.SymmetryBoundary(face="x0", type="pmc"))
    scene.add(
        gprMax.ThinWire(
            p1=(0, 0.004, 0.003),
            p2=(0, 0.004, 0.007),
            radius=0.1e-3,
        )
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "wire_symmetry",
        hide_progress_bars=True,
    )
    grid = captured["grid"]
    i, j, k = next(grid.thinwires[0].cells())
    targets = list(grid._thin_wire_h_targets("z", i, j, k))
    assert len(targets) == 3
    assert ("Hy", 0, j, k) in targets


def test_surrounding_h_stencil_cannot_touch_pml(monkeypatch, tmp_path):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.TimeWindow(time=1e-12))
    scene.add(
        gprMax.ThinWire(
            p1=(0.002, 0.01, 0.006),
            p2=(0.002, 0.01, 0.014),
            radius=0.1e-3,
        )
    )

    with pytest.raises(ValueError, match="magnetic component inside a PML"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "wire_adjacent_pml",
            hide_progress_bars=True,
        )


@pytest.mark.parametrize("radius", [0, -1e-4, float("inf"), float("nan"), 0.5e-3])
def test_invalid_radius_rejected(monkeypatch, tmp_path, radius):
    scene = _scene(dl=(1e-3, 1e-3, 1e-3))
    scene.add(
        gprMax.ThinWire(
            p1=(0.003, 0.003, 0.003),
            p2=(0.003, 0.003, 0.007),
            radius=radius,
        )
    )
    with pytest.raises(ValueError, match="radius"):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "wire_bad_radius",
            hide_progress_bars=True,
        )


def test_hash_command(monkeypatch, tmp_path: Path):
    input_file = tmp_path / "thin_wire.in"
    input_file.write_text(
        "#title: thin wire text command\n"
        "#dx_dy_dz: 0.001 0.001 0.001\n"
        "#domain: 0.012 0.012 0.012\n"
        "#pml_cells: 0\n"
        "#time_window: 1e-12\n"
        "#thin_wire: 0.004 0.004 0.003 0.004 0.004 0.007 0.0001\n"
    )

    captured = _capture_grid(monkeypatch)
    gprMax.run(
        inputfile=str(input_file),
        n=1,
        geometry_only=True,
        outputfile=tmp_path / "wire_hash",
        hide_progress_bars=True,
    )
    grid = captured["grid"]
    assert len(grid.thinwires) == 1
    assert grid.thinwires[0].wire_axis == "z"


def _cuda_parity_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=2e-10))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(gprMax.HertzianDipole(p1=(0.006, 0.01, 0.01), polarisation="z", waveform_id="w"))
    scene.add(
        gprMax.ThinWire(
            p1=(0.01, 0.01, 0.005),
            p2=(0.01, 0.01, 0.015),
            radius=0.1e-3,
        )
    )
    scene.add(gprMax.Rx(p1=(0.014, 0.01, 0.01)))
    return scene


@pytest.mark.skipif(not HAS_CUDA, reason="No CUDA device/pycuda available")
def test_cuda_uses_same_thin_wire_coefficients_as_cpu(tmp_path):
    cpu_path = tmp_path / "thin_wire_cpu"
    cuda_path = tmp_path / "thin_wire_cuda"
    gprMax.run(
        scenes=[_cuda_parity_scene()],
        n=1,
        outputfile=cpu_path,
        hide_progress_bars=True,
        cpu_precision="double",
    )
    gprMax.run(
        scenes=[_cuda_parity_scene()],
        n=1,
        outputfile=cuda_path,
        hide_progress_bars=True,
        gpu=[0],
        gpu_precision="double",
    )

    components = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    with h5py.File(str(cpu_path) + ".h5", "r") as output:
        cpu = {name: output[f"rxs/rx1/{name}"][:] for name in components}
    with h5py.File(str(cuda_path) + ".h5", "r") as output:
        cuda = {name: output[f"rxs/rx1/{name}"][:] for name in components}

    scale = max(float(np.max(np.abs(values))) for values in cpu.values())
    assert scale > 0
    for component in components:
        assert np.isfinite(cuda[component]).all()
        assert_allclose(cuda[component], cpu[component], rtol=2e-9, atol=2e-10 * scale)
