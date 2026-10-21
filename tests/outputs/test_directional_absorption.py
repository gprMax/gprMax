"""Directional electric loss: independent contractions and real FDTD solves."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import gprMax
from gprMax import config
from gprMax.hash_cmds_geometry import process_geometrycmds
from gprMax.materials import DispersiveMaterial, Material, create_directional_material
from gprMax.sar import (
    EDGE_OFFSETS,
    SARLocalPayload,
    SARMonitor,
    _material_loss_conductivity,
)


def _materials():
    values = [Material(i, axis) for i, axis in enumerate("xyz")]
    for item, sigma in zip(values, (0.1, 0.3, 0.6)):
        item.se = sigma
        item.mass_density = 1000.0
    return SimpleNamespace(materials=values)


def test_directional_record_preserves_order_and_density():
    grid = _materials()
    axes = tuple(grid.materials)
    xyz = create_directional_material(grid, axes)
    zyx = create_directional_material(grid, axes[::-1])
    assert xyz is create_directional_material(grid, axes)
    assert xyz.ID != zyx.ID
    assert xyz.directional_materials == axes
    assert zyx.directional_materials == axes[::-1]
    assert xyz.mass_density == 1000
    assert xyz.se == zyx.se  # Same scalar mean does not mean the same tensor.


@pytest.mark.parametrize(
    "command",
    (
        "#box: 0 0 0 0.01 0.01 0.01",
        "#sphere: 0.01 0.01 0.01 0.004",
        "#ellipsoid: 0.01 0.01 0.01 0.004 0.003 0.002",
        "#cylinder: 0 0 0 0.01 0.01 0.01 0.002",
        "#cone: 0 0 0 0.01 0.01 0.01 0.002 0.003",
        "#cylindrical_sector: z 0.01 0.01 0 0.01 0.005 0 90",
        "#triangle: 0 0 0 0.01 0 0 0 0.01 0 0.002",
    ),
)
@pytest.mark.parametrize("averaging", ("n", "y"))
def test_directional_tag_hash_syntax(command, averaging):
    (obj,) = process_geometrycmds([command + f" mat_x mat_y mat_z {averaging} tissue"])
    assert obj.kwargs["material_ids"] == ["mat_x", "mat_y", "mat_z"]
    assert obj.kwargs["tag"] == "tissue"


@pytest.mark.parametrize("kind", ("debye", "lorentz", "drude"))
def test_directional_loss_preserves_poles(kind, monkeypatch):
    monkeypatch.setattr(config, "sim_config", SimpleNamespace(em_consts={"e0": config.e0}))
    grid = _materials()
    pole = DispersiveMaterial(2, "z")
    pole.er, pole.se, pole.poles, pole.type = 2.0, 0.0, 1, kind
    pole.deltaer = [2.0]
    pole.tau = [1e-10 if kind == "debye" else 2e9]
    pole.alpha = [1e9]
    grid.materials[2] = pole
    tensor = create_directional_material(grid, grid.materials)
    frequencies = np.asarray((0.5e9, 1e9, 1.5e9))
    omega = 2 * np.pi * frequencies
    if kind == "debye":
        expected = config.e0 * 2 * omega**2 * 1e-10 / (1 + (omega * 1e-10) ** 2)
    elif kind == "lorentz":
        w0 = 2 * np.pi * 2e9
        expected = (
            config.e0
            * 2
            * w0**2
            * 2e9
            * omega**2
            / ((w0**2 - omega**2) ** 2 + (2e9 * omega) ** 2)
        )
    else:
        wp = 2 * np.pi * 2e9
        expected = config.e0 * wp**2 * 1e9 / (omega**2 + 1e18)
    loss = _material_loss_conductivity(grid, [tensor.numID], frequencies)
    np.testing.assert_allclose(loss[0, :, 0], 0.1, rtol=1e-14)
    np.testing.assert_allclose(loss[1, :, 0], 0.3, rtol=1e-14)
    np.testing.assert_allclose(loss[2, :, 0], expected, rtol=1e-14)


@pytest.mark.parametrize("active", (("Ex",), ("Ey", "Ez"), ("Ex", "Ey", "Ez")))
def test_component_contraction_and_mpi_catalogue(active, monkeypatch):
    monkeypatch.setattr(config, "sim_config", SimpleNamespace(em_consts={"e0": config.e0}))
    payloads = []
    field_values = {"Ex": 1 + 2j, "Ey": 3 - 1j, "Ez": 2 + 4j}
    expected = []
    for rank, x in enumerate((2, 6)):
        grid = _materials()
        tensor = create_directional_material(grid, grid.materials[:: 1 if rank == 0 else -1])
        cell = np.asarray(((x, 2, 2),), dtype=np.int32)
        # Both ranks deliberately use local numeric ID 3 for different tensors.
        payloads.append(
            SARLocalPayload(
                cell_indices=cell,
                tag_id=np.asarray([1]),
                material_id=np.asarray([3]),
                density=np.asarray([1000.0]),
                absorbed_power_density=np.zeros((1, 1)),
                excluded_pml_cell_count=0,
                edge_coordinates={c: cell[0] + EDGE_OFFSETS[c] for c in active},
                edge_dft={c: np.full((1, 4), field_values[c]) for c in active},
                material_definitions={3: tensor},
            )
        )
        sigmas = (0.1, 0.3, 0.6)[:: 1 if rank == 0 else -1]
        expected.append(
            0.5 * sum(sigmas["xyz".index(c[1])] * abs(field_values[c]) ** 2 for c in active)
        )
    monitor = SARMonitor.__new__(SARMonitor)
    monitor.frequencies = np.asarray([1e9])
    monitor.real_dtype = np.dtype(np.float64)
    monitor.grid = SimpleNamespace(materials=[])  # Neither material exists on the coordinator.
    monitor.edge_offsets = {c: EDGE_OFFSETS[c] for c in active}
    merged = monitor.merge_local_payloads(payloads)
    result = monitor._collocate_mpi_payload(merged, (10, 6, 6))
    assert result.material_id[0] != result.material_id[1]
    np.testing.assert_allclose(result.absorbed_power_density[0], expected, rtol=1e-14)


def directional_scene(*, axis="z", dispersive=False, directional=True, densities=(1000,) * 3):
    """With only the invariant E component active, transverse loss cannot matter."""
    scene = gprMax.Scene()
    invariant = "xyz".index(axis)
    domain = [0.032] * 3
    domain[invariant] = float("inf")
    upper = [0.032] * 3
    upper[invariant] = 0.001
    source = [0.012] * 3
    source[invariant] = 0
    low, high = [0.020] * 3, [0.021] * 3
    low[invariant], high[invariant] = 0, 0.001
    pml = [4] * 6
    pml[invariant] = pml[invariant + 3] = 0
    for obj in (
        gprMax.Domain(p1=domain),
        gprMax.DomainMode(mode="TM"),
        gprMax.Discretisation(p1=(0.001,) * 3),
        gprMax.PMLThickness(thickness=pml),
        gprMax.TimeWindow(time=3e-9),
        gprMax.OMPThreads(1),
        gprMax.Waveform(wave_type="ricker", amp=1, freq=1e9, id="pulse"),
        gprMax.HertzianDipole(p1=source, polarisation=axis, waveform_id="pulse"),
    ):
        scene.add(obj)
    for direction, density in zip("xyz", densities):
        scene.add(
            gprMax.Material(
                er=2,
                se=0.6 if direction == axis and not dispersive else 0,
                mr=1,
                sm=0,
                id=direction,
            )
        )
        if density is not None:
            scene.add(gprMax.MaterialDensity(density=density, material_ids=direction))
    if dispersive:
        scene.add(
            gprMax.AddDebyeDispersion(poles=1, er_delta=[2], tau=[1e-10], material_ids=[axis])
        )
    kwargs = {"material_ids": list("xyz")} if directional else {"material_id": axis}
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=upper, averaging=False, **kwargs))
    scene.add(gprMax.Box(p1=low, p2=high, averaging=False, tag="sample", **kwargs))
    sar = gprMax.SAR(
        frequencies=(0.75e9, 1e9, 1.25e9), tags="sample", waveform_id="pulse", id="dose"
    )
    rad = gprMax.Radiometry(
        frequencies=(0.75e9, 1e9, 1.25e9), tags="sample", waveform_id="pulse", id="rad"
    )
    scene.add(sar)
    scene.add(rad)
    return scene, sar, rad


def _verify_solve(sar, rad, axis, dispersive):
    monitor = rad._monitor
    e = np.mean(monitor.accumulators["E" + axis][:, monitor.cell_edge_indices["E" + axis]], axis=2)
    omega = 2 * np.pi * rad.result.frequency
    sigma = (
        (config.e0 * 2 * omega**2 * 1e-10 / (1 + (omega * 1e-10) ** 2))
        if dispersive
        else np.full(3, 0.6)
    )
    expected = 0.5 * sigma[:, None] * np.abs(e * rad.result.normalisation_scale[:, None]) ** 2
    assert np.all(rad.result.valid)
    np.testing.assert_allclose(rad.result.absorbed_power_density, expected, rtol=2e-6)
    np.testing.assert_allclose(sar.result.sar, expected / 1000, rtol=2e-6)


@pytest.mark.parametrize("axis", tuple("xyz"))
@pytest.mark.parametrize("dispersive", (False, True))
def test_directional_tm_solve_matches_isotropic(axis, dispersive, tmp_path):
    records = []
    for directional in (False, True):
        scene, sar, rad = directional_scene(
            axis=axis, dispersive=dispersive, directional=directional
        )
        gprMax.run(
            scenes=[scene],
            outputfile=tmp_path / str(directional),
            cpu_precision="double",
            hide_progress_bars=True,
        )
        _verify_solve(sar, rad, axis, dispersive)
        records.append(rad)
    np.testing.assert_array_equal(
        records[0]._monitor.accumulators["E" + axis], records[1]._monitor.accumulators["E" + axis]
    )
    np.testing.assert_array_equal(
        records[0].result.absorbed_power_density, records[1].result.absorbed_power_density
    )


@pytest.mark.parametrize("densities", ((None, 1000, 1000), (900, 1000, 1000)))
def test_directional_sar_rejects_missing_or_conflicting_density(densities, tmp_path):
    scene, _, _ = directional_scene(densities=densities)
    with pytest.raises(ValueError, match="same density on all three"):
        gprMax.run(scenes=[scene], outputfile=tmp_path / "invalid", hide_progress_bars=True)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ("cuda", "opencl"))
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("dispersive", (False, True))
def test_directional_device_solve(tmp_path, request, backend, precision, dispersive):
    scene, sar, rad = directional_scene(dispersive=dispersive)
    options = {
        "gpu"
        if backend == "cuda"
        else "opencl": [
            request.getfixturevalue("gpu_device" if backend == "cuda" else "opencl_device")
        ]
    }
    gprMax.run(
        scenes=[scene],
        outputfile=tmp_path / backend,
        gpu_precision=precision,
        hide_progress_bars=True,
        **options,
    )
    assert rad._monitor.collection_backend == f"{backend}_device"
    _verify_solve(sar, rad, "z", dispersive)
    expected = rad.result.absorbed_power_density.copy()
    scene, sar, rad = directional_scene(dispersive=dispersive)
    gprMax.run(
        scenes=[scene],
        outputfile=tmp_path / "cpu",
        cpu_precision=precision,
        hide_progress_bars=True,
    )
    np.testing.assert_allclose(
        expected, rad.result.absorbed_power_density, rtol=5e-4 if precision == "single" else 5e-7
    )


def three_d_scene(*, geometry=None, export=None):
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=(0.048, 0.032, 0.032)),
        gprMax.Discretisation(p1=(0.002,) * 3),
        gprMax.PMLThickness(thickness=2),
        gprMax.TimeWindow(time=2e-9),
        gprMax.OMPThreads(1),
        gprMax.Waveform(wave_type="ricker", amp=1, freq=1e9, id="pulse"),
        gprMax.HertzianDipole(p1=(0.01, 0.014, 0.014), polarisation="z", waveform_id="pulse"),
    ):
        scene.add(obj)
    for axis, sigma in zip("xyz", (0.1, 0.3, 0.6)):
        # On import, these unrelated, identically named materials must not
        # replace the tensor's original constitutive definitions.
        scene.add(gprMax.Material(er=99 if geometry else 2, se=sigma, mr=1, sm=0, id=axis))
        scene.add(gprMax.MaterialDensity(density=1000, material_ids=axis))
    if geometry:
        for _ in range(2):
            scene.add(
                gprMax.GeometryObjectsRead(
                    p1=(0, 0, 0),
                    geofile=str(geometry.with_suffix(".h5")),
                    material_database=geometry.name + "_materials",
                )
            )
    else:
        scene.add(
            gprMax.AddDebyeDispersion(poles=1, er_delta=[2], tau=[1e-10], material_ids=["z"])
        )
        scene.add(
            gprMax.Box(
                p1=(0.014, 0.01, 0.01),
                p2=(0.022, 0.022, 0.022),
                material_ids=list("xyz"),
                tag="target",
            )
        )
        scene.add(
            gprMax.Box(
                p1=(0.028, 0.01, 0.01),
                p2=(0.036, 0.022, 0.022),
                material_ids=list("zyx"),
                tag="target",
            )
        )
    if export:
        scene.add(
            gprMax.GeometryObjectsWrite(
                p1=(0, 0, 0), p2=(0.048, 0.032, 0.032), filename=str(export)
            )
        )
    outputs = []
    for cls in (gprMax.SAR, gprMax.Radiometry):
        output = cls(
            frequencies=(0.75e9, 1e9, 1.25e9),
            waveform_id="pulse",
            tags="target",
            id="dose" if cls is gprMax.SAR else "rad",
        )
        scene.add(output)
        outputs.append(output)
    return scene, *outputs


def test_directional_geometry_roundtrip_and_fixed_reuse(tmp_path):
    geometry = tmp_path / "tensor"
    scene, sar, rad = three_d_scene(export=geometry)
    gprMax.run(
        scenes=[scene],
        outputfile=tmp_path / "reference",
        cpu_precision="double",
        hide_progress_bars=True,
    )
    expected = rad.result.absorbed_power_density.copy()
    omega = 2 * np.pi * rad.result.frequency
    debye_loss = config.e0 * 2 * omega**2 * 1e-10 / (1 + (omega * 1e-10) ** 2)
    right = rad.result.cell_indices[:, 0] >= 12
    independent = np.zeros_like(expected)
    for component in ("Ex", "Ey", "Ez"):
        e = np.mean(
            rad._monitor.accumulators[component][:, rad._monitor.cell_edge_indices[component]],
            axis=2,
        )
        low = np.full(omega.size, 0.1)
        high = 0.6 + debye_loss
        sigma = (
            np.where(right[None, :], high[:, None], low[:, None])
            if component == "Ex"
            else np.where(right[None, :], low[:, None], high[:, None])
            if component == "Ez"
            else 0.3
        )
        independent += 0.5 * sigma * np.abs(e * rad.result.normalisation_scale[:, None]) ** 2
    np.testing.assert_allclose(expected, independent, rtol=1e-14)
    expected_fields = {
        component: values.copy() for component, values in rad._monitor.accumulators.items()
    }
    scene, sar, rad = three_d_scene(geometry=geometry, export=tmp_path / "reexport")
    gprMax.run(
        scenes=[scene],
        n=2,
        geometry_fixed=True,
        outputfile=tmp_path / "reused",
        cpu_precision="double",
        hide_progress_bars=True,
    )
    for component, values in expected_fields.items():
        np.testing.assert_array_equal(rad._monitor.accumulators[component], values)
    for run in (1, 2):
        with h5py.File(tmp_path / f"reused{run}.h5") as output:
            np.testing.assert_array_equal(
                output["radiometry/rad/absorbed_power_density"], expected
            )
            np.testing.assert_allclose(output["sar/dose/sar"], expected / 1000, rtol=1e-14)
    # A re-export uses new keys/names but must still reconstruct the tensors.
    scene, sar, rad = three_d_scene(geometry=tmp_path / "reexport")
    gprMax.run(
        scenes=[scene],
        outputfile=tmp_path / "again",
        cpu_precision="double",
        hide_progress_bars=True,
    )
    np.testing.assert_array_equal(rad.result.absorbed_power_density, expected)


def test_directional_geometry_rejects_voxel_only_fallback(tmp_path):
    geometry = tmp_path / "tensor"
    scene, _, _ = three_d_scene(export=geometry)
    gprMax.run(
        scenes=[scene],
        outputfile=tmp_path / "reference",
        geometry_only=True,
        hide_progress_bars=True,
    )
    with h5py.File(geometry.with_suffix(".h5"), "a") as data:
        del data["ID"]
    scene, _, _ = three_d_scene(geometry=geometry)
    with pytest.raises(
        ValueError, match="voxel-only scalar reconstruction would lose its anisotropy"
    ):
        gprMax.run(
            scenes=[scene],
            outputfile=tmp_path / "invalid",
            geometry_only=True,
            hide_progress_bars=True,
        )


@pytest.mark.parametrize("references", (None, [], ["missing"] * 3, "xyz"))
def test_directional_geometry_rejects_invalid_references(tmp_path, references):
    geometry = tmp_path / "tensor"
    scene, _, _ = three_d_scene(export=geometry)
    gprMax.run(
        scenes=[scene],
        outputfile=tmp_path / "reference",
        geometry_only=True,
        hide_progress_bars=True,
    )
    path = tmp_path / "tensor_materials.json"
    database = json.loads(path.read_text())
    for entry in database["materials"].values():
        if "directional_materials" in entry.get("metadata", {}):
            entry["metadata"]["directional_materials"] = references
    path.write_text(json.dumps(database), encoding="utf-8")
    scene, _, _ = three_d_scene(geometry=geometry)
    with pytest.raises(ValueError, match="Invalid x/y/z directional material references"):
        gprMax.run(
            scenes=[scene],
            outputfile=tmp_path / "invalid",
            geometry_only=True,
            hide_progress_bars=True,
        )


def _nest_directional_database(path):
    database = json.loads(path.read_text())
    entries = database["materials"]
    outer_key = next(
        key
        for key, entry in entries.items()
        if "directional_materials" in entry.get("metadata", {})
    )
    outer = entries[outer_key]
    references = outer["metadata"]["directional_materials"]
    nested_keys = []
    for axis, reference in enumerate(references):
        nested_key = f"nested_axis_{axis}"
        nested = json.loads(json.dumps(entries[reference]))
        nested["name"] = nested_key
        nested.setdefault("metadata", {})["original_id"] = nested_key
        entries[nested_key] = nested
        nested_keys.append(nested_key)
    middle_key = "nested_middle"
    middle = json.loads(json.dumps(outer))
    middle["name"] = middle_key
    middle["metadata"]["directional_materials"] = nested_keys
    middle["metadata"]["original_id"] = middle_key
    entries[middle_key] = middle
    outer["metadata"]["directional_materials"] = [middle_key] * 3
    path.write_text(json.dumps(database), encoding="utf-8")
    return outer_key, middle_key


def test_directional_geometry_resolves_nested_references(tmp_path):
    geometry = tmp_path / "tensor"
    scene, _, rad = three_d_scene(export=geometry)
    gprMax.run(
        scenes=[scene],
        outputfile=tmp_path / "reference",
        cpu_precision="double",
        hide_progress_bars=True,
    )
    expected = rad.result.absorbed_power_density.copy()
    _nest_directional_database(tmp_path / "tensor_materials.json")
    scene, _, rad = three_d_scene(geometry=geometry)
    gprMax.run(
        scenes=[scene],
        outputfile=tmp_path / "nested",
        cpu_precision="double",
        hide_progress_bars=True,
    )
    np.testing.assert_array_equal(rad.result.absorbed_power_density, expected)


def test_directional_geometry_rejects_nested_reference_cycles(tmp_path):
    geometry = tmp_path / "tensor"
    scene, _, _ = three_d_scene(export=geometry)
    gprMax.run(
        scenes=[scene],
        outputfile=tmp_path / "reference",
        geometry_only=True,
        hide_progress_bars=True,
    )
    outer_key, middle_key = _nest_directional_database(tmp_path / "tensor_materials.json")
    path = tmp_path / "tensor_materials.json"
    database = json.loads(path.read_text())
    database["materials"][middle_key]["metadata"]["directional_materials"] = [outer_key] * 3
    path.write_text(json.dumps(database), encoding="utf-8")
    scene, _, _ = three_d_scene(geometry=geometry)
    with pytest.raises(ValueError, match="Invalid x/y/z directional material references"):
        gprMax.run(
            scenes=[scene],
            outputfile=tmp_path / "invalid",
            geometry_only=True,
            hide_progress_bars=True,
        )


def test_directional_geometry_rejects_nested_density_conflict(tmp_path):
    geometry = tmp_path / "tensor"
    scene, _, _ = three_d_scene(export=geometry)
    gprMax.run(
        scenes=[scene],
        outputfile=tmp_path / "reference",
        geometry_only=True,
        hide_progress_bars=True,
    )
    _, middle_key = _nest_directional_database(tmp_path / "tensor_materials.json")
    path = tmp_path / "tensor_materials.json"
    database = json.loads(path.read_text())
    database["materials"][middle_key]["mass_density_kg_per_m3"] = 2000
    path.write_text(json.dumps(database), encoding="utf-8")
    scene, _, _ = three_d_scene(geometry=geometry)
    with pytest.raises(ValueError, match="Directional cell density"):
        gprMax.run(
            scenes=[scene],
            outputfile=tmp_path / "invalid",
            geometry_only=True,
            hide_progress_bars=True,
        )


def test_directional_record_cannot_be_loaded_as_scalar_material(tmp_path, monkeypatch):
    geometry = tmp_path / "tensor"
    scene, _, _ = three_d_scene(export=geometry)
    gprMax.run(
        scenes=[scene],
        outputfile=tmp_path / "reference",
        geometry_only=True,
        hide_progress_bars=True,
    )
    database = json.loads((tmp_path / "tensor_materials.json").read_text())
    key = next(
        key
        for key, entry in database["materials"].items()
        if "directional_materials" in entry.get("metadata", {})
    )
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=(0.02,) * 3),
        gprMax.Discretisation(p1=(0.001,) * 3),
        gprMax.TimeWindow(iterations=1),
        gprMax.PMLThickness(thickness=0),
    ):
        scene.add(obj)
    scene.add(gprMax.MaterialFromDatabase(database="tensor_materials", material=key, id="bad"))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="not loaded as scalar materials"):
        gprMax.run(
            scenes=[scene],
            outputfile=tmp_path / "invalid",
            geometry_only=True,
            hide_progress_bars=True,
        )


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ("cuda", "opencl"))
@pytest.mark.parametrize("precision", ("single", "double"))
def test_directional_device_3d_solve(tmp_path, request, backend, precision):
    results = []
    for device in (False, True):
        scene, _, rad = three_d_scene()
        options = {"cpu_precision": precision}
        if device:
            options = {
                "gpu_precision": precision,
                "gpu"
                if backend == "cuda"
                else "opencl": [
                    request.getfixturevalue("gpu_device" if backend == "cuda" else "opencl_device")
                ],
            }
        gprMax.run(
            scenes=[scene],
            outputfile=tmp_path / (backend if device else "cpu"),
            hide_progress_bars=True,
            **options,
        )
        results.append(rad.result.absorbed_power_density)
    np.testing.assert_allclose(
        results[1], results[0], rtol=5e-4 if precision == "single" else 5e-7
    )
