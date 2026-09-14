"""End-to-end field controls for volume overwrite and transparent imports."""

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.model import Model

SHAPES = ("box", "sphere", "ellipsoid", "cylinder", "cone", "triangle", "sector")
FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _scene():
    scene = gprMax.Scene()
    for obj in (
        gprMax.Discretisation(p1=(0.001,) * 3),
        gprMax.Domain(p1=(0.020,) * 3),
        gprMax.PMLThickness(thickness=0),
        gprMax.TimeWindow(iterations=1000),
        gprMax.OMPThreads(1),
        gprMax.Material(er=4, se=0, mr=7, sm=0, id="magnetic"),
        gprMax.Waveform(wave_type="ricker", amp=1, freq=2e9, id="pulse"),
        gprMax.HertzianDipole(polarisation="z", p1=(0.006, 0.010, 0.010), waveform_id="pulse"),
        gprMax.Rx(p1=(0.014, 0.010, 0.010)),
    ):
        scene.add(obj)
    return scene


def _primitive(shape, material, averaging):
    kwargs = dict(material_id=material, averaging=averaging)
    if shape == "box":
        return gprMax.Box(p1=(0.008,) * 3, p2=(0.012,) * 3, **kwargs)
    if shape == "sphere":
        return gprMax.Sphere(p1=(0.010,) * 3, r=0.0032, **kwargs)
    if shape == "ellipsoid":
        return gprMax.Ellipsoid(p1=(0.010,) * 3, xr=0.0032, yr=0.0022, zr=0.0042, **kwargs)
    if shape == "cylinder":
        return gprMax.Cylinder(
            p1=(0.007, 0.010, 0.010), p2=(0.013, 0.010, 0.010), r=0.0022, **kwargs
        )
    if shape == "cone":
        return gprMax.Cone(
            p1=(0.007, 0.010, 0.010), p2=(0.013, 0.010, 0.010), r1=0.0012, r2=0.0032, **kwargs
        )
    if shape == "triangle":
        return gprMax.Triangle(
            p1=(0.007, 0.007, 0.008),
            p2=(0.013, 0.007, 0.008),
            p3=(0.010, 0.014, 0.008),
            thickness=0.004,
            **kwargs,
        )
    return gprMax.CylindricalSector(
        normal="z",
        ctr1=0.010,
        ctr2=0.010,
        extent1=0.008,
        extent2=0.012,
        r=0.0032,
        start=0,
        end=270,
        **kwargs,
    )


def _capture(monkeypatch):
    captured = {}
    original = Model.build

    def build(self):
        original(self)
        captured["grid"] = self.G

    monkeypatch.setattr(Model, "build", build)
    return captured


def _solve(scene, path, captured, **options):
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=path,
        hide_progress_bars=True,
        log_level=40,
        cpu_precision="double",
        **options,
    )
    with h5py.File(path.with_suffix(".h5")) as output:
        traces = np.stack([output[f"rxs/rx1/{field}"][:] for field in FIELDS])
    grid = captured["grid"]
    return {
        name: getattr(grid, name).copy() for name in ("solid", "ID", "rigidE", "rigidH")
    }, traces


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("averaging", [False, True])
def test_replaced_volume_has_identical_geometry_and_fields(tmp_path, monkeypatch, shape, averaging):
    captured = _capture(monkeypatch)
    reference = _scene()
    reference.add(_primitive(shape, "free_space", averaging))
    overwritten = _scene()
    overwritten.add(_primitive(shape, "magnetic", False))
    overwritten.add(_primitive(shape, "free_space", averaging))
    expected, expected_fields = _solve(reference, tmp_path / "reference", captured)
    actual, actual_fields = _solve(overwritten, tmp_path / "overwritten", captured)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
    assert np.linalg.norm(expected_fields) > 0
    np.testing.assert_array_equal(actual_fields, expected_fields)


def test_transparent_full_component_import_preserves_rigid_geometry_and_fields(
    tmp_path, monkeypatch
):
    captured = _capture(monkeypatch)
    path = tmp_path / "transparent.h5"
    with h5py.File(path, "w") as output:
        output.attrs["dx_dy_dz"] = (0.001,) * 3
        output["data"] = np.full((8, 8, 8), -1, np.int16)
        output["ID"] = np.full((6, 9, 9, 9), -1, np.int16)
        output["rigidE"] = np.zeros((12, 8, 8, 8), np.int8)
        output["rigidH"] = np.zeros((6, 8, 8, 8), np.int8)
    materials = tmp_path / "materials.txt"
    materials.write_text("#material: 1 0 1 0 free_space\n")
    from gprMax.toolboxes.MaterialDatabase import convert_geometry

    path, database = convert_geometry(path, materials)
    results = []
    for imported in (False, True):
        scene = _scene()
        scene.add(_primitive("box", "magnetic", False))
        if imported:
            scene.add(
                gprMax.GeometryObjectsRead(
                    p1=(0.006,) * 3, geofile=path, material_database=database.stem
                )
            )
        results.append(_solve(scene, tmp_path / f"imported_{imported}", captured))
    for name in results[0][0]:
        np.testing.assert_array_equal(results[1][0][name], results[0][0][name], err_msg=name)
    assert np.linalg.norm(results[0][1]) > 0
    np.testing.assert_array_equal(results[1][1], results[0][1])


@pytest.mark.gpu
@pytest.mark.parametrize("shape", SHAPES)
def test_cuda_uploaded_overwrite_geometry_and_fields_match_cpu(
    tmp_path, monkeypatch, gpu_device, shape
):
    from gprMax.grid.cuda_grid import CUDAGrid

    captured = _capture(monkeypatch)
    original = CUDAGrid.htod_geometry_arrays
    uploads = []

    def upload(self, *args, **kwargs):
        original(self, *args, **kwargs)
        np.testing.assert_array_equal(self.ID_dev.get(), self.ID)
        uploads.append(True)

    monkeypatch.setattr(CUDAGrid, "htod_geometry_arrays", upload)
    reference = _scene()
    reference.add(_primitive(shape, "free_space", True))
    expected, expected_fields = _solve(reference, tmp_path / "cpu", captured)
    overwritten = _scene()
    overwritten.add(_primitive(shape, "magnetic", False))
    overwritten.add(_primitive(shape, "free_space", True))
    actual, actual_fields = _solve(
        overwritten, tmp_path / "cuda", captured, gpu=[gpu_device], gpu_precision="double"
    )
    assert uploads == [True]
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
    for family in (slice(0, 3), slice(3, 6)):
        scale = np.linalg.norm(expected_fields[family])
        assert scale > 0
        assert np.linalg.norm(actual_fields[family] - expected_fields[family]) / scale < 2e-10
