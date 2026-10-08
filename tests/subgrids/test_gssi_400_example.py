"""Keep the shipped subgrid example compatible with the contributed antenna."""

from pathlib import Path
import runpy

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.model import Model
from gprMax.toolboxes.GPRAntennaModels.GSSI import antenna_like_GSSI_400


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = (
    "examples/gpr/subgrids/gssi_400_over_fractal_subsurface.py",
    "reframe_tests/tests/src/subgrid_tests/gssi_400_over_fractal_subsurface.py",
)


def _load_example(monkeypatch, filename):
    calls = []
    with monkeypatch.context() as capture:
        capture.setattr(gprMax, "run", lambda **kwargs: calls.append(kwargs))
        example = runpy.run_path(str(ROOT / filename))

    assert len(calls) == 1
    assert calls[0]["scenes"] == [example["scene"]]
    assert calls[0]["subgrid"] is True
    assert calls[0]["autotranslate"] is True
    return example


@pytest.mark.unit
@pytest.mark.parametrize("filename", EXAMPLES)
def test_gssi_400_example_constructs_with_supported_antenna(monkeypatch, filename):
    """Execute every declaration, without allocating the full FDTD grids."""
    example = _load_example(monkeypatch, filename)

    assert example["dl_sg"] == 0.002
    assert example["dl"] == 0.010
    assert example["ratio"] == 5
    assert example["antenna_p"] == (1.5, 0.5, 1.53)

    # Preserve every antenna object and parameter, not just its outer size.
    expected = antenna_like_GSSI_400(*example["antenna_p"], resolution=0.002)
    actual = example["gssi_objects"]
    assert len(actual) == len(expected)
    for obj, reference in zip(actual, expected):
        assert type(obj) is type(reference)
        assert obj.kwargs == reference.kwargs

    fine = example["sg"]
    lower = np.asarray(fine.kwargs["p1"])
    upper = np.asarray(fine.kwargs["p2"])
    spacing = example["dl"]
    for point in (lower, upper):
        np.testing.assert_allclose(point / spacing, np.round(point / spacing), atol=1e-12, rtol=0)
    half_width = np.array([0.15, 0.15, 0])
    antenna_lower = np.array(example["antenna_p"]) - half_width
    antenna_upper = antenna_lower + example["antenna_case"]
    assert np.all(lower < antenna_lower)
    assert np.all(antenna_upper < upper)

    padding_cells = (
        fine.kwargs["is_os_sep"] * fine.kwargs["ratio"]
        + fine.kwargs["pml_separation"]
        + fine.kwargs["subgrid_pml_thickness"]
    )
    padding = padding_cells * example["dl_sg"]
    domain = np.array([example["x"], example["y"], example["z"]])
    assert np.all(lower - padding > 0)
    assert np.all(upper + padding < domain)

    soil = example["b2"]
    assert soil.autotranslate is False
    assert soil.kwargs["p1"] == (0, 0, 0)
    np.testing.assert_allclose(soil.kwargs["p2"][:2], (upper - lower + 2 * padding)[:2])
    # Convert the local soil surface back to global coordinates.
    assert soil.kwargs["p2"][2] + lower[2] - padding == pytest.approx(example["antenna_p"][2])
    assert example["b1"].kwargs["p2"][2] == example["antenna_p"][2]

    main_view = example["gv1"]
    assert main_view.kwargs["p2"] == tuple(domain)
    assert main_view.kwargs["dl"] == (spacing,) * 3
    snapshots = [obj for obj in example["scene"].output_objects if isinstance(obj, gprMax.Snapshot)]
    assert len(snapshots) == 50
    assert all(0 < obj.time <= example["tw"] for obj in snapshots)


@pytest.mark.unit
def test_gssi_400_reframe_scene_matches_downloadable_example():
    """The standalone ReFrame copy differs only by its copyright header."""
    scripts = [(ROOT / filename).read_text() for filename in EXAMPLES]
    assert scripts[0] == scripts[1][scripts[1].index('"""') :]


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    "geometry_only,backend",
    [(True, "cpu"), (False, "cpu"), pytest.param(False, "cuda", marks=pytest.mark.gpu)],
)
def test_gssi_400_example_builds_and_runs(tmp_path, monkeypatch, request, geometry_only, backend):
    """Build the complete geometry; shorten only the solve for smoke coverage."""
    gpu = request.getfixturevalue("gpu_device") if backend == "cuda" else None
    example = _load_example(monkeypatch, EXAMPLES[0])
    scene = example["scene"]
    scene.add(gprMax.OMPThreads(2))
    if not geometry_only:
        scene.single_use_objects = [obj for obj in scene.single_use_objects if not isinstance(obj, gprMax.TimeWindow)]
        scene.add(gprMax.TimeWindow(iterations=128))
        # Keep all snapshot definitions, rescheduled within the shorter run.
        snapshots = [obj for obj in scene.output_objects if isinstance(obj, gprMax.Snapshot)]
        for index, snapshot in enumerate(snapshots, start=1):
            snapshot.time = None
            snapshot.iterations = round(index * 127 / len(snapshots))

    built = []
    build = Model.build

    def inspect(model):
        build(model)
        main = model.G
        fine = model.subgrids[0]
        np.testing.assert_array_equal(main.size, (300, 100, 200))
        np.testing.assert_array_equal(fine.size, (220, 220, 160))
        np.testing.assert_allclose(fine.dl, (0.002,) * 3)
        assert main.dt / fine.dt == pytest.approx(5)
        soil = next(mat.numID for mat in fine.materials if mat.ID == "sandy_soil")
        # Include every transverse cell, right through the padding/PML.
        assert np.all(fine.solid[:, :, :35] == soil)
        assert fine.local_to_global((0, 0, 35))[2] == pytest.approx(1.53)
        assert len(fine.voltagesources) == 1
        feed = fine.voltagesources[0]
        np.testing.assert_allclose(fine.local_to_global(feed.coord), (1.42, 0.498, 1.538))
        rx = next(rx for rx in fine.rxs if rx.ID == "rxbowtie")
        np.testing.assert_allclose(fine.local_to_global(rx.coord), (1.582, 0.498, 1.538))
        assert len(main.snapshots) == 50
        assert all(0 <= snapshot.iteration < main.iterations for snapshot in main.snapshots)
        built.append(fine.iterations)

    monkeypatch.setattr(Model, "build", inspect)
    gprMax.run(
        scenes=[scene],
        outputfile=tmp_path / "model",
        geometry_only=geometry_only,
        subgrid=True,
        autotranslate=True,
        cpu_precision="double",
        gpu=None if gpu is None else [gpu],
        gpu_precision="double",
        hide_progress_bars=True,
        log_level=40,
    )
    assert len(built) == 1
    views = sorted(tmp_path.glob("*.vtkhdf"))
    assert len(views) == 2
    for view in views:
        with h5py.File(view) as output:
            assert "VTKHDF" in output
    if not geometry_only:
        with h5py.File(tmp_path / "model.h5") as output:
            receiver = output["subgrids/sg/rxs/rx1"]
            assert receiver.attrs["Name"] == "rxbowtie"
            trace = receiver["Ey"][...]
            assert trace.shape == (built[0],)
            assert np.isfinite(trace).all()
            assert np.max(np.abs(trace)) > 0
