"""Scene definitions must not own runtime grids after construction or a run."""

import gc
import weakref

import h5py
import numpy as np
import pytest

import gprMax
from gprMax import config
from gprMax.contexts import Context
from gprMax.model import Model
from gprMax.subgrids.user_objects import SubGridBase

pytestmark = pytest.mark.integration
BACKENDS = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]


def _scene(ratio, subgrids=1):
    scene = gprMax.Scene()
    for obj in (
        gprMax.Domain(p1=(0.120, 0.072, 0.072)),
        gprMax.Discretisation(p1=(0.003,) * 3),
        gprMax.TimeWindow(iterations=24),
        gprMax.PMLThickness(thickness=4),
        gprMax.OMPThreads(1),
    ):
        scene.add(obj)
    for index in range(subgrids):
        offset = index * 0.048
        fine = gprMax.SubGridHSG(
            p1=(0.024 + offset, 0.024, 0.024),
            p2=(0.042 + offset, 0.042, 0.042),
            ratio=ratio,
            id=f"fine{index}",
        )
        fine.add(gprMax.Waveform(wave_type="gaussian", amp=1, freq=1e10, id="pulse"))
        fine.add(
            gprMax.VoltageSource(
                p1=(0.033 + offset, 0.033, 0.033),
                polarisation="z",
                resistance=50,
                waveform_id="pulse",
                id="feed",
            )
        )
        fine.add(gprMax.Rx(p1=(0.036 + offset, 0.033, 0.033), id="inner"))
        scene.add(fine)
    scene.add(gprMax.Rx(p1=(0.054, 0.036, 0.033), id="outer"))
    return scene


def _run(scenes, path, *, gpu=None, **kwargs):
    return gprMax.run(
        scenes=scenes,
        outputfile=path,
        subgrid=True,
        autotranslate=True,
        gpu=None if gpu is None else [gpu],
        cpu_precision="double",
        gpu_precision="double",
        hide_progress_bars=True,
        log_level=50,
        **kwargs,
    )


@pytest.fixture
def runtime_references(monkeypatch):
    """Observe ownership without keeping grids or their arrays alive ourselves."""
    references = []
    setup = SubGridBase.setup

    def observe(user_object, grid, model):
        setup(user_object, grid, model)
        references.extend((weakref.ref(model.G), weakref.ref(grid)))

    monkeypatch.setattr(SubGridBase, "setup", observe)
    return references


def _assert_released(references):
    gc.collect()
    assert references
    assert all(reference() is None for reference in references)


def _check_sequential_release(monkeypatch, runtime_references, scenes, path, **kwargs):
    run_model = Context._run_model
    completed = []

    def observe(context, model_num):
        run_model(context, model_num)
        # Check between cases, not just when run() returns or a new run
        # replaces config.sim_config. Both the caller and config retain scenes.
        _assert_released(runtime_references)
        completed.append(model_num)

    monkeypatch.setattr(Context, "_run_model", observe)
    _run(scenes, path, n=len(scenes), **kwargs)
    assert len(completed) == len(scenes)
    assert config.sim_config.scenes is scenes
    assert all(obj.subgrid is None for scene in scenes for obj in scene.subgrid_objects)
    _assert_released(runtime_references)


@pytest.mark.parametrize("ratio", [1, 3])
@pytest.mark.parametrize("subgrids", [1, 2])
@pytest.mark.parametrize("geometry_only", [False, True])
def test_completed_scenes_release_all_grids(
    tmp_path, monkeypatch, runtime_references, ratio, subgrids, geometry_only
):
    scenes = [_scene(ratio, subgrids) for _ in range(3)]
    _check_sequential_release(
        monkeypatch,
        runtime_references,
        scenes,
        tmp_path / "scene",
        geometry_only=geometry_only,
    )
    assert len(runtime_references) == 2 * subgrids * len(scenes)


@pytest.mark.gpu
@pytest.mark.parametrize("ratio", [1, 3])
def test_cuda_completed_scenes_release_all_grids(
    tmp_path, monkeypatch, runtime_references, gpu_device, ratio
):
    scenes = [_scene(ratio, subgrids=2) for _ in range(3)]
    _check_sequential_release(
        monkeypatch, runtime_references, scenes, tmp_path / "cuda", gpu=gpu_device
    )
    assert len(runtime_references) == 4 * len(scenes)


def test_single_scene_releases_grids_in_persistent_process(
    tmp_path, monkeypatch, runtime_references
):
    scenes = [_scene(3, subgrids=2)]
    _check_sequential_release(monkeypatch, runtime_references, scenes, tmp_path / "single")
    assert len(runtime_references) == 4


@pytest.mark.parametrize("ratio", [1, 3])
def test_scene_can_be_built_again_after_releasing_grids(
    tmp_path, monkeypatch, runtime_references, ratio
):
    scene = _scene(ratio)
    _check_sequential_release(monkeypatch, runtime_references, [scene] * 3, tmp_path / "repeated")
    assert len(runtime_references) == 6


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("ratio", [1, 3])
def test_geometry_fixed_keeps_grids_only_during_run_and_preserves_outputs(
    tmp_path, monkeypatch, runtime_references, request, backend, ratio
):
    gpu = request.getfixturevalue("gpu_device") if backend == "cuda" else None
    scenes = [_scene(ratio) for _ in range(3)]
    _run(scenes, tmp_path / "rebuilt", n=3, gpu=gpu)
    _assert_released(runtime_references)
    runtime_references.clear()

    builds = []
    build = Model.build

    def observe(model):
        build(model)
        assert all(obj.subgrid is None for obj in scenes[0].subgrid_objects)
        # The model, not the scene definition, owns this pair throughout reuse.
        assert runtime_references[0]() is model.G
        assert runtime_references[1]() is model.subgrids[0]
        builds.append(config.get_model_config().reuse_geometry())

    monkeypatch.setattr(Model, "build", observe)
    _run([scenes[0]], tmp_path / "reused", n=3, geometry_fixed=True, gpu=gpu)
    assert builds == [False, True, True]
    assert len(runtime_references) == 2
    _assert_released(runtime_references)

    for index in range(1, 4):
        with h5py.File(tmp_path / f"rebuilt{index}.h5") as expected, h5py.File(
            tmp_path / f"reused{index}.h5"
        ) as actual:
            names = []
            expected.visititems(
                lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None
            )
            assert "subgrids/fine0/ports/feed/Vtotal" in names
            assert np.max(np.abs(expected["subgrids/fine0/rxs/rx1/Ez"][...])) > 0
            for name in names:
                np.testing.assert_array_equal(actual[name][...], expected[name][...], err_msg=name)


@pytest.mark.parametrize(
    "stage", ["setup", "tags", "main_geometry", "children", "model_build", "solve"]
)
def test_failed_run_does_not_leave_grids_owned_by_scene(
    tmp_path, monkeypatch, runtime_references, stage
):
    scene = _scene(3, subgrids=2)
    if stage == "setup":
        setup = SubGridBase.setup

        def fail_second_setup(user_object, grid, model):
            setup(user_object, grid, model)
            if grid.name == "fine1":
                raise RuntimeError("injected build failure")

        monkeypatch.setattr(SubGridBase, "setup", fail_second_setup)
    else:
        owner, method = {
            "tags": (gprMax.Scene, "initialise_geometry_tags"),
            "main_geometry": (gprMax.Scene, "process_geometry_objects"),
            "children": (gprMax.Scene, "process_subgrid_objects"),
            "model_build": (Model, "build_geometry"),
            "solve": (Model, "solve"),
        }[stage]

        def fail(*args, **kwargs):
            raise RuntimeError("injected build failure")

        monkeypatch.setattr(owner, method, fail)

    def run_and_discard_exception():
        # Do not retain pytest's exception info: its traceback legitimately
        # owns the failed model while an application is inspecting the error.
        with pytest.raises(RuntimeError, match="injected build failure"):
            _run([scene], tmp_path / "failed")

    run_and_discard_exception()
    assert all(obj.subgrid is None for obj in scene.subgrid_objects)
    assert len(runtime_references) == 4
    _assert_released(runtime_references)
