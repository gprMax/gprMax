"""End-to-end include/preflight and study-parameter regression coverage."""

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.hash_includes import static_hash_commands
from gprMax.studies import _find_hash_array_codebook, _find_hash_study

BASE = """#domain: 0.064 0.064 0.064
#dx_dy_dz: 0.002 0.002 0.002
#time_window: 24
#pml_cells: 0
#omp_threads: 1
#waveform: gaussian 1 1e10 w
#hertzian_dipole: z 0.032 0.032 0.032 w
#rx: 0.032 0.032 0.032 probe Ez
"""
BACKENDS = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.gpu),
    pytest.param("opencl", marks=pytest.mark.gpu),
]


def backend_options(request, backend):
    if backend == "cpu":
        return {"cpu_precision": "double"}
    index = request.getfixturevalue("gpu_device" if backend == "cuda" else "opencl_device")
    return {"gpu" if backend == "cuda" else "opencl": [index], "gpu_precision": "double"}


@pytest.mark.integration
@pytest.mark.parametrize("backend", BACKENDS)
def test_run_nested_includes_never_consults_cwd_study(tmp_path, monkeypatch, request, backend):
    model, caller = tmp_path / "model", tmp_path / "caller"
    nested = model / "nested"
    nested.mkdir(parents=True)
    caller.mkdir()
    source = model / "main.in"
    source.write_text(BASE + "#include_file: nested/outer.in\n")
    (nested / "outer.in").write_text("#include_file: inner.in\n")
    (nested / "inner.in").write_text("#material: 4 0 1 0 medium\n")
    (caller / "nested").mkdir()
    (caller / "nested/outer.in").write_text("#study: gpr phantom.csv\n")
    (model / "phantom.csv").write_text("case_id,object_id\noff,rx_1\n")
    monkeypatch.chdir(caller)
    gprMax.run(
        inputfile=source, hide_progress_bars=True, log_level=50, **backend_options(request, backend)
    )
    with h5py.File(source.with_suffix(".h5")) as output:
        assert "study" not in output
        assert np.count_nonzero(output["srcs/src1/excitation/samples"]) == 24
        assert np.count_nonzero(output["rxs/rx1/Ez"]) == 23


@pytest.mark.parametrize(
    "scanner, command",
    [
        (_find_hash_study, "#study: gpr cases.csv"),
        (_find_hash_array_codebook, "#array_codebook: modes.json"),
    ],
)
def test_preflight_shares_nested_directory_and_occurrence_policy(tmp_path, scanner, command):
    nested = tmp_path / "nested"
    nested.mkdir()
    main = tmp_path / "main.in"
    main.write_text("#include_file: nested/outer.in\n")
    (nested / "outer.in").write_text("#include_file: inner.in\n")
    (nested / "inner.in").write_text(command + "\n")
    result = scanner(main)
    path = result[1] if isinstance(result, tuple) else result
    assert path.parent == tmp_path  # Study CSV/codebook paths remain top-level relative.
    main.write_text("#include_file: nested/outer.in\n" * 2)
    with pytest.raises(ValueError, match="Only one"):
        scanner(main)


@pytest.mark.parametrize("alias", [False, True])
def test_preflight_rejects_include_cycles(tmp_path, alias):
    main, child = tmp_path / "main.in", tmp_path / "child.in"
    main.write_text("#include_file: child.in\n")
    if alias:
        try:
            (tmp_path / "alias.in").symlink_to(main)
        except OSError as error:
            if getattr(error, "winerror", None) != 1314:
                raise
            pytest.skip("Windows requires Developer Mode or symlink creation privilege")
    child.write_text(f"#include_file: {'alias.in' if alias else 'main.in'}\n")
    with pytest.raises(ValueError, match="cycle detected"):
        static_hash_commands(main)


def test_preflight_bounds_depth_and_does_not_execute_python(tmp_path):
    main = tmp_path / "main.in"
    main.write_text(
        "#python:\nraise RuntimeError('must not execute')\nprint('#study: gpr bad.csv')\n#end_python:\n#title: valid\n"
    )
    assert static_hash_commands(main) == ["#title: valid\n"]
    for index in range(101):
        (tmp_path / f"{index}.in").write_text(f"#include_file: {index+1}.in\n")
    with pytest.raises(ValueError, match="nesting exceeds"):
        static_hash_commands(tmp_path / "0.in")


@pytest.mark.parametrize("command", ["#study: gpr ignored.csv", "#array_codebook: ignored.json"])
def test_python_cannot_silently_introduce_run_controls(tmp_path, command):
    source = tmp_path / "dynamic.in"
    source.write_text(BASE + f"#python:\nprint({command!r})\n#end_python:\n")
    with pytest.raises(ValueError, match="must be declared literally"):
        gprMax.run(inputfile=source, hide_progress_bars=True, log_level=50)
    assert not list(tmp_path.glob("*.h5"))


@pytest.mark.parametrize(
    "study_type",
    [
        gprMax.GPRStudy,
        gprMax.SourceStudy,
        gprMax.PortStudy,
        gprMax.PlaneWaveStudy,
        gprMax.EigenmodeStudy,
    ],
)
def test_all_study_families_reject_common_invalid_parameters(study_type):
    study = study_type([gprMax.StudyCase("invalid", [gprMax.ObjectState("source", start=np.nan)])])
    with pytest.raises(ValueError, match="must be finite"):
        study.bind_scene(gprMax.Scene())


def study_scene(parameters):
    model = gprMax.Scene()
    source = gprMax.HertzianDipole(p1=(0.032,) * 3, polarisation="z", waveform_id="w")
    receiver = gprMax.Rx(p1=(0.032,) * 3, id="probe", outputs=["Ez"])
    for item in (
        gprMax.Domain(p1=(0.064,) * 3),
        gprMax.Discretisation(p1=(0.002,) * 3),
        gprMax.TimeWindow(iterations=24),
        gprMax.PMLThickness(thickness=0),
        gprMax.OMPThreads(1),
        gprMax.Waveform(wave_type="gaussian", amp=1, freq=1e10, id="w"),
        source,
        receiver,
    ):
        model.add(item)
    study = gprMax.GPRStudy([gprMax.StudyCase("case", [gprMax.ObjectState(source, **parameters)])])
    return model, study, receiver


@pytest.mark.parametrize("parameter", ["start", "stop"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_api_study_rejects_nonfinite_time_before_output(tmp_path, parameter, value):
    model, study, _ = study_scene({parameter: value})
    with pytest.raises(ValueError, match="must be finite"):
        gprMax.run(scenes=[model], study=study, outputfile=tmp_path / "invalid", log_level=50)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("parameter", ["active", "record"])
@pytest.mark.parametrize("value", [0, 1, "false", None, [], np.array([False])])
def test_study_rejects_nonboolean_flags(parameter, value):
    _, study, _ = study_scene({parameter: value})
    with pytest.raises(ValueError, match="must be a boolean"):
        study.reset_runtime()


@pytest.mark.integration
@pytest.mark.parametrize("value", [False, np.bool_(False), True, np.bool_(True)])
def test_api_study_python_and_numpy_flags_agree(tmp_path, value):
    model, study, _ = study_scene({"active": value})
    output = tmp_path / "study.h5"
    gprMax.run(
        scenes=[model], study=study, outputfile=output, hide_progress_bars=True, log_level=50
    )
    with h5py.File(output) as handle:
        assert bool(np.any(handle["srcs/src1/excitation/samples"])) == bool(value)
        assert bool(np.any(handle["rxs/rx1/Ez"])) == bool(value)


@pytest.mark.integration
def test_numpy_record_false_omits_receiver_data(tmp_path):
    model, study, receiver = study_scene({})
    study.cases[0].states.append(gprMax.ObjectState(receiver, record=np.bool_(False)))
    gprMax.run(scenes=[model], study=study, outputfile=tmp_path / "record", log_level=50)
    with h5py.File(tmp_path / "record.h5") as handle:
        assert not bool(handle["rxs/rx1"].attrs["Recorded"])
        assert "Ez" not in handle["rxs/rx1"]
