"""Reusable fixed-topology stateful-source studies."""

import json

import h5py
import numpy as np
import pytest

import gprMax


def _transmission_line_scene(amplitude=1.0):
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.012,) * 3))
    scene.add(gprMax.Discretisation(p1=(1e-3,) * 3))
    scene.add(gprMax.TimeWindow(time=2e-10))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.Waveform(wave_type="gaussian", amp=amplitude, freq=2e10, id="w"))
    source = gprMax.TransmissionLine(
        polarisation="z",
        p1=(0.006,) * 3,
        resistance=50,
        waveform_id="w",
    )
    scene.add(source)
    scene.add(gprMax.Rx(p1=(0.007, 0.006, 0.006), id="field"))
    return scene, source


def _magnetic_frill_scene(amplitude=1.0):
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl,) * 3))
    scene.add(gprMax.Domain(p1=(0.02,) * 3))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=2e-10))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=amplitude, freq=10e9, id="w"))
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.02, 0.02, dl), material_id="pec"))
    scene.add(
        gprMax.ThinWire(
            p1=(0.01, 0.01, 0),
            p2=(0.01, 0.01, 0.01),
            radius=0.1e-3,
        )
    )
    source = gprMax.MagneticFrillSource(
        p1=(0.01, 0.01, 0),
        polarisation="z",
        zcoax=50,
        waveform_id="w",
        start=0,
        stop=8e-11,
    )
    scene.add(source)
    scene.add(gprMax.Rx(p1=(0.014, 0.01, 0.005), id="field"))
    return scene, source


def _network_scene(amplitude=1.0, waveform_id="w", start=None, stop=None):
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.02,) * 3))
    scene.add(gprMax.Discretisation(p1=(0.002,) * 3))
    scene.add(gprMax.TimeWindow(time=4e-10))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=amplitude, freq=5e9, id="w"))
    scene.add(gprMax.Waveform(wave_type="gaussian", amp=amplitude, freq=3e9, id="w_alt"))
    scene.add(gprMax.RationalNetwork(id="source_50", conductance=1 / 50))
    scene.add(
        gprMax.NetworkTerminal(
            p1=(0.01,) * 3,
            polarisation="z",
            network_id="source_50",
            id="feed",
        )
    )
    excitation = gprMax.NetworkExcitation("feed", waveform_id, start=start, stop=stop)
    scene.add(excitation)
    scene.add(gprMax.NetworkPort("feed", reference_impedance=50))
    scene.add(gprMax.Rx(p1=(0.012, 0.01, 0.01), id="field"))
    return scene, excitation


def _two_network_scene(amplitudes=(1.0, 1.0)):
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.024, 0.02, 0.02)))
    scene.add(gprMax.Discretisation(p1=(0.002,) * 3))
    scene.add(gprMax.TimeWindow(time=4e-10))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=amplitudes[0], freq=5e9, id="w1"))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=amplitudes[1], freq=5e9, id="w2"))
    excitations = []
    for index, (position, waveform_id) in enumerate(
        (((0.008, 0.01, 0.01), "w1"), ((0.014, 0.01, 0.01), "w2")), start=1
    ):
        terminal_id = f"feed{index}"
        network_id = f"source{index}"
        scene.add(gprMax.RationalNetwork(id=network_id, conductance=1 / 50))
        scene.add(
            gprMax.NetworkTerminal(
                p1=position,
                polarisation="z",
                network_id=network_id,
                id=terminal_id,
            )
        )
        excitation = gprMax.NetworkExcitation(terminal_id, waveform_id)
        scene.add(excitation)
        scene.add(gprMax.NetworkPort(terminal_id, reference_impedance=50))
        excitations.append(excitation)
    scene.add(gprMax.Rx(p1=(0.012, 0.01, 0.01), id="field"))
    return scene, excitations


def _run_reused(tmp_path, name, factory, *, precision="double", **backend):
    scene, source = factory()
    study = gprMax.SourceStudy(
        [
            gprMax.StudyCase("full", [gprMax.ObjectState(source, scale=1)]),
            gprMax.StudyCase("half", [gprMax.ObjectState(source, scale=0.5)]),
        ]
    )
    gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=tmp_path / name,
        hide_progress_bars=True,
        log_level=30,
        **({"cpu_precision": precision} if not backend else {"gpu_precision": precision}),
        **backend,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("name", "factory", "paths"),
    [
        (
            "tl",
            _transmission_line_scene,
            ("tls/tl1/Vinc", "tls/tl1/Vtotal", "tls/tl1/Itotal", "rxs/rx1/Ez"),
        ),
        (
            "frill",
            _magnetic_frill_scene,
            ("frills/frill1/Vinc", "frills/frill1/Vtotal", "frills/frill1/Itot", "rxs/rx1/Ez"),
        ),
        (
            "network",
            _network_scene,
            ("ports/feed/Vgenerator", "ports/feed/Vtotal", "ports/feed/Inetwork", "rxs/rx1/Ez"),
        ),
    ],
)
def test_source_study_matches_fresh_runs(tmp_path, name, factory, paths):
    _run_reused(tmp_path, f"reused_{name}", factory)
    for index, amplitude in enumerate((1.0, 0.5), start=1):
        fresh, _ = factory(amplitude=amplitude)
        gprMax.run(
            scenes=[fresh],
            outputfile=tmp_path / f"fresh_{name}_{index}",
            hide_progress_bars=True,
            log_level=30,
            cpu_precision="double",
        )
        with h5py.File(tmp_path / f"reused_{name}{index}.h5") as reused, h5py.File(
            tmp_path / f"fresh_{name}_{index}.h5"
        ) as expected:
            for path in paths:
                np.testing.assert_allclose(reused[path], expected[path], rtol=3e-12, atol=3e-12)
            resolved = json.loads(reused["study/resolved_case"][()].decode())
            assert next(iter(resolved["objects"].values()))["scale"] == amplitude


@pytest.mark.integration
def test_source_study_multiple_drives_and_passive_omitted_terminal(tmp_path):
    scene, (feed1, feed2) = _two_network_scene()
    study = gprMax.SourceStudy(
        [
            gprMax.StudyCase("feed1", [gprMax.ObjectState(feed1, scale=1)]),
            gprMax.StudyCase(
                "both",
                [gprMax.ObjectState(feed1, scale=1), gprMax.ObjectState(feed2, scale=0.5)],
            ),
        ]
    )
    gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=tmp_path / "reused_two",
        hide_progress_bars=True,
        log_level=30,
        cpu_precision="double",
    )

    for index, amplitudes in enumerate(((1.0, 0.0), (1.0, 0.5)), start=1):
        fresh, _ = _two_network_scene(amplitudes=amplitudes)
        gprMax.run(
            scenes=[fresh],
            outputfile=tmp_path / f"fresh_two_{index}",
            hide_progress_bars=True,
            log_level=30,
            cpu_precision="double",
        )
        with h5py.File(tmp_path / f"reused_two{index}.h5") as reused, h5py.File(
            tmp_path / f"fresh_two_{index}.h5"
        ) as expected:
            for path in (
                "ports/feed1/Vgenerator",
                "ports/feed1/Vtotal",
                "ports/feed2/Vgenerator",
                "ports/feed2/Vtotal",
                "rxs/rx1/Ez",
            ):
                np.testing.assert_allclose(reused[path], expected[path], rtol=3e-12, atol=3e-12)
            if index == 1:
                assert np.all(reused["ports/feed2/Vgenerator"][...] == 0)
                assert np.max(np.abs(reused["ports/feed2/Vtotal"][...])) > 0


@pytest.mark.integration
def test_source_study_changes_waveform_and_drive_window_absolutely(tmp_path):
    scene, excitation = _network_scene()
    study = gprMax.SourceStudy(
        [
            gprMax.StudyCase("baseline", [gprMax.ObjectState(excitation)]),
            gprMax.StudyCase(
                "alternate",
                [
                    gprMax.ObjectState(
                        excitation,
                        waveform_id="w_alt",
                        start=2e-11,
                        stop=2e-10,
                        scale=0.75,
                    )
                ],
            ),
        ]
    )
    gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=tmp_path / "reused_waveform",
        hide_progress_bars=True,
        log_level=30,
        cpu_precision="double",
    )

    fresh, _ = _network_scene(
        amplitude=0.75,
        waveform_id="w_alt",
        start=2e-11,
        stop=2e-10,
    )
    gprMax.run(
        scenes=[fresh],
        outputfile=tmp_path / "fresh_waveform",
        hide_progress_bars=True,
        log_level=30,
        cpu_precision="double",
    )
    with h5py.File(tmp_path / "reused_waveform2.h5") as reused, h5py.File(
        tmp_path / "fresh_waveform.h5"
    ) as expected:
        for path in ("ports/feed/Vgenerator", "ports/feed/Vtotal", "rxs/rx1/Ez"):
            np.testing.assert_allclose(reused[path], expected[path], rtol=3e-12, atol=3e-12)
        resolved = json.loads(reused["study/resolved_case"][()].decode())
        applied = resolved["objects"]["network_excitation_1"]
        assert applied["waveform_id"] == "w_alt"
        assert applied["start"] == 2e-11
        assert applied["stop"] == 2e-10
        assert applied["scale"] == 0.75


@pytest.mark.integration
def test_hash_source_study_runs_all_csv_cases(tmp_path):
    cases = tmp_path / "source_cases.csv"
    cases.write_text(
        "case_id,object_id,active,scale\n"
        "full,network_excitation_1,true,1\n"
        "passive,network_excitation_1,false,1\n"
    )
    inputfile = tmp_path / "source_study.in"
    inputfile.write_text(
        "#title: reusable stateful-source study\n"
        "#dx_dy_dz: 0.002 0.002 0.002\n"
        "#domain: 0.02 0.02 0.02\n"
        "#pml_cells: 2\n"
        "#time_window: 4e-10\n"
        "#waveform: ricker 1 5e9 w\n"
        "#rational_network: source50 0.02 0 0\n"
        "#network_terminal: z 0.01 0.01 0.01 source50 feed\n"
        "#network_excitation: feed w\n"
        "#network_port: feed 50\n"
        f"#study: source {cases.name}\n"
    )
    gprMax.run(
        inputfile=inputfile,
        outputfile=tmp_path / "hash_source",
        hide_progress_bars=True,
        log_level=30,
        cpu_precision="double",
    )

    for index, active in enumerate((True, False), start=1):
        with h5py.File(tmp_path / f"hash_source{index}.h5") as output:
            assert output["study"].attrs["Type"] == "source"
            resolved = json.loads(output["study/resolved_case"][()].decode())
            assert resolved["objects"]["network_excitation_1"]["active"] is active
            assert output["srcs/src1"].attrs["StudyID"] == "network_excitation_1"
            if active:
                assert np.max(np.abs(output["ports/feed/Vgenerator"][...])) > 0
            else:
                assert np.all(output["ports/feed/Vgenerator"][...] == 0)


def _network_ntff_scene(amplitude=1.0):
    scene, excitation = _network_scene(amplitude=amplitude)
    scene.add(
        gprMax.NTFFSurface(
            p1=(0.006,) * 3,
            p2=(0.012,) * 3,
            id="surface",
            origin=(0.01,) * 3,
        )
    )
    scene.add(gprMax.KSIRFrequencyTransform("surface", "spectrum", (5e9,), save_surface_dft=False))
    scene.add(
        gprMax.KSIRFarField(
            theta=(90,),
            phi=(0,),
            transform_id="spectrum",
            id="pattern",
            outputs=("Etheta", "Ephi"),
        )
    )
    return scene, excitation


@pytest.mark.integration
def test_source_study_recreates_declarative_ntff_state(tmp_path):
    _run_reused(tmp_path, "reused_ntff", _network_ntff_scene)
    path = "ntff/surface/frequency/spectrum/far_field/pattern/fields"
    for index, amplitude in enumerate((1.0, 0.5), start=1):
        fresh, _ = _network_ntff_scene(amplitude=amplitude)
        gprMax.run(
            scenes=[fresh],
            outputfile=tmp_path / f"fresh_ntff_{index}",
            hide_progress_bars=True,
            log_level=30,
            cpu_precision="double",
        )
        with h5py.File(tmp_path / f"reused_ntff{index}.h5") as reused, h5py.File(
            tmp_path / f"fresh_ntff_{index}.h5"
        ) as expected:
            for component in ("Etheta", "Ephi"):
                np.testing.assert_allclose(
                    reused[f"{path}/{component}"],
                    expected[f"{path}/{component}"],
                    rtol=2e-11,
                    atol=1e-18,
                )


def test_source_study_csv_and_validation(tmp_path):
    table = tmp_path / "sources.csv"
    table.write_text(
        "case_id,object_id,active,waveform_id,start_s,stop_s,scale\n"
        "full,transmission_line_1,true,w,0,2e-10,1\n"
        "passive,transmission_line_1,false,w,0,2e-10,1\n"
    )
    study = gprMax.Study.from_csv("source", table)
    assert isinstance(study, gprMax.SourceStudy)
    assert study.cases[1].states[0].parameters["active"] is False

    scene, source = _transmission_line_scene()
    invalid = gprMax.SourceStudy(
        [gprMax.StudyCase("move", [gprMax.ObjectState(source, position=(0.005,) * 3)])]
    )
    with pytest.raises(ValueError, match="does not support parameter.*position"):
        invalid.bind_scene(scene)


def test_source_study_rejects_subgrid_source():
    scene, source = _transmission_line_scene()
    subgrid = gprMax.SubGridHSG(
        p1=(0.003,) * 3,
        p2=(0.009,) * 3,
        ratio=3,
        id="fine",
    )
    subgrid.add(
        gprMax.HertzianDipole(
            polarisation="x",
            p1=(0.006,) * 3,
            waveform_id="w",
        )
    )
    scene.add(subgrid)
    study = gprMax.SourceStudy([gprMax.StudyCase("run", [gprMax.ObjectState(source, scale=1)])])
    with pytest.raises(ValueError, match="HertzianDipole on subgrid 'fine'"):
        study.bind_scene(scene)


def _read_reused_field(tmp_path, name):
    traces = []
    for index in (1, 2):
        with h5py.File(tmp_path / f"{name}{index}.h5") as output:
            traces.append(output["rxs/rx1/Ez"][...])
    return traces


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["cuda", "opencl"])
@pytest.mark.parametrize(
    ("name", "factory"),
    [
        ("tl", _transmission_line_scene),
        ("frill", _magnetic_frill_scene),
        ("network", _network_scene),
    ],
)
def test_device_source_study_matches_cpu(tmp_path, request, backend, name, factory):
    if backend == "cuda":
        options = {"gpu": [request.getfixturevalue("gpu_device")]}
    else:
        options = {"opencl": [request.getfixturevalue("opencl_device")]}
    cpu_name = f"cpu_{backend}_{name}"
    device_name = f"{backend}_{name}"
    _run_reused(tmp_path, cpu_name, factory, precision="single")
    _run_reused(tmp_path, device_name, factory, precision="single", **options)

    expected = _read_reused_field(tmp_path, cpu_name)
    actual = _read_reused_field(tmp_path, device_name)
    scale = max(float(np.max(np.abs(trace))) for trace in expected)
    for cpu_trace, device_trace in zip(expected, actual):
        np.testing.assert_allclose(
            device_trace,
            cpu_trace,
            rtol=4e-4,
            atol=max(scale, 1e-12) * 4e-4,
        )
