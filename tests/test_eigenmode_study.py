"""Reusable eigenmode-port studies."""

from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax
import gprMax.sources as sources_module

INF = float("inf")


def _two_port_waveguide_scene(active_port=1, active_mode=1, modes=(1,)):
    multimode = max(modes) > 1
    domain_y = 0.08 if multimode else 0.05
    aperture_y = 0.075 if multimode else 0.045
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.06, domain_y, INF)))
    scene.add(gprMax.PMLThickness(thickness=(5, 0, 0, 5, 0, 0)))
    scene.add(gprMax.TimeWindow(time=0.4e-9))
    scene.add(gprMax.Waveform(wave_type="contsine", amp=1, freq=5e9, id="wave"))
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.06, 0.005, INF), material_id="pec"))
    scene.add(
        gprMax.Box(
            p1=(0, aperture_y, 0),
            p2=(0.06, domain_y, INF),
            material_id="pec",
        )
    )
    scene.add(gprMax.EigenmodeBand(id="band", fmin=5e9, fmax=5e9, points=1))
    scene.add(
        gprMax.EigenmodePort(
            port=1,
            p1=(0.010, 0.005, 0),
            p2=(0.010, aperture_y, INF),
            direction="+",
            modes=modes,
            anchors=(5e9,),
            plot_fields=False,
        )
    )
    scene.add(
        gprMax.EigenmodePort(
            port=2,
            p1=(0.050, 0.005, 0),
            p2=(0.050, aperture_y, INF),
            direction="-",
            modes=modes,
            anchors=(5e9,),
            plot_fields=False,
        )
    )
    excitation = gprMax.EigenmodeExcitation(
        port=active_port,
        mode=active_mode,
        waveform="wave",
        plot_waveform=False,
    )
    scene.add(excitation)
    return scene, excitation


def _study(excitation):
    return gprMax.EigenmodeStudy(
        [
            gprMax.StudyCase(
                "drive_port1_mode1",
                [gprMax.ObjectState(excitation, port=1, mode=1)],
            ),
            gprMax.StudyCase(
                "drive_port2_mode1",
                [gprMax.ObjectState(excitation, port=2, mode=1)],
            ),
        ]
    )


def _study_for_channels(excitation, channels):
    return gprMax.EigenmodeStudy(
        [
            gprMax.StudyCase(
                f"drive_port{port}_mode{mode}",
                [gprMax.ObjectState(excitation, port=port, mode=mode)],
            )
            for port, mode in channels
        ]
    )


@pytest.mark.integration
def test_eigenmode_study_matches_fresh_builds_without_resolving_modes(tmp_path, monkeypatch):
    solves = 0
    original_solve = sources_module.FDFD_1D_mode_solver.solve

    def counted_solve(solver):
        nonlocal solves
        solves += 1
        return original_solve(solver)

    monkeypatch.setattr(sources_module.FDFD_1D_mode_solver, "solve", counted_solve)
    scene, excitation = _two_port_waveguide_scene()
    study = _study(excitation)
    returned = gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=tmp_path / "reused",
        hide_progress_bars=True,
        log_level=30,
    )
    reused_solves = solves
    assert reused_solves == 2

    for port in (1, 2):
        fresh_scene, _ = _two_port_waveguide_scene(active_port=port)
        gprMax.run(
            scenes=[fresh_scene],
            outputfile=tmp_path / f"fresh{port}",
            hide_progress_bars=True,
            log_level=30,
        )

    assert returned["study"] is study.result
    assert study.result.s.shape[1:] == (2, 2)
    assert study.result.valid_s.any()
    for case_index in (1, 2):
        with h5py.File(tmp_path / f"reused{case_index}.h5") as reused, h5py.File(
            tmp_path / f"fresh{case_index}.h5"
        ) as fresh:
            response = reused["study/eigenmode_response"]
            assert response.attrs["InputPort"] == case_index
            assert response.attrs["InputMode"] == 1
            for port_index in (1, 2):
                for dataset in ("incident", "outgoing", "S"):
                    np.testing.assert_allclose(
                        reused[f"eigenmode_ports/port{port_index}/{dataset}"][...],
                        fresh[f"eigenmode_ports/port{port_index}/{dataset}"][...],
                        rtol=2e-12,
                        atol=2e-12,
                    )

    with h5py.File(tmp_path / "reused_study.h5") as output:
        assert output.attrs["StudyType"] == "eigenmode"
        assert output.attrs["Complete"]
        np.testing.assert_array_equal(output["channel_ports"], (1, 2))
        np.testing.assert_array_equal(output["channel_modes"], (1, 1))


def test_eigenmode_study_csv_factory_and_complete_channel_validation(tmp_path):
    cases = tmp_path / "eigenmode.csv"
    cases.write_text(
        "case_id,object_id,port,mode\n"
        "p1m1,eigenmode_excitation_1,1,1\n"
        "p2m1,eigenmode_excitation_1,2,1\n"
    )
    study = gprMax.Study.from_csv("eigenmode", cases)
    assert isinstance(study, gprMax.EigenmodeStudy)
    assert study.cases[1].states[0].parameters == {"port": 2, "mode": 1}

    scene, excitation = _two_port_waveguide_scene()
    incomplete = gprMax.EigenmodeStudy(
        [gprMax.StudyCase("only", [gprMax.ObjectState(excitation, port=1, mode=1)])]
    )
    with pytest.raises(ValueError, match="one case for every declared modal channel"):
        incomplete.bind_scene(scene)


@pytest.mark.integration
def test_hash_eigenmode_study_runs_and_writes_complete_smatrix(tmp_path):
    cases = tmp_path / "eigenmode.csv"
    cases.write_text(
        "case_id,object_id,port,mode\n"
        "p1m1,eigenmode_excitation_1,1,1\n"
        "p2m1,eigenmode_excitation_1,2,1\n"
    )
    inputfile = tmp_path / "eigenmode.in"
    inputfile.write_text(
        "#title: hash eigenmode study integration\n"
        "#domain_mode: TM\n"
        "#dx_dy_dz: 0.001 0.001 0.001\n"
        "#domain: 0.06 0.05 inf\n"
        "#pml_cells: 5 0 0 5 0 0\n"
        "#time_window: 4e-10\n"
        "#waveform: contsine 1 5e9 wave\n"
        "#box: 0 0 0 0.06 0.005 inf pec\n"
        "#box: 0 0.045 0 0.06 0.05 inf pec\n"
        "#eigenmode_band: band 5e9 5e9 1\n"
        "#eigenmode_port: 1 0.010 0.005 0 0.010 0.045 inf + 1 5e9 n\n"
        "#eigenmode_port: 2 0.050 0.005 0 0.050 0.045 inf - 1 5e9 n\n"
        "#eigenmode_excitation: 1 1 wave n\n"
        f"#study: eigenmode {cases.name}\n"
    )

    returned = gprMax.run(
        inputfile=inputfile,
        outputfile=tmp_path / "hash_eigenmode",
        hide_progress_bars=True,
        log_level=30,
    )

    assert isinstance(returned["study"], gprMax.EigenmodeStudyResult)
    with h5py.File(tmp_path / "hash_eigenmode_study.h5") as output:
        assert output.attrs["Complete"]
        assert output.attrs["CasesCompleted"] == 2
        np.testing.assert_array_equal(output["channel_ports"], (1, 2))
        np.testing.assert_array_equal(output["channel_modes"], (1, 1))
        assert output["S"].shape[1:] == (2, 2)


@pytest.mark.integration
def test_eigenmode_study_switches_cached_modes_without_new_fdfd_solves(tmp_path, monkeypatch):
    solves = 0
    original_solve = sources_module.FDFD_1D_mode_solver.solve

    def counted_solve(solver):
        nonlocal solves
        solves += 1
        return original_solve(solver)

    monkeypatch.setattr(sources_module.FDFD_1D_mode_solver, "solve", counted_solve)
    channels = ((1, 1), (1, 2), (2, 1), (2, 2))
    scene, excitation = _two_port_waveguide_scene(modes=(1, 2))
    study = _study_for_channels(excitation, channels)
    gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=tmp_path / "multimode",
        hide_progress_bars=True,
        log_level=30,
    )

    # One two-mode FDFD solution per port is cached during the only geometry
    # build; selecting all four source channels must not solve again.
    assert solves == 2
    assert study.result.s.shape[1:] == (4, 4)
    assert study.result.generalized_valid_s.any()
    np.testing.assert_array_equal(study.result.channel_ports, (1, 1, 2, 2))
    np.testing.assert_array_equal(study.result.channel_modes, (1, 2, 1, 2))


@pytest.mark.integration
def test_eigenmode_study_restart_preserves_completed_columns(tmp_path):
    output = tmp_path / "restart"
    scene, excitation = _two_port_waveguide_scene()
    study = _study(excitation)
    gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=output,
        hide_progress_bars=True,
        log_level=30,
    )
    with h5py.File(tmp_path / "restart_study.h5") as complete:
        original = complete["S"][:, :, 0].copy()

    restart_scene, restart_excitation = _two_port_waveguide_scene()
    restarted = _study(restart_excitation)
    gprMax.run(
        scenes=[restart_scene],
        study=restarted,
        i=2,
        outputfile=output,
        hide_progress_bars=True,
        log_level=30,
    )

    with h5py.File(tmp_path / "restart_study.h5") as complete:
        assert complete.attrs["Complete"]
        assert complete.attrs["CasesCompleted"] == 2
        np.testing.assert_array_equal(complete["S"][:, :, 0], original)


def test_modal_array_weights_and_embedded_response_synthesis():
    frequency = np.asarray((1e9, 2e9))
    excitations = (
        gprMax.ModalWeight(port=1, mode=1, power=4, phase_deg=90),
        gprMax.ModalWeight(port=2, mode=1, power=1, delay_s=0.25e-9),
    )
    weights = gprMax.modal_array_weights(frequency, (1, 2), (1, 1), excitations)
    np.testing.assert_allclose(weights[:, 0], 2j, atol=1e-14)
    np.testing.assert_allclose(
        weights[:, 1],
        np.exp(-2j * np.pi * frequency * 0.25e-9),
        atol=1e-14,
    )

    embedded = np.asarray(
        [
            [[1 + 0j, 2 + 0j], [3 + 0j, 4 + 0j]],
            [[5 + 0j, 6 + 0j], [7 + 0j, 8 + 0j]],
        ]
    )
    combined = gprMax.combine_embedded_modal_responses(embedded, weights)
    expected = embedded[..., 0] * weights[:, 0, np.newaxis]
    expected += embedded[..., 1] * weights[:, 1, np.newaxis]
    np.testing.assert_allclose(combined, expected)

    unused_nan = embedded.copy()
    unused_nan[..., 1] = np.nan
    first_only = weights.copy()
    first_only[:, 1] = 0
    np.testing.assert_allclose(
        gprMax.combine_embedded_modal_responses(unused_nan, first_only),
        embedded[..., 0] * weights[:, 0, np.newaxis],
    )
    with pytest.raises(ValueError, match="cannot be its frequency axis"):
        gprMax.combine_embedded_modal_responses(embedded, weights, channel_axis=0)

    s = np.full((2, 2, 2), np.nan + 1j * np.nan)
    s[:, :, 0] = np.asarray((0.5, 0.25))[np.newaxis, :]
    generalized_valid = np.zeros(s.shape, dtype=bool)
    generalized_valid[:, :, 0] = True
    study_result = gprMax.EigenmodeStudyResult(
        frequency=frequency,
        channel_ports=np.asarray((1, 2)),
        channel_modes=np.asarray((1, 1)),
        case_ids=("p1", "p2"),
        s=s,
        valid_s=generalized_valid.copy(),
        generalized_valid_s=generalized_valid,
        output_file=Path("unused.h5"),
    )
    np.testing.assert_allclose(
        study_result.outgoing((gprMax.ModalWeight(port=1, mode=1),)),
        np.asarray(((0.5, 0.25), (0.5, 0.25))),
    )


def _two_virtual_port_scene(active_port=1):
    dl = 1e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.06, 0.010, 0.012)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=0.4e-9))
    scene.add(gprMax.Waveform(wave_type="contsine", amp=1, freq=22e9, id="wave"))
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.06, dl, 0.012), material_id="pec"))
    scene.add(gprMax.Box(p1=(0, 0.009, 0), p2=(0.06, 0.010, 0.012), material_id="pec"))
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.06, 0.010, dl), material_id="pec"))
    scene.add(gprMax.Box(p1=(0, 0, 0.011), p2=(0.06, 0.010, 0.012), material_id="pec"))
    scene.add(gprMax.EigenmodeBand(id="band", fmin=22e9, fmax=22e9, points=1))
    for port, x, direction in ((1, 0.020, "+"), (2, 0.040, "-")):
        scene.add(
            gprMax.EigenmodePort(
                port=port,
                p1=(x, dl, dl),
                p2=(x, 0.009, 0.011),
                direction=direction,
                modes=(1,),
                anchors=(22e9,),
                plot_fields=False,
            )
        )
        scene.add(
            gprMax.VirtualWaveguide(
                port=port,
                length_cells=14,
                pml_cells=6,
                source_clearance_cells=3,
            )
        )
    excitation = gprMax.EigenmodeExcitation(
        port=active_port,
        mode=1,
        waveform="wave",
        plot_waveform=False,
    )
    scene.add(excitation)
    return scene, excitation


def _two_port_broadband_scene(active_port=1):
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.06, 0.05, INF)))
    scene.add(gprMax.PMLThickness(thickness=(5, 0, 0, 5, 0, 0)))
    scene.add(gprMax.TimeWindow(time=0.4e-9))
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.06, 0.005, INF), material_id="pec"))
    scene.add(gprMax.Box(p1=(0, 0.045, 0), p2=(0.06, 0.05, INF), material_id="pec"))
    scene.add(gprMax.EigenmodeBand(id="band", fmin=15e9, fmax=25e9, points=3))
    for port, x, direction in ((1, 0.010, "+"), (2, 0.050, "-")):
        scene.add(
            gprMax.EigenmodePort(
                port=port,
                p1=(x, 0.005, 0),
                p2=(x, 0.045, INF),
                direction=direction,
                modes=(1,),
                anchors="auto",
                plot_fields=False,
            )
        )
    excitation = gprMax.EigenmodeExcitation(
        port=active_port,
        mode=1,
        waveform="auto",
        plot_waveform=False,
    )
    scene.add(excitation)
    return scene, excitation


@pytest.mark.integration
def test_eigenmode_study_resets_and_switches_virtual_waveguides(tmp_path):
    scene, excitation = _two_virtual_port_scene()
    study = _study(excitation)
    gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=tmp_path / "virtual_reused",
        hide_progress_bars=True,
        log_level=30,
    )

    for active_port in (1, 2):
        fresh_scene, _ = _two_virtual_port_scene(active_port)
        gprMax.run(
            scenes=[fresh_scene],
            outputfile=tmp_path / f"virtual_fresh{active_port}",
            hide_progress_bars=True,
            log_level=30,
        )
        with h5py.File(tmp_path / f"virtual_reused{active_port}.h5") as reused, h5py.File(
            tmp_path / f"virtual_fresh{active_port}.h5"
        ) as fresh:
            for port_index in (1, 2):
                np.testing.assert_allclose(
                    reused[f"eigenmode_ports/port{port_index}/S"][...],
                    fresh[f"eigenmode_ports/port{port_index}/S"][...],
                    rtol=2e-10,
                    atol=2e-10,
                )


@pytest.mark.integration
def test_eigenmode_study_reuses_broadband_anchor_banks(tmp_path, monkeypatch):
    solve_models = []
    original_solve = sources_module.FDFD_1D_mode_solver.solve

    def counted_solve(solver):
        import gprMax.config as config

        solve_models.append(config.sim_config.current_model)
        return original_solve(solver)

    monkeypatch.setattr(sources_module.FDFD_1D_mode_solver, "solve", counted_solve)
    scene, excitation = _two_port_broadband_scene()
    study = _study(excitation)
    gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=tmp_path / "broadband_reused",
        hide_progress_bars=True,
        log_level=30,
    )
    reused_solve_count = len(solve_models)
    assert reused_solve_count > 2
    assert set(solve_models) == {0}

    fresh_scene, _ = _two_port_broadband_scene(active_port=2)
    gprMax.run(
        scenes=[fresh_scene],
        outputfile=tmp_path / "broadband_fresh2",
        hide_progress_bars=True,
        log_level=30,
    )
    assert len(solve_models) > reused_solve_count
    with h5py.File(tmp_path / "broadband_reused2.h5") as reused, h5py.File(
        tmp_path / "broadband_fresh2.h5"
    ) as fresh:
        for port_index in (1, 2):
            for dataset in ("incident", "outgoing", "S"):
                np.testing.assert_allclose(
                    reused[f"eigenmode_ports/port{port_index}/{dataset}"][...],
                    fresh[f"eigenmode_ports/port{port_index}/{dataset}"][...],
                    rtol=2e-10,
                    atol=2e-10,
                )


def _assert_device_eigenmode_study_matches_cpu(
    tmp_path, name, *, virtual=False, broadband=False, **device_options
):
    if virtual:
        scene_factory = _two_virtual_port_scene
    elif broadband:
        scene_factory = _two_port_broadband_scene
    else:
        scene_factory = _two_port_waveguide_scene
    cpu_scene, cpu_excitation = scene_factory()
    device_scene, device_excitation = scene_factory()
    cpu = _study(cpu_excitation)
    device = _study(device_excitation)
    gprMax.run(
        scenes=[cpu_scene],
        study=cpu,
        outputfile=tmp_path / "cpu",
        hide_progress_bars=True,
        log_level=30,
    )
    gprMax.run(
        scenes=[device_scene],
        study=device,
        outputfile=tmp_path / name,
        hide_progress_bars=True,
        log_level=30,
        **device_options,
    )
    np.testing.assert_array_equal(device.result.frequency, cpu.result.frequency)
    valid = device.result.generalized_valid_s & cpu.result.generalized_valid_s
    assert valid.any()
    np.testing.assert_allclose(
        device.result.s[valid],
        cpu.result.s[valid],
        rtol=5e-4,
        atol=5e-4,
    )


@pytest.mark.integration
@pytest.mark.gpu
def test_cuda_eigenmode_study_matches_cpu(tmp_path, gpu_device):
    _assert_device_eigenmode_study_matches_cpu(
        tmp_path,
        "cuda",
        broadband=True,
        gpu=[gpu_device],
    )


@pytest.mark.integration
@pytest.mark.gpu
def test_opencl_eigenmode_study_matches_cpu(tmp_path, opencl_device):
    _assert_device_eigenmode_study_matches_cpu(
        tmp_path,
        "opencl",
        broadband=True,
        opencl=[opencl_device],
    )


@pytest.mark.integration
@pytest.mark.gpu
def test_cuda_virtual_waveguide_eigenmode_study_matches_cpu(tmp_path, gpu_device):
    _assert_device_eigenmode_study_matches_cpu(
        tmp_path,
        "cuda_virtual",
        virtual=True,
        gpu=[gpu_device],
    )


@pytest.mark.integration
@pytest.mark.gpu
def test_opencl_virtual_waveguide_eigenmode_study_matches_cpu(tmp_path, opencl_device):
    _assert_device_eigenmode_study_matches_cpu(
        tmp_path,
        "opencl_virtual",
        virtual=True,
        opencl=[opencl_device],
    )
