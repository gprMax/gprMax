"""Reusable eigenmode-port studies."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax
import gprMax.sources as sources_module
from gprMax.studies import _deembed_modal_responses

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


def _add_centre_receiver(scene):
    scene.add(gprMax.Rx(p1=(0.03, 0.025, INF)))
    return scene


def _simultaneous_two_port_scene():
    scene, _ = _two_port_waveguide_scene(active_port=1)
    scene.add(
        gprMax.EigenmodeExcitation(
            port=2,
            mode=1,
            waveform="wave",
            amplitude=0.5,
            phase_deg=90,
            plot_waveform=False,
        )
    )
    return _add_centre_receiver(scene)


def _two_aperture_array_scene():
    dl = 2.5e-3
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.10, 0.065, 0.05)))
    scene.add(gprMax.PMLThickness(thickness=3))
    scene.add(gprMax.TimeWindow(time=0.8e-9))
    scene.add(gprMax.EigenmodeBand(id="band", fmin=8e9, fmax=12e9, points=3))
    for port, x0, x1 in ((1, 0.015, 0.040), (2, 0.060, 0.085)):
        # Four PEC walls form a short open-ended rectangular guide.
        # Use a rectangular, not square, aperture. A square guide has two
        # degenerate fundamental polarisations, so asking an iterative
        # eigensolver for only "mode 1" does not define a reproducible vector
        # within that two-dimensional eigenspace. That ambiguity can look like
        # a cross-polar error when independent embedded fields are compared
        # with a separately solved simultaneous excitation.
        scene.add(gprMax.Box(p1=(x0, 0.0200, 0.0125), p2=(x0 + dl, 0.0400, 0.025), material_id="pec"))
        scene.add(gprMax.Box(p1=(x1 - dl, 0.0200, 0.0125), p2=(x1, 0.0400, 0.025), material_id="pec"))
        scene.add(gprMax.Box(p1=(x0, 0.0200, 0.0125), p2=(x1, 0.0225, 0.025), material_id="pec"))
        scene.add(gprMax.Box(p1=(x0, 0.0375, 0.0125), p2=(x1, 0.0400, 0.025), material_id="pec"))
        scene.add(
            gprMax.EigenmodePort(
                port=port,
                p1=(x0 + dl, 0.0225, 0.0125),
                p2=(x1 - dl, 0.0375, 0.0125),
                direction="+",
                modes=(1,),
                # Keep the compact backend-parity model on one unambiguous
                # modal basis. Broadband anchor tracking has separate tests.
                anchors=(10e9,),
                plot_fields=False,
            )
        )
        scene.add(
            gprMax.VirtualWaveguide(
                port=port,
                length_cells=10,
                pml_cells=4,
                source_clearance_cells=2,
            )
        )
    excitation = gprMax.EigenmodeExcitation(
        port=1,
        mode=1,
        waveform="auto",
        plot_waveform=False,
    )
    scene.add(excitation)
    scene.add(gprMax.NTFFSurface(p1=(0.010, 0.0125, 0.010), p2=(0.0875, 0.050, 0.0375), id="box"))
    scene.add(gprMax.NTFFFrequencyTransform("box", "ff", (8e9, 10e9, 12e9)))
    scene.add(
        gprMax.NTFFFarField(
            theta=(0, 45, 90),
            phi=(0, 0, 0),
            transform_id="ff",
            id="cut",
            outputs=("Etheta", "Ephi"),
        )
    )
    return scene, excitation


def _two_aperture_array_study():
    scene, excitation = _two_aperture_array_scene()
    codebook = gprMax.ArrayCodebook(
        (
            gprMax.ArrayState(
                "broadside",
                (gprMax.ModalWeight(1, 1), gprMax.ModalWeight(2, 1)),
            ),
        ),
        embedded_far_fields=(gprMax.EmbeddedFarFieldSpec("ff", "cut"),),
    )
    study = gprMax.EigenmodeStudy(
        (
            gprMax.StudyCase("p1", (gprMax.ObjectState(excitation, port=1, mode=1),)),
            gprMax.StudyCase("p2", (gprMax.ObjectState(excitation, port=2, mode=1),)),
        ),
        codebook=codebook,
    )
    return scene, study, codebook


@pytest.mark.integration
def test_simultaneous_modal_drives_obey_linear_superposition(tmp_path):
    first_scene, _ = _two_port_waveguide_scene(active_port=1)
    _add_centre_receiver(first_scene)
    second_scene, second_excitation = _two_port_waveguide_scene(active_port=2)
    second_excitation.kwargs["amplitude"] = 0.5
    second_excitation.kwargs["phase_deg"] = 90
    _add_centre_receiver(second_scene)
    combined_scene = _simultaneous_two_port_scene()

    paths = {}
    for name, scene in (
        ("first", first_scene),
        ("second", second_scene),
        ("combined", combined_scene),
    ):
        path = tmp_path / name
        gprMax.run(
            scenes=[scene],
            outputfile=path,
            hide_progress_bars=True,
            log_level=30,
        )
        paths[name] = path.with_suffix(".h5")

    with h5py.File(paths["first"]) as first, h5py.File(paths["second"]) as second, h5py.File(
        paths["combined"]
    ) as combined:
        expected = first["rxs/rx1/Ez"][...] + second["rxs/rx1/Ez"][...]
        np.testing.assert_allclose(
            combined["rxs/rx1/Ez"][...],
            expected,
            rtol=1e-5,
            atol=1e-6 * np.max(np.abs(expected)),
        )
        for port_index in (1, 2):
            port = combined[f"eigenmode_ports/port{port_index}"]
            assert port.attrs["ResponseType"] == "driven"
            assert "S" not in port
        assert tuple(combined["eigenmode_ports/port1"].attrs["ExcitationModes"]) == (1,)
        assert tuple(combined["eigenmode_ports/port2"].attrs["ExcitationModes"]) == (1,)


@pytest.mark.integration
@pytest.mark.gpu
def test_cuda_simultaneous_modal_drives_match_cpu(tmp_path, gpu_device):
    cpu_output = tmp_path / "simultaneous_cpu"
    cuda_output = tmp_path / "simultaneous_cuda"
    gprMax.run(
        scenes=[_simultaneous_two_port_scene()],
        outputfile=cpu_output,
        hide_progress_bars=True,
        log_level=30,
    )
    gprMax.run(
        scenes=[_simultaneous_two_port_scene()],
        outputfile=cuda_output,
        gpu=[gpu_device],
        hide_progress_bars=True,
        log_level=30,
    )

    with h5py.File(cpu_output.with_suffix(".h5")) as cpu, h5py.File(cuda_output.with_suffix(".h5")) as cuda:
        np.testing.assert_allclose(
            cuda["rxs/rx1/Ez"][...],
            cpu["rxs/rx1/Ez"][...],
            rtol=2e-5,
            atol=2e-6 * np.max(np.abs(cpu["rxs/rx1/Ez"][...])),
        )
        for port_index in (1, 2):
            for dataset in ("incident", "outgoing"):
                np.testing.assert_allclose(
                    cuda[f"eigenmode_ports/port{port_index}/{dataset}"][...],
                    cpu[f"eigenmode_ports/port{port_index}/{dataset}"][...],
                    rtol=3e-5,
                    atol=1e-7,
                )


@pytest.mark.integration
@pytest.mark.gpu
def test_opencl_simultaneous_modal_drives_match_cpu(tmp_path, opencl_device):
    cpu_output = tmp_path / "simultaneous_cpu"
    opencl_output = tmp_path / "simultaneous_opencl"
    gprMax.run(
        scenes=[_simultaneous_two_port_scene()],
        outputfile=cpu_output,
        hide_progress_bars=True,
        log_level=30,
    )
    gprMax.run(
        scenes=[_simultaneous_two_port_scene()],
        outputfile=opencl_output,
        opencl=[opencl_device],
        hide_progress_bars=True,
        log_level=30,
    )

    with h5py.File(cpu_output.with_suffix(".h5")) as cpu, h5py.File(opencl_output.with_suffix(".h5")) as opencl:
        np.testing.assert_allclose(
            opencl["rxs/rx1/Ez"][...],
            cpu["rxs/rx1/Ez"][...],
            rtol=2e-5,
            atol=2e-6 * np.max(np.abs(cpu["rxs/rx1/Ez"][...])),
        )
        for port_index in (1, 2):
            for dataset in ("incident", "outgoing"):
                np.testing.assert_allclose(
                    opencl[f"eigenmode_ports/port{port_index}/{dataset}"][...],
                    cpu[f"eigenmode_ports/port{port_index}/{dataset}"][...],
                    rtol=3e-5,
                    atol=1e-7,
                )


@pytest.mark.integration
def test_two_modes_on_one_port_reuse_one_monitor(tmp_path):
    scene, _ = _two_port_waveguide_scene(active_port=1, modes=(1, 2))
    scene.add(
        gprMax.EigenmodeExcitation(
            port=1,
            mode=2,
            waveform="wave",
            amplitude=0.25,
            phase_deg=-45,
            plot_waveform=False,
        )
    )
    output = tmp_path / "two_modes_one_port"

    gprMax.run(
        scenes=[scene],
        outputfile=output,
        hide_progress_bars=True,
        log_level=30,
    )

    with h5py.File(output.with_suffix(".h5")) as handle:
        assert set(handle["eigenmode_ports"]) == {"port1", "port2"}
        source = handle["eigenmode_ports/port1"]
        assert tuple(source.attrs["ExcitationModes"]) == (1, 2)
        np.testing.assert_allclose(source.attrs["DriveAmplitudes"], (1, 0.25))
        np.testing.assert_allclose(source.attrs["DrivePhasesDegrees"], (0, -45))
        assert source.attrs["ResponseType"] == "driven"
        assert "S" not in source


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
    incident_columns = []
    outgoing_columns = []
    for case_index in (1, 2):
        with h5py.File(tmp_path / f"reused{case_index}.h5") as reused, h5py.File(
            tmp_path / f"fresh{case_index}.h5"
        ) as fresh:
            response = reused["study/eigenmode_response"]
            assert response.attrs["InputPort"] == case_index
            assert response.attrs["InputMode"] == 1
            np.testing.assert_array_equal(
                response["incident"],
                np.stack(
                    [reused[f"eigenmode_ports/port{port}/incident"][0] for port in (1, 2)],
                    axis=-1,
                ),
            )
            incident_columns.append(response["incident"][...])
            outgoing_columns.append(response["outgoing"][...])
            for port_index in (1, 2):
                for dataset in ("incident", "outgoing", "S"):
                    np.testing.assert_allclose(
                        reused[f"eigenmode_ports/port{port_index}/{dataset}"][...],
                        fresh[f"eigenmode_ports/port{port_index}/{dataset}"][...],
                        rtol=2e-12,
                        atol=2e-12,
                    )

    incident_matrix = np.stack(incident_columns, axis=-1)
    outgoing_matrix = np.stack(outgoing_columns, axis=-1)
    expected_s = np.asarray(
        [np.linalg.solve(incident.T, outgoing.T).T for incident, outgoing in zip(incident_matrix, outgoing_matrix)]
    )
    np.testing.assert_allclose(study.result.s, expected_s, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(study.result.incident_matrix, incident_matrix)
    np.testing.assert_array_equal(study.result.outgoing_matrix, outgoing_matrix)
    assert np.all(study.result.deembedding_valid)

    with h5py.File(tmp_path / "reused_study.h5") as output:
        assert output.attrs["StudyType"] == "eigenmode"
        assert output.attrs["Complete"]
        np.testing.assert_array_equal(output["channel_ports"], (1, 2))
        np.testing.assert_array_equal(output["channel_modes"], (1, 1))
        np.testing.assert_array_equal(output["incident_matrix"], incident_matrix)
        np.testing.assert_array_equal(output["outgoing_matrix"], outgoing_matrix)
        assert output.attrs["Deembedding"] == "conditioned_full_incident_matrix_solve"


def test_eigenmode_study_csv_factory_and_complete_channel_validation(tmp_path):
    cases = tmp_path / "eigenmode.csv"
    cases.write_text(
        "case_id,object_id,port,mode\n" "p1m1,eigenmode_excitation_1,1,1\n" "p2m1,eigenmode_excitation_1,2,1\n"
    )
    study = gprMax.Study.from_csv("eigenmode", cases)
    assert isinstance(study, gprMax.EigenmodeStudy)
    assert study.cases[1].states[0].parameters == {"port": 2, "mode": 1}

    scene, excitation = _two_port_waveguide_scene()
    incomplete = gprMax.EigenmodeStudy([gprMax.StudyCase("only", [gprMax.ObjectState(excitation, port=1, mode=1)])])
    with pytest.raises(ValueError, match="one case for every declared modal channel"):
        incomplete.bind_scene(scene)


@pytest.mark.integration
def test_hash_eigenmode_study_runs_and_writes_complete_smatrix(tmp_path):
    cases = tmp_path / "eigenmode.csv"
    cases.write_text(
        "case_id,object_id,port,mode\n" "p1m1,eigenmode_excitation_1,1,1\n" "p2m1,eigenmode_excitation_1,2,1\n"
    )
    codebook = tmp_path / "array.json"
    codebook.write_text(
        '{"schema":"gprMax-array-codebook-v1","states":['
        '{"id":"p1_only","drives":[{"port":1,"mode":1,"power_w":1.0}]}]}'
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
        f"#array_codebook: {codebook.name}\n"
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
        assert output["array_codebook"].attrs["Schema"] == "gprMax-array-codebook-v1"
        assert "array_states/p1_only/tarc" in output


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


def test_full_incident_matrix_deembedding_recovers_analytical_network_and_fields():
    """A conditioned solve removes passive-port incident-wave contamination."""

    frequency = np.asarray((1.0, 2.0, 3.0))
    true_s = np.asarray(
        [
            [[0.10 + 0.02j, 0.70 - 0.04j], [0.70 - 0.04j, -0.15 + 0.01j]],
            [[0.12 + 0.01j, 0.67 - 0.06j], [0.67 - 0.06j, -0.12 + 0.03j]],
            [[0.14 - 0.01j, 0.63 - 0.08j], [0.63 - 0.08j, -0.09 + 0.04j]],
        ],
        dtype=np.complex128,
    )
    incident = np.asarray(
        [
            [[1.0, 0.025 + 0.010j], [-0.018 + 0.006j, 0.96]],
            [[0.98, 0.030 + 0.008j], [-0.015 + 0.009j, 1.02]],
            [[1.03, 0.022 + 0.012j], [-0.020 + 0.004j, 0.99]],
        ],
        dtype=np.complex128,
    )
    outgoing = np.einsum("foi,fij->foj", true_s, incident)
    true_fields = np.asarray(
        [
            [[1.0 + 0.1j, 0.3 - 0.2j], [0.2, -0.6j]],
            [[0.9 + 0.2j, 0.4 - 0.1j], [0.3, -0.5j]],
            [[0.8 + 0.3j, 0.5], [0.4, -0.4j]],
        ],
        dtype=np.complex128,
    )
    raw_fields = np.einsum("fdi,fij->fdj", true_fields, incident)
    valid = np.ones(incident.shape, dtype=bool)

    recovered_s, s_valid, condition, basis_valid = _deembed_modal_responses(
        incident,
        outgoing,
        incident_valid=valid,
        response_valid=valid,
    )
    recovered_fields, field_valid, _, _ = _deembed_modal_responses(
        incident,
        raw_fields,
        incident_valid=valid,
        response_valid=np.ones(raw_fields.shape, dtype=bool),
    )

    np.testing.assert_allclose(recovered_s, true_s, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(recovered_fields, true_fields, rtol=2e-15, atol=2e-15)
    assert np.all(s_valid)
    assert np.all(field_valid)
    assert np.all(basis_valid)
    assert np.max(condition) < 1.1

    # The previous column-wise division leaves an error proportional to the
    # off-diagonal incident waves and therefore does not recover the network.
    diagonal_approximation = outgoing / np.diagonal(incident, axis1=1, axis2=2)[:, None, :]
    assert np.max(np.abs(diagonal_approximation - true_s)) > 1e-2


def test_full_incident_matrix_deembedding_rejects_singular_basis():
    incident = np.asarray([[[1.0, 1.0], [0.5, 0.5]]], dtype=np.complex128)
    response = np.ones_like(incident)
    recovered, valid, condition, basis_valid = _deembed_modal_responses(
        incident,
        response,
    )

    assert not basis_valid[0]
    assert not np.any(valid)
    assert condition[0] > 1e10
    assert np.all(np.isnan(recovered))


def test_two_port_active_network_metrics_match_closed_form(tmp_path):
    """Check active waves, power balance, and TARC for an exact two-port network."""

    frequency = np.asarray((1e9,))
    reflection = 0.2 + 0.1j
    transmission = 0.3 - 0.05j
    scattering = np.asarray(
        [[[reflection, transmission], [transmission, reflection]]],
        dtype=np.complex128,
    )
    valid = np.ones(scattering.shape, dtype=bool)
    study = gprMax.EigenmodeStudyResult(
        frequency=frequency,
        channel_ports=np.asarray((1, 2)),
        channel_modes=np.asarray((1, 1)),
        case_ids=("p1", "p2"),
        s=scattering,
        valid_s=valid,
        generalized_valid_s=valid,
        output_file=tmp_path / "unused.h5",
    )
    state = gprMax.ArrayState(
        "weighted",
        (
            gprMax.ModalWeight(1, 1, power=1.0),
            gprMax.ModalWeight(2, 1, power=0.5, phase_deg=60.0),
        ),
    )
    result = study.evaluate_array_state(state)

    incident = np.asarray((1.0, np.sqrt(0.5) * np.exp(1j * np.pi / 3)))
    outgoing = scattering[0] @ incident
    incident_power = np.sum(np.abs(incident) ** 2)
    reflected_power = np.sum(np.abs(outgoing) ** 2)
    np.testing.assert_allclose(result.incident[0], incident, atol=2e-16)
    np.testing.assert_allclose(result.outgoing[0], outgoing, atol=2e-16)
    np.testing.assert_allclose(result.active_reflection[0], outgoing / incident, atol=2e-16)
    np.testing.assert_allclose(result.incident_power, incident_power, atol=2e-16)
    np.testing.assert_allclose(result.reflected_power, reflected_power, atol=2e-16)
    np.testing.assert_allclose(
        result.accepted_power,
        incident_power - reflected_power,
        atol=2e-16,
    )
    np.testing.assert_allclose(result.tarc, np.sqrt(reflected_power / incident_power), atol=2e-16)


def test_array_codebook_json_and_coherent_state_metrics(tmp_path):
    codebook_path = tmp_path / "array.json"
    codebook_path.write_text(
        """{
          "schema": "gprMax-array-codebook-v1",
          "description": "two-element test",
          "embedded_far_fields": [
            {"transform_id": "ff", "output_id": "cut"}
          ],
          "states": [
            {
              "id": "quadrature",
              "drives": [
                {"port": 1, "mode": 1, "power_w": 1.0},
                {"port": 2, "mode": 1, "power_w": 1.0, "phase_deg": 90.0}
              ]
            }
          ]
        }"""
    )
    codebook = gprMax.ArrayCodebook.from_json(codebook_path)
    assert codebook.description == "two-element test"
    assert codebook.embedded_far_fields[0].key == "ff/cut"
    assert codebook.states[0].drives[1].phase_deg == 90
    canonical_path = tmp_path / "canonical.json"
    canonical_path.write_text(codebook.to_json())
    canonical = gprMax.ArrayCodebook.from_json(canonical_path)
    assert canonical.to_definition() == codebook.to_definition()

    frequency = np.asarray((1e9,))
    s = np.zeros((1, 2, 2), dtype=np.complex128)
    valid_s = np.ones(s.shape, dtype=bool)
    # Two orthogonal requested samples and two-point equal-weight sphere.
    bank = gprMax.EmbeddedFarFieldBank(
        transform_id="ff",
        output_id="cut",
        frequency=frequency,
        theta=np.asarray((0.0, 90.0)),
        phi=np.asarray((0.0, 0.0)),
        etheta=np.asarray([[[1, 0], [0, 1]]], dtype=np.complex128),
        ephi=np.zeros((1, 2, 2), dtype=np.complex128),
        sphere_theta=np.asarray((0.0, 90.0)),
        sphere_phi=np.asarray((0.0, 0.0)),
        sphere_weights=np.asarray((2 * np.pi, 2 * np.pi)),
        sphere_etheta=np.asarray([[[1, 0], [0, 1]]], dtype=np.complex128),
        sphere_ephi=np.zeros((1, 2, 2), dtype=np.complex128),
        valid=np.ones((1, 2), dtype=bool),
        impedance=1.0,
        theta_order=2,
        phi_order=1,
        enclosure_radius=1.0,
    )
    study_result = gprMax.EigenmodeStudyResult(
        frequency=frequency,
        channel_ports=np.asarray((1, 2)),
        channel_modes=np.asarray((1, 1)),
        case_ids=("p1", "p2"),
        s=s,
        valid_s=valid_s,
        generalized_valid_s=valid_s,
        output_file=tmp_path / "unused.h5",
        embedded_far_fields={bank.key: bank},
    )
    result = study_result.evaluate_array_state(codebook.states[0])
    np.testing.assert_allclose(result.incident, ((1, 1j),), atol=1e-15)
    np.testing.assert_allclose(result.incident_power, (2,))
    np.testing.assert_allclose(result.accepted_power, (2,))
    np.testing.assert_allclose(result.tarc, (0,))
    field = result.far_fields["ff/cut"]
    np.testing.assert_allclose(field.etheta, ((1, 1j),), atol=1e-15)
    np.testing.assert_allclose(field.radiated_power, (2 * np.pi,))
    np.testing.assert_allclose(field.directivity, ((1, 1),))


def test_two_element_hertzian_array_matches_analytical_array_factor(tmp_path):
    """Validate coherent synthesis and sphere integration against a closed form.

    The retained fields represent two identical z-directed Hertzian dipoles,
    separated by half a wavelength along x and driven in phase. Their array
    factor is ``2 cos(pi / 2 sin(theta) cos(phi))``. Integrating its power
    pattern gives the exact broadside directivity ``6 / (2 - 3 / pi**2)``.
    """

    frequency = np.asarray((1e9,))
    eta = 1.0
    theta_order = 64
    phi_order = 128
    mu, mu_weights = np.polynomial.legendre.leggauss(theta_order)
    theta_axis = np.arccos(mu[::-1])
    phi_axis = 2 * np.pi * np.arange(phi_order) / phi_order
    theta_grid, phi_grid = np.meshgrid(theta_axis, phi_axis, indexing="ij")
    sphere_theta_rad = theta_grid.ravel()
    sphere_phi_rad = phi_grid.ravel()
    sphere_weights = np.broadcast_to(
        mu_weights[::-1, np.newaxis] * (2 * np.pi / phi_order),
        theta_grid.shape,
    ).ravel()

    # q = k d = pi for half-wavelength separation. Each retained column is
    # the field per sqrt(W) of its own incident modal power wave.
    q = np.pi
    spatial_phase = 0.5 * q * np.sin(sphere_theta_rad) * np.cos(sphere_phi_rad)
    sphere_columns = np.stack(
        (
            np.sin(sphere_theta_rad) * np.exp(1j * spatial_phase),
            np.sin(sphere_theta_rad) * np.exp(-1j * spatial_phase),
        ),
        axis=-1,
    )[np.newaxis, ...]

    requested_theta_rad = np.deg2rad(np.asarray((90.0, 90.0)))
    requested_phi_rad = np.deg2rad(np.asarray((90.0, 0.0)))
    requested_phase = 0.5 * q * np.sin(requested_theta_rad) * np.cos(requested_phi_rad)
    requested_columns = np.stack(
        (
            np.sin(requested_theta_rad) * np.exp(1j * requested_phase),
            np.sin(requested_theta_rad) * np.exp(-1j * requested_phase),
        ),
        axis=-1,
    )[np.newaxis, ...]

    bank = gprMax.EmbeddedFarFieldBank(
        transform_id="analytical",
        output_id="principal",
        frequency=frequency,
        theta=np.rad2deg(requested_theta_rad),
        phi=np.rad2deg(requested_phi_rad),
        etheta=requested_columns,
        ephi=np.zeros_like(requested_columns),
        sphere_theta=np.rad2deg(sphere_theta_rad),
        sphere_phi=np.rad2deg(sphere_phi_rad),
        sphere_weights=sphere_weights,
        sphere_etheta=sphere_columns,
        sphere_ephi=np.zeros_like(sphere_columns),
        valid=np.ones((1, 2), dtype=bool),
        impedance=eta,
        theta_order=theta_order,
        phi_order=phi_order,
        enclosure_radius=1.0,
    )
    valid = np.ones((1, 2, 2), dtype=bool)
    study = gprMax.EigenmodeStudyResult(
        frequency=frequency,
        channel_ports=np.asarray((1, 2)),
        channel_modes=np.asarray((1, 1)),
        case_ids=("element1", "element2"),
        s=np.zeros((1, 2, 2), dtype=np.complex128),
        valid_s=valid,
        generalized_valid_s=valid,
        output_file=tmp_path / "unused.h5",
        embedded_far_fields={bank.key: bank},
    )
    state = study.evaluate_array_state(
        gprMax.ArrayState(
            "broadside",
            (gprMax.ModalWeight(1, 1), gprMax.ModalWeight(2, 1)),
        )
    )
    field = state.far_fields[bank.key]

    expected_field = 2 * np.sin(requested_theta_rad) * np.cos(requested_phase)
    # Integral of |Etheta|^2 over solid angle for q=pi:
    # (8 pi / 3) * (2 + 2 j0(pi) - j2(pi)), where j0(pi)=0 and
    # j2(pi)=3/pi^2.
    expected_field_integral = (8 * np.pi / 3) * (2 - 3 / np.pi**2)
    expected_radiated_power = expected_field_integral / (2 * eta)
    expected_broadside_directivity = 6 / (2 - 3 / np.pi**2)

    np.testing.assert_allclose(field.etheta[0], expected_field, atol=2e-15)
    np.testing.assert_allclose(field.radiated_power, expected_radiated_power, rtol=2e-13)
    np.testing.assert_allclose(
        field.directivity[0, 0],
        expected_broadside_directivity,
        rtol=2e-13,
    )
    assert abs(field.etheta[0, 1]) < 2e-15


def test_array_codebook_rejects_unknown_schema_and_duplicate_channels(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"schema": "wrong", "states": []}')
    with pytest.raises(ValueError, match="schema must be"):
        gprMax.ArrayCodebook.from_json(invalid)
    with pytest.raises(ValueError, match="duplicate port/mode"):
        gprMax.ArrayState(
            "bad",
            (
                gprMax.ModalWeight(1, 1),
                gprMax.ModalWeight(1, 1, phase_deg=90),
            ),
        )

    scene, excitation = _two_port_waveguide_scene()
    unavailable = gprMax.ArrayCodebook((gprMax.ArrayState("bad", (gprMax.ModalWeight(3, 1),)),))
    study = gprMax.EigenmodeStudy(_study(excitation).cases, codebook=unavailable)
    with pytest.raises(ValueError, match="unavailable port 3, mode 1"):
        study.bind_scene(scene)


def test_array_state_excludes_evanescent_output_from_power_accounting(tmp_path):
    frequency = np.asarray((1e9,))
    scattering = np.asarray([[[0.5, 0.0], [2.0, 3.0]]], dtype=np.complex128)
    physical = np.zeros(scattering.shape, dtype=bool)
    physical[:, 0, 0] = True
    generalized = np.ones(scattering.shape, dtype=bool)
    result = gprMax.EigenmodeStudyResult(
        frequency=frequency,
        channel_ports=np.asarray((1, 1)),
        channel_modes=np.asarray((1, 2)),
        case_ids=("mode1", "mode2"),
        s=scattering,
        valid_s=physical,
        generalized_valid_s=generalized,
        output_file=tmp_path / "unused.h5",
    )

    propagating = result.evaluate_array_state(gprMax.ArrayState("mode1", (gprMax.ModalWeight(1, 1),)))
    assert propagating.valid[0]
    np.testing.assert_allclose(propagating.outgoing, ((0.5, 2.0),))
    np.testing.assert_allclose(propagating.reflected_power, (0.25,))
    np.testing.assert_allclose(propagating.tarc, (0.5,))

    evanescent = result.evaluate_array_state(gprMax.ArrayState("mode2", (gprMax.ModalWeight(1, 2),)))
    assert not evanescent.valid[0]
    assert np.isnan(evanescent.reflected_power[0])


@pytest.mark.integration
def test_embedded_far_fields_reconstruct_each_independent_case(tmp_path):
    scene, study, codebook = _two_aperture_array_study()
    np.random.seed(1729)
    gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=tmp_path / "embedded",
        hide_progress_bars=True,
        log_level=30,
    )

    bank = study.result.embedded_far_fields["ff/cut"]
    for case_index in (1, 2):
        with h5py.File(tmp_path / f"embedded{case_index}.h5") as case:
            incident = np.stack(
                [case[f"eigenmode_ports/port{port}/incident"][0] for port in (1, 2)],
                axis=-1,
            )
            raw_etheta = case["ntff/box/frequency/ff/far_field/cut/fields/Etheta"][...]
            raw_ephi = case["ntff/box/frequency/ff/far_field/cut/fields/Ephi"][...]
        np.testing.assert_allclose(
            np.einsum("fdc,fc->fd", bank.etheta, incident),
            raw_etheta,
            rtol=2e-6,
            atol=2e-7 * np.max(np.abs(raw_etheta)),
        )
        np.testing.assert_allclose(
            np.einsum("fdc,fc->fd", bank.ephi, incident),
            raw_ephi,
            rtol=2e-6,
            atol=2e-7 * np.max(np.abs(raw_ephi)),
        )
    with h5py.File(tmp_path / "embedded_study.h5") as output:
        assert output["embedded_far_fields/ff/cut/Etheta"].shape[-1] == 2
        assert output["embedded_far_fields/ff/cut/raw_runs/Etheta"].shape[-1] == 2
        retained_raw_case = output[
            "embedded_far_fields/ff/cut/raw_runs/Etheta"
        ][..., 0].copy()
        retained_incident_case = output["incident_matrix"][..., 0].copy()
        definition = json.loads(output["array_codebook/definition"][()].decode())
        assert definition == codebook.to_definition()
        assert "array_states/broadside/far_fields/ff/cut/directivity" in output
        assert "array_states/broadside/far_fields/ff/cut/Etheta" in output
    reloaded = gprMax.EigenmodeStudyResult.from_hdf5(tmp_path / "embedded_study.h5")
    np.testing.assert_array_equal(
        reloaded.embedded_far_fields["ff/cut"].etheta,
        bank.etheta,
    )
    np.testing.assert_array_equal(reloaded.incident_matrix, study.result.incident_matrix)
    np.testing.assert_array_equal(reloaded.outgoing_matrix, study.result.outgoing_matrix)
    np.testing.assert_array_equal(
        reloaded.deembedding_condition_number,
        study.result.deembedding_condition_number,
    )
    assert reloaded.evaluate_codebook(codebook)[0].far_fields["ff/cut"].valid.any()

    # Restart at the second case. The first raw run field and incident/outgoing
    # column must be loaded before the aggregate is de-embedded again.
    restart_scene, restart_study, _ = _two_aperture_array_study()
    np.random.seed(1729)
    gprMax.run(
        scenes=[restart_scene],
        study=restart_study,
        i=2,
        outputfile=tmp_path / "embedded",
        hide_progress_bars=True,
        log_level=30,
    )
    np.testing.assert_allclose(
        restart_study.result.embedded_far_fields["ff/cut"].etheta,
        bank.etheta,
        rtol=2e-4,
        atol=2e-9,
    )
    with h5py.File(tmp_path / "embedded_study.h5") as output:
        np.testing.assert_array_equal(
            output["embedded_far_fields/ff/cut/raw_runs/Etheta"][..., 0],
            retained_raw_case,
        )
        np.testing.assert_array_equal(
            output["incident_matrix"][..., 0],
            retained_incident_case,
        )


def _assert_device_embedded_far_field_array_state_matches_cpu(tmp_path, backend, device):
    cpu_scene, cpu_study, cpu_codebook = _two_aperture_array_study()
    device_scene, device_study, device_codebook = _two_aperture_array_study()
    np.random.seed(1729)
    gprMax.run(
        scenes=[cpu_scene],
        study=cpu_study,
        outputfile=tmp_path / "embedded_cpu",
        cpu_precision="double",
        hide_progress_bars=True,
        log_level=30,
    )
    np.random.seed(1729)
    device_options = {"gpu": [device]} if backend == "cuda" else {"opencl": [device]}
    gprMax.run(
        scenes=[device_scene],
        study=device_study,
        outputfile=tmp_path / f"embedded_{backend}",
        gpu_precision="double",
        hide_progress_bars=True,
        log_level=30,
        **device_options,
    )

    def assert_scaled_close(actual, expected, tolerance):
        scale = np.nanmax(np.abs(expected))
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=tolerance,
            atol=tolerance * scale,
            equal_nan=True,
        )

    # Compare against the response scale so near-null cross-couplings do not
    # impose a meaningless large relative error.
    assert_scaled_close(device_study.result.s, cpu_study.result.s, 5e-3)
    for label, study in (("cpu", cpu_study), (backend, device_study)):
        bank = study.result.embedded_far_fields["ff/cut"]
        for case_index in (1, 2):
            with h5py.File(tmp_path / f"embedded_{label}{case_index}.h5") as case:
                incident = np.stack(
                    [case[f"eigenmode_ports/port{port}/incident"][0] for port in (1, 2)],
                    axis=-1,
                )
                for result_name, dataset_name in (("etheta", "Etheta"), ("ephi", "Ephi")):
                    raw = case[f"ntff/box/frequency/ff/far_field/cut/fields/{dataset_name}"][...]
                    np.testing.assert_allclose(
                        np.einsum(
                            "fdc,fc->fd",
                            getattr(bank, result_name),
                            incident,
                        ),
                        raw,
                        rtol=3e-6,
                        atol=3e-7 * np.max(np.abs(raw)),
                    )
    cpu_state = cpu_study.result.evaluate_codebook(cpu_codebook)[0]
    device_state = device_study.result.evaluate_codebook(device_codebook)[0]
    for name in ("tarc", "accepted_power"):
        assert_scaled_close(
            getattr(device_state, name),
            getattr(cpu_state, name),
            1e-2,
        )
    assert_scaled_close(
        device_state.far_fields["ff/cut"].radiated_power,
        cpu_state.far_fields["ff/cut"].radiated_power,
        1e-2,
    )
    assert_scaled_close(
        device_state.far_fields["ff/cut"].directivity,
        cpu_state.far_fields["ff/cut"].directivity,
        3e-2,
    )


@pytest.mark.integration
@pytest.mark.gpu
def test_cuda_embedded_far_field_array_state_matches_cpu(tmp_path, gpu_device):
    _assert_device_embedded_far_field_array_state_matches_cpu(tmp_path, "cuda", gpu_device)


@pytest.mark.integration
@pytest.mark.gpu
def test_opencl_embedded_far_field_array_state_matches_cpu(tmp_path, opencl_device):
    _assert_device_embedded_far_field_array_state_matches_cpu(tmp_path, "opencl", opencl_device)


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


def _two_virtual_port_broadband_scene(active_port=1, *, simultaneous=False):
    """Matched, lossless TE10 guide used for analytical multi-drive validation."""

    scene, excitation = _two_virtual_port_scene(active_port)
    time_window = next(item for item in scene.single_use_objects if isinstance(item, gprMax.TimeWindow))
    time_window.time = 1.2e-9
    band = next(item for item in scene.grid_objects if isinstance(item, gprMax.EigenmodeBand))
    band.kwargs.update(fmin=20e9, fmax=24e9, points=17)
    for port in (item for item in scene.grid_objects if isinstance(item, gprMax.EigenmodePort)):
        port.kwargs["anchors"] = (16.9e9, 22e9, 27e9)
    excitation.kwargs["waveform"] = "auto"
    if simultaneous:
        scene.add(
            gprMax.EigenmodeExcitation(
                port=2,
                mode=1,
                waveform="auto",
                amplitude=0.5,
                phase_deg=90,
                delay_s=7.5e-12,
                plot_waveform=False,
            )
        )
    return scene, excitation


@pytest.mark.integration
def test_simultaneous_drives_match_lossless_rectangular_waveguide_solution(tmp_path):
    """Validate driven modal waves against b=S*a and analytical TE10 propagation.

    Two independent excitations measure the complete incident and outgoing
    wave matrices.  Their right quotient gives the scattering matrix without
    treating the small residual reflection from either auxiliary termination
    as a new network excitation.  A third solve drives both ports with a
    complex broadband weight and must obey the resulting b=S*a relation.

    The physical guide is uniform, reciprocal, and lossless.  Its dominant
    TE10 propagation constant is beta=sqrt(k0**2-(pi/a)**2), so transmission
    over the distance L between reference planes is exp(-j*beta*L).
    """

    single_paths = []
    for active_port in (1, 2):
        scene, _ = _two_virtual_port_broadband_scene(active_port)
        output = tmp_path / f"rectangular_guide_port{active_port}"
        gprMax.run(
            scenes=[scene],
            outputfile=output,
            cpu_precision="double",
            hide_progress_bars=True,
            log_level=30,
        )
        single_paths.append(output.with_suffix(".h5"))

    simultaneous_scene, _ = _two_virtual_port_broadband_scene(1, simultaneous=True)
    simultaneous_path = tmp_path / "rectangular_guide_simultaneous"
    gprMax.run(
        scenes=[simultaneous_scene],
        outputfile=simultaneous_path,
        cpu_precision="double",
        hide_progress_bars=True,
        log_level=30,
    )

    incident_columns = []
    outgoing_columns = []
    frequency = None
    for path in single_paths:
        with h5py.File(path) as output:
            if frequency is None:
                frequency = output["eigenmode_ports/port1/frequency"][...]
            incident_columns.append(
                np.stack(
                    [output[f"eigenmode_ports/port{port}/incident"][0] for port in (1, 2)],
                    axis=-1,
                )
            )
            outgoing_columns.append(
                np.stack(
                    [output[f"eigenmode_ports/port{port}/outgoing"][0] for port in (1, 2)],
                    axis=-1,
                )
            )

    # A and B have shape (frequency, output channel, excitation case).
    incident_matrix = np.stack(incident_columns, axis=-1)
    outgoing_matrix = np.stack(outgoing_columns, axis=-1)
    scattering = np.asarray(
        [np.linalg.solve(incident_matrix[index].T, outgoing_matrix[index].T).T for index in range(frequency.size)]
    )

    with h5py.File(simultaneous_path.with_suffix(".h5")) as output:
        simultaneous_incident = np.stack(
            [output[f"eigenmode_ports/port{port}/incident"][0] for port in (1, 2)],
            axis=-1,
        )
        simultaneous_outgoing = np.stack(
            [output[f"eigenmode_ports/port{port}/outgoing"][0] for port in (1, 2)],
            axis=-1,
        )

    predicted_outgoing = np.einsum("fij,fj->fi", scattering, simultaneous_incident)
    driven_residual = np.linalg.norm(predicted_outgoing - simultaneous_outgoing, axis=1)
    driven_residual /= np.linalg.norm(simultaneous_outgoing, axis=1)
    assert np.max(driven_residual) < 3e-5

    # Reciprocity and losslessness are independent of the analytical phase.
    np.testing.assert_allclose(
        scattering[:, 0, 1],
        scattering[:, 1, 0],
        rtol=1e-5,
        atol=1e-8,
    )
    identity = np.eye(2)
    unitarity_error = max(np.linalg.norm(matrix.conj().T @ matrix - identity, ord=2) for matrix in scattering)
    assert unitarity_error < 3e-4

    wave_speed = 299_792_458.0
    broad_wall = 0.010
    reference_plane_spacing = 0.020
    cutoff = wave_speed / (2 * broad_wall)
    assert np.all(frequency > cutoff)
    propagation_constant = np.sqrt((2 * np.pi * frequency / wave_speed) ** 2 - (np.pi / broad_wall) ** 2)
    analytical_transmission = np.exp(-1j * propagation_constant * reference_plane_spacing)
    measured_transmission = scattering[:, 1, 0]
    assert np.max(np.abs(np.abs(measured_transmission) - 1)) < 2.5e-3
    phase_error = np.angle(measured_transmission / analytical_transmission)
    assert np.max(np.abs(phase_error)) < np.deg2rad(1.5)


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
    with h5py.File(tmp_path / "broadband_reused2.h5") as reused, h5py.File(tmp_path / "broadband_fresh2.h5") as fresh:
        for port_index in (1, 2):
            for dataset in ("incident", "outgoing", "S"):
                np.testing.assert_allclose(
                    reused[f"eigenmode_ports/port{port_index}/{dataset}"][...],
                    fresh[f"eigenmode_ports/port{port_index}/{dataset}"][...],
                    rtol=2e-10,
                    atol=2e-10,
                )


def _assert_device_eigenmode_study_matches_cpu(tmp_path, name, *, virtual=False, broadband=False, **device_options):
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
