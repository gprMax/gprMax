# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

"""Reusable finite-resistance voltage-port studies."""

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.ports import correct_s11_for_parallel_gap, correct_smatrix_for_parallel_gaps


def _two_port_scene(active_port=None, resistances=(50, 50)):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
    scene.add(gprMax.Domain(p1=(0.024, 0.024, 0.024)))
    scene.add(gprMax.TimeWindow(time=4e-10))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="on"))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=0, freq=5e9, id="off"))

    sources = []
    for index, (x, resistance) in enumerate(zip((0.010, 0.014), resistances), start=1):
        waveform_id = "on" if active_port in (None, index) else "off"
        source = gprMax.VoltageSource(
            p1=(x, 0.012, 0.012),
            polarisation="z",
            resistance=resistance,
            waveform_id=waveform_id,
            id=f"p{index}",
        )
        sources.append(source)
        scene.add(source)
    return scene, sources


def _two_port_study(sources):
    return gprMax.PortStudy(
        [
            gprMax.StudyCase("drive_p1", [gprMax.ObjectState(sources[0], scale=1)]),
            gprMax.StudyCase("drive_p2", [gprMax.ObjectState(sources[1], scale=1)]),
        ]
    )


@pytest.mark.integration
def test_two_port_study_matches_fresh_models_and_writes_smatrix(tmp_path):
    scene, sources = _two_port_scene()
    study = _two_port_study(sources)
    returned = gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=tmp_path / "ports",
        hide_progress_bars=True,
        log_level=30,
    )

    for active_port in (1, 2):
        fresh_scene, _ = _two_port_scene(active_port=active_port)
        gprMax.run(
            scenes=[fresh_scene],
            outputfile=tmp_path / f"fresh{active_port}",
            hide_progress_bars=True,
            log_level=30,
        )
    combined_scene, _ = _two_port_scene()
    gprMax.run(
        scenes=[combined_scene],
        outputfile=tmp_path / "fresh_combined",
        hide_progress_bars=True,
        log_level=30,
    )

    assert returned["study"] is study.result
    assert study.result.output_file == tmp_path / "ports_study.h5"
    assert study.result.s.shape[1:] == (2, 2)
    assert study.result.valid_s.any()

    for case_index in (1, 2):
        with h5py.File(tmp_path / f"ports{case_index}.h5") as reused, h5py.File(
            tmp_path / f"fresh{case_index}.h5"
        ) as fresh:
            assert reused["study/port_response"].attrs["DrivenPortID"] == f"p{case_index}"
            for port_id in ("p1", "p2"):
                np.testing.assert_array_equal(
                    reused[f"ports/{port_id}/Vtotal"],
                    fresh[f"ports/{port_id}/Vtotal"],
                )
            passive = 2 if case_index == 1 else 1
            assert np.all(reused[f"ports/p{passive}/Vgenerator"][...] == 0)
            assert np.all(reused[f"srcs/src{passive}/excitation/samples"][...] == 0)

    with h5py.File(tmp_path / "ports1.h5") as first, h5py.File(
        tmp_path / "ports2.h5"
    ) as second, h5py.File(tmp_path / "fresh_combined.h5") as combined:
        for port_id in ("p1", "p2"):
            np.testing.assert_allclose(
                first[f"ports/{port_id}/Vtotal"][...] + second[f"ports/{port_id}/Vtotal"][...],
                combined[f"ports/{port_id}/Vtotal"][...],
                rtol=2e-6,
                atol=2e-6,
            )

    with h5py.File(tmp_path / "ports_study.h5") as output:
        assert output.attrs["StudyType"] == "port"
        assert output.attrs["MatrixConvention"] == "S[frequency, output_port, input_port]"
        assert [item.decode() for item in output["port_ids"][...]] == ["p1", "p2"]
        source = output["S_source"][...]
        corrected = output["S"][...]
        valid = output["valid_S"][...].astype(bool)
        assert source.shape == corrected.shape == valid.shape
        assert valid.any()

        reciprocal = valid[:, 0, 1] & valid[:, 1, 0]
        assert reciprocal.any()
        np.testing.assert_allclose(
            source[reciprocal, 0, 1],
            source[reciprocal, 1, 0],
            rtol=2e-6,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            corrected[reciprocal, 0, 1],
            corrected[reciprocal, 1, 0],
            rtol=2e-6,
            atol=2e-6,
        )

        for input_index in range(2):
            with h5py.File(tmp_path / f"ports{input_index + 1}.h5") as case:
                drive = case[f"ports/p{input_index + 1}"]
                incident = drive["Vincident_spectrum"][...]
                for output_index in range(2):
                    reflected = case[f"ports/p{output_index + 1}/Vreflected_source_spectrum"][...]
                    expected = np.full(incident.shape, np.nan + 1j * np.nan, dtype=source.dtype)
                    np.divide(reflected, incident, out=expected, where=np.abs(incident) > 0)
                    finite = np.isfinite(source[:, output_index, input_index])
                    np.testing.assert_allclose(
                        source[finite, output_index, input_index],
                        expected[finite],
                        rtol=2e-6,
                        atol=2e-6,
                    )


def test_multiport_gap_correction_reduces_to_existing_one_port_formula():
    s11_source = np.asarray([0.1 + 0.2j, -0.25 + 0.05j], dtype=np.complex128)
    correction = np.asarray([0.02 + 0.1j, 0.03 + 0.2j], dtype=np.complex128)
    scalar, scalar_valid = correct_s11_for_parallel_gap(s11_source, correction, np.complex128)
    matrix, matrix_valid = correct_smatrix_for_parallel_gaps(
        s11_source[:, np.newaxis, np.newaxis],
        correction[:, np.newaxis],
        np.complex128,
    )

    np.testing.assert_allclose(matrix[:, 0, 0], scalar, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(matrix_valid, scalar_valid)


def test_multiport_gap_correction_recovers_coupled_two_port_matrix():
    identity = np.eye(2, dtype=np.complex128)
    device_admittance = np.asarray([[0.31 + 0.08j, -0.07 + 0.02j], [-0.07 + 0.02j, 0.44 + 0.11j]])
    gap = np.asarray([0.03 + 0.09j, 0.05 + 0.13j])

    def admittance_to_s(admittance):
        return np.linalg.solve(identity + admittance, identity - admittance)

    expected = admittance_to_s(device_admittance)
    source = admittance_to_s(device_admittance + np.diag(gap))
    corrected, valid = correct_smatrix_for_parallel_gaps(
        source[np.newaxis, ...], gap[np.newaxis, ...], np.complex128
    )

    assert valid[0]
    np.testing.assert_allclose(corrected[0], expected, rtol=1e-13, atol=1e-13)


@pytest.mark.integration
def test_port_study_uses_power_waves_for_unequal_reference_impedances(tmp_path):
    scene, sources = _two_port_scene(resistances=(50, 75))
    study = _two_port_study(sources)
    gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=tmp_path / "unequal",
        hide_progress_bars=True,
        log_level=30,
    )

    for input_index, input_impedance in enumerate((50, 75)):
        with h5py.File(tmp_path / f"unequal{input_index + 1}.h5") as output:
            incident = output[f"ports/p{input_index + 1}/Vincident_spectrum"][...]
            for output_index, output_impedance in enumerate((50, 75)):
                reflected = output[f"ports/p{output_index + 1}/Vreflected_source_spectrum"][...]
                expected = np.full(incident.shape, np.nan + 1j * np.nan)
                np.divide(reflected, incident, out=expected, where=np.abs(incident) > 0)
                expected *= np.sqrt(input_impedance / output_impedance)
                finite = np.isfinite(study.result.s_source[:, output_index, input_index])
                np.testing.assert_allclose(
                    study.result.s_source[finite, output_index, input_index],
                    expected[finite],
                    rtol=2e-6,
                    atol=2e-6,
                )


def test_port_study_rejects_hard_sources_and_incomplete_drive_schedule():
    scene, sources = _two_port_scene()
    sources[0].resistance = 0
    study = gprMax.PortStudy(
        [
            gprMax.StudyCase("drive_p1", [gprMax.ObjectState(sources[0])]),
            gprMax.StudyCase("drive_p2", [gprMax.ObjectState(sources[1])]),
        ]
    )
    with pytest.raises(ValueError, match="hard voltage sources"):
        study.bind_scene(scene)

    scene, sources = _two_port_scene()
    duplicate = gprMax.PortStudy(
        [
            gprMax.StudyCase("first", [gprMax.ObjectState(sources[0])]),
            gprMax.StudyCase("again", [gprMax.ObjectState(sources[0])]),
        ]
    )
    with pytest.raises(ValueError, match="exactly one case"):
        duplicate.bind_scene(scene)


@pytest.mark.integration
def test_port_study_restart_preserves_compatible_completed_columns(tmp_path):
    output = tmp_path / "restart"
    scene, sources = _two_port_scene()
    study = _two_port_study(sources)
    gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=output,
        hide_progress_bars=True,
        log_level=30,
    )
    with h5py.File(tmp_path / "restart_study.h5") as complete:
        original = complete["S_source"][...]
        assert complete.attrs["Complete"]

    restart_scene, restart_sources = _two_port_scene()
    restarted = _two_port_study(restart_sources)
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
        np.testing.assert_array_equal(complete["S_source"][...], original)


def test_port_study_csv_factory(tmp_path):
    cases = tmp_path / "ports.csv"
    cases.write_text(
        "case_id,object_id,active,scale\n"
        "drive_p1,voltage_source_1,true,1\n"
        "drive_p2,voltage_source_2,true,1\n"
    )

    study = gprMax.Study.from_csv("port", cases)

    assert isinstance(study, gprMax.PortStudy)
    assert [case.id for case in study.cases] == ["drive_p1", "drive_p2"]


@pytest.mark.integration
def test_hash_port_study_runs_and_writes_complete_smatrix(tmp_path):
    cases = tmp_path / "ports.csv"
    cases.write_text(
        "case_id,object_id,active,scale\n"
        "drive_p1,voltage_source_1,true,1\n"
        "drive_p2,voltage_source_2,true,1\n"
    )
    inputfile = tmp_path / "ports.in"
    inputfile.write_text(
        "#title: hash port study integration\n"
        "#dx_dy_dz: 0.002 0.002 0.002\n"
        "#domain: 0.024 0.024 0.024\n"
        "#pml_cells: 2\n"
        "#time_window: 4e-10\n"
        "#waveform: ricker 1 5e9 pulse\n"
        "#voltage_source: z 0.010 0.012 0.012 50 pulse 0 4e-10 p1 10\n"
        "#voltage_source: z 0.014 0.012 0.012 50 pulse 0 4e-10 p2 10\n"
        f"#study: port {cases.name}\n"
    )

    returned = gprMax.run(
        inputfile=inputfile,
        outputfile=tmp_path / "hash_ports",
        hide_progress_bars=True,
        log_level=30,
    )

    assert isinstance(returned["study"], gprMax.PortStudyResult)
    with h5py.File(tmp_path / "hash_ports_study.h5") as output:
        assert output.attrs["Complete"]
        assert output.attrs["CasesCompleted"] == 2
        assert [item.decode() for item in output["port_ids"][...]] == ["p1", "p2"]
        assert output["S"].shape[1:] == (2, 2)


def _assert_device_port_study_matches_cpu(tmp_path, device_name, **device_options):
    cpu_scene, cpu_sources = _two_port_scene()
    device_scene, device_sources = _two_port_scene()
    cpu = _two_port_study(cpu_sources)
    device = _two_port_study(device_sources)

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
        outputfile=tmp_path / device_name,
        hide_progress_bars=True,
        log_level=30,
        **device_options,
    )

    np.testing.assert_array_equal(device.result.frequency, cpu.result.frequency)
    valid = cpu.result.valid_s_source & device.result.valid_s_source
    assert valid.any()
    np.testing.assert_allclose(
        device.result.s_source[valid],
        cpu.result.s_source[valid],
        rtol=3e-4,
        atol=3e-4,
    )
    valid = cpu.result.valid_s & device.result.valid_s
    assert valid.any()
    np.testing.assert_allclose(device.result.s[valid], cpu.result.s[valid], rtol=3e-4, atol=3e-4)


@pytest.mark.integration
@pytest.mark.gpu
def test_cuda_port_study_matches_cpu(tmp_path, gpu_device):
    _assert_device_port_study_matches_cpu(tmp_path, "cuda", gpu=[gpu_device])


@pytest.mark.integration
@pytest.mark.gpu
def test_opencl_port_study_matches_cpu(tmp_path, opencl_device):
    _assert_device_port_study_matches_cpu(tmp_path, "opencl", opencl=[opencl_device])
