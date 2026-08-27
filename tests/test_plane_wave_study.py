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

"""Reusable discrete-plane-wave and RCS studies."""

import json

import h5py
import numpy as np
import pytest

import gprMax


def _two_dimensional_scene(phi=0, amplitude=1.0):
    inf = float("inf")
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(1e-3,) * 3))
    scene.add(gprMax.Domain(p1=(0.04, 0.04, inf)))
    scene.add(gprMax.TimeWindow(time=2e-10))
    scene.add(gprMax.PMLThickness(thickness=(3, 3, 0, 3, 3, 0)))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=amplitude, freq=8e9, id="pulse"))
    plane_wave = gprMax.DiscretePlaneWaveAngles(
        p1=(0.010, 0.010, inf),
        p2=(0.030, 0.030, inf),
        theta=90,
        phi=phi,
        psi=90,
        waveform_id="pulse",
    )
    scene.add(plane_wave)
    scene.add(gprMax.Rx(p1=(0.020, 0.020, inf), id="probe"))
    return scene, plane_wave


def _rcs_scene(phi=0):
    dl = 0.002
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl,) * 3))
    scene.add(gprMax.Domain(p1=(0.048,) * 3))
    scene.add(gprMax.TimeWindow(iterations=240))
    scene.add(gprMax.PMLThickness(thickness=3))
    scene.add(gprMax.OMPThreads(n=1))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=8e9, id="pulse"))
    plane_wave = gprMax.DiscretePlaneWaveAngles(
        p1=(0.018,) * 3,
        p2=(0.030,) * 3,
        theta=90,
        phi=phi,
        psi=90,
        waveform_id="pulse",
    )
    scene.add(plane_wave)
    scene.add(gprMax.Sphere(p1=(0.024,) * 3, r=0.004, material_id="pec"))
    scene.add(
        gprMax.NTFFSurface(
            p1=(0.012,) * 3,
            p2=(0.036,) * 3,
            id="surface",
            origin=(0.024,) * 3,
        )
    )
    scene.add(
        gprMax.KSIRFrequencyTransform(
            "surface",
            "spectrum",
            (8e9,),
            save_surface_dft=False,
            plane_wave_index=0,
        )
    )
    scene.add(
        gprMax.KSIRFarField(
            theta=(90, 90),
            phi=(0, 180),
            transform_id="spectrum",
            id="pattern",
            outputs=("Etheta", "rcs"),
        )
    )
    return scene, plane_wave


def _axial_scene(axis="x"):
    inf = float("inf")
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(1e-3,) * 3))
    scene.add(gprMax.Domain(p1=(0.04, 0.04, inf)))
    scene.add(gprMax.TimeWindow(time=2e-10))
    scene.add(gprMax.PMLThickness(thickness=(3, 3, 0, 3, 3, 0)))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=8e9, id="pulse"))
    plane_wave = gprMax.DiscretePlaneWaveAxial(
        p1=(0.010, 0.010, inf),
        p2=(0.030, 0.030, inf),
        axis=axis,
        psi=90,
        waveform_id="pulse",
    )
    scene.add(plane_wave)
    scene.add(gprMax.Rx(p1=(0.020, 0.020, inf), id="probe"))
    return scene, plane_wave


@pytest.mark.integration
def test_plane_wave_study_rebuilds_dpw_and_matches_fresh_runs(tmp_path):
    scene, plane_wave = _two_dimensional_scene()
    study = gprMax.PlaneWaveStudy(
        [
            gprMax.StudyCase(
                "x_incidence", [gprMax.ObjectState(plane_wave, phi=0, scale=1)]
            ),
            gprMax.StudyCase(
                "y_incidence", [gprMax.ObjectState(plane_wave, phi=90, scale=0.5)]
            ),
        ]
    )
    gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=tmp_path / "reused",
        hide_progress_bars=True,
        log_level=30,
        cpu_precision="double",
    )

    for index, (phi, amplitude) in enumerate(((0, 1), (90, 0.5)), start=1):
        fresh_scene, _ = _two_dimensional_scene(phi=phi, amplitude=1)
        gprMax.run(
            scenes=[fresh_scene],
            outputfile=tmp_path / f"fresh{index}",
            hide_progress_bars=True,
            log_level=30,
            cpu_precision="double",
        )
        with h5py.File(tmp_path / f"reused{index}.h5") as reused, h5py.File(
            tmp_path / f"fresh{index}.h5"
        ) as fresh:
            np.testing.assert_allclose(
                reused["rxs/rx1/Ez"],
                amplitude * fresh["rxs/rx1/Ez"][...],
                rtol=2e-12,
                atol=2e-12,
            )
            resolved = json.loads(reused["study/resolved_case"][()].decode())
            assert resolved["objects"]["plane_wave_1"]["scale"] == amplitude
            assert resolved["objects"]["plane_wave_1"]["actual_angles"][1] == phi


@pytest.mark.integration
def test_plane_wave_study_recreates_ntff_rcs_state_per_case(tmp_path):
    scene, plane_wave = _rcs_scene()
    study = gprMax.PlaneWaveStudy(
        [
            gprMax.StudyCase("x_incidence", [gprMax.ObjectState(plane_wave, phi=0)]),
            gprMax.StudyCase("y_incidence", [gprMax.ObjectState(plane_wave, phi=90)]),
        ]
    )
    gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=tmp_path / "reused_rcs",
        hide_progress_bars=True,
        log_level=30,
        cpu_precision="double",
    )

    path = "ntff/surface/frequency/spectrum/far_field/pattern/fields/rcs"
    for index, phi in enumerate((0, 90), start=1):
        fresh_scene, _ = _rcs_scene(phi=phi)
        gprMax.run(
            scenes=[fresh_scene],
            outputfile=tmp_path / f"fresh_rcs{index}",
            hide_progress_bars=True,
            log_level=30,
            cpu_precision="double",
        )
        with h5py.File(tmp_path / f"reused_rcs{index}.h5") as reused, h5py.File(
            tmp_path / f"fresh_rcs{index}.h5"
        ) as fresh:
            np.testing.assert_allclose(reused[path], fresh[path], rtol=2e-11, atol=1e-18)
            plane_wave_group = reused["ntff/surface/frequency/spectrum/plane_wave"]
            assert plane_wave_group.attrs["actual_angles"][1] == phi


@pytest.mark.integration
def test_plane_wave_study_rebuilds_axial_material_profile(tmp_path):
    scene, plane_wave = _axial_scene()
    study = gprMax.PlaneWaveStudy(
        [
            gprMax.StudyCase("x_incidence", [gprMax.ObjectState(plane_wave, axis="x")]),
            gprMax.StudyCase("y_incidence", [gprMax.ObjectState(plane_wave, axis="y")]),
        ]
    )
    gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=tmp_path / "reused_axial",
        hide_progress_bars=True,
        log_level=30,
        cpu_precision="double",
    )

    for index, axis in enumerate(("x", "y"), start=1):
        fresh_scene, _ = _axial_scene(axis=axis)
        gprMax.run(
            scenes=[fresh_scene],
            outputfile=tmp_path / f"fresh_axial{index}",
            hide_progress_bars=True,
            log_level=30,
            cpu_precision="double",
        )
        with h5py.File(tmp_path / f"reused_axial{index}.h5") as reused, h5py.File(
            tmp_path / f"fresh_axial{index}.h5"
        ) as fresh:
            np.testing.assert_allclose(
                reused["rxs/rx1/Ez"], fresh["rxs/rx1/Ez"], rtol=2e-12, atol=2e-12
            )


def test_plane_wave_study_csv_and_type_specific_validation(tmp_path):
    cases = tmp_path / "angles.csv"
    cases.write_text(
        "case_id,object_id,theta_deg,phi_deg,psi_deg,scale\n"
        "front,plane_wave_1,90,0,90,1\n"
        "side,plane_wave_1,90,90,90,0.5\n"
    )
    study = gprMax.Study.from_csv("plane_wave", cases)
    assert isinstance(study, gprMax.PlaneWaveStudy)
    assert study.cases[1].states[0].parameters == {
        "theta": 90,
        "phi": 90,
        "psi": 90,
        "scale": 0.5,
    }

    vectors = tmp_path / "vectors.csv"
    vectors.write_text(
        "case_id,object_id,m_x,m_y,m_z,psi_deg\n"
        "x,plane_wave_1,1,0,0,90\n"
        "reverse_y,plane_wave_1,0,-1,0,90\n"
    )
    vector_study = gprMax.Study.from_csv("plane_wave", vectors)
    assert vector_study.cases[1].states[0].parameters == {
        "psi": 90,
        "m_vec": (0, -1, 0),
    }

    vector_scene = gprMax.Scene()
    vector = gprMax.DiscretePlaneWaveVector(
        p1=(0.01,) * 3,
        p2=(0.02,) * 3,
        m_vec=(1, 0, 0),
        psi=0,
        waveform_id="pulse",
    )
    vector_scene.add(vector)
    wrong = gprMax.PlaneWaveStudy(
        [gprMax.StudyCase("wrong", [gprMax.ObjectState(vector, theta=90)])]
    )
    with pytest.raises(ValueError, match="unsupported parameter.*theta"):
        wrong.bind_scene(vector_scene)


@pytest.mark.integration
def test_hash_plane_wave_study_runs_all_csv_cases(tmp_path):
    cases = tmp_path / "angles.csv"
    cases.write_text(
        "case_id,object_id,theta_deg,phi_deg,psi_deg\n"
        "x,plane_wave_1,90,0,90\n"
        "y,plane_wave_1,90,90,90\n"
    )
    inputfile = tmp_path / "plane_wave.in"
    inputfile.write_text(
        "#title: reusable plane-wave study\n"
        "#dx_dy_dz: 0.002 0.002 0.002\n"
        "#domain: 0.04 0.04 0.04\n"
        "#pml_cells: 3\n"
        "#time_window: 2e-10\n"
        "#waveform: ricker 1 8e9 pulse\n"
        "#plane_wave_angles: 0.012 0.012 0.012 0.028 0.028 0.028 "
        "90 0 90 pulse\n"
        "#rx: 0.02 0.02 0.02\n"
        f"#study: plane_wave {cases.name}\n"
    )
    gprMax.run(
        inputfile=inputfile,
        outputfile=tmp_path / "hash",
        hide_progress_bars=True,
        log_level=30,
        cpu_precision="double",
    )

    for index, phi in enumerate((0, 90), start=1):
        with h5py.File(tmp_path / f"hash{index}.h5") as output:
            assert output["study"].attrs["Type"] == "plane_wave"
            resolved = json.loads(output["study/resolved_case"][()].decode())
            assert resolved["objects"]["plane_wave_1"]["actual_angles"][1] == phi
            assert output["study/source"][()].decode() == cases.read_text()


def test_plane_wave_study_rejects_multiple_or_zero_incident_sources():
    scene, plane_wave = _two_dimensional_scene()
    duplicate = gprMax.DiscretePlaneWaveAngles(
        p1=(0.010, 0.010, float("inf")),
        p2=(0.030, 0.030, float("inf")),
        theta=90,
        phi=90,
        psi=90,
        waveform_id="pulse",
    )
    scene.add(duplicate)
    study = gprMax.PlaneWaveStudy(
        [gprMax.StudyCase("one", [gprMax.ObjectState(plane_wave, phi=0)])]
    )
    with pytest.raises(ValueError, match="exactly one plane-wave template"):
        study.bind_scene(scene)

    single_scene, single = _two_dimensional_scene()
    zero = gprMax.PlaneWaveStudy(
        [gprMax.StudyCase("zero", [gprMax.ObjectState(single, scale=0)])]
    )
    with pytest.raises(ValueError, match="non-zero"):
        zero.bind_scene(single_scene)


def test_plane_wave_study_rejects_subgrid_excitation():
    scene, plane_wave = _two_dimensional_scene()
    subgrid = gprMax.SubGridHSG(
        p1=(0.01, 0.01, 0),
        p2=(0.03, 0.03, 0),
        ratio=3,
        id="fine",
    )
    subgrid.add(
        gprMax.HertzianDipole(
            polarisation="z",
            p1=(0.02, 0.02, 0),
            waveform_id="pulse",
        )
    )
    scene.add(subgrid)
    study = gprMax.PlaneWaveStudy(
        [gprMax.StudyCase("one", [gprMax.ObjectState(plane_wave, phi=0)])]
    )

    with pytest.raises(ValueError, match="HertzianDipole on subgrid 'fine'"):
        study.bind_scene(scene)


def _run_device_study(tmp_path, name, **backend):
    scene, plane_wave = _two_dimensional_scene()
    study = gprMax.PlaneWaveStudy(
        [
            gprMax.StudyCase("x", [gprMax.ObjectState(plane_wave, phi=0)]),
            gprMax.StudyCase("y", [gprMax.ObjectState(plane_wave, phi=90)]),
        ]
    )
    gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=tmp_path / name,
        hide_progress_bars=True,
        log_level=30,
        **backend,
    )
    traces = []
    for index in (1, 2):
        with h5py.File(tmp_path / f"{name}{index}.h5") as output:
            traces.append(output["rxs/rx1/Ez"][...])
    return traces


@pytest.mark.integration
@pytest.mark.gpu
def test_cuda_plane_wave_study_matches_cpu(tmp_path, gpu_device):
    cpu = _run_device_study(tmp_path, "cpu_cuda", cpu_precision="single")
    cuda = _run_device_study(
        tmp_path,
        "cuda",
        gpu=[gpu_device],
        gpu_precision="single",
    )
    scale = max(np.max(np.abs(trace)) for trace in cpu)
    for expected, actual in zip(cpu, cuda):
        np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4 * scale)


@pytest.mark.integration
@pytest.mark.gpu
def test_opencl_plane_wave_study_matches_cpu(tmp_path, opencl_device):
    cpu = _run_device_study(tmp_path, "cpu_opencl", cpu_precision="single")
    opencl = _run_device_study(
        tmp_path,
        "opencl",
        opencl=[opencl_device],
        gpu_precision="single",
    )
    scale = max(np.max(np.abs(trace)) for trace in cpu)
    for expected, actual in zip(cpu, opencl):
        np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4 * scale)
