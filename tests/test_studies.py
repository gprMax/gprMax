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

"""Tests for reusable-geometry parameter studies."""

import argparse
import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest

import gprMax
from gprMax.studies import Study, preflight_study_args
from gprMax.toolboxes.Plotting import plot_Ascan, plot_Bscan
from gprMax.toolboxes.Utilities.outputfiles_merge import get_output_data, merge_files


def _scene(
    source_position=(0.01, 0.01, 0.01),
    receiver_position=(0.015, 0.01, 0.01),
    waveform_amplitude=1.0,
):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=4e-10))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=waveform_amplitude, freq=5e9, id="wave"))
    source = gprMax.HertzianDipole(polarisation="z", p1=source_position, waveform_id="wave")
    receiver = gprMax.Rx(p1=receiver_position, id="probe")
    scene.add(source)
    scene.add(receiver)
    return scene, source, receiver


@pytest.mark.integration
def test_api_study_reuses_geometry_and_writes_exact_case_metadata(tmp_path):
    scene, source, receiver = _scene()
    study = gprMax.GPRStudy(
        [
            gprMax.StudyCase(
                "baseline",
                [
                    gprMax.ObjectState(source, scale=1.0),
                    gprMax.ObjectState(receiver, position=(0.015, 0.01, 0.01)),
                ],
            ),
            gprMax.StudyCase(
                "moved_scaled",
                [
                    gprMax.ObjectState(
                        "hertzian_dipole_1", position=(0.009, 0.01, 0.01), scale=0.5
                    ),
                    gprMax.ObjectState("probe", position=(0.014, 0.01, 0.01)),
                ],
            ),
        ]
    )
    output = tmp_path / "study"

    gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=output,
        hide_progress_bars=True,
        log_level=30,
    )
    fresh_scene, _, _ = _scene(
        source_position=(0.009, 0.01, 0.01),
        receiver_position=(0.014, 0.01, 0.01),
        waveform_amplitude=0.5,
    )
    fresh_output = tmp_path / "fresh"
    gprMax.run(
        scenes=[fresh_scene],
        outputfile=fresh_output,
        hide_progress_bars=True,
        log_level=30,
    )

    with h5py.File(tmp_path / "study1.h5") as first, h5py.File(
        tmp_path / "study2.h5"
    ) as second, h5py.File(tmp_path / "fresh.h5") as fresh:
        assert first["study"].attrs["CaseID"] == "baseline"
        assert second["study"].attrs["CaseID"] == "moved_scaled"
        assert not first["study"].attrs["GeometryReused"]
        assert second["study"].attrs["GeometryReused"]
        assert np.allclose(first["srcs/src1"].attrs["Position"], (0.01, 0.01, 0.01))
        assert np.allclose(second["srcs/src1"].attrs["Position"], (0.009, 0.01, 0.01))
        assert np.allclose(second["rxs/rx1"].attrs["Position"], (0.014, 0.01, 0.01))

        excitation1 = first["srcs/src1/excitation/samples"][:]
        excitation2 = second["srcs/src1/excitation/samples"][:]
        assert np.max(np.abs(excitation1)) > 0.9
        assert np.allclose(excitation2, 0.5 * excitation1)
        resolved = json.loads(second["study/resolved_case"][()].decode())
        assert resolved["objects"]["hertzian_dipole_1"]["scale"] == 0.5
        assert np.array_equal(second["rxs/rx1/Ez"][:], fresh["rxs/rx1/Ez"][:])


def test_hash_study_csv_preflight_sets_case_count_and_restart(tmp_path):
    csvfile = tmp_path / "cases.csv"
    csvfile.write_text(
        "case_id,object_id,active,x_m,y_m,z_m,waveform_id,start_s,stop_s,scale,record\n"
        "one,hertzian_dipole_1,true,,,,,,,1,\n"
        "two,hertzian_dipole_1,true,0.009,0.01,0.01,,,,0.5,\n"
        "three,hertzian_dipole_1,false,,,,,,,,\n"
    )
    inputfile = tmp_path / "model.in"
    inputfile.write_text(f"#study: gpr {csvfile.name}\n")

    args = argparse.Namespace(
        study=None,
        inputfile=str(inputfile),
        taskfarm=False,
        mpi=None,
        scenes=None,
        geometry_fixed=False,
        n=99,
        i=2,
    )
    study = preflight_study_args(args)

    assert isinstance(study, Study)
    assert [case.id for case in study.cases] == ["one", "two", "three"]
    assert args.geometry_fixed
    assert args.n == 2


@pytest.mark.integration
def test_study_record_false_keeps_layout_and_skips_receiver_data(tmp_path):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=4e-10))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="wave"))
    source = gprMax.HertzianDipole(polarisation="z", p1=(0.01, 0.01, 0.01), waveform_id="wave")
    scene.add(source)
    scene.add(gprMax.Rx(p1=(0.014, 0.01, 0.01), id="first"))
    scene.add(gprMax.Rx(p1=(0.016, 0.01, 0.01), id="second"))
    study = gprMax.GPRStudy(
        [
            gprMax.StudyCase("all", [gprMax.ObjectState(source, scale=1.0)]),
            gprMax.StudyCase(
                "mute_second",
                [
                    gprMax.ObjectState(source, scale=1.0),
                    gprMax.ObjectState("rx_2", position=(0.017, 0.01, 0.01), record=False),
                ],
            ),
            gprMax.StudyCase("restored", [gprMax.ObjectState(source, scale=1.0)]),
        ]
    )

    gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=tmp_path / "selective",
        hide_progress_bars=True,
        log_level=30,
    )

    with h5py.File(tmp_path / "selective1.h5") as all_rx, h5py.File(
        tmp_path / "selective2.h5"
    ) as muted, h5py.File(tmp_path / "selective3.h5") as restored:
        assert all_rx.attrs["nrx"] == 2
        assert muted.attrs["nrx"] == 2
        assert bool(all_rx["rxs/rx2"].attrs["Recorded"])
        assert not bool(muted["rxs/rx2"].attrs["Recorded"])
        assert "Ez" in all_rx["rxs/rx2"]
        assert "Ez" not in muted["rxs/rx2"]
        assert np.array_equal(all_rx["rxs/rx1/Ez"][:], muted["rxs/rx1/Ez"][:])
        assert np.allclose(muted["rxs/rx2"].attrs["Position"], (0.017, 0.01, 0.01))
        assert bool(restored["rxs/rx2"].attrs["Recorded"])
        assert np.array_equal(all_rx["rxs/rx2/Ez"][:], restored["rxs/rx2/Ez"][:])
        resolved = json.loads(muted["study/resolved_case"][()].decode())
        assert resolved["objects"]["rx_2"]["record"] is False
        assert resolved["objects"]["rx_1"]["record"] is True


@pytest.mark.integration
def test_muted_study_trace_stays_out_of_plots_and_merges_as_nan(tmp_path, monkeypatch):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=4e-10))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="wave"))
    source = gprMax.HertzianDipole(polarisation="z", p1=(0.01, 0.01, 0.01), waveform_id="wave")
    scene.add(source)
    scene.add(gprMax.Rx(p1=(0.014, 0.01, 0.01), id="first"))
    scene.add(gprMax.Rx(p1=(0.016, 0.01, 0.01), id="second"))
    study = gprMax.GPRStudy(
        [
            gprMax.StudyCase("all", [gprMax.ObjectState(source, scale=1.0)]),
            gprMax.StudyCase(
                "mute_second",
                [
                    gprMax.ObjectState(source, scale=1.0),
                    gprMax.ObjectState("rx_2", record=False),
                ],
            ),
        ]
    )
    gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=tmp_path / "selective",
        hide_progress_bars=True,
        log_level=30,
    )
    recorded, muted = tmp_path / "selective1.h5", tmp_path / "selective2.h5"

    figures = []
    monkeypatch.setattr(plot_Ascan, "handle_plot_output", lambda _plt, fig, *a, **k: figures.append(fig))
    monkeypatch.setattr(plot_Bscan, "handle_plot_output", lambda _plt, fig, *a, **k: figures.append(fig))
    try:
        with pytest.warns(UserWarning, match="was not recorded"):
            plot_Ascan.mpl_plot(muted, outputs=["Ez"], show=False)
        assert len(figures) == 1
    finally:
        plt.close("all")

    samples, _ = plot_Bscan.gather_receiver_outputs(muted, "Ez")
    assert samples.shape[1] == 1

    merged = merge_files([recorded, muted], tmp_path / "selective_merged.h5")
    data, _ = get_output_data(merged, 1, "Ez", receiver_name="second")
    assert np.isfinite(data[:, 0]).all()
    assert np.isnan(data[:, 1]).all()
    with h5py.File(merged, "r") as handle:
        assert "Recorded" not in handle["rxs/rx2"].attrs
        np.testing.assert_array_equal(handle["trace_metadata/rxs/rx2/Recorded"], [True, False])

    figures.clear()
    samples, dt, offset = plot_Bscan.gather_receiver_outputs(merged, "Ez", return_time_offset=True)
    try:
        plot_Bscan.mpl_plot(
            merged, samples, dt, 1, "Ez", show=False, trace_group="gathered", time_offset=offset
        )
        assert len(figures) == 1
    finally:
        plt.close("all")

    with pytest.raises(ValueError, match="was not recorded"):
        get_output_data(muted, 2, "Ez")


@pytest.mark.integration
def test_hash_study_record_false_from_csv(tmp_path):
    csvfile = tmp_path / "record.csv"
    csvfile.write_text(
        "case_id,object_id,x_m,y_m,z_m,record\n"
        "all,rx_1,0.015,0.01,0.01,true\n"
        "muted,rx_1,0.016,0.01,0.01,false\n"
    )
    inputfile = tmp_path / "model.in"
    inputfile.write_text(
        "#title: hash study record\n"
        "#dx_dy_dz: 0.001 0.001 0.001\n"
        "#domain: 0.02 0.02 0.02\n"
        "#pml_cells: 0\n"
        "#time_window: 4e-10\n"
        "#waveform: ricker 1 5e9 wave\n"
        "#hertzian_dipole: z 0.01 0.01 0.01 wave\n"
        "#rx: 0.015 0.01 0.01\n"
        f"#study: gpr {csvfile.name}\n"
    )

    gprMax.run(
        inputfile=inputfile,
        outputfile=tmp_path / "hash_record",
        hide_progress_bars=True,
        log_level=30,
    )

    with h5py.File(tmp_path / "hash_record1.h5") as recorded, h5py.File(
        tmp_path / "hash_record2.h5"
    ) as muted:
        assert "Ez" in recorded["rxs/rx1"]
        assert bool(recorded["rxs/rx1"].attrs["Recorded"])
        assert "Ez" not in muted["rxs/rx1"]
        assert not bool(muted["rxs/rx1"].attrs["Recorded"])


@pytest.mark.integration
def test_hash_study_runs_all_cases_and_disables_omitted_source(tmp_path):
    csvfile = tmp_path / "cases.csv"
    csvfile.write_text(
        "case_id,object_id,active,x_m,y_m,z_m,scale\n"
        "on,hertzian_dipole_1,true,0.01,0.01,0.01,1\n"
        "on,rx_1,,0.015,0.01,0.01,\n"
        "off,rx_1,,0.014,0.01,0.01,\n"
    )
    inputfile = tmp_path / "model.in"
    inputfile.write_text(
        "#title: hash study integration\n"
        "#dx_dy_dz: 0.001 0.001 0.001\n"
        "#domain: 0.02 0.02 0.02\n"
        "#pml_cells: 0\n"
        "#time_window: 4e-10\n"
        "#waveform: ricker 1 5e9 wave\n"
        "#hertzian_dipole: z 0.01 0.01 0.01 wave\n"
        "#rx: 0.015 0.01 0.01\n"
        f"#study: gpr {csvfile.name}\n"
    )
    output = tmp_path / "hash_study"

    gprMax.run(
        inputfile=inputfile,
        outputfile=output,
        hide_progress_bars=True,
        log_level=30,
    )

    with h5py.File(tmp_path / "hash_study1.h5") as active, h5py.File(
        tmp_path / "hash_study2.h5"
    ) as inactive:
        assert active["study"].attrs["CaseID"] == "on"
        assert inactive["study"].attrs["CaseID"] == "off"
        assert np.max(np.abs(active["srcs/src1/excitation/samples"][:])) > 0.9
        assert np.all(inactive["srcs/src1/excitation/samples"][:] == 0)
        assert inactive["study/source"][()].decode() == csvfile.read_text()
        resolved = json.loads(inactive["study/resolved_case"][()].decode())
        assert not resolved["objects"]["hertzian_dipole_1"]["active"]


@pytest.mark.integration
def test_magnetic_dipole_study_scales_whole_step_excitation(tmp_path):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=4e-10))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="wave"))
    source = gprMax.MagneticDipole(polarisation="z", p1=(0.01, 0.01, 0.01), waveform_id="wave")
    scene.add(source)
    scene.add(gprMax.Rx(p1=(0.015, 0.01, 0.01)))
    study = gprMax.GPRStudy(
        [
            gprMax.StudyCase("one", [gprMax.ObjectState(source, scale=1.0)]),
            gprMax.StudyCase("quarter", [gprMax.ObjectState(source, scale=0.25)]),
        ]
    )

    gprMax.run(
        scenes=[scene],
        study=study,
        outputfile=tmp_path / "magnetic",
        hide_progress_bars=True,
        log_level=30,
    )

    with h5py.File(tmp_path / "magnetic1.h5") as first, h5py.File(
        tmp_path / "magnetic2.h5"
    ) as second:
        excitation1 = first["srcs/src1/excitation/samples"][:]
        excitation2 = second["srcs/src1/excitation/samples"][:]
        assert np.max(np.abs(excitation1)) > 0.9
        assert np.allclose(excitation2, 0.25 * excitation1)
        assert second["srcs/src1"].attrs["StudyID"] == "magnetic_dipole_1"


def test_study_csv_rejects_partial_position_and_duplicate_object(tmp_path):
    partial = tmp_path / "partial.csv"
    partial.write_text("case_id,object_id,x_m,y_m,z_m\none,rx_1,0.1,,0.2\n")
    with pytest.raises(ValueError, match="provided together"):
        Study.from_csv("gpr", partial)

    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(
        "case_id,object_id,scale\none,hertzian_dipole_1,1\none,hertzian_dipole_1,0.5\n"
    )
    with pytest.raises(ValueError, match="more than once"):
        Study.from_csv("gpr", duplicate)
