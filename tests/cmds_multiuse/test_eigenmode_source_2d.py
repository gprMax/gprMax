import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax
import gprMax.config as config

INF = float("inf")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _scene(mode):
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.06, 0.05, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=0.12e-9))
    scene.add(gprMax.Waveform(wave_type="contsine", amp=1, freq=5e9, id="eig_pulse"))
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.06, 0.005, INF), material_id="pec"))
    scene.add(gprMax.Box(p1=(0, 0.045, 0), p2=(0.06, 0.05, INF), material_id="pec"))
    scene.add(
        gprMax.EigenmodeSource(
            normal="x",
            direction="+",
            p1=(0.005, 0),
            p2=(0.045, INF),
            w=0.015,
            mode_index=0,
            frequency=5e9,
            waveform_id="eig_pulse",
        )
    )
    scene.add(gprMax.Rx(p1=(0.035, 0.025, INF)))
    return scene


def _dielectric_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.08, 0.08, INF)))
    scene.add(gprMax.PMLThickness(thickness=(5, 5, 0, 5, 5, 0)))
    scene.add(gprMax.TimeWindow(time=0.14e-9))
    scene.add(gprMax.Material(er=9, se=0, mr=1, sm=0, id="slab_core"))
    scene.add(gprMax.Waveform(wave_type="contsine", amp=1, freq=5e9, id="eig_pulse"))
    scene.add(
        gprMax.Box(
            p1=(0, 0.03, 0),
            p2=(0.08, 0.05, INF),
            material_id="slab_core",
        )
    )
    scene.add(
        gprMax.EigenmodeSource(
            normal="x",
            direction="+",
            p1=(0.005, 0),
            p2=(0.075, INF),
            w=0.015,
            mode_index=0,
            frequency=5e9,
            waveform_id="eig_pulse",
        )
    )
    scene.add(gprMax.Rx(p1=(0.05, 0.04, INF)))
    return scene


def _pmc_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TE"))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=(0.06, 0.05, INF)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=0.12e-9))
    scene.add(gprMax.Waveform(wave_type="contsine", amp=1, freq=5e9, id="eig_pulse"))
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.06, 0.005, INF), material_id="pmc"))
    scene.add(gprMax.Box(p1=(0, 0.045, 0), p2=(0.06, 0.05, INF), material_id="pmc"))
    scene.add(
        gprMax.EigenmodeSource(
            normal="x",
            direction="+",
            p1=(0.005, 0),
            p2=(0.045, INF),
            w=0.015,
            mode_index=0,
            frequency=5e9,
            waveform_id="eig_pulse",
        )
    )
    scene.add(gprMax.Rx(p1=(0.035, 0.025, INF)))
    return scene


@pytest.mark.parametrize(
    ("mode", "live_components", "dead_components"),
    [
        ("TM", ("Ez", "Hx", "Hy"), ("Ex", "Ey", "Hz")),
        ("TE", ("Ey", "Hz"), ("Ez", "Hx", "Hy")),
    ],
)
def test_2d_eigenmode_injection_updates_only_live_system(tmp_path, mode, live_components, dead_components):
    output = tmp_path / f"eigenmode_{mode.lower()}"
    gprMax.run(
        scenes=[_scene(mode)],
        n=1,
        outputfile=output,
        hide_progress_bars=True,
    )

    with h5py.File(output.with_suffix(".h5"), "r") as handle:
        receiver = handle["rxs/rx1"]
        for component in live_components:
            assert np.max(np.abs(receiver[component][...])) > 0
        for component in dead_components:
            assert np.max(np.abs(receiver[component][...])) == 0


def test_2d_dielectric_mode_decays_before_source_boundary(tmp_path):
    output = tmp_path / "eigenmode_dielectric"
    scene = _dielectric_scene()
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output,
        hide_progress_bars=True,
    )

    with h5py.File(output.with_suffix(".h5"), "r") as handle:
        assert np.max(np.abs(handle["rxs/rx1/Ez"][...])) > 0


def test_2d_pmc_mode_enforces_magnetic_wall_and_injects(tmp_path):
    output = tmp_path / "eigenmode_pmc"
    scene = _pmc_scene()
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output,
        hide_progress_bars=True,
    )

    with h5py.File(output.with_suffix(".h5"), "r") as handle:
        assert np.max(np.abs(handle["rxs/rx1/Hz"][...])) > 0


def test_eigenmode_source_rejected_with_mpi(monkeypatch):
    # build() checks MPI before touching grid at all, so a bare object
    # stands in for the grid here - mirrors the fast, direct-build style
    # used for #magnetic_frill_source's own MPI-rejection test.
    monkeypatch.setattr(config, "sim_config", type("_SC", (), {})())
    config.sim_config.general = {"solver": "cpu"}
    config.sim_config.mpi = True
    source = gprMax.EigenmodeSource(
        normal="x",
        direction="+",
        p1=(0.005, 0),
        p2=(0.045, INF),
        w=0.015,
        mode_index=0,
        frequency=5e9,
        waveform_id="eig_pulse",
    )
    with pytest.raises(ValueError, match="MPI"):
        source.build(grid=None)


def test_2d_eigenmode_normal_cannot_be_invariant_axis(tmp_path):
    scene = _scene("TM")
    scene.grid_objects = [obj for obj in scene.grid_objects if not isinstance(obj, gprMax.EigenmodeSource)]
    scene.add(
        gprMax.EigenmodeSource(
            normal="z",
            direction="+",
            p1=(0.005, 0.005),
            p2=(0.045, 0.045),
            w=INF,
            mode_index=0,
            frequency=5e9,
            waveform_id="eig_pulse",
        )
    )
    with pytest.raises(ValueError):
        gprMax.run(
            scenes=[scene],
            n=1,
            geometry_only=True,
            outputfile=tmp_path / "bad_normal",
            hide_progress_bars=True,
        )


@pytest.mark.parametrize("mode", ["TM", "TE"])
@pytest.mark.parametrize("invariant_axis", [0, 1, 2])
def test_2d_eigenmode_builds_for_every_invariant_axis(tmp_path, mode, invariant_axis):
    letters = "xyz"
    normal_axis = (invariant_axis + 1) % 3
    transverse_axis = (invariant_axis + 2) % 3
    domain = [0.04, 0.04, 0.04]
    domain[invariant_axis] = INF

    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Discretisation(p1=(1e-3, 1e-3, 1e-3)))
    scene.add(gprMax.Domain(p1=tuple(domain)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=2e-12))
    scene.add(gprMax.Waveform(wave_type="contsine", amp=1, freq=10e9, id="w"))

    lower_wall_end = [0.04, 0.04, 0.04]
    lower_wall_end[invariant_axis] = INF
    lower_wall_end[transverse_axis] = 0.005
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=tuple(lower_wall_end), material_id="pec"))
    upper_wall_start = [0, 0, 0]
    upper_wall_end = [0.04, 0.04, 0.04]
    upper_wall_start[transverse_axis] = 0.035
    upper_wall_end[invariant_axis] = INF
    scene.add(
        gprMax.Box(
            p1=tuple(upper_wall_start),
            p2=tuple(upper_wall_end),
            material_id="pec",
        )
    )

    full_lower = [0, 0, 0]
    full_upper = [0, 0, 0]
    full_lower[normal_axis] = full_upper[normal_axis] = 0.01
    full_lower[transverse_axis] = 0.005
    full_upper[transverse_axis] = 0.035
    full_lower[invariant_axis] = 0
    full_upper[invariant_axis] = INF
    source_transverse_axes = [axis for axis in range(3) if axis != normal_axis]
    scene.add(
        gprMax.EigenmodeSource(
            normal=letters[normal_axis],
            direction="+",
            p1=tuple(full_lower[axis] for axis in source_transverse_axes),
            p2=tuple(full_upper[axis] for axis in source_transverse_axes),
            w=0.01,
            mode_index=0,
            frequency=10e9,
            waveform_id="w",
        )
    )

    gprMax.run(
        scenes=[scene],
        n=1,
        geometry_only=True,
        outputfile=tmp_path / f"{mode}_{letters[invariant_axis]}",
        hide_progress_bars=True,
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("2d_tm/pec_waveguide/pec_waveguide.in"),
        Path("2d_te/pec_waveguide/pec_waveguide.in"),
        Path("2d_tm/dielectric_slab/dielectric_slab.in"),
        Path("2d_te/pmc_waveguide/pmc_waveguide.in"),
        Path("2d_tm/dielectric_bend/dielectric_bend.in"),
        Path("2d_te/dielectric_bend/dielectric_bend.in"),
    ],
)
def test_2d_example_builds_with_eight_timestamps(tmp_path, relative_path):
    source = REPOSITORY_ROOT / "eigensource_test_run" / relative_path
    copied_input = tmp_path / source.name
    shutil.copyfile(source, copied_input)

    assert copied_input.read_text().count("#snapshot:") == 8
    gprMax.run(
        inputfile=copied_input,
        n=1,
        geometry_only=True,
        outputfile=tmp_path / source.stem,
        hide_progress_bars=True,
    )
    assert len(list(tmp_path.glob(f"{source.stem}_eigenmode_*_fields.png"))) == 1
