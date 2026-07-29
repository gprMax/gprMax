"""Build-time validation for the initial single-edge RxPort interface."""

import pytest

import gprMax


def _base_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.02, 0.02, 0.02)))
    scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
    scene.add(gprMax.TimeWindow(time=2e-11))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    return scene


def _run_geometry(scene, tmp_path, name, **kwargs):
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=tmp_path / name,
        geometry_only=True,
        hide_progress_bars=True,
        **kwargs,
    )


def test_port_requires_exactly_one_colocated_voltage_source(tmp_path):
    scene = _base_scene()
    scene.add(gprMax.RxPort((0.01, 0.01, 0.01), id="feed"))

    with pytest.raises(ValueError, match="exactly one voltage source.*found 0"):
        _run_geometry(scene, tmp_path, "no_source")


def test_port_rejects_hard_voltage_source(tmp_path):
    scene = _base_scene()
    scene.add(gprMax.VoltageSource((0.01, 0.01, 0.01), "z", 0, "pulse"))
    scene.add(gprMax.RxPort((0.01, 0.01, 0.01), id="feed"))

    with pytest.raises(ValueError, match="non-zero voltage-source resistance"):
        _run_geometry(scene, tmp_path, "hard_source")


def test_port_rejects_ambiguous_colocated_sources(tmp_path):
    scene = _base_scene()
    scene.add(gprMax.VoltageSource((0.01, 0.01, 0.01), "z", 50, "pulse"))
    scene.add(gprMax.VoltageSource((0.01, 0.01, 0.01), "x", 50, "pulse"))
    scene.add(gprMax.RxPort((0.01, 0.01, 0.01), id="feed"))

    with pytest.raises(ValueError, match="exactly one voltage source.*found 2"):
        _run_geometry(scene, tmp_path, "ambiguous_source")


def test_port_rejects_geometry_fixed_even_for_first_run(tmp_path):
    scene = _base_scene()
    scene.add(gprMax.VoltageSource((0.01, 0.01, 0.01), "z", 50, "pulse"))
    scene.add(gprMax.RxPort((0.01, 0.01, 0.01), id="feed"))

    with pytest.raises(ValueError, match="does not support geometry-fixed"):
        _run_geometry(scene, tmp_path, "geometry_fixed", geometry_fixed=True)


def test_port_rejects_voltage_source_on_pec_edge(tmp_path):
    scene = _base_scene()
    scene.add(
        gprMax.Box(
            p1=(0.008, 0.008, 0.008),
            p2=(0.012, 0.012, 0.012),
            material_id="pec",
        )
    )
    scene.add(gprMax.VoltageSource((0.01, 0.01, 0.01), "z", 50, "pulse"))
    scene.add(gprMax.RxPort((0.01, 0.01, 0.01), id="feed"))

    with pytest.raises(ValueError, match="voltage source on a PEC edge"):
        _run_geometry(scene, tmp_path, "pec_source")


def test_port_rejects_second_monitor_on_same_source(tmp_path):
    scene = _base_scene()
    scene.add(gprMax.VoltageSource((0.01, 0.01, 0.01), "z", 50, "pulse"))
    scene.add(gprMax.RxPort((0.01, 0.01, 0.01), id="feed1"))
    scene.add(gprMax.RxPort((0.01, 0.01, 0.01), id="feed2"))

    with pytest.raises(ValueError, match="source already has an RxPort output"):
        _run_geometry(scene, tmp_path, "duplicate_source")


def test_port_rejects_dispersive_material_on_source_edge(tmp_path):
    scene = _base_scene()
    scene.add(gprMax.Material(er=4, se=0, mr=1, sm=0, id="dispersive"))
    scene.add(
        gprMax.AddDebyeDispersion(
            poles=1,
            er_delta=(3,),
            tau=(1e-11,),
            material_ids=("dispersive",),
        )
    )
    scene.add(
        gprMax.Box(
            p1=(0.008, 0.008, 0.008),
            p2=(0.012, 0.012, 0.012),
            material_id="dispersive",
        )
    )
    scene.add(gprMax.VoltageSource((0.01, 0.01, 0.01), "z", 50, "pulse"))
    scene.add(gprMax.RxPort((0.01, 0.01, 0.01), id="feed"))

    with pytest.raises(ValueError, match="dispersive material"):
        _run_geometry(scene, tmp_path, "dispersive_source")
