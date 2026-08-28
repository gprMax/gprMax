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

"""Build-time validation for source-owned voltage-port outputs."""

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


def test_hard_voltage_source_defaults_to_50_ohm_reference_impedance(tmp_path):
    scene = _base_scene()
    source = gprMax.VoltageSource((0.01, 0.01, 0.01), "z", 0, "pulse")
    scene.add(source)

    _run_geometry(scene, tmp_path, "hard_source")
    assert source._source.reference_impedance == 50
    assert source._monitor.output_id == "port1"


def test_uppercase_hard_source_polarisation_still_builds(tmp_path):
    scene = _base_scene()
    source = gprMax.VoltageSource((0.01, 0.01, 0.01), "Z", 0, "pulse")
    scene.add(source)

    _run_geometry(scene, tmp_path, "uppercase_hard_source")

    assert source._source.polarisation == "z"
    assert source._monitor.output_id == "port1"


def test_finite_resistance_source_at_domain_minimum_retains_port(tmp_path):
    scene = _base_scene()
    source = gprMax.VoltageSource((0.0, 0.0, 0.01), "z", 50, "pulse")
    scene.add(source)

    _run_geometry(scene, tmp_path, "finite_boundary_source")

    assert source._source is not None
    assert source._monitor.output_id == "port1"


@pytest.mark.parametrize(
    ("polarisation", "point"),
    (
        ("x", (0.01, 0.0, 0.0)),
        ("y", (0.0, 0.01, 0.0)),
        ("z", (0.0, 0.0, 0.01)),
    ),
)
def test_hard_source_at_domain_minimum_remains_valid_without_port(
    tmp_path, capsys, polarisation, point
):
    scene = _base_scene()
    source = gprMax.VoltageSource(point, polarisation, 0, "pulse")
    scene.add(source)

    _run_geometry(scene, tmp_path, f"hard_{polarisation}_boundary")

    assert source._source is not None
    assert source._monitor is None
    assert source._source.port_id is None
    assert "automatic port output is disabled" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="port result is not available"):
        _ = source.result


def test_boundary_hard_source_does_not_consume_an_automatic_port_id(tmp_path):
    scene = _base_scene()
    boundary = gprMax.VoltageSource((0.0, 0.0, 0.01), "z", 0, "pulse")
    monitored = gprMax.VoltageSource((0.01, 0.01, 0.01), "z", 50, "pulse")
    scene.add(boundary)
    scene.add(monitored)

    _run_geometry(scene, tmp_path, "boundary_then_monitored")

    assert boundary._monitor is None
    assert monitored._monitor.output_id == "port1"


def test_port_rejects_reference_impedance_different_from_finite_resistance(tmp_path):
    scene = _base_scene()
    scene.add(
        gprMax.VoltageSource(
            (0.01, 0.01, 0.01),
            "z",
            50,
            "pulse",
            reference_impedance=75,
        )
    )
    with pytest.raises(ValueError, match="only valid for a zero-resistance hard source"):
        _run_geometry(scene, tmp_path, "finite_reference_mismatch")


def test_colocated_sources_get_independent_automatic_ports(tmp_path):
    scene = _base_scene()
    first = gprMax.VoltageSource((0.01, 0.01, 0.01), "z", 50, "pulse")
    second = gprMax.VoltageSource((0.01, 0.01, 0.01), "x", 50, "pulse")
    scene.add(first)
    scene.add(second)

    _run_geometry(scene, tmp_path, "colocated_sources")
    assert first._monitor.output_id == "port1"
    assert second._monitor.output_id == "port2"


def test_port_supports_geometry_fixed_first_run(tmp_path):
    scene = _base_scene()
    scene.add(gprMax.VoltageSource((0.01, 0.01, 0.01), "z", 50, "pulse"))

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

    with pytest.raises(ValueError, match="voltage source on a PEC edge"):
        _run_geometry(scene, tmp_path, "pec_source")


def test_port_rejects_duplicate_explicit_ids(tmp_path):
    scene = _base_scene()
    scene.add(gprMax.VoltageSource((0.008, 0.01, 0.01), "z", 50, "pulse", id="feed"))
    scene.add(gprMax.VoltageSource((0.012, 0.01, 0.01), "z", 50, "pulse", id="feed"))

    with pytest.raises(ValueError, match="port output ID 'feed' is already in use"):
        _run_geometry(scene, tmp_path, "duplicate_source")


def test_finite_resistance_port_accepts_dispersive_material_on_source_edge(tmp_path):
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
    source = gprMax.VoltageSource((0.01, 0.01, 0.01), "z", 50, "pulse")
    scene.add(source)

    _run_geometry(scene, tmp_path, "dispersive_source")
    assert source._monitor.background_is_dispersive
