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

"""End-to-end source geometry metadata written by real user objects."""

from pathlib import Path

import h5py
import pytest

import gprMax


INF = float("inf")


def _strings(dataset):
    return [value.decode() if isinstance(value, bytes) else str(value) for value in dataset[...]]


def _geometry(filename: Path, role: str):
    with h5py.File(filename, "r") as output:
        fields = output["VTKHDF/FieldData"]
        return {
            "ids": _strings(fields[f"{role}_geometry_ids"]),
            "types": _strings(fields[f"{role}_geometry_types"]),
            "kinds": _strings(fields[f"{role}_geometry_kinds"]),
            "bounds": fields[f"{role}_geometry_bounds"][...],
        }


@pytest.mark.integration
def test_real_2d_tfsf_source_writes_a_rectangle(tmp_path):
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
    scene.add(gprMax.Domain(p1=(0.04, 0.04, INF)))
    scene.add(gprMax.TimeWindow(iterations=20))
    scene.add(gprMax.PMLThickness(thickness=(3, 3, 0, 3, 3, 0)))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=8e9, id="pulse"))
    scene.add(
        gprMax.DiscretePlaneWaveAxial(
            p1=(0.01, 0.01, INF),
            p2=(0.03, 0.03, INF),
            axis="x",
            psi=90,
            waveform_id="pulse",
        )
    )
    scene.add(
        gprMax.GeometryView(
            p1=(0, 0, 0),
            p2=(0.04, 0.04, INF),
            dl=(0.002, 0.002, 0.002),
            output_type="n",
            filename="tfsf_geometry",
        )
    )

    gprMax.run(
        scenes=[scene],
        outputfile=tmp_path / "tfsf",
        geometry_only=True,
        hide_progress_bars=True,
    )
    geometry = _geometry(tmp_path / "tfsf_geometry.vtkhdf", "source")

    assert geometry["ids"] == ["plane_wave_1"]
    assert geometry["types"] == ["DiscretePlaneWave"]
    assert geometry["kinds"] == ["rectangle"]
    assert geometry["bounds"][0] == pytest.approx((0.01, 0.03, 0.01, 0.03, 0, 0))


@pytest.mark.integration
def test_real_eigenmode_source_writes_its_port_plane(tmp_path):
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
    scene.add(gprMax.Domain(p1=(0.08, 0.08, INF)))
    scene.add(gprMax.TimeWindow(iterations=300))
    scene.add(gprMax.PMLThickness(thickness=(3, 3, 0, 3, 3, 0)))
    scene.add(gprMax.Material(er=9, se=0, mr=1, sm=0, id="slab_core"))
    scene.add(gprMax.Box(p1=(0, 0.03, 0), p2=(0.08, 0.05, INF), material_id="slab_core"))
    scene.add(gprMax.EigenmodeBand(id="band", fmin=4e9, fmax=6e9, points=3))
    scene.add(
        gprMax.EigenmodePort(
            port=1,
            p1=(0.02, 0.006, 0),
            p2=(0.02, 0.074, INF),
            direction="+",
            modes=(1,),
            anchors=(5e9,),
            plot_fields=False,
        )
    )
    scene.add(
        gprMax.EigenmodePort(
            port=2,
            p1=(0.06, 0.006, 0),
            p2=(0.06, 0.074, INF),
            direction="-",
            modes=(1,),
            anchors=(5e9,),
            plot_fields=False,
        )
    )
    scene.add(
        gprMax.EigenmodeExcitation(
            port=1,
            mode=1,
            waveform="auto",
            plot_waveform=False,
        )
    )
    scene.add(
        gprMax.GeometryView(
            p1=(0, 0, 0),
            p2=(0.08, 0.08, INF),
            dl=(0.002, 0.002, 0.002),
            output_type="n",
            filename="port_geometry",
        )
    )

    gprMax.run(
        scenes=[scene],
        outputfile=tmp_path / "port",
        geometry_only=True,
        hide_progress_bars=True,
    )
    geometry = _geometry(tmp_path / "port_geometry.vtkhdf", "source")
    receivers = _geometry(tmp_path / "port_geometry.vtkhdf", "receiver")

    assert geometry["ids"] == ["port1"]
    assert geometry["types"] == ["EigenmodePort"]
    assert geometry["kinds"] == ["plane"]
    assert geometry["bounds"][0] == pytest.approx((0.02, 0.02, 0.006, 0.074, 0, 0.002))
    assert receivers["ids"] == ["port2"]
    assert receivers["types"] == ["EigenmodePort"]
    assert receivers["kinds"] == ["plane"]
    assert receivers["bounds"][0] == pytest.approx((0.06, 0.06, 0.006, 0.074, 0, 0.002))


@pytest.mark.integration
def test_real_zero_amplitude_voltage_port_is_a_receiver(tmp_path):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
    scene.add(gprMax.Domain(p1=(0.024, 0.024, 0.024)))
    scene.add(gprMax.TimeWindow(iterations=20))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=8e9, id="drive"))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=0, freq=8e9, id="silent"))
    scene.add(
        gprMax.VoltageSource(
            p1=(0.008, 0.01, 0.01),
            polarisation="z",
            resistance=50,
            waveform_id="drive",
            id="tx",
        )
    )
    scene.add(
        gprMax.VoltageSource(
            p1=(0.016, 0.01, 0.01),
            polarisation="z",
            resistance=50,
            waveform_id="silent",
            id="rx_port",
        )
    )
    scene.add(gprMax.Rx(p1=(0.012, 0.014, 0.01), id="field_probe"))
    scene.add(
        gprMax.GeometryView(
            p1=(0, 0, 0),
            p2=(0.024, 0.024, 0.024),
            dl=(0.002, 0.002, 0.002),
            output_type="n",
            filename="voltage_ports_geometry",
        )
    )

    gprMax.run(
        scenes=[scene],
        outputfile=tmp_path / "voltage_ports",
        geometry_only=True,
        hide_progress_bars=True,
    )
    filename = tmp_path / "voltage_ports_geometry.vtkhdf"
    sources = _geometry(filename, "source")
    receivers = _geometry(filename, "receiver")

    assert sources["ids"] == ["tx"]
    assert sources["types"] == ["VoltageSourcePort"]
    assert receivers["ids"] == ["field_probe", "rx_port"]
    assert receivers["types"] == ["Rx", "VoltageSourcePort"]
    with h5py.File(filename, "r") as output:
        fields = output["VTKHDF/FieldData"]
        assert _strings(fields["receiver_ids"]) == ["field_probe"]


@pytest.mark.integration
def test_real_passive_virtual_waveguide_is_a_receiver_interface(tmp_path):
    dl = 0.002
    domain = (0.06, 0.02, 0.024)
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=domain))
    scene.add(gprMax.TimeWindow(iterations=100))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.06, 0.002, 0.024), material_id="pec"))
    scene.add(gprMax.Box(p1=(0, 0.018, 0), p2=(0.06, 0.02, 0.024), material_id="pec"))
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.06, 0.02, 0.002), material_id="pec"))
    scene.add(gprMax.Box(p1=(0, 0, 0.022), p2=domain, material_id="pec"))
    scene.add(gprMax.EigenmodeBand(id="band", fmin=10e9, fmax=10e9, points=1))
    scene.add(
        gprMax.EigenmodePort(
            port=1,
            p1=(0.03, 0.002, 0.002),
            p2=(0.03, 0.018, 0.022),
            direction="+",
            modes=(1,),
            anchors=(10e9,),
            plot_fields=False,
        )
    )
    scene.add(
        gprMax.VirtualWaveguide(
            port=1,
            length_cells=10,
            pml_cells=4,
            source_clearance_cells=2,
        )
    )
    scene.add(
        gprMax.GeometryView(
            p1=(0, 0, 0),
            p2=domain,
            dl=(dl, dl, dl),
            output_type="n",
            filename="virtual_interface_geometry",
        )
    )

    gprMax.run(
        scenes=[scene],
        outputfile=tmp_path / "virtual_interface",
        geometry_only=True,
        hide_progress_bars=True,
    )
    filename = tmp_path / "virtual_interface_geometry.vtkhdf"
    receivers = _geometry(filename, "receiver")

    assert receivers["ids"] == ["port1"]
    assert receivers["types"] == ["VirtualWaveguideInterface"]
    assert receivers["kinds"] == ["plane"]
    assert receivers["bounds"][0] == pytest.approx(
        (0.03, 0.03, 0.002, 0.018, 0.002, 0.022)
    )
    with h5py.File(filename, "r") as output:
        fields = output["VTKHDF/FieldData"]
        assert "source_geometry_schema_version" in fields
        assert "source_geometry_ids" not in fields
