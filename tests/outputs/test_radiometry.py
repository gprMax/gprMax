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

"""Shared absorbed-power and density-independent radiometry output tests."""

import h5py
import numpy as np
import pytest

import gprMax
from gprMax import config
from gprMax.hash_cmds_file import get_user_objects
from gprMax.user_objects.cmds_output import Radiometry


def _lossy_scene(*, plane_wave=False):
    dl = 0.002
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.032, 0.024, 0.024)))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(time=2e-9))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.Material(er=2.5, se=0.2, mr=1, sm=0, id="lossy"))
    scene.add(
        gprMax.Box(
            p1=(0.016, 0.006, 0.006),
            p2=(0.024, 0.018, 0.018),
            material_id="lossy",
            tag="target",
        )
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1e9, id="pulse"))
    if plane_wave:
        scene.add(
            gprMax.DiscretePlaneWaveAxial(
                p1=(0.008, 0.004, 0.004),
                p2=(0.026, 0.020, 0.020),
                axis="x",
                psi=90,
                waveform_id="pulse",
            )
        )
    else:
        scene.add(
            gprMax.VoltageSource(
                p1=(0.012, 0.012, 0.012),
                polarisation="z",
                resistance=50,
                waveform_id="pulse",
            )
        )
    return scene


@pytest.mark.parametrize(
    ("command", "normalisation", "waveform_id", "target_power", "target_flux"),
    (
        ("#radiometry: 1e9 1e9 1 pulse 2 10 rad body\n", "waveform", "pulse", None, None),
        (
            "#radiometry: 1e9 1e9 1 pulse current_moment 0.2 10 rad body\n",
            "current_moment",
            "pulse",
            None,
            None,
        ),
        (
            "#radiometry: 1e9 1e9 1 pulse incident_flux 3 10 rad body\n",
            "incident_flux",
            "pulse",
            None,
            3.0,
        ),
        (
            "#radiometry: 1e9 1e9 1 accepted_power 4 feed 10 rad body\n",
            "accepted_power",
            None,
            4.0,
            None,
        ),
    ),
)
def test_radiometry_hash_normalisation_forms(
    command, normalisation, waveform_id, target_power, target_flux
):
    output = get_user_objects([command], checkessential=False)[0]

    assert isinstance(output, Radiometry)
    assert output.normalisation == normalisation
    assert output.waveform_id == waveform_id
    assert output.target_power == target_power
    assert output.target_flux == target_flux
    assert output.tags == ("body",)


def test_radiometry_does_not_require_material_density(tmp_path):
    scene = _lossy_scene()
    output = gprMax.Radiometry(
        frequencies=(1e9,),
        tags="target",
        waveform_id="pulse",
        id="local_source",
        target_amplitude=1.0,
    )
    scene.add(output)

    filename = tmp_path / "radiometry_no_density"
    gprMax.run(scenes=[scene], n=1, outputfile=filename, hide_progress_bars=True)

    assert output.result.valid[0]
    assert np.all(np.isfinite(output.result.absorbed_power_density))
    np.testing.assert_allclose(
        output.result.normalised_absorption_density,
        output.result.absorbed_power_density,
    )
    with h5py.File(str(filename) + ".h5", "r") as data:
        group = data["radiometry/local_source"]
        assert "density" not in group
        assert "sar" not in group
        assert group.attrs["NormalisedAbsorptionMeaning"] == (
            "absorbed power per squared source-native amplitude"
        )


def test_radiometry_port_weighting_is_independent_of_requested_power(tmp_path):
    scene = _lossy_scene()
    voltage_source = next(
        item for item in scene.grid_objects if isinstance(item, gprMax.VoltageSource)
    )
    voltage_source.id = "feed"
    one_watt = gprMax.Radiometry(
        frequencies=(1e9,),
        tags="target",
        id="one_watt",
        normalisation="incident_power",
        port_id="feed",
        target_power=1.0,
    )
    four_watt = gprMax.Radiometry(
        frequencies=(1e9,),
        tags="target",
        id="four_watt",
        normalisation="incident_power",
        port_id="feed",
        target_power=4.0,
    )
    scene.add(one_watt)
    scene.add(four_watt)

    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=tmp_path / "radiometry_port",
        hide_progress_bars=True,
    )

    np.testing.assert_allclose(
        four_watt.result.absorbed_power_density,
        4 * one_watt.result.absorbed_power_density,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        four_watt.result.normalised_absorption_density,
        one_watt.result.normalised_absorption_density,
        rtol=2e-6,
    )


def test_radiometry_state_is_reset_for_geometry_fixed_runs(tmp_path):
    scene = _lossy_scene()
    scene.add(
        gprMax.Radiometry(
            frequencies=(1e9,),
            tags="target",
            waveform_id="pulse",
            id="reused",
        )
    )
    filename = tmp_path / "radiometry_reused"

    gprMax.run(
        scenes=[scene],
        n=2,
        geometry_fixed=True,
        outputfile=filename,
        hide_progress_bars=True,
    )

    with h5py.File(f"{filename}1.h5", "r") as first, h5py.File(f"{filename}2.h5", "r") as second:
        np.testing.assert_array_equal(
            first["radiometry/reused/normalised_absorption_density"][...],
            second["radiometry/reused/normalised_absorption_density"][...],
        )


def test_plane_wave_incident_flux_normalisation_produces_cross_section_density(tmp_path):
    scene = _lossy_scene(plane_wave=True)
    output = gprMax.Radiometry(
        frequencies=(1e9,),
        tags="target",
        waveform_id="pulse",
        id="plane_wave",
        normalisation="incident_flux",
        target_flux=2.0,
    )
    scene.add(output)

    filename = tmp_path / "radiometry_plane_wave"
    gprMax.run(scenes=[scene], n=1, outputfile=filename, hide_progress_bars=True)

    assert output.result.valid[0]
    assert output.result.incident_flux[0] > 0
    free_space_impedance = np.sqrt(config.m0 / config.e0)
    expected_flux = 0.5 * np.abs(output.result.source_spectrum[0]) ** 2 / free_space_impedance
    assert output.result.incident_flux[0] == pytest.approx(expected_flux, rel=1e-12)
    scaled_flux = output.result.incident_flux[0] * np.abs(output.result.normalisation_scale[0]) ** 2
    assert scaled_flux == pytest.approx(2.0, rel=2e-6)
    np.testing.assert_allclose(
        output.result.normalised_absorption_density,
        output.result.absorbed_power_density / 2.0,
    )
    with h5py.File(str(filename) + ".h5", "r") as data:
        group = data["radiometry/plane_wave"]
        assert group.attrs["NormalisedAbsorptionDensityUnits"] == "1/m"
        assert group.attrs["IntegratedNormalisedAbsorptionUnits"] == "m2"
        assert np.all(np.isfinite(group["tags/target/normalised_absorption"][...]))
