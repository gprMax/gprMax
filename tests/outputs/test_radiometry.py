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


def _lossy_scene(*, plane_wave=False, conductivity=0.2):
    dl = 0.002
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.032, 0.024, 0.024)))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(time=2e-9))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.Material(er=2.5, se=conductivity, mr=1, sm=0, id="lossy"))
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


@pytest.mark.parametrize(
    "source_kind,normalisation,expected_units",
    (
        ("voltage", "waveform", "V"),
        ("current", "waveform", "A"),
        ("current", "current_moment", "A m"),
    ),
)
def test_radiometry_source_normalisation_unit_labels_match_saved_values(
    tmp_path, source_kind, normalisation, expected_units
):
    scene = _lossy_scene()
    if source_kind == "current":
        voltage_source = next(
            item for item in scene.grid_objects if isinstance(item, gprMax.VoltageSource)
        )
        scene.grid_objects.remove(voltage_source)
        scene.add(
            gprMax.HertzianDipole(p1=(0.012, 0.012, 0.012), polarisation="z", waveform_id="pulse")
        )
    target_amplitude = 0.01
    output = gprMax.Radiometry(
        frequencies=(1e9,),
        tags="target",
        waveform_id="pulse",
        id="normalised",
        normalisation=normalisation,
        target_amplitude=target_amplitude,
    )
    scene.add(output)
    filename = tmp_path / f"radiometry_{source_kind}_{normalisation}"
    gprMax.run(scenes=[scene], n=1, outputfile=filename, hide_progress_bars=True)

    result = output.result
    assert result.valid[0]
    np.testing.assert_array_equal(
        result.normalised_absorption_density,
        result.absorbed_power_density / target_amplitude**2,
    )
    with h5py.File(str(filename) + ".h5", "r") as data:
        group = data["radiometry/normalised"]
        # Unit metadata must describe the arrays actually written, without
        # applying another numerical scaling during serialization.
        np.testing.assert_array_equal(
            group["absorbed_power_density"][...], result.absorbed_power_density
        )
        np.testing.assert_array_equal(
            group["normalised_absorption_density"][...], result.normalised_absorption_density
        )
        assert group.attrs["TargetAmplitudeUnits"] == expected_units
        assert group.attrs["NormalisedAbsorptionDensityUnits"] == f"W/m3/({expected_units})2"
        assert group.attrs["IntegratedNormalisedAbsorptionUnits"] == f"W/({expected_units})2"
        assert group["tags/target"].attrs["NormalisedAbsorptionUnits"] == f"W/({expected_units})2"


@pytest.mark.parametrize("conductivity", (0.0, 0.2))
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_absorption_tag_summaries_preserve_invalid_frequencies_and_valid_zeros(
    tmp_path, conductivity
):
    scene = _lossy_scene(conductivity=conductivity)
    scene.add(gprMax.MaterialDensity(density=1000, material_ids="lossy"))
    # This tag exists but has no sampled physical cells because it lies in PML.
    scene.add(
        gprMax.Box(
            p1=(0.0, 0.0, 0.0),
            p2=(0.002, 0.002, 0.002),
            material_id="lossy",
            tag="empty",
        )
    )
    for output_class, output_id in ((gprMax.SAR, "sar"), (gprMax.Radiometry, "rad")):
        scene.add(
            output_class(
                frequencies=(1e9, 20e9),
                tags=("target", "empty"),
                waveform_id="pulse",
                id=output_id,
                spectrum_limit="nyquist",
            )
        )
    filename = tmp_path / "absorption_band_summaries"
    gprMax.run(scenes=[scene], n=1, outputfile=filename, hide_progress_bars=True)

    with h5py.File(str(filename) + ".h5", "r") as data:
        for path in ("sar/sar", "radiometry/rad"):
            group = data[path]
            np.testing.assert_array_equal(group["valid"][...], (1, 0))
            assert np.all(np.isnan(group["absorbed_power_density"][1]))
            for tag in ("target", "empty"):
                summary = group[f"tags/{tag}"]
                power = summary["absorbed_power"][...]
                assert np.isnan(power[1])
                if tag == "empty" or conductivity == 0:
                    assert power[0] == 0.0
                else:
                    assert power[0] > 0
                if path.startswith("sar/"):
                    average = summary["mass_average_sar"][...]
                    assert np.isnan(average[1])
                    if tag == "empty":
                        assert np.isnan(average[0])
                    if tag == "target" and conductivity == 0:
                        assert average[0] == 0.0
                else:
                    weighting = summary["normalised_absorption"][...]
                    assert np.isnan(weighting[1])
                    if tag == "empty" or conductivity == 0:
                        assert weighting[0] == 0.0


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


@pytest.mark.parametrize("window", ("rectangular", "hann", "hann_end"))
def test_plane_wave_incident_flux_normalisation_produces_cross_section_density(tmp_path, window):
    scene = _lossy_scene(plane_wave=True)
    output = gprMax.Radiometry(
        frequencies=(1e9,),
        tags="target",
        waveform_id="pulse",
        id="plane_wave",
        normalisation="incident_flux",
        target_flux=2.0,
        window=window,
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
