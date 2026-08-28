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

"""End-to-end KSIR, antenna, TFSF, and RCS coverage with HSG subgrids."""

import h5py
import numpy as np
import pytest

import gprMax


def _base_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.09, 0.09, 0.09)))
    scene.add(gprMax.Discretisation(p1=(0.003, 0.003, 0.003)))
    scene.add(gprMax.TimeWindow(time=1e-9))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.OMPThreads(1))
    subgrid = gprMax.SubGridHSG(
        p1=(0.03, 0.03, 0.03),
        p2=(0.06, 0.06, 0.06),
        ratio=3,
        id="fine_grid",
    )
    scene.add(subgrid)
    return scene, subgrid


def test_antenna_metrics_use_subgrid_port_and_time_step(tmp_path):
    scene, subgrid = _base_scene()
    source_position = (0.045, 0.045, 0.045)
    subgrid.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    subgrid.add(
        gprMax.VoltageSource(
            p1=source_position,
            polarisation="z",
            resistance=50,
            waveform_id="pulse",
            id="feed",
            spectrum_limit="nyquist",
        )
    )

    scene.add(
        gprMax.NTFFSurface(
            p1=(0.015, 0.015, 0.015),
            p2=(0.075, 0.075, 0.075),
            id="surface",
        )
    )
    scene.add(
        gprMax.KSIRFrequencyTransform(
            surface_id="surface",
            id="band",
            frequencies=(5e9,),
        )
    )
    scene.add(gprMax.KSIRAntennaPorts("band", ("fine_grid/feed",)))
    pattern = gprMax.KSIRFarField(
        theta=(90,),
        phi=(0,),
        transform_id="band",
        id="broadside",
        outputs=("gain", "realized_gain", "radiation_efficiency"),
    )
    scene.add(pattern)

    output = tmp_path / "subgrid_antenna"
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output,
        subgrid=True,
        autotranslate=True,
        hide_progress_bars=True,
        cpu_precision="double",
    )

    assert np.isfinite(pattern.result.fields["gain"]).all()
    with h5py.File(output.with_suffix(".h5"), "r") as handle:
        port_power = handle["ntff/surface/frequency/band/far_field/broadside/port_power"]
        assert port_power["port_ids"].asstr()[...].tolist() == ["fine_grid/feed"]
        assert port_power["source_types"].asstr()[...].tolist() == ["VoltageSource"]
        assert port_power["gain_valid"][0] == 1


def test_tfsf_and_rcs_include_scatterer_inside_subgrid(tmp_path):
    scene, subgrid = _base_scene()
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    scene.add(
        gprMax.DiscretePlaneWaveAxial(
            axis="x",
            psi=90,
            p1=(0.018, 0.018, 0.018),
            p2=(0.072, 0.072, 0.072),
            waveform_id="pulse",
        )
    )
    subgrid.add(
        gprMax.Sphere(
            p1=(0.045, 0.045, 0.045),
            r=0.004,
            material_id="pec",
        )
    )
    scene.add(
        gprMax.NTFFSurface(
            p1=(0.012, 0.012, 0.012),
            p2=(0.078, 0.078, 0.078),
            id="surface",
        )
    )
    scene.add(
        gprMax.KSIRFrequencyTransform(
            surface_id="surface",
            id="scattering",
            frequencies=(5e9,),
        )
    )
    backscatter = gprMax.KSIRFarField(
        theta=(90,),
        phi=(180,),
        transform_id="scattering",
        id="backscatter",
        outputs=("rcs",),
    )
    scene.add(backscatter)

    output = tmp_path / "subgrid_rcs"
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output,
        subgrid=True,
        autotranslate=True,
        hide_progress_bars=True,
        cpu_precision="double",
    )

    rcs = backscatter.result.fields["rcs"]
    assert rcs.shape == (1, 1)
    assert np.isfinite(rcs).all()
    assert np.all(rcs >= 0)
    assert np.max(rcs) > 0


def test_ntff_surface_cannot_cut_subgrid_coupling_region(tmp_path):
    scene, _ = _base_scene()
    scene.add(
        gprMax.NTFFSurface(
            p1=(0.024, 0.024, 0.024),
            p2=(0.066, 0.066, 0.066),
            id="cutting_surface",
        )
    )
    scene.add(
        gprMax.KSIRFrequencyTransform(
            surface_id="cutting_surface",
            id="band",
            frequencies=(5e9,),
        )
    )

    with pytest.raises(ValueError, match="outer coupling surface of subgrid 'fine_grid'"):
        gprMax.run(
            scenes=[scene],
            n=1,
            outputfile=tmp_path / "cutting_ksir",
            subgrid=True,
            autotranslate=True,
            geometry_only=True,
            hide_progress_bars=True,
        )


def test_tfsf_surface_cannot_cut_subgrid_coupling_region(tmp_path):
    scene, _ = _base_scene()
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    scene.add(
        gprMax.DiscretePlaneWaveAxial(
            axis="x",
            psi=90,
            p1=(0.018, 0.018, 0.018),
            p2=(0.060, 0.060, 0.060),
            waveform_id="pulse",
        )
    )

    with pytest.raises(ValueError, match="outer coupling surface of subgrid 'fine_grid'"):
        gprMax.run(
            scenes=[scene],
            n=1,
            outputfile=tmp_path / "cutting_tfsf",
            subgrid=True,
            autotranslate=True,
            geometry_only=True,
            hide_progress_bars=True,
        )


def test_tfsf_source_must_be_defined_on_main_grid(tmp_path):
    scene, subgrid = _base_scene()
    subgrid.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    subgrid.add(
        gprMax.DiscretePlaneWaveAxial(
            axis="x",
            psi=90,
            p1=(0.033, 0.033, 0.033),
            p2=(0.057, 0.057, 0.057),
            waveform_id="pulse",
        )
    )

    with pytest.raises(ValueError, match="must be defined on the main grid"):
        gprMax.run(
            scenes=[scene],
            n=1,
            outputfile=tmp_path / "subgrid_tfsf_source",
            subgrid=True,
            autotranslate=True,
            geometry_only=True,
            hide_progress_bars=True,
        )
