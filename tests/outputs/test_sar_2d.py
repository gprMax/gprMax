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

"""TM/TE live-plane and dimensional semantics for SAR outputs."""

from itertools import product

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.sar import EDGE_OFFSETS, edge_offsets_for_mode

INF = float("inf")


@pytest.mark.parametrize(
    "mode,active",
    (
        ("2D TMx", ("Ex",)),
        ("2D TMy", ("Ey",)),
        ("2D TMz", ("Ez",)),
        ("2D TEx", ("Ey", "Ez")),
        ("2D TEy", ("Ex", "Ez")),
        ("2D TEz", ("Ex", "Ey")),
    ),
)
def test_sar_edge_offsets_preserve_yee_geometry(mode, active):
    offsets = edge_offsets_for_mode(mode)
    axis = "xyz".index(mode[-1])

    assert tuple(offsets) == active
    for component, component_offsets in offsets.items():
        assert component_offsets.shape == (4, 3)
        if "TM" in mode:
            np.testing.assert_array_equal(component_offsets, EDGE_OFFSETS[component])
        else:
            assert np.all(component_offsets[:, axis] == 0)
            assert np.unique(component_offsets, axis=0).shape[0] == 2


def _scene(mode, axis):
    dl = 0.002
    dimensions = [0.016, 0.018, 0.020]
    dimensions[axis] = INF
    lower = [0.004, 0.004, 0.004]
    upper = [0.012, 0.014, 0.016]
    lower[axis] = upper[axis] = INF
    source = [0.008, 0.008, 0.010]
    source[axis] = INF
    family = mode[:2]
    active = tuple(edge_offsets_for_mode(f"2D {mode}{'xyz'[axis]}"))
    polarisation = active[0][-1].lower()

    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.DomainMode(mode=family))
    scene.add(gprMax.Domain(p1=tuple(dimensions)))
    scene.add(gprMax.TimeWindow(time=0.6e-9))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.Material(er=4, se=0.5, mr=1, sm=0, id="tissue"))
    scene.add(gprMax.MaterialDensity(density=1000, material_ids="tissue"))
    scene.add(
        gprMax.Box(
            p1=tuple(lower),
            p2=tuple(upper),
            material_id="tissue",
            tag="target",
        )
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1e9, id="pulse"))
    scene.add(
        gprMax.VoltageSource(
            p1=tuple(source),
            polarisation=polarisation,
            resistance=50,
            waveform_id="pulse",
        )
    )
    output = gprMax.SAR(
        frequencies=(1e9,),
        waveform_id="pulse",
        tags="target",
        id="target_sar",
        spectrum_limit="nyquist",
    )
    scene.add(output)
    return scene, output, active


@pytest.mark.integration
@pytest.mark.parametrize("family,axis", tuple(product(("TM", "TE"), range(3))))
def test_sar_2d_samples_only_live_plane_and_active_fields(tmp_path, family, axis):
    scene, output, active = _scene(family, axis)
    mode = f"2D {family}{'xyz'[axis]}"
    filename = tmp_path / f"sar_{family.lower()}{'xyz'[axis]}"

    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=filename,
        hide_progress_bars=True,
        cpu_precision="double",
    )

    live_index = 0 if family == "TM" else 1
    assert np.all(output.result.cell_indices[:, axis] == live_index)
    assert tuple(output._monitor.accumulators) == active
    assert np.all(np.isfinite(output.result.sar))
    assert np.all(output.result.sar >= 0)

    with h5py.File(str(filename) + ".h5", "r") as data:
        group = data["sar/target_sar"]
        assert group.attrs["Dimensionality"] == 2
        assert group.attrs["ModelMode"] == mode
        assert group.attrs["InvariantAxis"] == "xyz"[axis]
        assert group.attrs["LiveInvariantIndex"] == live_index
        assert group.attrs["IntegrationMeasure"] == "per_unit_invariant_length"
        assert tuple(item.decode() for item in group.attrs["ActiveElectricComponents"]) == active
        summary = group["tags/target"]
        assert summary.attrs["MassPerLengthUnits"] == "kg/m"
        assert "Mass" not in summary.attrs
        assert "absorbed_power_per_length" in summary
        assert "absorbed_power" not in summary


@pytest.mark.parametrize("family", ("TM", "TE"))
def test_sar_2d_rejects_3d_mass_averaging(tmp_path, family):
    scene, output, _ = _scene(family, 2)
    output.averaging_masses = (0.001,)

    with pytest.raises(ValueError, match="spatial mass averaging is not yet available in 2-D"):
        gprMax.run(
            scenes=[scene],
            n=1,
            outputfile=tmp_path / f"sar_{family.lower()}_mass",
            hide_progress_bars=True,
        )


@pytest.mark.integration
@pytest.mark.parametrize("family", ("TM", "TE"))
def test_radiometry_2d_uses_per_unit_length_without_density(tmp_path, family):
    scene, sar_output, _ = _scene(family, 2)
    scene.output_objects.remove(sar_output)
    scene.grid_objects = [
        item for item in scene.grid_objects if not isinstance(item, gprMax.MaterialDensity)
    ]
    output = gprMax.Radiometry(
        frequencies=(1e9,),
        waveform_id="pulse",
        tags="target",
        id="target_radiometry",
        spectrum_limit="nyquist",
    )
    scene.add(output)
    filename = tmp_path / f"radiometry_{family.lower()}z"

    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=filename,
        hide_progress_bars=True,
        cpu_precision="double",
    )

    assert np.all(output.result.valid)
    with h5py.File(str(filename) + ".h5", "r") as data:
        group = data["radiometry/target_radiometry"]
        assert group.attrs["Dimensionality"] == 2
        assert group.attrs["IntegrationMeasure"] == "per_unit_invariant_length"
        assert "absorbed_power_per_length" in group["tags/target"]
        assert "density" not in group


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("family", ("TM", "TE"))
def test_cuda_sar_2d_matches_cpu(tmp_path, request, family):
    cpu_scene, _, _ = _scene(family, 2)
    cuda_scene, _, _ = _scene(family, 2)
    cpu_path = tmp_path / f"sar_{family.lower()}_cpu"
    cuda_path = tmp_path / f"sar_{family.lower()}_cuda"

    gprMax.run(
        scenes=[cpu_scene],
        n=1,
        outputfile=cpu_path,
        hide_progress_bars=True,
        cpu_precision="single",
    )
    gprMax.run(
        scenes=[cuda_scene],
        n=1,
        outputfile=cuda_path,
        hide_progress_bars=True,
        gpu=[request.getfixturevalue("gpu_device")],
        gpu_precision="single",
    )

    with h5py.File(str(cpu_path) + ".h5", "r") as cpu, h5py.File(
        str(cuda_path) + ".h5", "r"
    ) as cuda:
        cpu_group = cpu["sar/target_sar"]
        cuda_group = cuda["sar/target_sar"]
        assert cuda_group.attrs["CollectionBackend"] == "cuda_device"
        for dataset in ("absorbed_power_density", "sar"):
            reference = cpu_group[dataset][...]
            scale = max(float(np.nanmax(np.abs(reference), initial=0.0)), 1e-18)
            np.testing.assert_allclose(
                cuda_group[dataset][...],
                reference,
                rtol=5e-4,
                atol=5e-4 * scale,
                equal_nan=True,
            )
