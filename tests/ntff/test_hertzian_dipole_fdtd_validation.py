"""End-to-end KSIR validation against the Hertzian-dipole pattern."""

import h5py
import numpy as np
import pytest

import gprMax

pytestmark = pytest.mark.integration


def test_cpu_hertzian_dipole_pattern_matches_closed_form(tmp_path):
    dl = 0.002
    frequency = 5e9
    centre = (0.05, 0.05, 0.05)
    theta = np.arange(0.0, 181.0, 2.0)
    phi = np.arange(0.0, 361.0, 2.0)

    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl,) * 3))
    scene.add(gprMax.Domain(p1=(0.1,) * 3))
    scene.add(gprMax.TimeWindow(time=1e-9))
    scene.add(gprMax.OMPThreads(n=2))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=frequency, id="pulse"))
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=centre, waveform_id="pulse"))
    scene.add(
        gprMax.NTFFSurface(
            p1=(0.034,) * 3,
            p2=(0.066,) * 3,
            id="dipole_surface",
            origin=centre,
        )
    )
    scene.add(
        gprMax.KSIRFrequencyTransform(
            "dipole_surface",
            "dipole_spectrum",
            (frequency,),
            save_surface_dft=False,
        )
    )
    e_plane = gprMax.KSIRFarField(
        theta=theta,
        phi=np.zeros(theta.shape),
        transform_id="dipole_spectrum",
        id="e_plane",
        outputs=("Etheta",),
    )
    h_plane = gprMax.KSIRFarField(
        theta=np.full(phi.shape, 90.0),
        phi=phi,
        transform_id="dipole_spectrum",
        id="h_plane",
        outputs=("Etheta",),
    )
    scene.add(e_plane)
    scene.add(h_plane)

    outputfile = tmp_path / "hertzian_dipole"
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=outputfile,
        hide_progress_bars=True,
        cpu_precision="double",
    )

    e_magnitude = np.abs(e_plane.result.fields["Etheta"][0])
    peak = np.max(e_magnitude)
    e_normalized = e_magnitude / peak
    h_normalized = np.abs(h_plane.result.fields["Etheta"][0]) / peak
    analytical = np.abs(np.sin(np.deg2rad(theta)))
    interior = slice(1, -1)
    nonzero_h = h_normalized[h_normalized > np.finfo(h_normalized.dtype).eps]

    assert np.sqrt(np.mean((e_normalized[interior] - analytical[interior]) ** 2)) < 0.01
    assert np.max(np.abs(e_normalized[interior] - analytical[interior])) < 0.02
    assert max(e_normalized[0], e_normalized[-1]) < 1e-8
    assert np.sqrt(np.mean((h_normalized - 1) ** 2)) < 0.01
    assert 20 * np.log10(np.max(nonzero_h) / np.min(nonzero_h)) < 0.1

    with h5py.File(outputfile.with_suffix(".h5"), "r") as output:
        group = output["ntff/dipole_surface/frequency/dipole_spectrum"]
        assert group.attrs["collection_backend"] == "cython_openmp"
        assert "surface_dft" not in group
