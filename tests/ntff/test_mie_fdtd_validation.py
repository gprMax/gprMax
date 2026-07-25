"""End-to-end CPU TFSF/KSIR validation against PEC-sphere Mie scattering."""

import h5py
import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import c

import gprMax
from testing.validation.mie_pec import pec_mie_amplitudes, pec_sphere_bistatic_rcs

FREQUENCY = 5e9
SPHERE_RADIUS = 0.0096
SPHERE_CENTRE = (0.048, 0.048, 0.048)
SCATTERING_ANGLES = np.arange(0.0, 181.0, 10.0)


def _mie_scene():
    dl = 0.0016
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl,) * 3))
    scene.add(gprMax.Domain(p1=(0.096,) * 3))
    scene.add(gprMax.TimeWindow(iterations=900))
    scene.add(gprMax.OMPThreads(n=4))
    scene.add(gprMax.PMLThickness(thickness=10))
    scene.add(
        gprMax.Waveform(
            wave_type="ricker", amp=1, freq=FREQUENCY, id="plane_pulse"
        )
    )
    scene.add(
        gprMax.DiscretePlaneWaveAxial(
            p1=(0.032,) * 3,
            p2=(0.064,) * 3,
            axis="x",
            psi=90,
            waveform_id="plane_pulse",
        )
    )
    scene.add(
        gprMax.Sphere(
            p1=SPHERE_CENTRE,
            r=SPHERE_RADIUS,
            material_id="pec",
        )
    )
    scene.add(
        gprMax.KSIRSurface(
            p1=(0.024,) * 3,
            p2=(0.072,) * 3,
            id="mie_surface",
            origin=SPHERE_CENTRE,
        )
    )
    transform = gprMax.KSIRFrequencyTransform(
        surface_id="mie_surface",
        id="mie_spectrum",
        frequencies=(FREQUENCY,),
        save_surface_dft=False,
    )
    far_field = gprMax.KSIRFarField(
        theta=np.full(SCATTERING_ANGLES.shape, 90.0),
        phi=SCATTERING_ANGLES,
        transform_id="mie_spectrum",
        id="mie_pattern",
        outputs=("Etheta", "Ephi", "rcs"),
    )
    scene.add(transform)
    scene.add(far_field)
    return scene, transform, far_field


def test_cpu_tfsf_ksir_pec_sphere_matches_mie(tmp_path):
    """Exercise the production FDTD scattering and HDF5 paths."""

    scene, transform, far_field = _mie_scene()
    outputfile = tmp_path / "mie_sphere"
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=outputfile,
        hide_progress_bars=True,
        cpu_precision="double",
    )

    simulated = np.asarray(far_field.result.fields["rcs"][0])
    analytic = pec_sphere_bistatic_rcs(
        FREQUENCY,
        SPHERE_RADIUS,
        np.deg2rad(SCATTERING_ANGLES),
        polarisation="perpendicular",
    )
    simulated_pattern = simulated / np.max(simulated)
    analytic_pattern = analytic / np.max(analytic)
    pattern_rms_error = np.sqrt(np.mean((simulated_pattern - analytic_pattern) ** 2))
    relative_l2_error = np.linalg.norm(simulated - analytic) / np.linalg.norm(analytic)
    backscatter_error = abs(simulated[-1] - analytic[-1]) / analytic[-1]

    monitor = transform._compiled_outputs.transform_monitor(transform.ID)
    incident_electric = monitor.result.incident_electric[0, 2]
    wavenumber = 2 * np.pi * FREQUENCY / c
    perpendicular, _ = pec_mie_amplitudes(
        wavenumber * SPHERE_RADIUS,
        np.deg2rad(SCATTERING_ANGLES),
    )
    analytic_amplitude = -1j * np.conj(perpendicular) / wavenumber
    numerical_amplitude = far_field.result.fields["Etheta"][0] / incident_electric
    complex_relative_l2_error = np.linalg.norm(
        numerical_amplitude - analytic_amplitude
    ) / np.linalg.norm(analytic_amplitude)
    phase_rms_error = np.sqrt(
        np.mean(np.angle(numerical_amplitude / analytic_amplitude) ** 2)
    )

    assert np.all(np.isfinite(simulated))
    assert np.all(simulated > 0)
    assert pattern_rms_error < 0.12
    assert relative_l2_error < 0.35
    assert backscatter_error < 0.25
    assert complex_relative_l2_error < 0.25
    assert phase_rms_error < 0.20

    with h5py.File(str(outputfile) + ".h5", "r") as output:
        group = output["ntff/mie_surface/frequency/mie_spectrum"]
        assert group.attrs["collection_backend"] == "cython_openmp"
        assert "surface_dft" not in group
        assert_allclose(group["far_field/mie_pattern/fields/rcs"][:], simulated[None, :])
