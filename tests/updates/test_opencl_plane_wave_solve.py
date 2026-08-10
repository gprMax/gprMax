"""CPU/GPU parity for discrete plane-wave execution paths."""

from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

import gprMax

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.slow]

COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _run_trace(
    scene,
    output: Path,
    *,
    opencl_device=None,
    cuda_device=None,
    precision="single",
):
    if opencl_device is not None and cuda_device is not None:
        raise ValueError("Select at most one GPU backend")
    if opencl_device is not None:
        options = {"opencl": [opencl_device], "gpu_precision": precision}
    elif cuda_device is not None:
        options = {"gpu": [cuda_device], "gpu_precision": precision}
    else:
        options = {"cpu_precision": precision}
    gprMax.run(
        scenes=[scene],
        outputfile=output,
        hide_progress_bars=True,
        **options,
    )
    with h5py.File(output.with_suffix(".h5"), "r") as result:
        return {component: result[f"rxs/rx1/{component}"][:] for component in COMPONENTS}


def _assert_trace_parity(cpu, opencl, rtol=2e-5):
    scale = max(np.max(np.abs(values)) for values in cpu.values())
    assert scale > 0
    for component in COMPONENTS:
        assert_allclose(opencl[component], cpu[component], rtol=rtol, atol=rtol * scale)


def _standard_lorentz_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.OMPThreads(n=1))
    scene.add(gprMax.Discretisation(p1=(1e-3,) * 3))
    scene.add(gprMax.Domain(p1=(20e-3,) * 3))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=2e-10))
    scene.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id="lorentz"))
    scene.add(
        gprMax.AddLorentzDispersion(
            poles=2,
            er_delta=(1.0, 0.5),
            omega=(8e9, 15e9),
            delta=(1e9, 1.5e9),
            material_ids=("lorentz",),
        )
    )
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(20e-3,) * 3, material_id="lorentz"))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(
        gprMax.DiscretePlaneWaveVector(
            p1=(4e-3,) * 3,
            p2=(16e-3,) * 3,
            m_vec=(1, 2, 1),
            psi=35,
            waveform_id="w",
            material_id="lorentz",
        )
    )
    scene.add(gprMax.Rx(p1=(10e-3,) * 3))
    return scene


def _standard_drude_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.OMPThreads(n=1))
    scene.add(gprMax.Discretisation(p1=(1e-3,) * 3))
    scene.add(gprMax.Domain(p1=(20e-3,) * 3))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=2e-10))
    scene.add(gprMax.Material(er=3, se=0, mr=1, sm=0, id="drude"))
    scene.add(
        gprMax.AddDrudeDispersion(
            poles=2,
            omega=(1e9, 2e9),
            alpha=(0.8e9, 1.2e9),
            material_ids=("drude",),
        )
    )
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(20e-3,) * 3, material_id="drude"))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w"))
    scene.add(
        gprMax.DiscretePlaneWaveVector(
            p1=(4e-3,) * 3,
            p2=(16e-3,) * 3,
            m_vec=(1, 2, 1),
            psi=35,
            waveform_id="w",
            material_id="drude",
        )
    )
    scene.add(gprMax.Rx(p1=(10e-3,) * 3))
    return scene


def _layered_axial_scene():
    inf = float("inf")
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode="TM"))
    scene.add(gprMax.OMPThreads(n=1))
    scene.add(gprMax.Discretisation(p1=(1e-3,) * 3))
    scene.add(gprMax.Domain(p1=(0.04, 0.02, inf)))
    scene.add(gprMax.TimeWindow(time=3e-10))
    scene.add(gprMax.PMLThickness(thickness=(3, 3, 0, 3, 3, 0)))
    scene.add(gprMax.Material(er=2.5, se=0, mr=1, sm=0, id="debye"))
    scene.add(
        gprMax.AddDebyeDispersion(
            poles=2,
            er_delta=(2.0, 0.75),
            tau=(1e-11, 4e-11),
            material_ids=("debye",),
        )
    )
    scene.add(
        gprMax.Box(
            p1=(0.025, 0, inf),
            p2=(0.04, 0.02, inf),
            material_id="debye",
        )
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=8e9, id="w"))
    scene.add(
        gprMax.DiscretePlaneWaveAxial(
            p1=(0.008, 0.006, inf),
            p2=(0.032, 0.014, inf),
            axis="x",
            psi=90,
            waveform_id="w",
        )
    )
    scene.add(gprMax.Rx(p1=(0.020, 0.010, inf)))
    return scene


def _two_dimensional_scene(mode):
    inf = float("inf")
    psi = 90 if mode == "TM" else 0
    scene = gprMax.Scene()
    scene.add(gprMax.DomainMode(mode=mode))
    scene.add(gprMax.Discretisation(p1=(1e-3,) * 3))
    scene.add(gprMax.Domain(p1=(0.04, 0.04, inf)))
    scene.add(gprMax.TimeWindow(time=2e-10))
    scene.add(gprMax.PMLThickness(thickness=(3, 3, 0, 3, 3, 0)))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=8e9, id="w"))
    scene.add(
        gprMax.DiscretePlaneWaveAngles(
            p1=(0.010, 0.010, inf),
            p2=(0.030, 0.030, inf),
            theta=90,
            phi=26.565051177,
            psi=psi,
            waveform_id="w",
        )
    )
    scene.add(gprMax.Rx(p1=(0.020, 0.020, inf)))
    return scene


def _multiple_windowed_scene():
    scene = gprMax.Scene()
    scene.add(gprMax.OMPThreads(n=1))
    scene.add(gprMax.Discretisation(p1=(1e-3,) * 3))
    scene.add(gprMax.Domain(p1=(20e-3,) * 3))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(time=2e-10))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=10e9, id="w1"))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=0.4, freq=7e9, id="w2"))
    for waveform_id, m_vec, psi, start, stop in (
        ("w1", (1, 0, 0), 0, 2e-11, 1.4e-10),
        ("w2", (0, 1, 0), 90, 4e-11, 1.7e-10),
    ):
        scene.add(
            gprMax.DiscretePlaneWaveVector(
                p1=(4e-3,) * 3,
                p2=(16e-3,) * 3,
                m_vec=m_vec,
                psi=psi,
                waveform_id=waveform_id,
                start=start,
                stop=stop,
            )
        )
    scene.add(gprMax.Rx(p1=(10e-3,) * 3))
    return scene


@pytest.mark.parametrize(
    "name,scene_factory",
    [
        ("standard_lorentz", _standard_lorentz_scene),
        ("standard_drude", _standard_drude_scene),
        ("layered_axial", _layered_axial_scene),
        ("tm", lambda: _two_dimensional_scene("TM")),
        ("te", lambda: _two_dimensional_scene("TE")),
        ("multiple_windowed", _multiple_windowed_scene),
    ],
)
def test_opencl_plane_wave_traces_match_cpu(tmp_path, opencl_device, name, scene_factory):
    cpu = _run_trace(scene_factory(), tmp_path / f"{name}_cpu")
    opencl = _run_trace(
        scene_factory(),
        tmp_path / f"{name}_opencl",
        opencl_device=opencl_device,
    )

    _assert_trace_parity(cpu, opencl)


@pytest.mark.parametrize(
    "name,scene_factory",
    [
        ("standard_lorentz", _standard_lorentz_scene),
        ("layered_axial", _layered_axial_scene),
    ],
)
def test_cuda_device_source_initialisation_matches_cpu(tmp_path, gpu_device, name, scene_factory):
    cpu = _run_trace(scene_factory(), tmp_path / f"{name}_cpu_cuda_reference")
    cuda = _run_trace(
        scene_factory(),
        tmp_path / f"{name}_cuda",
        cuda_device=gpu_device,
    )

    _assert_trace_parity(cpu, cuda)


def _rcs_scene(formulation):
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(0.004,) * 3))
    scene.add(gprMax.Domain(p1=(0.08,) * 3))
    scene.add(gprMax.TimeWindow(time=4e-10))
    scene.add(gprMax.PMLThickness(thickness=3))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    scene.add(
        gprMax.DiscretePlaneWaveAxial(
            p1=(0.028,) * 3,
            p2=(0.052,) * 3,
            axis="x",
            psi=90,
            waveform_id="pulse",
        )
    )
    scene.add(gprMax.Sphere(p1=(0.04,) * 3, r=0.006, material_id="pec"))
    scene.add(
        gprMax.NTFFSurface(
            p1=(0.02,) * 3,
            p2=(0.06,) * 3,
            id="surface",
            origin=(0.04,) * 3,
        )
    )
    transform_type, far_field_type = {
        "ksir": (gprMax.KSIRFrequencyTransform, gprMax.KSIRFarField),
        "equivalent_current": (gprMax.NTFFFrequencyTransform, gprMax.NTFFFarField),
    }[formulation]
    transform = transform_type("surface", "spectrum", (5e9,), plane_wave_index=0)
    far_field = far_field_type(
        (90,),
        (180,),
        "spectrum",
        id="backscatter",
        outputs=("rcs",),
    )
    scene.add(transform)
    scene.add(far_field)
    return scene, transform, far_field


@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("formulation", ("ksir", "equivalent_current"))
def test_opencl_rcs_incident_reference_matches_cpu(tmp_path, opencl_device, precision, formulation):
    results = {}
    for backend in ("cpu", "opencl"):
        scene, transform, far_field = _rcs_scene(formulation)
        options = (
            {"opencl": [opencl_device], "gpu_precision": precision}
            if backend == "opencl"
            else {"cpu_precision": precision}
        )
        gprMax.run(
            scenes=[scene],
            outputfile=tmp_path / f"rcs_{formulation}_{backend}_{precision}",
            hide_progress_bars=True,
            **options,
        )
        monitor = transform._compiled_outputs.transform_monitor(transform.ID)
        results[backend] = (
            monitor.result.incident_electric.copy(),
            far_field.result.fields["rcs"].copy(),
        )

    assert_allclose(results["opencl"][0], results["cpu"][0], rtol=2e-5)
    assert_allclose(results["opencl"][1], results["cpu"][1], rtol=2e-5)
    assert np.all(np.isfinite(results["opencl"][1]))
