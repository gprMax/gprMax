"""CUDA/CPU parity for the reusable grouped KSIR interface."""

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import gprMax

try:
    import pycuda.driver as _cuda_driver

    _cuda_driver.init()
    HAS_CUDA = _cuda_driver.Device.count() > 0
except Exception:
    HAS_CUDA = False


def _scene():
    dl = 0.004
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.08, 0.08, 0.08)))
    scene.add(gprMax.TimeWindow(time=2e-10))
    scene.add(gprMax.PMLThickness(thickness=3))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.04, 0.04, 0.04), waveform_id="pulse"))
    scene.add(
        gprMax.KSIRSurface(
            p1=(0.028, 0.028, 0.028),
            p2=(0.052, 0.052, 0.052),
            id="surface",
        )
    )
    transform = gprMax.KSIRFrequencyTransform("surface", "spectrum", (5e9,))
    time_receiver = gprMax.KSIRTimeRx(
        ((0.064, 0.04, 0.042), (0.068, 0.04, 0.042)),
        "surface",
        id="time",
        outputs=("Ez", "Hy"),
        time_origin="first_arrival",
    )
    frequency_receiver = gprMax.KSIRFrequencyRx(
        (0.064, 0.04, 0.042),
        "spectrum",
        id="frequency",
        outputs=("Ez",),
    )
    far_field = gprMax.KSIRFarField(
        (30, 90, 150),
        (0, 0, 0),
        "spectrum",
        id="far",
        outputs=("Etheta", "Ephi"),
    )
    for item in (transform, time_receiver, frequency_receiver, far_field):
        scene.add(item)
    return scene, transform, time_receiver, frequency_receiver, far_field


@pytest.mark.skipif(not HAS_CUDA, reason="No CUDA device/pycuda available")
@pytest.mark.parametrize(
    "precision,real_dtype,complex_dtype,rtol",
    [
        ("single", np.dtype("float32"), np.dtype("complex64"), 1e-3),
        ("double", np.dtype("float64"), np.dtype("complex128"), 2e-10),
    ],
)
def test_cuda_reusable_outputs_match_cpu(tmp_path, precision, real_dtype, complex_dtype, rtol):
    cpu_scene, cpu_transform, cpu_time, cpu_frequency, cpu_far = _scene()
    cuda_scene, cuda_transform, cuda_time, cuda_frequency, cuda_far = _scene()
    gprMax.run(
        scenes=[cpu_scene],
        n=1,
        outputfile=tmp_path / f"cpu_{precision}",
        hide_progress_bars=True,
        cpu_precision=precision,
    )
    gprMax.run(
        scenes=[cuda_scene],
        n=1,
        outputfile=tmp_path / f"cuda_{precision}",
        hide_progress_bars=True,
        gpu=[0],
        gpu_precision=precision,
    )

    assert cuda_time.result.fields["Ez"].dtype == real_dtype
    assert cuda_frequency.result.fields["Ez"].dtype == complex_dtype
    assert cuda_far.result.fields["Etheta"].dtype == complex_dtype
    assert_array_equal(cuda_time.result.valid_lengths, cpu_time.result.valid_lengths)
    assert_allclose(cuda_time.result.time_origins, cpu_time.result.time_origins, rtol=0, atol=0)

    for component in ("Ez", "Hy"):
        expected = cpu_time.result.fields[component]
        scale = np.max(np.abs(expected))
        assert_allclose(
            cuda_time.result.fields[component],
            expected,
            rtol=rtol,
            atol=rtol * scale,
        )
    for component in cuda_transform.surface_data:
        expected = cpu_transform.surface_data[component]
        actual = cuda_transform.surface_data[component]
        assert_allclose(
            actual.field, expected.field, rtol=rtol, atol=rtol * np.max(np.abs(expected.field))
        )
        assert_allclose(
            actual.normal_derivative,
            expected.normal_derivative,
            rtol=2 * rtol,
            atol=2 * rtol * np.max(np.abs(expected.normal_derivative)),
        )
    assert_allclose(
        cuda_frequency.result.fields["Ez"],
        cpu_frequency.result.fields["Ez"],
        rtol=2 * rtol,
        atol=2 * rtol * np.max(np.abs(cpu_frequency.result.fields["Ez"])),
    )
    assert_allclose(
        cuda_far.result.fields["Etheta"],
        cpu_far.result.fields["Etheta"],
        rtol=2 * rtol,
        atol=2 * rtol * np.max(np.abs(cpu_far.result.fields["Etheta"])),
    )

    with h5py.File(tmp_path / f"cuda_{precision}.h5", "r") as output:
        time_group = output["ntff/surface/time/time"]
        frequency_group = output["ntff/surface/frequency/spectrum"]
        assert time_group.attrs["solver"] == "cuda"
        assert time_group.attrs["collection_backend"] == "cuda_device"
        assert frequency_group.attrs["solver"] == "cuda"
        assert frequency_group.attrs["collection_backend"] == "cuda_device"
        assert frequency_group["surface_dft/Ez/field"].dtype == complex_dtype
