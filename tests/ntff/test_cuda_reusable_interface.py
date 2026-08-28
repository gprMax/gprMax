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

"""CUDA/CPU parity for the reusable grouped KSIR interface."""

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import gprMax

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

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
        gprMax.NTFFSurface(
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


def _antenna_scene():
    dl = 0.004
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.08, 0.08, 0.08)))
    scene.add(gprMax.TimeWindow(time=4e-10))
    scene.add(gprMax.PMLThickness(thickness=3))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    scene.add(
        gprMax.VoltageSource(
            polarisation="z",
            p1=(0.04, 0.04, 0.04),
            resistance=50,
            waveform_id="pulse",
            id="feed",
        )
    )
    scene.add(
        gprMax.NTFFSurface(
            p1=(0.028, 0.028, 0.028),
            p2=(0.052, 0.052, 0.052),
            id="surface",
        )
    )
    scene.add(gprMax.KSIRFrequencyTransform("surface", "spectrum", (5e9,)))
    scene.add(gprMax.KSIRAntennaPorts("spectrum", ("feed",)))
    far_field = gprMax.KSIRFarField(
        (30, 90, 150),
        (0, 0, 0),
        "spectrum",
        id="far",
        outputs=(
            "directivity",
            "gain",
            "realized_gain",
            "radiation_efficiency",
            "total_efficiency",
        ),
    )
    scene.add(far_field)
    return scene, far_field


def _equivalent_current_scene():
    dl = 0.004
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.08, 0.08, 0.08)))
    scene.add(gprMax.TimeWindow(time=1e-9))
    scene.add(gprMax.PMLThickness(thickness=3))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.04, 0.04, 0.04), waveform_id="pulse"))
    scene.add(
        gprMax.NTFFSurface(
            p1=(0.028, 0.028, 0.028),
            p2=(0.052, 0.052, 0.052),
            id="surface",
        )
    )
    transform = gprMax.NTFFFrequencyTransform("surface", "current", (5e9,))
    far_field = gprMax.NTFFFarField(
        (30, 90, 150),
        (0, 0, 0),
        "current",
        id="far",
        outputs=("Etheta", "Ephi"),
    )
    scene.add(transform)
    scene.add(far_field)
    return scene, transform, far_field


def _equivalent_current_time_scene():
    dl = 0.004
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.08, 0.08, 0.08)))
    scene.add(gprMax.TimeWindow(time=3e-10))
    scene.add(gprMax.PMLThickness(thickness=3))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    scene.add(gprMax.HertzianDipole(polarisation="z", p1=(0.04, 0.04, 0.04), waveform_id="pulse"))
    scene.add(
        gprMax.NTFFSurface(
            p1=(0.028, 0.028, 0.028),
            p2=(0.052, 0.052, 0.052),
            id="surface",
        )
    )
    far_field = gprMax.NTFFTimeFarField(
        (0, 30, 90, 150, 180),
        (0, 0, 0, 0, 0),
        "surface",
        id="transient",
        outputs=("Etheta", "Ephi"),
    )
    scene.add(far_field)
    return scene, far_field


def _layered_equivalent_current_time_scene():
    dl = 0.004
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.08, 0.08, 0.08)))
    scene.add(gprMax.TimeWindow(time=4e-9))
    scene.add(gprMax.PMLThickness(thickness=3))
    scene.add(gprMax.Material(er=2.5, se=0, mr=1, sm=0, id="substrate"))
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.08, 0.08, 0.04), material_id="substrate"))
    scene.add(gprMax.Box(p1=(0, 0, 0), p2=(0.08, 0.08, 0.016), material_id="pec"))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    scene.add(gprMax.HertzianDipole(polarisation="x", p1=(0.04, 0.04, 0.032), waveform_id="pulse"))
    scene.add(
        gprMax.NTFFSurface(
            p1=(0.028, 0.028, 0.016),
            p2=(0.052, 0.052, 0.052),
            id="surface",
            origin=(0.04, 0.04, 0.04),
            omit_faces=("z0",),
        )
    )
    scene.add(
        gprMax.NTFFLayeredBackground(
            id="grounded",
            axis="z",
            materials=("free_space", "substrate", "pec"),
            interfaces=(0.04, 0.016),
        )
    )
    scene.add(
        gprMax.NTFFLayeredTimeTransform(
            surface_id="surface",
            id="transient",
            background_id="grounded",
            impulse_tolerance=1e-2,
            max_impulses=1_000,
        )
    )
    far_field = gprMax.NTFFLayeredTimeFarField(
        theta=(20, 35, 50),
        phi=(0, 30, 90),
        transform_id="transient",
        id="layered",
        outputs=("Etheta", "Ephi"),
    )
    scene.add(far_field)
    return scene, far_field


@pytest.mark.skipif(not HAS_CUDA, reason="No CUDA device/pycuda available")
@pytest.mark.parametrize(
    "precision,rtol",
    [("single", 3e-4), ("double", 2e-11)],
)
def test_cuda_equivalent_current_time_far_fields_match_cpu(tmp_path, gpu_device, precision, rtol):
    cpu_scene, cpu_far = _equivalent_current_time_scene()
    cuda_scene, cuda_far = _equivalent_current_time_scene()
    gprMax.run(
        scenes=[cpu_scene],
        outputfile=tmp_path / f"equivalent_time_cpu_{precision}",
        hide_progress_bars=True,
        cpu_precision=precision,
    )
    gprMax.run(
        scenes=[cuda_scene],
        outputfile=tmp_path / f"equivalent_time_cuda_{precision}",
        hide_progress_bars=True,
        gpu=[gpu_device],
        gpu_precision=precision,
    )

    assert_allclose(cuda_far.result.times, cpu_far.result.times, rtol=0, atol=0)
    field_scale = max(np.max(np.abs(cpu_far.result.fields[component])) for component in ("Etheta", "Ephi"))
    assert field_scale > 0
    for component in ("Etheta", "Ephi"):
        expected = cpu_far.result.fields[component]
        assert_allclose(
            cuda_far.result.fields[component],
            expected,
            rtol=rtol,
            # The cross-polarised component is analytically zero for this
            # axis-aligned dipole. Scale absolute round-off against the
            # physical co-polar field rather than numerical cancellation.
            atol=rtol * field_scale,
        )


@pytest.mark.skipif(not HAS_CUDA, reason="No CUDA device/pycuda available")
@pytest.mark.parametrize(
    "precision,rtol",
    [("single", 8e-4), ("double", 1e-7)],
)
def test_cuda_layered_equivalent_current_time_far_fields_match_cpu(tmp_path, gpu_device, precision, rtol):
    cpu_scene, cpu_far = _layered_equivalent_current_time_scene()
    cuda_scene, cuda_far = _layered_equivalent_current_time_scene()
    gprMax.run(
        scenes=[cpu_scene],
        outputfile=tmp_path / f"layered_time_cpu_{precision}",
        hide_progress_bars=True,
        cpu_precision=precision,
    )
    gprMax.run(
        scenes=[cuda_scene],
        outputfile=tmp_path / f"layered_time_cuda_{precision}",
        hide_progress_bars=True,
        gpu=[gpu_device],
        gpu_precision=precision,
    )

    assert_allclose(cuda_far.result.times, cpu_far.result.times, rtol=0, atol=0)
    field_scale = max(np.max(np.abs(cpu_far.result.fields[component])) for component in ("Etheta", "Ephi"))
    assert field_scale > 0
    for component in ("Etheta", "Ephi"):
        assert_allclose(
            cuda_far.result.fields[component],
            cpu_far.result.fields[component],
            rtol=rtol,
            atol=rtol * field_scale,
        )
    with h5py.File(tmp_path / f"layered_time_cuda_{precision}.h5", "r") as output:
        group = output["ntff/surface/time_far_field/layered"]
        assert group.attrs["solver"] == "cuda"
        assert group.attrs["collection_backend"] == "cuda_device_layered"


def test_opencl_layered_equivalent_current_time_far_fields_match_cpu(tmp_path, opencl_device):
    cpu_scene, cpu_far = _layered_equivalent_current_time_scene()
    opencl_scene, opencl_far = _layered_equivalent_current_time_scene()
    gprMax.run(
        scenes=[cpu_scene],
        outputfile=tmp_path / "layered_time_cpu_opencl",
        hide_progress_bars=True,
        cpu_precision="single",
    )
    gprMax.run(
        scenes=[opencl_scene],
        outputfile=tmp_path / "layered_time_opencl",
        hide_progress_bars=True,
        opencl=[opencl_device],
        gpu_precision="single",
    )

    assert_allclose(opencl_far.result.times, cpu_far.result.times, rtol=0, atol=0)
    field_scale = max(np.max(np.abs(cpu_far.result.fields[component])) for component in ("Etheta", "Ephi"))
    for component in ("Etheta", "Ephi"):
        assert_allclose(
            opencl_far.result.fields[component],
            cpu_far.result.fields[component],
            rtol=1e-3,
            atol=1e-3 * field_scale,
        )
    with h5py.File(tmp_path / "layered_time_opencl.h5", "r") as output:
        group = output["ntff/surface/time_far_field/layered"]
        assert group.attrs["solver"] == "opencl"
        assert group.attrs["collection_backend"] == "opencl_device_layered"


def _plane_wave_rcs_scene():
    dl = 0.004
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.Domain(p1=(0.08, 0.08, 0.08)))
    scene.add(gprMax.TimeWindow(time=4e-10))
    scene.add(gprMax.PMLThickness(thickness=3))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    scene.add(
        gprMax.DiscretePlaneWaveAxial(
            p1=(0.028, 0.028, 0.028),
            p2=(0.052, 0.052, 0.052),
            axis="x",
            psi=90,
            waveform_id="pulse",
        )
    )
    scene.add(
        gprMax.NTFFSurface(
            p1=(0.02, 0.02, 0.02),
            p2=(0.06, 0.06, 0.06),
            id="surface",
            origin=(0.04, 0.04, 0.04),
        )
    )
    transform = gprMax.KSIRFrequencyTransform(
        "surface",
        "spectrum",
        (5e9,),
        save_surface_dft=False,
        plane_wave_index=0,
    )
    far_field = gprMax.KSIRFarField(
        (90,),
        (180,),
        "spectrum",
        id="backscatter",
        outputs=("rcs",),
    )
    scene.add(transform)
    scene.add(far_field)
    return scene, transform, far_field


@pytest.mark.skipif(not HAS_CUDA, reason="No CUDA device/pycuda available")
@pytest.mark.parametrize(
    "precision,real_dtype,complex_dtype,rtol",
    [
        ("single", np.dtype("float32"), np.dtype("complex64"), 1e-3),
        ("double", np.dtype("float64"), np.dtype("complex128"), 2e-10),
    ],
)
def test_cuda_reusable_outputs_match_cpu(tmp_path, gpu_device, precision, real_dtype, complex_dtype, rtol):
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
        gpu=[gpu_device],
        gpu_precision=precision,
    )

    assert cuda_time.result.fields["Ez"].dtype == real_dtype
    assert cuda_frequency.result.fields["Ez"].dtype == complex_dtype
    assert cuda_far.result.fields["Etheta"].dtype == complex_dtype
    assert_array_equal(cuda_time.result.valid_lengths, cpu_time.result.valid_lengths)
    assert_array_equal(
        cuda_time.result.fully_supported_lengths,
        cpu_time.result.fully_supported_lengths,
    )
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
        assert_allclose(actual.field, expected.field, rtol=rtol, atol=rtol * np.max(np.abs(expected.field)))
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
        assert_array_equal(
            time_group["fully_supported_lengths"][:],
            cuda_time.result.fully_supported_lengths,
        )
        assert_allclose(
            time_group["terminal_field_ratios"][:],
            cuda_time.result.terminal_field_ratios,
        )
        assert_array_equal(
            time_group["terminal_decay_ok"][:],
            cuda_time.result.terminal_decay_ok,
        )
        assert frequency_group.attrs["solver"] == "cuda"
        assert frequency_group.attrs["collection_backend"] == "cuda_device"
        assert frequency_group["surface_dft/Ez/field"].dtype == complex_dtype


@pytest.mark.skipif(not HAS_CUDA, reason="No CUDA device/pycuda available")
@pytest.mark.parametrize(
    "precision,complex_dtype,rtol",
    [
        ("single", np.dtype("complex64"), 2e-3),
        ("double", np.dtype("complex128"), 2e-10),
    ],
)
def test_cuda_equivalent_current_far_field_matches_cpu(tmp_path, gpu_device, precision, complex_dtype, rtol):
    cpu_scene, _, cpu_far = _equivalent_current_scene()
    cuda_scene, cuda_transform, cuda_far = _equivalent_current_scene()
    gprMax.run(
        scenes=[cpu_scene],
        n=1,
        outputfile=tmp_path / f"cpu_current_{precision}",
        hide_progress_bars=True,
        cpu_precision=precision,
    )
    gprMax.run(
        scenes=[cuda_scene],
        n=1,
        outputfile=tmp_path / f"cuda_current_{precision}",
        hide_progress_bars=True,
        gpu=[gpu_device],
        gpu_precision=precision,
    )

    pattern_scale = np.max(np.abs(cpu_far.result.fields["Etheta"]))
    assert pattern_scale > 0
    for component in ("Etheta", "Ephi"):
        expected = cpu_far.result.fields[component]
        assert_allclose(
            cuda_far.result.fields[component],
            expected,
            rtol=rtol,
            # Ephi is a symmetry null for this dipole and is therefore
            # dominated by round-off. Scale the absolute tolerance by the
            # non-zero physical pattern rather than by the null itself.
            atol=rtol * pattern_scale,
        )
    assert cuda_far.result.fields["Etheta"].dtype == complex_dtype
    assert set(cuda_transform.surface_data) == {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}
    with h5py.File(tmp_path / f"cuda_current_{precision}.h5", "r") as output:
        group = output["ntff/surface/frequency/current"]
        assert group.attrs["formulation"] == "equivalent_current"
        assert group.attrs["collection_backend"] == "cuda_device"


@pytest.mark.skipif(not HAS_CUDA, reason="No CUDA device/pycuda available")
def test_cuda_antenna_metrics_match_cpu(tmp_path, gpu_device):
    cpu_scene, cpu_far = _antenna_scene()
    cuda_scene, cuda_far = _antenna_scene()
    gprMax.run(
        scenes=[cpu_scene],
        n=1,
        outputfile=tmp_path / "cpu_antenna",
        hide_progress_bars=True,
        cpu_precision="single",
    )
    gprMax.run(
        scenes=[cuda_scene],
        n=1,
        outputfile=tmp_path / "cuda_antenna",
        hide_progress_bars=True,
        gpu=[gpu_device],
        gpu_precision="single",
    )

    for output in cpu_far.result.fields:
        assert_allclose(
            cuda_far.result.fields[output],
            cpu_far.result.fields[output],
            rtol=2e-3,
            atol=2e-3 * np.nanmax(np.abs(cpu_far.result.fields[output])),
        )
    assert_allclose(
        cuda_far.result.radiation_metrics.radiated_power,
        cpu_far.result.radiation_metrics.radiated_power,
        rtol=2e-3,
    )
    assert_allclose(
        cuda_far.result.port_metrics.accepted_power,
        cpu_far.result.port_metrics.accepted_power,
        rtol=2e-3,
    )
    assert_allclose(
        cuda_far.result.port_metrics.incident_power,
        cpu_far.result.port_metrics.incident_power,
        rtol=2e-3,
    )

    with h5py.File(tmp_path / "cuda_antenna.h5", "r") as output:
        group = output["ntff/surface/frequency/spectrum/far_field/far"]
        assert group["port_power/port_ids"].asstr()[...].tolist() == ["feed"]
        assert group["port_power/gain_valid"][0] == 1


@pytest.mark.skipif(not HAS_CUDA, reason="No CUDA device/pycuda available")
@pytest.mark.parametrize(
    "precision,complex_dtype,rtol",
    [
        ("single", np.dtype("complex64"), 2e-4),
        ("double", np.dtype("complex128"), 2e-10),
    ],
)
def test_cuda_plane_wave_rcs_incident_reference_matches_cpu(tmp_path, gpu_device, precision, complex_dtype, rtol):
    cpu_scene, cpu_transform, _ = _plane_wave_rcs_scene()
    cuda_scene, cuda_transform, cuda_far = _plane_wave_rcs_scene()
    gprMax.run(
        scenes=[cpu_scene],
        n=1,
        outputfile=tmp_path / f"cpu_plane_wave_rcs_{precision}",
        hide_progress_bars=True,
        cpu_precision=precision,
    )
    gprMax.run(
        scenes=[cuda_scene],
        n=1,
        outputfile=tmp_path / f"cuda_plane_wave_rcs_{precision}",
        hide_progress_bars=True,
        gpu=[gpu_device],
        gpu_precision=precision,
    )

    cpu_monitor = cpu_transform._compiled_outputs.transform_monitor(cpu_transform.ID)
    cuda_monitor = cuda_transform._compiled_outputs.transform_monitor(cuda_transform.ID)
    assert cuda_monitor.result.incident_electric.dtype == complex_dtype
    assert_allclose(
        cuda_monitor.result.incident_electric,
        cpu_monitor.result.incident_electric,
        rtol=rtol,
        atol=rtol * np.max(np.abs(cpu_monitor.result.incident_electric)),
    )
    assert np.all(np.isfinite(cuda_far.result.fields["rcs"]))

    with h5py.File(tmp_path / f"cuda_plane_wave_rcs_{precision}.h5", "r") as output:
        transform = output["ntff/surface/frequency/spectrum"]
        assert transform.attrs["solver"] == "cuda"
        assert transform.attrs["collection_backend"] == "cuda_device"
        assert np.all(np.isfinite(transform["far_field/backscatter/fields/rcs"][:]))
