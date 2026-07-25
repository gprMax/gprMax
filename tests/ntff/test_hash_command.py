"""Text-input coverage for the reusable positional KSIR interface."""

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.hash_cmds_file import get_user_objects
from gprMax.ntff.interface import validate_identifier
from gprMax.user_objects.cmds_output import (
    KSIRFarField,
    KSIRFarFieldArray,
    KSIRFrequencyRx,
    KSIRFrequencyRxArray,
    KSIRFrequencyRxSpherical,
    KSIRFrequencyTransform,
    KSIRSurface,
    KSIRTimeRx,
    KSIRTimeRxArray,
    KSIRTimeRxSpherical,
    _ksir_array_points,
    _ksir_spherical_coordinates,
)


def _parse(*commands):
    return get_user_objects([f"{item}\n" for item in commands], checkessential=False)


def test_positional_hash_commands_create_public_objects():
    objects = _parse(
        "#ksir_surface: 0.02 0.03 0.04 0.08 0.09 0.10 surface1",
        "#ksir_frequency: surface1 spectrum1 1e8 2e8 hann",
        "#ksir_time_rx: 0.12 0.06 0.07 surface1 time1 Ez Hy first_arrival",
        "#ksir_time_rx_spherical: 0.4 90 30 surface1 time2 Etheta Ephi simulation",
        "#ksir_time_rx_array: 0.12 0.05 0.05 0.14 0.05 0.05 0.01 0.01 0.01 surface1 time3 Ez",
        "#ksir_frequency_rx: 0.12 0.06 0.07 spectrum1 freq1 Ex Ez",
        "#ksir_frequency_rx_spherical: 0.4 90 30 spectrum1 freq2 Etheta Ephi",
        "#ksir_frequency_rx_array: 0.12 0.05 0.05 0.14 0.05 0.05 0.01 0.01 0.01 spectrum1 freq3 Ez",
        "#ksir_far_field: 90 30 spectrum1 far1 Etheta Ephi radiation_intensity",
        "#ksir_far_field_array: 0 180 90 0 360 180 spectrum1 far2 Etheta Ephi",
    )

    assert [type(item) for item in objects] == [
        KSIRSurface,
        KSIRFrequencyTransform,
        KSIRTimeRx,
        KSIRTimeRxSpherical,
        KSIRTimeRxArray,
        KSIRFrequencyRx,
        KSIRFrequencyRxSpherical,
        KSIRFrequencyRxArray,
        KSIRFarField,
        KSIRFarFieldArray,
    ]
    transform = objects[1]
    assert transform.frequencies == (1e8, 2e8)
    assert transform.window == "hann"
    time_receiver = objects[2]
    assert time_receiver.ID == "time1"
    assert time_receiver.outputs == ("Ez", "Hy")
    assert time_receiver.time_origin == "first_arrival"


def test_defaults_and_optional_parameter_positions_are_unambiguous():
    objects = _parse(
        "#ksir_surface: 0.02 0.02 0.02 0.08 0.08 0.08 s",
        "#ksir_frequency: s f 1e8",
        "#ksir_time_rx: 0.1 0.05 0.05 s",
        "#ksir_far_field: 90 0 f",
    )

    assert objects[1].window == "rectangular"
    assert objects[2].ID is None
    assert objects[2].outputs is None
    assert objects[2].time_origin == "simulation"
    assert objects[3].ID is None
    assert objects[3].outputs is None


def test_cartesian_array_accepts_zero_step_only_on_fixed_axes():
    points = _ksir_array_points((1.0, 2.0, 3.0), (1.2, 2.0, 3.0), (0.1, 0.0, 0.0), np.float64)

    assert points.shape == (3, 3)
    np.testing.assert_allclose(points[:, 0], (1.0, 1.1, 1.2))
    with pytest.raises(ValueError, match="positive step"):
        _ksir_array_points((1.0, 2.0, 3.0), (1.2, 2.0, 3.0), (0.0, 0.0, 0.0), np.float64)

    single_points = _ksir_array_points(
        (0.12, 0.05, 0.05),
        (0.14, 0.05, 0.05),
        (0.01, 0.0, 0.0),
        np.float32,
    )
    assert single_points.shape == (3, 3)


def test_spherical_coordinates_require_positive_radius_and_physical_theta():
    with pytest.raises(ValueError, match="radii must be positive"):
        _ksir_spherical_coordinates(0, 90, 0, np.float64)
    with pytest.raises(ValueError, match="theta must lie"):
        _ksir_spherical_coordinates(1, 181, 0, np.float64)


@pytest.mark.parametrize("identifier", ["", " ", ".", "..", "a/b", "a\x00b"])
def test_hdf5_identifiers_reject_invalid_path_components(identifier):
    with pytest.raises(ValueError, match="HDF5 path component"):
        validate_identifier("KSIR ID", identifier)


@pytest.mark.parametrize(
    "command",
    [
        "#ksir_surface: 0 0 0 1 1 1",
        "#ksir_frequency: surface transform",
        "#ksir_time_rx: 1 2 3",
        "#ksir_time_rx: 1 2 3 surface first_arrival",
        "#ksir_time_rx_array: 0 0 0 1 1 1 0.1 0.1 surface",
        "#ksir_frequency_rx: 1 2 3",
        "#ksir_far_field: 90 0",
        "#ksir_far_field_array: 0 180 5 0 360 transform",
        "#ksir_field_extension: 0 0 0 1 1 1 2 2 2 components=Ez",
        "#ksir_near_to_far: 0 0 0 1 1 1",
    ],
)
def test_malformed_or_retired_hash_commands_are_rejected(command):
    with pytest.raises((ValueError, KeyError, SyntaxError)):
        _parse(command)


def test_reusable_commands_run_and_write_grouped_hdf5(tmp_path):
    inputfile = tmp_path / "reusable_ksir.in"
    inputfile.write_text(
        "#domain: 0.08 0.08 0.08\n"
        "#dx_dy_dz: 0.004 0.004 0.004\n"
        "#time_window: 2e-10\n"
        "#pml_cells: 3\n"
        "#waveform: ricker 1 5e9 pulse\n"
        "#hertzian_dipole: z 0.04 0.04 0.04 pulse\n"
        "#ksir_surface: 0.028 0.028 0.028 0.052 0.052 0.052 surf1\n"
        "#ksir_frequency: surf1 spec1 5e9 rectangular\n"
        "#ksir_time_rx: 0.064 0.04 0.042 surf1 time1 Ez first_arrival\n"
        "#ksir_time_rx_spherical: 0.03 90 0 surf1 time2 Etheta simulation\n"
        "#ksir_frequency_rx: 0.064 0.04 0.042 spec1 freq1 Ez\n"
        "#ksir_frequency_rx_spherical: 0.03 90 0 spec1 freq2 Etheta\n"
        "#ksir_far_field: 90 0 spec1 far1 Etheta Ephi radiation_intensity\n"
    )
    outputfile = tmp_path / "reusable_ksir"

    gprMax.run(
        inputfile=str(inputfile),
        n=1,
        outputfile=outputfile,
        hide_progress_bars=True,
        cpu_precision="single",
    )

    with h5py.File(str(outputfile) + ".h5", "r") as output:
        surface = output["ntff/surf1"]
        assert surface.attrs["closure"] == "closed"
        transform = surface["frequency/spec1"]
        assert transform.attrs["phasor_time_sign"] == "exp(+j*omega*t)"
        assert transform["frequencies"].dtype == np.float32
        assert transform["surface_dft/Ez/field"].dtype == np.complex64
        assert transform["receivers/freq1"].attrs["range_normalized"] == 0
        assert transform["far_field/far1"].attrs["range_normalized"] == 1
        assert np.isfinite(transform["receivers/freq1/fields/Ez"][...]).all()
        assert np.isfinite(transform["receivers/freq2/fields/Etheta"][...]).all()
        assert np.isfinite(transform["far_field/far1/fields/Etheta"][...]).all()
        assert np.isfinite(transform["far_field/far1/fields/radiation_intensity"][...]).all()
        assert surface["time/time1"].attrs["time_origin"] == "first_arrival"
        assert surface["time/time2"].attrs["coordinate_system"] == "spherical"
        assert surface["time/time1/fields/Ez"].dtype == np.float32


def test_surface_on_symmetry_plane_is_completed_automatically(tmp_path):
    inputfile = tmp_path / "symmetric_ksir.in"
    inputfile.write_text(
        "#domain: 0.04 0.08 0.08\n"
        "#dx_dy_dz: 0.004 0.004 0.004\n"
        "#time_window: 2e-10\n"
        "#pml_cells: 3\n"
        "#symmetry_boundary: x0 pmc\n"
        "#waveform: ricker 1 5e9 pulse\n"
        "#hertzian_dipole: z 0.008 0.04 0.04 pulse\n"
        "#ksir_surface: 0 0.028 0.028 0.016 0.052 0.052 half\n"
        "#ksir_time_rx: 0.024 0.04 0.042 half fields Ez\n"
    )
    outputfile = tmp_path / "symmetric_ksir"

    gprMax.run(
        inputfile=str(inputfile),
        n=1,
        outputfile=outputfile,
        hide_progress_bars=True,
        cpu_precision="double",
    )

    with h5py.File(str(outputfile) + ".h5", "r") as output:
        surface = output["ntff/half"]
        assert surface.attrs["closure"] == "symmetry"
        assert surface.attrs["omitted_faces"].tolist() == [b"x0"]
        assert surface.attrs["symmetry_plane_types"].tolist() == [b"pmc"]
        assert surface.attrs["symmetry_image_count"] == 2
        np.testing.assert_allclose(surface.attrs["physical_origin"], (0.0, 0.04, 0.04))
