"""Text-input coverage for the reusable positional NTFF interface."""

from pathlib import Path

import h5py
import numpy as np
import pytest

import gprMax
from gprMax.hash_cmds_file import get_user_objects
from gprMax.ntff.interface import validate_identifier
from gprMax.user_objects.cmds_output import (
    KSIRAntennaPorts,
    KSIRFarField,
    KSIRFarFieldArray,
    KSIRFrequencyRx,
    KSIRFrequencyRxArray,
    KSIRFrequencyRxSpherical,
    KSIRFrequencyTransform,
    KSIRTimeRx,
    KSIRTimeRxArray,
    KSIRTimeRxSpherical,
    NTFFAntennaPorts,
    NTFFFarField,
    NTFFFarFieldArray,
    NTFFFrequencyTransform,
    NTFFSurface,
    NTFFTimeFarField,
    NTFFTimeFarFieldArray,
    _ksir_array_points,
    _ksir_spherical_coordinates,
)


def _parse(*commands):
    return get_user_objects([f"{item}\n" for item in commands], checkessential=False)


def test_positional_hash_commands_create_public_objects():
    objects = _parse(
        "#ntff_surface: 0.02 0.03 0.04 0.08 0.09 0.10 surface1",
        "#ksir_frequency: surface1 spectrum1 1e8 2e8 hann",
        "#ksir_time_rx: 0.12 0.06 0.07 surface1 time1 Ez Hy first_arrival",
        "#ksir_time_rx_spherical: 0.4 90 30 surface1 time2 Etheta Ephi simulation",
        "#ksir_time_rx_array: 0.12 0.05 0.05 0.14 0.05 0.05 0.01 0.01 0.01 surface1 time3 Ez",
        "#ksir_frequency_rx: 0.12 0.06 0.07 spectrum1 freq1 Ex Ez",
        "#ksir_frequency_rx_spherical: 0.4 90 30 spectrum1 freq2 Etheta Ephi",
        "#ksir_frequency_rx_array: 0.12 0.05 0.05 0.14 0.05 0.05 0.01 0.01 0.01 spectrum1 freq3 Ez",
        "#ksir_far_field: 90 30 spectrum1 far1 Etheta Ephi radiation_intensity",
        "#ksir_far_field_array: 0 180 90 0 360 180 spectrum1 far2 Etheta Ephi",
        "#ksir_antenna_ports: spectrum1 feed1 feed2",
        "#ntff_frequency: surface1 spectrum2 1e8 2e8 rectangular",
        "#ntff_far_field: 90 30 spectrum2 ecfar1 Etheta Ephi",
        "#ntff_far_field_array: 0 180 90 0 360 180 spectrum2 ecfar2 Etheta Ephi",
        "#ntff_antenna_ports: spectrum2 feed1 feed2",
        "#ntff_time_far_field: 90 30 surface1 ectime1 Etheta Ephi",
        "#ntff_time_far_field_array: 0 180 90 0 360 180 surface1 ectime2 Etheta Ephi",
    )

    assert [type(item) for item in objects] == [
        NTFFSurface,
        KSIRFrequencyTransform,
        NTFFFrequencyTransform,
        KSIRAntennaPorts,
        NTFFAntennaPorts,
        KSIRTimeRx,
        KSIRTimeRxSpherical,
        KSIRTimeRxArray,
        KSIRFrequencyRx,
        KSIRFrequencyRxSpherical,
        KSIRFrequencyRxArray,
        KSIRFarField,
        KSIRFarFieldArray,
        NTFFFarField,
        NTFFFarFieldArray,
        NTFFTimeFarField,
        NTFFTimeFarFieldArray,
    ]
    transform = objects[1]
    assert transform.frequencies == (1e8, 2e8)
    assert transform.window == "hann"
    assert objects[3].port_ids == ("feed1", "feed2")
    time_receiver = objects[5]
    assert time_receiver.ID == "time1"
    assert time_receiver.outputs == ("Ez", "Hy")
    assert time_receiver.time_origin == "first_arrival"
    assert objects[2].frequencies == (1e8, 2e8)


def test_defaults_and_optional_parameter_positions_are_unambiguous():
    objects = _parse(
        "#ntff_surface: 0.02 0.02 0.02 0.08 0.08 0.08 s",
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


def test_ntff_surface_accepts_multiple_omitted_huygens_faces():
    objects = _parse(
        "#ntff_surface: 0.02 0.02 0.02 0.08 0.08 0.08 open_surface x0 xmax z0",
        "#ntff_frequency: open_surface spectrum 1e8",
        "#ntff_far_field: 90 0 spectrum",
    )

    assert objects[0].omit_faces == ("x0", "xmax", "z0")


def test_open_huygens_surface_rejects_ksir_ramahi(tmp_path):
    inputfile = tmp_path / "open_ksir.in"
    inputfile.write_text(
        "#domain: 0.08 0.08 0.08\n"
        "#dx_dy_dz: 0.004 0.004 0.004\n"
        "#time_window: 2e-10\n"
        "#pml_cells: 3\n"
        "#material: 4 0 1 0 feed\n"
        "#waveform: ricker 1 5e9 pulse\n"
        "#hertzian_dipole: z 0.02 0.04 0.04 pulse\n"
        "#box: 0 0.036 0.036 0.040 0.044 0.044 feed\n"
        "#ntff_surface: 0.028 0.028 0.028 0.052 0.052 0.052 surf x0\n"
        "#ksir_frequency: surf spectrum 5e9\n"
        "#ksir_far_field: 90 0 spectrum pattern Etheta\n"
    )

    with pytest.raises(ValueError, match="KSIR/Ramahi.*requires all six"):
        gprMax.run(
            inputfile=str(inputfile),
            n=1,
            outputfile=tmp_path / "open_ksir",
            hide_progress_bars=True,
        )


def test_open_huygens_surface_allows_source_through_omitted_face(tmp_path):
    inputfile = tmp_path / "open_huygens.in"
    inputfile.write_text(
        "#domain: 0.08 0.08 0.08\n"
        "#dx_dy_dz: 0.004 0.004 0.004\n"
        "#time_window: 2e-10\n"
        "#pml_cells: 3\n"
        "#material: 4 0 1 0 feed\n"
        "#waveform: ricker 1 5e9 pulse\n"
        "#hertzian_dipole: z 0.02 0.04 0.04 pulse\n"
        "#box: 0 0.036 0.036 0.040 0.044 0.044 feed\n"
        "#ntff_surface: 0.028 0.028 0.028 0.052 0.052 0.052 surf x0\n"
        "#ntff_frequency: surf spectrum 5e9\n"
        "#ntff_far_field: 90 0 spectrum pattern Etheta Ephi\n"
    )
    outputfile = tmp_path / "open_huygens"

    gprMax.run(
        inputfile=str(inputfile),
        n=1,
        outputfile=outputfile,
        hide_progress_bars=True,
    )

    with h5py.File(str(outputfile) + ".h5", "r") as output:
        surface = output["ntff/surf"]
        assert surface.attrs["closure"] == "huygens_open"
        assert surface.attrs["omitted_faces"].tolist() == [b"x0"]


def test_open_huygens_surface_supports_two_feed_openings(tmp_path):
    inputfile = tmp_path / "two_openings.in"
    inputfile.write_text(
        "#domain: 0.08 0.08 0.08\n"
        "#dx_dy_dz: 0.004 0.004 0.004\n"
        "#time_window: 2e-10\n"
        "#pml_cells: 3\n"
        "#material: 4 0 1 0 feed\n"
        "#waveform: ricker 1 5e9 pulse\n"
        "#hertzian_dipole: z 0.02 0.04 0.04 pulse\n"
        "#box: 0 0.036 0.036 0.080 0.044 0.044 feed\n"
        "#ntff_surface: 0.028 0.028 0.028 0.052 0.052 0.052 surf x0 xmax\n"
        "#ntff_frequency: surf spectrum 5e9\n"
        "#ntff_far_field: 90 0 spectrum pattern Etheta Ephi\n"
    )
    outputfile = tmp_path / "two_openings"

    gprMax.run(
        inputfile=str(inputfile),
        n=1,
        outputfile=outputfile,
        hide_progress_bars=True,
    )

    with h5py.File(str(outputfile) + ".h5", "r") as output:
        surface = output["ntff/surf"]
        normals = surface["frequency/spectrum/surface_dft/Ex/patch_normals"][...]
        assert surface.attrs["closure"] == "huygens_open"
        assert surface.attrs["omitted_faces"].tolist() == [b"x0", b"xmax"]
        assert not np.any(normals[:, 0])


def test_open_huygens_surface_can_omit_a_pec_backplane(tmp_path):
    inputfile = tmp_path / "pec_backplane.in"
    inputfile.write_text(
        "#domain: 0.08 0.08 0.08\n"
        "#dx_dy_dz: 0.004 0.004 0.004\n"
        "#time_window: 2e-10\n"
        "#pml_cells: 3\n"
        "#waveform: ricker 1 5e9 pulse\n"
        "#hertzian_dipole: z 0.04 0.04 0.044 pulse\n"
        "#box: 0.012 0.012 0.012 0.068 0.068 0.028 pec\n"
        "#ntff_surface: 0.024 0.024 0.028 0.056 0.056 0.056 surf z0\n"
        "#ntff_frequency: surf spectrum 5e9\n"
        "#ntff_far_field: 45 0 spectrum pattern Etheta Ephi\n"
    )
    outputfile = tmp_path / "pec_backplane"

    gprMax.run(
        inputfile=str(inputfile),
        n=1,
        outputfile=outputfile,
        hide_progress_bars=True,
    )

    with h5py.File(str(outputfile) + ".h5", "r") as output:
        surface = output["ntff/surf"]
        normals = surface["frequency/spectrum/surface_dft/Ex/patch_normals"][...]
        assert surface.attrs["omitted_faces"].tolist() == [b"z0"]
        assert not np.any(normals[:, 2] < 0)


def test_open_huygens_surface_rejects_source_beyond_a_sampled_face(tmp_path):
    inputfile = tmp_path / "wrong_opening.in"
    inputfile.write_text(
        "#domain: 0.08 0.08 0.08\n"
        "#dx_dy_dz: 0.004 0.004 0.004\n"
        "#time_window: 2e-10\n"
        "#pml_cells: 3\n"
        "#waveform: ricker 1 5e9 pulse\n"
        "#hertzian_dipole: z 0.04 0.02 0.04 pulse\n"
        "#ntff_surface: 0.028 0.028 0.028 0.052 0.052 0.052 surf x0\n"
        "#ntff_frequency: surf spectrum 5e9\n"
        "#ntff_far_field: 90 0 spectrum pattern Etheta\n"
    )

    with pytest.raises(ValueError, match="configured omitted faces"):
        gprMax.run(
            inputfile=str(inputfile),
            n=1,
            outputfile=tmp_path / "wrong_opening",
            hide_progress_bars=True,
        )


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
        "#ntff_surface: 0 0 0 1 1 1",
        "#ksir_frequency: surface transform",
        "#ksir_time_rx: 1 2 3",
        "#ksir_time_rx: 1 2 3 surface first_arrival",
        "#ksir_time_rx_array: 0 0 0 1 1 1 0.1 0.1 surface",
        "#ksir_frequency_rx: 1 2 3",
        "#ksir_far_field: 90 0",
        "#ksir_far_field_array: 0 180 5 0 360 transform",
        "#ksir_antenna_ports: transform",
        "#ntff_frequency: surface transform",
        "#ntff_far_field: 90 0",
        "#ntff_far_field_array: 0 180 5 0 360 transform",
        "#ntff_antenna_ports: transform",
        "#ntff_time_far_field: 90 0",
        "#ntff_time_far_field_array: 0 180 5 0 360 surface",
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
        "#time_window: 1e-9\n"
        "#pml_cells: 3\n"
        "#waveform: ricker 1 5e9 pulse\n"
        "#hertzian_dipole: z 0.04 0.04 0.04 pulse\n"
        "#ntff_surface: 0.028 0.028 0.028 0.052 0.052 0.052 surf1\n"
        "#ksir_frequency: surf1 spec1 5e9 rectangular\n"
        "#ksir_time_rx: 0.064 0.04 0.042 surf1 time1 Ez first_arrival\n"
        "#ksir_time_rx_spherical: 0.03 90 0 surf1 time2 Etheta simulation\n"
        "#ksir_frequency_rx: 0.064 0.04 0.042 spec1 freq1 Ez\n"
        "#ksir_frequency_rx_spherical: 0.03 90 0 spec1 freq2 Etheta\n"
        "#ksir_far_field: 90 0 spec1 far1 Etheta Ephi radiation_intensity "
        "directivity directivity_dbi\n"
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
        assert np.isfinite(transform["far_field/far1/fields/directivity"][...]).all()
        assert np.isfinite(transform["far_field/far1/radiated_power"][...]).all()
        assert surface["time/time1"].attrs["time_origin"] == "first_arrival"
        assert surface["time/time2"].attrs["coordinate_system"] == "spherical"
        assert surface["time/time1/fields/Ez"].dtype == np.float32


def test_equivalent_current_commands_run_from_shared_surface(tmp_path):
    inputfile = tmp_path / "equivalent_current.in"
    inputfile.write_text(
        "#domain: 0.08 0.08 0.08\n"
        "#dx_dy_dz: 0.004 0.004 0.004\n"
        "#time_window: 2e-10\n"
        "#pml_cells: 3\n"
        "#waveform: ricker 1 5e9 pulse\n"
        "#hertzian_dipole: z 0.04 0.04 0.04 pulse\n"
        "#ntff_surface: 0.028 0.028 0.028 0.052 0.052 0.052 surf\n"
        "#ntff_frequency: surf ec 5e9 rectangular\n"
        "#ntff_far_field_array: 0 180 45 0 0 1 ec pattern Etheta Ephi\n"
        "#ksir_frequency: surf ksir 5e9 rectangular\n"
        "#ksir_far_field_array: 0 180 45 0 0 1 ksir pattern Etheta Ephi\n"
    )
    outputfile = tmp_path / "equivalent_current"

    gprMax.run(
        inputfile=str(inputfile),
        n=1,
        outputfile=outputfile,
        hide_progress_bars=True,
        cpu_precision="double",
    )

    with h5py.File(str(outputfile) + ".h5", "r") as output:
        surface = output["ntff/surf"]
        transform = surface["frequency/ec"]
        far_field = transform["far_field/pattern"]
        assert surface.attrs["formulation"] == "shared_ntff_surface"
        assert transform.attrs["formulation"] == "equivalent_current"
        assert transform.attrs["collection_backend"] == "cython_openmp"
        assert np.isfinite(far_field["fields/Etheta"][...]).all()
        assert np.isfinite(far_field["fields/Ephi"][...]).all()
        equivalent = np.abs(far_field["fields/Etheta"][0])
        ksir = np.abs(surface["frequency/ksir/far_field/pattern/fields/Etheta"][0])
        equivalent /= np.max(equivalent)
        expected = np.sin(np.deg2rad(np.arange(0, 181, 45)))
        np.testing.assert_allclose(equivalent, expected, rtol=0, atol=0.08)
        assert np.isfinite(ksir).all()


def test_1997_time_far_field_stores_only_complete_retarded_window(tmp_path):
    inputfile = tmp_path / "equivalent_current_time.in"
    inputfile.write_text(
        "#domain: 0.08 0.08 0.08\n"
        "#dx_dy_dz: 0.004 0.004 0.004\n"
        "#time_window: 3e-10\n"
        "#pml_cells: 3\n"
        "#waveform: ricker 1 5e9 pulse\n"
        "#hertzian_dipole: z 0.04 0.04 0.04 pulse\n"
        "#ntff_surface: 0.028 0.028 0.028 0.052 0.052 0.052 surf\n"
        "#ntff_time_far_field_array: 0 180 45 0 0 1 surf transient "
        "Etheta Ephi Ex Hphi\n"
    )
    outputfile = tmp_path / "equivalent_current_time"

    gprMax.run(
        inputfile=str(inputfile),
        n=1,
        outputfile=outputfile,
        hide_progress_bars=True,
        cpu_precision="double",
    )

    with h5py.File(str(outputfile) + ".h5", "r") as output:
        group = output["ntff/surf/time_far_field/transient"]
        assert group.attrs["formulation"] == "equivalent_current_1997"
        assert group.attrs["interpolation"] == "linear"
        assert group.attrs["range_normalized"] == 1
        assert group["times"].size < 3e-10 / output.attrs["dt"]
        assert np.all(np.diff(group["times"][...]) > 0)
        for name in ("Etheta", "Ephi", "Ex", "Hphi"):
            values = group[f"fields/{name}"][...]
            assert values.shape[0] == 5
            assert np.isfinite(values).all()


def test_antenna_metrics_run_from_single_voltage_port(tmp_path):
    inputfile = tmp_path / "antenna_metrics.in"
    inputfile.write_text(
        "#domain: 0.08 0.08 0.08\n"
        "#dx_dy_dz: 0.004 0.004 0.004\n"
        "#time_window: 4e-10\n"
        "#pml_cells: 3\n"
        "#waveform: ricker 1 5e9 pulse\n"
        "#voltage_source: z 0.04 0.04 0.04 50 pulse\n"
        "#rx_port: 0.04 0.04 0.04 feed\n"
        "#ntff_surface: 0.028 0.028 0.028 0.052 0.052 0.052 surf\n"
        "#ksir_frequency: surf band 5e9 rectangular\n"
        "#ksir_antenna_ports: band feed\n"
        "#ksir_far_field: 90 0 band broadside Etheta Ephi directivity "
        "directivity_dbi gain gain_dbi realized_gain realized_gain_dbi "
        "radiation_efficiency total_efficiency\n"
    )
    outputfile = tmp_path / "antenna_metrics"

    gprMax.run(
        inputfile=str(inputfile),
        n=1,
        outputfile=outputfile,
        hide_progress_bars=True,
        cpu_precision="double",
    )

    with h5py.File(str(outputfile) + ".h5", "r") as output:
        group = output["ntff/surf/frequency/band/far_field/broadside"]
        assert group.attrs["radiation_quadrature_theta_order"] >= 12
        assert group.attrs["radiation_quadrature_phi_order"] >= 24
        assert group.attrs["maximum_directivity_sampling"] == "full-sphere quadrature plus requested directions"
        assert group["port_power/port_ids"].asstr()[...].tolist() == ["feed"]
        assert group["port_power/incident_voltage_per_port"].shape == (1, 1)
        assert group["port_power/terminal_voltage_per_port"].shape == (1, 1)
        assert group["port_power/terminal_current_per_port"].shape == (1, 1)
        assert group["port_power"].attrs["spectral_power_units"] == "W s^2"
        assert group["port_power/gain_valid"][0] == 1
        assert group["port_power/realized_gain_valid"][0] == 1
        for name in (
            "directivity",
            "directivity_dbi",
            "gain",
            "gain_dbi",
            "realized_gain",
            "realized_gain_dbi",
            "radiation_efficiency",
            "total_efficiency",
        ):
            assert np.isfinite(group[f"fields/{name}"][...]).all()


def test_eigenmode_port_normalises_gain_and_realized_gain(tmp_path):
    inputfile = (
        Path(__file__).parents[2] / "examples" / "features" / "eigenmode_sources" / "dielectric_rod_antenna_3d.in"
    )
    outputfile = tmp_path / "eigenmode_antenna"

    gprMax.run(
        inputfile=str(inputfile),
        n=1,
        outputfile=outputfile,
        hide_progress_bars=True,
    )

    with h5py.File(str(outputfile) + ".h5", "r") as output:
        far_field = output["ntff/radiation_surface/frequency/antenna_band/far_field/full_sphere"]
        port_power = far_field["port_power"]
        modal_port = port_power["modal_ports/port1"]

        assert port_power["port_ids"].asstr()[...].tolist() == ["port1"]
        assert port_power["representations"].asstr()[...].tolist() == ["modal_power_waves"]
        assert np.isnan(port_power["reference_impedances"][0])
        assert np.all(port_power["mesh_valid"][...] == 1)
        assert np.all(port_power["terminal_valid"][...] == 1)
        assert np.all(port_power["gain_valid"][...] == 1)
        assert np.all(port_power["realized_gain_valid"][...] == 1)
        assert modal_port.attrs["port_id"] == "port1"
        assert modal_port["incident"].shape == (1, 9)
        assert modal_port["outgoing"].shape == (1, 9)
        assert modal_port["power_matrix"].shape == (9, 1, 1)
        assert modal_port["electric_cross_power_matrix"].shape == (9, 1, 1)
        # This deliberately coarse antenna model is only a smoke test for the
        # modal-port/NTFF normalization path. Its upper-band reflection is
        # about 0.35 on the CI mesh, so keep the bound above that physical
        # response while still catching a broken incident/outgoing split.
        assert np.all(np.abs(output["eigenmode_ports/port1/S"][0]) < 0.4)
        radiation_efficiency = far_field["fields/radiation_efficiency"][...]
        total_efficiency = far_field["fields/total_efficiency"][...]
        directivity_dbi = far_field["fields/directivity_dbi"][...]
        gain_dbi = far_field["fields/gain_dbi"][...]
        realized_gain_dbi = far_field["fields/realized_gain_dbi"][...]
        assert np.all((0 < radiation_efficiency) & (radiation_efficiency <= 1))
        assert np.all((0 < total_efficiency) & (total_efficiency <= 1))
        assert np.all(gain_dbi <= directivity_dbi + 1e-12)
        assert np.all(realized_gain_dbi <= gain_dbi + 1e-12)
        assert np.isfinite(gain_dbi).all()
        assert np.isfinite(realized_gain_dbi).all()


def test_multiport_gain_keeps_zero_amplitude_port_in_power_balance(tmp_path):
    inputfile = tmp_path / "multiport_metrics.in"
    inputfile.write_text(
        "#domain: 0.08 0.08 0.08\n"
        "#dx_dy_dz: 0.004 0.004 0.004\n"
        "#time_window: 4e-10\n"
        "#pml_cells: 3\n"
        "#waveform: ricker 1 5e9 driven\n"
        "#waveform: ricker 0 5e9 terminated\n"
        "#voltage_source: z 0.036 0.04 0.04 50 driven\n"
        "#voltage_source: z 0.044 0.04 0.04 50 terminated\n"
        "#rx_port: 0.036 0.04 0.04 element1\n"
        "#rx_port: 0.044 0.04 0.04 element2\n"
        "#ntff_surface: 0.024 0.028 0.028 0.056 0.052 0.052 surf\n"
        "#ksir_frequency: surf band 5e9 rectangular\n"
        "#ksir_antenna_ports: band element1 element2\n"
        "#ksir_far_field: 90 0 band broadside gain realized_gain\n"
    )
    outputfile = tmp_path / "multiport_metrics"

    gprMax.run(
        inputfile=str(inputfile),
        n=1,
        outputfile=outputfile,
        hide_progress_bars=True,
        cpu_precision="double",
    )

    with h5py.File(str(outputfile) + ".h5", "r") as output:
        group = output["ntff/surf/frequency/band/far_field/broadside"]
        port_power = group["port_power"]
        assert port_power["port_ids"].asstr()[...].tolist() == ["element1", "element2"]
        assert port_power["incident_power_per_port"][0, 0] > 0
        assert port_power["incident_power_per_port"][1, 0] == 0
        assert port_power["incident_voltage_per_port"].shape == (2, 1)
        assert port_power["terminal_voltage_per_port"].shape == (2, 1)
        assert port_power["terminal_current_per_port"].shape == (2, 1)
        assert port_power["terminal_valid"][0] == 1
        assert np.isfinite(port_power["accepted_power_per_port"][:, 0]).all()
        assert np.isfinite(group["fields/gain"][...]).all()
        assert np.isfinite(group["fields/realized_gain"][...]).all()


def test_automatic_transmission_line_port_can_normalise_gain(tmp_path):
    inputfile = tmp_path / "transmission_line_gain.in"
    inputfile.write_text(
        "#domain: 0.08 0.08 0.08\n"
        "#dx_dy_dz: 0.004 0.004 0.004\n"
        "#time_window: 4e-10\n"
        "#pml_cells: 3\n"
        "#waveform: ricker 1 5e9 pulse\n"
        "#transmission_line: z 0.04 0.04 0.04 50 pulse\n"
        "#ntff_surface: 0.028 0.028 0.028 0.052 0.052 0.052 surf\n"
        "#ksir_frequency: surf band 5e9 rectangular\n"
        "#ksir_antenna_ports: band tl1\n"
        "#ksir_far_field: 90 0 band broadside gain realized_gain\n"
    )
    outputfile = tmp_path / "transmission_line_gain"

    gprMax.run(
        inputfile=str(inputfile),
        n=1,
        outputfile=outputfile,
        hide_progress_bars=True,
        cpu_precision="double",
    )

    with h5py.File(str(outputfile) + ".h5", "r") as output:
        group = output["ntff/surf/frequency/band/far_field/broadside"]
        assert group["port_power/port_ids"].asstr()[...].tolist() == ["tl1"]
        assert group["port_power/source_types"].asstr()[...].tolist() == ["TransmissionLine"]
        assert group["port_power/gain_valid"][0] == 1
        assert np.isfinite(group["fields/gain"][...]).all()
        assert np.isfinite(group["fields/realized_gain"][...]).all()


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
        "#ntff_surface: 0 0.028 0.028 0.016 0.052 0.052 half\n"
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
