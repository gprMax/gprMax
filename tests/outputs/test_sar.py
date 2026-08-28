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

"""CPU tagged-cell frequency-domain SAR output tests."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import gprMax
from gprMax import config
from gprMax.grid.mpi_grid import MPIGrid
from gprMax.hash_cmds_file import get_user_objects
from gprMax.materials import DispersiveMaterial, Material
from gprMax.ports import PortPowerSpectrum
from gprMax.sar import (
    EDGE_OFFSETS,
    SARLocalPayload,
    SARMonitor,
    _material_loss_conductivity,
    _mpi_owned_edge_mask,
    _pml_cell_mask,
)
from gprMax.user_objects.cmds_output import SAR


def _scene(spectrum_limit=10, dispersive=False, averaging_masses=()):
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.024, 0.024, 0.024)))
    scene.add(gprMax.Discretisation(p1=(0.002, 0.002, 0.002)))
    scene.add(gprMax.TimeWindow(time=2e-9))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.Material(er=4, se=0.5, mr=1, sm=0, id="tissue"))
    if dispersive:
        scene.add(
            gprMax.AddDebyeDispersion(
                poles=1,
                er_delta=(2.0,),
                tau=(1e-10,),
                material_ids=("tissue",),
            )
        )
    scene.add(gprMax.MaterialDensity(density=1000, material_ids="tissue"))
    scene.add(
        gprMax.Box(
            p1=(0.004, 0.004, 0.004),
            p2=(0.020, 0.020, 0.020),
            material_id="tissue",
            tag="target",
        )
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1e9, id="pulse"))
    scene.add(
        gprMax.VoltageSource(
            p1=(0.012, 0.012, 0.012),
            polarisation="z",
            resistance=50,
            waveform_id="pulse",
        )
    )
    output = SAR(
        frequencies=(0.75e9, 1e9, 1.25e9),
        waveform_id="pulse",
        tags="target",
        id="target_sar",
        spectrum_limit=spectrum_limit,
        averaging_masses=averaging_masses,
    )
    scene.add(output)
    return scene, output


def test_sar_cpu_integration_writes_sparse_tagged_output(tmp_path):
    scene, api_output = _scene()
    output = tmp_path / "sar"
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output,
        hide_progress_bars=True,
        cpu_precision="single",
    )

    assert api_output.ID == "target_sar"
    assert api_output.result.sar.shape[0] == 3
    with h5py.File(str(output) + ".h5", "r") as data:
        sar = data["sar/target_sar"]
        assert sar.attrs["Units"] == "W/kg"
        assert sar.attrs["MinimumWavelengthCells"] == 10
        assert sar["frequency"].shape == (3,)
        assert sar["sar"].shape[0] == 3
        assert sar["sar"].shape[1] == sar["cell_indices"].shape[0]
        assert np.all(sar["density"][...] == 1000)
        assert np.all(sar["valid"][...])
        assert np.all(np.isfinite(sar["sar"][...]))
        assert np.all(sar["sar"][...] >= 0)
        assert "tags/target/mass_average_sar" in sar
        assert "spatial_average" not in sar


def test_sar_rejects_frequency_beyond_requested_mesh_limit(tmp_path):
    scene, _ = _scene(spectrum_limit=10)
    scene.output_objects[-1].frequencies = (20e9,)
    with pytest.raises(ValueError, match="cells per shortest wavelength"):
        gprMax.run(
            scenes=[scene],
            n=1,
            outputfile=tmp_path / "invalid_sar",
            hide_progress_bars=True,
        )


def test_sar_hash_command_uses_explicit_frequency_range_and_tags():
    objects = get_user_objects(
        ["#sar: 1e8 3e8 3 pulse 1 8 dosimetry brain eyes\n"],
        checkessential=False,
    )
    assert len(objects) == 1
    output = objects[0]
    assert isinstance(output, SAR)
    np.testing.assert_allclose(output.frequencies, (1e8, 2e8, 3e8))
    assert output.waveform_id == "pulse"
    assert output.tags == ("brain", "eyes")
    assert output.spectrum_limit == 8
    assert output.averaging_masses == ()


def test_sar_python_api_defaults_to_local_cell_output():
    output = SAR(frequencies=(1e9,), waveform_id="pulse", tags="target")
    assert output.averaging_masses == ()


def test_sar_hash_command_accepts_explicit_spatial_averaging_masses():
    objects = get_user_objects(
        ["#sar: 1e8 3e8 3 pulse 1 8 dosimetry " "spatial_average 3 0.001 0.01 0.05 brain eyes\n"],
        checkessential=False,
    )
    output = objects[0]
    assert output.averaging_masses == pytest.approx((0.001, 0.01, 0.05))
    assert output.tags == ("brain", "eyes")


def test_sar_hash_power_normalisation_accepts_spatial_averaging_masses():
    objects = get_user_objects(
        [
            "#sar: 1e9 1e9 1 pulse accepted_power 1 feed 10 dose "
            "spatial_average 2 0.001 0.01 body\n"
        ],
        checkessential=False,
    )
    output = objects[0]
    assert output.averaging_masses == pytest.approx((0.001, 0.01))
    assert output.tags == ("body",)


def test_sar_hash_command_accepts_explicit_nyquist_research_mode():
    objects = get_user_objects(
        ["#sar: 1e8 1e8 1 pulse 1 nyquist full brain\n"],
        checkessential=False,
    )
    assert objects[0].spectrum_limit == "nyquist"


def test_sar_hash_command_accepts_port_power_normalisation():
    objects = get_user_objects(
        ["#sar: 1e9 1e9 1 pulse accepted_power 2.5 feed 10 dose body\n"],
        checkessential=False,
    )
    output = objects[0]
    assert output.normalisation == "accepted_power"
    assert output.target_power == pytest.approx(2.5)
    assert output.port_id == "feed"
    assert output.tags == ("body",)


def test_sar_hash_port_power_normalisation_does_not_require_waveform():
    objects = get_user_objects(
        ["#sar: 1e9 1e9 1 accepted_power 2.5 feed 10 dose body\n"],
        checkessential=False,
    )
    output = objects[0]
    assert output.waveform_id is None
    assert output.normalisation == "accepted_power"
    assert output.target_power == pytest.approx(2.5)
    assert output.port_id == "feed"
    assert output.tags == ("body",)


def test_sar_hash_current_moment_normalisation_is_explicit():
    objects = get_user_objects(
        ["#sar: 1e9 1e9 1 pulse current_moment 0.25 10 dose body\n"],
        checkessential=False,
    )
    output = objects[0]
    assert output.waveform_id == "pulse"
    assert output.normalisation == "current_moment"
    assert output.target_amplitude == pytest.approx(0.25)
    assert output.port_id is None
    assert output.tags == ("body",)


def test_sar_hash_plane_wave_incident_flux_normalisation_is_explicit():
    output = get_user_objects(
        ["#sar: 1e9 1e9 1 pulse incident_flux 2.5 10 dose body\n"],
        checkessential=False,
    )[0]

    assert output.waveform_id == "pulse"
    assert output.normalisation == "incident_flux"
    assert output.target_flux == pytest.approx(2.5)
    assert output.port_id is None
    assert output.tags == ("body",)


def test_sar_cell_formula_matches_homogeneous_analytical_value():
    """For a peak phasor, SAR = sigma |E|^2 / (2 rho)."""

    monitor = SARMonitor.__new__(SARMonitor)
    monitor._next_iteration = monitor.grid_iterations = 1
    monitor._source_samples = np.asarray([1.0], dtype=np.float64)
    monitor.frequencies = np.asarray([1e9])
    monitor.grid_dt = 1e-12
    monitor.window = np.ones(1)
    monitor.complex_dtype = np.dtype(np.complex128)
    monitor.real_dtype = np.dtype(np.float64)
    monitor.target_amplitude = 1.0
    monitor.source_floor_db = -40.0
    monitor.mesh_valid = np.asarray([True])
    monitor.cells_per_wavelength = np.asarray([20.0])
    monitor.limiting_material = np.asarray(["tissue"])
    monitor.cells = np.asarray([[0, 0, 0]], dtype=np.int32)
    monitor.cell_tag_ids = np.asarray([1], dtype=np.uint8)
    monitor.cell_material_ids = np.asarray([2], dtype=np.uint32)
    monitor.density = np.asarray([1000.0])
    monitor.cell_material_loss = np.asarray([[0.5]])
    monitor.cell_edge_indices = {
        component: np.asarray([[0, 1, 2, 3]]) for component in ("Ex", "Ey", "Ez")
    }
    # engineering_dft([1]) = dt; choose Ex=10 V/m and Ey=Ez=0.
    monitor.accumulators = {
        "Ex": np.full((1, 4), 10 * monitor.grid_dt, dtype=np.complex128),
        "Ey": np.zeros((1, 4), dtype=np.complex128),
        "Ez": np.zeros((1, 4), dtype=np.complex128),
    }
    monitor.result = None
    monitor.averaging_masses = ()
    monitor.normalisation = "waveform"

    result = monitor.finalise()

    assert result.absorbed_power_density[0, 0] == pytest.approx(25.0)
    assert result.sar[0, 0] == pytest.approx(0.025)


def test_mpi_sar_edge_ownership_is_unique_and_includes_outer_boundary():
    lower = SimpleNamespace(
        lower_extent=np.asarray((0, 0, 0), dtype=np.int32),
        upper_extent=np.asarray((5, 4, 4), dtype=np.int32),
        negative_halo_offset=np.asarray((0, 0, 0), dtype=bool),
        global_size=np.asarray((10, 4, 4), dtype=np.int32),
    )
    upper = SimpleNamespace(
        lower_extent=np.asarray((4, 0, 0), dtype=np.int32),
        upper_extent=np.asarray((10, 4, 4), dtype=np.int32),
        negative_halo_offset=np.asarray((1, 0, 0), dtype=bool),
        global_size=np.asarray((10, 4, 4), dtype=np.int32),
    )
    coordinates = np.asarray(((4, 2, 2), (5, 2, 2), (10, 2, 2)), dtype=np.int32)

    np.testing.assert_array_equal(_mpi_owned_edge_mask(lower, coordinates), (True, False, False))
    np.testing.assert_array_equal(_mpi_owned_edge_mask(upper, coordinates), (False, True, True))


def test_mpi_sar_gather_adds_no_collective_when_no_monitors():
    grid = SimpleNamespace(sar_monitors=[])

    assert MPIGrid.gather_sar_payloads(grid) is None


def test_mpi_sar_internal_pml_mask_is_localised_from_global_coordinates():
    spec = SimpleNamespace(xs=3, xf=7, ys=1, yf=3, zs=2, zf=4)
    grid = SimpleNamespace(
        nx=6,
        ny=5,
        nz=5,
        lower_extent=np.asarray((4, 0, 0), dtype=np.int32),
        pmls={
            "thickness": {name: 0 for name in ("x0", "xmax", "y0", "ymax", "z0", "zmax")},
            "internal_specs": [spec],
        },
    )

    mask = _pml_cell_mask(grid)

    expected = np.zeros((6, 5, 5), dtype=bool)
    expected[0:3, 1:3, 2:4] = True
    np.testing.assert_array_equal(mask, expected)


def test_mpi_sar_payload_collocates_edges_across_partition_corner(monkeypatch):
    cell = np.asarray(((5, 5, 5),), dtype=np.int32)
    required = {
        component: cell[:, np.newaxis, :] + offsets[np.newaxis, :, :]
        for component, offsets in EDGE_OFFSETS.items()
    }
    payloads = []
    for rank in range(2):
        edge_coordinates = {}
        edge_dft = {}
        for component, coordinates in required.items():
            coordinates = coordinates.reshape(-1, 3)
            selection = np.arange(coordinates.shape[0]) % 2 == rank
            edge_coordinates[component] = coordinates[selection]
            value = 2.0 if component == "Ex" else 0.0
            edge_dft[component] = np.full(
                (1, np.count_nonzero(selection)), value, dtype=np.complex128
            )
        payloads.append(
            SARLocalPayload(
                cell_indices=cell if rank == 0 else np.empty((0, 3), dtype=np.int32),
                tag_id=np.asarray((1,), dtype=np.uint8)
                if rank == 0
                else np.empty(0, dtype=np.uint8),
                material_id=np.asarray((3,), dtype=np.uint32)
                if rank == 0
                else np.empty(0, dtype=np.uint32),
                density=np.asarray((1000.0,)) if rank == 0 else np.empty(0),
                absorbed_power_density=np.zeros((1, 1 if rank == 0 else 0)),
                excluded_pml_cell_count=rank,
                edge_coordinates=edge_coordinates,
                edge_dft=edge_dft,
            )
        )

    monitor = SARMonitor.__new__(SARMonitor)
    monitor.frequencies = np.asarray((1e9,))
    monitor.real_dtype = np.dtype(np.float64)
    monitor.grid = SimpleNamespace()
    monkeypatch.setattr(
        "gprMax.sar._material_loss_conductivity",
        lambda *args: np.ones((1, 1), dtype=np.float64),
    )

    merged = monitor.merge_local_payloads(payloads)
    collocated = monitor._collocate_mpi_payload(merged, (10, 10, 10))

    np.testing.assert_array_equal(collocated.cell_indices, cell)
    assert collocated.absorbed_power_density[0, 0] == pytest.approx(2.0)
    assert collocated.excluded_pml_cell_count == 1


def test_sar_debye_loss_conductivity_matches_closed_form(monkeypatch):
    material = DispersiveMaterial(2, "debye_tissue")
    material.er = 4.0
    material.se = 0.2
    material.poles = 1
    material.type = "debye"
    material.deltaer = [2.0]
    material.tau = [1e-10]
    frequencies = np.asarray((0.5e9, 1.0e9, 2.0e9))
    omega = 2 * np.pi * frequencies
    monkeypatch.setattr(config, "sim_config", SimpleNamespace(em_consts={"e0": config.e0}))
    expected = material.se + (
        omega
        * config.e0
        * material.deltaer[0]
        * omega
        * material.tau[0]
        / (1 + (omega * material.tau[0]) ** 2)
    )

    actual = _material_loss_conductivity(
        SimpleNamespace(materials=[material]),
        np.asarray((material.numID,)),
        frequencies,
    )[:, 0]

    np.testing.assert_allclose(actual, expected, rtol=1e-14)


def test_sar_perfect_conductors_have_zero_finite_volume_loss(monkeypatch):
    """Ideal conductors must not create the indeterminate product inf * 0."""

    pec = Material(0, "pec")
    pec.se = float("inf")
    pmc = Material(1, "pmc")
    pmc.sm = float("inf")
    frequencies = np.asarray((0.5e9, 1.0e9, 2.0e9))
    monkeypatch.setattr(config, "sim_config", SimpleNamespace(em_consts={"e0": config.e0}))

    actual = _material_loss_conductivity(
        SimpleNamespace(materials=[pec, pmc]),
        np.asarray((pec.numID, pmc.numID)),
        frequencies,
    )

    np.testing.assert_array_equal(actual, np.zeros((3, 2)))
    assert np.all(np.isfinite(actual))


def test_sar_rejects_non_finite_non_pec_material_loss(monkeypatch):
    material = Material(2, "invalid")
    material.se = float("nan")
    monkeypatch.setattr(config, "sim_config", SimpleNamespace(em_consts={"e0": config.e0}))

    with pytest.raises(ValueError, match="non-finite electric loss"):
        _material_loss_conductivity(
            SimpleNamespace(materials=[material]),
            np.asarray((material.numID,)),
            np.asarray((1e9,)),
        )


def test_sar_incident_power_normalisation_scales_quadratically(monkeypatch):
    monitor = SARMonitor.__new__(SARMonitor)
    monitor._next_iteration = monitor.grid_iterations = 1
    monitor._source_samples = np.asarray([1.0])
    monitor.frequencies = np.asarray([1e9])
    monitor.grid_dt = 1.0
    monitor.window = np.ones(1)
    monitor.window_name = "rectangular"
    monitor.complex_dtype = np.dtype(np.complex128)
    monitor.real_dtype = np.dtype(np.float64)
    monitor.target_amplitude = 1.0
    monitor.source_floor_db = -40.0
    monitor.mesh_valid = np.asarray([True])
    monitor.cells_per_wavelength = np.asarray([20.0])
    monitor.limiting_material = np.asarray(["tissue"])
    monitor.cells = np.asarray([[0, 0, 0]], dtype=np.int32)
    monitor.cell_tag_ids = np.asarray([1], dtype=np.uint8)
    monitor.cell_material_ids = np.asarray([2], dtype=np.uint32)
    monitor.density = np.asarray([1000.0])
    monitor.cell_material_loss = np.asarray([[0.5]])
    monitor.cell_edge_indices = {
        component: np.asarray([[0, 1, 2, 3]]) for component in ("Ex", "Ey", "Ez")
    }
    monitor.accumulators = {
        "Ex": np.full((1, 4), 10.0, dtype=np.complex128),
        "Ey": np.zeros((1, 4), dtype=np.complex128),
        "Ez": np.zeros((1, 4), dtype=np.complex128),
    }
    monitor.result = None
    monitor.averaging_masses = ()
    monitor.normalisation = "incident_power"
    monitor.port_id = "feed"
    monitor.target_power = 8.0
    monitor.grid = SimpleNamespace()

    monkeypatch.setattr("gprMax.sar.port_output_registry", lambda grid: {"feed": object()})
    monkeypatch.setattr(
        "gprMax.sar.evaluate_port_power_spectrum",
        lambda *args, **kwargs: PortPowerSpectrum(
            port_id="feed",
            source_type="test",
            reference_impedance=50,
            frequency=np.asarray([1e9]),
            incident_voltage=np.asarray([1 + 0j]),
            terminal_voltage=np.asarray([1 + 0j]),
            terminal_current=np.asarray([1 + 0j]),
            incident_power=np.asarray([2.0]),
            accepted_power=np.asarray([1.0]),
            mesh_valid=np.asarray([True]),
            terminal_valid=np.asarray([True]),
        ),
    )

    result = monitor.finalise()

    assert result.normalisation_scale[0] == pytest.approx(2.0)
    assert result.sar[0, 0] == pytest.approx(0.1)
    assert np.isnan(result.source_spectrum[0])
    assert result.source_relative_db[0] == pytest.approx(0.0)
    assert result.incident_power[0] == pytest.approx(2.0)


def test_sar_current_moment_normalises_the_hertzian_source_length():
    monitor = SARMonitor.__new__(SARMonitor)
    monitor._next_iteration = monitor.grid_iterations = 1
    monitor._source_samples = np.asarray([1.0])
    monitor._source_length = 0.25
    monitor.frequencies = np.asarray([1e9])
    monitor.grid_dt = 1.0
    monitor.window = np.ones(1)
    monitor.complex_dtype = np.dtype(np.complex128)
    monitor.real_dtype = np.dtype(np.float64)
    monitor.target_amplitude = 1.0
    monitor.source_floor_db = -40.0
    monitor.mesh_valid = np.asarray([True])
    monitor.cells_per_wavelength = np.asarray([20.0])
    monitor.limiting_material = np.asarray(["tissue"])
    monitor.cells = np.asarray([[0, 0, 0]], dtype=np.int32)
    monitor.cell_tag_ids = np.asarray([1], dtype=np.uint8)
    monitor.cell_material_ids = np.asarray([2], dtype=np.uint32)
    monitor.density = np.asarray([1000.0])
    monitor.cell_material_loss = np.asarray([[0.5]])
    monitor.cell_edge_indices = {
        component: np.asarray([[0, 1, 2, 3]]) for component in ("Ex", "Ey", "Ez")
    }
    monitor.accumulators = {
        "Ex": np.full((1, 4), 10.0, dtype=np.complex128),
        "Ey": np.zeros((1, 4), dtype=np.complex128),
        "Ez": np.zeros((1, 4), dtype=np.complex128),
    }
    monitor.result = None
    monitor.averaging_masses = ()
    monitor.normalisation = "current_moment"

    result = monitor.finalise()

    # Unit current moment requires four times the field amplitude produced by
    # the 0.25 m, unit-current source, hence sixteen times its SAR.
    assert result.normalisation_scale[0] == pytest.approx(4.0)
    assert result.sar[0, 0] == pytest.approx(0.4)


def test_sar_real_port_power_normalisation_scales_with_target_power(tmp_path):
    scene, one_watt = _scene()
    one_watt.normalisation = "incident_power"
    one_watt.port_id = "feed"
    one_watt.target_power = 1.0
    one_watt.waveform_id = None
    voltage_source = next(
        item for item in scene.grid_objects if isinstance(item, gprMax.VoltageSource)
    )
    voltage_source.id = "feed"
    four_watt = SAR(
        frequencies=one_watt.frequencies,
        tags="target",
        id="target_sar_4W",
        normalisation="incident_power",
        port_id="feed",
        target_power=4.0,
        averaging_masses=(),
    )
    scene.add(four_watt)

    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=tmp_path / "sar_power",
        hide_progress_bars=True,
    )

    assert np.all(one_watt.result.valid)
    assert np.all(four_watt.result.valid)
    np.testing.assert_allclose(four_watt.result.sar, 4 * one_watt.result.sar, rtol=2e-6)


def test_sar_hertzian_current_and_current_moment_normalisations_agree(tmp_path):
    dl = 0.002
    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.024, 0.024, 0.024)))
    scene.add(gprMax.Discretisation(p1=(dl, dl, dl)))
    scene.add(gprMax.TimeWindow(time=2e-9))
    scene.add(gprMax.PMLThickness(thickness=2))
    scene.add(gprMax.OMPThreads(1))
    scene.add(gprMax.Material(er=4, se=0.5, mr=1, sm=0, id="tissue"))
    scene.add(gprMax.MaterialDensity(density=1000, material_ids="tissue"))
    scene.add(
        gprMax.Box(
            p1=(0.004, 0.004, 0.004),
            p2=(0.020, 0.020, 0.020),
            material_id="tissue",
            tag="target",
        )
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=1e9, id="pulse"))
    scene.add(
        gprMax.HertzianDipole(p1=(0.012, 0.012, 0.012), polarisation="z", waveform_id="pulse")
    )
    current = gprMax.SAR(
        frequencies=(1e9,),
        waveform_id="pulse",
        tags="target",
        id="current",
        target_amplitude=3.0,
    )
    moment = gprMax.SAR(
        frequencies=(1e9,),
        waveform_id="pulse",
        tags="target",
        id="moment",
        target_amplitude=3.0 * dl,
        normalisation="current_moment",
    )
    scene.add(current)
    scene.add(moment)

    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=tmp_path / "hertzian_moment",
        hide_progress_bars=True,
    )

    np.testing.assert_allclose(moment.result.sar, current.result.sar, rtol=2e-6)
    assert moment._monitor._source_length == pytest.approx(dl)


@pytest.mark.integration
def test_sar_eigenmode_port_power_needs_no_waveform_id(tmp_path):
    from tests.test_virtual_waveguide_integration import _uniform_waveguide_scene

    scene = _uniform_waveguide_scene(normal_axis=0, direction="+")
    # Keep the perturbing test load weak enough that the compact 1 mm guide
    # remains above the solver's absolute three-cell propagation guard over
    # the continuous-sine spectral estimate.
    scene.add(gprMax.Material(er=1.0, se=1e-6, mr=1, sm=0, id="lossy"))
    scene.add(gprMax.MaterialDensity(density=1000, material_ids="lossy"))
    scene.add(
        gprMax.Box(
            p1=(0.030, 0.003, 0.003),
            p2=(0.034, 0.007, 0.009),
            material_id="lossy",
            tag="target",
        )
    )
    output = gprMax.SAR(
        frequencies=(22e9,),
        tags="target",
        id="modal_sar",
        normalisation="accepted_power",
        port_id="port1",
        target_power=1.0,
        spectrum_limit="nyquist",
    )
    scene.add(output)

    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=tmp_path / "eigenmode_sar",
        hide_progress_bars=True,
        log_level=30,
    )

    assert output.waveform_id is None
    assert output.result.valid[0]
    assert output.result.incident_power[0] > 0
    assert output.result.normalising_power[0] > 0
    assert np.all(np.isfinite(output.result.sar[0]))


def test_sar_state_is_reset_for_geometry_fixed_runs(tmp_path):
    scene, _ = _scene()
    output = tmp_path / "sar_reuse"
    gprMax.run(
        scenes=[scene],
        n=2,
        geometry_fixed=True,
        outputfile=output,
        hide_progress_bars=True,
    )

    results = []
    for index in (1, 2):
        with h5py.File(f"{output}{index}.h5", "r") as data:
            results.append(data["sar/target_sar/sar"][...])
    np.testing.assert_array_equal(results[0], results[1])


def test_sar_supports_dispersive_material_loss(tmp_path):
    scene, _ = _scene(dispersive=True)
    output = tmp_path / "sar_debye"
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=output,
        hide_progress_bars=True,
    )
    with h5py.File(str(output) + ".h5", "r") as data:
        absorbed = data["sar/target_sar/absorbed_power_density"][...]
        assert np.all(np.isfinite(absorbed))
        assert np.all(absorbed >= 0)


def test_sar_tagged_pec_integration_is_finite_and_zero(tmp_path):
    scene, output = _scene(spectrum_limit="nyquist")
    box = next(item for item in scene.geometry_objects if isinstance(item, gprMax.Box))
    box.kwargs["material_id"] = "pec"
    box.kwargs["p2"] = (0.008, 0.008, 0.008)
    scene.add(gprMax.MaterialDensity(density=7800, material_ids="pec"))

    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=tmp_path / "sar_pec",
        hide_progress_bars=True,
    )

    assert np.all(np.isfinite(output.result.absorbed_power_density))
    assert np.all(np.isfinite(output.result.sar))
    np.testing.assert_array_equal(output.result.absorbed_power_density, 0)
    np.testing.assert_array_equal(output.result.sar, 0)


def test_sar_rejects_selected_material_without_density(tmp_path):
    scene, _ = _scene()
    scene.grid_objects = [
        item for item in scene.grid_objects if not isinstance(item, gprMax.MaterialDensity)
    ]
    with pytest.raises(ValueError, match="requires finite positive mass density"):
        gprMax.run(
            scenes=[scene],
            n=1,
            outputfile=tmp_path / "missing_density",
            hide_progress_bars=True,
        )


def test_sar_excludes_tagged_boundary_pml_cells(tmp_path):
    scene, output = _scene()
    scene.grid_objects = [item for item in scene.grid_objects if not isinstance(item, gprMax.Box)]
    scene.add(
        gprMax.Box(
            p1=(0, 0, 0),
            p2=(0.024, 0.024, 0.024),
            material_id="tissue",
            tag="target",
        )
    )
    filename = tmp_path / "sar_pml_exclusion"
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=filename,
        hide_progress_bars=True,
    )

    # 12 cells per axis with two PML cells on each face leaves 8^3 cells.
    assert output.result.cell_indices.shape == (8**3, 3)
    assert np.all((output.result.cell_indices >= 2) & (output.result.cell_indices < 10))
    with h5py.File(str(filename) + ".h5", "r") as data:
        sar = data["sar/target_sar"]
        assert sar.attrs["PMLCellPolicy"] == "excluded"
        assert sar.attrs["ExcludedPMLCellCount"] == 12**3 - 8**3


def test_sar_writes_standard_spatial_average_groups(tmp_path):
    scene, _ = _scene(averaging_masses=(0.001, 0.01))
    filename = tmp_path / "sar_spatial_average"
    gprMax.run(
        scenes=[scene],
        n=1,
        outputfile=filename,
        hide_progress_bars=True,
    )
    with h5py.File(str(filename) + ".h5", "r") as data:
        group = data["sar/target_sar/spatial_average"]
        assert group.attrs["DensityModel"] == "constant within each tagged FDTD cell"
        assert "1g" in group
        assert "10g" in group
        assert np.all(np.isfinite(group["1g/peak_sar"][...]))
        # This test object has only 4.096 g of selected physical tissue.
        assert np.all(np.isnan(group["10g/peak_sar"][...]))


def test_sar_pml_mask_includes_internal_slabs():
    grid = SimpleNamespace(
        nx=8,
        ny=7,
        nz=6,
        pmls={
            "thickness": {
                "x0": 1,
                "xmax": 0,
                "y0": 0,
                "ymax": 1,
                "z0": 0,
                "zmax": 0,
            },
            "internal_specs": (SimpleNamespace(xs=3, xf=5, ys=2, yf=4, zs=1, zf=5),),
        },
    )

    mask = _pml_cell_mask(grid)

    assert np.all(mask[0, :, :])
    assert np.all(mask[:, -1, :])
    assert np.all(mask[3:5, 2:4, 1:5])
    assert not mask[2, 2, 2]
