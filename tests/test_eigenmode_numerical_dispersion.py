"""Integration checks for Yee dispersion from material slices to modal ports."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
from gprMax.impedance_surfaces import SurfaceImpedanceModel
from gprMax.materials import DispersiveMaterial, Material
from gprMax.sources import EigenmodeSource


@pytest.fixture(autouse=True)
def _solver_constants(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            em_consts={
                "e0": config.e0,
                "m0": config.m0,
                "c": config.c,
                "z0": np.sqrt(config.m0 / config.e0),
            },
            dtypes={"float_or_double": np.float64, "complex": np.complex128},
            geometry_only=False,
        ),
    )


def _source_grid(domain="3D", normal_axis=0):
    """Small anisotropic Yee guide, with PEC assigned at actual E locations."""
    transverse_axes = tuple(axis for axis in range(3) if axis != normal_axis)
    u_axis, v_axis = transverse_axes
    nu, nv = 12, 8 if domain == "3D" else 1
    shape = np.ones(3, dtype=int)
    shape[normal_axis], shape[u_axis], shape[v_axis] = 3, nu + 1, nv + 1
    spacing = np.empty(3)
    spacing[normal_axis], spacing[u_axis], spacing[v_axis] = 1.8e-3, 0.8e-3, 1.1e-3
    dielectric = Material(0, "dielectric")
    dielectric.er = 4.0
    pec = Material(1, "pec")
    pec.se = np.inf
    ids = np.zeros((6, *shape), dtype=np.uint32)

    def pec_faces(component, axis):
        index = [slice(None)] * 4
        index[0] = component
        index[axis + 1] = [0, -1]
        ids[tuple(index)] = pec.numID

    pec_faces(normal_axis, u_axis)
    if domain != "TE":
        pec_faces(v_axis, u_axis)
    if domain == "3D":
        pec_faces(u_axis, v_axis)
        pec_faces(normal_axis, v_axis)
    grid = SimpleNamespace(
        materials=[dielectric, pec],
        ID=ids,
        solid=np.zeros(tuple(shape - 1), dtype=np.uint32),
        dl=spacing,
        dx=spacing[0],
        dy=spacing[1],
        dz=spacing[2],
        dt=1.5e-12,
        iterations=256,
        eigenmodeports=[],
    )
    source = EigenmodeSource(grid)
    source.normal_axis = normal_axis
    source.transverse_axes = transverse_axes
    source.transverse_start = np.array((0, 0), dtype=np.int32)
    source.transverse_stop = np.array((nu, nv), dtype=np.int32)
    source.plane_index = 1
    source.direction = "+"
    source.frequency = 20e9
    source.mode_index = source.mode_count = source.port_index = 1
    source.mode_indices = (1,)
    source.dft_start = source.dft_stop = source.frequency
    source.dft_points = 1
    source.plot_fields = source.plot_waveform = False
    if domain != "3D":
        source.invariant_axis = v_axis
        source.physical_transverse_axis = u_axis
        source.domain_polarization = domain
    return source, grid


@pytest.mark.parametrize("domain", ("3D", "TM", "TE"))
@pytest.mark.parametrize("normal_axis", (0, 2))
def test_grid_setup_uses_yee_phase_for_source_and_port(domain, normal_axis):
    source, grid = _source_grid(domain, normal_axis)

    source.grid_init(grid)

    omega = 2 * np.pi * source.frequency
    discrete_omega = 2 * np.sin(omega * grid.dt / 2) / grid.dt
    transverse_spacing = grid.dl[source.transverse_axes[0]]
    # TE has a constant H profile between the two PEC walls. The other
    # guides have the fundamental sine profile across 12 transverse cells.
    transverse_wavenumber = (
        0.0 if domain == "TE" else 2 * np.sin(np.pi / 24) / transverse_spacing
    )
    normal_spacing = grid.dl[normal_axis]
    discrete_beta = np.sqrt(4 * (discrete_omega / config.c) ** 2 - transverse_wavenumber**2)
    beta = 2 * np.arcsin(discrete_beta * normal_spacing / 2) / normal_spacing
    expected_neff = beta * config.c / omega

    solver = source.mode_solver
    assert solver.fdtd_dt == grid.dt
    assert solver.propagation_spacing == normal_spacing
    assert source.complex_neff == pytest.approx(expected_neff, rel=2e-10, abs=1e-11)
    assert source.port_anchor_neff[0, 0] == pytest.approx(expected_neff, rel=2e-10)
    assert omega * source._magnetic_modal_time_offset(grid) == pytest.approx(
        omega * grid.dt / 2 + beta * normal_spacing / 2,
        rel=2e-10,
    )

    monitor = source.port_monitor
    assert monitor.neff[0, 0] == pytest.approx(expected_neff, rel=2e-10)
    amplitude = 0.8 + 0.35j
    monitor.electric_dft[0] = monitor.electric_gram[0] @ np.array([amplitude])
    monitor.magnetic_dft[0] = monitor.magnetic_gram[0] @ np.array(
        [amplitude * np.exp(-1j * beta * normal_spacing / 2)]
    )
    result = monitor.finalise(grid)
    assert result.power_wave_valid[0, 0]
    assert result.incident[0, 0] == pytest.approx(amplitude, rel=2e-10)
    assert result.outgoing[0, 0] == pytest.approx(0.0, abs=2e-10)


@pytest.mark.parametrize("electric", (True, False))
@pytest.mark.parametrize("mpi", (False, True))
def test_conductive_material_slice_satisfies_harmonic_yee_update(monkeypatch, electric, mpi):
    source, grid = _source_grid()
    material = grid.materials[0]
    material.se = 0.12
    material.mr = 1.6
    material.sm = 2.5e4
    grid.ID.fill(material.numID)
    if mpi:
        # Exercise the actual MPI extraction and ownership loop without
        # requiring a separate process or an installed MPI launcher.
        monkeypatch.setitem(sys.modules, "mpi4py", SimpleNamespace(MPI=SimpleNamespace(SUM=object())))
        grid.global_size = grid.ID.shape[1:]
        grid.rank = 0
        grid.get_rank_from_coordinate = lambda coordinate: 0
        grid.global_to_local_coordinate = lambda coordinate: coordinate
        grid.comm = SimpleNamespace(Allreduce=lambda local, total, op: np.copyto(total, local))
        source.global_plane_index = source.plane_index
        source.global_transverse_start = source.transverse_start.copy()

    tensors = source._extract_local_complex_property_tensors(grid, electric=electric)
    theta = 2 * np.pi * source.frequency * grid.dt
    discrete_omega = 2 * np.sin(theta / 2) / grid.dt
    if electric:
        material.calculate_update_coeffsE(grid)
        feedback, coupling, vacuum_property = material.CA, material.srce, config.e0
    else:
        material.calculate_update_coeffsH(grid)
        feedback, coupling, vacuum_property = material.DA, material.srcm, config.m0
    # A unit harmonic field goes from exp(-j theta/2) to exp(+j theta/2)
    # in one step. Solve the implemented update for the driving curl.
    required_curl = (np.exp(0.5j * theta) - feedback * np.exp(-0.5j * theta)) / coupling
    for tensor in tensors:
        np.testing.assert_allclose(
            1j * discrete_omega * vacuum_property * tensor,
            required_curl,
            rtol=2e-14,
            atol=2e-14,
        )


@pytest.fixture(params=("debye", "lorentz", "drude", "inclusive"))
def dispersive_material(request):
    material = DispersiveMaterial(0, request.param)
    material.type = request.param
    material.er = 2.0
    material.poles = 1
    if request.param == "debye":
        material.deltaer = [4.0]
        material.tau = [8e-12]
    elif request.param == "lorentz":
        material.deltaer = [3.0]
        material.tau = [15e9]
        material.alpha = [8e9]
    elif request.param == "drude":
        material.tau = [25e9]
        material.alpha = [10e9]
    else:
        # Mixed Debye/Drude representation. Its constant conductivity must
        # stay paired with the analytic Drude pole in this partial fix.
        plasma_omega = 2 * np.pi * 25e9
        material.poles = 2
        material.inclusive_w = [4.0 / 8e-12, -(plasma_omega**2 / 10e9)]
        material.inclusive_q = [-1.0 / 8e-12, -10e9]
        material.inclusive_conductivity = config.e0 * plasma_omega**2 / 10e9
    return material


@pytest.mark.parametrize("mpi", (False, True))
def test_dispersive_static_conductivity_matches_harmonic_yee_increment(
    monkeypatch, dispersive_material, mpi
):
    source, grid = _source_grid()
    material = dispersive_material
    grid.materials[0] = material
    grid.ID.fill(material.numID)
    monkeypatch.setattr(
        config,
        "get_model_config",
        lambda: SimpleNamespace(
            materials={"maxpoles": material.poles, "dispersivedtype": np.complex128}
        ),
    )
    if mpi:
        monkeypatch.setitem(sys.modules, "mpi4py", SimpleNamespace(MPI=SimpleNamespace(SUM=object())))
        grid.global_size = grid.ID.shape[1:]
        grid.rank = 0
        grid.get_rank_from_coordinate = lambda coordinate: 0
        grid.global_to_local_coordinate = lambda coordinate: coordinate
        grid.comm = SimpleNamespace(Allreduce=lambda local, total, op: np.copyto(total, local))
        source.global_plane_index = source.plane_index
        source.global_transverse_start = source.transverse_start.copy()

    theta = 2 * np.pi * source.frequency * grid.dt
    discrete_omega = 2 * np.sin(theta / 2) / grid.dt

    def required_curl():
        material.calculate_update_coeffsE(grid)
        return (
            np.exp(0.5j * theta) - material.CA * np.exp(-0.5j * theta)
        ) / material.srce

    baseline = source._extract_local_complex_property_tensors(grid, electric=True)
    # Pole and inclusive-conductivity responses remain physical-frequency
    # values; only the independent static conductivity is compensated.
    for tensor in baseline:
        np.testing.assert_allclose(tensor, material.calculate_er(source.frequency), rtol=1e-14)
    baseline_curl = required_curl()
    material.se = 0.2
    conductive = source._extract_local_complex_property_tensors(grid, electric=True)
    # Pole-state feedback is unchanged by se, so subtracting the two updates
    # isolates the exact static-conductivity contribution to the curl.
    curl_increment = required_curl() - baseline_curl
    for original, updated in zip(baseline, conductive):
        np.testing.assert_allclose(
            1j * discrete_omega * config.e0 * (updated - original),
            curl_increment,
            rtol=2e-13,
            atol=2e-14,
        )
    assert material.se == 0.2


def test_dispersive_response_without_timestep_is_unchanged(dispersive_material):
    source, _ = _source_grid()
    dispersive_material.se = 0.2

    assert source._complex_er(dispersive_material) == dispersive_material.calculate_er(source.frequency)


@pytest.mark.parametrize("fdtd_dt", (None, 1.5e-12))
def test_dispersive_pec_is_masked_before_sampling(monkeypatch, dispersive_material, fdtd_dt):
    source, _ = _source_grid()
    dispersive_material.se = np.inf

    def unexpected_sampling(frequency):
        raise AssertionError("PEC must not evaluate an infinite conductive response")

    monkeypatch.setattr(dispersive_material, "calculate_er", unexpected_sampling)

    assert source._complex_er(dispersive_material, fdtd_dt) == source.FDFD_PEC_PROPERTY


def test_source_surface_row_uses_same_discrete_frequency_as_bulk_curls():
    source, grid = _source_grid()
    epsilon0 = config.e0
    area = 0.5 * grid.dy * grid.dz
    electric_mass = epsilon0 * 2.5 * area
    conductive_mass = 0.03 * area
    length, resistance = 0.5 * grid.dy, 50.0
    a_plus = electric_mass / grid.dt + conductive_mass / 2
    a_minus = electric_mass / grid.dt - conductive_mass / 2
    grid.surface_impedance_models = {"wall": SurfaceImpedanceModel("wall", D=resistance)}
    grid.impedance_surfaces = SimpleNamespace(
        model_ids=("wall",),
        edge_info=np.array([[0, 1, 2, 2, 0, 2, 0, 1]], dtype=np.int32),
        edge_fraction=np.array([0.5]),
        edge_params=np.array([[a_plus, a_minus]]),
        port_normal=np.array([[1]], dtype=np.int32),
        port_info=np.array([[0]], dtype=np.int32),
        port_g=np.array([-length]),
        h_info=np.array([[1, 1, 2, 2], [2, 1, 2, 2]], dtype=np.int32),
        h_weight=np.array([0.5 * grid.dy, -0.5 * grid.dz]),
    )

    boundary = source._build_surface_impedance_fdfd_boundary(grid)

    assert len(boundary.rows) == 1
    theta = 2 * np.pi * source.frequency * grid.dt
    discrete_omega = 2 * np.sin(theta / 2) / grid.dt
    required_curl = (
        (a_plus * np.exp(1j * theta) - a_minus) * np.exp(-0.5j * theta)
        + length * np.cos(theta / 2) / resistance
    )
    represented_curl = 1j * discrete_omega * epsilon0 * area * boundary.rows[0].relative_permittivity
    assert represented_curl == pytest.approx(required_curl, rel=2e-14, abs=1e-16)
