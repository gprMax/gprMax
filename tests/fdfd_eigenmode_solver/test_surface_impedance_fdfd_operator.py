"""Topology and exact integral-row tests for impedance-aware FDFD."""

from types import SimpleNamespace

import numpy as np
import pytest

import gprMax.config as config
from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver
from gprMax.fdfd_eigenmode_solver.surface_impedance_operator import (
    BoundaryAmpereRow,
    BoundaryMagneticTerm,
    FDFDSurfaceBoundary,
    boundary_edge_relative_permittivity,
    evaluate_surface_ade,
)
from gprMax.impedance_surfaces import SurfaceImpedanceModel
from gprMax.surface_impedance_presets import fit_metal_surface_impedance


@pytest.fixture(autouse=True)
def _solver_constants(monkeypatch):
    monkeypatch.setattr(
        config,
        "sim_config",
        SimpleNamespace(
            em_consts={
                "e0": 8.8541878128e-12,
                "m0": 1.25663706212e-6,
                "c": 299792458.0,
                "z0": 376.73031366686166,
            }
        ),
    )


def _solver_inputs(nu=2, nv=2):
    return dict(
        frequency=3.0e9,
        du=0.7e-3,
        dv=0.9e-3,
        mode_index=0,
        eps_r_uu=np.ones((nu, nv + 1)),
        eps_r_vv=np.ones((nu + 1, nv)),
        eps_r_ww=np.ones((nu + 1, nv + 1)),
        mu_r_uu=np.ones((nu + 1, nv)),
        mu_r_vv=np.ones((nu, nv + 1)),
        mu_r_ww=np.ones((nu, nv)),
    )


def _row_entries(matrix, row):
    values = matrix.getrow(row)
    return dict(zip(values.indices.tolist(), values.data.tolist()))


def _rectangular_impedance_boundary(nu, nv, du, dv, response):
    """Return exact clipped rows for a dielectric rectangle in metal."""
    epsilon0 = config.sim_config.em_consts["e0"]
    dw = min(du, dv)
    electric_retained = (
        np.ones((nu, nv + 1), dtype=bool),
        np.ones((nu + 1, nv), dtype=bool),
        np.ones((nu + 1, nv + 1), dtype=bool),
    )
    magnetic_retained = (
        np.ones((nu + 1, nv), dtype=bool),
        np.ones((nu, nv + 1), dtype=bool),
        np.ones((nu, nv), dtype=bool),
    )
    rows = []

    def relative_permittivity(area, port_lengths):
        return boundary_edge_relative_permittivity(
            response=response,
            epsilon0=epsilon0,
            retained_dual_area=area,
            electric_mass=epsilon0 * area,
            conductive_mass=0.0,
            port_lengths=port_lengths,
        )

    # Transverse edges have one clipped half-face.  Their longitudinal-H
    # phase term is implicit in P/Q; only the ordinary H_w line is supplied.
    transverse_area = 0.5 * dv * dw
    transverse_epsilon = relative_permittivity(transverse_area, (dw,))
    for i in range(nu):
        rows.append(
            BoundaryAmpereRow(
                0,
                (i, 0),
                transverse_area,
                transverse_epsilon,
                (BoundaryMagneticTerm(2, (i, 0), dw),),
            )
        )
        rows.append(
            BoundaryAmpereRow(
                0,
                (i, nv),
                transverse_area,
                transverse_epsilon,
                (BoundaryMagneticTerm(2, (i, nv - 1), -dw),),
            )
        )

    transverse_area = 0.5 * du * dw
    transverse_epsilon = relative_permittivity(transverse_area, (dw,))
    for j in range(nv):
        rows.append(
            BoundaryAmpereRow(
                1,
                (0, j),
                transverse_area,
                transverse_epsilon,
                (BoundaryMagneticTerm(2, (0, j), -dw),),
            )
        )
        rows.append(
            BoundaryAmpereRow(
                1,
                (nu, j),
                transverse_area,
                transverse_epsilon,
                (BoundaryMagneticTerm(2, (nu - 1, j), dw),),
            )
        )

    # A longitudinal edge owns one retained quadrant at a corner and two on
    # a flat wall.  Sum each quarter-cell circulation; duplicate H samples
    # merge in the solver to the ordinary one-sided Yee derivative.
    for i in range(nu + 1):
        for j in range(nv + 1):
            if i not in (0, nu) and j not in (0, nv):
                continue
            terms = {}
            retained_quadrants = 0
            for di in (-1, 0):
                for dj in (-1, 0):
                    cell_i, cell_j = i + di, j + dj
                    if not (0 <= cell_i < nu and 0 <= cell_j < nv):
                        continue
                    retained_quadrants += 1
                    if di == 0:
                        key = (1, (cell_i, j))
                        terms[key] = terms.get(key, 0.0) + 0.5 * dv
                    else:
                        key = (1, (cell_i, j))
                        terms[key] = terms.get(key, 0.0) - 0.5 * dv
                    if dj == 0:
                        key = (0, (i, cell_j))
                        terms[key] = terms.get(key, 0.0) - 0.5 * du
                    else:
                        key = (0, (i, cell_j))
                        terms[key] = terms.get(key, 0.0) + 0.5 * du

            port_lengths = []
            if i in (0, nu):
                port_lengths.extend([0.5 * dv] * ((j > 0) + (j < nv)))
            if j in (0, nv):
                port_lengths.extend([0.5 * du] * ((i > 0) + (i < nu)))
            area = retained_quadrants * du * dv / 4
            rows.append(
                BoundaryAmpereRow(
                    2,
                    (i, j),
                    area,
                    relative_permittivity(area, port_lengths),
                    tuple(
                        BoundaryMagneticTerm(axis, index, weight)
                        for (axis, index), weight in terms.items()
                    ),
                )
            )

    return FDFDSurfaceBoundary.create(
        electric_retained=electric_retained,
        magnetic_retained=magnetic_retained,
        rows=rows,
    )


def test_surface_topology_and_clipped_ampere_rows_are_installed_exactly():
    inputs = _solver_inputs()
    electric_retained = [
        np.ones((2, 3), dtype=bool),
        np.ones((3, 2), dtype=bool),
        np.ones((3, 3), dtype=bool),
    ]
    magnetic_retained = [
        np.ones((3, 2), dtype=bool),
        np.ones((2, 3), dtype=bool),
        np.ones((2, 2), dtype=bool),
    ]
    # These exclusions deliberately do not use PEC/PMC masks.  In particular,
    # excluding E_u must not also remove its collocated, interface-normal H_v.
    electric_retained[0][1, 2] = False
    electric_retained[2][2, 2] = False
    magnetic_retained[0][2, 1] = False
    magnetic_retained[2][1, 1] = False

    area_u = 1.3e-6
    area_v = 1.7e-6
    area_w = 2.1e-6
    boundary = FDFDSurfaceBoundary.create(
        electric_retained=electric_retained,
        magnetic_retained=magnetic_retained,
        rows=(
            BoundaryAmpereRow(
                0,
                (0, 0),
                area_u,
                2.0 - 0.3j,
                (
                    BoundaryMagneticTerm(2, (0, 0), 0.8e-3),
                    BoundaryMagneticTerm(2, (0, 0), 0.2e-3),
                ),
            ),
            BoundaryAmpereRow(
                1,
                (0, 0),
                area_v,
                2.5 - 0.4j,
                (BoundaryMagneticTerm(2, (0, 0), -0.6e-3),),
            ),
            BoundaryAmpereRow(
                2,
                (1, 1),
                area_w,
                3.0 - 0.5j,
                (
                    BoundaryMagneticTerm(0, (1, 0), 0.9e-3),
                    BoundaryMagneticTerm(1, (0, 1), -0.7e-3),
                ),
            ),
        ),
    )
    baseline = FDFD_2D_mode_solver(**inputs)
    solver = FDFD_2D_mode_solver(**inputs, surface_boundary=boundary)

    assert solver.eps_r_uu[0, 0] == 2.0 - 0.3j
    assert solver.eps_r_vv[0, 0] == 2.5 - 0.4j
    assert solver.eps_r_ww[1, 1] == 3.0 - 0.5j
    assert not solver.free_eu_mask[solver._flat_index(1, 2, solver.shape_eu[0])]
    assert solver.free_hv_mask[solver._flat_index(1, 2, solver.shape_hv[0])]
    assert not solver.free_ew_mask[solver._flat_index(2, 2, solver.shape_ew[0])]
    assert not solver.free_hu_mask[solver._flat_index(2, 1, solver.shape_hu[0])]
    assert not solver.free_hw_mask[solver._flat_index(1, 1, solver.shape_hw[0])]
    assert solver.surface_boundary_rows[0].magnetic_terms == (
        BoundaryMagneticTerm(2, (0, 0), 1.0e-3),
    )

    eu_row = solver._flat_index(0, 0, solver.shape_eu[0])
    ev_row = solver._flat_index(0, 0, solver.shape_ev[0])
    ew_row = solver._flat_index(1, 1, solver.shape_ew[0])
    hw_column = solver._flat_index(0, 0, solver.shape_hw[0])
    hu_column = solver._flat_index(1, 0, solver.shape_hu[0])
    hv_column = solver._flat_index(0, 1, solver.shape_hv[0])
    assert _row_entries(solver.DHV_HW_TO_HV, eu_row) == pytest.approx(
        {hw_column: 1.0e-3 / (area_u * solver.k0)}
    )
    assert _row_entries(solver.DHU_HW_TO_HU, ev_row) == pytest.approx(
        {hw_column: 0.6e-3 / (area_v * solver.k0)}
    )
    assert _row_entries(solver.DHV_HU_TO_EW, ew_row) == pytest.approx(
        {hu_column: -0.9e-3 / (area_w * solver.k0)}
    )
    assert _row_entries(solver.DHU_HV_TO_EW, ew_row) == pytest.approx(
        {hv_column: -0.7e-3 / (area_w * solver.k0)}
    )

    unaffected = solver._flat_index(1, 0, solver.shape_eu[0])
    assert _row_entries(solver.DHV_HW_TO_HV, unaffected) == pytest.approx(
        _row_entries(baseline.DHV_HW_TO_HV, unaffected)
    )


def test_surface_row_cannot_reference_excluded_magnetic_component():
    inputs = _solver_inputs()
    electric_retained = (
        np.ones((2, 3), dtype=bool),
        np.ones((3, 2), dtype=bool),
        np.ones((3, 3), dtype=bool),
    )
    magnetic_retained = (
        np.ones((3, 2), dtype=bool),
        np.ones((2, 3), dtype=bool),
        np.ones((2, 2), dtype=bool),
    )
    magnetic_retained[2][0, 0] = False
    boundary = FDFDSurfaceBoundary.create(
        electric_retained=electric_retained,
        magnetic_retained=magnetic_retained,
        rows=(
            BoundaryAmpereRow(
                0,
                (0, 0),
                1e-6,
                1 - 0.1j,
                (BoundaryMagneticTerm(2, (0, 0), 1e-3),),
            ),
        ),
    )

    with pytest.raises(ValueError, match="excluded magnetic"):
        FDFD_2D_mode_solver(**inputs, surface_boundary=boundary)


@pytest.mark.integration
def test_copper_exact_boundary_rows_match_rectangular_te10_attenuation():
    width = 22.86e-3
    height = 10.16e-3
    frequency = 10e9
    nu, nv = 60, 30
    du, dv = width / nu, height / nv
    constants = config.sim_config.em_consts
    dt = 0.99 / (
        constants["c"] * np.sqrt(du**-2 + dv**-2 + min(du, dv) ** -2)
    )
    fit = fit_metal_surface_impedance("copper", 1e8, 1e11, 16)
    model = SurfaceImpedanceModel(
        "copper",
        A=fit.A,
        B=fit.B,
        C=fit.C,
        D=fit.D,
        fit_fmin_hz=fit.fmin_hz,
        fit_fmax_hz=fit.fmax_hz,
    )
    discrete = model.discretise(dt)
    response = evaluate_surface_ade(
        frequency_hz=frequency,
        dt=dt,
        F=discrete.F,
        G=discrete.G,
        L=discrete.L,
        Z0=discrete.Z0,
    )
    boundary = _rectangular_impedance_boundary(nu, nv, du, dv, response)
    cutoff = constants["c"] / (2 * width)
    expected_neff = np.sqrt(1 - (cutoff / frequency) ** 2)
    solver = FDFD_2D_mode_solver(
        frequency=frequency,
        du=du,
        dv=dv,
        mode_index=0,
        eps_r_uu=np.ones((nu, nv + 1)),
        eps_r_vv=np.ones((nu + 1, nv)),
        eps_r_ww=np.ones((nu + 1, nv + 1)),
        mu_r_uu=np.ones((nu + 1, nv)),
        mu_r_vv=np.ones((nu, nv + 1)),
        mu_r_ww=np.ones((nu, nv)),
        guess=-(expected_neff**2),
        surface_boundary=boundary,
    )
    solver.solve()

    k0 = 2 * np.pi * frequency / constants["c"]
    beta = k0 * expected_neff
    cutoff_wavenumber = np.pi / width
    effective_impedance = 1 / response.admittance
    expected_alpha = effective_impedance.real / constants["z0"] * (
        k0 / (beta * height)
        + 2 * cutoff_wavenumber**2 / (k0 * beta * width)
    )
    calculated_alpha = -k0 * solver.modal_complex_neff.imag

    assert solver.modal_complex_neff.real == pytest.approx(expected_neff, rel=4e-4)
    assert calculated_alpha == pytest.approx(expected_alpha, rel=0.02)
    assert calculated_alpha > 0
