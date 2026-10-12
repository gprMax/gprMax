"""C4v staircase circular PEC/copper modes versus smooth-cylinder Bessel fields."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv, jvp, jn_zeros, jnp_zeros

import gprMax.config as config
from gprMax.fdfd_eigenmode_solver.fdfd_2d_mode_solver import FDFD_2D_mode_solver
from gprMax.fdfd_eigenmode_solver.surface_impedance_operator import (
    BoundaryAmpereRow, BoundaryMagneticTerm, FDFDSurfaceBoundary,
    boundary_edge_relative_permittivity, evaluate_surface_ade)
from gprMax.impedance_surfaces import SurfaceImpedanceModel
from gprMax.surface_impedance_presets import fit_conductivity_surface_impedance
from testing.validation.eigenmode_tracking.identification.tracking_classification_edges import analyse
from testing.validation.validate_fdfd_eigenmodes import _configured_solver_constants
from testing.validation.eigenmode_tracking.profiles.waveguide_modal_profiles import vector, matching

ROOT = Path(__file__).resolve().parents[4]
WINDOW, RADIUS = .024, .01
FREQUENCIES = (39e9, 40e9, 41e9)
MODES = 21


def retained_masks(air):
    """Independent edge/face supports for a propagation-invariant voxel bore."""
    nu, nv = air.shape
    u = np.zeros((nu, nv+1), int)
    v = np.zeros((nu+1, nv), int)
    w = np.zeros((nu+1, nv+1), int)
    u[:, :-1] += air; u[:, 1:] += air
    v[:-1, :] += air; v[1:, :] += air
    for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)):
        w[i:i+nu, j:j+nv] += air
    return u, v, w


def voxel_surface_boundary(air, spacing, response):
    """Clipped Ampere contours on every air/metal face of a voxel bore.

    Each longitudinal dual-face quadrant contributes its retained area and
    magnetic circulation. Each air/metal transition adds one half-edge
    surface-current port. This includes convex and concave staircase corners.
    """
    nu, nv = air.shape
    u, v, w = retained_masks(air)
    d = spacing
    e0 = config.sim_config.em_consts['e0']
    rows = []

    def inside(i, j):
        return 0 <= i < nu and 0 <= j < nv and bool(air[i, j])

    def add(axis, index, area, lengths, terms):
        epsilon = boundary_edge_relative_permittivity(
            response=response, epsilon0=e0, retained_dual_area=area,
            electric_mass=e0*area, conductive_mass=0., port_lengths=lengths)
        rows.append(BoundaryAmpereRow(axis, index, area, epsilon,
                    tuple(BoundaryMagneticTerm(a, ij, weight)
                          for (a, ij), weight in terms.items() if weight)))

    for i, j in zip(*np.where(u == 1)):
        upper = inside(i, j)
        add(0, (i, j), d*d/2, (d,), {(2, (i, j if upper else j-1)): d if upper else -d})
    for i, j in zip(*np.where(v == 1)):
        right = inside(i, j)
        add(1, (i, j), d*d/2, (d,), {(2, (i if right else i-1, j)): -d if right else d})
    for i, j in zip(*np.where((w > 0) & (w < 4))):
        terms = {}
        for di in (-1, 0):
            for dj in (-1, 0):
                if not inside(i+di, j+dj):
                    continue
                for key, weight in (((1, (i+di, j)), d/2 if di == 0 else -d/2),
                                    ((0, (i, j+dj)), -d/2 if dj == 0 else d/2)):
                    terms[key] = terms.get(key, 0.) + weight
        ports = sum(inside(i-1, j+dj) != inside(i, j+dj) for dj in (-1, 0))
        ports += sum(inside(i+di, j-1) != inside(i+di, j) for di in (-1, 0))
        add(2, (i, j), w[i, j]*d*d/4, (d/2,)*ports, terms)
    return FDFDSurfaceBoundary.create(electric_retained=(u > 0, v > 0, w > 0),
                                      magnetic_retained=(v > 0, u > 0, air), rows=rows)


def circular_solver(frequency, cells, wall, count=MODES+6):
    d = WINDOW/cells
    x = (np.arange(cells)+.5-cells/2)*d
    air = x[:, None]**2+x[None, :]**2 < RADIUS**2
    u, v, w = retained_masks(air)
    kwargs = {}
    if wall == 'PEC':
        kwargs.update(pec_u_mask=u < 2, pec_v_mask=v < 2, pec_w_mask=w < 4)
    else:
        dt = .99*d/(config.sim_config.em_consts['c']*np.sqrt(3))
        fit = fit_conductivity_surface_impedance(5.96e7, 1e9, 120e9, 16)
        model = SurfaceImpedanceModel('copper', A=fit.A, B=fit.B, C=fit.C, D=fit.D)
        discrete = model.discretise(dt)
        response = evaluate_surface_ade(frequency_hz=frequency, dt=dt, F=discrete.F,
                                       G=discrete.G, L=discrete.L, Z0=discrete.Z0)
        kwargs['surface_boundary'] = voxel_surface_boundary(air, d, response)
    solver = FDFD_2D_mode_solver(frequency, d, d, count-1,
        np.ones(u.shape), np.ones(v.shape), np.ones(w.shape),
        np.ones(v.shape), np.ones(u.shape), np.ones(air.shape), **kwargs)
    solver.retain_tracking_operator = True
    solver.calculate_diagnostics = True
    solver.solve()
    if wall == 'copper':
        solver.edge_validation_metadata = dict(sigma_S_per_m=5.96e7, ade_dt=dt,
            fit_frequency_bounds_Hz=(1e9, 120e9), fit_max_relative_error=fit.max_relative_error,
            surface_rows=len(kwargs['surface_boundary'].rows), admittance=response.admittance)
    return solver


def reference_modes():
    modes = []
    for family, zeros in (('TE', jnp_zeros), ('TM', jn_zeros)):
        for m in range(7):
            for n, root in enumerate(zeros(m, 3), 1):
                for angular in (('cos', 'sin') if m else ('cos',)):
                    modes.append((float(root), family, m, n, angular))
    return sorted(modes)[:MODES]


def bessel_profile(m, root, angular, x, y):
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    kc = root/RADIUS
    cosine, sine = np.cos(m*theta), np.sin(m*theta)
    angle, derivative = (cosine, -m*sine) if angular == 'cos' else (sine, m*cosine)
    radial = jv(m, kc*r)
    psi = radial*angle
    dr = kc*jvp(m, kc*r)*angle
    dt = np.divide(radial*derivative, r, out=np.zeros_like(r), where=r > 0)
    dx = np.cos(theta)*dr-np.sin(theta)*dt
    dy = np.sin(theta)*dr+np.cos(theta)*dt
    dx = np.where(r == 0, kc/2 if m == 1 and angular == 'cos' else 0., dx)
    dy = np.where(r == 0, kc/2 if m == 1 and angular == 'sin' else 0., dy)
    return psi, dx, dy, r <= RADIUS


def analytical_fields(mode, cells, frequency, c, z0):
    root, family, m, _, angular = mode
    kc2, k0 = (root/RADIUS)**2, 2*np.pi*frequency/c
    beta = np.sqrt(k0*k0-kc2+0j)
    fields = []
    for component, (nx, ny, ox, oy) in enumerate(((cells, cells+1, .5, 0),
            (cells+1, cells, 0, .5), (cells+1, cells+1, 0, 0),
            (cells+1, cells, 0, .5), (cells, cells+1, .5, 0), (cells, cells, .5, .5))):
        x = (np.arange(nx)+ox-cells/2)[:, None]*WINDOW/cells
        y = (np.arange(ny)+oy-cells/2)[None, :]*WINDOW/cells
        psi, dx, dy, mask = bessel_profile(m, root, angular, x, y)
        if family == 'TE':
            value = (-1j*k0*z0*dy/kc2, 1j*k0*z0*dx/kc2, 0*psi,
                     -1j*beta*dx/kc2, -1j*beta*dy/kc2, psi)[component]
        else:
            value = (-1j*beta*dx/kc2, -1j*beta*dy/kc2, psi,
                     1j*k0/z0*dy/kc2, -1j*k0/z0*dx/kc2, 0*psi)[component]
        fields.append(value*mask)
    return fields[:3], fields[3:]


def mode_name(mode):
    _, family, m, n, angular = mode
    return f'{family}{m}{n}' + (f' {angular}' if m else '')


def run(output, wall, meshes=(96, 192)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    name = f'C4v_cylindrical_{wall}'
    destination = output/f'{name}_results.json'
    if destination.exists():
        raise ValueError('Choose a fresh output directory to preserve run provenance.')
    np.random.seed(20261012)
    record = dict(name=name, status='running', started_utc=datetime.now(timezone.utc).isoformat(),
        wall=wall, radius_m=RADIUS, window_m=WINDOW, meshes=list(meshes), requested_modes=MODES,
        candidate_modes=MODES+6, frequency_Hz=FREQUENCIES, seed=20261012, cases=[], source_sha256={})
    for relative in ('testing/validation/eigenmode_tracking/profiles/cylindrical_modal_profiles.py',
            'testing/validation/eigenmode_tracking/identification/tracking_classification_edges.py',
            'testing/validation/eigenmode_tracking/profiles/waveguide_modal_profiles.py',
            'gprMax/eigenmode_tracking.py', 'gprMax/eigenmode_symmetry.py',
            'gprMax/fdfd_eigenmode_solver/fdfd_2d_mode_solver.py',
            'gprMax/fdfd_eigenmode_solver/surface_impedance_operator.py',
            'gprMax/surface_impedance_presets.py'):
        record['source_sha256'][relative] = hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
    def save():
        destination.write_text(json.dumps(record, indent=2)+'\n')
    save()
    try:
        with _configured_solver_constants():
            c, z0 = (config.sim_config.em_consts[k] for k in ('c', 'z0'))
            modes = reference_modes()
            record['analytical_modes'] = [dict(root=m[0], label=mode_name(m)) for m in modes]
            for cells in meshes:
                print('SOLVE', name, cells, flush=True)
                bank = [circular_solver(f, cells, wall) for f in FREQUENCIES]
                report = analyse(f'{name}_{cells}', bank, MODES, output, 'circular')
                assert report['detected']['detected'] == 'c4v'
                assert report['minimum_power_eigenvalue'] > 0
                data = np.load(output/f'{name}_{cells}_fields.npz')
                count = len(report['anchors'][1])
                assert count == MODES, count
                scores, assignments, matrices, beta_error = [], [], [], []
                roots = np.array([m[0] for m in modes])
                for k, frequency in enumerate(FREQUENCIES):
                    analytical = np.column_stack([vector(*analytical_fields(m, cells, frequency, c, z0), z0) for m in modes])
                    numerical = np.column_stack([vector(
                        [data[f'E_{k}_{j}_{q}'] for q in range(3)],
                        [data[f'H_{k}_{j}_{q}'] for q in range(3)], z0) for j in range(count)])
                    matrix = matching(numerical, analytical)
                    # Analytical labels are diagnostics, never inputs to tracking.
                    dominant = matrix.argmax(axis=1)
                    family_scores = []
                    errors = []
                    for j, column in enumerate(dominant):
                        q = np.linalg.qr(analytical[:, np.isclose(roots, roots[column], rtol=1e-10)])[0]
                        family_scores.append(float(np.linalg.norm(q.conj().T @ numerical[:, j])**2 / np.linalg.norm(numerical[:, j])**2))
                        beta_reference = np.sqrt((2*np.pi*frequency/c)**2-(roots[column]/RADIUS)**2)
                        errors.append(float(abs(bank[k].complex_neff[j].real*2*np.pi*frequency/c-beta_reference)/beta_reference))
                    scores.append(family_scores); assignments.append(dominant.tolist())
                    matrices.append(matrix); beta_error.append(errors)
                assert assignments[0] == assignments[1] == assignments[2], assignments
                result = dict(cells=cells, spacing_m=WINDOW/cells, diameter_cells=2*RADIUS*cells/WINDOW,
                    dominant_reference_indices=assignments[1],
                    labels=[mode_name(modes[j]) for j in assignments[1]],
                    full_vector_matching=matrices[1].tolist(), PEC_subspace_overlap=scores,
                    minimum_PEC_subspace_overlap=float(np.min(scores)),
                    relative_beta_error=beta_error, maximum_relative_beta_error=float(np.max(beta_error)),
                    maximum_Maxwell_residual=report['max_field_residual'], groups=report['groups'],
                    neff=report['neff'])
                if wall == 'copper':
                    result['copper'] = report['copper']
                    attenuation = np.array([-s.complex_neff[:count].imag*2*np.pi*s.frequency/c for s in bank])
                    assert np.all(np.isfinite(attenuation)) and np.all(attenuation > 0)
                    result['attenuation_Np_per_m'] = attenuation.tolist()
                assert result['maximum_Maxwell_residual'] < 1e-7, result['maximum_Maxwell_residual']
                assert result['minimum_PEC_subspace_overlap'] > .9, result['minimum_PEC_subspace_overlap']
                record['cases'].append(result); save()
                if cells == meshes[-1]:
                    plot_profiles(data, modes, assignments[1], cells, c, z0, output, name)
                    plot_metrics(record, modes, output)
        record['status'] = 'complete'
    except Exception as exc:
        record.update(status='failed', error=str(exc))
        raise
    finally:
        save()
    print('COMPLETE', name, [(r['cells'], r['minimum_PEC_subspace_overlap']) for r in record['cases']], flush=True)


def plot_profiles(data, modes, assignment, cells, c, z0, output, name):
    for page in range((MODES+5)//6):
        fig, axes = plt.subplots(3, 4, figsize=(11, 9), layout='constrained')
        for slot, j in enumerate(range(page*6, min(page*6+6, MODES))):
            mode = modes[assignment[j]]
            ae, ah = analytical_fields(mode, cells, FREQUENCIES[1], c, z0)
            reference = ah[2] if mode[1] == 'TE' else ae[2]
            numeric = data[f'{"H" if mode[1] == "TE" else "E"}_1_{j}_2']
            numeric = numeric*np.exp(-1j*np.angle(np.vdot(reference, numeric)))
            for column, field, title in ((0, numeric.real/np.max(abs(numeric)), 'Tracked'),
                                         (1, reference.real/np.max(abs(reference)), 'PEC Bessel')):
                ax = axes[slot//2, 2*(slot%2)+column]
                im = ax.imshow(field.T, origin='lower', extent=(-12, 12, -12, 12),
                    cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
                ax.add_patch(plt.Circle((0, 0), 10, fill=False, color='.25', lw=.5))
                ax.set(title=f'{j+1}: {mode_name(mode)} · {title}', xlabel='x (mm)', ylabel='y (mm)',
                       xticks=[-10, 0, 10], yticks=[-10, 0, 10])
        for unused in range(2*min(6, MODES-page*6), 12):
            axes.flat[unused].set_visible(False)
        fig.colorbar(im, ax=axes, shrink=.7, label='Phase-aligned normalized longitudinal field')
        fig.suptitle(f'{name} · 40 GHz · {cells} × {cells} cells\nDominant PEC reference: TE → Hz; TM → Ez')
        fig.savefig(output/f'{name}_profiles_{page+1}.png', dpi=180)
        plt.close(fig)


def plot_metrics(record, modes, output):
    name = record['name']
    final = record['cases'][-1]
    fig, ax = plt.subplots(figsize=(10, 8), layout='constrained')
    im = ax.imshow(final['full_vector_matching'], vmin=0, vmax=1, cmap='viridis')
    ax.set_xticks(range(MODES), [mode_name(m) for m in modes], rotation=90)
    ax.set_yticks(range(MODES), [f'{j+1}: {s}' for j, s in enumerate(final['labels'])])
    ax.set(title=f'{name} · 40 GHz · full-vector squared overlap', xlabel='Smooth PEC Bessel reference', ylabel='Tracked channel')
    fig.colorbar(im, ax=ax, label='Squared normalized overlap')
    fig.savefig(output/f'{name}_matching.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for result in record['cases']:
        label = f"{result['cells']} × {result['cells']} ({result['spacing_m']*1e3:.3f} mm)"
        axes[0].semilogy(range(1, MODES+1), np.maximum(1-np.array(result['PEC_subspace_overlap']).min(axis=0), 1e-16), 'o-', label=label)
        axes[1].semilogy(range(1, MODES+1), np.array(result['relative_beta_error']).max(axis=0), 'o-', label=label)
    for ax in axes:
        ax.set(xlabel='Tracked channel', xticks=range(1, MODES+1, 2)); ax.grid(alpha=.25); ax.legend()
    axes[0].set(title='Worst smooth-PEC subspace mismatch', ylabel='1 − projection score')
    axes[1].set(title='Worst propagation-constant difference', ylabel='|Re β − βPEC| / βPEC')
    fig.suptitle(f'{name} · 39 / 40 / 41 GHz · staircase refinement')
    fig.savefig(output/f'{name}_convergence.png', dpi=180); plt.close(fig)
    if record['wall'] == 'copper':
        fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
        for result in record['cases']:
            ax.plot(range(1, MODES+1), np.array(result['attenuation_Np_per_m'])[1]*20/np.log(10), 'o-', label=f"{result['cells']} × {result['cells']}")
        ax.set(title=f'{name} · 40 GHz', xlabel='Tracked channel', ylabel='Attenuation (dB/m)')
        ax.legend(); ax.grid(alpha=.25)
        fig.savefig(output/f'{name}_attenuation.png', dpi=180); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wall', choices=('PEC', 'copper'), required=True)
    parser.add_argument('--cells', type=int, nargs='+', default=[96, 192])
    args = parser.parse_args()
    run(args.output, args.wall, tuple(args.cells))
