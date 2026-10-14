"""Two persistent doublets cross in a uniformly anisotropic square PEC bore.

The sweep varies epsilon_z at fixed frequency, not a dispersive frequency
sweep. Independent continuum TE/TM solutions compare beta and all six fields.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, epsilon_0, mu_0

from gprMax.eigenmode_config import EigenmodeTrackingConfig
from gprMax.eigenmode_tracking import track_solver_bank
from gprMax.sources import EigenmodeSource
from testing.eigenmode_support.subspaces.expanded_mode_tracking import fd_solver
from testing.eigenmode_support.subspaces.general_subspace_tracking import stress_rotation
from testing.eigenmode_support.validate_fdfd_eigenmodes import _configured_solver_constants


WIDTH, FREQUENCY, EPSILON_T = .02, 20e9, 2.


def beta(family, m, n, epsilon_z):
    kc2 = (np.pi/WIDTH)**2*(m*m+n*n)
    return np.sqrt(EPSILON_T*(2*np.pi*FREQUENCY/c)**2-
                   kc2*(1 if family == 'TE' else EPSILON_T/epsilon_z))


def analytical_fields(family, m, n, epsilon_z, cells):
    k0, kc2 = 2*np.pi*FREQUENCY/c, (np.pi/WIDTH)**2*(m*m+n*n)
    propagation = beta(family, m, n, epsilon_z)
    components = []
    for component, (nu, nv, ox, oy) in enumerate(((cells, cells+1, .5, 0),
            (cells+1, cells, 0, .5), (cells+1, cells+1, 0, 0),
            (cells+1, cells, 0, .5), (cells, cells+1, .5, 0), (cells, cells, .5, .5))):
        x, y = (np.arange(nu)+ox)[:, None]*WIDTH/cells, (np.arange(nv)+oy)[None, :]*WIDTH/cells
        kx, ky = m*np.pi/WIDTH, n*np.pi/WIDTH
        if family == 'TE':
            psi = np.cos(kx*x)*np.cos(ky*y)
            dx, dy = -kx*np.sin(kx*x)*np.cos(ky*y), -ky*np.cos(kx*x)*np.sin(ky*y)
            value = (-1j*k0*dy/kc2, 1j*k0*dx/kc2, 0*psi,
                     -1j*propagation*dx/kc2, -1j*propagation*dy/kc2, psi)[component]
        else:
            psi = np.sin(kx*x)*np.sin(ky*y)
            dx, dy = kx*np.cos(kx*x)*np.sin(ky*y), ky*np.sin(kx*x)*np.cos(ky*y)
            value = (-1j*propagation*epsilon_z/EPSILON_T*dx/kc2,
                     -1j*propagation*epsilon_z/EPSILON_T*dy/kc2, psi,
                     1j*k0*epsilon_z*dy/kc2, -1j*k0*epsilon_z*dx/kc2, 0*psi)[component]
        components.append(np.asarray(value, complex))
    return components


def run(output, meshes=(64, 96)):
    output.mkdir(parents=True, exist_ok=True)
    discrete_crossings = [2*(1+(np.sin(np.pi/(2*n))/np.sin(np.pi/n))**2) for n in meshes]
    axis = np.sort(np.r_[np.linspace(2.3, 2.7, 21), 2.5, discrete_crossings])
    axis = np.unique(axis)
    definitions = [('TE', 2, 0), ('TE', 0, 2), ('TM', 1, 2), ('TM', 2, 1)]
    labels = ['TE20', 'TE02', 'TM12', 'TM21']
    z0 = np.sqrt(mu_0/epsilon_0)
    root = Path(__file__).resolve().parents[4]
    record = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  geometry='Square PEC bore uniformly filled with diag(2, 2, epsilon_z)',
                  width_m=WIDTH, frequency_Hz=FREQUENCY, epsilon_z=axis.tolist(),
                  analytical_crossing_epsilon_z=2.5, labels=labels, cases=[], source_sha256={})
    for p in [Path(__file__), root/'gprMax/eigenmode_tracking.py']:
        record['source_sha256'][p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    with _configured_solver_constants():
        for cells in meshes:
            print('SOLVE anisotropic square', cells, flush=True)
            bank = [fd_solver(FREQUENCY, (cells, cells), WIDTH/cells, (2, 2, ez), count=14) for ez in axis]
            crossing = 2*(1+(np.sin(np.pi/(2*cells))/np.sin(np.pi/cells))**2)
            exact_anchor = int(np.argmin(abs(axis-crossing)))
            stress_rotation(bank[exact_anchor], [4, 5, 6, 7])
            owner = EigenmodeSource(None)
            owner.tracking_config = EigenmodeTrackingConfig(extra_candidates=0)
            owner.normal_axis = 2; owner.invariant_axis = None; owner.port_index = 1
            owner.fallback_frequency = axis[0]; owner._tracking_impedance = z0
            owner.degenerate = (); owner.mode_polarizations = {}
            mapping = track_solver_bank(owner, axis, bank, 8)
            # At the low-epsilon anchor these branches occupy ranks 5--8.
            selected = [4, 5, 6, 7]
            assert (5, 6) in owner.degenerate and (7, 8) in owner.degenerate
            assert (5, 6, 7, 8) not in owner.degenerate
            assert set(mapping[0, 4:6]) == {4, 5} and set(mapping[-1, 4:6]) == {6, 7}
            numeric_beta = np.array([[s.complex_neff[j].real*2*np.pi*FREQUENCY/c for j in selected] for s in bank])
            reference_beta = np.array([[beta(*definition, ez) for definition in definitions] for ez in axis])
            scores, fitted_profiles, numeric_profiles = [], [], []
            for k, (s, ez) in enumerate(zip(bank, axis)):
                refs = [analytical_fields(*definition, ez, cells) for definition in definitions]
                nums = [[getattr(s, name)[..., j]*(1 if q < 3 else z0)
                         for q, name in enumerate(('Eu', 'Ev', 'Ew', 'Hu', 'Hv', 'Hw'))] for j in selected]
                a = np.column_stack([np.concatenate([v.ravel() for v in row]) for row in nums])
                b = np.column_stack([np.concatenate([v.ravel() for v in row]) for row in refs])
                row_scores, fitted = [], []
                for pair in ([0, 1], [2, 3]):
                    span = b[:, pair]; q = np.linalg.qr(span)[0]
                    for j in pair:
                        row_scores.append(float(np.linalg.norm(q.conj().T @ a[:, j])**2/np.linalg.norm(a[:, j])**2))
                        coef = np.linalg.lstsq(span, a[:, j], rcond=None)[0]
                        fitted.append([sum(v*refs[g][component] for v, g in zip(coef, pair)) for component in range(6)])
                scores.append(row_scores)
                if k in (0, len(axis)//2, len(axis)-1):
                    fitted_profiles.append(fitted); numeric_profiles.append(nums)
            case = dict(cells=cells, persistent_groups=owner.degenerate,
                        candidate_indices=mapping[:, selected].tolist(), beta_fdfd_per_m=numeric_beta.tolist(),
                        beta_analytical_per_m=reference_beta.tolist(), full_vector_subspace_overlap=scores,
                        maximum_relative_beta_error=float(np.max(abs(numeric_beta-reference_beta)/reference_beta)),
                        minimum_profile_overlap=float(np.min(scores)),
                        maximum_Maxwell_residual=float(np.max(owner.tracking_diagnostics['residuals'])),
                        numerical_crossing_epsilon_z=float(crossing),
                        exact_crossing_stress='Seeded complex unitary mixture of all four coincident modes',
                        exact_crossing_anchor=exact_anchor)
            assert case['maximum_relative_beta_error'] < .003, case
            assert case['minimum_profile_overlap'] > .999, case
            assert case['maximum_Maxwell_residual'] < 1e-8, case
            record['cases'].append(case)
            print('COMPLETE', cells, case['maximum_relative_beta_error'], case['minimum_profile_overlap'], flush=True)
            if cells == meshes[-1]:
                fig, axes = plt.subplots(3, 8, figsize=(20, 8), constrained_layout=True)
                for row in range(3):
                    for j in range(4):
                        component = 5 if j < 2 else 2
                        a, b = numeric_profiles[row][j][component], fitted_profiles[row][j][component]
                        phase = np.exp(-1j*np.angle(b.flat[np.argmax(abs(b))]))
                        a, b = a*phase, b*phase
                        a = a*np.exp(-1j*np.angle(np.vdot(b, a)))
                        for side, field in enumerate((a, b)):
                            ax = axes[row, 2*j+side]
                            im = ax.imshow((field/max(abs(field).max(), 1e-300)).real.T, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1, extent=(0, 20, 0, 20))
                            ax.set(title=f'{labels[j]} · {"FDFD" if side == 0 else "Analytical"}', xlabel='x (mm)', ylabel='y (mm)')
                    axes[row, 0].set_ylabel(f'εz = {axis[[0,len(axis)//2,-1][row]]:.2f}\ny (mm)')
                fig.colorbar(im, ax=axes, shrink=.7, label='Phase-aligned normalized longitudinal field')
                fig.suptitle('Persistent TE20/02 and TM12/21 doublets before, near, and after their crossing\nAnalytical fields oriented within each pair; FDFD fields unchanged')
                fig.savefig(output/'modal_profiles.png', dpi=140); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for pair, color, name in [([0, 1], 'tab:blue', 'TE20/02 pair'), ([2, 3], 'tab:orange', 'TM12/21 pair')]:
        axes[0].plot(axis, reference_beta[:, pair].mean(axis=1), color=color, label=f'{name}: analytical')
        for case in record['cases']:
            axes[0].plot(axis, np.array(case['beta_fdfd_per_m'])[:, pair].mean(axis=1), '.', color=color, ms=3 if case['cells']==meshes[0] else 5, label=f"{name}: {case['cells']}² FDFD")
    axes[0].axvline(2.5, color='.5', ls=':'); axes[0].set(xlabel='Axial relative permittivity εz', ylabel='Propagation constant β (rad/m)', title='Branch crossing with persistent degeneracy')
    for case in record['cases']:
        axes[1].semilogy(axis, np.max(abs(np.array(case['beta_fdfd_per_m'])-reference_beta)/reference_beta, axis=1), '.-', label=f"{case['cells']}²")
    axes[1].set(xlabel='Axial relative permittivity εz', ylabel='Maximum relative β error', title='Independent continuum comparison')
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(output/'propagation_constants.png', dpi=160); plt.close(fig)
    record['status'] = 'complete'
    (output/'results.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cells', nargs='+', type=int, default=[64, 96])
    args = parser.parse_args()
    run(args.output, tuple(args.cells))
