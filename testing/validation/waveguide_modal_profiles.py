"""Undriven C4v square and C2v 2:1 rectangle, PEC and copper: tracked modes versus independently sampled analytical fields.

Run: python -m testing.validation.waveguide_modal_profiles --case C4v_square --wall PEC --output DIR
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

import gprMax.config as config
from testing.validation.expanded_mode_tracking import fd_solver
from testing.validation.tracking_classification_edges import analyse, copper_solver
from testing.validation.validate_fdfd_eigenmodes import _configured_solver_constants

ROOT = Path(__file__).resolve().parents[2]
WIDTH = .02
CASES = {'C4v_square': (.02, 18, (39e9, 40e9, 41e9)),
         'C2v_rectangle_2to1': (.01, 35, (98e9, 100e9, 102e9))}


def analytical_fields(family, m, n, cells, frequency, c, z0, height=WIDTH):
    """Continuum +z fields, exp(+jwt), sampled at each native Yee location."""
    kx, ky = m*np.pi/WIDTH, n*np.pi/height
    kc2 = kx*kx + ky*ky
    k0 = 2*np.pi*frequency/c
    beta = np.sqrt(k0*k0-kc2+0j)
    nv = round(cells*height/WIDTH)
    shapes = ((cells, nv+1), (cells+1, nv), (cells+1, nv+1),
              (cells+1, nv), (cells, nv+1), (cells, nv))
    offsets = ((.5, 0), (0, .5), (0, 0), (0, .5), (.5, 0), (.5, .5))
    fields = []
    for component, (shape, offset) in enumerate(zip(shapes, offsets)):
        x = (np.arange(shape[0])+offset[0])[:, None]*WIDTH/cells
        y = (np.arange(shape[1])+offset[1])[None, :]*WIDTH/cells
        if family == 'TE':
            longitudinal = np.cos(kx*x)*np.cos(ky*y)
            dx = -kx*np.sin(kx*x)*np.cos(ky*y)
            dy = -ky*np.cos(kx*x)*np.sin(ky*y)
            value = (-1j*k0*z0*dy/kc2, 1j*k0*z0*dx/kc2,
                     0*longitudinal, -1j*beta*dx/kc2, -1j*beta*dy/kc2,
                     longitudinal)[component]
        else:
            longitudinal = np.sin(kx*x)*np.sin(ky*y)
            dx = kx*np.cos(kx*x)*np.sin(ky*y)
            dy = ky*np.sin(kx*x)*np.cos(ky*y)
            value = (-1j*beta*dx/kc2, -1j*beta*dy/kc2, longitudinal,
                     1j*k0/z0*dy/kc2, -1j*k0/z0*dx/kc2,
                     0*longitudinal)[component]
        fields.append(np.asarray(value, complex))
    return fields[:3], fields[3:]


def vector(e, h, z0):
    return np.concatenate([a.ravel() for a in e]+[(z0*a).ravel() for a in h])


def matching(numerical, analytical):
    a = numerical/np.linalg.norm(numerical, axis=0)
    b = analytical/np.linalg.norm(analytical, axis=0)
    return abs(a.conj().T @ b)**2


def run(output, name, wall='PEC'):
    height, count, frequencies = CASES[name]
    name = f'{name}_{wall}'
    np.random.seed(20261012)
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(started_utc=datetime.now(timezone.utc).isoformat(), status='running',
                   name=name, wall=wall, seed=20261012, width_m=WIDTH, height_m=height, frequencies_Hz=frequencies, requested_modes=count,
                   candidate_modes=count+6, meshes=[64, 96], cases=[], source_sha256={})
    for relative in ('testing/validation/waveguide_modal_profiles.py',
                     'testing/validation/tracking_classification_edges.py',
                     'testing/validation/expanded_mode_tracking.py',
                     'gprMax/eigenmode_tracking.py', 'gprMax/eigenmode_symmetry.py',
                     'gprMax/fdfd_eigenmode_solver/fdfd_2d_mode_solver.py',
                     'testing/validation/modal_field_templates.py',
                     'tests/fdfd_eigenmode_solver/test_surface_impedance_fdfd_operator.py',
                     'gprMax/surface_impedance_presets.py'):
        summary['source_sha256'][relative] = hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
    destination = output/f'{name}_results.json'
    destination.write_text(json.dumps(summary, indent=2)+'\n')
    with _configured_solver_constants():
        c, z0 = (config.sim_config.em_consts[k] for k in ('c', 'z0'))
        for cells in summary['meshes']:
            nv = round(cells*height/WIDTH)
            bank = [(copper_solver(f, (cells, nv), count=count+6, fit_max_frequency=120e9)
                     if wall == 'copper' else
                     fd_solver(f, (cells, nv), WIDTH/cells, (1, 1, 1), count=count+6))
                    for f in frequencies]
            report = analyse(f'{name}_{cells}', bank, count, output, 'rectangular')
            data = np.load(output/f'{name}_{cells}_fields.npz')
            labels = [r['templates'][0]['name'] for r in report['anchors'][1]]
            assert report['detected']['detected'] == name.split('_')[0].lower()
            assert report['minimum_power_eigenvalue'] > 0
            assert len(labels) == count and len(set(labels)) == count, labels
            assert all([r['templates'][0]['name'] for r in row] == labels for row in report['anchors'])
            scores, matrices, subspace_scores = [], [], []
            for k, frequency in enumerate(frequencies):
                numerical, analytical = [], []
                for j, label in enumerate(labels):
                    family, indices = label.split('_')
                    m, n = map(int, indices)
                    e = [data[f'E_{k}_{j}_{q}'] for q in range(3)]
                    h = [data[f'H_{k}_{j}_{q}'] for q in range(3)]
                    ae, ah = analytical_fields(family, m, n, cells, frequency, c, z0, height)
                    numerical.append(vector(e, h, z0))
                    analytical.append(vector(ae, ah, z0))
                numerical = np.column_stack(numerical)
                analytical = np.column_stack(analytical)
                matrix = matching(numerical, analytical)
                cutoff_keys = [int(label.split('_')[1][0])**2 +
                               (WIDTH/height*int(label.split('_')[1][1]))**2 for label in labels]
                projected = []
                for j, key in enumerate(cutoff_keys):
                    q = np.linalg.qr(analytical[:, np.isclose(cutoff_keys, key)])[0]
                    projected.append(float(np.linalg.norm(q.conj().T @ numerical[:, j])**2 /
                                           np.linalg.norm(numerical[:, j])**2))
                subspace_scores.append(projected)
                matrices.append(matrix)
                scores.append(matrix.diagonal().tolist())
            case = dict(cells=cells, shape=[cells, nv], labels=labels, full_vector_overlap=scores,
                        PEC_subspace_overlap=subspace_scores,
                        minimum_PEC_subspace_overlap=float(np.min(subspace_scores)),
                        minimum_overlap=float(np.min(scores)),
                        maximum_off_diagonal=float(max(np.max(a-np.diag(a.diagonal())) for a in matrices)),
                        maximum_Maxwell_residual=report['max_field_residual'],
                        minimum_longitudinal_overlap=min(r['templates'][0]['score'] for row in report['anchors'] for r in row),
                        groups=report['groups'])
            case['neff'] = report['neff']
            if wall == 'copper':
                case['copper'] = report['copper']
                attenuation = np.array([-s.complex_neff[:count].imag*2*np.pi*s.frequency/c
                                        for s in bank])
                case['attenuation_Np_per_m'] = attenuation.tolist()
                assert np.all(np.isfinite(attenuation)) and np.all(attenuation > 0), attenuation
            summary['cases'].append(case)
            assert case['minimum_PEC_subspace_overlap'] > .999, case
            if wall == 'PEC':
                assert case['minimum_overlap'] > .999, case
            assert case['maximum_Maxwell_residual'] < (1e-8 if wall == 'copper' else 1e-9), case
            if cells == 96:
                plot_fields(data, labels, cells, c, z0, output, name, height, frequencies[1])
                plot_matching(matrices[1], summary['cases'], labels, output, name, frequencies)
                if wall == 'copper':
                    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
                    for result in summary['cases']:
                        ax.plot(range(1, count+1), np.array(result['attenuation_Np_per_m'])[1]*20/np.log(10),
                                'o-', label=f"{result['shape'][0]} × {result['shape'][1]} cells")
                    ax.set(xlabel='Tracked channel', ylabel='Attenuation (dB/m)',
                           title=f'{name} · {frequencies[1]/1e9:g} GHz · copper 5.96e7 S/m')
                    ax.grid(alpha=.25); ax.legend()
                    fig.savefig(output/f'{name}_attenuation.png', dpi=180)
                    plt.close(fig)
    summary['status'] = 'complete'
    destination.write_text(json.dumps(summary, indent=2)+'\n')
    print('COMPLETE', [(r['cells'], r['minimum_overlap']) for r in summary['cases']], flush=True)


def plot_fields(data, labels, cells, c, z0, output, name, height, frequency):
    # Three pages keep each of the 18 numerical/reference pairs readable.
    for page in range((len(labels)+5)//6):
        fig, axes = plt.subplots(3, 4, figsize=(11, 9), layout='constrained')
        for slot, j in enumerate(range(page*6, min(page*6+6, len(labels)))):
            family, indices = labels[j].split('_')
            ae, ah = analytical_fields(family, *map(int, indices), cells, frequency, c, z0, height)
            reference = ah[2] if family == 'TE' else ae[2]
            numerical = data[f'{"H" if family == "TE" else "E"}_1_{j}_2']
            numerical = numerical*np.exp(-1j*np.angle(np.vdot(reference, numerical)))
            numerical = numerical.real/np.max(abs(numerical))
            reference = reference.real/np.max(abs(reference))
            for column, field, title in ((0, numerical, 'Tracked'), (1, reference, 'PEC theory')):
                ax = axes[slot//2, 2*(slot%2)+column]
                im = ax.imshow(field.T, origin='lower', extent=(0, 20, 0, height*1e3),
                               cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
                ax.set(title=f'{j+1}: {family}{indices} — {title}', xlabel='x (mm)', ylabel='y (mm)')
                ax.set_xticks([0, 10, 20]); ax.set_yticks([0, height*500, height*1000])
        for unused in range(2*min(6, len(labels)-page*6), 12):
            axes.flat[unused].set_visible(False)
        fig.colorbar(im, ax=axes, shrink=.7, label='Phase-aligned normalized longitudinal field')
        fig.suptitle(f'{name} · {frequency/1e9:g} GHz · {cells} × {round(cells*height/WIDTH)} cells · modes {page*6+1}–{min(page*6+6,len(labels))}\n'
                     'Dominant PEC component: TE → Hz; TM → Ez. Copper modes may mix TE and TM.')
        fig.savefig(output/f'{name}_profiles_{page+1}.png', dpi=180)
        plt.close(fig)


def plot_matching(matrix, cases, labels, output, name, frequencies):
    count = len(labels)
    fig, (ax, error) = plt.subplots(1, 2, figsize=(16, 9), layout='constrained')
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis')
    names = [s.replace('_', '') for s in labels]
    ax.set_xticks(range(count), names, rotation=90)
    ax.set_yticks(range(count), [f'{j+1}: {s}' for j, s in enumerate(names)])
    ax.set(xlabel='PEC analytical reference field', ylabel='Tracked channel', title=f'{name} · full-vector squared overlap · {frequencies[1]/1e9:g} GHz')
    fig.colorbar(im, ax=ax, label='Squared normalized overlap')
    for case in cases:
        mismatch = 1-np.array(case['PEC_subspace_overlap' if name.endswith('copper') else 'full_vector_overlap'])
        error.semilogy(range(1, count+1), np.maximum(mismatch.max(axis=0), 1e-16), 'o-',
                       label=f"{case['shape'][0]} × {case['shape'][1]} cells")
    error.set(xlabel='Tracked channel', ylabel='Worst 1 − overlap across the three frequency anchors',
              title='PEC degenerate-subspace mismatch' if name.endswith('copper') else 'PEC individual-profile mismatch',
              xticks=range(1, count+1, 2 if count > 20 else 1))
    error.grid(alpha=.25); error.legend()
    fig.savefig(output/f'{name}_matching.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--wall', choices=('PEC', 'copper'), default='PEC')
    args = parser.parse_args()
    try:
        run(args.output, args.case, args.wall)
    except Exception as exc:
        result = args.output/f'{args.case}_{args.wall}_results.json'
        if result.exists():
            record = json.loads(result.read_text())
            record.update(status='failed', error=str(exc))
            result.write_text(json.dumps(record, indent=2)+'\n')
        raise
