"""Render compact PEC comparisons from retained, independently checked runs.

JSON run provenance is preserved. Pass the original local NPZ field archive
to regenerate the representative profile pages; no new simulation is implied.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, epsilon_0, mu_0

from testing.eigenmode_support.profiles.waveguide_modal_profiles import analytical_fields
from testing.eigenmode_support.profiles import cylindrical_modal_profiles as cylinder


def run(directory, archive):
    record = json.loads((directory/'results.json').read_text())
    circular = 'radius_m' in record
    frequencies = np.array(record['frequency_Hz' if circular else 'frequencies_Hz'])
    final = record['cases'][-1]
    labels = final['labels']
    analytic_modes = cylinder.reference_modes() if circular else None
    z0 = np.sqrt(mu_0/epsilon_0)
    beta_cases = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for case in record['cases']:
        neff = np.array(case['neff'])[..., 0]
        numeric = neff*2*np.pi*frequencies[:, None]/c
        if circular:
            assignment = np.array(case['dominant_reference_indices'])
            if assignment.ndim == 1:
                assignment = np.tile(assignment, (len(frequencies), 1))
            cutoff = np.array([[analytic_modes[j][0]/record['radius_m'] for j in row] for row in assignment])
            profile_score = np.array(case['PEC_subspace_overlap'])
        else:
            indices = [tuple(map(int, label.split('_')[1])) for label in case['labels']]
            cutoff = np.array([np.pi*np.sqrt((m/record['width_m'])**2+(n/record['height_m'])**2) for m, n in indices])
            profile_score = np.array(case['full_vector_overlap'])
        theory = np.sqrt((2*np.pi*frequencies[:, None]/c)**2-cutoff**2)
        error = abs(numeric-theory)/theory
        beta_cases.append(dict(cells=case['cells'], beta_fdfd_per_m=numeric.tolist(),
                               beta_analytical_per_m=theory.tolist(), maximum_relative_beta_error=float(error.max())))
        x = np.arange(1, len(labels)+1)
        axes[0].plot(x, numeric[1], 'o', ms=3, label=f"FDFD {case['cells']} cells across window")
        axes[1].semilogy(x, np.maximum(error.max(axis=0), 1e-16), '.-', label=f"{case['cells']} cells")
        axes[2].semilogy(x, np.maximum(1-profile_score.min(axis=0), 1e-16), '.-', label=f"{case['cells']} cells")
    axes[0].plot(x, theory[1], 'k_', ms=9, label='Analytical')
    axes[0].set(title=f'{frequencies[1]/1e9:g} GHz', ylabel='Propagation constant β (rad/m)')
    axes[1].set(title='All frequency anchors', ylabel='Worst relative β error')
    axes[2].set(title='Full-vector modal profile', ylabel='Worst 1 − analytical overlap')
    for ax in axes:
        ax.set_xlabel('Tracked channel'); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.savefig(directory/'comparison.png', dpi=160); plt.close(fig)
    (directory/'propagation_constants.json').write_text(json.dumps(dict(
        description='Derived from the retained run; not a new simulation', frequency_Hz=frequencies.tolist(), cases=beta_cases), indent=2)+'\n')
    with np.load(archive) as data:
        count = min(16, len(labels))
        for page in range((count+7)//8):
            fig, axes = plt.subplots(4, 4, figsize=(12, 11), constrained_layout=True)
            for slot, j in enumerate(range(8*page, min(8*page+8, count))):
                if circular:
                    index = np.array(final['dominant_reference_indices'])
                    mode = analytic_modes[index[1, j] if index.ndim == 2 else index[j]]
                    e, h = cylinder.analytical_fields(mode, final['cells'], frequencies[1], c, z0)
                    family = mode[1]; reference = h[2] if family == 'TE' else e[2]
                    extent = (-12, 12, -12, 12)
                else:
                    family, indices = labels[j].split('_'); m, n = map(int, indices)
                    e, h = analytical_fields(family, m, n, final['cells'], frequencies[1], c, z0, record['height_m'])
                    reference = h[2] if family == 'TE' else e[2]
                    extent = (0, record['width_m']*1e3, 0, record['height_m']*1e3)
                numeric = data[f'{"H" if family == "TE" else "E"}_1_{j}_2']
                numeric = numeric*np.exp(-1j*np.angle(np.vdot(reference, numeric)))
                for side, field in enumerate((numeric, reference)):
                    ax = axes[slot//2, 2*(slot % 2)+side]
                    im = ax.imshow((field/max(abs(field).max(), 1e-300)).real.T, origin='lower', extent=extent, cmap='RdBu_r', vmin=-1, vmax=1)
                    ax.set(title=f'{j+1}: {labels[j]} · {"Tracked" if side == 0 else "Analytical"}', xlabel='x (mm)', ylabel='y (mm)')
            for ax in list(axes.flat)[2*min(8, count-8*page):]:
                ax.set_visible(False)
            fig.colorbar(im, ax=axes, shrink=.7, label='Phase-aligned normalized longitudinal field')
            fig.suptitle(f'{record["name"]} · {frequencies[1]/1e9:g} GHz\nRepresentative modes {page*8+1}–{min((page+1)*8,count)}; TE → Hz, TM → Ez')
            fig.savefig(directory/f'modal_profiles_{page+1}.png', dpi=140); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    run(args.directory, args.archive)
