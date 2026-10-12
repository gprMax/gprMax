"""Plot saved tracking banks without rerunning or changing their provenance.

Run with one or more result directories as positional arguments. NPZ field
archives are local run products; JSON metrics and generated PNGs are retained.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_directory(directory):
    for report in sorted(directory.glob('*.json')):
        data = json.loads(report.read_text())
        if 'neff' not in data or 'frequency_GHz' not in data:
            continue
        neff = np.asarray(data['neff'])
        values = neff[..., 0] if neff.ndim == 3 else neff.real
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.plot(data['frequency_GHz'], values, '.-')
        ax.set(xlabel='Frequency (GHz)', ylabel='Real effective index',
               title=data.get('name', report.stem).replace('_', ' '))
        ax.grid(alpha=.25)
        fig.savefig(report.with_name(report.stem + '_tracks.png'), dpi=160)
        plt.close(fig)
        archive = report.with_name(report.stem + '_fields.npz')
        if not archive.exists():
            continue
        with np.load(archive) as fields:
            keys = [tuple(map(int, key.split('_')[1:])) for key in fields.files
                    if key.startswith('E_')]
            anchor = sorted({key[0] for key in keys})[len({key[0] for key in keys}) // 2]
            modes = sorted({key[1] for key in keys if key[0] == anchor})
            fig, axes = plt.subplots((len(modes) + 3) // 4, 4,
                                     figsize=(12, 3 * ((len(modes) + 3) // 4)),
                                     constrained_layout=True, squeeze=False)
            for ax, mode in zip(axes.flat, modes):
                components = [fields[f'E_{anchor}_{mode}_{c}'] for c in range(3)]
                component = max(range(3), key=lambda c: np.linalg.norm(components[c]))
                field = abs(components[component])
                scale = float(field.max())
                ax.imshow((field / scale if scale else field).T, origin='lower',
                          cmap='viridis', vmin=0, vmax=1)
                ax.set_title(f'Mode {mode + 1}: normalized |E{"xyz"[component]}|')
                ax.set(xlabel='Grid index u', ylabel='Grid index v')
            for ax in list(axes.flat)[len(modes):]:
                ax.set_visible(False)
            fig.suptitle(f'{report.stem}: anchor {anchor + 1}; each panel normalized separately')
            fig.savefig(report.with_name(report.stem + '_fields.png'), dpi=130)
            plt.close(fig)


def plot_persistent(report):
    data = json.loads(report.read_text())
    cases = [(name, case) for name, case in data['cases'].items() if 'candidate_indices' in case]
    fig, axes = plt.subplots(1, len(cases), figsize=(11, 4), constrained_layout=True)
    for ax, (name, case) in zip(np.atleast_1d(axes), cases):
        ax.plot(data['frequency_parameter'], np.asarray(case['candidate_indices']) + 1, '.-')
        ax.set(title=name.replace('_', ' '), xlabel='Sweep parameter', ylabel='Assigned candidate (1-based)')
        ax.grid(alpha=.25)
    fig.savefig(report.with_name('persistent_assignments.png'), dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    args = parser.parse_args()
    for directory in args.directories:
        plot_directory(directory)
        report = directory / 'results.json'
        if report.exists() and 'frequency_parameter' in json.loads(report.read_text()):
            plot_persistent(report)
