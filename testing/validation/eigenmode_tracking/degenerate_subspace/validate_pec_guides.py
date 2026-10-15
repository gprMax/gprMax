"""Reproduce compact PEC field/propagation comparisons without excitation."""

import argparse
from pathlib import Path
import shutil

from testing.eigenmode_support.profiles.waveguide_modal_profiles import run as rectangular
from testing.eigenmode_support.profiles.cylindrical_modal_profiles import run as cylindrical
from .plot_pec_comparison import run as plot


def run(output, scratch, case='all'):
    choices = {'rectangular': 'C2v_rectangle_2to1', 'square': 'C4v_square',
               'cylindrical': 'C4v_cylindrical'}
    selected = choices if case == 'all' else {case: choices[case]}
    for geometry, name in selected.items():
        name += '_PEC'
        raw, destination = scratch/name, output/name
        if destination.exists():
            raise ValueError(f'Choose a fresh output directory: {destination} already exists')
        raw.mkdir(parents=True, exist_ok=True)
        if geometry == 'cylindrical':
            cylindrical(raw, 'PEC')
            cells = 192
        else:
            rectangular(raw, choices[geometry], 'PEC')
            cells = 96
        destination.mkdir(parents=True)
        shutil.copy2(raw/f'{name}_results.json', destination/'results.json')
        plot(destination, raw/f'{name}_{cells}_fields.npz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['all', 'rectangular', 'square', 'cylindrical'], default='all')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.scratch, args.case)
