"""Plot saved analytical and crossing evidence without rerunning solvers."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    base = Path('testing/validation')
    crossing = json.loads((base/'circular_crossing_identity_20261024/results.json').read_text())['crossing']
    output = base/'circular_crossing_identity_20261024'
    f = np.array([r['frequency']/1e9 for r in crossing])
    mean = np.array([np.mean(list(r['analytical_neff'].values())) for r in crossing])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    colors = {39: '#1864ab', 40: '#d9480f', 41: '#868e96'}
    for channel in (39, 40, 41):
        y = [next(v['neff'][0] for v in r['channels'] if v['channel'] == channel) for r in crossing]
        axes[0].plot(f, (np.array(y)-mean)*1e3, '-o', ms=3, color=colors[channel], label=f'FDFD channel {channel}')
    for name, style in (('TE0,3', '--'), ('HE4,2', ':')):
        y = np.array([r['analytical_neff'][name] for r in crossing])
        axes[0].plot(f, (y-mean)*1e3, style, color='black', lw=1.6, label='Analytical '+name.replace(',', ''))
    axes[0].set(xlabel='Frequency (GHz)', ylabel=r'$10^3(n_{eff}-\overline{n}_{analytic})$', title='(a) Continuum crossing; grid avoided crossing')
    axes[0].legend(fontsize=8)
    for key, label, color in (('TE03', 'TE03 reference', '#1864ab'), ('HE42_sin', 'HE42 reference, A2 component', '#d9480f')):
        y = [next(v['overlaps'][key] for v in r['channels'] if v['channel'] == 39) for r in crossing]
        axes[1].plot(f, y, '-o', ms=4, color=color, label=label)
    axes[1].set(xlabel='Frequency (GHz)', ylabel='Squared full-vector field overlap', ylim=(0, 1.03), title='(b) Channel 39 exchanges modal character')
    axes[1].legend(fontsize=8)
    for name, color, marker in (('EH1,1', '#1864ab', 'o'), ('HE3,1', '#d9480f', 's')):
        values = []
        for cells in (40, 64, 96):
            r = json.loads((base/f'analytical_dielectric_identity_20261024/circular_dielectric_40GHz_{cells}.json').read_text())
            values.append(min(v['squared_field_overlap'] for v in r['channels'] if v['analytical'] == name))
        axes[2].plot([40, 64, 96], values, marker=marker, color=color, label=name.replace(',', ''))
    axes[2].set(xlabel='Cells across 40 mm window', ylabel='Squared full-vector field overlap', ylim=(.7, 1.01), xticks=[40, 64, 96], title='(c) Circular-reference agreement at 40 GHz')
    axes[2].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.22); ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(output/'identity_comparison.png', dpi=180)
    fig.savefig(output/'identity_comparison.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
