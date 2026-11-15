"""Generate the native-symmetry guide figures and verify every drawn pattern.

Run from the repository root with::

    python -m testing.eigenmode_support.symmetry.illustrate_native_groups

Blue squares represent a scalar longitudinal-permittivity contrast on Yee
nodes, not mode fields. The runtime detector certifies each illustrated group
with uniform transverse tensors before any files are written.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

from gprMax.eigenmode_symmetry import detect_symmetry


BLUE = '#24759a'
ORANGE = '#c77918'
PURPLE = '#7553a0'
INK = '#263746'


def patterns():
    x, y = np.meshgrid(np.arange(-8, 9), np.arange(-8, 9), indexing='ij')
    arm = ((x >= 1) & (x <= 5) & (y >= 0) & (y <= 1)) | (
        (x >= 4) & (x <= 5) & (y >= 1) & (y <= 4))
    zed = ((abs(y) <= 1) & (abs(x) <= 5)) | (
        (x >= 3) & (x <= 5) & (y >= 1) & (y <= 4))
    def triangle(u, v, extent=5):
        return (v >= -extent+1) & (v <= extent) & (2*abs(u) <= extent-v)
    return {
        'c4v': (abs(x) <= 5) & (abs(y) <= 5) & ((abs(x) <= 2) | (abs(y) <= 2)),
        'c4': np.logical_or.reduce([np.rot90(arm, k) for k in range(4)]),
        'c2v': x*x/36 + y*y/9 <= 1,
        'c2v_diagonal': (x+y)**2/72 + (x-y)**2/18 <= 1,
        'c2': zed | zed[::-1, ::-1],
        'cs_u': triangle(x, y),
        'cs_v': triangle(y, x),
        'cs_diagonal': triangle(x-y, x+y, 6),
        'cs_antidiagonal': triangle(x+y, x-y, 6),
        'c1': ((x >= -4) & (x <= 4) & (y >= -4) & (y <= -2)) | (
            (x >= -4) & (x <= -2) & (y >= -4) & (y <= 5)),
    }


def certify(samples, expected):
    n = samples.shape[0]-1
    arrays = dict(eps_r_uu=np.ones((n, n+1)), eps_r_vv=np.ones((n+1, n)),
                  eps_r_ww=1.+samples, mu_r_uu=np.ones((n+1, n)),
                  mu_r_vv=np.ones((n, n+1)), mu_r_ww=np.ones((n, n)))
    for prefix, material in (('pec', 'eps_r'), ('pmc', 'mu_r')):
        for axis in 'uvw':
            arrays[f'{prefix}_{axis}_mask'] = np.zeros_like(arrays[f'{material}_{axis}{axis}'], dtype=bool)
    detected = detect_symmetry([SimpleNamespace(Nu=n, Nv=n, du=1., dv=1., surface_boundary=None, **arrays)])
    if detected['embedding'] != expected:
        raise ValueError(f'Illustration {expected} actually has {detected["embedding"]} symmetry.')


def save(figure, output, name):
    for extension in ('svg', 'png'):
        figure.savefig(output/f'{name}.{extension}', dpi=180, facecolor='white',
                       metadata={'Creator': 'gprMax native symmetry illustration', 'Date': None}
                       if extension == 'svg' else None)
        if extension == 'svg':
            path = output/f'{name}.{extension}'
            lines = path.read_text(encoding='utf-8').splitlines()
            path.write_bytes(('\n'.join(line.rstrip() for line in lines)+'\n').encode('utf-8'))
    plt.close(figure)


def group_figure(output):
    examples = patterns()
    for key, samples in examples.items():
        certify(samples, key)
    titles = ('C4v', 'C4', 'C2v · axis mirrors', 'C2v · diagonal mirrors', 'C2',
              'Cs · u-mirror', 'Cs · v-mirror', 'Cs · diagonal', 'Cs · antidiagonal', 'C1')
    subtitles = ('90° + four mirrors', '90°; no mirrors', '180° + two mirrors',
                 '180° + two mirrors', '180°; no mirrors',
                 'u → −u', 'v → −v', '(u, v) → (v, u)', '(u, v) → (−v, −u)', 'Identity only')
    mirrors = {'c4v': ('u', 'v', 'd', 'a'), 'c2v': ('u', 'v'), 'c2v_diagonal': ('d', 'a'),
               'cs_u': ('u',), 'cs_v': ('v',), 'cs_diagonal': ('d',), 'cs_antidiagonal': ('a',)}
    lines = {'u': ([0, 0], [-7, 7]), 'v': ([-7, 7], [0, 0]),
             'd': ([-7, 7], [-7, 7]), 'a': ([-7, 7], [7, -7])}
    fig, axes = plt.subplots(2, 5, figsize=(15.5, 7.4))
    fig.subplots_adjust(left=.025, right=.985, top=.85, bottom=.13, wspace=.16, hspace=.45)
    fig.suptitle('Native transverse symmetries on a square Yee grid', fontsize=19, y=.97, color=INK)
    fig.text(.5, .915, 'All groups preserve the propagation direction. Each example is checked by the symmetry detector.',
             ha='center', fontsize=11, color=INK)
    x, y = np.meshgrid(np.arange(-8, 9), np.arange(-8, 9), indexing='ij')
    for ax, (key, samples), title, subtitle in zip(axes.flat, examples.items(), titles, subtitles):
        ax.scatter(x, y, s=4, color='#d6dee3', zorder=0)
        ax.scatter(x[samples], y[samples], s=19, marker='s', color=BLUE, linewidths=0, zorder=1)
        for mirror in mirrors.get(key, ()):
            ax.plot(*lines[mirror], color=ORANGE, linewidth=1.6, linestyle=(0, (4, 3)), zorder=2)
        if key in ('c4v', 'c4', 'c2v', 'c2v_diagonal', 'c2'):
            angle = 90 if key in ('c4v', 'c4') else 180
            theta = np.deg2rad(np.linspace(15, 15+angle, 100))
            arc_x, arc_y = 7*np.cos(theta), 7*np.sin(theta)
            ax.plot(arc_x, arc_y, color=PURPLE, linewidth=1.7, zorder=3)
            ax.add_patch(FancyArrowPatch((arc_x[-5], arc_y[-5]), (arc_x[-1], arc_y[-1]),
                         arrowstyle='-|>', mutation_scale=12, color=PURPLE, zorder=3))
        ax.set(xlim=(-8.6, 8.6), ylim=(-8.6, 8.6), aspect='equal', xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set_edgecolor('#cbd5db')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10, color=INK)
        ax.text(.5, -.1, subtitle, transform=ax.transAxes, ha='center', va='top', fontsize=10, color=INK)
    handles = [Line2D([], [], marker='s', color=BLUE, linestyle='none', label='Sampled material contrast'),
               Line2D([], [], color=ORANGE, linestyle='--', label='Mirror line'),
               Line2D([], [], color=PURPLE, label='Smallest nonzero rotation')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .035), ncol=3, frameon=False, fontsize=11)
    fig.text(.5, .015, 'u points right; v points up. Diagonal mirrors and quarter turns require equal transverse spacing.',
             ha='center', fontsize=10, color=INK)
    save(fig, output, 'mode_tracking_native_symmetries')


def centre_figure(output):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))
    fig.subplots_adjust(left=.06, right=.98, top=.79, bottom=.19, wspace=.22)
    fig.suptitle('The active port centre can differ from the window centre', fontsize=17, y=.96, color=INK)
    titles = ('Closed guide: inactive exterior padding', 'Open structure: active air padding')
    notes = ('Use the enclosed guide centre for symmetry.',
             'The asymmetric truncation remains part of the problem.')
    for i, (ax, title, note) in enumerate(zip(axes, titles, notes)):
        ax.add_patch(Rectangle((0, 0), 17, 13, facecolor='#e0e5e8' if i == 0 else '#f5f8fa',
                               edgecolor=INK, linewidth=1.5))
        ax.add_patch(Rectangle((2, 3), 8, 8, facecolor='white' if i == 0 else '#c4dfe9',
                               edgecolor=INK if i == 0 else BLUE, linewidth=2))
        for x in range(3, 10):
            ax.plot([x, x], [3, 11], color='#e5eaf0', linewidth=.55)
        for y in range(4, 11):
            ax.plot([2, 10], [y, y], color='#e5eaf0', linewidth=.55)
        ax.scatter([6], [7], marker='+', s=110, linewidths=2.2, color=BLUE, zorder=4)
        ax.scatter([8.5], [6.5], marker='x', s=60, linewidths=1.8, color=ORANGE, zorder=4)
        ax.annotate('Guide centre' if i == 0 else 'Core centre', (6, 7), xytext=(3.6, 9.2),
                    color=BLUE, fontsize=10, arrowprops=dict(arrowstyle='-', color=BLUE))
        ax.annotate('Window\ncentre', (8.5, 6.5), xytext=(11.5, 8.5),
                    color=ORANGE, fontsize=10, arrowprops=dict(arrowstyle='-', color=ORANGE))
        ax.text(13.3, 3.4, 'Inactive\nPEC / SIBC\nexterior' if i == 0 else 'Active\nair',
                ha='center', va='center', fontsize=10, color=INK)
        ax.set(xlim=(-.5, 17.5), ylim=(-.5, 13.5), aspect='equal', xticks=[], yticks=[])
        ax.set_axis_off()
        ax.set_title(title, fontsize=12, fontweight='bold', color=INK, pad=10)
        ax.text(.5, -.09, note, transform=ax.transAxes, ha='center', va='top', fontsize=10, color=INK)
    fig.text(.5, .065, 'Only proven inactive padding is excluded. The solve and exported field-array shapes stay unchanged.',
             ha='center', fontsize=11, color=INK)
    save(fig, output, 'mode_tracking_port_centres')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('images_shared'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'svg.hashsalt': 'gprmax-native-symmetry'})
    group_figure(args.output)
    centre_figure(args.output)
    print(f'Certified all 10 group embeddings; wrote SVG and PNG figures to {args.output}.')


if __name__ == '__main__':
    main()
