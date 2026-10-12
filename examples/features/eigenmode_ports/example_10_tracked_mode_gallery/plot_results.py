"""Plot tracked modal banks; no excitation or scattering data is used."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle


def _centred(field, shape):
    for axis, count in enumerate(shape):
        if field.shape[axis] == count + 1:
            field = (np.take(field, range(count), axis=axis)
                     + np.take(field, range(1, count+1), axis=axis)) / 2
        elif field.shape[axis] != count:
            raise ValueError('Unexpected native Yee field shape')
    return field


def plot_case(path):
    path = Path(path)
    case = path.name.removesuffix('.modes.h5')
    with h5py.File(path, 'r') as output:
        group = output['ports/1']
        frequencies = group['frequencies'][:]
        modes = group['mode_indices'][:]
        neff = group['neff'][:]
        valid = group['anchor_mode_reference_valid'][:]
        if not np.all(valid) or not np.all(np.isfinite(neff)):
            raise ValueError(f'{case}: incomplete or nonfinite tracked mode bank')
        tracking = group['mode_tracking']
        if not np.all(tracking['numerical_valid'][:]) or tracking['unresolved_interval_modes'].size:
            raise ValueError(f'{case}: unresolved or numerically invalid tracking')
        symmetry = group.get('symmetry_detection')
        detection = symmetry.attrs['Detected'] if symmetry is not None else 'not applied'
        details = json.loads(symmetry.attrs['Details']) if symmetry is not None else {'groups': []}
        middle = int(np.argmin(abs(frequencies - (frequencies[0]+frequencies[-1])/2)))
        fields = {name: group[f'fields/{name}/values'][middle] for name in ('Ex','Ey','Ez','Hx','Hy','Hz')}
        shape = tuple(min(a.shape[dim+1] for a in fields.values()) for dim in (0,1))
        native = group['fields/Ex']
        u = native['u'][:]
        v = native['v'][:]
        if len(u) == shape[0]+1: u = (u[:-1]+u[1:])/2
        if len(v) == shape[1]+1: v = (v[:-1]+v[1:])/2
        residual = float(np.max(tracking['combined_residuals'][:]))
        groups = tracking.attrs['AutomaticDegenerateGroups']
    colours = [f'C{i%10}' for i in range(len(modes))]
    figure = Figure(figsize=(13,5), layout='constrained'); FigureCanvasAgg(figure)
    axes = figure.subplots(1,2)
    for j, mode in enumerate(modes):
        for ax, values in zip(axes, (neff.real, neff.imag)):
            ax.plot(frequencies/1e9, values[:,j], marker='.', color=colours[j],
                    linestyle='-' if j<10 else '--', label=f'Mode {mode}')
    for ax, label in zip(axes, ('Re effective index', 'Im effective index')):
        ax.set(xlabel='Frequency (GHz)', ylabel=label); ax.grid(alpha=.2)
    axes[0].legend(ncol=4, fontsize=8)
    figure.suptitle(f'{case.replace("_", " ").title()} — automatic tracking, {detection.upper()}')
    dispersion = path.with_name(f'{case}_dispersion.png')
    figure.savefig(dispersion, dpi=160)
    outputs = [dispersion.name]
    x, y = np.meshgrid(u*1e3, v*1e3, indexing='ij')
    centre = (u[0]+u[-1])*500
    phase = []
    for j in range(len(modes)):
        electric = np.concatenate([fields[f'E{a}'][j].ravel() for a in 'xyz'])
        phase.append(np.exp(-1j*np.angle(electric[np.argmax(abs(electric))])))
    for family in ('E','H'):
        rows = (len(modes)+3)//4
        figure = Figure(figsize=(14,3.3*rows), layout='constrained'); FigureCanvasAgg(figure)
        axes = figure.subplots(rows,4,squeeze=False)
        for j, ax in enumerate(axes.flat):
            if j >= len(modes): ax.set_visible(False); continue
            a, b, c = [_centred(fields[family+component][j], shape) for component in 'xyz']
            magnitude = np.sqrt(abs(a)**2+abs(b)**2+abs(c)**2)
            scale = max(float(np.max(magnitude)), np.finfo(float).tiny)
            ax.pcolormesh(x, y, magnitude/scale, cmap='viridis', vmin=0, vmax=1, shading='nearest')
            step = max(1, max(shape)//12)
            ix = (slice(None,None,step),slice(None,None,step))
            ax.quiver(x[ix], y[ix], (a*phase[j]).real[ix]/scale,
                      (b*phase[j]).real[ix]/scale, color='white', pivot='mid', scale=15)
            if case.endswith('circular'):
                boundary = Circle((centre,centre),10,fill=False,edgecolor='tomato',linewidth=1)
            else:
                height = 14 if case.endswith('rectangular') else 20
                boundary = Rectangle((centre-10,centre-height/2),20,height,fill=False,edgecolor='tomato',linewidth=1)
            ax.add_patch(boundary)
            ax.set(aspect='equal', xlabel='x (mm)', ylabel='y (mm)',
                   title=f'Mode {modes[j]} | Re n = {neff[middle,j].real:.4f}')
            ax.tick_params(labelsize=8)
        figure.suptitle(f'{case.replace("_", " ").title()} | {family} fields at {frequencies[middle]/1e9:g} GHz\n'
                       'Colour: magnitude / peak (0–1); arrows: transverse field at a common phase; red: guide boundary')
        image = path.with_name(f'{case}_{family}_modes.png')
        figure.savefig(image, dpi=140)
        outputs.append(image.name)
    return dict(case=case, mode_count=len(modes), anchor_count=len(frequencies),
                detected_symmetry=detection, symmetry_groups=details['groups'],
                automatic_groups=groups, maximum_residual=residual,
                minimum_real_neff=float(neff.real.min()), plots=outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('banks', type=Path, nargs='+', help='one or more .modes.h5 files')
    for bank in parser.parse_args().banks:
        print(json.dumps(plot_case(bank), indent=2))


if __name__ == '__main__':
    main()
