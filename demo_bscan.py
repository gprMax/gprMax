"""Generate a B-scan demo visualization."""

import os
import sys
import types
import importlib

# Mock gprMax modules
gprMax_mod = types.ModuleType('gprMax')
sys.modules['gprMax'] = gprMax_mod

exceptions_mod = types.ModuleType('gprMax.exceptions')
class CmdInputError(Exception):
    pass
exceptions_mod.CmdInputError = CmdInputError
sys.modules['gprMax.exceptions'] = exceptions_mod
gprMax_mod.exceptions = exceptions_mod

receivers_mod = types.ModuleType('gprMax.receivers')
class Rx:
    allowableoutputs = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 'Ix', 'Iy', 'Iz']
    defaultoutputs = allowableoutputs[:-3]
receivers_mod.Rx = Rx
sys.modules['gprMax.receivers'] = receivers_mod
gprMax_mod.receivers = receivers_mod

utilities_mod = types.ModuleType('gprMax.utilities')
sys.modules['gprMax.utilities'] = utilities_mod
gprMax_mod.utilities = utilities_mod

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo_output')
os.makedirs(OUT_DIR, exist_ok=True)


def create_bscan_data():
    """Create realistic B-scan data (hyperbolic reflections)."""
    n_traces = 80
    n_samples = 500
    dt = 1e-10
    fc = 800e6

    t = np.linspace(0, (n_samples - 1) * dt, n_samples)
    bscan = np.zeros((n_samples, n_traces))

    # Direct wave at ~1 ns
    t_direct = 1e-9
    for trace in range(n_traces):
        ricker = (1 - 2 * (np.pi * fc * (t - t_direct))**2) * \
                 np.exp(-(np.pi * fc * (t - t_direct))**2)
        bscan[:, trace] += ricker * 80

    # Hyperbolic reflector 1 (buried pipe at trace=30, depth ~10 ns)
    x_center1 = 30
    t_apex1 = 10e-9
    velocity = 0.1
    dx = 0.05
    for trace in range(n_traces):
        dist = abs(trace - x_center1) * dx
        t_ref = np.sqrt(t_apex1**2 + (2 * dist / (velocity * 1e9))**2 * 1e18)
        ricker = (1 - 2 * (np.pi * fc * (t - t_ref))**2) * \
                 np.exp(-(np.pi * fc * (t - t_ref))**2)
        attn = np.exp(-0.03 * abs(trace - x_center1))
        bscan[:, trace] += ricker * 40 * attn

    # Hyperbolic reflector 2 (trace=55, depth ~20 ns)
    x_center2 = 55
    t_apex2 = 20e-9
    for trace in range(n_traces):
        dist = abs(trace - x_center2) * dx
        t_ref = np.sqrt(t_apex2**2 + (2 * dist / (velocity * 1e9))**2 * 1e18)
        ricker = (1 - 2 * (np.pi * fc * (t - t_ref))**2) * \
                 np.exp(-(np.pi * fc * (t - t_ref))**2)
        attn = np.exp(-0.04 * abs(trace - x_center2))
        bscan[:, trace] += ricker * 25 * attn

    # Flat layer at ~35 ns with slight undulation
    t_layer = 35e-9
    for trace in range(n_traces):
        t_var = t_layer + 0.5e-9 * np.sin(2 * np.pi * trace / n_traces)
        ricker = (1 - 2 * (np.pi * fc * (t - t_var))**2) * \
                 np.exp(-(np.pi * fc * (t - t_var))**2)
        bscan[:, trace] += ricker * 20

    # Add noise
    bscan += np.random.normal(0, 1.5, bscan.shape)

    return bscan, dt


def render_bscan(filename, outputdata, dt, rxnumber, rxcomponent):
    """Render B-scan using the same logic as enhanced plot_Bscan.py."""
    basename = os.path.basename(filename)
    time_max_ns = outputdata.shape[0] * dt * 1e9

    fig = plt.figure(num=basename + ' - rx' + str(rxnumber),
                     figsize=(20, 10), facecolor='w', edgecolor='w')

    plt.imshow(outputdata,
               extent=[0, outputdata.shape[1], time_max_ns, 0],
               interpolation='nearest', aspect='auto', cmap='seismic',
               vmin=-np.amax(np.abs(outputdata)),
               vmax=np.amax(np.abs(outputdata)))

    plt.xlabel('Trace number')
    plt.ylabel('Time (ns)')
    plt.title('B-scan - {} (rx{}, {})'.format(basename, rxnumber, rxcomponent))

    ax = fig.gca()
    ax.grid(which='both', axis='both', linestyle='-.')

    cb = plt.colorbar()
    cb.set_label('Field strength [V/m]')

    plt.tight_layout()
    return plt


print("Generating B-scan demo...\n")

bscan_data, dt = create_bscan_data()

plt.close('all')
p = render_bscan('cylinder_Bscan_2D.out', bscan_data, dt, 1, 'Ex')
p.savefig(os.path.join(OUT_DIR, 'demo6_bscan.png'), dpi=150, bbox_inches='tight')
plt.close('all')
print("[1/1] B-scan view -> demo6_bscan.png")
print(f"\nSaved to: {os.path.abspath(OUT_DIR)}")
