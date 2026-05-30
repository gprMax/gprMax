"""Generate demo A-scan visualizations to showcase enhancements."""

import os
import sys
import types

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

import numpy as np
utilities_mod = types.ModuleType('gprMax.utilities')
def fft_power(signal, dt):
    N = len(signal)
    freqs = np.fft.rfftfreq(N, d=dt)
    fft_vals = np.fft.rfft(signal)
    power = 20 * np.log10(np.abs(fft_vals) / np.max(np.abs(fft_vals)) + 1e-30)
    return freqs, power
utilities_mod.fft_power = fft_power
sys.modules['gprMax.utilities'] = utilities_mod
gprMax_mod.utilities = utilities_mod

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tools.plot_Ascan import mpl_plot

# Output directory
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo_output')
os.makedirs(OUT_DIR, exist_ok=True)


def create_realistic_mock(filepath, nrx=1):
    """Create a realistic-looking GPR A-scan mock file."""
    iterations = 1000
    dt = 5e-11  # 50 ps time step -> 50 ns total window

    with h5py.File(filepath, 'w') as f:
        f.attrs['nrx'] = nrx
        f.attrs['dt'] = dt
        f.attrs['Iterations'] = iterations

        t = np.linspace(0, (iterations - 1) * dt, iterations)
        fc = 800e6  # 800 MHz center frequency

        for rx in range(1, nrx + 1):
            t0 = 1.5e-9
            ricker = (1 - 2 * (np.pi * fc * (t - t0))**2) * \
                     np.exp(-(np.pi * fc * (t - t0))**2)

            t1 = 8e-9
            reflection1 = 0.4 * (1 - 2 * (np.pi * fc * (t - t1))**2) * \
                           np.exp(-(np.pi * fc * (t - t1))**2)

            t2 = 18e-9
            reflection2 = -0.15 * (1 - 2 * (np.pi * fc * (t - t2))**2) * \
                           np.exp(-(np.pi * fc * (t - t2))**2)

            noise = np.random.normal(0, 0.02, iterations)

            ex_data = (ricker + reflection1 + reflection2 + noise) * 50
            ey_data = ex_data * 0.1 + np.random.normal(0, 0.5, iterations)
            ez_data = ex_data * 0.05 + np.random.normal(0, 0.3, iterations)
            hx_data = ex_data * 0.002
            hy_data = ey_data * 0.002
            hz_data = ez_data * 0.002

            f.create_dataset(f'/rxs/rx{rx}/Ex', data=ex_data)
            f.create_dataset(f'/rxs/rx{rx}/Ey', data=ey_data)
            f.create_dataset(f'/rxs/rx{rx}/Ez', data=ez_data)
            f.create_dataset(f'/rxs/rx{rx}/Hx', data=hx_data)
            f.create_dataset(f'/rxs/rx{rx}/Hy', data=hy_data)
            f.create_dataset(f'/rxs/rx{rx}/Hz', data=hz_data)


mock_file = os.path.join(OUT_DIR, 'cylinder_Ascan_2D.out')
create_realistic_mock(mock_file)

print("Generating demo plots...\n")

# Demo 1: Default single-component
plt.close('all')
p = mpl_plot(mock_file, outputs=['Ex'])
p.savefig(os.path.join(OUT_DIR, 'demo1_default_ex.png'), dpi=150, bbox_inches='tight')
plt.close('all')
print("[1/5] Default single-component -> demo1_default_ex.png")

# Demo 2: Normalized
plt.close('all')
p = mpl_plot(mock_file, outputs=['Ex'], normalize=True)
p.savefig(os.path.join(OUT_DIR, 'demo2_normalized.png'), dpi=150, bbox_inches='tight')
plt.close('all')
print("[2/5] Normalized               -> demo2_normalized.png")

# Demo 3: No grid
plt.close('all')
p = mpl_plot(mock_file, outputs=['Ex'], show_grid=False)
p.savefig(os.path.join(OUT_DIR, 'demo3_no_grid.png'), dpi=150, bbox_inches='tight')
plt.close('all')
print("[3/5] No-grid                  -> demo3_no_grid.png")

# Demo 4: Multi-component
plt.close('all')
p = mpl_plot(mock_file, outputs=['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'])
p.savefig(os.path.join(OUT_DIR, 'demo4_multi_component.png'), dpi=150, bbox_inches='tight')
plt.close('all')
print("[4/5] Multi-component          -> demo4_multi_component.png")

# Demo 5: FFT
plt.close('all')
p = mpl_plot(mock_file, outputs=['Ex'], fft=True)
p.savefig(os.path.join(OUT_DIR, 'demo5_fft.png'), dpi=150, bbox_inches='tight')
plt.close('all')
print("[5/5] FFT view                 -> demo5_fft.png")

os.unlink(mock_file)
print(f"\nAll demos saved to: {os.path.abspath(OUT_DIR)}")
