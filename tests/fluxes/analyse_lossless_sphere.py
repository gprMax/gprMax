"""Analyse Mie scattering flux output vs Mie theory.

Copyright (C) 2025: Quandela
Author: Quentin David
License: GNU GPL v3 or later.

Loads a flux output file (default autodetects one of Mie_scattering_cpu_fluxes.out,
Mie_scattering_gpu_fluxes.out, or Mie_scattering_fluxes.out), computes scattering
efficiency σ/(π r^2) and compares to PyMieScatt theoretical Qsca.

CLI examples:
    python analyse_lossless_sphere.py            # auto-detect file
    python analyse_lossless_sphere.py --file custom_fluxes.out --radius 1e-6

Outputs a figure Mie_scattering.png (or <basename>.png if --file specified) and prints a distance metric.
"""

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import PyMieScatt as ps
from scipy.constants import c

def load_flux(filename):
    with h5py.File(filename, 'r') as f:
        # infer wavelengths from box (all boxes share same wavelengths)
        wavelengths = f['boxes']['box1']['wavelengths'][()]
        incident = f['scattering']['incidents']['incident1']['values'][()]
        scattered = f['boxes']['box1']['values'][()]
    return wavelengths, incident, scattered

def compute_eff(wavelengths, incident_flux, scattered_flux, r):
    incident_norm = incident_flux / (2*r)**2
    eff = scattered_flux / incident_norm / (np.pi * r**2)
    return eff

def theory_eff(wavelengths, r, n_rel=2.0):
    # PyMieScatt expects size in nm, wavelength in nm
    wl_nm = wavelengths * 1e9
    diameter_nm = 2*r * 1e9
    return np.array([ps.MieQ(n_rel, wl, diameter_nm, asDict=True)['Qsca'] for wl in wl_nm])

def main():
    p = argparse.ArgumentParser(description='Analyse Mie scattering flux output.')
    p.add_argument('--file', help='Flux output file path (defaults to auto-detect standard names).')
    p.add_argument('--radius', type=float, default=1e-6, help='Sphere radius (m).')
    p.add_argument('--nrel', type=float, default=2.0, help='Relative refractive index.')
    args = p.parse_args()
    filename = args.file
    if not filename:
        # Try common defaults in order
        for cand in ["Mie_scattering_cpu_fluxes.out", "Mie_scattering_gpu_fluxes.out", "Mie_scattering_fluxes.out"]:
            try:
                with open(cand, 'rb'):
                    filename = cand
                    break
            except FileNotFoundError:
                continue
        if not filename:
            p.error('No flux file found (searched standard names); specify one with --file')

    wavelengths, incident, scattered = load_flux(filename)
    eff_sim = compute_eff(wavelengths, incident, scattered, args.radius)
    eff_th = theory_eff(wavelengths, args.radius, args.nrel)
    dist = np.linalg.norm(eff_sim - eff_th)/np.linalg.norm(eff_th)

    x_axis = 2*np.pi*args.radius / wavelengths
    fig, ax = plt.subplots(dpi=150)
    ax.loglog(x_axis, eff_sim, 'ro-', label='simulation', alpha=0.7)
    ax.loglog(x_axis, eff_th, 'k--', label='theory', alpha=0.7)
    ax.set_xlabel('2πr/λ')
    ax.set_ylabel('scattering efficiency σ/πr²')
    ax.set_title('Mie Scattering of a Lossless Dielectric Sphere')
    ax.grid(which='both', ls=':')
    ax.legend(loc='upper right')
    fig.text(0.5, 0.88, f"Relative L2 distance: {dist:.2e}", ha='center', fontsize=8,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    # Derive PNG name: if user passed a file, reuse its base (strip trailing _fluxes.out)
    if args.file:
        base = args.file
        if base.endswith('_fluxes.out'):
            base = base[:-11]
        outpng = f"{base}.png"
    else:
        outpng = 'Mie_scattering.png'
    fig.tight_layout()
    plt.savefig(outpng)
    print(f"Saved {outpng}; distance={dist:.3e}")

if __name__ == '__main__':
    main()
