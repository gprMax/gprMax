"""Mie scattering simulation driver (flux/scattering example).

Copyright (C) 2025: Quandela
Author: Quentin David
License: GNU GPL v3 or later (see root LICENSE file).

Generates a gprMax input file for a dielectric sphere inside a box_flux, runs
the scattering (empty + with sphere) simulation, producing
Mie_scattering_<backend>_fluxes.out.

CLI:
  python run_lossless_sphere.py --backend cpu
  python run_lossless_sphere.py --backend gpu

Dependencies: scipy, numpy, gprMax, (GPUtil for GPU discovery).
"""

import argparse
import numpy as np
from scipy.constants import c
from gprMax.gprMax import api as run_sim
import os

# Directory where this script resides – all generated inputs/outputs constrained here
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def detect_gpu():
	try:
		import GPUtil
		devs = GPUtil.getAvailable()
		return devs if devs else None
	except Exception:
		return None

def build_and_run(radius=1e-6, nfrq=100, backend='cpu'):
	r = radius
	wvl_min = 2*np.pi*r/10
	wvl_max = 2*np.pi*r/2
	frq_min = c/wvl_max
	frq_max = c/wvl_min
	frq_cen = 0.5*(frq_min+frq_max)
	std = wvl_max/c - wvl_min/c
	resolution = 25
	dl = 1.0e-6/resolution
	dpml = 0.5*wvl_max
	dair = 0.5*wvl_max
	pmlcells = int(np.ceil(dpml/dl))
	s = 2*(dpml + dair + r)
	timewindow = 150e-15
	title = f"Mie_scattering_{backend}"
	inputfname = os.path.join(SCRIPT_DIR, f"mie_{backend}.in")
	with open(inputfname, 'w') as f:
		f.write(f"#title: {title}\n")
		f.write(f"#domain: {s} {s} {s}\n")
		f.write(f"#dx_dy_dz: {dl} {dl} {dl}\n")
		f.write(f"#time_window: {timewindow}\n")
		f.write(f"#pml_cells: {pmlcells}\n")
		f.write(f"#time_step_stability_factor: 0.5\n")
		f.write(f"#waveform: gaussian 1 {frq_cen} my_gaussian {std}\n")
		f.write(f"#plane_voltage_source: z {dpml + 2*dl} 0 0 {dpml + 2*dl} {s} {s} 0 my_gaussian\n")
		f.write("#material: 4 0 1 0 my_material\n")
		f.write(f"#box_flux: {s/2} {s/2} {s/2} {r} {r} {r} {r} {r} {r} {wvl_min} {wvl_max} {nfrq}\n")
		f.write("#scattering:\n")
		f.write(f"#sphere: {s/2} {s/2} {s/2} {r} my_material\n")
		f.write("#scattering_end:\n")
	# run
	gpu_arg = None
	if backend == 'gpu':
		gpu_ids = detect_gpu()
		if not gpu_ids:
			raise RuntimeError('No GPU available for backend=gpu')
		gpu_arg = gpu_ids
	run_sim(inputfile=inputfname, gpu=gpu_arg, outputdir=SCRIPT_DIR)
	return inputfname

def main():
	parser = argparse.ArgumentParser(description='Run Mie scattering flux example (CPU/GPU).')
	parser.add_argument('--backend', choices=['cpu','gpu'], default='cpu')
	parser.add_argument('--radius', type=float, default=1e-6, help='Sphere radius (m)')
	parser.add_argument('--nfrq', type=int, default=100, help='Number of wavelength samples')
	args = parser.parse_args()
	build_and_run(radius=args.radius, nfrq=args.nfrq, backend=args.backend)

if __name__ == '__main__':
	main()

