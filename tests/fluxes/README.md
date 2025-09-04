Fluxes scattering example
==========================

This directory demonstrates usage of the flux / scattering feature with a dielectric sphere (Mie scattering efficiency).

Contents:
 - lossless_sphere_scattering.in: base input template (sphere + flux box + scattering block).
 - run_lossless_sphere.py: generates a parametric input and runs the simulation (CPU or GPU) writing Mie_scattering_<backend>_fluxes.out.
 - analyse_lossless_sphere.py: loads the fluxes output, computes scattering efficiency, compares to Mie theory (PyMieScatt), and saves a plot.
 - Mie_scattering_<backend>.png / *_fluxes.out: example outputs (can be regenerated).

Prerequisites:
 - PyMieScatt (required for the analysis script to compute the theoretical Mie scattering efficiency).
 - GPUtil (only needed when using --backend gpu to auto-detect an available CUDA device).

Install (user environment):
   pip install PyMieScatt
   # Only if you plan to run on GPU
   pip install GPUtil

If you use a managed environment (conda/venv), install these packages there. The core gprMax run itself does not require PyMieScatt; it is only used for post-processing comparison.

Usage:
 1. Run simulation (CPU):
    python run_lossless_sphere.py --backend cpu
 2. Run simulation (GPU):
    python run_lossless_sphere.py --backend gpu
 3. Analyse (auto-detect backend file or specify):
    python analyse_lossless_sphere.py --backend cpu
    python analyse_lossless_sphere.py --backend gpu

Arguments:
 --backend {cpu,gpu}  Select execution device. GPU requires a detected CUDA device.
 --nfrq N             Number of wavelength samples (default 100).
 --radius R           Sphere radius in metres (default 1e-6).

Notes:
 - Incident and scattered fluxes are used to compute efficiency σ/(π r^2).
 - The run script writes a temporary generated input file and invokes gprMax API.
 - PyMieScatt must be installed for the analysis step.
