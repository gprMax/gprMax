Fluxes scattering example
==========================

This directory demonstrates usage of the flux / scattering feature with a dielectric sphere (Mie scattering efficiency).

Contents:
 - lossless_sphere_scattering.in: base input template (sphere + flux box + scattering block).
 - run_lossless_sphere.py: generates a parametric input and runs the simulation (CPU or GPU) writing Mie_scattering_<backend>_fluxes.out inside this directory.
 - analyse_lossless_sphere.py: loads a flux output (auto‑detects a standard name or use --file), computes scattering efficiency, compares to Mie theory (PyMieScatt), and saves a plot.
 - Mie_scattering_<backend>.png / *_fluxes.out: example outputs (can be regenerated).

Dependencies:
 - Simulation: numpy, scipy, gprMax (already in core environment), optional GPUtil (for GPU auto‑detection).
 - Analysis (theory comparison): PyMieScatt (and its own SciPy dependency).

IMPORTANT (environment / SciPy compatibility):
 PyMieScatt currently pins / requires an older SciPy version than the one recommended for gprMax. To avoid downgrading SciPy in your main gprMax environment, create a small separate environment purely for the analysis step. Example with venv:

   python -m venv mieenv
   source mieenv/bin/activate
   pip install PyMieScatt matplotlib h5py numpy
   # (Run only the analyse_lossless_sphere.py script from this env; keep simulations in the main env.)

Or with conda:

   conda create -n mieenv python=3.11 PyMieScatt matplotlib h5py numpy
   conda activate mieenv

You can also simply try installing PyMieScatt in the main environment; if it attempts to downgrade SciPy, abort and use a dedicated env as above.

Quick install (analysis env, pip example):

   pip install PyMieScatt
   # Optional if you need GPU discovery for the run script (can stay in main env):
   pip install GPUtil

Workflow:
 1. Run simulation (CPU):
   python run_lossless_sphere.py --backend cpu
 2. (Optional) Run simulation (GPU):
   python run_lossless_sphere.py --backend gpu
 3. Analyse (auto‑detects first matching flux file):
   python analyse_lossless_sphere.py
   # or specify a file if multiple present
   python analyse_lossless_sphere.py --file Mie_scattering_cpu_fluxes.out

Run script arguments:
 --backend {cpu,gpu}  Execution device (GPU requires a CUDA device; auto detection via GPUtil if installed).
 --nfrq N             Number of wavelength samples (default 100).
 --radius R           Sphere radius in metres (default 1e-6).

Analysis script arguments:
 --file PATH          Specific flux file (otherwise searches standard names in current directory).
 --radius R           Sphere radius (must match run; default 1e-6).
 --nrel N             Relative refractive index (default 2.0).

Notes:
 - Efficiency computed: σ / (π r²) from integrated scattered flux vs incident reference.
 - Run script constrains all generated inputs & outputs to this directory.
 - Incident fluxes are stored with sign (orientation of Poynting vector); analysis uses magnitude appropriately.
 - PyMieScatt only needed for theoretical curve; simulation itself does not depend on it.
