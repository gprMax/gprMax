C4v cylindrical staircase: PEC and copper tracked modes
=======================================================

Run from the repository root in the gprMax Python environment::

   python examples/features/eigenmode_ports/example_13_C4v_cylindrical/tracked_modes.py

The example solves 21 modes of an air-filled, 10-mm-radius cylindrical bore
at 39, 40, and 41 GHz. It runs PEC and copper counterparts without excitation
or time stepping. The copper conductivity is 5.96e7 S/m, represented by a
16-term ADE surface-impedance fit over 1–120 GHz.

The default 96 × 96 and 192 × 192 modal windows span 24 mm, giving transverse
steps of 0.25 and 0.125 mm (80 and 160 cells across the bore diameter).
Use ``--cells 96 192`` to select meshes, ``--wall PEC`` or ``--wall copper``
for one material, and a fresh ``--output DIR`` to preserve earlier runs.

A Cartesian voxel circle has C4v symmetry; the smooth analytical cylinder
has continuous rotational symmetry. The staircase can split angular
partners and perturb their propagation constants. Analytical comparison
therefore includes both sine/cosine Bessel partners, and equal-cutoff TE/TM
families where applicable. Profiles never enter the tracking algorithm.

Outputs include native complex E/H fields (NPZ), source hashes and metrics
(JSON), phase-aligned modal-field comparisons, a full-vector Bessel matching
matrix, mesh comparisons for profile and propagation-constant error, and
copper attenuation. All field samples enter the profile metric, including
the discrepancy between the smooth and staircased boundaries; no near-wall
samples are discarded. Copper profiles use smooth PEC references, while
attenuation describes the staircased SIBC model rather than an independently
validated smooth-cylinder loss formula.

See the tracking theory validation section and its cylindrical modal-profile
gallery for recorded results and the reproducible solver implementation in
``testing/validation/eigenmode_tracking/profiles/cylindrical_modal_profiles.py``.
