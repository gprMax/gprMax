C2v_rectangle_2to1: tracked PEC and copper modes
=====================================================

Run from the repository root with the gprMax Python environment::

   python examples/features/eigenmode_ports/example_12_C2v_rectangle_2to1/tracked_modes.py

Use ``--wall PEC`` or ``--wall copper`` to run one material, and ``--output DIR``
to select a fresh output directory. There is no excitation or time stepping.
The script invokes ``testing/validation/waveguide_modal_profiles.py`` and uses
64 and 96 cells across the 20-mm width, with square transverse cells.
It writes native complex Yee fields (NPZ), source hashes and metrics (JSON),
phase-aligned longitudinal field comparisons, full-vector matching matrices,
mesh-comparison plots, and copper attenuation plots.

The C4v square tracks 18 modes at 39/40/41 GHz; the C2v 2:1 rectangle tracks
35 modes at 98/100/102 GHz, including TE61 and TE23. Labels are assigned only
for validation after automatic tracking; analytical profiles never enter the
tracking algorithm. Copper has conductivity 5.96e7 S/m and an ADE surface
impedance fitted over 1–120 GHz. Copper modes can mix PEC-degenerate TE/TM
profiles; the matrix shows that mixing, while the subspace score measures
agreement with the complete PEC family. The plotted TE/TM name identifies
the dominant PEC longitudinal component, not an exact copper mode family.

See the tracking theory documentation's validation section for the recorded
results, overlap definitions, and all field galleries.
