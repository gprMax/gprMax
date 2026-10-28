Example 9: square anisotropic guide with automatic branch crossing
==================================================================

A 16 × 16 mm square PEC bore is filled with diag(2.25, 1, 1). Its anisotropic
material reduces the sampled symmetry to C2v. The y-polarized TE10 and
x-polarized TE02 branches start as modes 3 and 4 and exchange raw eigenvalue
ranks near 14.4 GHz on this mesh. Two lower-order branches remain above them. Automatic
tracking preserves these channel labels at both ports through the crossing.

Both ports request ``modes=(3, 4)``, ``tracking="auto"``, explicit frequency
anchors, and ``plot_fields=True``. The source uses the automatic pulse,
and both longitudinal terminations use sixteen-cell PMLs. PEC walls are
built after the anisotropic fill to preserve shared wall samples. The probe
is off the bore centre at x = 10 mm, y = 6 mm so it sees both polarizations;
TE02 has an Ex node at the centre.

Run from the repository root::

   python examples/features/eigenmode_ports/example_9_C2v_square_anisotropic_crossing/auto_mode_crossing.py --geometry-only
   python examples/features/eigenmode_ports/example_9_C2v_square_anisotropic_crossing/auto_mode_crossing.py --mode 3
   python examples/features/eigenmode_ports/example_9_C2v_square_anisotropic_crossing/plot_results.py --mode 3
   python examples/features/eigenmode_ports/example_9_C2v_square_anisotropic_crossing/auto_mode_crossing.py --mode 4
   python examples/features/eigenmode_ports/example_9_C2v_square_anisotropic_crossing/plot_results.py --mode 4

Geometry-only figures show stable tracked profiles while dispersion curves
cross. Full runs measure modal reflection and transmission over 13.1–15.8 GHz.
The plotter shows co- and cross-channel S11/S21 and the interior probe's Ex/Ey
traces, and marks the crossing inferred from the retained anchors. Change only
``--mode`` between the two launched polarizations. Use ``--output`` on the
runner and ``--input`` on the plotter for a custom output stem.

Recorded simulation plots
-------------------------

.. image:: square_crossing_mode3_results.png
   :width: 100%
   :alt: Mode 3 reflection, transmission, and interior Ey polarization.

.. image:: square_crossing_mode4_results.png
   :width: 100%
   :alt: Mode 4 reflection, transmission, and interior Ex polarization.

Both launched-mode results were regenerated on 24 October 2026. ``recorded_results.json`` retains source hashes, commands, power-wave validity and scattering measurements.
