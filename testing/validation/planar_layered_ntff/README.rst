Planar-layered NTFF validation
==============================

``validate_point_dipole.py`` compares the production FDTD transform with a
direct frequency-domain point-current solution in a three-layer medium.  The
Huygens surface crosses both material interfaces.  The analytical calculation
uses the exact discrete Hertzian source history stored in the output file, so
the comparison includes phase and does not depend on an assumed continuous
source waveform.

Run the double-precision CPU case with::

    python -m testing.validation.planar_layered_ntff.validate_point_dipole

The formulation is based on Çapoğlu, Taflove, and Backman, IEEE Transactions
on Antennas and Propagation 60(4), 1878--1885 (2012),
https://doi.org/10.1109/TAP.2012.2186253.  The validation is analogous to the
paper's layered Hertzian-dipole experiment, but its geometry is deliberately
smaller so it remains practical as a repeatable repository validation.

The retained 1 mm model covers nine frequencies from 1--3 GHz and both
observation half-spaces. Its maximum vector-field error normalised to the
analytical peak is 2.304 percent and its RMS error is 0.857 percent. A 2 mm
run gives 4.738 and 1.722 percent, respectively, providing a mesh-refinement
check rather than only a single retained comparison.
