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

Additional analytical comparisons
=================================

Run the published eight-layer Çapoğlu reproduction with::

    python -m testing.validation.planar_layered_ntff.validate_capoglu_paper --gpu 0

It uses nine differently oriented point-current sources and compares the
closed production Huygens surface with the same point currents evaluated
directly in the unperturbed stack. The maximum real/imaginary curve RMS error
relative to each analytical curve peak is 0.477 percent, below the one-percent
maximum reported in the paper.

The Engheta and Smith comparisons use independent asymptotic dipole
expressions::

    python -m testing.validation.planar_layered_ntff.validate_engheta_interfacial_dipoles --gpu 0
    python -m testing.validation.planar_layered_ntff.validate_smith_dipole_height --gpu 0

The Engheta cases place vertical and horizontal electric dipoles directly on
interfaces with refractive-index ratios two and four. Their worst RMS and
pointwise normalised-power differences are 0.363 and 1.059 percent. The Smith
cases place a horizontal dipole at :math:`h/\lambda_0=0.1`, 0.2, and 0.35
above an :math:`\epsilon_r=9` half-space; their corresponding worst errors
are 0.271 and 0.755 percent.

Finite-radius GPR antenna verification
======================================

Run the GSSI-like 1.5 GHz energy-pattern study with::

    python -m testing.validation.planar_layered_ntff.validate_gssi_energy_convergence --gpu 0

This is a verification of the Warren--Giannopoulos finite-distance
methodology rather than an independent analytical solution. It samples 25
receiver circles from 0.10--0.58 m over a lossless :math:`\epsilon_r=5`
half-space and compares their time-integrated field-energy shapes with a
broadband layered NTFF result. At 0.58 m the RMS differences are 1.38 percent
in the E plane and 3.44 percent in the H plane. See the analytical-comparison
page in the user documentation for equations, plots, bandwidth checks, and
the 2 mm versus published 1 mm resolution caveat.
