Planar-layered NTFF validation
==============================

The direct time-domain transmission-line kernel has independent closed-form
checks for the dielectric half-space and ungrounded dielectric slab in
``tests/ntff/test_layered_time.py``. They evaluate the impulse amplitudes and
delays in Eqs. (53)--(55) and (71)--(75) of Çapoğlu's 2007 thesis for both TE
and TM voltage/current responses. The tests account explicitly for gprMax's
factor-two Green-response convention and observation-medium impedance
normalisation. Random multilayer tests and an end-to-end FDTD comparison then
check the direct impulse trains against the established frequency-domain
recursion.

``validate_capoglu_grounded_time.py`` reproduces the PEC-grounded slab in
Figure 11 of the thesis. It models horizontal and vertical Hertzian electric
dipoles 1 mm below the air interface of a 2 mm, :math:`\epsilon_r=2.5` slab,
using the published 0.1 mm spatial step and 5 ps differentiated-Gaussian
pulse. The transform surface omits only its lower face, which coincides with
the terminal PEC plane. The independent reference evaluates the short-circuit
echo series in Eqs. (59), (63), and (65), using the stored source samples and
the physical centre of the Yee source edge. No amplitude, phase, or time
alignment is fitted.

Run the double-precision CPU benchmark with::

    python -m testing.validation.planar_layered_ntff.validate_capoglu_grounded_time

or exercise the device-resident CUDA collector with, for example::

    python -m testing.validation.planar_layered_ntff.validate_capoglu_grounded_time --gpu 0 --precision double

The retained HED maximum and RMS errors are 0.494 and 0.149 percent of the
analytical peak; the VED values are 0.915 and 0.287 percent. The result checks
the PEC short-circuit recursion, direct retarded-time propagation, the
terminal-face omission, and both tangential and normal source orientations.
The retained CPU and double-precision CUDA error metrics agree to better than
``3e-14`` in absolute normalised error on the validation system.
The thesis's Figure 12 and later microstrip examples remain illustrative
rather than quantitative independent reference curves.

Independent PEC image and reflection checks
===========================================

``validate_grounded_dipoles.py`` compares four bare-PEC electric/magnetic
dipole configurations with exact image theory and two dielectric-coated PEC
electric-dipole configurations with an independent short-circuited TE/TM
plane-wave-spectrum calculation. It retains complex field, power-pattern,
and maximum-directivity errors at 1.5, 2.0, and 2.5 GHz. The electric cases
remain below 0.049 percent pointwise complex-field error; the deliberately
retained worst case is the tangential magnetic source, whose maximum power
difference is 5.07 percent at this mesh. Halving the cell size from 1.5 mm to
0.75 mm reduces its 2.5 GHz complex-field, power, and maximum-directivity
differences from 3.83, 5.07, and 3.19 percent to 1.89, 2.47, and 1.56 percent,
respectively.

``validate_grounded_slab_reflection.py`` uses total-minus-incident DPW fields
to measure the complex normal-incidence reflection of a 12 mm,
:math:`\epsilon_r=4` PEC-backed slab over 0.4--7.0 GHz. The exact reference
is formed from :math:`Z_{\rm in}=jZ_1\tan(k_1d)`. Its retained maximum
magnitude and phase errors are :math:`2.11\times10^{-8}` and 0.0553 degrees.

Run both with::

    python -m testing.validation.planar_layered_ntff.validate_grounded_dipoles
    python -m testing.validation.planar_layered_ntff.validate_grounded_slab_reflection

The magnetic-source refinement can be repeated on a CUDA device with::

    python -m testing.validation.planar_layered_ntff.validate_grounded_dipoles \
        --case magnetic_tangential_bare --dl 0.00075 --gpu 0

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
