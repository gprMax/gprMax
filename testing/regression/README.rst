=================
Regression models
=================

This directory contains larger behavioural and diagnostic model matrices that
do not use an independent analytical reference solution. They exercise
expected solver behaviour and are separate from the analytical cases in
``testing/validation`` and the focused automated tests in ``tests``.

``eigenmode_sources`` exercises mode dimensionality, directionality,
boundaries, and broadband interpolation.

``impulse_response`` verifies receiver histories synthesised from one impulse
run against separate direct FDTD runs using several built-in waveforms. Run
it from the repository root with::

    python -m testing.regression.impulse_response.validate_waveform_synthesis
