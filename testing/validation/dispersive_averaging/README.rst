Dispersive-interface averaging validation
=========================================

This directory contains reusable analytical references and manual FDTD
validations for dispersive-interface averaging. Generated HDF5, CSV, JSON,
and plot files are written below ``results/`` and are not tracked.

The validations cover:

* normal-incidence dielectric, Debye, Lorentz, and Drude half spaces;
* finite multilayer stacks containing mixed dispersion families;
* construction-order sensitivity of staircased multilayers; and
* a Debye-core/Lorentz-shell sphere compared with the analytical
  Aden--Kerker series.

Run the shorter analytical reference tests with::

    python -m pytest tests/materials/test_dispersive_averaging_references.py

Run the FDTD studies from the repository root, for example::

    python -m testing.validation.dispersive_averaging.validate_halfspace_comparison
    python -m testing.validation.dispersive_averaging.validate_multilayer_fdtd
    python -m testing.validation.dispersive_averaging.validate_core_shell_fdtd --gpu 0
    python -m testing.validation.dispersive_averaging.run_pole_reduction_study

Pass ``--help`` to a script for precision, accelerator, output-directory,
case-selection, and cache-reuse options. The full sphere models are intended
as manual validation runs rather than routine pytest tests.

The ``reduction`` module is an offline research utility for evaluating
constrained pole-reduction strategies over a specified frequency band. Pole
reduction is deliberately not applied automatically by the solver: a reduced
model is band limited and must satisfy an application-specific error bound.
