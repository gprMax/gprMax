==========================
Other numerical codes
==========================

This directory contains reproducible comparisons between gprMax and other
numerical solvers. These are inter-code comparisons, not correctness
validation: discretisation, geometry, material, and source representations
can differ, and neither solver is treated as ground truth.

Analytical reference cases are kept in ``testing/validation``. Automated
regressions belong in ``tests``; larger behavioural and backend comparisons
are kept in ``testing/regression`` and ``testing/backend_consistency``.

``matlab_mom`` contains the current comparisons with MATLAB Antenna Toolbox
Method-of-Moments models.
