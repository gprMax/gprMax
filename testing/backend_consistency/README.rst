==========================
Backend-consistency checks
==========================

These checks compare equivalent calculations across solver backends,
precision modes, source orientations, or other configurations. Agreement is
useful regression evidence, but it is not validation against an analytical
solution.

The focused CPU/GPU checks suitable for routine automation belong in
``tests``. The scripts here retain larger manual studies and their compact
plots and metrics.
