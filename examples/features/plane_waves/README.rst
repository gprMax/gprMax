===================
Plane-wave examples
===================

``dielectric_sphere_tfsf.in`` and ``dielectric_sphere_tfsf.py`` are equivalent
hash-command and Python API models. They launch a discrete plane wave in the
positive x direction through a total-field/scattered-field (TFSF) box that
contains a dielectric sphere.

The receiver inside the box records the incident plus scattered (total) field.
The receiver beyond the x-maximum face records scattered field only. This
placement makes the field separation performed by the TFSF surface explicit.

The plane-wave direction-vector form is used because the background is
homogeneous free space. Use the axial form for normally incident layered-media
models, where the auxiliary one-dimensional grid must reproduce the sequence
of materials along the propagation axis.
