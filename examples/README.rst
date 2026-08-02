==============
gprMax examples
==============

The examples are organised by application so that related hash-command input
files and Python API models can be found together.

``gpr/``
    Ground-penetrating radar examples, divided into basic models, antenna
    models, materials, and subgridding applications.

``antennas/``
    General antenna examples that are not specific to GPR. This directory is
    organised by antenna family. The ``wire_dipole`` example keeps its
    transmission-line and voltage-port feeds together for comparison.

``features/``
    Small models demonstrating solver or model-building features independently
    of a particular application. This includes plane waves and generic
    subgridding examples.

``jupyter-notebooks/``
    Interactive tutorials. These are retained separately while their older
    paths and commands are modernised.

Equivalent text input and Python API versions of a model are stored in the
same directory wherever both are available. Simulation outputs and generated
geometry or plotting files should not be committed to this directory.

Radar cross-section examples
============================

User-facing radar cross-section examples will be stored in a top-level
``rcs/`` directory. Detailed analytical comparisons, convergence studies, and
generated results belong in the validation suite rather than here.
