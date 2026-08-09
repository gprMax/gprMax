pytest test suite
=================

The ``tests`` directory is the automated pytest test suite. It contains
focused unit tests as well as compact integration and hardware tests. The
larger, manually run scientific comparisons in ``testing/validation`` are
kept separate from this suite.

Installation
------------

Install the development requirements, including pytest, in the active
environment::

    python -m pip install -r requirements.txt

Running tests
-------------

Run the complete suite::

    python -m pytest

Run the usual CPU development selection, excluding real-GPU and unusually
slow tests::

    python -m pytest -m "not gpu and not slow"

This is also the selection run automatically for pull requests and pushes
to ``devel`` by the GitHub Actions pytest workflow.

Run only compact integration tests::

    python -m pytest -m integration

Run tests that require a real GPU, selecting device 1 in this example::

    python -m pytest -m gpu --gpu-device 1

The CUDA index can instead be set with ``GPRMAX_TEST_GPU``. Tests skip
themselves when the selected CUDA device is unavailable. Tests that inspect
generated GPU source or use mocks are not marked ``gpu`` because they do
not require real hardware.

Use pytest's duration report when deciding whether a test needs the
``slow`` marker::

    python -m pytest --durations=25

Markers
-------

``integration``
    Exercises several gprMax components together or executes a complete,
    compact model. This includes the automated FDTD comparisons with a
    Hertzian-dipole closed form and PEC-sphere Mie theory.

``gpu``
    Executes on a real GPU. This marker may overlap with ``integration``.

``slow``
    Normally takes more than 10 seconds on a development machine. It may be
    combined with either of the other markers.

Analytical helper functions tested without running an FDTD model are normal
unit tests; there is deliberately no separate ``analytical`` marker.
