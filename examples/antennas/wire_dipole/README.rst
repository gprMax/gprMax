====================
Wire-dipole examples
====================

This directory keeps alternative feeds for the same 150 mm wire dipole close
together:

``antenna_wire_dipole_fs.in`` and ``antenna_wire_dipole_fs.py``
    Equivalent input methods for a 50 Ohm, one-cell resistive voltage source
    with a coincident receiver port. This is the recommended example. The HDF5
    ``/ports/feed`` group directly contains its frequency axis, corrected
    complex ``S11``, ``Zin``, and ``Yin``.

``antenna_wire_dipole_pattern.in`` and ``antenna_wire_dipole_pattern.py``
    Equivalent hash-command and Python API models. They add a closed NTFF
    surface and request full-sphere directivity, gain, realized gain,
    radiation efficiency, and total efficiency at 950 MHz.
    ``plot_wire_dipole_pattern.py`` reads those quantities directly from the
    HDF5 output.

``antenna_wire_dipole_transmission_line.py``
    Uses gprMax's one-dimensional transmission-line feed. It is retained for
    studies that need the explicit line voltage/current histories and the
    independent current-wave terminal check.

The two feeds are different numerical source models and should not be expected
to produce identical broadband terminal responses without a convergence and
reference-plane study.
