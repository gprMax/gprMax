Toolboxes is a sub-package where useful Python modules contributed by users are stored.

******************
GPR Antenna Models
******************

Information
===========

Optional optimisation keyword names are checked. Unknown names raise
``TypeError`` instead of being ignored, even if their value is ``None`` or
zero. Supply the documented parameter set for the selected antenna.

The package features models of antennas similar to commercial GPR antennas. The following antenna models are included:

======================== ============= ============= ========================================================================================================================================================================================================================= ================
Manufacturer/Model       Dimensions    Resolution(s) Author/Contact                                                                                                                                                                                                            Attribution/Cite
======================== ============= ============= ========================================================================================================================================================================================================================= ================
GSSI 1.5GHz (Model 5100) 170x108x45mm  1, 2mm        Craig Warren (craig.warren@northumbria.ac.uk), Northumbria University, UK                                                                                                                                                 1
GSSI 2GHz palm antenna   86x86x68mm    1mm           Ourania Patsia, Antonios Giannopoulos, and Iraklis Giannakis, The University of Edinburgh, UK                                                                                                                              3
MALA 1.2GHz              184x109x46mm  1, 2mm        Craig Warren (craig.warren@northumbria.ac.uk), Northumbria University, UK                                                                                                                                                 —
GSSI 400MHz              300x300x170mm 2mm           Sam Stadler (sam.stadler@leibniz-liag.de), `Leibniz Institute for Applied Geophysics <https://www.leibniz-liag.de/en/research/methods/electromagnetic-methods/ground-penetrating-radar.html>`_, Germany                   2
======================== ============= ============= ========================================================================================================================================================================================================================= ================

**License**: `Creative Commons Attribution-ShareAlike 4.0 International License <http://creativecommons.org/licenses/by-sa/4.0/>`_

**Attributions/citations**:

1. Giannakis, I., Giannopoulos, A., & Warren, C. (2019). Realistic FDTD GPR antenna models optimised using a novel linear/non-linear Full Waveform Inversion. *IEEE Transactions on Geoscience and Remote Sensing*, 57(3), 1768-1778. (https://doi.org/10.1109/TGRS.2018.2869027)
2. Stadler, S., & Igel, J. (2022). Developing realistic FDTD GPR antenna surrogates by means of particle swarm optimization. *IEEE Transactions on Antennas and Propagation*, 70(6), 4259-4272. (https://doi.org/10.1109/TAP.2022.3142335)
3. Patsia, O., Giannopoulos, A., & Giannakis, I. (2024). Developing a realistic numerical equivalent of a GPR antenna transducer using global optimizers. *Near Surface Geophysics*, 22(2), 140-151. (https://doi.org/10.1002/nsg.12280)

Package contents
================

* ``GSSI.py`` is a module containing models of antennas similar to those manufactured by `Geophysical Survey Systems, Inc. (GSSI) <http://www.geophysical.com>`_.
* ``MALA.py`` is a module containing models of antennas similar to those manufactured by `MALA Geoscience <http://www.malags.com/>`_.

Descriptions of how the models were created can be found in the aforementioned attributions.

GSSI 2 GHz receiver-port update
-------------------------------

``antenna_like_GSSI_2000`` now uses a zero-amplitude voltage source with the
original 200008.0107 Ohm receiver resistance. Its rigid gap retains er=1.056;
only the resistance has moved from material conductivity to the source, so
the original loading and ``gssi2000_rxbowtie`` Ey output are preserved at
the default orientation (see rotation below).
The automatic voltage-source ports now have explicit IDs ``gssi2000_tx``
and ``gssi2000_rx`` (update references to the former automatic Tx ``portN``).
Receiver voltage is saved in ``/ports/gssi2000_rx/Vtotal`` with its own
half-step time axis. Zero amplitude does not mean zero resistance or a clamp.

Use the driven Tx port for its input impedance and S11. The passive receiver's
generator is zero, so its own input S11/impedance cannot be inferred from this
run; drive Rx separately with Tx terminated to characterise that input.
For NTFF gain/efficiency, associate **both** physical ports, including the
passive receiver, e.g. after declaring an NTFF transform ``antenna_band``:

.. code-block:: python

    scene.add(gprMax.NTFFAntennaPorts(
        transform_id='antenna_band',
        port_ids=('gssi2000_tx', 'gssi2000_rx'),
    ))

Radiation-efficiency normalisation treats the Rx termination as an external
port load rather than internal material loss, using both ports' net accepted
power. This changes the power-accounting boundary, not the solved fields.

The default ``spectrum_limit=10`` retains normal mesh-validity checks. The
legacy highly conducting materials may leave no valid positive-frequency
bins. ``spectrum_limit='nyquist'`` explicitly requests research spectra/power
normalisation, not a mesh-convergence certificate; respect validity metadata
and validate numerical convergence before interpreting impedance or efficiency.
No NTFF surface or frequency band is added automatically.

The legacy conducting materials can also trigger the solver's
underresolution preflight. Reproducing that original model may require the
explicit ``gprMax.run(..., allow_underresolved=True)`` research override.
This does not certify mesh convergence or change the port validity flags;
placement and rotation do not relax either check or replace these materials
with PEC.

How to use the package
======================

The antenna models can be accessed from within a block of Python code in an input file. The models are inserted at location x,y,z. The coordinates are relative to the geometric centre of the antenna in the x-y plane and the bottom of the antenna skid in the z direction. The supported cubic spatial resolutions are listed in the table above and are selected with the keyword argument, e.g. ``resolution=0.002``.

.. note::

    Use survey steps that are multiples of the spatial resolution. Every
    antenna model rounds its placement once and preserves its geometry, but
    sub-cell steps can still map successive positions to the same grid point.
    Moving a complete antenna requires rebuilding its geometry;
    ``src_steps``/``rx_steps`` move sources/receivers only.

GSSI 2 GHz placement and in-plane rotation
------------------------------------------

This model requires a **1 mm cubic grid**, including when placed in a subgrid.
The supplied ``(x, y, z)`` is the horizontal centre of the case at the bottom
of the skid, not a corner or the three-dimensional centre. It is rounded once
to the nearest grid point using gprMax's usual rounding rule; a warning gives
the actual position when it moves. All geometry and feeds are then placed
using integer-cell offsets. This preserves the one-cell gaps even at half-cell
positions that previously could collapse an edge. Grid-aligned placements at
the default orientation retain their original geometry and loading. Normal
domain and subgrid bounds still apply: this does not allow an antenna to
extend outside the permitted region.

The optional keyword ``rotation`` accepts **0, 90, 180 or 270 degrees** only.
Positive rotation is counterclockwise when looking from +z towards the x-y
plane, about the snapped placement point. The complete antenna rotates,
including its geometry, transmitter, passive receiver load and field monitor.
Its z coordinates and the surrounding model are unchanged. This is an exact
grid-aligned rotation, not arbitrary-angle voxelisation. The other toolbox
antenna models offer the same option (see below).

.. code-block:: python

    from gprMax.toolboxes.GPRAntennaModels.GSSI import antenna_like_GSSI_2000

    # Add to a scene that already declares a 1 mm cubic grid and large enough domain.
    for obj in antenna_like_GSSI_2000(0.125, 0.125, 0.040, rotation=90):
        scene.add(obj)

For an x-y survey, create a scene per antenna position/orientation and call
the factory with that position and rotation. The antenna's fixed material,
waveform and port IDs are retained, so do not add multiple copies to the same
grid. Keep the ground/targets unchanged; do not rotate the complete scene or
change the source polarisation independently.

Output components and polarity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The field receiver remains ``gssi2000_rxbowtie`` and the ports remain
``gssi2000_tx`` and ``gssi2000_rx``. Outputs follow the **global positive-axis**
convention, not an antenna-local axis. The transmitter waveform changes sign
where needed to preserve the physical excitation of the rotated antenna.

========= ====================== ==========================
Rotation  Field dataset          Antenna-local voltage
========= ====================== ==========================
0         ``Ey``                 ``Vtotal``
90        ``Ex``                 ``-Vtotal``
180       ``Ey``                 ``-Vtotal``
270       ``Ex``                 ``Vtotal``
========= ====================== ==========================

The same sign conversion applies to the selected electric-field component
and directed port-current traces when comparing antenna-local polarities.
Read port voltages directly using their own saved time axis; do not assume
they share the raw field receiver's sample times. Consistent voltage/current
sign reversals leave impedance and power unchanged. Rotation relative to a
fixed target can, of course, change the physical response.

MALA 1.2 GHz, GSSI 1.5 GHz and GSSI 400 MHz placement and rotation
------------------------------------------------------------------

``antenna_like_MALA_1200``, ``antenna_like_GSSI_1500`` and
``antenna_like_GSSI_400`` place their geometry the same way. The supplied
``(x, y, z)`` (horizontal centre of the case at the bottom of the skid) is
rounded once to the nearest grid point; a warning gives the actual position
when it moves. Every part is then placed using whole-cell offsets from that
point, so the antenna has the same shape wherever it is placed. Previously
each coordinate was rounded separately: where a dimension fell on a half cell,
floating-point round-off decided the rounding, and parts could grow, shrink
or move by one cell from one placement to the next, including at grid-aligned
placements (the MALA case is 109 mm wide, a half cell from its centre at
1 mm).

Where a dimension is an odd number of cells, it is split with the half cell
below the placement point. At 1 mm the MALA case spans ``y - 55`` to
``y + 54`` cells, and at 2 mm the GSSI 1.5 GHz case spans ``x - 43`` to
``x + 42`` cells. Grid-aligned placements of the GSSI 1.5 GHz model at 1 mm
and the GSSI 400 MHz model keep their original geometry. The MALA model at
both resolutions and the GSSI 1.5 GHz model at 2 mm previously changed with
placement, so their geometry now matches one of the previous placements
rather than all of them.

The ``rotation`` keyword, rotation convention, output components and
polarity table are the same as for the GSSI 2 GHz model above. Each field
receiver is ``rxbowtie``. At 90 and 180 degrees the transmitter waveform is
negated to keep the antenna-local excitation. The GSSI 400 MHz model reads its
excitation from a file, which cannot be scaled; at those rotations the file's
columns are passed on as user waveforms with the same IDs, interpolation and
fill value, with the transmitter column negated.

.. code-block:: python

    from gprMax.toolboxes.GPRAntennaModels.MALA import antenna_like_MALA_1200

    for obj in antenna_like_MALA_1200(0.150, 0.150, 0.050, rotation=90):
        scene.add(obj)

Example
-------

To include an antenna model similar to a GSSI 1.5 GHz antenna at a location 0.125m, 0.094m, 0.100m (x,y,z) using a 2mm cubic spatial resolution:

.. code-block:: none

    import gprMax
    from gprMax.toolboxes.GPRAntennaModels.GSSI import antenna_like_GSSI_1500

    scene = gprMax.Scene()

    # Import antenna model and add to model
    dl = 0.002
    ant_pos = (0.125, 0.094, 0.100)
    gssi_objects = antenna_like_GSSI_1500(ant_pos[0], ant_pos[1], ant_pos[2],
                                          resolution=dl)
    for obj in gssi_objects:
        scene.add(obj)

.. figure:: ../../images_shared/antenna_like_GSSI_1500.png
    :width: 600 px

    FDTD geometry mesh showing an antenna model similar to a GSSI 1.5 GHz antenna (skid removed for illustrative purposes).

.. figure:: ../../images_shared/antenna_like_GSSI_400.png
    :width: 600 px

    FDTD geometry mesh showing an antenna model similar to a GSSI 400 MHz antenna (skid removed for illustrative purposes).

.. figure:: ../../images_shared/antenna_like_MALA_1200.png
    :width: 600 px

    FDTD geometry mesh showing an antenna model similar to a MALA 1.2GHz antenna (skid removed for illustrative purposes).
