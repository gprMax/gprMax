Reusable examples and antenna validation
----------------------------------------

The main examples use the same four sections: parameters, model, objective and
launch. Larger files put supporting checks after the main workflow and signpost
each call. Choose the closest example, copy it, and edit the first three sections
to define your own experiment.

.. list-table:: Which example to copy
   :header-rows: 1
   :widths: 30 70

   * - File
     - Purpose
   * - ``start_here.py``
     - Small dielectric block; one parameter and a receiver-peak objective.
   * - ``optimiser_choices.py``
     - Change the optimiser while importing the same starter model and objective.
   * - ``waveform_matching.py``
     - Two parameters; load reference data, align waveforms and minimise normalised RMS error.
   * - ``thin_wire_dipole.py``
     - Change a symmetric dipole length so its S11 dip approaches 1 GHz.
   * - ``rectangular_patch.py``
     - Change patch length, width and feed offset; minimise S11 at 3.1 GHz.
   * - ``inspect_output.py``
     - Explore an existing output file and demonstrate public readers; no simulation.
   * - ``scipy_integer.py``
     - Direct SciPy helper for a single bounded integer.

``dielectric_block.py``, ``compare_dipole_optimisers.py`` and the ``advanced``
examples demonstrate explicit campaign/benchmark machinery. Start with the files
above before using those additional interfaces.

Waveform matching: custom processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Copy :download:`waveform_matching.py <../../gprMax/toolboxes/Optimisation/examples/waveform_matching.py>`
and :download:`reference_waveform.npz <../../gprMax/toolboxes/Optimisation/examples/reference_waveform.npz>`
to the same folder. The bundled data are synthetic, generated with width 12 mm
and relative permittivity 4. See
:download:`reference notes <../../gprMax/toolboxes/Optimisation/examples/reference_waveform.md>`.

Replace the reference file and loading block for measured data. Check time and
amplitude units, align sampling times and define the error calculation in
``evaluate``. The example saves ``comparison.npz`` for every scored trial so you
can inspect the reference, simulated trace and residual. The optimiser receives
only the normalised RMS error. Width is rounded to the example's 2 mm mesh by its
model builder; permittivity is continuous.

Thin-wire dipole: resonance location
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`thin_wire_dipole.py <../../gprMax/toolboxes/Optimisation/examples/thin_wire_dipole.py>`
uses two wire arms and a one-cell voltage-source feed gap. The adjustable
``arm_cells`` is 54–83 cells per arm. With fixed ``DZ_M=0.001``, total length is
``(2 * arm_cells + 1) * DZ_M``: 109–167 mm in symmetric 2 mm increments.

The objective is the absolute difference, in Hz, between the estimated S11-dip
frequency and 1 GHz. The helper at the end checks the spectrum and fits the dip
between bins. This differs from minimising S11 directly at 1 GHz. The 100 ns
time window gives approximately 10 MHz independent frequency resolution;
interpolation is an estimate within that simulation, not a finer mesh or a
longer recording. If you refine dz, review the cell-count bounds to preserve the
physical lengths you intend to search.

Rectangular patch: literature-based validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`rectangular_patch.py <../../gprMax/toolboxes/Optimisation/examples/rectangular_patch.py>`
is based on N. Jin and Y. Rahmat-Samii, “Parallel particle swarm optimization and
finite-difference time-domain (PSO/FDTD) algorithm for multiband and wide-band
patch antenna designs,” *IEEE Transactions on Antennas and Propagation*,
53(11), 3459–3468, 2005.
`Paper DOI <https://doi.org/10.1109/TAP.2005.858842>`_. The example follows the
single-frequency rectangular-patch case in Section III and compares against
Antenna I in Table I. It is an adaptation with an equivalent magnetic-frill feed.

The fixed board/ground is 60 × 60 mm, substrate height 3 mm and relative
permittivity 2.2. The mesh is 1 mm and the time window 100 ns. A 3 mm thin-wire
probe spans the substrate, with a magnetic-frill source representing the coaxial
aperture at the ground. Its assumed radius is 0.2 mm and reference impedance
50 ohms. These feed assumptions are documented in the source and
:download:`patch notes <../../gprMax/toolboxes/Optimisation/examples/rectangular_patch.md>`.

.. list-table:: Physical parameter ranges, in millimetres
   :header-rows: 1
   :widths: 26 37 37

   * - Parameter
     - Paper's stated space
     - Reusable example
   * - Patch length L
     - 0 < L < 44
     - 26, 28, 30, 32, 34, 36
   * - Patch width W
     - 0 < W < 44
     - 12, 14, …, 44
   * - Feed offset x
     - 0 < x < 22; x < L/2
     - 1, 2, …, 12; always inside the patch

The feed offset is measured **from the patch centre**, not the edge. Positive x
in the parameter dictionary moves the feed toward the model's negative x axis.
Even L/W keep the centre and centreline on the fixed design lattice. These are
physical millimetre increments; ``Integer(..., step=2)`` expresses them through
the public interface for all five optimisers. This is a restricted benchmark
space, not the paper's full search.

The criterion is ``20 * log10(abs(S11(3.1 GHz)))``. Lower is better. The paper's
single-frequency fitness adds 50 to this dB value; that constant does not change
which candidate minimises it. The example does not optimise the broadband
Antenna II criterion.

Both tested searches found **L = 30 mm, W = 18 mm, x = 4 mm**, the Table I
Antenna I dimensions. The independently simulated reference and the selected
result in this model give **S11 = -31.5375 dB at 3.1 GHz**; the paper reports
approximately -28 dB. The agreement is in the design dimensions, not an exact
reproduction of the paper's feed, spectrum or PSO implementation.

.. list-table:: Recorded runs, seed 7 and ten candidates per population
   :header-rows: 1
   :widths: 14 27 21 20 18

   * - Method
     - First reference design
     - Total proposals
     - New simulations
     - Stop
   * - PSO
     - Population 31, proposal 304
     - 550
     - 68
     - User stopped after population 55
   * - DE
     - Population 47, proposal 467
     - 470
     - 100
     - Reference score reached

The reference geometry was not seeded. Cached outputs were reused only after a
candidate was proposed; counts of new simulations therefore differ from proposal
counts. PSO used an optional restart after 20 stagnant populations, which first
occurred after population 51, after finding the reference. DE used
``DE/rand/1/bin``. The two single-seed runs validate the connection between the
optimisers, geometry and objective; they do not establish an algorithm ranking.

.. figure:: ../../gprMax/toolboxes/Optimisation/examples/images/patch_comparison.png
   :alt: Recorded PSO and DE patch convergence and best-design S11 comparison.
   :width: 95%

   Recorded benchmark runs. The experiment's cached runs and stopping settings
   differ from the short default example.

For a fresh DE run, save this as a script and use a new directory:

.. code-block:: python

   from gprMax.toolboxes.Optimisation import LocalPool
   from gprMax.toolboxes.Optimisation.examples.rectangular_patch import run_example

   if __name__ == "__main__":
       run_example(
           "results/patch_de", optimiser="de", evaluations=2000,
           population_size=10, seed=7, target_value=-30.0,
           execution=LocalPool(workers=2, cpu_threads_per_worker=2),
       )

This first checks one starting design, then starts the search. The starting
simulation is additional to the optimisation budget and does not seed the
optimiser. Change ``"de"`` to ``"pso"`` or another supported method. Set
``target_value=None`` to use the evaluation limit alone. The source file's
default budget is only 40 proposals; it is a short trial and need not find the
reference. Fresh 100 ns simulations can make longer runs take hours.

The reusable interface does not preload the experiment's private caches or
resume its checkpoints. The model still needs mesh/time-window and feed checks
for any new antenna application. This benchmark validates framework operation
on this model; it is not a mesh-convergence study.
