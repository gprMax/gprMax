****************
Marimo toolbox
****************

The Marimo toolbox provides interactive dashboards for creating introductory
models, monitoring simulations, and inspecting A-scan and B-scan output. The
dashboards were contributed by Gaurav Sharma (``alphaleporus``) through Google
Summer of Code 2026.

Install the optional dependencies with:

.. code-block:: console

   $ pip install "gprMax[marimo]"

From a source checkout, an A-scan comparison can be generated and opened with:

.. code-block:: console

   $ python -m gprMax examples/gpr/basic/cylinder_Ascan_2D.in
   $ python -m gprMax toolboxes/Marimo/examples/cylinder_Ascan_2D_background.in
   $ marimo run toolboxes/Marimo/ascan_dashboard.py

The reference model is a target-free **background** model: it retains the
dielectric half-space and is not a free-space calculation. The subtraction is
valid only when the target and background runs otherwise have identical
geometry, excitation, receivers, grid, and time sampling.

The reusable readers respect the per-component sample interval and Yee-time
offset stored by current gprMax HDF5 output. Source excitation histories and
their metadata are also exposed. The introductory dashboards currently read
root-grid receivers; subgrid receiver groups are not yet presented in their
user interface.

The complete module list, processing assumptions, limitations, and examples
are given in the :download:`Marimo toolbox README
<../../toolboxes/Marimo/README.md>`.
