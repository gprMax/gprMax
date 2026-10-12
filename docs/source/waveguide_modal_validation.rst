C4v and C2v waveguides: analytical modal-profile validation
==========================================================================================

These undriven modal solves compare automatic tracking with independently
sampled analytical PEC fields. The **C4v square** is 20 × 20 mm and retains
18 modes at 39, 40, and 41 GHz. The **C2v 2:1 rectangle** is 20 × 10 mm and
retains 35 modes at 98, 100, and 102 GHz, including TE61 and TE23. Both have
air filling and +z propagation. The two transverse meshes have 64 and 96
cells across the width: square cells of 0.3125 and 0.208333 mm, respectively.
Thus the square uses 64 × 64 and 96 × 96 cells, and the rectangle uses
64 × 32 and 96 × 48 cells. Six extra candidate modes close the search bank.

Each geometry has PEC and copper counterparts. Copper uses conductivity
5.96 × 10⁷ S/m and a 16-term ADE surface-impedance fit over 1–120 GHz. The
boundary response is evaluated with the mesh's 0.99-Courant time step;
the interior modal operator uses continuous time and longitudinal propagation.
These runs assess the modal fields and tracking, without excitation or FDTD
time stepping. Copper attenuation is a solver result, not an independent
analytical attenuation validation.

What is compared
----------------

For transverse dimensions a and b, the analytical longitudinal fields are
Hz = cos(mπx/a) cos(nπy/b) for TE modes and
Ez = sin(mπx/a) sin(nπy/b) for TM modes. All transverse components follow
from Maxwell's equations with the continuum propagation constant
β² = (ω/c)² − (mπ/a)² − (nπ/b)². Each component is sampled at its own Yee
location. Analytical labels are assigned after tracking using longitudinal
Fourier coefficients; they never select or rotate the tracked basis.

The matching matrix uses all six complex field components, with
F = [Ex, Ey, Ez, Z₀Hx, Z₀Hy, Z₀Hz] concatenated over their native samples:

.. math::

   M_{ij}=\frac{|F_i^\dagger A_j|^2}{(F_i^\dagger F_i)(A_j^\dagger A_j)}.

This sampled field metric is invariant to global phase and amplitude. It is
not a power-wave scattering coefficient. Reference columns follow the
computed channels' dominant analytical labels; the computation itself is
not reordered. The mesh comparison plots the worst mismatch over all three
frequency anchors. Field galleries show phase-aligned, independently
peak-normalized Hz for a TE label or Ez for a TM label, at the middle anchor
on the finer mesh. The other components still enter the matching metric.

Copper can split PEC degeneracies and produce TE/TM hybrid modes. Therefore
its matching matrices retain off-diagonal weights, and its mesh plots use
projection onto the **complete equal-cutoff PEC subspace**. With Q an
orthonormal basis of those analytical fields, the subspace score is
‖Q†F‖²/‖F‖². A copper TE/TM label denotes its dominant PEC component, not an
exact TE/TM eigenmode. The dominant-component pictures alone do not show all
of that mixing. Finite conductivity leaves a physical difference from the
PEC reference even as the mesh is refined.

Recorded fine-mesh results
--------------------------

The minima below include every retained mode and all three anchors on the
96-cell-wide mesh. The Maxwell residual is evaluated on the final tracked
fields. JSON reports retain both meshes, individual scores, complex effective
indices, source hashes, and copper attenuation.

.. list-table:: Double-precision CPU results
   :header-rows: 1

   * - Case
     - Minimum individual overlap
     - Minimum PEC-subspace overlap
     - Maximum Maxwell residual
   * - C4v square PEC
     - 0.999999951
     - 0.999999959
     - 7.133e-13
   * - C2v rectangle 2to1 PEC
     - 0.999999511
     - 0.999999652
     - 2.391e-14
   * - C4v square copper
     - 0.999995382
     - 0.999999885
     - 6.838e-10
   * - C2v rectangle 2to1 copper
     - 0.522481989
     - 0.999998531
     - 4.287e-12

The PEC 2:1 rectangle retains separate TE61 and TE23 channels despite their
equal cutoff, since 6² + 4·1² = 2² + 4·3². Their TE/TM partners form a
four-dimensional parent group. The field comparisons and matching matrix
show the separate profiles after automatic subdivision. Copper lifts the
accidental degeneracy; the lossy eigenmodes are tracked without forcing
unequal-eigenvalue modes into a PEC-degenerate basis.

Reproduce the examples
----------------------

Run from the repository root in the gprMax Python environment:

.. code-block:: console

   python examples/features/eigenmode_ports/example_11_C4v_square/tracked_modes.py --output results/C4v
   python examples/features/eigenmode_ports/example_12_C2v_rectangle_2to1/tracked_modes.py --output results/C2v

Each command runs both wall materials. Use ``--wall PEC`` or ``--wall copper``
to select one. The shared implementation is
``testing/validation/eigenmode_tracking/profiles/waveguide_modal_profiles.py``. Outputs include all six
complex native Yee components (NPZ), tracking diagnostics and run provenance
(JSON), modal-field galleries, matching matrices, and copper attenuation.
Choose a fresh output directory to keep separate run provenance.

C4v square PEC
--------------

:download:`Recorded results <../../testing/validation/eigenmode_tracking/profiles/results/C4v_square_PEC/results.json>`.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C4v_square_PEC/C4v_square_PEC_matching.png
   :alt: C4v square PEC: full-vector matching to PEC analytical fields and mesh comparison. The right panel measures individual-profile overlap.
   :width: 100%

   C4v square PEC: full-vector matching to PEC analytical fields and mesh comparison. The right panel measures individual-profile overlap.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C4v_square_PEC/C4v_square_PEC_profiles_1.png
   :alt: C4v square PEC: tracked modes 1–6, alongside their PEC analytical longitudinal reference components.
   :width: 100%

   C4v square PEC: tracked modes 1–6, alongside their PEC analytical longitudinal reference components.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C4v_square_PEC/C4v_square_PEC_profiles_2.png
   :alt: C4v square PEC: tracked modes 7–12, alongside their PEC analytical longitudinal reference components.
   :width: 100%

   C4v square PEC: tracked modes 7–12, alongside their PEC analytical longitudinal reference components.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C4v_square_PEC/C4v_square_PEC_profiles_3.png
   :alt: C4v square PEC: tracked modes 13–18, alongside their PEC analytical longitudinal reference components.
   :width: 100%

   C4v square PEC: tracked modes 13–18, alongside their PEC analytical longitudinal reference components.

C2v rectangle 2to1 PEC
----------------------

:download:`Recorded results <../../testing/validation/eigenmode_tracking/profiles/results/C2v_rectangle_2to1_PEC/results.json>`.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C2v_rectangle_2to1_PEC/C2v_rectangle_2to1_PEC_matching.png
   :alt: C2v rectangle 2to1 PEC: full-vector matching to PEC analytical fields and mesh comparison. The right panel measures individual-profile overlap.
   :width: 100%

   C2v rectangle 2to1 PEC: full-vector matching to PEC analytical fields and mesh comparison. The right panel measures individual-profile overlap.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C2v_rectangle_2to1_PEC/C2v_rectangle_2to1_PEC_profiles_1.png
   :alt: C2v rectangle 2to1 PEC: tracked modes 1–6, alongside their PEC analytical longitudinal reference components.
   :width: 100%

   C2v rectangle 2to1 PEC: tracked modes 1–6, alongside their PEC analytical longitudinal reference components.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C2v_rectangle_2to1_PEC/C2v_rectangle_2to1_PEC_profiles_2.png
   :alt: C2v rectangle 2to1 PEC: tracked modes 7–12, alongside their PEC analytical longitudinal reference components.
   :width: 100%

   C2v rectangle 2to1 PEC: tracked modes 7–12, alongside their PEC analytical longitudinal reference components.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C2v_rectangle_2to1_PEC/C2v_rectangle_2to1_PEC_profiles_3.png
   :alt: C2v rectangle 2to1 PEC: tracked modes 13–18, alongside their PEC analytical longitudinal reference components.
   :width: 100%

   C2v rectangle 2to1 PEC: tracked modes 13–18, alongside their PEC analytical longitudinal reference components.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C2v_rectangle_2to1_PEC/C2v_rectangle_2to1_PEC_profiles_4.png
   :alt: C2v rectangle 2to1 PEC: tracked modes 19–24, alongside their PEC analytical longitudinal reference components.
   :width: 100%

   C2v rectangle 2to1 PEC: tracked modes 19–24, alongside their PEC analytical longitudinal reference components.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C2v_rectangle_2to1_PEC/C2v_rectangle_2to1_PEC_profiles_5.png
   :alt: C2v rectangle 2to1 PEC: tracked modes 25–30, alongside their PEC analytical longitudinal reference components.
   :width: 100%

   C2v rectangle 2to1 PEC: tracked modes 25–30, alongside their PEC analytical longitudinal reference components.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C2v_rectangle_2to1_PEC/C2v_rectangle_2to1_PEC_profiles_6.png
   :alt: C2v rectangle 2to1 PEC: tracked modes 31–35, alongside their PEC analytical longitudinal reference components.
   :width: 100%

   C2v rectangle 2to1 PEC: tracked modes 31–35, alongside their PEC analytical longitudinal reference components.

C4v square copper
-----------------

:download:`Recorded results <../../testing/validation/eigenmode_tracking/profiles/results/C4v_square_copper/results.json>`.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C4v_square_copper/C4v_square_copper_matching.png
   :alt: C4v square copper: full-vector matching to PEC analytical fields and mesh comparison. The right panel measures complete PEC-subspace overlap.
   :width: 100%

   C4v square copper: full-vector matching to PEC analytical fields and mesh comparison. The right panel measures complete PEC-subspace overlap.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C4v_square_copper/C4v_square_copper_attenuation.png
   :alt: C4v square copper: attenuation at the middle frequency on both meshes.
   :width: 100%

   C4v square copper: attenuation at the middle frequency on both meshes.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C4v_square_copper/C4v_square_copper_profiles_1.png
   :alt: C4v square copper: tracked modes 1–6, alongside their PEC analytical longitudinal reference components.
   :width: 100%

   C4v square copper: tracked modes 1–6, alongside their PEC analytical longitudinal reference components.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C4v_square_copper/C4v_square_copper_profiles_2.png
   :alt: C4v square copper: tracked modes 7–12, alongside their PEC analytical longitudinal reference components.
   :width: 100%

   C4v square copper: tracked modes 7–12, alongside their PEC analytical longitudinal reference components.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C4v_square_copper/C4v_square_copper_profiles_3.png
   :alt: C4v square copper: tracked modes 13–18, alongside their PEC analytical longitudinal reference components.
   :width: 100%

   C4v square copper: tracked modes 13–18, alongside their PEC analytical longitudinal reference components.

C2v rectangle 2to1 copper
-------------------------

:download:`Recorded results <../../testing/validation/eigenmode_tracking/profiles/results/C2v_rectangle_2to1_copper/results.json>`.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C2v_rectangle_2to1_copper/C2v_rectangle_2to1_copper_matching.png
   :alt: C2v rectangle 2to1 copper: full-vector matching to PEC analytical fields and mesh comparison. The right panel measures complete PEC-subspace overlap.
   :width: 100%

   C2v rectangle 2to1 copper: full-vector matching to PEC analytical fields and mesh comparison. The right panel measures complete PEC-subspace overlap.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C2v_rectangle_2to1_copper/C2v_rectangle_2to1_copper_attenuation.png
   :alt: C2v rectangle 2to1 copper: attenuation at the middle frequency on both meshes.
   :width: 100%

   C2v rectangle 2to1 copper: attenuation at the middle frequency on both meshes.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C2v_rectangle_2to1_copper/C2v_rectangle_2to1_copper_profiles_1.png
   :alt: C2v rectangle 2to1 copper: tracked modes 1–6, alongside their PEC analytical longitudinal reference components.
   :width: 100%

   C2v rectangle 2to1 copper: tracked modes 1–6, alongside their PEC analytical longitudinal reference components.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C2v_rectangle_2to1_copper/C2v_rectangle_2to1_copper_profiles_2.png
   :alt: C2v rectangle 2to1 copper: tracked modes 7–12, alongside their PEC analytical longitudinal reference components.
   :width: 100%

   C2v rectangle 2to1 copper: tracked modes 7–12, alongside their PEC analytical longitudinal reference components.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C2v_rectangle_2to1_copper/C2v_rectangle_2to1_copper_profiles_3.png
   :alt: C2v rectangle 2to1 copper: tracked modes 13–18, alongside their PEC analytical longitudinal reference components.
   :width: 100%

   C2v rectangle 2to1 copper: tracked modes 13–18, alongside their PEC analytical longitudinal reference components.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C2v_rectangle_2to1_copper/C2v_rectangle_2to1_copper_profiles_4.png
   :alt: C2v rectangle 2to1 copper: tracked modes 19–24, alongside their PEC analytical longitudinal reference components.
   :width: 100%

   C2v rectangle 2to1 copper: tracked modes 19–24, alongside their PEC analytical longitudinal reference components.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C2v_rectangle_2to1_copper/C2v_rectangle_2to1_copper_profiles_5.png
   :alt: C2v rectangle 2to1 copper: tracked modes 25–30, alongside their PEC analytical longitudinal reference components.
   :width: 100%

   C2v rectangle 2to1 copper: tracked modes 25–30, alongside their PEC analytical longitudinal reference components.

.. figure:: ../../testing/validation/eigenmode_tracking/profiles/results/C2v_rectangle_2to1_copper/C2v_rectangle_2to1_copper_profiles_6.png
   :alt: C2v rectangle 2to1 copper: tracked modes 31–35, alongside their PEC analytical longitudinal reference components.
   :width: 100%

   C2v rectangle 2to1 copper: tracked modes 31–35, alongside their PEC analytical longitudinal reference components.
