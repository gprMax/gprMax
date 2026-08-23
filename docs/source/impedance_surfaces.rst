.. _impedance-surfaces:

****************************************************
Surface-Impedance Volumes, ADEs, and Modal Injection
****************************************************

Purpose and present scope
=========================

The surface-impedance implementation represents an electrically opaque
conductor without filling its skin depth with FDTD cells. The conductor
interior is excluded from the field update and its boundary obeys a local,
scalar surface-impedance condition. A rational auxiliary differential
equation (ADE) supplies frequency dispersion. The same time-discrete ADE is
reduced harmonically for the FDFD eigenmode solve, so the mode launched into
FDTD sees the boundary law that FDTD actually advances.

Three user objects have distinct roles:

* :class:`gprMax.SurfaceImpedance` defines a reusable boundary material;
* :class:`gprMax.ImpedanceBox` directly marks an axis-aligned closed volume;
* :class:`gprMax.ImpedanceVolume` converts the surviving cells of a tagged
  volumetric object.

Despite the feature name, the implemented geometry is a **one-sided boundary
of a closed opaque volume**. It is not a zero-thickness, transmissive sheet.
Fields on the conductor side do not exist. A sheet separating two retained
field regions needs a two-sided sheet transition condition and is discussed
under :ref:`impedance-future-work`.

The first implementation is intended for microwave and radio-frequency
models in which a local scalar surface impedance is appropriate. In
particular, the common-metal presets describe thick, smooth, non-magnetic
bulk metal at 293 K. They are not thin-film, rough-surface, alloy,
temperature-dependent, ferromagnetic, or optical material models.

Quick start
===========

Python API
----------

The shortest model is a constant surface resistance on a closed box:

.. code-block:: python

    import gprMax

    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.30, 0.20, 0.15)))
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.TimeWindow(time=8e-9))

    scene.add(gprMax.SurfaceImpedance(id='wall', resistance=50.0))
    scene.add(gprMax.ImpedanceBox(
        p1=(0.10, 0.07, 0.05),
        p2=(0.20, 0.13, 0.10),
        surface_impedance_id='wall',
    ))

A common-metal preset creates a passive dispersive realization over a stated
band:

.. code-block:: python

    scene.add(gprMax.SurfaceImpedance(
        id='copper_wall',
        preset='copper',
        fit_fmin_hz=8e9,
        fit_fmax_hz=12e9,
        fit_order=16,
    ))

The accepted preset names are ``copper``, ``silver``, and ``gold``. The
case-insensitive element-symbol aliases ``Cu``, ``Ag``, and ``Au`` are also
accepted. If the fit limits are omitted, the defaults are 1 MHz and 100 GHz.
``fit_order`` is the number of candidate Foster relaxation poles and must be
between 4 and 64. Numerically negligible branches are removed, so the stored
realization order can be smaller than the candidate order.

Any supported cell-occupying geometry can become an impedance volume through
its tag:

.. code-block:: python

    scene.add(gprMax.SurfaceImpedance(id='silver_body', preset='silver'))
    scene.add(gprMax.Sphere(
        p1=(0.15, 0.10, 0.075),
        r=0.025,
        material_id='free_space',
        tag='target_body',
    ))
    scene.add(gprMax.ImpedanceVolume(
        geometry_tag='target_body',
        surface_impedance_id='silver_body',
    ))

``ImpedanceVolume`` must occur after all geometry that forms the selected tag.
It converts only cells which still carry that tag at that point in scene
order. Several primitives can use one tag to form a union. An ordinary object
can overwrite part of the tag before conversion to form a cavity, and later
geometry can overwrite impedance markers under the usual last-object-wins
rule.

The Python API also accepts a real state-space model. For example, the
following stable first-order model tends from 70 Ohms at DC to 50 Ohms at
high frequency:

.. code-block:: python

    import numpy as np

    rate = 2 * np.pi * 1e9
    scene.add(gprMax.SurfaceImpedance(
        id='first_order_wall',
        A=((-rate,),),
        B=(rate,),
        C=(20.0,),
        D=50.0,
        fit_fmin_hz=1e7,
        fit_fmax_hz=5e9,
    ))

The matrix ``A`` and vectors ``B`` and ``C`` must be real. ``A`` must be
strictly Hurwitz, and the realization dimensions must agree. The default
path rejects a negative feedthrough and checks discrete passivity during grid
construction. ``allow_active=True`` is an expert escape hatch for research
models; it removes the passivity guarantee and should not be used for an
ordinary passive boundary.

Hash-command input
------------------

The corresponding material forms are:

.. code-block:: text

    #surface_impedance: resistive_wall 50
    #surface_impedance: copper_default copper
    #surface_impedance: copper_xband Cu 8e9 12e9 16

The text format does not accept arbitrary ``A, B, C, D`` matrices; use the
Python API for a custom realization. Apply a material with either a direct
box or a tag conversion:

.. code-block:: text

    #impedance_box: 0.10 0.07 0.05 0.20 0.13 0.10 resistive_wall

    #sphere: 0.15 0.10 0.075 0.025 free_space n target_body
    #impedance_volume: target_body copper_default

See :ref:`input-hash-cmds` and :ref:`input-api` for the complete command and
constructor signatures.

Geometry semantics
==================

Supported volume sources
------------------------

``ImpedanceBox`` is the simplest and fastest way to express an axis-aligned
rectangular volume. The tag-driven path is deliberately independent of the
primitive which wrote the cell map. It supports the current volumetric
voxelizers, including boxes, spheres, ellipsoids, axis-aligned and oblique
cylinders and cones when their rasterized boundaries are manifold,
finite-thickness cylindrical sectors, triangular prisms, fractal boxes, and
tagged imported voxel geometry.

The final rasterized cells are authoritative. Curved and oblique surfaces are
therefore represented by the same Yee-aligned staircase as other gprMax
geometry. Two impedance regions that touch through a face are one opaque
excluded region at that interface; an internal metal-to-metal face is not
given a second boundary condition. Disconnected tagged bodies are permitted
provided every resulting boundary is closed and manifold.

The volume must occupy at least one cell along every non-empty axis and there
must be at least one retained cell between the complete impedance region and
each domain boundary. Its surface must not intersect a PML.

Yee degrees of freedom
----------------------

Let a candidate electric edge be oriented along Cartesian axis ``a``. Four
cell-centred quadrants surround that edge in the transverse ``b-c`` plane.
The compiler classifies the edge as follows:

* zero metal quadrants: use the ordinary Yee update;
* four metal quadrants: remove the electric degree of freedom;
* one to three metal quadrants: retain one boundary electric degree of
  freedom and compile a sparse impedance row.

The retained dielectric quadrants determine the electric mass and the
conductive mass,

.. math::

   m_{\epsilon,e}
      = \sum_{q\in\mathcal R(e)}
        \epsilon_0\epsilon_{r,q}\frac{\Delta b\Delta c}{4},
   \qquad
   m_{\sigma,e}
      = \sum_{q\in\mathcal R(e)}
        \sigma_q\frac{\Delta b\Delta c}{4}.

Thus a flat boundary retains one half of the ordinary dual area. A convex
box edge retains three quarters; re-entrant staircase configurations can
retain one quarter. Heterogeneous, non-dispersive retained quadrants are
integrated independently. A dispersive dielectric immediately outside the
boundary is currently rejected because its volume memory variables have not
yet been coupled to the locally implicit surface row.

Each retained quadrant also contributes half-line magnetic-circulation
segments. Duplicate segments are coalesced. Every metal/dielectric face
adjacent to the electric edge creates a surface-current port, so one electric
edge at a staircase corner can own more than one independent ADE state. The
edge is shared, but there is one port per adjacent face and per surface model.

Interior H components are assigned zero update coefficients. At an interface,
the normal H degree of freedom remains associated with the retained material;
tangential boundary action is supplied by the surface current. A pattern in
which only two diagonally opposite quadrants are metal is non-manifold at a
Yee edge and is rejected. Refining the mesh or increasing a thin object's
thickness or radius normally removes this ambiguity.

Supported solver configurations
-------------------------------

The current implementation supports three-dimensional main-grid CPU models.
The following combinations are deliberately rejected:

* CUDA, OpenCL, and Metal field solvers;
* MPI domain decomposition and subgrids;
* symmetry boundaries or thin wires in the same grid;
* an impedance boundary which intersects a PML;
* an electric source, rational-network terminal, or transmission-line edge
  which overlaps a boundary electric edge;
* a dispersive retained material immediately outside the boundary;
* a ``VirtualWaveguide`` termination.

An axial discrete plane wave is unsupported because it samples the completed
geometry to construct its layered auxiliary line. A homogeneous vector/angle
discrete plane wave is supported only when the complete impedance boundary is
strictly inside its total-field/scattered-field box.

A direct three-dimensional :class:`gprMax.EigenmodePort` may cross an
impedance guide. The guide boundary must be invariant along the propagation
direction through both cells adjacent to the modal plane, and the modal
window must contain the complete retained aperture and every required
boundary H degree of freedom. A guide end cap normal to the propagation axis
therefore cannot cross the solve plane.

Conventions and continuous boundary model
=========================================

Time, propagation, and current
------------------------------

gprMax uses

.. math::

   \mathbf F(\mathbf r,t)
      = \operatorname{Re}\left\{
        \widetilde{\mathbf F}(\mathbf r)e^{+j\omega t}\right\},
   \qquad
   \widetilde F(\omega)=\int F(t)e^{-j\omega t}\,\mathrm dt.

A forward mode in local coordinate ``w`` varies as

.. math::

   \widetilde{\mathbf F}(u,v,w)
      = \widetilde{\mathbf F}(u,v)e^{-j\beta w},
   \qquad
   \beta=\beta_r-j\alpha,
   \quad \alpha>0.

Consequently ``Im(beta)`` and ``Im(n_eff)`` are negative for a passive lossy
forward mode, and

.. math::

   \alpha=-\operatorname{Im}\beta
          =-k_0\operatorname{Im}n_{\mathrm{eff}}.

Let :math:`\hat{\mathbf n}_m` point from the excluded metal into the retained
dielectric. The surface current and scalar boundary law are

.. math::

   \mathbf K=\hat{\mathbf n}_m\times\mathbf H,
   \qquad
   \mathbf E_t=Z_s\mathbf K.

For an electric edge with unit tangent :math:`\hat{\mathbf t}_p`, one compiled
port uses

.. math::

   e_p=\hat{\mathbf t}_p\cdot\mathbf E_t,
   \qquad
   k_p=\hat{\mathbf t}_p\cdot\mathbf K.

This convention makes the time-average surface loss

.. math::

   P_{\mathrm{wall}}
      =\frac12\operatorname{Re}
        \int_\Gamma \mathbf E_t\cdot\mathbf K^*\,\mathrm dS,

which is non-negative when :math:`\operatorname{Re}Z_s\ge 0`.

Rational state-space impedance
------------------------------

The reusable continuous model is

.. math::

   \widehat Z(s)=D+C(sI-A)^{-1}B,
   \qquad s=j\omega.

It is impedance, rather than admittance, because retaining the surface
current in the local boundary relation avoids fitting a very large
:math:`1/Z_s` for a good conductor. The implementation accepts a proper real
realization only: there is no proportional :math:`sE` term. Dynamic models
may declare a finite validity band; the default custom-model band is unbounded,
while metal presets always use a finite fit band. FDTD can advance a
realization over its whole discrete spectrum after it passes the passivity
check, but accuracy outside a finite fitted band is the user's responsibility.
An impedance-aware FDFD solve refuses to extrapolate a declared physical or
bilinear-warped evaluation frequency.

At construction time gprMax checks that coefficients are finite, dimensions
agree, and every eigenvalue of ``A`` has strictly negative real part. Unless
``allow_active`` is selected, the direct term must be non-negative. At the
actual FDTD time step the code additionally checks the mapped unit-circle
response for negative real impedance.

Common-metal Foster presets
---------------------------

For a thick good conductor, the target under the stated time convention is

.. math::

   Z_{\mathrm{gc}}(j\omega)
      =(1+j)\sqrt{\frac{\omega\mu_0}{2\sigma}}
      =(1+j)\sqrt{\pi f\mu_0\rho}.

The stored 293 K bulk-pure-metal resistivities are:

.. list-table:: Common-metal preset data
   :header-rows: 1
   :widths: 18 24 24

   * - Preset
     - Resistivity (Ohm metre)
     - Conductivity (S/m)
   * - copper
     - :math:`1.676\times10^{-8}`
     - :math:`5.966\times10^{7}`
   * - silver
     - :math:`1.586\times10^{-8}`
     - :math:`6.305\times10^{7}`
   * - gold
     - :math:`2.192\times10^{-8}`
     - :math:`4.562\times10^{7}`

The values come from the recommended resistivity tables of Matula at 293 K
[MAT1979]_. The measured reference quantity is stored as
resistivity and inverted only when constructing the target impedance.

The fit uses a positive-real Foster form

.. math::

   \widehat Z(s)
      =R_0+\sum_{m=1}^{N}R_m\frac{s}{s+a_m},
   \qquad R_0,R_m\ge0,\quad a_m>0.

Candidate relaxation rates span two decades below and above the requested
fit band. Non-negative least squares fits real and imaginary parts with
relative-error weighting. Because every coefficient is non-negative, the
result is passive over the complete frequency axis, not only at the sample
points. The advertised maximum fit error still applies only inside the
requested band.

The realization stored by gprMax is

.. math::

   A=-\operatorname{diag}(a_m),\qquad
   B_m=\sqrt{R_ma_m},\qquad
   C_m=-\sqrt{R_ma_m},\qquad
   D=R_0+\sum_m R_m.

This scaling balances the first-order input and output coupling while
preserving the Foster transfer function.

Exact trapezoidal ADE
=====================

Continuous state equation
-------------------------

For one boundary port, the continuous state relation is

.. math::

   \dot{\mathbf x}=A\mathbf x+B k,
   \qquad
   e=C\mathbf x+Dk.

The state is convolution memory for the surface law; it is not an electric or
magnetic field inside the conductor. gprMax stores :math:`e^n` and
:math:`\mathbf x^n` at integer electric times and :math:`k^{n+1/2}` at the
magnetic half time.

Discrete runtime coefficients
-----------------------------

Trapezoidal integration gives

.. math::

   \mathbf x^{n+1}=F\mathbf x^n+Gk^{n+1/2},

where

.. math::

   F=\left(I-\frac{\Delta t}{2}A\right)^{-1}
     \left(I+\frac{\Delta t}{2}A\right),
   \qquad
   G=\left(I-\frac{\Delta t}{2}A\right)^{-1}\Delta t B.

The impedance output is centred at the same half time as the surface current:

.. math::

   \frac{e^{n+1}+e^n}{2}
      =L\mathbf x^n+Z_0 k^{n+1/2},

.. math::

   L=\frac12C(I+F),
   \qquad
   Z_0=D+\frac12CG.

``F``, ``G``, ``L``, and ``Z0`` are the exact runtime data shared by the FDTD
kernel and the FDFD reduction. gprMax requires ``Z0`` to be finite and
strictly positive. A constant resistance is the order-zero case:
``F``, ``G``, and ``L`` are empty and ``Z0`` is the resistance.

Discrete passivity
------------------

The trapezoidal map sends a discrete phase :math:`\theta` to

.. math::

   s_b=j\frac{2}{\Delta t}\tan\left(\frac{\theta}{2}\right).

For passive user models, gprMax samples this warped response from DC towards
the unit-circle Nyquist point and requires a non-negative real impedance to
floating-point tolerance. It also requires a strictly Hurwitz continuous
``A``. These checks complement, rather than replace, ordinary mesh
convergence and long-time decay tests.

Sparse locally implicit FDTD algorithm
======================================

Integral Ampere row
-------------------

The geometry compiler creates one integrated Ampere row per retained boundary
electric edge:

.. math::

   \left(\frac{m_{\epsilon,e}}{\Delta t}
          +\frac{m_{\sigma,e}}{2}\right)e^{n+1}
   =\left(\frac{m_{\epsilon,e}}{\Delta t}
          -\frac{m_{\sigma,e}}{2}\right)e^n
    +r_H^{n+1/2}
    +\sum_{p\in\mathcal P(e)}g_pk_p^{n+1/2}.

Here :math:`r_H` is the clipped line integral of the ordinary retained H
samples. :math:`g_p` includes the oriented surface-current line metric; with
the current compiler convention it is the negative of the accumulated
positive face length. Surface normals and magnetic signs are generated from
the voxel topology rather than hard-coded in the ADE kernel.

Define

.. math::

   a_+=\frac{m_\epsilon}{\Delta t}+\frac{m_\sigma}{2},
   \qquad
   a_-=\frac{m_\epsilon}{\Delta t}-\frac{m_\sigma}{2},
   \qquad
   h_p^n=L_p\mathbf x_p^n.

The scalar port relation gives

.. math::

   k_p^{n+1/2}
      =\frac{\tfrac12(e^{n+1}+e^n)-h_p^n}{Z_{0,p}}.

Substitution eliminates every port current from the local solve. The exact
expression executed by the Python reference path and Cython kernel is

.. math::

   d_e=a_+-\sum_p\frac{g_p}{2Z_{0,p}},

.. math::

   e^{n+1}=\frac{
      a_-e^n+r_H
      +\displaystyle\sum_p\left(
         \frac{g_pe^n}{2Z_{0,p}}-\frac{g_ph_p^n}{Z_{0,p}}
       \right)}{d_e}.

The kernel then recovers every :math:`k_p^{n+1/2}` and advances

.. math::

   \mathbf x_p^{n+1}=F_p\mathbf x_p^n+G_pk_p^{n+1/2}.

This analytic scalar elimination naturally handles several faces sharing one
electric edge; it does not perform a dense solve at each time step.

Packed data and update order
----------------------------

The compiler packs:

* boundary edge component/index, H range, and port range;
* :math:`a_+`, :math:`a_-`, and retained dual-area fraction;
* H component/index records and signed half-line weights;
* port model, unique state offset, :math:`g_p`, face normal, and face area;
* one copy of ``F, G, L, Z0`` for each used material model.

The dense material arrays use private ``surface-hold`` and ``volume-void``
rows. The ordinary electric update preserves a boundary value without
applying a full-cell curl, and zeros interior fields. After the magnetic
update and its source corrections, the ordinary electric stages and electric
source corrections run. The sparse impedance update then replaces every held
boundary E value with the locally implicit result. This makes the impedance
row authoritative and prevents a hard source from silently overwriting it;
explicit electric-edge overlaps are rejected during compilation.

Each boundary electric edge and all of its port-state slices have one owner.
The Cython implementation can therefore use OpenMP ``prange`` without atomics
or cross-edge state races. Runtime and state storage scale with boundary area
and rational-model order, rather than conductor volume:

.. math::

   \text{work}\sim O(N_{\Gamma,E}\overline N_p^2),
   \qquad
   \text{state}\sim O(N_{\Gamma,K}\overline N_p),

where the matrix-vector state update gives the quadratic order dependence for
a general dense realization. The metal presets have diagonal ``F``, although
the present packed kernel retains the general dense loop.

Exact surface-ADE reduction for FDFD
====================================

Algorithmic impedance
---------------------

For a physical solve frequency :math:`f`, set

.. math::

   \theta=2\pi f\Delta t,\qquad
   z=e^{j\theta},\qquad
   c_\theta=\cos(\theta/2),\qquad
   \Omega=\frac{2}{\Delta t}\sin(\theta/2).

The discrete state recurrence has the exact harmonic response

.. math::

   Z_{\mathrm{alg}}(f,\Delta t)
      =Z_0+L(zI-F)^{-1}G.

The midpoint boundary equation is

.. math::

   c_\theta\widetilde e
      =Z_{\mathrm{alg}}\widetilde k,
   \qquad
   Y_{\mathrm{alg}}
      =\frac{\widetilde k}{\widetilde e}
      =\frac{c_\theta}{Z_{\mathrm{alg}}}.

For a trapezoidal realization,

.. math::

   Z_{\mathrm{alg}}
      =\widehat Z\!\left(
         j\frac{2}{\Delta t}\tan\frac{\theta}{2}
       \right).

The mode frequency must be positive and below temporal Nyquist. For a dynamic
model, both :math:`f` and the bilinear-warped frequency

.. math::

   f_b=\frac{\tan(\pi f\Delta t)}{\pi\Delta t}

must lie inside the model's declared fit band. This catches a subtle form of
ADE extrapolation near Nyquist.

Clipped row in the existing P/Q solver
--------------------------------------

The implemented FDFD path eliminates the scalar surface currents into the
electric coefficient rather than appending them as eigenproblem unknowns.
For boundary edge ``e`` with retained dual area :math:`A_e`, attached port
lengths :math:`\ell_p`, and the same integrated masses used by FDTD, define

.. math::

   \epsilon_{r,e}^{\mathrm{eff}}
      =\frac{
         j\Omega m_{\epsilon,e}
         +c_\theta m_{\sigma,e}
         +\displaystyle\sum_p\ell_pY_{\mathrm{alg},p}
       }{
         j\omega\epsilon_0A_e
       }.

This makes the ordinary physical-frequency material term reproduce the exact
discrete-time surface load:

.. math::

   j\omega\epsilon_0A_e\epsilon_{r,e}^{\mathrm{eff}}
      =j\Omega m_{\epsilon,e}
       +c_\theta m_{\sigma,e}
       +\sum_p\ell_p\frac{c_\theta}{Z_{\mathrm{alg},p}}.

The standard rectangular finite-difference curl row is replaced by the
compiled clipped line circulation, normalized by :math:`A_e k_0`. Independent
retained masks remove metal-interior E and H degrees of freedom without
misusing PEC masks; in particular, interface-normal H can remain present when
a collocated tangential E is a valid impedance-boundary unknown. The existing
P/Q reduction then solves for complex :math:`n_{\mathrm{eff}}` and selects the
passive forward branch.

.. important::

   The surface ADE, midpoint factor, boundary electric mass, conductivity,
   and clipped transverse curl are reduced exactly for the FDTD time step.
   The surrounding FDFD eigensolver still uses physical :math:`\omega` and
   :math:`k_0` in its established P/Q normalization. The effective
   permittivity above maps the discrete boundary row into that normalization;
   this is **not** a fully time-discrete P/Q bulk Yee eigenproblem. Keep modal
   anchors comfortably below Nyquist and perform mesh/time-step convergence
   when bulk numerical dispersion is material to the result.

The longitudinal :math:`\beta` coupling remains the standard implicit P/Q
term. The source-plane mapper checks that the omitted longitudinal H weights
form equal and opposite contributions in the two cells adjacent to the plane.
That is why a changing wall cross-section or an impedance end cap at the
modal plane is rejected.

Eigenmode solution and FDTD injection
=====================================

A direct modal solve reuses the component IDs, retained masks, clipped H
weights, dual fractions, port models, and FDTD ``dt`` from the already
compiled three-dimensional grid. There is no separately redrawn FDFD wall.
This shared geometry is as important as sharing ``F, G, L, Z0``: a half-cell
area or sign mismatch would change both loss and mode phase.

For a rectangular impedance guide, a typical source/monitor definition is:

.. code-block:: python

    scene.add(gprMax.EigenmodeBand(
        id='copper_te10', fmin=8e9, fmax=12e9, points=21,
    ))
    scene.add(gprMax.EigenmodePort(
        port=1,
        p1=(0.04, guide_y0, guide_z0),
        p2=(0.04, guide_y1, guide_z1),
        direction='+',
        modes=(1,),
        anchors=(8e9, 9e9, 10e9, 11e9, 12e9),
        plot_fields=False,
    ))
    scene.add(gprMax.EigenmodePort(
        port=2,
        p1=(0.08, guide_y0, guide_z0),
        p2=(0.08, guide_y1, guide_z1),
        direction='-',
        modes=(1,),
        anchors=(8e9, 9e9, 10e9, 11e9, 12e9),
        plot_fields=False,
    ))
    scene.add(gprMax.EigenmodeExcitation(
        port=1, mode=1, waveform='auto', plot_waveform=False,
    ))

The complete four-wall geometry is shown in
``testing/validation/validate_impedance_copper_waveguide_s21.py``.

The mode solver returns fields on their native component-specific Yee grids.
Lossy walls generally produce genuinely complex field profiles. The source
therefore uses in-phase/quadrature synthesis when a single real profile is
insufficient. The equivalent-current TF/SF correction injects tangential E
into the magnetic update and tangential H into the electric update. Magnetic
coefficients include the :math:`\Delta t/2` temporal staggering and the
half-normal-cell spatial phase. Coordinate-basis handedness is applied when a
global y-normal plane gives a left-handed local ``(u,v,w)`` ordering.

Modal receivers project the total fields onto the same tracked FDFD basis and
separate forward and backward coefficients. For two downstream planes at
:math:`w_1` and :math:`w_2`, a uniform single-mode guide should give

.. math::

   \frac{b(w_2,f)}{b(w_1,f)}
      =\exp[-j\beta(f)(w_2-w_1)].

This two-plane ratio removes source amplitude and most startup sensitivity.
The source-plane ratio :math:`b_1/a_1` independently measures launch mismatch.
See :doc:`eigenmode_port` for broadband anchor tracking, one-watt
normalization, I/Q source synthesis, modal DFTs, and the complete HDF5 modal
schema.

Analytical rectangular-guide reference
--------------------------------------

For the lossless TE10 mode of a guide with width :math:`a`, height :math:`b`,
and air filling,

.. math::

   k=\frac{2\pi f}{c},\qquad
   k_c=\frac{\pi}{a},\qquad
   \beta_0=\sqrt{k^2-k_c^2}.

First-order wall perturbation gives the attenuation

.. math::

   \alpha(f)=\frac{R_s(f)}{\eta_0}
      \left[
        \frac{k}{\beta_0b}
        +\frac{2k_c^2}{k\beta_0a}
      \right].

For a complex local surface impedance, define the real geometry factor

.. math::

   Q(f)=\frac{1}{\eta_0}
      \left[
        \frac{k}{\beta_0b}
        +\frac{2k_c^2}{k\beta_0a}
      \right].

To first order,

.. math::

   \beta(f)\simeq
      \beta_0+Q\operatorname{Im}Z_s
      -jQ\operatorname{Re}Z_s,
   \qquad
   S_{21}^{\mathrm{theory}}(f)=e^{-j\beta(f)L}.

The physical copper reference uses
:math:`Z_s=(1+j)\sqrt{\pi f\mu_0\rho_{\mathrm{Cu}}}`. The discrete FDFD
operator test instead uses the exact algorithmic
:math:`Z_{\mathrm{eff}}=Z_{\mathrm{alg}}/c_\theta`, because its purpose is
to isolate FDFD/FDTD boundary compatibility at a finite time step. These are
different comparisons and should not be conflated.

The end-to-end copper validation also removes the known lossless spatial and
temporal dispersion of its cubic Yee grid from its diagnostic phase
comparison.
For cubic spacing :math:`\Delta`, its analytical lossless reference is

.. math::

   \beta_Y=\frac{2}{\Delta}\sin^{-1}\!\left\{
      \Delta\left[
        \left(\frac{\sin(\pi f\Delta t)}{c\Delta t}\right)^2
        -\left(\frac{\sin(\pi\Delta/(2a))}{\Delta}\right)^2
      \right]^{1/2}
   \right\}.

That diagnostic reference replaces :math:`\beta_0` by :math:`\beta_Y` in the
phase term while retaining the continuum perturbation factor :math:`Q`. The
pure continuum result is written separately so mesh dispersion remains
visible rather than being mistaken for a copper-boundary error. It is not a
release gate: a finite-record ratio between two passive planes can contain a
small projection ripple comparable with the already small conductor loss.

Validation and benchmarking
===========================

Normal-incidence reflection
---------------------------

``testing.validation.validate_impedance_box_reflection`` runs all six face
orientations. A loaded simulation is subtracted from a geometrically
identical free-space run, time-gated before edge diffraction, and de-embedded
with the axial Yee wavenumber. It compares against the exact algorithm
executed by the ADE and half-cell boundary, not a continuous Fresnel formula.

For a flat free-space wall with normal cell size :math:`\Delta`, define

.. math::

   Y_\Gamma
      =j\epsilon_0\frac{\Delta}{\Delta t}\sin(\theta/2)
       +\frac{c_\theta}{Z_{\mathrm{alg}}}.

If :math:`k_Y` is the axial Yee wavenumber, the reference at the boundary-E
plane is

.. math::

   \Gamma_{\mathrm{Yee}}
      =\frac{e^{+jk_Y\Delta/2}-\eta_0Y_\Gamma}
             {e^{-jk_Y\Delta/2}+\eta_0Y_\Gamma}.

This catches the three common errors separately: wrong surface-normal signs,
omitting bilinear ADE warping, and using a full rather than half electric dual
cell. The automated gates are 0.005 magnitude RMSE, 0.7 degrees phase RMSE,
and 0.01 complex relative L2 error for each requested face.

FDFD attenuation and modal launch
---------------------------------

The focused FDFD operator test constructs a copper-lined rectangular guide,
compares :math:`-k_0\operatorname{Im}n_{\mathrm{eff}}` with the TE10
perturbation result using :math:`Z_{\mathrm{alg}}/c_\theta`, and applies a 2%
relative tolerance.

``testing.validation.validate_impedance_modal_injection`` uses a deliberately
larger constant resistance so attenuation is measurable in a compact model.
It checks both the source-plane backward wave and the attenuation obtained
from two passive modal planes. Its acceptance gates are -20 dB maximum source
reflection and 12% relative L2 attenuation error. The wall ends are outside
the causal return window, so termination reflections cannot make the test
pass or fail.

``testing.validation.validate_impedance_copper_waveguide_s21`` is the
physical common-metal case. It uses a 1.6 mm by 0.8 mm copper-lined guide on a
0.1 mm cubic grid. TE10 is evaluated from 130 to 150 GHz, below the 187.37 GHz
next-mode cutoff, over a 20 mm reference-plane spacing. The independent
good-conductor formula predicts 0.102--0.117 dB insertion loss, making copper
loss materially larger than in the initial microwave-scale test.

The release checks deliberately separate the two requested milestones. First,
the attenuation :math:`-k_0\operatorname{Im}n_{\mathrm{eff}}` stored for each
exact FDFD anchor is compared with :math:`Q\operatorname{Re}Z_s`; the relative
L2 gate is 1%. Second, the driven FDTD port must have maximum
:math:`S_{11}<-20` dB after the complex modal field is injected. The raw
two-passive-plane propagation factor
:math:`T_{32}=b_3^+/b_2^+` and the perturbative
:math:`e^{-j\beta L}` curve are still written to CSV and plotted for
inspection, but are not acceptance gates. This distinction prevents a small
finite-record projection ripple from being misidentified as an error in the
copper ADE. The workflow still exercises the copper preset, Foster fit, exact
FDFD boundary reduction, complex modal source, FDTD ADE, and modal projection.

Run the validations from the repository root:

.. code-block:: console

    python -m testing.validation.validate_impedance_box_reflection --threads 4
    python -m testing.validation.validate_impedance_modal_injection --threads 4
    python -m testing.validation.validate_impedance_copper_waveguide_s21 --threads 4

Use ``--reuse`` to reanalyse compatible cached solver output when a driver
supports it. Each validation exits non-zero when an acceptance criterion
fails and writes numerical data plus a machine-readable summary below its
selected output directory.

Sparse-kernel performance
-------------------------

``testing.benchmarking.benchmark_impedance_box`` alternates otherwise
identical baseline and impedance runs, times the full solve, and also isolates
the sparse kernel and a bulk-plus-surface hot loop:

.. code-block:: console

    python -m testing.benchmarking.benchmark_impedance_box \
        --cells 80 --iterations 250 --threads 4 --repeats 3 \
        --kernel-iterations 1000 --hot-iterations 250 --hot-repeats 3

Report ``boundary_edges`` and ``surface_ports`` with timing results: overhead
depends on boundary area, fit order, compiler, CPU, thread count, and the
surface-to-volume ratio. The most useful outputs are sparse edge updates per
second, median solve overhead, and bulk-plus-surface hot-loop overhead.
Wall-clock construction time should not be mixed with time-stepping overhead.

HDF5 reproducibility metadata
=============================

Every output file containing a surface definition has a root group
``/surface_impedance_models``. Its attributes are:

* ``SchemaVersion``;
* ``TimeConvention = exp(+j*omega*t)``;
* ``FourierAnalysisConvention = exp(-j*omega*t)``;
* ``SurfaceNormalConvention = metal_to_retained_dielectric``;
* ``SurfaceCurrentConvention = K = n_m cross H``;
* ``Units = SI``;
* ``FDTDTimeStep``.

Each definition appears as ``model1``, ``model2``, and so on, in definition
order. A model group stores ``ID``, ``ModelHashSHA256``, ``Order``, ``D``, fit
limits, ``AllowActive``, ``UsedByCompiledBoundary``, ``Preset``,
``Provenance``, and, when available, ``FitMaximumRelativeError``. The
continuous ``A``, ``B``, and ``C`` arrays are datasets. A preset additionally
stores reference temperature, resistivity, and conductivity.

For every model used by the compiled boundary, ``fdtd_discrete`` stores the
exact ``F``, ``G``, and ``L`` arrays and the ``TimeStep``, ``Z0``, and
``PassivityChecked`` attributes. A defined but unused model has no discrete
subgroup. The model hash covers ID, continuous coefficients, fit limits,
preset, and provenance. The current schema does not serialize the complete
sparse geometry/port map, so geometry reproducibility still depends on the
input model and normal gprMax output metadata.

An eigenmode port additionally stores ``anchor_complex_neff`` below its
``/eigenmode_ports/portN`` group. Its complex array has shape
``(anchor frequency, mode)`` and is aligned with the
``CandidateAnchorFrequencies`` attribute. The associated
``anchor_mode_valid`` and ``anchor_mode_reference_valid`` datasets identify
the usable rows. This is the exact FDFD propagation constant bank used for
broadband interpolation and makes the FDFD attenuation comparison
reproducible from the output file.

For example, inspect one used realization with:

.. code-block:: python

    import h5py

    with h5py.File('model.h5', 'r') as output:
        models = output['surface_impedance_models']
        wall = models['model1']
        print(wall.attrs['ID'], wall.attrs['ModelHashSHA256'])
        A = wall['A'][...]
        B = wall['B'][...]
        C = wall['C'][...]
        F = wall['fdtd_discrete/F'][...]
        G = wall['fdtd_discrete/G'][...]
        L = wall['fdtd_discrete/L'][...]
        Z0 = wall['fdtd_discrete'].attrs['Z0']

Implementation map
==================

The main implementation seams are:

.. list-table:: Surface-impedance implementation files
   :header-rows: 1
   :widths: 42 58

   * - Module
     - Responsibility
   * - ``gprMax/impedance_surfaces.py``
     - Model validation/discretization, voxel-boundary compilation, packed
       records, and the Python reference update.
   * - ``gprMax/surface_impedance_presets.py``
     - Metal reference data, good-conductor target, and passive Foster NNLS
       fit.
   * - ``gprMax/cython/impedance_surface.pyx``
     - OpenMP sparse locally implicit FDTD update.
   * - ``gprMax/user_objects/cmds_geometry/impedance_box.py``
     - Direct axis-aligned marker geometry.
   * - ``gprMax/user_objects/cmds_geometry/impedance_volume.py``
     - Tag-driven conversion of arbitrary surviving volumetric voxels.
   * - ``gprMax/fdfd_eigenmode_solver/surface_impedance_operator.py``
     - Exact ADE harmonic response and boundary-row effective coefficient.
   * - ``gprMax/fdfd_eigenmode_solver/fdfd_2d_mode_solver.py``
     - Independent retained E/H masks and clipped P/Q curl-row replacement.
   * - ``gprMax/sources.py``
     - Maps the compiled three-dimensional boundary onto a direct modal plane
       and validates propagation invariance.
   * - ``gprMax/eigenmode_ports.py``
     - Broadband modal interpolation and HDF5 storage of the exact complex
       FDFD effective-index anchors and their validity masks.
   * - ``gprMax/fields_outputs.py``
     - Continuous/discrete model metadata and provenance.

Troubleshooting
===============

``A must be strictly Hurwitz``
    Move every continuous pole into the open left half-plane. A pole on the
    imaginary axis is not accepted, even if a short run appears bounded.

``non-positive discrete feedthrough Z0``
    The locally implicit elimination divides by ``Z0``. Use a proper passive
    realization with sufficient positive direct term, or re-express the fit
    in a positive-real Foster form. Do not add an arbitrary epsilon merely to
    bypass the check.

``non-passive on the discrete band``
    The trapezoidal unit-circle response has negative real impedance. Refit
    the model with passive constraints. ``allow_active=True`` disables this
    protection but also removes the passive-stability expectation.

``frequency is outside ... fit band``
    Expand the preset/custom fit band to include every eigenmode anchor and
    its bilinear-warped frequency. Close to Nyquist the warped frequency can
    be much higher than the physical anchor. Reducing ``dt`` also reduces the
    difference.

``voxel topology is non-manifold at a Yee edge``
    Two metal quadrants touch only diagonally. Refine the mesh, thicken the
    feature, increase a curved object's radius, or alter its position so the
    rasterized boundary has an unambiguous inside and outside.

``must have at least one retained cell on every side`` or PML intersection
    Move or shorten the impedance volume. The excluded region cannot touch a
    domain boundary, and the boundary itself cannot be inside PML cells.

``boundary does not yet support a dispersive retained material``
    Put a non-dispersive dielectric next to the wall in this implementation.
    A future extension must combine both the bulk polarization memory and the
    surface-current memory in the same clipped electric row.

``eigenmodes require a propagation-invariant boundary``
    Move the modal plane into a uniform section, extend the guide through both
    adjacent normal cells, remove an end cap at the plane, and make the modal
    window include the complete aperture.

Unexpected gain or positive ``Im(n_eff)``
    Check the :math:`e^{+j\omega t}` and :math:`e^{-j\beta w}` conventions,
    the metal-to-dielectric normal, fit passivity, and forward-mode selection.
    For the implemented convention, a passive forward mode has
    :math:`\operatorname{Im}n_{\mathrm{eff}}<0`.

Large source-plane reflection
    Confirm that source and monitor anchors cover the significant waveform
    band, the guide is uniform at the source plane, wall-end returns are
    outside the measurement gate, and the modal window contains all boundary
    DOFs. Then refine the Yee grid and repeat. Do not infer boundary failure
    from a trace contaminated by a nearby open guide end.

Metal result disagrees with measurement
    Check temperature, purity, alloying, plating thickness, surface roughness,
    magnetic permeability, and whether several skin depths fit inside the
    real conductor. The preset is a local semi-infinite good-conductor model,
    not a universal named-material database.

.. _impedance-future-work:

Extension guide
===============

Zero-thickness sheets
---------------------

A sheet must retain fields on both sides and impose a jump relation, for
example a generalized sheet transition condition. It cannot be implemented
by marking a one-cell-thick opaque volume: doing so removes interior degrees
of freedom, changes the physical thickness with mesh refinement, and creates
the wrong one-sided topology. A sheet extension should introduce two-sided
ports at selected Yee faces, define unambiguous ownership where sheets meet,
and derive both FDTD and FDFD rows from the same discrete recurrence.

Tensor and nonlocal surfaces
----------------------------

A local tensor impedance couples two tangential currents at one face. Replace
the scalar ``Z0`` elimination with a small block solve and store a vector ADE
state per face. A nonlocal :math:`Z(\omega,k_t)` additionally couples
neighbouring faces or introduces tangential surface derivatives; it cannot
reuse the independent-port kernel unchanged.

Accelerators and MPI
--------------------

An accelerator backend needs device-resident packed edge, port, model, and
state arrays plus a sparse boundary kernel after its electric update. MPI
needs deterministic ownership of a boundary E edge and its ADE states,
magnetic halo availability before the surface solve, and a cross-rank modal
plane assembly. The current compiler rejects these paths rather than silently
falling back to a different boundary.

Dispersive retained media and fully discrete bulk FDFD
------------------------------------------------------

Supporting a dispersive exterior requires a locally coupled row containing
both volume-polarization and surface-current states, including corner
averaging policy. A fully time-discrete modal solver would also replace the
remaining physical-frequency P/Q bulk symbols by the exact leapfrog temporal
and longitudinal spatial symbols. Those are separate extensions: the present
solver already makes the surface ADE exact, and its physical-frequency P/Q
bulk normalization must not be described as fully time discrete.

References
==========

The Yee grid convention follows [YEE1966]_. Background on
surface-impedance FDTD boundaries is given by [MAL1992]_ and [BEG1992]_. The
preset resistivity provenance is [MAT1979]_.
