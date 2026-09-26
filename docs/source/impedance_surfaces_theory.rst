.. _impedance-theory:

****************************************
Surface impedance: theory and validation
****************************************

For API parameters, model setup, and troubleshooting, start with
:doc:`impedance_surfaces`. This page defines the boundary model and its
discrete implementation, then records the validation evidence. Runtime
restrictions are listed once in the practical guide; a derivation here is
not a claim of support for additional geometries or backends.

The `published HTML guide <https://docs.gprmax.com/en/latest/impedance_surfaces_theory.html>`_
renders the equations and cross-references in full.

.. contents:: On this page
   :local:
   :depth: 2

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
          =-k_0\operatorname{Im}n_{\mathrm{eff}},
   \qquad k_0=\frac{2\pi f}{c}.

Let :math:`\hat{\mathbf n}_m` point from the excluded metal into the retained
dielectric. The surface current and scalar boundary law are

.. math::

   \mathbf K=\hat{\mathbf n}_m\times\mathbf H,
   \qquad
   \mathbf E_t=Z_s\mathbf K.

The boundary treatment uses the collocated tangential E and H method of
Kobidze [KOB2010]_. Both fields in the impedance condition are evaluated at
the conductor surface. In this implementation, the boundary magnetic field
is represented by the surface current :math:`\mathbf K` and coupled to the
boundary E degree of freedom through the clipped circulation and local ADE
update described below.

Here, a *surface-current port* is an internal boundary coupling, not a
user-defined voltage or eigenmode measurement port. For an electric edge
with unit tangent :math:`\hat{\mathbf t}_p`, it uses

.. math::

   e_p=\hat{\mathbf t}_p\cdot\mathbf E_t,
   \qquad
   k_p=\hat{\mathbf t}_p\cdot\mathbf K.

With this convention, the time-averaged power dissipated at the surface is

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
realization internally: there is no proportional :math:`sE` term. Users
select resistance, a preset, or conductivity rather than supplying the
realization coefficients. Every fitted model must specify a finite validity
band. FDTD can advance the passive realization over its whole discrete
spectrum, but the specified fit accuracy applies only inside that band. An
impedance-aware FDFD solve refuses to extrapolate a declared physical or
bilinear-warped evaluation frequency.

At construction time gprMax verifies that the generated coefficients are
finite, their dimensions are consistent, every eigenvalue of ``A`` has strictly negative
real part, and the direct term is non-negative. At the actual FDTD time step
the code additionally checks the mapped unit-circle response for negative
real impedance. Active surface impedances are not part of the public API.

.. _impedance-metal-theory:

Common-metal Foster presets
---------------------------

For a thick good conductor, the target under the stated time convention is

.. math::

   Z_{\mathrm{gc}}(j\omega)
      =(1+j)\sqrt{\frac{\omega\mu_0}{2\sigma}}
      =(1+j)\sqrt{\pi f\mu_0\rho}.

The stored resistivities for pure bulk metals at 293 K are:

.. list-table:: Common-metal preset data at 293 K
   :class: api-parameters
   :header-rows: 1
   :widths: 18 24 24 22

   * - Preset
     - Resistivity (Ohm metre)
     - Conductivity (S/m)
     - Source
   * - aluminium
     - :math:`2.650\times10^{-8}`
     - :math:`3.774\times10^{7}`
     - [DES1984A]_
   * - copper
     - :math:`1.676\times10^{-8}`
     - :math:`5.966\times10^{7}`
     - [MAT1979]_
   * - gold
     - :math:`2.192\times10^{-8}`
     - :math:`4.562\times10^{7}`
     - [MAT1979]_
   * - molybdenum
     - :math:`5.340\times10^{-8}`
     - :math:`1.873\times10^{7}`
     - [DES1984S]_
   * - palladium
     - :math:`1.054\times10^{-7}`
     - :math:`9.488\times10^{6}`
     - [MAT1979]_
   * - silver
     - :math:`1.586\times10^{-8}`
     - :math:`6.305\times10^{7}`
     - [MAT1979]_
   * - tungsten
     - :math:`5.280\times10^{-8}`
     - :math:`1.894\times10^{7}`
     - [DES1984S]_
   * - zinc
     - :math:`5.964\times10^{-8}`
     - :math:`1.677\times10^{7}`
     - [DES1984S]_

The measured reference quantity is stored as resistivity and inverted only
when constructing the target impedance. Named presets describe pure bulk
metal at the reference temperature, not an alloy or plated finish.

The fit uses a positive-real Foster form

.. math::

   \widehat Z(s)
      =R_0+\sum_{m=1}^{N}R_m\frac{s}{s+a_m},
   \qquad R_0,R_m\ge0,\quad a_m>0.

The target is first normalised by the lower fit frequency and by its impedance
scale. For each requested runtime order, deterministic logarithmic relaxation
grids with several out-of-band extensions are tested. Active subsets from
slightly overcomplete grids supply additional non-uniform starting points.
The pole locations are then refined in logarithmic frequency by deterministic
bounded Powell searches. At every nonlinear evaluation, column-scaled bounded
least squares fits the direct term and Foster residues to the real and
imaginary parts of the target with relative-error weighting. The residues remain
non-negative, so the result is passive over the complete frequency axis, not
only at the sample points. A separate grid of at least 16,385 points certifies
the reported maximum and RMS errors.

Automatic order selection tests Foster models with 1 through 64 poles in
ascending order. It stops at the first count whose deterministic local searches
produce a certified maximum complex relative error no larger than
``fit_tolerance``. This is a sequential local model-order search, rather than a
claim of a mathematical global optimum over all possible pole locations. An
explicit integer is an exact state count.
The normalised good-conductor problem depends on bandwidth ratio rather than
conductivity or absolute frequency, so each ratio-and-pole-count realization
is cached and scaled to the requested band and metal. That realization is
independent of the requested tolerance; tolerance is used only to accept or
reject its certified error during order selection. Consequently every
common-metal preset selects the same pole count for the same frequency ratio
and tolerance. The reported fit error still applies only inside the requested
band.

For the default 0.2% tolerance, representative selections are:

.. list-table:: Automatic Foster order versus fitted bandwidth
   :class: api-parameters
   :header-rows: 1
   :widths: 28 18 22 24

   * - Fit band
     - Band ratio
     - Selected poles
     - Maximum relative error
   * - 10--10.1 GHz
     - 1.01
     - 1
     - 0.10304%
   * - 8--12 GHz
     - 1.5
     - 2
     - 0.14848%
   * - 8--16 GHz
     - 2
     - 3
     - 0.03695%
   * - 0.1--10 GHz
     - 100
     - 8
     - 0.1341%
   * - 1 MHz--100 GHz
     - 100,000
     - 13
     - 0.1412%

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

``F``, ``G``, ``L``, and ``Z0`` define the exact discrete law used by both the
FDTD kernel and FDFD reduction. The packed FDTD coefficients store its diagonal local form
described below. gprMax requires ``Z0`` to be finite and strictly positive. A
constant resistance is the order-zero case: ``F``, ``G``, and ``L`` are empty
and ``Z0`` is the resistance.

Local Foster recurrence
-----------------------

For the supported metal realization, :math:`A` and therefore :math:`F` are
diagonal. The time-step kernel does not store or multiply the zero off-diagonal
entries. For pole :math:`m`, define

.. math::

   f_m=F_{mm},\qquad q_m=L_mG_m,\qquad y_{p,m}^n=L_mx_{p,m}^n.

The history and state update for surface-current port :math:`p` then reduce to

.. math::

   h_p^n=\sum_m y_{p,m}^n,
   \qquad
   y_{p,m}^{n+1}=f_m y_{p,m}^n+q_m k_p^{n+1/2}.

The read-only :math:`f_m,q_m` coefficients are shared by every port using a
material, while every port owns its own local :math:`y_{p,m}` history. The
history is summed once, the locally implicit electric edge is solved, and the
independent pole states are then advanced in place. No second state buffer or
dense pole-to-pole matrix product is required.

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


.. _impedance-geometry-theory:

Discrete geometry and contacts
==============================

Rasterized topology
-------------------

The final rasterized cell ownership is authoritative. The topology check runs
after all geometry objects and ordered overwrites have produced the cells used
by the solver. Curved and oblique surfaces are therefore represented by the
same Yee-aligned staircase as other gprMax geometry, and two individually
valid primitives can still produce an invalid final union or cutout.

Impedance cells must connect locally through full voxel faces. Around a Yee
edge, one impedance quadrant, two adjacent impedance quadrants, or three
impedance quadrants form valid flat or staircased boundaries. Exactly two
diagonally opposite impedance quadrants touch only through that edge. This
non-manifold pattern is rejected because it does not define an unambiguous
clipped H circulation and local boundary normal.

At every grid vertex, the impedance cells and the retained cells in the
incident ``2 x 2 x 2`` voxel neighbourhood must each be connected through
voxel faces whenever the respective set is non-empty. This rejects both
vertex-only impedance contacts and a pinched retained region. The check is
binary across surface models: cells assigned any surface-impedance ID count
as impedance cells, even when the IDs differ. Consequently, two impedance
regions that meet through a face form one excluded volume at that interface;
the internal metal-to-metal face receives no boundary condition.

This is a local rule, not a requirement that every impedance object in the
model belong to one connected component. Separate bodies remain valid when
they do not touch through only an edge or vertex and every resulting boundary
is closed and manifold. To repair an invalid contact, add or thicken impedance
cells so the regions connect through a full voxel face, or move them apart so
they share neither an edge nor a vertex. Refining the mesh, increasing a thin
feature's thickness or a curved object's radius, or adjusting its position can
also produce an unambiguous rasterized boundary.

Yee degrees of freedom
----------------------

Let a candidate electric edge be oriented along Cartesian axis ``a``. Four
cell-centred quadrants surround that edge in the transverse ``b-c`` plane.
The compiler classifies the edge as follows:

* zero metal quadrants: use the ordinary Yee update;
* four metal quadrants: remove the electric degree of freedom;
* one, two adjacent, or three metal quadrants: retain one boundary electric
  degree of freedom and compile a sparse impedance row;
* exactly two diagonally opposite metal quadrants: reject the non-manifold
  edge contact before any field-update data are compiled.

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
retain one quarter. Heterogeneous retained quadrants are integrated
independently, including Debye, Lorentz, Drude, and inclusive mixtures.
Polarization histories use only the retained areas; repeated quadrants of
the same material share a history with their areas summed.

Each retained quadrant also contributes half-line magnetic-circulation
segments. Duplicate segments are coalesced. Every metal/dielectric face
adjacent to the electric edge creates a surface-current port, so one electric
edge at a staircase corner can own more than one independent ADE state. The
edge is shared, but there is one port per adjacent face and per surface model.

Interior H components are assigned zero update coefficients. At an interface,
the normal H degree of freedom remains associated with the retained material;
tangential boundary action is supplied by the surface current. The local
face-connectivity checks above ensure that this clipped update is compiled
only for an unambiguous manifold boundary.

Contacts with PEC and PMC volumes
---------------------------------

Isotropic PEC, PMC, and surface-impedance volumes may coexist and touch.
Custom materials with infinite electric or magnetic conductivity receive the
same treatment as the built-in ``pec`` and ``pmc`` materials.

If any incident quadrant is PEC, the shared tangential E component is forced
to zero and has no surface-impedance update or surface-current state. This
precedence is independent of the drawing order of adjacent volumes. An
existing PEC component constraint is also preserved. A diagonal PEC/SIBC
contact with two separating retained quadrants produces an aggregated warning
with a count and up to three example Yee-edge coordinates. Face-sharing
PEC/SIBC contacts do not produce this warning.

PMC compatibility uses the existing volume solver's H constraints. A contour
sample whose final H-component material is PMC is omitted from the sparse
circulation because its update coefficients force it to zero. This does not
remove a quadrant's electric storage or conductivity contribution, change the
dual area, or delete an impedance-current port. It produces the same update
as retaining that sample with H equal to zero. This feature does not change
the PMC volume's boundary discretisation. Domain symmetry planes use the
separate treatment described below.

Directional PEC/PMC material mixtures at the impedance boundary are rejected:
their averaged voxel material does not retain the directional information
needed by this compiler. Use isotropic conductor volumes for these contacts.

The runnable example
``examples/features/impedance_surface/pec_pmc_contacts.py`` places a copper
SIBC block between PEC and PMC volumes. It writes a geometry view and electric
field traces at the PEC, PMC, and exposed SIBC contacts::

    python examples/features/impedance_surface/pec_pmc_contacts.py --output-dir results/contacts

Domain symmetry planes
----------------------

PEC and PMC ``SymmetryBoundary`` planes are supported on all six domain faces,
including intersecting planes. An impedance volume may meet a symmetry plane;
its boundary must satisfy the topology rules after reflection across that
plane. The cut through the impedance interior does not create an end cap.

On a PEC plane, tangential E remains zero and has no impedance-current state.
On a PMC plane, the clipped electric update integrates only the physical
quadrants and closes the contour along the plane, where tangential H is zero.
The reduced electric area gives the same update as the full mirrored geometry
with odd tangential H. Surface-current ports also use their physical lengths;
reported surface areas describe the modeled portion of the geometry.

The contact example can cut all three conductor volumes at the ``x0`` plane
and add a receiver on that plane::

    python examples/features/impedance_surface/pec_pmc_contacts.py --symmetry pmc --output-dir results/symmetry

Use ``--symmetry pec`` for the corresponding PEC plane. The ordinary source,
PML, and solver restrictions in :doc:`impedance_surfaces` still apply.

Sparse locally implicit FDTD algorithm
======================================

.. _impedance-stability-theory:

Clipped-circulation stability
-----------------------------

The public timestep cap is applied as

.. math::

   \Delta t_{\mathrm{used}}=\min(f_{\mathrm{user}},0.99)
       \Delta t_{\mathrm{CFL,rounded}}.

Here the default requested factor is one. The operational consequences and
user command are in :ref:`impedance-automatic-timestep`.

For a homogeneous nondispersive exterior, retained voxel quadrants give
matched electric and magnetic energy weights: each retained voxel contributes
one quarter of its volume to an incident E edge and one half to an incident H
face. With these weights, the compiled clipped Ampere circulation is the
weighted transpose of the retained Faraday curl. The resulting geometric
stability bound is the ordinary Cartesian CFL bound; quarter-cell electric
areas alone do not imply an additional timestep reduction.

A strict margin below that bound matters. A retained one-voxel cavity can
attain the bound exactly. Before the automatic SIBC margin was introduced,
a source-free CPU audit of a passive 1 MOhm cavity at the rounded Cartesian
CFL timestep showed approximately 198-fold field-norm
amplification in double precision and severe late-time growth in single
precision over 200,000 steps. A timestep factor of 0.99 kept both runs bounded.
The tested copper-preset cavity also remained bounded at that timestep.
The :ref:`impedance-automatic-timestep` now applies the 0.99 margin without
requiring an explicit user command.

A historical audit using the ordinary rounded timestep and complete CPU solver
confirmed that, on the 1 mm test mesh, ``dt`` is rounded down by two binary64
ULPs and is below the mathematical CFL limit. Nevertheless, the stored
float32 coefficients of the 1 MOhm cavity give an effective Courant factor
of approximately 1.000000018 and predict the observed growth. A factor of
0.9999999 removed exponential growth in that case but still allowed large
transient amplification. See
``testing/validation/impedance_surface/default_cfl_report.md`` for the
coefficient derivation and checks of the actual timestep rounding.

This is a demonstrated margin for the tested cases, not an unconditional
stability guarantee for dispersive bulk media, PML, or every contact/source
configuration. Surface passivity and amplification eigenvalues should be
checked together with the geometric energy balance and actual long-time
field growth.

Reproduce the audit with
``python -m testing.validation.impedance_surface.stability --steps 20000``.
Use ``--historical`` to disable the automatic cap within this validation
driver and reproduce the pre-protection results; this diagnostic switch is
not a simulation input option. Protected results use separate filenames so
the historical records remain available.
The derivation, scope, recorded results, and precision-sensitive reproducer
are in ``testing/validation/impedance_surface/stability_report.md``.

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

The kernel then recovers every :math:`k_p^{n+1/2}` and advances each local
Foster pole

.. math::

   y_{p,m}^{n+1}=f_m y_{p,m}^n+q_mk_p^{n+1/2}.

This analytic scalar elimination naturally handles several faces sharing one
electric edge; it does not perform a dense solve at each time step. A flat
boundary edge normally has one port and one current. A convex manifold edge
can have two face ports: they share the scalar electric solve but retain
separate histories and separate :math:`k_p` values.

Dispersive retained quadrants
-----------------------------

At a dispersive contact the same scalar elimination includes both bulk
polarization and surface-current histories. Let :math:`A_q` be the retained
area assigned to material :math:`q`, and let :math:`EA_q,EB_q` be its ordinary
bulk dispersive electric coefficients before division. The electric row uses

.. math::

   a_+^{\mathrm{disp}}=\sum_q A_q EA_q,
   \qquad a_-^{\mathrm{disp}}=\sum_q A_q EB_q,
   \qquad r_H\longmapsto r_H-\Phi^n.

Each retained material pole owns the area-scaled complex history
:math:`S=\epsilon_0 A_q T`. Using the bulk recurrence coefficients
``eqt``, ``zt``, and ``eqt2`` gives

.. math::

   f=\mathrm{eqt},\qquad b=\epsilon_0 A_q\mathrm{zt},
   \qquad c=\mathrm{eqt2},\qquad
   \Phi^n=\sum_{q,m}\operatorname{Re}(c_{q,m}S_{q,m}^n),

.. math::

   S_{q,m}^{n+1}=f_{q,m}S_{q,m}^n+b_{q,m}(e^n-e^{n+1}).

Both the instantaneous polarization contribution and Drude's equivalent
conductivity are included in :math:`EA_q,EB_q`. Updating only the history
term would omit their implicit contribution to the electric solve.
The sparse kernel advances :math:`S` with the final boundary E, after the
bulk A/B stages. The private held material prevents those stages from
advancing the same histories. Resetting the grid clears both polarization
and surface-current states.

Real and imaginary parts occupy separate columns in the field precision.
Per pole there are six real coefficients and two real state values. Each
boundary edge also stores a pole offset and two corrections to the physical
non-dispersive :math:`a_+,a_-` values. These arrays are empty when no
dispersive material touches the boundary. The magnetic circulation and
surface-port geometry are unchanged.

Packed data and update order
----------------------------

The geometry compiler stores:

* boundary edge component/index, H range, and port range;
* :math:`a_+`, :math:`a_-`, and retained dual-area fraction;
* H component/index records and signed half-line weights;
* the precomputed old-E coefficient and inverse scalar denominator for each
  boundary edge;
* port model, unique state offset, :math:`g_p`, :math:`g_p/Z_{0,p}`,
  :math:`1/Z_{0,p}`, face normal, and face area;
* one copy of the :math:`f_m,q_m` vectors and :math:`Z_0` for each used
  material model;
* one in-place :math:`y_{p,m}` value per port and selected pole.

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
or cross-edge state races. The one or two histories on an edge are thread-local
scalars. Runtime work and state storage scale linearly with boundary area and
the selected Foster order, rather than conductor volume:

.. math::

   \text{work}\sim O(N_{\Gamma,E}+N_{\Gamma,K}\overline N_p),
   \qquad
   \text{state}\sim O(N_{\Gamma,K}\overline N_p),

where :math:`N_{\Gamma,K}` is the number of local surface-current ports and
:math:`\overline N_p` is their average selected pole count. Shared model
coefficient storage is also linear in pole count. The optimized representation
therefore avoids both the former quadratic dense-matrix arithmetic and its
second per-port state buffer.

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

For the diagonal Foster representation this is equivalently the local pole sum

.. math::

   Z_{\mathrm{alg}}(f,\Delta t)
      =Z_0+\sum_m\frac{q_m}{z-f_m}.

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

Clipped row in the P/Q solver
-----------------------------

The implemented FDFD path eliminates the scalar surface currents into the
electric coefficient rather than appending them as eigenproblem unknowns.
For boundary edge ``e`` with retained dual area :math:`A_e`, attached port
lengths :math:`\ell_p`, and the same integrated masses used by FDTD, the
dispersion-compensated solver uses

.. math::

   \epsilon_{r,e}^{\mathrm{eff}}
      =\frac{
         j\Omega m_{\epsilon,e}
         +c_\theta m_{\sigma,e}
         +\displaystyle\sum_p\ell_pY_{\mathrm{alg},p}
       }{
         j\Omega\epsilon_0A_e
       }.

This makes the material term with the leapfrog temporal symbol reproduce the
exact discrete-time surface load:

.. math::

   j\Omega\epsilon_0A_e\epsilon_{r,e}^{\mathrm{eff}}
      =j\Omega m_{\epsilon,e}
       +c_\theta m_{\sigma,e}
       +\sum_p\ell_p\frac{c_\theta}{Z_{\mathrm{alg},p}}.

The standard rectangular finite-difference curl row is replaced by the
compiled clipped line circulation, normalized by :math:`A_e k_{0,\mathrm{operator}}`, where
:math:`k_{0,\mathrm{operator}}=\Omega/c` is ``solver.operator_k0``.
Independent retained masks remove metal-interior E and H
degrees of freedom without
misusing PEC masks; in particular, interface-normal H can remain present when
a collocated tangential E is a valid impedance-boundary unknown. The existing
P/Q reduction solves for :math:`\lambda=-n_{\mathrm{operator}}^2`, with
:math:`n_{\mathrm{operator}}=K_w/k_{0,\mathrm{operator}}`. The normal spacing :math:`\Delta w`
then determines the phase propagation constant and public effective index:

.. math::

   \beta=\frac{2}{\Delta w}\sin^{-1}\left(\frac{K_w\Delta w}{2}\right),
   \qquad n_{\mathrm{eff}}=\frac{\beta}{k_0}.

The passive forward branch gives attenuation and evanescent decay in positive
``w``. Modal field reconstruction uses :math:`n_{\mathrm{operator}}`; source
and monitor spatial phases use :math:`\beta`.

Low-level calls that omit ``fdtd_dt`` retain the physical-frequency P/Q
normalization: the coefficient denominator uses :math:`j\omega\epsilon_0A_e`
and :math:`k_{0,\mathrm{operator}}=k_0=\omega/c`, while the boundary numerator still comes from its
exact discrete recurrence. See :doc:`eigenmode_port_theory` for both optional grid
parameters.

.. important::

   The surface ADE, midpoint factor, boundary electric mass, conductivity,
   clipped transverse curl, and retained boundary polarization histories are
   reduced exactly for the FDTD time step.
   Eigenmode sources also use the owning grid's leapfrog temporal symbol and
   longitudinal spatial difference. Bulk nondispersive conductivity includes
   its midpoint factor. Away from the surface, bulk dispersive poles still use their
   analytic physical-frequency response rather than the exact volume ADE
   transfer, so the general dispersive bulk eigenproblem is not fully time
   discrete. Keep modal anchors below Nyquist and check mesh/time-step
   convergence.

The longitudinal :math:`K_w` coupling remains the standard implicit P/Q
term. The source-plane mapper checks that the omitted longitudinal H weights
form equal and opposite contributions in the two cells adjacent to the plane.
That is why a changing wall cross-section or an impedance end cap at the
modal plane is rejected.

Dispersive retained media and volume ADEs
-----------------------------------------

Dispersive retained quadrants are supported by the coupled scalar row above.
The modal boundary mapper also includes their exact discrete response. For
:math:`z=\exp(j\theta)`, a complex pole contributes the real-output transfer

.. math::

   H_m(z)=\tfrac12\left[\frac{c_m b_m}{z-f_m}
                  +\frac{c_m^* b_m^*}{z-f_m^*}\right].

If :math:`\delta_\pm=a_\pm^{\mathrm{disp}}-a_\pm`, the additional
midpoint-time Ampere load is

.. math::

   \Delta Y=\delta_+e^{j\theta/2}-\delta_-e^{-j\theta/2}
           -2j\sin(\theta/2)\sum_m H_m(z).

Both conjugate branches are required for Lorentz materials; taking the real
part of a transfer evaluated at complex :math:`z` gives an incorrect phase.
The modal mapper adds :math:`\Delta Y/(j\Omega\epsilon_0 A_{\rm ret})`
to the boundary relative permittivity. Interior bulk dispersive rows still
use their analytic physical-frequency response, so making the entire bulk
eigenproblem exact in time remains separate work.

Eigenmode solution and FDTD injection
=====================================

A direct modal solve reuses the component IDs, retained masks, clipped H
weights, dual fractions, port models, FDTD ``dt``, and normal cell spacing
from the already compiled grid. There is no separately
redrawn FDFD wall.
This shared geometry is as important as sharing the exact discrete ADE law
(``f``, ``q``, and ``Z0`` in its local Foster form): a half-cell area or sign
mismatch would change both loss and mode phase.

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
``testing/validation/impedance_surface/validate_copper_wall_waveguide.py``.

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
See :doc:`eigenmode_port_theory` for broadband anchor tracking, one-watt
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
temporal dispersion of its cubic Yee grid from its phase comparison.
For cubic spacing :math:`\Delta`, its analytical lossless reference is

.. math::

   \beta_Y=\frac{2}{\Delta}\sin^{-1}\!\left\{
      \Delta\left[
        \left(\frac{\sin(\pi f\Delta t)}{c\Delta t}\right)^2
        -\left(\frac{\sin(\pi\Delta/(2a))}{\Delta}\right)^2
      \right]^{1/2}
   \right\}.

That reference replaces :math:`\beta_0` by :math:`\beta_Y` in the phase term
while retaining the continuum perturbation factor :math:`Q`. The pure
continuum result is written separately so mesh dispersion remains visible
rather than being mistaken for a copper-boundary error. The FDTD attenuation
comparison uses :math:`-\ln|S_{21}|/L`, so its loss acceptance criterion is
independent of this phase correction.


.. _impedance-pmc-theory:

Exact voxel-face PMC
====================

Use ``SurfaceImpedance(id='wall', resistance=float('inf'))`` or
``#surface_impedance: wall resistance inf`` to impose the exact
:math:`Z_s\to+\infty` limit. Its surface admittance and surface current are
zero. The retained dual area and clipped H circulation remain active:

.. math::

   E_e^{n+1}=E_e^n+\frac{\Delta t}{\epsilon A_e}
     \left(r_{H,e}^{n+1/2}+A_e Q_e^{n+1/2}\right)

for a lossless nondispersive retained host. This locates the PMC at the
main-voxel face; it does not zero the nearest retained half-cell H samples.
The built-in volume material ``pmc`` retains its existing discretisation.
Using it, or a material with infinite magnetic conductivity, emits a warning:
its H constraints can shift a flat wall's effective reflection plane by half
a cell. Use infinite SIBC resistance for PMC at the voxel face. The warning
is emitted once per affected grid; unused declarations, internal TE storage
constraints, and ``SymmetryBoundary(type='pmc', ...)`` do not trigger it.
The derivation, reflection-plane tests, cavity modes, and long-run results
are in ``testing/validation/sibc_based_pmc/README.md``.


.. _impedance-pml-theory:

PML and virtual-guide coupling
==============================

The supported extrusion, host, and aperture requirements are in
:ref:`sibc-pml`. The auxiliary guide retains independent surface histories.

In 2D, only the live invariant layer is physical: index zero for TM and
index one for TE. The compiler projects only active field components onto
that layer and excludes artificial invariant-axis end faces. The 1D modal
solve preserves Faraday derivatives while replacing the relevant Ampere
and constitutive rows with the clipped derivative and discrete ADE response.
Modal power is per metre and independent of synthetic invariant-axis spacing.
The auxiliary guide extrudes full-mass boundary rows, retains the active
Yee staggering, and allocates independent surface histories.

Modal excitation supplies its missing longitudinal circulation to the
implicit surface solve and its ADE history. This applies to direct and
virtual-guide sources, including exact PMC, whose tangential electric field
need not vanish. See :ref:`eigenmode-virtual-coupling` for the aperture's
shared E/H samples and propagation-direction signs.

The coupling adds the PML correction to the magnetic circulation. For
retained dual area :math:`A_e`, let :math:`Q_e` be the signed correction
to the ordinary curl from the native PML histories. The surface equation is

.. math::

   d_e E_e^{n+1}
   = a_e E_e^n + r_{H,e}^{n+1/2}
     - \Phi_e^n - \sum_p (g_p/Z_{0p})h_p^n + A_e Q_e^{n+1/2},

where :math:`d_e` is the local implicit denominator, :math:`a_e` its
old-field numerator, :math:`\Phi_e^n` the bulk-polarization history load,
and :math:`h_p` the sum of the Foster histories at port :math:`p`.
The denominator and numerator include the retained-area sums of electric
storage, conductivity, and instantaneous polarization terms. Uniform
extrusion makes the longitudinal clipped-H fraction
cancel against the retained area, so the native stretched derivative is
applicable. For the continuous stretched-coordinate interpretation and its
limitations, see Steven G. Johnson's `Notes on Perfectly Matched Layers
<https://arxiv.org/abs/2108.05348>`_. The discrete coupling above follows
from the implemented clipped circulation. It preserves the physical
:math:`E^n` while collecting the PML forcing. After the ordinary local solve,
it applies

.. math::

   \Delta E_e = A_e Q_e/d_e,\qquad
    \Delta y_{pm}=q_{pm}\Delta E_e/(2Z_{0p}),\qquad
    \Delta S_{em}=-b_{em}\Delta E_e.

Here :math:`S_{em}` is a bulk-polarization state satisfying
:math:`S_{em}^{n+1}=f_{em}S_{em}^n+b_{em}(E_e^n-E_e^{n+1})`.
The geometric curl scaling does not require a homogeneous or lossless host.
Different retained materials contribute their own masses and polarization
histories; no effective bulk pole model is assigned to the PML capture row.
That row has unit curl-correction coefficient and zero ordinary curl and
bulk-polarization update, preventing duplicate constitutive updates.

This is algebraically the same coupled solve, including the midpoint-time
surface-current and bulk-polarization histories. Adding the PML increment to :math:`E^n` before
solving would give the wrong damping and history for finite impedance.
The automatic time-step factor of at most 0.99 also applies to these models.

Reproducible comparisons with longer physical guides, pulse absorption,
and virtual/continuous guide comparisons are in
``testing/validation/impedance_surface/validate_sibc_pml.py`` and
``testing/validation/impedance_surface/virtual_waveguide.py``. Exact PMC
image comparisons are in
``testing/validation/sibc_based_pmc/pml_mirror.py``. These test the supported
extruded configurations; they do not establish stability for arbitrary
PML profiles or unsupported bulk media.
Lossy and Debye/Lorentz/Drude bulk contacts, including layered interfaces,
are checked against independent mirrored bulk grids in
``tests/impedance_surfaces/test_pmc_pml.py``. Finite surface and bulk histories
are compared with a simultaneous dense solve in
``tests/impedance_surfaces/test_lossy_dispersive_pml.py``. The copper microstrip
comparison is ``testing/validation/impedance_surface/validate_microstrip_pml.py``.
CPU virtual guides extrude the complete retained-area conductivity and bulk
pole coefficients, with independent polarization and Foster histories on
every auxiliary edge. Their ordinary dispersive aperture samples use the
same bulk ADE recurrence as the continuous grid. Comparisons in
``tests/test_virtual_waveguide_impedance.py`` cover lossy, Debye, Lorentz,
and Drude contacts, mixed hosts, both precisions, rotated 3D and TE/TM
guides, and active excitation. The comparison fixes the PML profile in both
models: cropping opaque padding otherwise changes the cross-section average
used to choose the default maximum PML conductivity.
The profile audit retained in
``testing/validation/impedance_surface/results/sibc_pml/profile_audit.json``
found late growth with two duplicated unshifted HORIPML terms in both an
ordinary PEC guide and an SIBC guide. Reducing the time step to 0.99 alone
does not guarantee stability for arbitrary custom PML parameters.
The follow-up investigation identified the negative real part of the
unshifted product stretch as the cause. A second frequency-shift profile
tracking ``1.1 * sigma1`` removed this failure in the PEC and
finite/dispersive SIBC tests; see :ref:`pml-higher-order-stability` and
``testing/validation/impedance_surface/results/pml_profile_investigation/README.md``.

.. _impedance-pml-profile-theory:

Second-order HORIPML profile guard
==================================

HORIPML multiplies its two stretching factors. Duplicating the unshifted
first-order defaults is therefore not a safe way to create a second-order
absorber. For :math:`\kappa_1=\kappa_2=1`,
:math:`\alpha_1=\alpha_2=0`, and identical :math:`\sigma`, the product has

.. math::

    \operatorname{Re}S(\omega)=1-\frac{\sigma^2}{(\omega\epsilon_0)^2}.

This is negative at low frequencies and can amplify evanescent fields.
The resulting growing mode persists as the time step is reduced; a CFL
safety factor does not repair this profile.

During PML coefficient construction, gprMax rejects second-order HORIPML
profiles with negative real total stretch at any resolved E or H sample,
beyond floating-point tolerance. It checks the analytical quadratic in
``(omega * epsilon0)**2`` over all positive frequencies, including terminal
samples and the full profile before MPI partitioning. The error identifies
the slab/profile and sample, reports its parameters, and suggests a repair.
Boundary, internal, and virtual-guide absorbers share this check. Explicit
``sigmamax=0`` remains zero; only ``None`` requests automatic conductivity.
Passing this check excludes the demonstrated failure mechanism, rather than
certifying stability for every geometry and material.

For a classical/CFS pairing with :math:`\alpha_1=0`, a pointwise condition
which removes this mechanism is :math:`\alpha_2\geq\sigma_1/\kappa_1`
with :math:`\kappa_1\kappa_2\geq1`. Alpha and sigma use the same units in
these input commands. Match the grading at every electric and magnetic
sample, including polynomial order and direction. For the duplicated
unit-kappa quartic profile, setting the second alpha profile to
``1.1 * sigma1`` with matching quartic grading kept the tested PEC and
finite/dispersive SIBC guides bounded for 20,000 steps.

This condition addresses the negative-real-stretch mechanism, not every
possible material, mesh, or PML stability issue. The reproducible diagnosis,
input recipe, time-step controls, and reflection results are in
``testing/validation/impedance_surface/results/pml_profile_investigation/README.md``.
MRIPML adds its pole terms and has different parameter normalization; do
not apply the HORIPML product argument to that formulation.

Validation and benchmarking
===========================

The 2D matrix and long-run drivers complement the comparisons below:

.. code-block:: console

   python -m testing.validation.impedance_surface.validate_2d --section matrix
   python -m testing.validation.impedance_surface.validate_2d --section late

They cover all TE/TM axes and propagation directions, both precisions,
constant and fitted impedance, and representative 20,000-step guides with
PML. The recorded PMC evidence remains under
``testing/validation/sibc_based_pmc``; general surface evidence remains under
``testing/validation/impedance_surface``. Their configuration-specific
bounds do not certify arbitrary PML profiles or unsupported host media.

FDFD attenuation and modal launch
---------------------------------

The focused FDFD operator test constructs a copper-lined rectangular guide,
compares :math:`-k_0\operatorname{Im}n_{\mathrm{eff}}` with
the TE10 perturbation result using :math:`Z_{\mathrm{alg}}/c_\theta`, and
applies a 2% relative tolerance.

``testing.validation.impedance_surface.validate_copper_wall_waveguide`` is the
physical common-metal case. It uses a 1.6 mm by 0.8 mm copper-lined guide on a
0.1 mm cubic grid. TE10 is evaluated from 130 to 150 GHz, below the 187.37 GHz
next-mode cutoff, over a 40 mm reference-plane spacing. The independent
good-conductor formula predicts 0.204--0.234 dB insertion loss, making copper
loss materially larger than in the initial microwave-scale test.

The comparison uses 21 uniform validation frequencies from 130 to 150 GHz,
and each is an exact source-port FDFD anchor. The copper excitation spans
120--150 GHz on a 31-point, 1 GHz DFT grid. The source port uses all 31 bins
plus guards at 100, 110, 160, and 170 GHz, for 35 anchors. Each passive port uses
only 11 anchors: the four guards and seven uniformly spaced anchors from 120
to 150 GHz. Dense source anchors preserve exact in-band ``neff`` validation
and modal injection; sparse guarded passive anchors avoid repeating
unnecessary FDFD solves while retaining smooth modal interpolation. Starting
the excitation sufficiently above the 93.69 GHz TE10 cutoff prevents a slow
near-cutoff tail from entering the finite record.

The copper surface explicitly uses ``fit_order='auto'`` over 80--180 GHz with
a 0.2% tolerance and selects three poles. The model uses a 210 mm domain and a
500 ps record. The active source is at 90 mm and the passive planes are at 105
and 145 mm. The finite walls extend almost to the domain ends, leaving 97.415
ps between the record endpoint and the conservative earliest wall-end return.
This gives the source response time to settle while retaining a causally
isolated one-way propagation measurement.

The copper release checks cover three milestones. The attenuation
:math:`-k_0\operatorname{Im}n_{\mathrm{eff}}` stored for
each exact in-band FDFD anchor is compared with :math:`Q\operatorname{Re}Z_s`
using a 1% relative L2 error threshold. The driven FDTD port must have maximum
:math:`S_{11}<-20` dB after the
complex modal field is injected. Finally, attenuation obtained from the FDTD
two-plane propagation factor is compared with the same perturbation theory
using a 2% relative L2 error threshold. The workflow therefore exercises the copper
preset, Foster fit, exact FDFD boundary reduction, complex modal source, FDTD
ADE, modal projection, and propagation loss in one accepted result.

In the retained double-precision four-thread result with bulk
dispersion compensation, the impedance fit error is 0.026023%, the FDFD and
FDTD attenuation errors are 0.681438% and 0.759867%,
and maximum reflection is -101.0893 dB. The four-thread rerun on 2026-09-04
completed in 147.019 s including analysis and plot generation.

Run the validation from the repository root:

.. code-block:: console

    python -m testing.validation.impedance_surface.validate_copper_wall_waveguide --threads 4

Use ``--reuse`` to reanalyse compatible cached solver output. The validation
exits non-zero when an acceptance criterion fails and writes numerical data
plus a machine-readable summary below its selected output directory.

Sparse-kernel performance
-------------------------

``testing.benchmarking.benchmark_impedance_box`` alternates otherwise
identical baseline, resistive, automatic-order copper, and explicit-order
copper runs. It times the full solve and also isolates the sparse kernel and a
bulk-plus-surface hot loop:

.. code-block:: console

    python -m testing.benchmarking.benchmark_impedance_box \
        --cells 80 --iterations 250 --threads 4 --repeats 3 \
        --fit-band 8e9 12e9 --fit-tolerance 2e-3 \
        --explicit-orders 4 8 16 32 \
        --kernel-iterations 1000 --kernel-repeats 3 \
        --hot-iterations 250 --hot-repeats 3 \
        --output impedance_box_benchmark.json

The JSON records requested and selected order, boundary edges, surface ports,
per-port state values and bytes, packed coefficient bytes, edge/port/pole
update rates, median solve overhead, and bulk-plus-surface hot-loop overhead.
The resistive case separates fixed sparse-boundary cost from pole cost, while
the explicit-order sweep reveals order scaling. Results depend on compiler,
CPU, thread count, boundary area, and surface-to-volume ratio; timing values
are not portable CI thresholds. Wall-clock construction time should not be
mixed with time-stepping overhead.

Dispersive-contact benchmark
----------------------------

The box benchmark accepts ``--exterior debye``, ``lorentz``, ``drude``, or
``mixed``. The same exterior is used for its ordinary-grid baseline. The
dispersive hot loop includes both electric A/B stages, and the JSON records
boundary polarization pole count, state bytes, coefficient/index bytes, and
polarization update rate in addition to the surface-current measurements.

The combined sweep also runs pulse-driven Python/Cython comparisons,
late-time decay checks, and convergence toward an ordinary PEC box as the
surface resistance decreases:

.. code-block:: console

    python -m testing.benchmarking.benchmark_dispersive_impedance \
        --cells 48 --iterations 250 --threads 4 --repeats 3 \
        --explicit-orders 4 8 --kernel-iterations 1000 --kernel-repeats 3 \
        --hot-iterations 250 --hot-repeats 3 \
        --output testing/benchmarking/results/dispersive_impedance_2026-09-07.json

The recorded 2026-09-07 sweep passes all four driven exterior cases. Maximum
Python/Cython receiver relative L2 error is :math:`1.17\times10^{-15}`.
Reducing resistance from 0.01 to 0.001 Ohm reduces error against the PEC
receiver trace approximately tenfold in every case. The final 200-sample
peak is at most :math:`1.42\times10^{-4}` of the run peak. These are discrete
solver and limiting-case checks, not a general continuum-accuracy or
unconditional-stability guarantee.

For the 6,912-edge timing box, a uniform two-pole exterior adds 221,184 state
bytes and 801,796 coefficient/index bytes. The mixed exterior needs slightly
more state at material seams. Full raw repeats, hardware information,
acceptance criteria, and a readable summary are saved in
``testing/benchmarking/results/dispersive_impedance_2026-09-07.json`` and the
adjacent ``.md`` report. Timing noise on a shared desktop prevents using
these samples as a material speed ranking or a portable performance limit.

Analytical sphere and plane-wave comparisons
--------------------------------------------

Two additional CPU validations compare the driven surface boundary with
analytical electromagnetic solutions:

.. code-block:: console

    python -m testing.validation.impedance_surface.validate_conductor_sphere --threads 4
    python -m testing.validation.impedance_surface.validate_reflection_phase --threads 4

The sphere case compares 2--7 GHz backscatter and complex angular scattering
for a 16 mm radius sphere with both impedance-boundary Mie theory and the
full homogeneous conducting-sphere Mie series. Conductivities are
:math:`10^3` and :math:`5.8\times10^7` S/m. Refining the staircased boundary
from 1.5 to 0.75 mm reduces backscatter RMS errors from 1.30/1.26 dB to
0.855/0.832 dB. Both fine meshes pass the 1 dB RMS, 2 dB maximum and 15%
complex angular-pattern error thresholds. The coarse results are convergence
diagnostics. The bulk reference evaluates scaled Bessel ratios, so the
copper interior does not overflow or require numerical skin-depth cells.

The planar case compares the complex electric reflection coefficient

.. math::

   \Gamma = \frac{Z_s-\eta}{Z_s+\eta}, \qquad
   \eta=\sqrt{\frac{\mu_0}{\epsilon_0\epsilon_r(\omega)}}

over 1--8 GHz, including a Debye material directly touching the wall.
Its relative permittivity is :math:`2.5+2/(1+j\omega\,80\mathrm{ps})`.
Matching incident-reference runs and an independent discrete propagation
symbol remove the 30 mm receiver-to-wall propagation phase. Transverse
PEC/PMC symmetry faces generate a uniform TEM wave; remote end boundaries
cannot return within the 4 ns analysis record. The 0.5 mm mesh has a maximum
continuum phase RMS error of 0.001497 degree across the four host/conductivity
pairs. Both 1 and 0.5 mm meshes pass all plane-wave acceptance checks. The additional
discrete prediction includes time staggering, the retained half-cell mass,
Debye recurrence and bilinear impedance realization; its maximum phase RMS
error is 0.00005387 degree. A negative-control regression removes the
boundary polarisation history and fails the 0.0002 degree discrete error threshold.

Both drivers return a nonzero status on failed acceptance checks and support
``--reuse`` for compatible local caches. Retained CSV, figures and summaries
are under ``testing/validation/impedance_surface/results/conductor_sphere``
and ``results/reflection_phase``. Methods, reference formulas, qualifications
and recorded comparisons are in
``testing/validation/impedance_surface/analytical_validation_report.md``.


Reproducibility metadata
========================

See :ref:`impedance-output` for the stored surface realization, discrete
coefficients, fit provenance, and modal anchor metadata.

Implementation map
==================

The main implementation files are:

.. list-table:: Surface-impedance implementation files
   :class: api-parameters
   :header-rows: 1
   :widths: 42 58

   * - Module
     - Responsibility
   * - ``gprMax/impedance_surfaces.py``
     - Model validation/discretization, voxel-boundary compilation, packed
       records, and the Python reference update.
   * - ``gprMax/surface_impedance_presets.py``
     - Metal reference data, good-conductor target, and passive Foster
       bounded least-squares (BVLS) fit.
   * - ``gprMax/cython/impedance_surface.pyx``
     - OpenMP sparse locally implicit FDTD update.
   * - ``gprMax/user_objects/cmds_geometry/cmds_geometry.py``
     - Geometry-only material resolution, marker creation, and rejection of
       sheet, line, and directional surface-impedance assignments.
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

Internal realization diagnostics
================================

These checks concern generated coefficients and low-level implementation
work; the public API does not accept arbitrary state-space coefficients.
If a preset or conductivity fit fails one of these checks, report the input
and fit band rather than editing the internal coefficients to bypass it.

``A must be strictly Hurwitz``
    Move every continuous pole into the open left half-plane. A pole on the
    imaginary axis is not accepted, even if a short run appears bounded.

``non-positive discrete feedthrough Z0``
    The locally implicit elimination divides by ``Z0``. Use a proper passive
    realization with sufficient positive direct term, or re-express the fit
    in a positive-real Foster form. Do not add an arbitrary epsilon merely to
    bypass the check.

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

References
==========

The Yee grid convention follows [YEE1966]_. The collocated tangential E and H
surface-impedance method used here is described by [KOB2010]_. Background on
surface-impedance FDTD boundaries is given by [MAL1992]_ and [BEG1992]_. The
preset resistivity provenance is [MAT1979]_, [DES1984A]_, and [DES1984S]_.
