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

The :class:`gprMax.SurfaceImpedance` object defines a reusable boundary
material. Assign its ID directly to the ``material_id`` of an ordinary
cell-occupying geometry object. No separate impedance-geometry object or
conversion step is required.

Despite the feature name, the implemented geometry is a **one-sided boundary
of a closed opaque volume**. It is not a zero-thickness, transmissive sheet.
Fields on the conductor side do not exist. A sheet separating two retained
field regions needs a two-sided sheet transition condition and is discussed
under :ref:`impedance-future-work`.

The first implementation is intended for microwave and radio-frequency
models in which a local scalar surface impedance is appropriate. In
particular, the common-metal presets describe thick, smooth, non-magnetic
bulk metal at 293 K. They are not thin-film, rough-surface, alloy,
temperature-dependent, ferromagnetic, or optical material models. Fitted
bands are capped at 300 GHz and must satisfy
:math:`\sigma/(\omega\epsilon_0)\geq100` at their upper edge.

Quick start
===========

Python API
----------

A constant surface resistance remains available for idealized algorithmic
studies on a closed box:

.. warning::

   A frequency-independent, purely real surface impedance does not represent
   the causal, frequency-dispersive response of a physical conductor. gprMax
   emits a warning when this form is built. Use a fitted metal preset or bulk
   conductivity over the intended frequency band for physically representative
   conductor loss.

.. code-block:: python

    import gprMax

    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(0.30, 0.20, 0.15)))
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.TimeWindow(time=8e-9))

    scene.add(gprMax.SurfaceImpedance(id='wall', resistance=50.0))
    scene.add(gprMax.Box(
        p1=(0.10, 0.07, 0.05),
        p2=(0.20, 0.13, 0.10),
        material_id='wall',
        averaging='n',
    ))

A common-metal preset creates a passive dispersive realization over a stated
band:

.. code-block:: python

    scene.add(gprMax.SurfaceImpedance(
        id='copper_wall',
        preset='copper',
        fit_frequency_range=(8e9, 12e9),
        fit_order='auto',
        fit_tolerance=2e-3,
        plot_fit=False,
    ))

The accepted preset names are ``aluminium``, ``copper``, ``gold``,
``molybdenum``, ``palladium``, ``silver``, ``tungsten``, and ``zinc``.
Case-insensitive element symbols are accepted, as is the spelling
``aluminum``. A finite ``fit_frequency_range`` is mandatory. By default,
``fit_order='auto'`` tests actual runtime pole counts from 1 through 64 and
selects the first count that reaches ``fit_tolerance``. An integer requests
exactly that many runtime poles. With an explicit order, ``fit_tolerance`` is
a diagnostic threshold: a miss produces a build warning but does not override
the requested order. The selected count is available as ``fit_pole_count`` on
the surface object and as ``fit_result.selected_pole_count`` on its fit result.
For example, the default
0.2% tolerance needs two poles over 8--12 GHz, rather than a fixed high order
for every bandwidth.

``plot_fit=False`` writes the intended-versus-fitted impedance plot during a
geometry-only run. Set it to ``True`` to also write the plot during a full
field solve. The setting never opens an interactive window; it writes a
headless PNG beside the selected output stem.

Any supported cell-occupying geometry can use the surface-impedance ID in the
same position as an ordinary material ID:

.. code-block:: python

    scene.add(gprMax.SurfaceImpedance(
        id='silver_body',
        preset='silver',
        fit_frequency_range=(8e9, 12e9),
    ))
    scene.add(gprMax.Sphere(
        p1=(0.15, 0.10, 0.075),
        r=0.025,
        material_id='silver_body',
        averaging='n',
    ))

Geometry retains the usual ordered overwrite semantics. Several primitives
with the same surface-impedance ID can form a union. A later ordinary object
can overwrite part of that region to form a cavity, and later geometry can
replace either kind of material under the normal last-object-wins rule. Tags
remain optional semantic metadata and are not needed to assign an impedance.

Alternatively, specify a positive bulk conductivity directly. It uses the
same passive good-conductor target and fitting machinery as a named preset:

.. code-block:: python

    scene.add(gprMax.SurfaceImpedance(
        id='alloy_wall',
        conductivity=3.2e7,
        fit_frequency_range=(8e9, 12e9),
        fit_order=12,
    ))

The public object deliberately does not accept state-space coefficients.
gprMax retains ``A, B, C, D`` internally for its ADE and writes them to HDF5
for exact reproducibility.

Hash-command input
------------------

The corresponding material forms are:

.. code-block:: text

    #surface_impedance: resistive_wall resistance 50
    #surface_impedance: copper_xband preset Cu 8e9 12e9 auto 2e-3 n
    #surface_impedance: alloy_wall conductivity 3.2e7 8e9 12e9 12

The ``resistance`` form is the idealized boundary described in the warning
above and generates the same build warning for hash-command input.

The final ``n`` above plots during geometry-only runs; ``y`` also plots during
full runs. Apply the surface-impedance ID directly as the geometry material:

.. code-block:: text

    #box: 0.10 0.07 0.05 0.20 0.13 0.10 resistive_wall n
    #sphere: 0.15 0.10 0.075 0.025 copper_xband n

See :ref:`input-hash-cmds` and :ref:`input-api` for the complete command and
constructor signatures.

Geometry semantics
==================

Supported geometry
------------------

Surface-impedance IDs can be assigned directly to boxes, spheres,
ellipsoids, axis-aligned and oblique cylinders and cones when their rasterized
boundaries are manifold, finite-thickness cylindrical sectors, and
finite-thickness triangular prisms. A ``FractalBox`` can also use a
surface-impedance ID as its ``mixing_model_id``, but it must set
``n_materials=1`` because one opaque volume cannot contain a graded set of
surface models, and it must be used with a roughness, grass, or water
modifier. An unmodified one-material ``FractalBox`` retains its existing
instruction to use ``Box`` instead.

The assignment must be scalar. In the Python API, use ``material_id`` rather
than directional ``material_ids``. Hash-command geometry must likewise use
the surface-impedance ID as its sole material identifier. Surface-impedance
geometry is not dielectric-smoothed.

Plates, electric or magnetic edges, zero-thickness triangles and cylindrical
sectors, and boxes that collapse to zero cells on any axis are rejected.
Those sheet and line objects have retained fields on both sides and require a
two-sided sheet transition condition rather than the implemented one-sided
opaque-volume boundary.

Impedance geometry cannot yet be round-tripped through
``GeometryObjectsWrite`` and ``GeometryObjectsRead``. Recreate the
``SurfaceImpedance`` definition and native geometry in the destination scene.

The volume must occupy at least one cell along every non-empty axis and there
must be at least one retained cell between the complete impedance region and
each domain boundary. Its surface must not intersect a PML.

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
tangential boundary action is supplied by the surface current. The local
face-connectivity checks above ensure that this clipped update is compiled
only for an unambiguous manifold boundary.

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
          =-k_{0,\mathrm{physical}}\operatorname{Im}n_{\mathrm{eff}},
   \qquad k_{0,\mathrm{physical}}=\frac{2\pi f}{c}.

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
realization internally: there is no proportional :math:`sE` term. Users
select resistance, a preset, or conductivity rather than supplying the
realization coefficients. Every fitted model has a finite mandatory validity
band. FDTD can advance the passive realization over its whole discrete
spectrum, but accuracy is advertised only inside that band. An
impedance-aware FDFD solve refuses to extrapolate a declared physical or
bilinear-warped evaluation frequency.

At construction time gprMax verifies that the generated coefficients are
finite, dimensions agree, every eigenvalue of ``A`` has strictly negative
real part, and the direct term is non-negative. At the actual FDTD time step
the code additionally checks the mapped unit-circle response for negative
real impedance. Active surface impedances are not part of the public API.

Common-metal Foster presets
---------------------------

For a thick good conductor, the target under the stated time convention is

.. math::

   Z_{\mathrm{gc}}(j\omega)
      =(1+j)\sqrt{\frac{\omega\mu_0}{2\sigma}}
      =(1+j)\sqrt{\pi f\mu_0\rho}.

The stored 293 K bulk-pure-metal resistivities are:

.. list-table:: Common-metal preset data at 293 K
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
imaginary target with relative-error weighting. The residues remain
non-negative, so the result is passive over the complete frequency axis, not
only at the sample points. A separate grid of at least 16,385 points certifies
the reported maximum and RMS errors.

Automatic order tests one, two, and then increasing actual Foster state counts
through 64. It stops at the first count whose deterministic local searches
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
and tolerance. The advertised error still applies only inside the requested
band.

For the default 0.2% tolerance, representative selections are:

.. list-table:: Automatic Foster order versus fitted bandwidth
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
FDTD kernel and FDFD reduction. The FDTD pack stores its diagonal local form
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

The kernel then recovers every :math:`k_p^{n+1/2}` and advances each local
Foster pole

.. math::

   y_{p,m}^{n+1}=f_m y_{p,m}^n+q_mk_p^{n+1/2}.

This analytic scalar elimination naturally handles several faces sharing one
electric edge; it does not perform a dense solve at each time step. A flat
boundary edge normally has one port and one current. A convex manifold edge
can have two face ports: they share the scalar electric solve but retain
separate histories and separate :math:`k_p` values.

Packed data and update order
----------------------------

The compiler packs:

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

For the diagonal Foster pack this is equivalently the local pole sum

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
compiled clipped line circulation, normalized by :math:`A_e k_0`, where
:math:`k_0=\Omega/c`. Independent retained masks remove metal-interior E and H
degrees of freedom without
misusing PEC masks; in particular, interface-normal H can remain present when
a collocated tangential E is a valid impedance-boundary unknown. The existing
P/Q reduction solves for :math:`\lambda=-n_{\mathrm{operator}}^2`, with
:math:`n_{\mathrm{operator}}=K_w/k_0`. The normal spacing :math:`\Delta w`
then determines the phase propagation constant and public effective index:

.. math::

   \beta=\frac{2}{\Delta w}\sin^{-1}\left(\frac{K_w\Delta w}{2}\right),
   \qquad n_{\mathrm{eff}}=\frac{\beta}{k_{0,\mathrm{physical}}}.

The passive forward branch gives attenuation and evanescent decay in positive
``w``. Modal field reconstruction uses :math:`n_{\mathrm{operator}}`; source
and monitor spatial phases use :math:`\beta`.

Low-level calls that omit ``fdtd_dt`` retain the physical-frequency P/Q
normalization: the coefficient denominator uses :math:`j\omega\epsilon_0A_e`
and :math:`k_0=\omega/c`, while the boundary numerator still comes from its
exact discrete recurrence. See :doc:`eigenmode_port` for both optional grid
parameters.

.. important::

   The surface ADE, midpoint factor, boundary electric mass, conductivity,
   and clipped transverse curl are reduced exactly for the FDTD time step.
   Eigenmode sources also use the owning grid's leapfrog temporal symbol and
   longitudinal spatial difference. Bulk nondispersive conductivity includes
   its midpoint factor. Bulk dispersive material poles still use their
   analytic physical-frequency response rather than the exact volume ADE
   transfer, so the general dispersive bulk eigenproblem is not fully time
   discrete. Keep modal anchors below Nyquist and check mesh/time-step
   convergence.

The longitudinal :math:`K_w` coupling remains the standard implicit P/Q
term. The source-plane mapper checks that the omitted longitudinal H weights
form equal and opposite contributions in the two cells adjacent to the plane.
That is why a changing wall cross-section or an impedance end cap at the
modal plane is rejected.

Eigenmode solution and FDTD injection
=====================================

A direct modal solve reuses the component IDs, retained masks, clipped H
weights, dual fractions, port models, FDTD ``dt``, and normal cell spacing
from the already compiled three-dimensional grid. There is no separately
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
comparison uses :math:`-\ln|S_{21}|/L`, so its loss gate is independent of this
phase correction.

Validation and benchmarking
===========================

FDFD attenuation and modal launch
---------------------------------

The focused FDFD operator test constructs a copper-lined rectangular guide,
compares :math:`-k_{0,\mathrm{physical}}\operatorname{Im}n_{\mathrm{eff}}` with
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
:math:`-k_{0,\mathrm{physical}}\operatorname{Im}n_{\mathrm{eff}}` stored for
each exact in-band FDFD anchor is compared with :math:`Q\operatorname{Re}Z_s`
using a 1% relative L2 gate. The driven FDTD port must have maximum
:math:`S_{11}<-20` dB after the
complex modal field is injected. Finally, attenuation obtained from the FDTD
two-plane propagation factor is compared with the same perturbation theory
using a 2% relative L2 gate. The workflow therefore exercises the copper
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

HDF5 reproducibility metadata
=============================

Every output file containing a surface definition has a root group
``/surface_impedance_models``. Its attributes are:

* ``SchemaVersion = 3``;
* ``TimeConvention = exp(+j*omega*t)``;
* ``FourierAnalysisConvention = exp(-j*omega*t)``;
* ``SurfaceNormalConvention = metal_to_retained_dielectric``;
* ``SurfaceCurrentConvention = K = n_m cross H``;
* ``Units = SI``;
* ``FDTDTimeStep``.

Each definition appears as ``model1``, ``model2``, and so on, in definition
order. A model group stores ``ID``, ``ModelHashSHA256``, ``Order``, ``D``, fit
limits, source kind, conductivity, preset/provenance, requested and selected
pole counts, fit method and tolerance, maximum/RMS errors, plotting policy, and
whether the compiled boundary uses it. The continuous ``A``, ``B``, and ``C``
arrays are datasets. A preset additionally stores reference temperature and
resistivity.

For every model used by the compiled boundary, ``fdtd_discrete`` stores the
exact local ``f`` and ``q`` pole vectors and the ``TimeStep``, ``Z0``, and
``PassivityChecked`` attributes. Together with the continuous realization,
these reproduce both the in-place FDTD recurrence and the equivalent FDFD
transfer. A defined but unused model has no discrete subgroup. The model hash
covers the continuous model and fitting provenance. The current schema does
not serialize the complete
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
        f = wall['fdtd_discrete/f'][...]
        q = wall['fdtd_discrete/q'][...]
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
    The trapezoidal unit-circle response has negative real impedance. This is
    an internal fit failure for the public passive models and should be
    reported with the input and fit band.

``frequency is outside ... fit band``
    Expand the preset or conductivity fit band to include every eigenmode anchor and
    its bilinear-warped frequency. Close to Nyquist the warped frequency can
    be much higher than the physical anchor. Reducing ``dt`` also reduces the
    difference.

``impedance-volume voxel topology is non-manifold at a Yee edge``
    The error reports the edge orientation and grid index where two impedance
    quadrants touch only diagonally. Connect the impedance cells through a
    full voxel face, or move them apart so they no longer share the reported
    edge. Refining the mesh, thickening the feature, increasing a curved
    object's radius, or adjusting its position can change the final
    rasterization.

``impedance-volume voxel topology is non-manifold at grid vertex``
    The error reports the vertex index and whether the impedance cells,
    retained cells, or both are not face-connected within its incident
    ``2 x 2 x 2`` neighbourhood. Reshape the geometry so both sets connect
    through voxel faces, or separate the impedance regions so they share
    neither an edge nor a vertex. The check includes contacts between
    different surface-impedance IDs.

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

Dispersive retained media and volume ADEs
-----------------------------------------

Supporting a dispersive exterior requires a locally coupled row containing
both volume-polarization and surface-current states, including corner
averaging policy. The modal solver already uses the leapfrog temporal and
longitudinal spatial symbols when constructed by an eigenmode source. An
exact treatment of bulk dispersive poles would additionally replace their
analytic physical-frequency response with the corresponding FDTD volume-ADE
transfer. This remaining extension is separate from the exact surface-ADE
reduction and the existing compensation of Yee numerical dispersion.

References
==========

The Yee grid convention follows [YEE1966]_. Background on
surface-impedance FDTD boundaries is given by [MAL1992]_ and [BEG1992]_. The
preset resistivity provenance is [MAT1979]_, [DES1984A]_, and [DES1984S]_.
