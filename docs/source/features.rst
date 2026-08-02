.. _capabilities:

*****************
Software Features
*****************

This section highlights some of the key features of gprMax that are useful for GPR modelling as well as more general electromagnetic simulations.

Python API
==========

There is now a **Python API**, which includes all the functionality of the input file (hash) commands as well as several more advanced features. It allows users to access to gprMax functions directly from Python by importing the gprMax module. This method is recommended for those who prefer to use Python or need access to specific API-only advanced features, and is described in the :ref:`Python API <input-api>` section. There are several advantages to using the Python API:

1. Users can take advantage of the Python language - for instance, the structural elements of Python can be utilised more easily.
2. gprMax objects can be used directly within functions, classes, modules and packages. In this way, collections of components can be defined, reused, and modified. For example, complex targets can be imported from a separate module and combined with an antenna from another module.
3. The API can interface with other Python libraries. For example, the API could be used to create a parametric antenna and the external library Scipy could then be used to optimise its parameters.

Two-dimensional TM and TE modes
===============================

Models that are invariant in one Cartesian direction can use either a
transverse-magnetic (TM) or transverse-electric (TE) field reduction. The
invariant axis is declared with ``inf`` in ``#domain`` and the polarisation is
selected with ``#domain_mode``. The same modes are available through the
Python API. Sources, receivers, snapshots, material construction, fractal
geometry, and imported geometry objects observe the reduced component set.
See :ref:`input-hash-cmds` and :ref:`guidance` for the command syntax and field
equations.

Symmetry boundaries
===================

PEC and PMC symmetry planes can replace the PML on selected model faces. PMC
planes use an image-theory ghost-node update, while PEC planes constrain the
tangential electric field through the Yee material construction. Multiple
planes may be combined. KSIR transforms using an NTFF integration surface can
use a symmetry plane as an omitted face that is completed by image theory. See
``#symmetry_boundary`` in :ref:`input-hash-cmds` for supported solvers and
current restrictions.

Excitation options
==================

Models can be excited using the range of local sources available in gprMax:
Hertzian electric and magnetic dipoles, hard or resistive voltage sources, and
one-dimensional transmission-line feeds. The dipoles provide idealised local
radiators, while voltage and transmission-line sources can feed explicit
antenna geometries.

Plane-wave excitation is available through a total-field/scattered-field
(TFSF) surface. gprMax uses the finite-difference time-domain discrete plane
wave (FDTD-DPW) formulation of Tan and Potter [TAN2010]_. Its auxiliary
one-dimensional FDTD grid reproduces the numerical propagation of the main
grid, forming a nearly perfectly matched TFSF source with very low numerical
leakage into the scattered-field region. Plane waves can be specified by
propagation angles or an integer direction vector in a homogeneous background;
an axial form is available for normally incident layered-media models.

The source and plane-wave commands are described in
:ref:`input-hash-cmds`.

.. _ntff-formulations:

Near-to-far-field transformations
==================================

gprMax provides two complementary closed-surface formulations. The Kirchhoff
surface-integral representation (KSIR) reconstructs finite-distance fields as
well as far fields in the time and frequency domains. The conventional
Love-equivalent-current formulation provides an independent far-zone result,
using a direct frequency-domain transform or the modified time-domain method
of Giannopoulos *et al.* [GIAFF1997]_. A single ``NTFFSurface`` can be reused
by both formulations, so their results can be compared without changing the
FDTD model or integration surface.

The available formulations are summarised below.

.. list-table:: NTFF formulations
   :header-rows: 1
   :widths: 22 22 16 18 22

   * - Formulation
     - Result
     - Domain
     - Finite distance
     - Implementation notes
   * - KSIR
     - Electric and magnetic fields
     - Time or frequency
     - Yes
     - Symmetry completion is supported
   * - Love currents
     - Far-zone fields
     - Frequency
     - No
     - Six physical faces are required
   * - Modified Love currents [GIAFF1997]_
     - Far-zone fields
     - Time
     - No
     - CPU and six physical faces are required

Definitions and conventions
---------------------------

The integration surface :math:`S` must enclose all radiating sources, or the
complete TFSF box and scatterer for a scattering calculation. It lies in a
homogeneous, linear, lossless and non-dispersive background with

.. math::

    c_b=\frac{1}{\sqrt{\mu_b\epsilon_b}},\qquad
    \eta_b=\sqrt{\frac{\mu_b}{\epsilon_b}},\qquad
    k=\frac{\omega}{c_b}.

The unit normal :math:`\hat{\mathbf n}` points out of the enclosed volume,
:math:`\mathbf r'` denotes a source point on :math:`S`, and
:math:`\mathbf r_0` is the phase origin (the surface centre for hash-command
inputs). Spherical angles use :math:`\theta` from ``+z`` and :math:`\phi`
from ``+x`` towards ``+y``.

Frequency-domain results use the electrical-engineering convention

.. math::

    \mathbf E(t)=\Re\{\widetilde{\mathbf E}(\omega)e^{+j\omega t}\},
    \qquad
    \widetilde{\mathbf E}(\omega)=
    \int \mathbf E(t)e^{-j\omega t}\,\mathrm dt,

so an outward wave contains :math:`e^{-jkr}`. Far-zone datasets store the
range-normalised quantities

.. math::

    \mathbf F_E(\hat{\mathbf r},\omega)
    =\lim_{r\rightarrow\infty}r e^{+jkr}
      \widetilde{\mathbf E}(\mathbf r,\omega),\qquad
    \mathbf F_H=\frac{1}{\eta_b}\hat{\mathbf r}\times\mathbf F_E.

They therefore have no observation-radius parameter. KSIR finite-distance
receivers instead retain the physical ``1/R`` and ``1/R**2`` dependence.

KSIR
----

The Kirchhoff surface-integral representation (KSIR) in gprMax is based on the
formulation introduced by Ramahi [RAM1997]_. For any Cartesian electric- or
magnetic-field component :math:`\psi`, the time-domain field outside a closed
surface :math:`S` is

.. math::

    \psi(\mathbf r,t) = \frac{1}{4\pi}\oint_S
    \left[
    -\frac{1}{R}\frac{\partial\psi(\mathbf r',t_R)}{\partial n'}
    + \frac{\hat{\mathbf n}'\mathbin{\cdot}\hat{\mathbf R}}{R^2}
      \psi(\mathbf r',t_R)
    + \frac{\hat{\mathbf n}'\mathbin{\cdot}\hat{\mathbf R}}{c_b R}
      \frac{\partial\psi(\mathbf r',t_R)}{\partial t}
    \right] \mathrm{d}S',

where :math:`\mathbf R=\mathbf r-\mathbf r'`, :math:`R=|\mathbf R|`,
:math:`\hat{\mathbf R}=\mathbf R/R`, :math:`\hat{\mathbf n}'` is the outward
surface normal, :math:`t_R=t-R/c_b` is the retarded time, and :math:`c_b` is
the wave speed in the homogeneous background medium. Each requested field
component is reconstructed independently from that component and its
outward-normal derivative; KSIR does not require equivalent electric and
magnetic surface currents.

gprMax extends the original time-domain presentation by directly accumulating
frequency-domain surface phasors. With the electrical-engineering convention
:math:`\psi(t)=\Re\{\widetilde{\psi}(\omega)e^{+j\omega t}\}` and forward
transform kernel :math:`e^{-j\omega t}`, the exact finite-distance form used
by gprMax is

.. math::

    \widetilde{\psi}(\mathbf r,\omega) = \frac{1}{4\pi}\oint_S
    e^{-j k R}
    \left[
    -\frac{1}{R}\frac{\partial\widetilde{\psi}(\mathbf r',\omega)}
      {\partial n'}
    + (\hat{\mathbf n}'\mathbin{\cdot}\hat{\mathbf R})
      \left(\frac{1}{R^2}+\frac{j k}{R}\right)
      \widetilde{\psi}(\mathbf r',\omega)
    \right] \mathrm{d}S',

where :math:`k=\omega/c_b`. Its far-zone limit supplies the range-normalized
radiation and scattering fields.

The implementation also uses a Yee-aware interpolation approach. The common
logical box defines six closed faces, but each Cartesian component is sampled
on its own correctly offset Yee surface; the six components are not first
forced onto one Huygens surface. For each component and face, two samples of
that *same* component straddle the mathematical component surface and are
centred and differenced as

.. math::

    \psi_S = \frac{\psi_{\mathrm{out}}+\psi_{\mathrm{in}}}{2},
    \qquad
    \frac{\partial\psi_S}{\partial n'} =
    \frac{\psi_{\mathrm{out}}-\psi_{\mathrm{in}}}{\Delta n}.

This retains a centred normal derivative without introducing cross-component
spatial interpolation. Electric samples remain at integer Yee time levels
and magnetic samples at half-integer levels; the frequency transform includes
those actual sample times. Time-domain fractional propagation delays are
deposited between their two neighbouring output samples.

A reusable integration surface can feed multiple Cartesian or spherical
observation points and frequency transforms. CPU collection is implemented
with Cython/OpenMP; CUDA, OpenCL, and Metal keep collection state and
time-domain storage on the device during the FDTD iterations. See the KSIR
command reference in :ref:`input-hash-cmds` and the HDF5 schema in
:ref:`output`.

Equivalent electric and magnetic currents
------------------------------------------

Unlike KSIR, the equivalent-current formulation first collocates the
tangential Yee fields at common cell-face centres. Arithmetic interpolation
is used only in the directions required by the Yee staggering. The outward
Love currents are then

.. math::

    \mathbf J_s=\hat{\mathbf n}\times\mathbf H,
    \qquad
    \mathbf M_s=-\hat{\mathbf n}\times\mathbf E.

Frequency-domain far field
^^^^^^^^^^^^^^^^^^^^^^^^^^

The frequency-domain transform follows the conventional closed-surface FDTD
construction of Luebbers *et al.* [LUE1991]_. Define

.. math::

    \mathbf N(\hat{\mathbf r},\omega)=
    \oint_S\mathbf J_s(\mathbf r',\omega)
    e^{+jk\hat{\mathbf r}\cdot(\mathbf r'-\mathbf r_0)}\,\mathrm dS',
    \qquad
    \mathbf L(\hat{\mathbf r},\omega)=
    \oint_S\mathbf M_s(\mathbf r',\omega)
    e^{+jk\hat{\mathbf r}\cdot(\mathbf r'-\mathbf r_0)}\,\mathrm dS'.

For the engineering convention stated above, the stored electric far field is

.. math::

    \mathbf F_E=-\frac{jk}{4\pi}
    \left[
    \eta_b\left(\mathbf N-
    \hat{\mathbf r}(\hat{\mathbf r}\cdot\mathbf N)\right)
    -\hat{\mathbf r}\times\mathbf L
    \right].

This supplies radiation patterns, antenna quantities, and RCS independently
of the scalar KSIR construction. Direct frequency accumulation avoids storing
the complete surface-field history.

Modified 1997 time-domain far field
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`\tau=t-r/c_b` be reduced time and let a dot denote a time
derivative. The range-normalised transient electric field implemented by
gprMax is

.. math::

    \mathbf F_E(\hat{\mathbf r},\tau)
    =-\frac{1}{4\pi c_b}\oint_S
    \left[
    \eta_b\dot{\mathbf J}_{s,t}
    -\hat{\mathbf r}\times\dot{\mathbf M}_s
    \right]
    \left(\tau+\frac{\hat{\mathbf r}\cdot
    (\mathbf r'-\mathbf r_0)}{c_b}\right)\,\mathrm dS',

where

.. math::

    \mathbf J_{s,t}=\mathbf J_s-
    \hat{\mathbf r}(\hat{\mathbf r}\cdot\mathbf J_s)

is transverse to the observation direction. The magnetic far field follows
from :math:`\mathbf F_H=(\hat{\mathbf r}\times\mathbf F_E)/\eta_b`.

The original Luebbers time-domain construction [LUE1991]_ interpolates the
electric and magnetic equivalent-current contributions onto a common time
level. The modification of Giannopoulos *et al.* [GIAFF1997]_ instead retains
their natural Yee staggering. In gprMax,
:math:`\mathbf M_s^n=-\hat{\mathbf n}\times\mathbf E^n` is differenced at
:math:`(n-1/2)\Delta t`, whereas
:math:`\mathbf J_s^{n+1/2}=\hat{\mathbf n}\times\mathbf H^{n+1/2}` is
differenced at :math:`n\Delta t`. The two contributions are deposited
independently; linear interpolation is used only for the generally fractional
propagation delay to the reduced-time grid. Thus the extra Yee-time
interpolation removed by the 1997 method is not reintroduced by the
implementation.

Only the interval supported by every integration patch is returned. This
removes the range-dependent zero prefix and prevents an incomplete
retarded-time tail from being presented as a physical late-time response.

Equivalent-current outputs are far-zone quantities and therefore have no
radius parameter. KSIR remains the appropriate choice when finite-distance or
near-field reconstruction is required. The frequency-domain
equivalent-current collector supports CPU, CUDA, OpenCL, and Metal; its
angular evaluation is Cython/OpenMP post-processing. The one-step transient
implementation currently uses the CPU solver.

Subgridding
===========

Including finely detailed objects or regions of high dielectric strength in FDTD modeling can dramatically increase the computational burden of the method. This is because the conditionally stable nature of the algorithm requires a minimum time step for a given spatial discretization. Thus, when the spatial discretization is lowered, either to reduce numerical dispersion or include small-sized features, the time step must be reduced. Also, the number of spatial cells is increased. One approach to reducing the overall computational cost is to introduce local finely discretized regions into a coarser finite-difference grid. This approach is known as subgridding. The computing time is reduced since there are fewer cells to solve. Also, there are fewer iterations since the coarse time step is maintained in the coarse region. gprMax uses a new Huygens subgridding (HSG) algorithm with a novel artificial loss mechanism called the switched Huygens subgridding (SHSG). For a detailed description of subgridding and the SHSG method please read [HAR2021]_. Examples of how to use the subgridding functionality can be found in the :ref:`Advanced features <examples-subgrid>` section.

Dispersive materials
====================

gprMax has always included the ability to represent dispersive materials using a single-pole Debye model. Many materials can be adequately represented using this approach for the typical frequency ranges associated with GPR. However, multi-pole Debye, Drude and Lorentz functions are often used to simulate the electric susceptibility of materials such as: water [PIE2009]_, human tissue [IRE2013]_, cold plasma [LI2013]_, gold [VIA2005]_, and soils [BER1998]_, [GIAK2012]_, [TEI1998]_. Electric susceptibility relates the polarization density to the electric field, and includes both the real and imaginary parts of the complex electric permittivity variation. In the new version of gprMax a recursive convolution based method is used to express dispersive properties as apparent current density sources [GIA2014]_. A major advantage of this implementation is that it creates an inclusive susceptibility function that holds, as special cases, Debye, Drude and Lorentz materials. For further details see the :ref:`material commands section <materials>`.

Realistic soils, heterogeneous objects and rough surfaces
=========================================================

The inclusion of improved models of soils is important for many GPR simulations. gprMax can now be used to create soils with more realistic dielectric and geometrical properties. A semi-empirical model, initially suggested by [DOB1985]_, is used to describe the dielectric properties of the soil. The model relates relative permittivity of the soil to bulk density, sand particle density, sand fraction, clay fraction and water volumetric fraction. Using this approach, a more realistic soil with a stochastic distribution of the aforementioned parameters can be modelled. The real and imaginary parts of this semi-empirical model can be approximated using a multi-pole Debye function plus a conductive term. This can now be achieved in gprMax using the new dispersive material functionality. For further details see the :ref:`material commands section <materials>`.

Fractals are scale invariant functions which can express the topography of the earth for a wide range of scales with sufficient detail [TUR1987]_. For this reason fractals have been chosen to represent the topography of soils. Fractals can be generated by the convolution of Gaussian noise with an inverse Fourier transform of :math:`\frac{1}{kb}`, where :math:`k` is the wavenumber and :math:`b` is a constant related to the fractal dimension [TUR1997]_. gprMax can now generate heterogeneous volumes (boxes) with realistic soil properties that can have rough surfaces applied. For further details see the :ref:`fractal object building commands section <fractals>`.

Fractal correlated noise [TUR1997]_ is used to describe the stochastic distribution of the properties of soils. This approach has been chosen because it has been shown that soil-related environmental properties frequently obey fractal laws [BUR1981]_, [HILL1998]_. For further details see the :ref:`material commands section <materials>` and the :ref:`fractal object building commands section <fractals>`.

.. _antennas:

Library of antenna models
=========================

gprMax now includes Python modules with pre-defined models of antennas that behave similarly to commercial antennas [WAR2011]_ [STA2017]_. Currently models of antennas similar to `Geophysical Survey Systems, Inc. (GSSI) <http://www.geophysical.com>`_ 1.5 GHz (Model 5100) antenna, and 400 MHz antenna, as well as `MALA Geoscience <http://www.malags.com/>`_ 1.2 GHz antenna are included. By taking advantage of our Python API, using such complex structures in a model is straightforward without having to be built step-by-step by the user. For further details see the :ref:`Python API <input-api>` section.

Anisotropic materials
=====================

It is possible to specify objects that have diagonal anisotropy which allows materials such as wood and fibre-reinforced composites, often imaged with GPR, to be more accurately modelled. Standard isotropic objects specify one material identifier that defines the same properties in x, y, and z directions. However, every volumetric object building command can also be specified with three material identifiers, which allows properties for the x, y, and z directions to be separately defined.

Dielectric smoothing
====================

At the boundaries between different materials in a model there is the question of what electric and magnetic material properties to use?

* Should the last object to be defined at that location dictate the electric and magnetic properties?
* Should an average set of electric and magnetic properties of the materials of the objects that share that location be used?

This latter option is often referred to as dielectric smoothing and has been shown to result in more accurate simulations [LUE1994]_ [BOU1996]_ [WHI2009]_. To address this question gprMax includes an option to turn dielectric smoothing on or off for volumetric object building commands. The default behaviour (if no option is specified) is for dielectric smoothing to be on. The option can be specified with a single character ``y`` (on) or ``n`` (off) given after the material identifier in each object command. When dielectric smoothing is on, gprMax uses an arithmetic mean for the four cells surrounding each electric-field edge and, by default, a harmonic mean for the two cells normal to each magnetic-field edge. The harmonic magnetic average follows continuity of the normal magnetic flux density. The earlier arithmetic magnetic behaviour remains available for reproducing results from older versions; see ``#magnetic_averaging`` in :ref:`input-hash-cmds`.

Perfectly Matched Layer (PML) absorbing boundary conditions
===========================================================

With increased research into quantitative information from GPR, it has become
necessary for models to have more efficient and better-performing Perfectly
Matched Layer (PML) absorbing boundary conditions. Since 2005 gprMax has
featured PML absorbing boundary conditions based on the uniaxial PML (UPML)
[GED1998]_ formulation. A PML based on a recursive integration approach to the
complex frequency shifted (CFS) PML has been adopted since the major
redevelopment of gprMax (v3), and it is used exclusively.

Both Higher-Order Recursive Integration PML (HORIPML) [GIA2012]_ and Multipole
Recursive Integration PML (MRIPML) [GIA2018]_ formulations are available. The
higher-order formulation combines CFS stretching functions as a product,
whereas the multipole formulation combines constituent CFS poles as a sum and
can provide advanced broadband and late-time boundary absorption. First- and
second-order configurations are currently supported.

The formulation, thickness on each model boundary, and the parameters of every
CFS term are fully customisable. Advanced users can set the minimum and maximum
values, polynomial grading profile, and grading direction independently for
:math:`\alpha`, :math:`\kappa`, and :math:`\sigma`. This allows the PML to be
optimised for a particular application. RIPML corrections are applied after
the standard FDTD field updates and are agnostic to the underlying medium, so
the same formulation can be used with dispersive and anisotropic materials.
See the PML command reference in :ref:`input-hash-cmds`.

Open source, robust, file formats
=================================

Alongside improvements to the input file there is a new output file format – `HDF5 <http://www.hdfgroup.org/HDF5/>`_ – to manage the larger and more complex data sets that are being generated. HDF5 is a robust, portable and extensible format with a number of free readers available. For further details see the :ref:`Simulation Output <output>` section.

In addition, the `Visualization Toolkit (VTK) <http://www.vtk.org>`_ is being used for improved handling and viewing of the detailed 3D FDTD geometry meshes. The VTK is an open-source system for 3D computer graphics, image processing and visualisation. It also has a number of free readers available including `Paraview <http://www.paraview.org>`_. For further details see the :ref:`geometry view command <geometryview>`.

.. note::

    As of June 2025, gprMax uses the `VTKHDF file format
    <https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html#vtkhdf-file-format>`_
    rather than the previous `XML file format
    <https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html#xml-file-formats>`_
    in order to better support parallel I/O. The Paraview macro has been
    updated to reflect this change.
