Gradient-first mode separation investigation
===========================================

The proposed TE/TM, directional-gradient, then symmetry pipeline is useful as
a basis-selection strategy, but it does not uniquely separate every physical
mode and is not established as better than the current implementation. Keep
symmetry certification and the existing fallbacks. The most useful extension
would be an optional total-gradient probe for unresolved equivalent copies,
with field-history continuation deciding their identities across frequency.

This investigation interprets the request's ``TE/TE`` as TE/TM. It changes no
runtime selection or public controls. The offline prototype, counterexample
tests, raw results, and source-hash manifest accompany this report.

Repository state
----------------

Fetched https://github.com/gprMax/gprMax.git on 24 October 2026. Rebasing
``master`` onto ``upstream/master`` advanced it from ``d068a023`` to
``8d010441b4a6223dea01e8653a4aa62434eb13a4`` without conflicts or local commits
to replay. Created and checked out ``auto-tracking-improve`` at that commit.
The pre-existing untracked ``.worktrees/`` directory was retained.

What the current code does
--------------------------

``gprMax/eigenmode_symmetry.py`` detects common sampled-material and boundary
symmetry, certifies closure of the full vector fields, then calls
``longitudinal_blocks``. That function solves the TE/TM-content problem before
decomposing each family into irreducible symmetry blocks. Next,
``axis_sensitive_blocks`` uses directional gradients to orient compatible
equal-parity singlets, including square TE20/TE02. Finally,
``orient_irreducible_block`` uses transverse electric moments for suitable
doublets and mirror parity otherwise. Thus the current implementation already
combines longitudinal, symmetry, and directional information.

``gprMax/eigenmode_tracking.py::_polarize_resolved_pairs`` also already lets
users specify moment-based polarization within resolved doublets. It rejects
zero or ill-conditioned moments and validates power, continuity, and mixed
Maxwell residuals. Requiring C4v E for every user mix would narrow the present
capability: an admissible degenerate pair without C4v can also support valid
mixing when its moments and residuals pass.

Operator meaning
----------------

Use full projected matrices, including off-diagonal terms, rather than scores
of the arbitrary raw eigenvectors. For native longitudinal sample matrices,
define

.. math::

   M=E_w^\dagger E_w+Z_{\rm ref}^2H_w^\dagger H_w,\quad
   L_E=E_w^\dagger E_w,\quad D=K_u-K_v,\quad S=K_u+K_v.

Solve ``L_E c = rho M c``, then ``D c = eta M c`` within each unresolved
rho family. These generalized eigenproblems remove arbitrary input scaling
and basis mixing. TEM fields make M singular; quasi-TEM fields can make it
insignificant or poorly conditioned. Retain the current fallback in those
cases. Hybrid rho values are useful descriptors, not universal mode labels.

In a rectangular homogeneous guide, the continuum scalar-mode scores are
``eta = (m*pi/a)^2 - (n*pi/b)^2`` and
``s = (m*pi/a)^2 + (n*pi/b)^2``. Their sum and difference determine the two
squared directional wave numbers. This explains the success for rectangular
index pairs. In a homogeneous PEC guide, integration by parts gives
``S = kc^2 M`` on a fixed-cutoff TE or TM eigenspace. Consequently S provides
no additional distinction *inside that degenerate eigenspace*.

Circular counterexamples
------------------------

Exact unit-disk Bessel fields were integrated with 96-point radial
Gauss-Legendre quadrature and 128 uniform angular samples. Gradients are
analytical and integrated only inside the disk; padding nonzero boundary Hz
with zeros would create a false gradient contribution. The scalar Bessel
construction follows the circular-waveguide equations in
`MIT's waveguide notes <https://fab.cba.mit.edu/classes/862.16/notes/circuits.pdf>`_.
The operator and moment conclusions below are derived and checked here.

For a fixed angular order m and radial root n, both cosine and sine partners
share rho. The projection of D onto their span vanishes for m >= 2, whereas
S is ``kc^2 M``. Adding S therefore cannot select individual angular partners.
For m=2 they are distinct C4v singlets; for m=3 they form one E doublet.
Symmetry/parity information resolves a basis that these two gradient measures
cannot choose. Even a physically valid degenerate pair has no unique individual
basis until an orientation convention is specified.

Axisymmetric m=0 modes also have D=0. S distinguishes their radial roots:

.. list-table:: Unit-radius scores for the first two axisymmetric radial orders
   :header-rows: 1

   * - Family
     - D norm
     - S eigenvalues
   * - TE
     - 4.1e-15
     - 14.68197064, 49.21845632
   * - TM
     - 1.8e-15
     - 5.78318596, 30.47126234

These different radial orders already have distinct cutoffs and propagation
constants. They are compared as descriptors, not authorized for field mixing.
S is helpful for recognizing radial scale, but is not an extra universally
necessary separator within an already certified degenerate parent span.

Equal absolute directional scores and C4v E
------------------------------------------

The candidate idea has a sound symmetry basis. In an M-orthonormal frame of a
closed C4v span, a quarter turn sends D to -D, while axis mirrors preserve D.
Thus the spectral subspaces of D squared, equivalently equal absolute eta,
are symmetry-invariant. This makes them reasonable *candidate spans* when
formed with a scale-aware tolerance inside one compatible longitudinal family
and one admissible persistent degeneracy group.

It is not an E certificate. Square TE20/TE02 have opposite directional scores
of equal magnitude, but a half turn is +I on their span; they comprise A2+B2.
For an E doublet, a half turn is -I. Check the full vector actions, closure,
group relations, and irreducibility on each candidate, not only traces or
longitudinal-field parity. A zero-score candidate may contain several singlets
or multiple E copies and must be decomposed further. Never collect all modes
with a matching scalar across different propagation constants or TE/TM families.

In repeated E copies, S and D squared need not commute, despite each being
symmetry-invariant. A positive-gradient matrix counterexample is included in
the tests; it is an algebraic limitation, not a claimed Maxwell geometry.
Hence ordering these probes can select different copy bases. Neither choice
automatically establishes physical branch identity; retain frequency-history
information and unresolved subspaces.

Electric moments after E certification
-------------------------------------

E symmetry permits a transverse vector moment but does not require one.
The TE11 disk pair has nonzero, full-rank moments. The TE31 E pair has zero
integrated transverse E by angular cancellation. Uniform PEC TM modes have
zero integrated transverse E because the integral of grad(Ez) reduces to a
boundary integral with Ez=0. TM11 is therefore another E counterexample.
Numerical moment norms in these checks are below 4e-14 after normalizing the
longitudinal fields; common electromagnetic prefactors are omitted.

Use the electric-moment transformation only for a resolved pair with
significant, well-conditioned rank-two moments. Otherwise use the established
mirror/continuation basis. If a user wants to mix a zero-moment pair, a future
explicit complex coefficient or parity-basis angle interface would express
that request; a transverse-moment direction cannot define it. Apply every
accepted transformation to all E/H components and retain the existing
propagation, power, measurement-conditioning, and residual checks.

Measured comparison and validation
----------------------------------

At 20 GHz, the native full-vector FDFD checks used a 20 mm square PEC guide
with relative permittivity 2 on 12x12 and 20x20 meshes. Seeded complex rotations
were applied inside the exact eigenspaces before identification. The groups
were TE10/TE01, TE20/TE02, and TE12/TE21/TM12/TM21.

Both the prototype and current implementation achieved minimum longitudinal
analytical-template scores greater than ``1-1e-10`` in all six cases. The
prototype's maximum mixed eigen-residual was ``6.35e-15``. The TE20/TE02
half-turn trace was +2; E pairs had trace -2 per pair. These are positive
controls for separation and the false E candidate, not evidence of improved
scattering accuracy or speed. Lossy, anisotropic, hybrid, and open-port
performance of a replacement pipeline has not been established here.

The four-file pytest run with a writable temporary directory produced
65 passes and one integration failure. The separate resolved-polarization
suite passed all 39 tests, giving 104 passes and one failure overall. The failure occurs in the installed
compiled magnetic PML extension: the Python caller and current Cython source
pass 23 arguments including ``pmc_mask``, while the loaded binary accepts 22.
The binary predates this investigation. The failure prevents claiming a full
source/receiver end-to-end pass; it does not occur in mode separation. See
``pytest.log`` for the complete failure. An earlier run could not create
pytest's default temporary directory, resolved by selecting a workspace path.

Reproduce from the repository root, selecting a fresh output directory::

   python -m testing.eigenmode_support.identification.gradient_first_investigation --output NEW_DIRECTORY
   python -m pytest tests/test_gradient_first_investigation.py tests/test_eigenmode_longitudinal.py tests/test_eigenmode_parity_orientation.py tests/test_eigenmode_auto_symmetry.py -q -p no:cacheprovider --basetemp=FRESH_TEMP_DIRECTORY
   python -m pytest tests/test_eigenmode_resolved_polarization.py -q -p no:cacheprovider --basetemp=ANOTHER_FRESH_TEMP_DIRECTORY

Recommended direction
---------------------

Keep persistent eigenvalue groups and continuation first. Retain longitudinal
family separation, then use directional or total-gradient probes only where
they produce a stable, validated split. Equal absolute directional scores can
reduce the symmetry candidate search, but retain full symmetry decomposition
for unresolved candidates. Resolve each E copy before moment-based user
polarization, with parity fallback for zero moments. This preserves the useful
part of the proposal without asserting that three scalar operators uniquely
identify arbitrary modes. The present evidence supports this guarded hybrid
approach; it does not support replacing the current default pipeline.
