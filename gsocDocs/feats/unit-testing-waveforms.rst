Unit Tests — Waveforms
======================

**Branch:** ``feat/unit-testing-waveforms``

**Module under test:** ``gprMax/waveforms.py``

**Test file:** ``tests/unit/waveforms/test_waveforms.py``

**Shared fixture:** ``tests/unit/conftest.py``

Scope
-----

Verifies that ``Waveform.calculate_value(time, dt)`` returns mathematically
correct amplitudes for each of the 12 supported ``type`` strings, that the
``amp`` and ``freq`` parameters scale the output as documented, and that error
paths behave predictably.

Tests do not exercise the FDTD solver, source injection, or any I/O. They are
pure-Python regression checks against closed-form references.

Test Infrastructure
-------------------

``tests/unit/conftest.py`` provides a single pytest fixture:

``make_waveform(wave_type, freq=1e9, amp=1.0)``
   Factory returning a configured ``Waveform`` instance. ``userfunc`` is left
   as ``None`` and must be set explicitly by tests that exercise
   ``type="user"``. Tests use ``freq=1e9`` (1 GHz) and ``dt=1e-12`` (1 ps)
   unless otherwise noted.

Test Catalog
------------

Each entry below lists the assertion, the property it verifies, and the most
likely source location to inspect on failure.

TestGaussian
^^^^^^^^^^^^

``test_peaks_at_chi_with_unit_value``
   Asserts ``calculate_value(chi) == 1.0`` for ``type="gaussian"`` where
   ``chi = 1 / freq``. Verifies the centering coefficient produced by
   ``calculate_coefficients`` (``waveforms.py:77``) matches the closed form
   ``exp(-zeta * delta**2)`` with ``delta = time - chi``. Failure indicates a
   wrong ``chi`` formula or an incorrect exponent at ``waveforms.py:99``.

``test_symmetric_around_chi``
   Asserts ``calculate_value(chi - delta) == calculate_value(chi + delta)``.
   The gaussian is even around its peak by definition. Failure indicates an
   off-by-one in the time index or a sign asymmetry in the kernel.

TestGaussianDot
^^^^^^^^^^^^^^^

``test_zero_at_chi``
   Asserts ``calculate_value(chi) == 0`` for ``type="gaussiandot"``. The first
   derivative of any function is zero at the function's peak. Failure
   indicates the derivative formula at ``waveforms.py:103`` has been altered
   so the ``delay`` factor no longer multiplies the exponential cleanly.

``test_antisymmetric_around_chi``
   Asserts ``calculate_value(chi - delta) == -calculate_value(chi + delta)``.
   The first derivative of an even function is odd about the symmetry axis.
   Failure indicates loss of sign symmetry in the derivative branch.

TestGaussianDotNorm
^^^^^^^^^^^^^^^^^^^

``test_peaks_at_unit_magnitude``
   Samples 5000 points over ``[0, 2/freq]`` and asserts
   ``max|calculate_value| == 1.0``. The ``norm`` suffix is defined by the
   normalisation factor ``sqrt(e / (2 * zeta))`` at ``waveforms.py:107``,
   which scales the peak magnitude to unity. Failure indicates a wrong
   normalisation factor or a wrong ``zeta`` for the ``gaussiandotnorm`` type.

TestGaussianDotDotNorm
^^^^^^^^^^^^^^^^^^^^^^

``test_trough_at_chi``
   Asserts ``calculate_value(chi) == -1.0`` for ``type="gaussiandotdotnorm"``
   with ``chi = sqrt(2) / freq``. At ``delta = 0`` the formula at
   ``waveforms.py:118-125`` reduces to
   ``(2 * zeta * -1 * exp(0)) / (2 * zeta) = -1``. Failure indicates either
   the wrong ``chi`` for the ricker family (``waveforms.py:80``) or a wrong
   normalisation factor at ``waveforms.py:118``.

TestRicker
^^^^^^^^^^

``test_is_negated_gaussiandotdotnorm``
   For ``t in {0.5, 1.0, 1.5, 2.0} * 1e-9``, asserts
   ``ricker(t) == -gaussiandotdotnorm(t)``. The ricker branch at
   ``waveforms.py:127-133`` is implemented as the explicit negation of the
   gaussiandotdotnorm formula. Failure indicates the sign in front of the
   ricker expression has been changed or the normalisation factor differs.

TestPrimeVsDot
^^^^^^^^^^^^^^

``test_gaussianprime_matches_gaussiandot_today``
   Asserts equality across five sample times. Both types share the
   ``elif`` branch at ``waveforms.py:101`` and both land in the
   gaussian-family coefficient branch at ``waveforms.py:70-78``, producing
   identical output. **This contradicts the source comment at
   ``waveforms.py:45-53``** which describes ``gaussianprime`` as the
   derivative of the base gaussian and ``gaussiandot`` as a user-tuned
   waveform with a distinct centre frequency. The test pins current
   behaviour. If the source is later changed so ``gaussiandot`` lands in
   the ricker-family coefficient branch, this test must be inverted to
   assert inequality.

``test_gaussiandoubleprime_differs_from_gaussiandotdot``
   Asserts the two types produce values differing by more than 50 % at
   ``t = 1 ns``. They share the formula at ``waveforms.py:110-114`` but
   use different coefficients:

   - ``gaussiandoubleprime`` → ``chi = 1/freq``, ``zeta = 2 * pi**2 * freq**2``
   - ``gaussiandotdot``      → ``chi = sqrt(2)/freq``, ``zeta = pi**2 * freq**2``

   Their distinctness is intentional and matches the comment at
   ``waveforms.py:45-53``. Failure indicates ``gaussiandoubleprime`` has
   been moved to the ricker-family coefficient branch (or vice versa),
   collapsing the two waveforms into one.

TestSine
^^^^^^^^

``test_zero_at_origin``
   Asserts ``calculate_value(0) == 0``. Direct consequence of
   ``sin(2 * pi * f * 0) == 0`` at ``waveforms.py:136``.

``test_unit_at_quarter_period``
   Asserts ``calculate_value(1/(4*freq)) == 1.0`` since
   ``sin(2 * pi * f * 1/(4*f)) == sin(pi/2) == 1``. Failure indicates a
   wrong frequency multiplier in the sine expression.

``test_silenced_after_one_cycle``
   Asserts ``calculate_value(1.5 / freq) == 0`` (exact integer 0, not
   approximate). Verifies the truncation branch at ``waveforms.py:137-138``
   that zeroes the output once ``time * freq > 1``. Failure indicates the
   ``> 1`` threshold has changed or the branch has been removed.

TestContSine
^^^^^^^^^^^^

``test_zero_at_origin``
   Asserts ``calculate_value(0) == 0``. At ``time = 0`` the ramp factor at
   ``waveforms.py:142`` evaluates to 0, zeroing the output.

``test_ramp_saturates_to_pure_sine``
   For ``t = 5.123 / freq`` (well past the ramp threshold of ``4 / freq``),
   asserts ``calculate_value(t) == sin(2 * pi * freq * t)``. Verifies the
   ramp clamp at ``waveforms.py:143``. Failure indicates the ramp formula
   ``0.25 * time * freq`` or the ``min(ramp, 1)`` clamp has been altered.

TestImpulse
^^^^^^^^^^^

``test_unit_at_origin``
   Asserts ``calculate_value(0, dt=1e-12) == 1``. Matches the ``time == 0``
   branch at ``waveforms.py:149``.

``test_unit_within_dt``
   Asserts ``calculate_value(dt/2, dt=dt) == 1``. Matches the ``time < dt``
   branch at ``waveforms.py:149``.

``test_zero_after_dt``
   Asserts ``calculate_value(2*dt, dt=dt) == 0``. Matches the
   ``time >= dt`` branch at ``waveforms.py:151``. Failure in any of the
   three impulse tests indicates the ``dt`` boundary condition has changed.

TestUser
^^^^^^^^

``test_calls_userfunc_and_scales_by_amp``
   Sets ``w.userfunc = lambda t: 4.0`` and ``amp = 3.0``; asserts
   ``calculate_value(0.5e-9) == 12.0``. Verifies that
   ``waveforms.py:155`` calls ``self.userfunc`` and the final
   ``ampvalue *= self.amp`` at ``waveforms.py:157`` applies. Failure
   indicates either the ``user`` branch has been removed or the amplitude
   multiplication has been skipped for that branch.

``test_passes_time_to_userfunc``
   Sets ``w.userfunc`` to a function that captures its argument; asserts the
   captured value equals the requested ``time``. Verifies that the source
   does not transform ``time`` before passing it to the user callback.

TestAmplitudeScaling
^^^^^^^^^^^^^^^^^^^^

``test_doubling_amp_doubles_output`` (parametrised over 11 types)
   For each ``type`` in ``ALL_TYPES``, asserts that two ``Waveform``
   instances with ``amp = 1.0`` and ``amp = 2.0`` produce outputs in
   exact 1:2 ratio at ``t = 0.5 ns``. Verifies that the final
   ``ampvalue *= self.amp`` at ``waveforms.py:157`` is executed regardless
   of which ``if/elif`` branch ran. Failure on a single ``type`` indicates
   that the ``amp`` multiplication has been moved inside one of the
   branches, missing the others.

``test_zero_amp_zero_output`` (parametrised over 11 types)
   For each ``type``, asserts ``calculate_value(0.5e-9) == 0`` when
   ``amp = 0``. Linearity edge case. Failure indicates the same scaling
   bug as above, or that one of the branches assigns ``ampvalue`` to a
   non-zero constant after the ``*= self.amp`` line.

TestCausality
^^^^^^^^^^^^^

``test_near_zero_at_origin`` (parametrised over ``gaussian``,
``gaussiandotnorm``, ``gaussiandotdotnorm``, ``ricker``)
   Asserts ``abs(calculate_value(0)) < 1e-6`` for each normalised
   gaussian-family type. The ``chi`` time-shift at ``waveforms.py:77`` and
   ``waveforms.py:80`` exists precisely so that ``t = 0`` lands on the far
   left tail of the exponential. Failure indicates ``chi`` has been
   removed or set incorrectly; without it the waveform would discharge
   non-trivial energy on the first time step, breaking causality at the
   source.

   Non-normalised variants (``gaussiandot``, ``gaussiandotdot``, etc.) are
   not tested here because their absolute peak magnitudes are O(1e10), so
   "near zero at origin" cannot be expressed as an absolute tolerance.

TestErrors
^^^^^^^^^^

``test_unknown_type_currently_raises_unbound_local``
   With ``type = "notarealtype"``, asserts ``calculate_value`` raises
   ``UnboundLocalError``. The ``if / elif`` chain at ``waveforms.py:96-156``
   has no ``else`` branch, so ``ampvalue`` is never assigned for unknown
   types and the final ``ampvalue *= self.amp`` at ``waveforms.py:157``
   accesses an unbound local. This test is a tripwire: when a follow-up
   PR adds an ``else`` clause raising ``ValueError``, this test must be
   updated in the same commit to assert ``pytest.raises(ValueError)``.

Running
-------

From the repository root, with the project installed in editable mode
(``pip install -e .``)::

    python -m pytest tests/unit/ -v

Filter to the waveforms suite::

    python -m pytest tests/unit/waveforms/ -v

Run a single test::

    python -m pytest tests/unit/waveforms/test_waveforms.py::TestGaussian::test_peaks_at_chi_with_unit_value -v

Stop on first failure (useful while iterating)::

    python -m pytest tests/unit/ -x
