Toolboxes is a sub-package where useful Python modules contributed by users are stored.

********
Plotting
********

Information
===========

This package is intended to provide some basic scripts to get started with plotting outputs from simulations.

Package contents
================

plot_Ascan.py
-------------

This module uses matplotlib to plot the time history for the electric and magnetic field components, and currents for all receivers in a model (each receiver gets a separate figure window). Usage (from the top-level gprMax directory) is:

.. code-block:: none

    python -m gprMax.toolboxes.Plotting.plot_Ascan outputfile

where ``outputfile`` is the name of output file including the path.

There are optional command line arguments:

* ``--outputs`` to specify a subset of the default output components (``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy``, ``Hz``, ``Ix``, ``Iy`` or ``Iz``) to plot. By default all electric and magnetic field components are plotted.
* ``-fft`` to plot the Fast Fourier Transform (FFT) of a single output component

For example to plot the ``Ez`` output component with it's FFT:

.. code-block:: none

    python -m gprMax.toolboxes.Plotting.plot_Ascan my_outputfile.h5 --outputs Ez -fft

Any supported component subset may include currents, for example
``--outputs Ez Ix Iz-``. Each curve uses its dataset's ``SampleInterval`` and
``TimeSampleOffset``. H-field and receiver loop-current traces therefore
normally start at ``-dt/2``, not zero; electric traces normally start at zero.


plot_Bscan.py
-------------

gprMax produces a separate output file for each trace (A-scan) in the
B-scan. These must first be combined with the
``gprMax.toolboxes.Utilities.outputfiles_merge`` module. ``plot_Bscan.py`` then uses
matplotlib to plot the resulting real, time-domain receiver-data matrix.
Usage from the top-level gprMax directory is:

.. code-block:: none

    python -m gprMax.toolboxes.Plotting.plot_Bscan outputfile rx-component

where:

* ``outputfile`` is the name of output file including the path
* ``rx-component`` is the name of the receiver output component (``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy``, ``Hz``, ``Ix``, ``Iy`` or ``Iz``) to plot

Merged antenna terminal voltages can be plotted without adding a point
receiver. Select ``Vtotal`` and identify the merged port or feed group:

.. code-block:: none

    python -m gprMax.toolboxes.Plotting.plot_Bscan antenna_merged.h5 Vtotal \
        --trace-group ports/receive

Voltage-source and rational-network ports use ``ports/<ID>``. Transmission
lines and magnetic frills use paths such as ``tls/tl1`` and
``frills/frill1``. S-parameters, impedance, and spectra are intentionally not
B-scan quantities and cannot be selected here.

Image pixel centres are placed at the physical sample times. Gathering
receivers rejects mismatched time axes. For Python callers,
``get_output_data(..., return_time_offset=True)`` and
``gather_receiver_outputs(..., return_time_offset=True)`` return
``(samples, dt, offset)``; pass the offset as ``time_offset=offset`` to
``plot_Bscan.mpl_plot``. The default two-value loader return is unchanged.
Direct matrix plotting without an explicit offset uses the receiver-component
convention; terminal voltages should always use the offset from the loader.

plot_port.py
------------

This module is the terminal-output counterpart to ``plot_Ascan.py``. An A-scan
plots local field or Ampere-loop samples from ``/rxs``; this module plots the
authoritative source-terminal quantities already calculated by gprMax. It
supports voltage-source ports, rational-network ports, transmission lines,
magnetic-frill ports, and ports inside subgrids. It does not repeat the S11 or
input-impedance calculation during plotting.

Run it with:

.. code-block:: none

    python -m gprMax.toolboxes.Plotting.plot_port outputfile --save

All stored ports are plotted by default. Use a repeatable ``--port`` option to
select one or more port IDs or complete HDF5 paths:

.. code-block:: none

    python -m gprMax.toolboxes.Plotting.plot_port outputfile \
        --port feed --port subgrids/fine_grid/ports/feed2 --save

Every port receives uniquely named parameter and signal figures, so plots from
the same model cannot overwrite one another. ``--output-dir`` selects their
directory; ``--format`` accepts ``png``, ``pdf``, or ``svg``; and ``--dpi``
controls raster resolution. Without ``--save`` the figures are displayed
interactively.

The parameter figure plots stored complex S11, input impedance, and input
admittance using their respective validity masks. Transmission-line
current-deembedding checks and voltage-source uncorrected source-plane values
are included when present. Invalid finite research values are hidden by
default; ``--show-invalid`` displays them as grey dotted lines. ``--validity``
adds a figure showing the stored source-band, mesh, gap-correction, and
line-propagation masks. It also plots the incident spectrum relative to its
peak and the cells per minimum wavelength when those diagnostics are stored,
together with the thresholds used to construct the masks.

The signal figure adapts to the available schema. It can include generator,
incident, reflected, and total voltage; incident, terminal, Ampere-loop, and
network currents; and their stored spectra. Missing quantities are not
estimated. ``--parameters-only`` suppresses this figure. ``--fmin`` and
``--fmax`` limit displayed frequencies in hertz, while ``--tmin`` and
``--tmax`` limit histories in seconds. Axes are automatically presented in
suitable engineering units.

``--list-ports`` prints all discoverable paths. The internal plotting data
model stores a collection of named S-parameter traces. It currently contains
S11 because local gprMax terminal outputs are one-port results; it can accept
additional Sij traces in future. Complete multiport matrices already exist in
``PortStudy`` and ``EigenmodeStudy`` aggregate files; use the study result
readers and examples in :doc:`studies` rather than treating a local S11 trace
as a matrix column.

The former ``plot_antenna_params`` entry point remains as a compatibility
alias for the new plotter. Its legacy reconstruction of S11, Zin, and a
voltage/field transfer ratio labelled S21 has been retired. That calculation
did not include the present source, mesh, gap, and discrete-line validity
corrections, and a field-to-source ratio is not a power-wave S-parameter.


.. _waveforms:

plot_source_wave.py
--------------------

This module uses Matplotlib to plot built-in waveforms in the time domain
and, optionally, their power spectra. Run it from the repository root:

.. code-block:: none

    python -m gprMax.toolboxes.Plotting.plot_source_wave type amp freq timewindow dt

where:

* ``type`` is a built-in waveform name, such as ``gaussian`` or ``ricker``.
* ``amp`` is the amplitude of the waveform
* ``freq`` is the centre frequency of the waveform (Hertz). In the case of the Gaussian waveform it is related to the pulse width.
* ``timewindow`` is the time window (seconds) to view the waveform, i.e. the time window of the proposed simulation
* ``dt`` is the time step (seconds) to view waveform, i.e. the time step of the proposed simulation

Optional command-line arguments are:

* ``-fft`` to include the waveform's power spectrum, calculated using the FFT.
* ``-save`` to save a PNG without opening a plot window.


Definitions of the built-in waveforms and example plots are shown using the parameters: amplitude of one, centre frequency of 1GHz, time window of 6ns, and a time step of 1.926ps.

The :ref:`modulated Gaussian example <waveform-modulated-gaussian>` below
also shows how to choose a custom bandwidth with a user-defined waveform.

gaussian
^^^^^^^^

A Gaussian waveform.

.. math:: W(t) = e^{-\zeta(t-\chi)^2}

where :math:`\zeta = 2\pi^2f^2`, :math:`\chi=\frac{1}{f}` and :math:`f` is the frequency.

.. figure:: ../../images_shared/gaussian.png

    Example of the ``gaussian`` waveform - time domain and power spectrum.

.. _waveform-modulated-gaussian:

gauspulse (modulated Gaussian)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A cosine carrier is multiplied by a Gaussian envelope. The built-in
``gauspulse`` uses the default bandwidth of MATLAB's
`gauspuls <https://www.mathworks.com/help/signal/ref/gauspuls.html>`_:
fractional bandwidth :math:`b=0.5` at :math:`r=-6` dB. Only amplitude and
carrier frequency are supplied in the waveform command.

.. math::

    W(t)=A e^{-a(t-t_0)^2}\cos\left[2\pi f_0(t-t_0)\right],
    \qquad a=-\frac{(\pi f_0 b)^2}{4\ln(10^{r/20})},
    \qquad t_0=\sqrt{-\frac{\ln(10^{-60/20})}{a}}.

MATLAB's pulse is centred at zero. gprMax shifts it by :math:`t_0`, the
default -60 dB envelope cutoff, so the leading half is present after the
source starts. At 1 GHz the peak is at approximately 2.781 ns and the
-6 dB band extends from 0.75 to 1.25 GHz. The envelope at time zero is
:math:`10^{-3}` of its peak, **not exactly zero**. The analytic trailing
tail is not truncated automatically; the source's start/stop window still
applies. A source start time adds to this intrinsic delay. Use a time window
of at least :math:`2t_0` to include the pulse down to -60 dB on both sides,
plus enough time to record the model's response.

.. figure:: ../../images_shared/modulated_gaussian.png
    :alt: Gaussian-modulated cosine at 1 GHz with its envelope and normalised power spectrum.

    Built-in ``gauspulse``: time trace with its envelope and power spectrum
    normalised to 0 dB at the peak. Zero padding smooths the plotted spectrum
    but does not increase its physical frequency resolution.

Use the existing waveform hash command:

.. code-block:: none

    #waveform: gauspulse 1 1e9 pulse

The equivalent Python API definition is:

.. code-block:: python

    scene.add(gprMax.Waveform(wave_type='gauspulse', amp=1, freq=1e9, id='pulse'))

Assign ``pulse`` to a source as usual. A :download:`complete hash-command
model <../../examples/features/waveforms/gauspulse.in>` uses a resistive
voltage source and records its automatic port and a nearby receiver.
To plot the waveform with the standard plotting command:

.. code-block:: console

    python -m gprMax.toolboxes.Plotting.plot_source_wave gauspulse 1 1e9 6e-9 1.926e-12 -fft -save

The :download:`gallery plotting script
<../../examples/features/waveforms/modulated_gaussian.py>` adds the envelope
and bandwidth markers shown above. It also exports a two-column sample
table, if needed for comparison with another tool:

.. code-block:: console

    python -m examples.features.waveforms.modulated_gaussian --output-dir waveform_preview

For a different bandwidth, reference level, phase or delay, use the existing
user-defined interface instead of extending the hash-command syntax. For
example, a 70%-bandwidth pulse with the same -6 dB reference level:

.. code-block:: python

    from scipy.signal import gausspulse

    delay = gausspulse('cutoff', fc=1e9, bw=0.7, bwr=-6, tpr=-60)

    def custom_pulse(time):
        return float(gausspulse(time - delay, fc=1e9, bw=0.7, bwr=-6))

    scene.add(gprMax.Waveform(wave_type='user', user_func=custom_pulse, id='custom'))

SciPy spells this function ``gausspulse``; MATLAB spells it ``gauspuls``;
the gprMax built-in type is ``gauspulse``. A callable returns the complete
amplitude. For custom samples in a hash-command model, use
``#excitation_file``. User-defined waveforms cannot drive discrete plane
waves, but the built-in ``gauspulse`` can.


gaussiandot
^^^^^^^^^^^

First derivative of a Gaussian waveform.

.. math:: W(t) = -2 \zeta (t-\chi) e^{-\zeta(t-\chi)^2}

where :math:`\zeta = 2\pi^2f^2`, :math:`\chi=\frac{1}{f}` and :math:`f` is the frequency.

.. figure:: ../../images_shared/gaussiandot.png

    Example of the ``gaussiandot`` waveform - time domain and power spectrum.


gaussiandotnorm
^^^^^^^^^^^^^^^

Normalised first derivative of a Gaussian waveform.

.. math:: W(t) = -2 \sqrt{\frac{e}{2\zeta}} \zeta (t-\chi) e^{-\zeta(t-\chi)^2}

where :math:`\zeta = 2\pi^2f^2`, :math:`\chi=\frac{1}{f}` and :math:`f` is the frequency.

.. figure:: ../../images_shared/gaussiandotnorm.png

    Example of the ``gaussiandotnorm`` waveform - time domain and power spectrum.


gaussiandotdot
^^^^^^^^^^^^^^

Second derivative of a Gaussian waveform.

.. math:: W(t) = 2\zeta \left(2\zeta(t-\chi)^2 - 1 \right) e^{-\zeta(t-\chi)^2}

where :math:`\zeta = \pi^2f^2`, :math:`\chi=\frac{\sqrt{2}}{f}` and :math:`f` is the frequency.

.. figure:: ../../images_shared/gaussiandotdot.png

    Example of the ``gaussiandotdot`` waveform - time domain and power spectrum.


gaussiandotdotnorm
^^^^^^^^^^^^^^^^^^

Normalised second derivative of a Gaussian waveform.

.. math:: W(t) = \left( 2\zeta (t-\chi)^2 - 1 \right) e^{-\zeta(t-\chi)^2}

where :math:`\zeta = \pi^2f^2`, :math:`\chi=\frac{\sqrt{2}}{f}` and :math:`f` is the frequency.

.. figure:: ../../images_shared/gaussiandotdotnorm.png

    Example of the ``gaussiandotdotnorm`` waveform - time domain and power spectrum.


ricker
^^^^^^

A Ricker (or Mexican Hat) waveform which is the negative, normalised second derivative of a Gaussian waveform.

.. math:: W(t) = - \left( 2\zeta (t-\chi)^2 -1 \right) e^{-\zeta(t-\chi)^2}

where :math:`\zeta = \pi^2f^2`, :math:`\chi=\frac{\sqrt{2}}{f}` and :math:`f` is the frequency.

.. figure:: ../../images_shared/ricker.png

    Example of the ``ricker`` waveform - time domain and power spectrum.


sine
^^^^

A single cycle of a sine waveform.

.. math:: W(t) = R\sin(2\pi ft)

and

.. math::

    R =
    \begin{cases}
    1 &\text{if $ft\leq1$}, \\
    0 &\text{if $ft>1$}.
    \end{cases}

:math:`f` is the frequency

.. figure:: ../../images_shared/sine.png

    Example of the ``sine`` waveform - time domain and power spectrum.


contsine
^^^^^^^^

A continuous sine waveform with a linear amplitude ramp over the first four
cycles. The ramp reaches one at :math:`t=4/f`, not at the end of the first cycle.

.. math:: W(t) = R\sin(2\pi ft)

and

.. math::

    R =
    \begin{cases}
    R_cft &\text{if $R_cft\leq 1$}, \\
    1 &\text{if $R_cft>1$}.
    \end{cases}

where :math:`R_c` is set to :math:`0.25` and :math:`f` is the frequency.

.. figure:: ../../images_shared/contsine.png

    Example of the ``contsine`` waveform - time domain and power spectrum.


impulse
^^^^^^^

A unit-amplitude discrete impulse, not a unit-area continuous-time Dirac
delta. Within the source's active time window it is evaluated as:

.. math::

    W(t) =
    \begin{cases}
    1 &\text{if $0\leq t<\Delta t$}, \\
    0 &\text{if $t\geq\Delta t$}.
    \end{cases}

Here :math:`t` is time relative to the source start and :math:`\Delta t` is
the model's time step. This gives one nonzero sample on either the whole-
or half-step source lattice. A hard voltage source still clamps its edge
to zero after that sample until its stop time; see :ref:`voltage_source`.

.. figure:: ../../images_shared/impulse.png
    :width: 350 px

    Example of the ``impulse`` waveform - time domain.

.. note::
    * The impulse waveform should be used with care!
    * The impulse response of a model, i.e. when the source in the model is excited using the impulse waveform, is not likely to be useful when viewed in isolation.
    * However, the impulse response of a model can be convolved with different inputs (waveforms) to provide valid outputs without having to run a separate model for each different input (waveform).
    * The impulse response of the model can only be legitimately convolved with inputs (waveforms) that respect the limits of numerical dispersion in the original model, i.e. if a waveform contains frequencies that will not propagate correctly (due to numerical dispersion) in the original model, then the convolution of the waveform with the impulse response will not be valid.
