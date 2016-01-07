
.. _plotting:

********
Plotting
********

A-scans
=======

plot_Ascan.py
-------------

This module uses matplotlib to plot the time history for the electric and magnetic field components for any receivers in a model (each receiver gets a separate figure window). Usage (from the top-level gprMax directory) is:

.. code-block:: none

    python -m tools.plot_Ascan outputfile

where ``outputfile`` is the name of output file including the path.

There are optional command line arguments:

* ``--fields`` to specify a subset of the default field components (``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy`` or ``Hz``) as a list to plot. By default all field components are plotted.
* ``-fft`` to plot the Fast Fourier Transform (FFT) of a single field component

For example to plot the ``Ez`` field component with it's FFT:

.. code-block:: none

    python -m tools.plot_Ascan my_outputfile.out --fields Ez -fft

B-scans
=======

plot_Bscan.py
-------------

gprMax produces a separate output file for each trace (A-scan) in the B-scan. These must be combined into a single file using the ``outputfiles_merge.py`` module (described in the :ref:`other utilities section <utils>`). This module uses matplotlib to plot an image of the B-scan. Usage (from the top-level gprMax directory) is:

.. code-block:: none

    python -m tools.plot_Bscan outputfile --field fieldcomponent

where:

* ``outputfile`` is the name of output file including the path
* ``--field`` is the name of field component to plot, e.g. ``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy`` or ``Hz``


.. _waveforms:

Built-in waveforms
==================

This section describes the definitions of the functions that are used to create the built-in waveforms, and how to plot them.

plot_builtin_wave.py
--------------------

This module uses matplotlib to plot one of the built-in waveforms and it's FFT. Usage (from the top-level gprMax directory) is:

.. code-block:: none

    python -m tools.plot_builtin_wave type amp freq timewindow dt

where:

* ``type`` is the type of waveform, e.g. gaussian, ricker etc...
* ``amp`` is the amplitude of the waveform
* ``freq`` is the centre frequency of the waveform
* ``timewindow`` is the time window to view the waveform, i.e. the time window of the proposed simulation
* ``dt`` is the time step to view waveform, i.e. the time step of the proposed simulation

There is an optional command line argument:

* ``-fft`` to plot the Fast Fourier Transform (FFT) of the waveform

Example plots of all the built-in waveforms are shown using the parameters: amplitude of one, frequency of 1GHz, time window of 6ns, and a time step of 1.926ps.

gaussian
^^^^^^^^

A Gaussian waveform.

.. math:: I = e^{-\zeta(t-\chi)^2}

where :math:`I` is the current, :math:`\zeta = 2\pi^2f^2`, :math:`\chi=\frac{1}{f}` and :math:`f` is the frequency.

.. figure:: images/gaussian.png

    Example of the ``gaussian`` waveform - time domain and power spectrum.


gaussiandot
^^^^^^^^^^^

First derivative of a Gaussian waveform.

.. math:: I = -2 \zeta (t-\chi) e^{-\zeta(t-\chi)^2}

where :math:`I` is the current, :math:`\zeta = 2\pi^2f^2`, :math:`\chi=\frac{1}{f}` and :math:`f` is the frequency.

.. figure:: images/gaussiandot.png

    Example of the ``gaussiandot`` waveform - time domain and power spectrum.


gaussiandotnorm
^^^^^^^^^^^^^^^

Normalised first derivative of a Gaussian waveform.

.. math:: I = -2 \sqrt{\frac{e}{2\zeta}} \zeta (t-\chi) e^{-\zeta(t-\chi)^2}

where :math:`I` is the current, :math:`\zeta = 2\pi^2f^2`, :math:`\chi=\frac{1}{f}` and :math:`f` is the frequency.

.. figure:: images/gaussiandotnorm.png

    Example of the ``gaussiandotnorm`` waveform - time domain and power spectrum.


gaussiandotdot
^^^^^^^^^^^^^^

Second derivative of a Gaussian waveform.

.. math:: I = 2\zeta \left(2\zeta(t-\chi)^2 - 1 \right) e^{-\zeta(t-\chi)^2}

where :math:`I` is the current, :math:`\zeta = \pi^2f^2`, :math:`\chi=\frac{\sqrt{2}}{f}` and :math:`f` is the frequency.

.. figure:: images/gaussiandotdot.png

    Example of the ``gaussiandotdot`` waveform - time domain and power spectrum.


gaussiandotdotnorm
^^^^^^^^^^^^^^^^^^

Normalised second derivative of a Gaussian waveform.

.. math:: I = \left( 2\zeta (t-\chi)^2 - 1 \right) e^{-\zeta(t-\chi)^2}

where :math:`I` is the current, :math:`\zeta = \pi^2f^2`, :math:`\chi=\frac{\sqrt{2}}{f}` and :math:`f` is the frequency.

.. figure:: images/gaussiandotdotnorm.png

    Example of the ``gaussiandotdotnorm`` waveform - time domain and power spectrum.


ricker
^^^^^^

A Ricker (or Mexican Hat) waveform which is the negative, normalised second derivative of a Gaussian waveform.

.. math:: I = - \left( 2\zeta (t-\chi)^2 -1 \right) e^{-\zeta(t-\chi)^2}

where :math:`I` is the current, :math:`\zeta = \pi^2f^2`, :math:`\chi=\frac{\sqrt{2}}{f}` and :math:`f` is the frequency.

.. figure:: images/ricker.png

    Example of the ``ricker`` waveform - time domain and power spectrum.


sine
^^^^

A single cycle of a sine waveform.

.. math:: I = R\sin(2\pi ft)

and

.. math::

    R =
    \begin{cases}
    1 &\text{if $ft\leq1$}, \\
    0 &\text{if $ft>1$}.
    \end{cases}

:math:`I` is the current, :math:`t` is time and :math:`f` is the frequency.

.. figure:: images/sine.png

    Example of the ``sine`` waveform - time domain and power spectrum.


contsine
^^^^^^^^

A continuous sine waveform. In order to avoid introducing noise into the calculation the amplitude of the waveform is modulated for the first cycle of the sine wave (ramp excitation).

.. math:: I = R\sin(2\pi ft)

and

.. math::

    R =
    \begin{cases}
    R_cft &\text{if $R\leq 1$}, \\
    1 &\text{if $R>1$}.
    \end{cases}

where :math:`I` is the current, :math:`R_c` is set to :math:`0.25`, :math:`t` is time and :math:`f` is the frequency.

.. figure:: images/contsine.png

    Example of the ``contsine`` waveform - time domain and power spectrum.


