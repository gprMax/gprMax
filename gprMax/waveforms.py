# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

import logging

import numpy as np

logger = logging.getLogger(__name__)


class Waveform:
    """Definitions of waveform shapes that can be used with sources."""

    types = [
        "gaussian",
        "gauspulse",
        "gaussiandot",
        "gaussiandotnorm",
        "gaussiandotdot",
        "gaussiandotdotnorm",
        "gaussianprime",
        "gaussiandoubleprime",
        "ricker",
        "sine",
        "contsine",
        "impulse",
        "user",
    ]

    # Information about specific waveforms:
    #
    # gaussianprime and gaussiandoubleprime waveforms are the first derivative
    # and second derivative of the 'base' gaussian waveform, i.e. the centre
    # frequencies of the waveforms will rise for the first and second derivatives.
    #
    # gaussiandot, gaussiandotnorm, gaussiandotdot, gaussiandotdotnorm,
    # ricker waveforms have their centre frequencies specified by the user,
    # i.e. they are not derived from the 'base' gaussian

    def __init__(self):
        self.ID = None
        self.type = None
        self.amp = 1
        self.freq = None
        self.userfunc = None
        self.chi = 0
        self.zeta = 0
        self.delay = 0

    def _require_finite_amplitude(self):
        """Reject non-finite amplitude scalings before they enter the solve."""
        try:
            finite = bool(np.isfinite(self.amp))
        except TypeError:
            finite = False
        if not finite:
            raise ValueError(f"Waveform {self.ID!r} requires a finite amplitude scaling.")

    def _require_positive_frequency(self):
        """Reject missing or non-physical excitation frequencies.

        Every built-in waveform — including ``impulse``, whose frequency is
        otherwise unused — requires a finite positive frequency. This matches
        the ``#waveform`` command, the Python API documentation, and the
        impulse-response toolbox, so a direct :class:`Waveform` use fails
        with the same clear error instead of a late ``ZeroDivisionError`` or
        silently unphysical samples.
        """
        try:
            finite = bool(np.isfinite(self.freq))
        except TypeError:
            finite = False
        if not finite or not self.freq > 0:
            raise ValueError(
                f"Waveform {self.ID!r} of type {self.type!r} requires "
                "a finite excitation frequency greater than zero."
            )

    def _require_valid_sample_time(self, time, dt):
        """Reject non-finite sample times and non-positive timesteps."""
        try:
            finite_time = bool(np.isfinite(time))
        except TypeError:
            finite_time = False
        if not finite_time:
            raise ValueError(f"Waveform {self.ID!r} requires a finite sample time.")
        try:
            finite_dt = bool(np.isfinite(dt))
        except TypeError:
            finite_dt = False
        if not finite_dt or not dt > 0:
            raise ValueError(f"Waveform {self.ID!r} requires a finite timestep greater than zero.")

    def calculate_coefficients(self):
        """Calculates coefficients (used to calculate values) for specific
        waveforms.
        """

        if self.type == "user":
            return
        if self.type not in self.types:
            raise ValueError(f"Unknown waveform type {self.type!r}")
        self._require_positive_frequency()

        if self.type in [
            "gaussian",
            "gaussiandot",
            "gaussiandotnorm",
            "gaussianprime",
            "gaussiandoubleprime",
        ]:
            self.chi = 1 / self.freq
            self.zeta = 2 * np.pi**2 * self.freq**2
        elif self.type in ["gaussiandotdot", "gaussiandotdotnorm", "ricker"]:
            self.chi = np.sqrt(2) / self.freq
            self.zeta = np.pi**2 * self.freq**2
        elif self.type == "gauspulse":
            # MATLAB gauspuls / scipy.signal.gausspulse defaults: fractional
            # bandwidth 0.5 at -6 dB. Shift the zero-centred cosine to the
            # -60 dB envelope cutoff so both halves fit at positive times.
            # Keep the analytic DPW evaluator in cython/plane_wave.pyx in sync.
            self.zeta = -(np.pi * self.freq * 0.5) ** 2 / (4 * np.log(10 ** (-6 / 20)))
            self.chi = np.sqrt(-np.log(10 ** (-60 / 20)) / self.zeta)

    def calculate_value(self, time, dt):
        """Calculates the value of the waveform at a specific time.

        Args:
            time: float for absolute time.
            dt: float for absolute time discretisation.

        Returns:
            ampvalue: float for calculated value for waveform.
        """

        if self.type not in self.types:
            raise ValueError(f"Unknown waveform type {self.type!r}")
        self._require_finite_amplitude()
        self._require_valid_sample_time(time, dt)
        if self.type == "user" and not callable(self.userfunc):
            raise ValueError(f"User waveform {self.ID!r} requires a callable 'userfunc'.")
        self.calculate_coefficients()

        # Waveforms
        if self.type == "gaussian":
            delay = time - self.chi
            ampvalue = np.exp(-self.zeta * delay**2)

        elif self.type == "gauspulse":
            delay = time - self.chi
            ampvalue = np.exp(-self.zeta * delay**2) * np.cos(2 * np.pi * self.freq * delay)

        elif self.type in ["gaussiandot", "gaussianprime"]:
            delay = time - self.chi
            ampvalue = -2 * self.zeta * delay * np.exp(-self.zeta * delay**2)

        elif self.type == "gaussiandotnorm":
            delay = time - self.chi
            normalise = np.sqrt(np.exp(1) / (2 * self.zeta))
            ampvalue = -2 * self.zeta * delay * np.exp(-self.zeta * delay**2) * normalise

        elif self.type in ["gaussiandotdot", "gaussiandoubleprime"]:
            delay = time - self.chi
            ampvalue = (
                2 * self.zeta * (2 * self.zeta * delay**2 - 1) * np.exp(-self.zeta * delay**2)
            )

        elif self.type == "gaussiandotdotnorm":
            delay = time - self.chi
            normalise = 1 / (2 * self.zeta)
            ampvalue = (
                2
                * self.zeta
                * (2 * self.zeta * delay**2 - 1)
                * np.exp(-self.zeta * delay**2)
                * normalise
            )

        elif self.type == "ricker":
            delay = time - self.chi
            normalise = 1 / (2 * self.zeta)
            ampvalue = -(
                (2 * self.zeta * (2 * self.zeta * delay**2 - 1) * np.exp(-self.zeta * delay**2))
                * normalise
            )

        elif self.type == "sine":
            ampvalue = np.sin(2 * np.pi * self.freq * time)
            if time * self.freq > 1:
                ampvalue = 0

        elif self.type == "contsine":
            rampamp = 0.25
            ramp = rampamp * time * self.freq
            ramp = min(ramp, 1)

            ampvalue = ramp * np.sin(2 * np.pi * self.freq * time)

        elif self.type == "impulse":
            # time < dt condition required to do impulsive magnetic dipole
            if time == 0 or time < dt:
                ampvalue = 1
            elif time >= dt:
                ampvalue = 0

        elif self.type == "user":
            try:
                ampvalue = float(self.userfunc(time))
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f"User waveform {self.ID!r} 'userfunc' must accept a single float "
                    f"(time in seconds) and return a numeric value (failed at time {time:g} s)."
                ) from err
            if not np.isfinite(ampvalue):
                raise ValueError(f"User waveform {self.ID!r} returned a non-finite value at time {time:g} s")

        else:
            raise ValueError(f"Unknown waveform type {self.type!r}")

        ampvalue *= self.amp

        return ampvalue
