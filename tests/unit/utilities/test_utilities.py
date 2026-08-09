"""Rounding, sorting and the small helpers everything else is built on.

Nothing in this file is more than a dozen lines of source, and that is
precisely the risk. These functions are called from geometry construction,
from grid discretisation and from the source and receiver placement code —
places where the caller assumes "round" means what Python's ``round`` means.
It does not.

Two deliberate departures from the standard library run through this suite:

* ``round_int`` uses ``ROUND_HALF_DOWN``: a tie goes toward **zero**, so
  ``0.5`` is ``0`` and ``-0.5`` is ``0``. Python's built-in ``round`` uses
  banker's rounding (ties to even), giving ``round(0.5) == 0`` but
  ``round(1.5) == 2`` — the two agree on the first and disagree on the second.
* ``round_float`` uses ``ROUND_FLOOR``: it truncates toward **negative
  infinity**, not toward zero, so ``-1.2345`` at two places is ``-1.24`` and
  not ``-1.23``.

Both choices are correct for their purpose — a cell index must not be rounded
up past the domain edge — but neither is discoverable from the call site. The
tables below exist so that a change to either is a failing test rather than a
geometry that is one cell out.

``get_terminal_width`` is included because two other modules build padded
banner strings from it, and its zero-width fallback is the only branch that
matters on a CI runner with no tty.
"""

import numpy as np
import pytest

from gprMax.utilities.utilities import (
    atoi,
    fft_power,
    get_terminal_width,
    logo,
    natural_keys,
    round32,
    round_float,
    round_int,
    round_value,
    timer,
)


class TestTerminalWidth:
    """``get_terminal_width`` — the width every banner is padded to."""

    def test_returns_the_reported_width(self, monkeypatch):
        monkeypatch.setattr(
            "gprMax.utilities.utilities.get_terminal_size", lambda: (137, 40)
        )

        assert get_terminal_width() == 137

    def test_a_zero_width_falls_back_to_one_hundred(self, monkeypatch):
        """A pipe or a CI log can report zero columns.

        Without the fallback the padding expressions elsewhere
        (``'-' * (get_terminal_width() - 1 - len(s))``) would go negative and
        silently produce no separator at all.
        """
        monkeypatch.setattr(
            "gprMax.utilities.utilities.get_terminal_size", lambda: (0, 0)
        )

        assert get_terminal_width() == 100

    def test_only_the_first_element_is_used(self, monkeypatch):
        """Columns, not lines — the two are easy to transpose."""
        monkeypatch.setattr(
            "gprMax.utilities.utilities.get_terminal_size", lambda: (80, 24)
        )

        assert get_terminal_width() == 80

    def test_the_real_call_returns_a_positive_integer(self):
        """Unpatched, on whatever the runner is."""
        width = get_terminal_width()

        assert isinstance(width, int) and width > 0


class TestAtoi:
    """``atoi`` — convert if it looks like a number, otherwise pass through."""

    def test_digits_become_an_integer(self):
        assert atoi("42") == 42

    def test_non_digits_are_returned_unchanged(self):
        assert atoi("model") == "model"

    def test_an_empty_string_is_returned_unchanged(self):
        """``"".isdigit()`` is ``False``, so this must not raise."""
        assert atoi("") == ""

    def test_leading_zeros_are_dropped(self):
        assert atoi("007") == 7

    def test_a_negative_sign_prevents_conversion(self):
        """``isdigit`` rejects the minus sign, so ``"-1"`` stays a string.

        Harmless for its actual use — sorting file names — but it means
        ``atoi`` is not a general string-to-int conversion.
        """
        assert atoi("-1") == "-1"

    def test_a_decimal_point_prevents_conversion(self):
        assert atoi("1.5") == "1.5"


class TestNaturalKeys:
    """``natural_keys`` — the sort order humans expect for numbered files."""

    def test_digits_and_text_are_split_apart(self):
        assert natural_keys("model12") == ["model", 12, ""]

    def test_a_string_with_no_digits_is_a_single_element(self):
        assert natural_keys("model") == ["model"]

    def test_it_sorts_numerically_not_lexically(self):
        """The whole point: ``model10`` must come after ``model9``."""
        names = ["model10", "model9", "model1"]

        assert sorted(names, key=natural_keys) == ["model1", "model9", "model10"]

    def test_lexical_sorting_would_get_this_wrong(self):
        """Stated explicitly so the value of the helper is visible."""
        names = ["model10", "model9"]

        assert sorted(names) != sorted(names, key=natural_keys)

    def test_multiple_number_groups_are_all_converted(self):
        assert natural_keys("snap_2_of_10.vti") == [
            "snap_",
            2,
            "_of_",
            10,
            ".vti",
        ]

    def test_it_orders_snapshot_files_correctly(self):
        names = [f"model_snaps/snapshot{n}.vti" for n in (1, 2, 10, 20, 3)]

        assert [
            n for n in sorted(names, key=natural_keys)
        ] == [f"model_snaps/snapshot{n}.vti" for n in (1, 2, 3, 10, 20)]


class TestRoundInt:
    """``round_int`` — ties toward zero, not toward even."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (0.0, 0),
            (0.4, 0),
            (0.6, 1),
            (1.4, 1),
            (1.6, 2),
            (2.4, 2),
            (-0.4, 0),
            (-0.6, -1),
            (-1.4, -1),
            (-1.6, -2),
            (10.0, 10),
        ],
    )
    def test_non_tie_values_round_to_nearest(self, value, expected):
        assert round_int(value) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            (0.5, 0),
            (1.5, 1),
            (2.5, 2),
            (3.5, 3),
            (-0.5, 0),
            (-1.5, -1),
            (-2.5, -2),
        ],
    )
    def test_ties_go_toward_zero(self, value, expected):
        """``ROUND_HALF_DOWN`` — magnitude never increases on a tie.

        This is the property the grid code depends on: a coordinate exactly
        half a cell past the last node must not round to a node that does not
        exist.
        """
        assert round_int(value) == expected

    @pytest.mark.parametrize("value", [1.5, 3.5, -1.5, -3.5])
    def test_it_disagrees_with_the_builtin_on_odd_ties(self, value):
        """Stated as a test so the difference is not folded into a comment.

        The two rules coincide on *even* ties — banker's rounding sends
        ``2.5`` to ``2``, and so does half-down — which is why the difference
        is easy to miss in casual testing. It shows up only on odd ties.
        """
        assert round_int(value) != round(value)

    @pytest.mark.parametrize("value", [0.5, 2.5, -0.5, -2.5])
    def test_it_agrees_with_the_builtin_on_even_ties(self, value):
        assert round_int(value) == round(value)

    def test_it_returns_a_python_int(self):
        result = round_int(2.7)

        assert isinstance(result, int) and not isinstance(result, bool)

    def test_an_integer_argument_is_accepted(self):
        assert round_int(7) == 7

    def test_a_numpy_float_is_rejected(self):
        """``decimal.Decimal`` accepts ``float`` and ``int``, but not ``np.float32``.

        Callers holding a value from an array must convert first. Pinned
        because the failure is a bare ``TypeError`` from inside ``decimal``
        with no mention of gprMax, and is written up in
        ``notes/bugs/utilities-rounding-rejects-numpy-scalars.md``.
        """
        with pytest.raises(TypeError):
            round_int(np.float32(2.5))

    def test_a_numpy_float64_is_accepted(self):
        """``np.float64`` subclasses ``float``, so it slips through.

        The inconsistency with ``np.float32`` is the reason the restriction is
        worth writing down.
        """
        assert round_int(np.float64(2.5)) == 2


class TestRoundFloat:
    """``round_float`` — truncation toward negative infinity."""

    @pytest.mark.parametrize(
        "value, places, expected",
        [
            (1.2345, 2, 1.23),
            (1.2399, 2, 1.23),
            (1.0, 2, 1.0),
            (1.2345, 3, 1.234),
            (1.2345, 1, 1.2),
            (0.999, 2, 0.99),
        ],
    )
    def test_positive_values_truncate_downward(self, value, places, expected):
        assert round_float(value, places) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "value, places, expected",
        [
            (-1.2345, 2, -1.24),
            (-1.2301, 2, -1.24),
            (-0.001, 2, -0.01),
            (-1.0, 2, -1.0),
        ],
    )
    def test_negative_values_also_go_downward(self, value, places, expected):
        """Toward −∞, so a negative number becomes *more* negative.

        ``ROUND_FLOOR`` is not ``ROUND_DOWN``; the latter truncates toward
        zero and would give ``-1.23`` for the first case.
        """
        assert round_float(value, places) == pytest.approx(expected)

    def test_it_is_not_symmetric_about_zero(self):
        """The consequence of ``ROUND_FLOOR``, stated once and plainly."""
        assert round_float(1.235, 2) != -round_float(-1.235, 2)

    def test_zero_places_gives_an_integral_value(self):
        """``"1."`` with no zeros is still a valid quantisation target."""
        assert round_float(1.9, 0) == pytest.approx(1.0)

    def test_it_returns_a_python_float(self):
        assert isinstance(round_float(1.5, 2), float)

    def test_many_places_leave_the_value_alone(self):
        assert round_float(0.1, 15) == pytest.approx(0.1)

    def test_a_negative_place_count_is_silently_treated_as_zero(self):
        """``'0' * -1`` is the empty string, so the target becomes ``"1."``.

        ``Decimal("1.")`` is a valid integral quantiser, so a negative place
        count neither raises nor multiplies by a power of ten — it rounds to
        a whole number. Surprising, but harmless, and pinned so the behaviour
        is documented somewhere.
        """
        assert round_float(1.9, -1) == pytest.approx(1.0)


class TestRoundValue:
    """``round_value`` — the dispatcher the rest of the package calls."""

    def test_zero_places_uses_the_integer_rounding(self):
        assert round_value(2.5) == round_int(2.5)

    def test_zero_places_is_the_default(self):
        assert round_value(2.5) == 2

    def test_a_nonzero_place_count_uses_the_float_rounding(self):
        assert round_value(1.2345, 2) == round_float(1.2345, 2)

    def test_zero_places_returns_an_integer(self):
        assert isinstance(round_value(2.5), int)

    def test_nonzero_places_returns_a_float(self):
        assert isinstance(round_value(2.5, 1), float)

    def test_the_return_type_depends_on_the_place_count(self):
        """One argument changes the *type* of the result, not just its value.

        Worth an explicit test: a caller passing ``decimalplaces`` through
        from configuration gets an ``int`` or a ``float`` depending on data.
        """
        assert type(round_value(3.0, 0)) is not type(round_value(3.0, 1))


class TestRound32:
    """``round32`` — round *up* to a multiple of 32, for kernel block sizes."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (0, 0),
            (1, 32),
            (31, 32),
            (32, 32),
            (33, 64),
            (64, 64),
            (65, 96),
            (1000, 1024),
        ],
    )
    def test_it_rounds_up_to_the_next_multiple(self, value, expected):
        assert round32(value) == expected

    def test_an_exact_multiple_is_unchanged(self):
        """Otherwise every launch would over-allocate by a whole block."""
        assert round32(256) == 256

    def test_the_result_is_always_a_multiple_of_thirty_two(self):
        assert all(round32(n) % 32 == 0 for n in range(0, 200))

    def test_the_result_is_never_smaller_than_the_input(self):
        assert all(round32(n) >= n for n in range(0, 200))

    def test_it_returns_a_python_int(self):
        """``np.ceil`` returns a float; the cast matters for array shapes."""
        assert isinstance(round32(33), int)

    def test_a_float_input_is_accepted(self):
        assert round32(32.5) == 64

    def test_a_string_input_is_accepted(self):
        """``float(value)`` first, so a value straight from a hash command works."""
        assert round32("33") == 64


class TestFftPower:
    """``fft_power`` — the spectrum used for numerical-dispersion analysis."""

    def test_the_maximum_power_is_zero_decibels(self):
        """The whole array is shifted so the peak sits at 0 dB."""
        waveform = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 256))

        _, power = fft_power(waveform, 1 / 256)

        assert np.amax(power) == pytest.approx(0.0)

    def test_no_power_exceeds_the_peak(self):
        waveform = np.random.default_rng(0).normal(size=128)

        _, power = fft_power(waveform, 1e-9)

        assert np.all(power <= 0.0)

    def test_the_frequency_bins_match_the_waveform_length(self):
        waveform = np.ones(64)

        freqs, power = fft_power(waveform, 1e-9)

        assert freqs.size == power.size == 64

    def test_the_first_bin_is_direct_current(self):
        freqs, _ = fft_power(np.ones(32), 1e-9)

        assert freqs[0] == 0.0

    def test_the_bin_spacing_is_the_reciprocal_of_the_record_length(self):
        dt = 1e-9
        n = 64

        freqs, _ = fft_power(np.ones(n), dt)

        assert freqs[1] == pytest.approx(1 / (n * dt))

    def test_the_peak_lands_on_the_signal_frequency(self):
        """A pure tone must show up in the bin it belongs to."""
        n, dt, frequency = 256, 1e-9, 50e6
        t = np.arange(n) * dt
        waveform = np.sin(2 * np.pi * frequency * t)

        freqs, power = fft_power(waveform, dt)

        assert abs(freqs[np.argmax(power)]) == pytest.approx(frequency, rel=0.02)

    def test_an_all_zero_waveform_does_not_produce_infinities(self):
        """``log10(0)`` is ``-inf``; the function replaces non-finite values.

        A zeroed receiver trace is entirely normal — a source that has not
        fired yet — so this path is reached in ordinary use.
        """
        _, power = fft_power(np.zeros(32), 1e-9)

        assert np.all(np.isfinite(power))

    def test_a_waveform_with_a_zero_bin_does_not_produce_infinities(self):
        """A real signal can still have an exactly zero frequency component."""
        waveform = np.array([1.0, -1.0] * 16)

        _, power = fft_power(waveform, 1e-9)

        assert np.all(np.isfinite(power))

    def test_scaling_the_waveform_does_not_change_the_spectrum(self):
        """Power is relative to its own maximum, so amplitude cancels out."""
        waveform = np.random.default_rng(1).normal(size=64)

        _, power = fft_power(waveform, 1e-9)
        _, scaled = fft_power(waveform * 1000, 1e-9)

        assert power == pytest.approx(scaled)


class TestTimer:
    """``timer`` — a monotonic clock, wrapped for a single import point."""

    def test_it_returns_a_float(self):
        assert isinstance(timer(), float)

    def test_it_does_not_go_backwards(self):
        """``perf_counter``, not ``time`` — immune to the system clock moving.

        Solve times are differences of two of these, so a wall-clock jump
        during a long run would otherwise produce a negative duration.
        """
        first = timer()
        second = timer()

        assert second >= first

    def test_it_is_the_performance_counter(self):
        """Pinned by behaviour: two calls sit within one counter tick."""
        from time import perf_counter

        assert abs(timer() - perf_counter()) < 1.0


class TestLogo:
    """``logo`` — the banner printed once at startup."""

    @pytest.fixture(autouse=True)
    def fixed_width(self, monkeypatch):
        """Pin the terminal width; the banner is padded to it."""
        monkeypatch.setattr(
            "gprMax.utilities.utilities.get_terminal_width", lambda: 100
        )

    def test_it_returns_a_string(self):
        assert isinstance(logo("4.0.0"), str)

    def test_the_version_appears(self):
        assert "4.0.0" in logo("4.0.0")

    def test_the_current_year_appears_in_the_copyright(self):
        """Generated from the clock, so a stale year is a real possibility."""
        import datetime

        assert str(datetime.datetime.now().year) in logo("4.0.0")

    def test_the_project_url_appears(self):
        assert "www.gprmax.com" in logo("4.0.0")

    def test_the_authors_appear(self):
        assert "Craig Warren" in logo("4.0.0")

    def test_the_licence_is_named(self):
        assert "GNU General Public License" in logo("4.0.0")

    def test_no_line_exceeds_the_terminal_width(self):
        """The reason the width is read at all.

        Colour escape sequences are stripped first — they occupy no columns
        but do occupy characters.
        """
        import re

        text = re.sub(r"\x1b\[[0-9;]*m", "", logo("4.0.0"))

        assert max(len(line) for line in text.split("\n")) <= 100

    def test_it_does_not_print(self, capsys):
        """The caller logs the returned string; the function is pure."""
        logo("4.0.0")

        assert capsys.readouterr().out == ""
