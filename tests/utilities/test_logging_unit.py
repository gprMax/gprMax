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

"""The custom log level, the colour formatter, and the logger setup.

gprMax does not use ``print``. Everything a user sees — the banner, the host
description, the per-model progress, the memory warnings — arrives through
``logger.basic(...)``, a method that does not exist in the standard library.
``gprMax/utilities/logging.py`` creates it at **import time**, by mutating the
``logging`` module itself:

* ``logging.addLevelName(25, "BASIC")`` registers the name;
* ``logging.BASIC = 25`` publishes the constant;
* ``logging.Logger.basic = basic`` bolts a method onto the stdlib class.

None of those are reversible, and all three are process-wide: any library in
the same interpreter now sees a ``BASIC`` level it never asked for. That is a
deliberate trade — it makes ``logger.basic`` available everywhere without an
import — but it means the tests below are asserting on the state of the
*standard library*, not on a gprMax object. They are written to be read that
way.

Level 25 sits between ``INFO`` (20) and ``WARNING`` (30), which is what makes
it useful: a user running at the default level sees ``BASIC`` output, and
``--log-level 30`` silences it while leaving warnings intact. The numeric
ordering is therefore a contract, not an implementation detail, and it is
pinned below.

``logging_config`` is tested through the handlers it installs rather than
through captured output. ``caplog`` attaches its own handler at the root and
``propagate`` is set to ``False`` here, so the two do not compose; inspecting
``logger.handlers`` is both more direct and more honest about what the
function actually does.
"""

import logging

import pytest

from gprMax.utilities.logging import BASIC_NUM, MAPPING, CustomFormatter, logging_config


@pytest.fixture
def temporary_logger():
    """A uniquely named logger, torn down after the test.

    ``logging.getLogger`` caches by name in a process-wide dictionary, so
    reusing ``"gprMax"`` here would leave handlers attached for every later
    test in the session — including the ones in other directories that assert
    on log output.
    """
    created = []

    def _make(name):
        created.append(name)
        return logging.getLogger(name)

    yield _make

    for name in created:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)
        logger.propagate = True


class TestTheBasicLevel:
    """Level 25 — registered in the standard library at import."""

    def test_the_number_is_twenty_five(self):
        assert BASIC_NUM == 25

    def test_it_sits_between_info_and_warning(self):
        """The property that makes the level useful, not its value.

        A user at the default ``INFO`` threshold sees ``BASIC`` messages;
        raising the threshold to ``WARNING`` hides them without hiding
        warnings.
        """
        assert logging.INFO < BASIC_NUM < logging.WARNING

    def test_the_name_is_registered_with_the_standard_library(self):
        assert logging.getLevelName(BASIC_NUM) == "BASIC"

    def test_the_reverse_lookup_also_works(self):
        """``getLevelName`` is bidirectional; ``--log-level BASIC`` needs this."""
        assert logging.getLevelName("BASIC") == BASIC_NUM

    def test_the_constant_is_published_on_the_logging_module(self):
        """``logging.BASIC`` — so callers need not import from gprMax."""
        assert logging.BASIC == BASIC_NUM

    def test_the_method_is_attached_to_every_logger(self):
        """Including loggers created before gprMax was imported.

        ``Logger.basic`` is set on the *class*, so it appears retroactively.
        """
        assert callable(logging.getLogger("anything-at-all").basic)


class TestBasicMessages:
    """``logger.basic(...)`` — what the method actually emits."""

    def test_a_message_is_recorded_at_level_twenty_five(self, caplog):
        caplog.set_level(logging.DEBUG)

        logging.getLogger("test-basic").basic("hello")

        assert caplog.records[0].levelno == BASIC_NUM

    def test_the_level_name_on_the_record_is_basic(self, caplog):
        caplog.set_level(logging.DEBUG)

        logging.getLogger("test-basic").basic("hello")

        assert caplog.records[0].levelname == "BASIC"

    def test_the_message_survives(self, caplog):
        caplog.set_level(logging.DEBUG)

        logging.getLogger("test-basic").basic("the payload")

        assert caplog.records[0].getMessage() == "the payload"

    def test_arguments_are_interpolated_lazily(self, caplog):
        """``%``-style arguments, as with every other level method."""
        caplog.set_level(logging.DEBUG)

        logging.getLogger("test-basic").basic("%d cores", 6)

        assert caplog.records[0].getMessage() == "6 cores"

    def test_it_is_suppressed_above_its_level(self, caplog):
        """The point of putting it below ``WARNING``."""
        caplog.set_level(logging.WARNING)

        logging.getLogger("test-basic").basic("should not appear")

        assert caplog.records == []

    def test_a_warning_still_passes_at_that_threshold(self, caplog):
        """Paired with the previous test: only ``BASIC`` is silenced."""
        caplog.set_level(logging.WARNING)

        logging.getLogger("test-basic").warning("should appear")

        assert len(caplog.records) == 1

    def test_the_enabled_check_happens_before_the_log_call(self, caplog):
        """``isEnabledFor`` guards the body, so a disabled call is cheap.

        Asserted through behaviour: nothing is recorded, and no formatting
        error is raised by the deliberately broken format string.
        """
        caplog.set_level(logging.CRITICAL)

        logging.getLogger("test-basic").basic("%d", "not a number")

        assert caplog.records == []


class TestTheColourMapping:
    """``MAPPING`` — one colour per level name."""

    def test_every_standard_level_has_a_colour(self):
        assert set(MAPPING) == {
            "DEBUG",
            "INFO",
            "BASIC",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

    def test_warnings_and_errors_are_visually_distinct(self):
        """The reason the mapping exists at all."""
        assert MAPPING["WARNING"] != MAPPING["ERROR"]

    def test_the_quiet_levels_share_one_colour(self):
        assert MAPPING["DEBUG"] == MAPPING["INFO"] == MAPPING["BASIC"]


class TestCustomFormatter:
    """``CustomFormatter`` — colour wrapped around level name and message."""

    @staticmethod
    def _record(level=logging.WARNING, msg="a message", args=()):
        return logging.LogRecord(
            name="test",
            level=level,
            pathname=__file__,
            lineno=1,
            msg=msg,
            args=args,
            exc_info=None,
        )

    def test_the_message_appears_in_the_output(self):
        formatted = CustomFormatter("%(message)s").format(self._record())

        assert "a message" in formatted

    def test_the_level_name_appears_when_the_pattern_asks_for_it(self):
        formatted = CustomFormatter("%(levelname)s: %(message)s").format(self._record())

        assert "WARNING" in formatted

    def test_colour_escapes_are_added(self):
        formatted = CustomFormatter("%(message)s").format(self._record())

        assert "\x1b[" in formatted

    def test_the_colour_is_reset_afterwards(self):
        """Otherwise every subsequent line in the terminal stays coloured."""
        formatted = CustomFormatter("%(message)s").format(self._record())

        assert formatted.endswith("\x1b[0m")

    @pytest.mark.parametrize(
        "level",
        [logging.DEBUG, logging.INFO, BASIC_NUM, logging.WARNING, logging.ERROR, logging.CRITICAL],
    )
    def test_every_level_formats(self, level):
        formatted = CustomFormatter("%(levelname)s %(message)s").format(self._record(level=level))

        assert "a message" in formatted

    def test_an_unmapped_level_gets_the_fallback_colour(self):
        """A level gprMax did not register still formats rather than raising."""
        from colorama import Fore

        formatted = CustomFormatter("%(message)s").format(self._record(level=5))

        assert Fore.BLUE in formatted

    def test_the_original_record_is_not_modified(self):
        """The formatter copies first.

        A file handler attached to the same logger must not receive escape
        sequences because a console handler formatted the record first.
        """
        record = self._record()

        CustomFormatter("%(message)s").format(record)

        assert record.levelname == "WARNING" and record.msg == "a message"

    def test_a_message_with_no_arguments_formats(self):
        """The only shape gprMax actually uses — every call site is an f-string."""
        formatted = CustomFormatter("%(message)s").format(self._record(msg="6 cores"))

        assert "6 cores" in formatted

    def test_a_message_with_arguments_formats_once(self):
        """The formatter supports the standard library's lazy interpolation."""
        formatted = CustomFormatter("%(message)s").format(self._record(msg="%d cores", args=(6,)))

        assert "6 cores" in formatted


class TestLoggingConfig:
    """``logging_config`` — the setup call the CLI and API both make."""

    def test_a_handler_is_installed(self, temporary_logger):
        logger = temporary_logger("test-config-handler")

        logging_config(name="test-config-handler")

        assert len(logger.handlers) == 1

    def test_the_handler_writes_to_stdout(self, temporary_logger):
        """Not stderr — gprMax's normal output is not an error stream."""
        import sys

        logger = temporary_logger("test-config-stdout")

        logging_config(name="test-config-stdout")

        assert logger.handlers[0].stream is sys.stdout

    def test_the_logger_itself_is_set_to_debug(self, temporary_logger):
        """Filtering happens on the handler, so the logger must let all through.

        This is what allows a file handler at ``DEBUG`` to coexist with a
        console handler at ``INFO``.
        """
        logger = temporary_logger("test-config-level")

        logging_config(name="test-config-level")

        assert logger.level == logging.DEBUG

    def test_the_requested_level_lands_on_the_handler(self, temporary_logger):
        logger = temporary_logger("test-config-handler-level")

        logging_config(name="test-config-handler-level", level=logging.WARNING)

        assert logger.handlers[0].level == logging.WARNING

    def test_propagation_is_disabled(self, temporary_logger):
        """Otherwise every message would also reach the root logger's handlers,
        printing it twice for anyone who called ``basicConfig``.
        """
        logger = temporary_logger("test-config-propagate")

        logging_config(name="test-config-propagate")

        assert logger.propagate is False

    def test_calling_twice_does_not_accumulate_handlers(self, temporary_logger):
        """The API can be called repeatedly in one interpreter session."""
        logger = temporary_logger("test-config-twice")

        logging_config(name="test-config-twice")
        logging_config(name="test-config-twice")

        assert len(logger.handlers) == 1

    def test_the_default_name_is_gprmax(self):
        """Every module logger is ``gprMax.<something>``, so this is the root."""
        import inspect

        signature = inspect.signature(logging_config)

        assert signature.parameters["name"].default == "gprMax"

    def test_the_default_level_is_info(self):
        import inspect

        signature = inspect.signature(logging_config)

        assert signature.parameters["level"].default == logging.INFO


class TestFormatStyles:
    """``format_style`` — the terse default and the diagnostic alternative."""

    def test_the_standard_style_is_the_message_alone(self, temporary_logger):
        logger = temporary_logger("test-style-std")

        logging_config(name="test-style-std", format_style="std")

        assert logger.handlers[0].formatter._fmt == "%(message)s"

    def test_the_full_style_carries_the_source_location(self, temporary_logger):
        logger = temporary_logger("test-style-full")

        logging_config(name="test-style-full", format_style="full")

        assert "%(lineno)d" in logger.handlers[0].formatter._fmt

    def test_the_full_style_carries_a_timestamp(self, temporary_logger):
        logger = temporary_logger("test-style-time")

        logging_config(name="test-style-time", format_style="full")

        assert "%(asctime)s" in logger.handlers[0].formatter._fmt

    def test_debug_level_forces_the_full_style(self, temporary_logger):
        """``--log-level DEBUG`` upgrades the format without a second flag.

        Anyone asking for debug output wants to know where it came from.
        """
        logger = temporary_logger("test-style-debug")

        logging_config(name="test-style-debug", level=logging.DEBUG, format_style="std")

        assert "%(lineno)d" in logger.handlers[0].formatter._fmt

    def test_the_formatter_is_the_colour_one(self, temporary_logger):
        logger = temporary_logger("test-style-colour")

        logging_config(name="test-style-colour")

        assert isinstance(logger.handlers[0].formatter, CustomFormatter)


class TestFileLogging:
    """``log_file=True`` — a second, uncoloured handler on disk."""

    @pytest.fixture(autouse=True)
    def in_tmp_path(self, monkeypatch, tmp_path):
        """The file name is relative, so run in a scratch directory."""
        monkeypatch.chdir(tmp_path)

    def test_a_second_handler_is_added(self, temporary_logger):
        logger = temporary_logger("test-file-count")

        logging_config(name="test-file-count", log_file=True)

        assert len(logger.handlers) == 2

    def test_the_second_handler_writes_to_a_file(self, temporary_logger):
        logger = temporary_logger("test-file-type")

        logging_config(name="test-file-type", log_file=True)

        assert isinstance(logger.handlers[1], logging.FileHandler)

    def test_the_file_name_starts_with_the_logger_name(self, temporary_logger, tmp_path):
        temporary_logger("test-file-name")

        logging_config(name="test-file-name", log_file=True)

        assert list(tmp_path.glob("test-file-name-log-*.txt"))

    def test_the_file_handler_records_everything(self, temporary_logger):
        """``DEBUG`` on disk regardless of the console level.

        A user reporting a problem can be asked for the log file without
        being asked to re-run at a different verbosity.
        """
        logger = temporary_logger("test-file-level")

        logging_config(name="test-file-level", level=logging.WARNING, log_file=True)

        assert logger.handlers[1].level == logging.DEBUG

    def test_the_file_is_not_coloured(self, temporary_logger):
        """Escape sequences in a text file are noise, not colour."""
        logger = temporary_logger("test-file-plain")

        logging_config(name="test-file-plain", log_file=True)

        assert not isinstance(logger.handlers[1].formatter, CustomFormatter)

    def test_the_file_always_uses_the_full_format(self, temporary_logger):
        """Even when the console is terse."""
        logger = temporary_logger("test-file-format")

        logging_config(name="test-file-format", format_style="std", log_file=True)

        assert "%(lineno)d" in logger.handlers[1].formatter._fmt

    def test_messages_reach_the_file(self, temporary_logger, tmp_path):
        """End to end: configure, log, close, read back."""
        logger = temporary_logger("test-file-content")
        logging_config(name="test-file-content", log_file=True)

        logger.basic("written to disk")
        logger.handlers[1].flush()

        written = next(tmp_path.glob("test-file-content-log-*.txt")).read_text()
        assert "written to disk" in written

    def test_the_level_name_reaches_the_file(self, temporary_logger, tmp_path):
        logger = temporary_logger("test-file-levelname")
        logging_config(name="test-file-levelname", log_file=True)

        logger.basic("a basic message")
        logger.handlers[1].flush()

        written = next(tmp_path.glob("test-file-levelname-log-*.txt")).read_text()
        assert "BASIC" in written


pytestmark = pytest.mark.unit
