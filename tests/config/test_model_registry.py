"""The model-config registry — which ``ModelConfig`` the package sees.

A simulation is a *sequence* of models, but almost every module in gprMax
reaches for its configuration through a single no-argument call::

    from gprMax.config import get_model_config
    config = get_model_config()

There is no argument, no lookup key and no object passed down the call chain.
The answer comes from two pieces of mutable state on the module-level
``sim_config``: a list of ``ModelConfig`` slots (``model_configs``) and an
integer cursor into it (``current_model``). The context loop moves the cursor
between models; everything downstream silently follows.

That indirection is why this file exists. **Every other directory under
``tests/unit/`` replaces ``get_model_config`` with a stub**, precisely so it
does not have to care about the cursor — which means the real
``get_model_config`` / ``set_model_config`` / ``set_current_model`` triad has
never been executed by a test, despite being the single most-called piece of
configuration code in the package. A cursor left pointing at the wrong slot
would not raise: model 2 would quietly be written with model 1's output path,
materials and dispersive dtypes.

The tests below therefore assert three separate things:

* the plumbing — set a config, get it back, on the slot asked for;
* the *default* argument, which is where the cursor enters (both getters and
  both setters take ``model_num=None`` and mean "whatever ``current_model``
  says"), and which is the behaviour every caller in the package relies on;
* the failure mode — an unset slot raises ``ValueError`` and logs, rather than
  returning ``None`` for a caller to trip over later.

``sim_config`` is installed as the real module global here, not a double.
``restore_config_globals`` in ``conftest.py`` puts the original back.
"""

import pytest


class TestTheCursor:
    """``current_model`` — the integer that decides what everyone sees."""

    def test_starts_at_the_first_model(self, install_sim_config):
        assert install_sim_config(n=3).current_model == 0

    def test_set_current_model_moves_it(self, install_sim_config):
        sim_config = install_sim_config(n=3)

        sim_config.set_current_model(2)

        assert sim_config.current_model == 2

    def test_the_cursor_is_not_range_checked(self, install_sim_config):
        """``set_current_model`` is a bare assignment.

        Out-of-range values are accepted silently and only fail later, at the
        indexing in ``get_model_config``. Pinned so a future bounds check is a
        deliberate change rather than an accident.
        """
        sim_config = install_sim_config(n=1)

        sim_config.set_current_model(99)

        assert sim_config.current_model == 99


class TestStoringAndRetrieving:
    """``set_model_config`` / ``get_model_config`` on an explicit slot."""

    def test_a_config_can_be_stored_and_retrieved(self, install_sim_config):
        sim_config = install_sim_config(n=2)

        sim_config.set_model_config("model-zero", 0)

        assert sim_config.get_model_config(0) == "model-zero"

    def test_slots_are_independent(self, install_sim_config):
        sim_config = install_sim_config(n=3)

        sim_config.set_model_config("first", 0)
        sim_config.set_model_config("third", 2)

        assert sim_config.model_configs == ["first", None, "third"]

    def test_storing_twice_replaces(self, install_sim_config):
        sim_config = install_sim_config(n=1)

        sim_config.set_model_config("old", 0)
        sim_config.set_model_config("new", 0)

        assert sim_config.get_model_config(0) == "new"

    def test_the_stored_object_is_returned_by_identity(self, install_sim_config):
        """No copying, no wrapping — the caller gets the same object back.

        ``ModelConfig`` is mutated in place all through a model run (memory
        tallies, dispersive dtypes, the output path), so a copy anywhere in
        this path would silently discard those updates.
        """
        sim_config = install_sim_config(n=1)
        model_config = object()

        sim_config.set_model_config(model_config, 0)

        assert sim_config.get_model_config(0) is model_config


class TestTheDefaultSlot:
    """``model_num=None`` means "the current model" — for both directions."""

    def test_getting_without_a_number_follows_the_cursor(self, install_sim_config):
        sim_config = install_sim_config(n=3)
        sim_config.set_model_config("first", 0)
        sim_config.set_model_config("third", 2)

        sim_config.set_current_model(2)

        assert sim_config.get_model_config() == "third"

    def test_storing_without_a_number_follows_the_cursor(self, install_sim_config):
        sim_config = install_sim_config(n=3)
        sim_config.set_current_model(1)

        sim_config.set_model_config("current")

        assert sim_config.model_configs == [None, "current", None]

    def test_moving_the_cursor_changes_the_answer(self, install_sim_config):
        """The whole point of the indirection, in one test.

        Nothing is passed between these two calls; only the cursor moved.
        """
        sim_config = install_sim_config(n=2)
        sim_config.set_model_config("first", 0)
        sim_config.set_model_config("second", 1)

        sim_config.set_current_model(0)
        before = sim_config.get_model_config()
        sim_config.set_current_model(1)
        after = sim_config.get_model_config()

        assert (before, after) == ("first", "second")

    def test_slot_zero_is_not_confused_with_the_default(self, install_sim_config):
        """``model_num=0`` is falsy, so the guard must test ``is None``.

        A truthiness check here would make model 0 unaddressable whenever the
        cursor sat elsewhere.
        """
        sim_config = install_sim_config(n=2)
        sim_config.set_model_config("first", 0)
        sim_config.set_model_config("second", 1)
        sim_config.set_current_model(1)

        assert sim_config.get_model_config(0) == "first"


class TestUnsetSlots:
    """An empty slot is an error, not a ``None``."""

    def test_every_slot_starts_empty(self, install_sim_config):
        assert install_sim_config(n=4).model_configs == [None] * 4

    def test_getting_an_unset_slot_raises(self, install_sim_config):
        sim_config = install_sim_config(n=2)

        with pytest.raises(ValueError):
            sim_config.get_model_config(1)

    def test_getting_an_unset_current_model_raises(self, install_sim_config):
        sim_config = install_sim_config(n=2)
        sim_config.set_current_model(1)
        sim_config.set_model_config("first", 0)

        with pytest.raises(ValueError):
            sim_config.get_model_config()

    def test_the_missing_model_number_is_logged(self, install_sim_config, caplog):
        """The raise is bare, so the log line carries the whole diagnosis."""
        sim_config = install_sim_config(n=3)

        with pytest.raises(ValueError):
            sim_config.get_model_config(2)

        assert "model 2" in caplog.text

    def test_a_cursor_past_the_end_raises_index_error_not_value_error(self, install_sim_config):
        """The out-of-range case is a different failure from the unset case.

        ``get_model_config`` indexes before it checks for ``None``, so a
        cursor beyond the list raises ``IndexError`` with no log line. That
        distinction matters when reading a traceback: ``ValueError`` means the
        model was never configured, ``IndexError`` means the *index arithmetic*
        is wrong — the ``-i``/``-n`` mismatch written up in
        ``notes/bugs/config-model-index-range-mismatch.md``.
        """
        sim_config = install_sim_config(n=1)
        sim_config.set_current_model(1)

        with pytest.raises(IndexError):
            sim_config.get_model_config()


class TestTheModuleLevelHelper:
    """``config.get_model_config()`` — the function the whole package imports."""

    def test_it_delegates_to_the_installed_simulation_config(self, install_sim_config):
        from gprMax import config

        sim_config = install_sim_config(n=2)
        sim_config.set_model_config("current", 0)

        assert config.get_model_config() == "current"

    def test_it_takes_no_arguments(self, install_sim_config):
        """Callers cannot ask for a specific model through this door.

        The only way to change the answer is to move the cursor, which is why
        ``set_current_model`` is called exactly once per model by the context
        loop and nowhere else.
        """
        from gprMax import config

        install_sim_config(n=2)

        with pytest.raises(TypeError):
            config.get_model_config(0)

    def test_it_follows_the_cursor(self, install_sim_config):
        from gprMax import config

        sim_config = install_sim_config(n=2)
        sim_config.set_model_config("first", 0)
        sim_config.set_model_config("second", 1)
        sim_config.set_current_model(1)

        assert config.get_model_config() == "second"

    def test_it_reads_the_global_at_call_time(self, install_sim_config):
        """The lookup is late-bound, so replacing the global takes effect.

        This is what lets every other test directory swap in a stand-in
        ``sim_config`` and have the real ``get_model_config`` return it.
        """
        from gprMax import config

        first = install_sim_config(n=1)
        first.set_model_config("from-the-first", 0)
        second = install_sim_config(n=1)
        second.set_model_config("from-the-second", 0)

        assert config.get_model_config() == "from-the-second"

    def test_it_fails_loudly_when_no_simulation_config_is_installed(self):
        """The shipped value of the global is ``None``.

        Importing ``gprMax`` does not create a ``SimulationConfig``; the API
        and the CLI both build one. Calling before that point is an
        ``AttributeError`` on ``None``, not a helpful message.
        """
        from gprMax import config

        config.sim_config = None

        with pytest.raises(AttributeError):
            config.get_model_config()


class TestTheCursorAlsoSelectsTheScene:
    """One cursor, two lists — scenes are indexed by the same integer."""

    def test_the_scene_follows_the_cursor(self, install_sim_config):
        sim_config = install_sim_config(n=2, scenes=["first", "second"])

        sim_config.set_current_model(1)

        assert sim_config.get_scene() == "second"

    def test_a_model_config_reads_its_scene_through_its_own_number(
        self, install_sim_config, make_model_config
    ):
        """``ModelConfig.get_scene`` passes ``self.model_num``, not the cursor.

        So a ``ModelConfig`` keeps pointing at its own scene even if the
        cursor has already moved on — the one place in this file where the
        answer does *not* depend on ``current_model``.
        """
        sim_config = install_sim_config(n=2, scenes=["first", "second"])
        model_config = make_model_config(model_num=1, sim_config=sim_config)

        sim_config.set_current_model(0)

        assert model_config.get_scene() == "second"

    def test_scene_slots_and_config_slots_are_the_same_length(self, install_sim_config):
        """Both default to ``number_of_models``, so one cursor addresses both."""
        sim_config = install_sim_config(n=5)

        assert len(sim_config.scenes) == len(sim_config.model_configs) == 5


pytestmark = pytest.mark.unit
