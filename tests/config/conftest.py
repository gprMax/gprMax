"""Shared fixtures for the configuration test suite.

**This suite is the odd one out, deliberately.** Every other directory under
``tests/unit/`` replaces ``config.sim_config`` and ``config.get_model_config``
with ``SimpleNamespace`` stand-ins. That was the right call each time — it
keeps those suites fast and independent of a global. But it means the real
``SimulationConfig`` and ``ModelConfig`` have never been executed by a test,
and neither has the model-index bookkeeping that decides which config
``get_model_config()`` returns. That is the largest untested surface in the
package, and it sits underneath everything.

So here the real classes are built and driven. Two consequences shape every
fixture below.

**The construction order is a hard precondition.** ``ModelConfig.__init__``
reads the module-level ``sim_config`` — for the banner string, for
``model_end``, and for ``args.n`` — so a ``ModelConfig`` cannot exist before a
``SimulationConfig`` has been installed as the global. ``make_model_config``
enforces that by installing one if the test has not.

**Four host probes have to be neutralised.** ``SimulationConfig.__init__``
calls ``get_host_info()`` unconditionally, plus ``detect_cuda_gpus()``,
``detect_opencl()`` and ``detect_metal()`` on the accelerator paths. All four
are imported *by name* into ``gprMax.config``, so they must be patched in that
namespace — patching ``gprMax.utilities.host_info.get_host_info`` has no
effect, because the name was already bound at import. ``get_terminal_width``
is patched for the same reason and to make the banner string deterministic
across the three CI runners, whose terminal widths differ.

Because these tests mutate ``gprMax.config.sim_config`` — the global every
other suite monkeypatches — ``restore_config_globals`` saves and restores it
around every test. Without it, a failure here would cascade into unrelated
suites depending on file ordering.
"""

import argparse
from types import SimpleNamespace

import pytest

# A terminal width fixed for the whole suite. The real value differs between
# an interactive shell (variable), pytest with no tty (80) and the three CI
# runners, so any assertion on the banner string has to pin it.
TERMINAL_WIDTH = 100

# A host-info dictionary with the same nine keys the real probe returns, in
# the same order, with values that are obviously synthetic.
FAKE_HOST_INFO = {
    "hostname": "test-host",
    "machineID": "Test Manufacturer Test Model",
    "sockets": 2,
    "cpuID": "Test CPU @ 1.00GHz",
    "osversion": "Test OS 1.0",
    "hyperthreading": False,
    "logicalcores": 8,
    "physicalcores": 4,
    "ram": 16 * 1024**3,
}


@pytest.fixture(autouse=True)
def restore_config_globals(request):
    """Save and restore every module-level name this suite writes.

    ``sim_config`` is a plain module attribute with no reset hook, and these
    tests assign to it directly. Restoring it is what keeps the rest of
    ``tests/unit/`` order-independent.
    """
    if request.node.get_closest_marker("unit") is None:
        yield
        return

    from gprMax import config

    saved = config.sim_config
    yield
    config.sim_config = saved


@pytest.fixture(autouse=True)
def no_host_probes(monkeypatch, request):
    """Replace the four hardware probes and the terminal-width lookup.

    Patched on ``gprMax.config`` rather than at their definition sites,
    because ``config.py`` imports them by name. Without this every
    ``SimulationConfig()`` would shell out to ``wmic`` / ``sysctl`` /
    ``lscpu``, making the suite slow, non-deterministic and dependent on the
    runner's hardware.
    """
    if request.node.get_closest_marker("unit") is None:
        return

    from gprMax import config

    monkeypatch.setattr(config, "get_host_info", lambda: dict(FAKE_HOST_INFO))
    monkeypatch.setattr(config, "detect_cuda_gpus", lambda: {})
    monkeypatch.setattr(config, "detect_opencl", lambda: {})
    monkeypatch.setattr(config, "detect_metal", lambda: {})
    monkeypatch.setattr(config, "get_terminal_width", lambda: TERMINAL_WIDTH)


@pytest.fixture
def make_args():
    """Factory for the ``argparse.Namespace`` a ``SimulationConfig`` takes.

    Starts from ``gprMax.args_defaults`` — the same dictionary the API and
    the CLI both fill in — so the defaults under test are the production
    defaults rather than a guess. ``inputfile`` is supplied because
    ``_set_input_file_path`` does ``Path(args.inputfile)`` and the shipped
    default of ``None`` would raise ``TypeError``.
    """
    from gprMax import gprMax as gprMax_module

    def _make(**overrides):
        args = argparse.Namespace(**gprMax_module.args_defaults)
        args.inputfile = "model.in"
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    return _make


@pytest.fixture
def make_sim_config(make_args):
    """Factory for a real ``SimulationConfig``.

    Does **not** install it as the global — tests that need that call
    ``install_sim_config`` — so that construction-time behaviour can be
    tested without side effects.
    """
    from gprMax import config

    def _make(**overrides):
        return config.SimulationConfig(make_args(**overrides))

    return _make


@pytest.fixture
def install_sim_config(make_sim_config):
    """Build a real ``SimulationConfig`` and install it as the global.

    Returns the instance. ``restore_config_globals`` undoes the assignment
    after the test.
    """
    from gprMax import config

    def _install(**overrides):
        sim_config = make_sim_config(**overrides)
        config.sim_config = sim_config
        return sim_config

    return _install


@pytest.fixture
def make_model_config(install_sim_config):
    """Factory for a real ``ModelConfig``.

    Installs a ``SimulationConfig`` first, because ``ModelConfig.__init__``
    reads the global. Pass ``sim_config=...`` to reuse one already installed;
    otherwise a fresh one is built from ``args_defaults`` plus any overrides.
    """
    from gprMax import config

    def _make(model_num=0, sim_config=None, **overrides):
        if sim_config is None:
            install_sim_config(**overrides)
        else:
            config.sim_config = sim_config
        return config.ModelConfig(model_num)

    return _make


@pytest.fixture
def fake_device():
    """A stand-in compute device, as ``detect_*`` would return.

    The accelerator branches store whatever the probe hands back, so any
    object works. Named so an assertion failure is readable.
    """

    def _make(name="fake-device", total_memory=8 * 1024**3):
        return SimpleNamespace(name=name, total_memory=lambda: total_memory)

    return _make
