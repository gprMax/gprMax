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

"""``set_omp_threads`` — the five environment variables that size the solver.

The CPU solver's inner loops are OpenMP-parallel Cython. OpenMP is configured
entirely through environment variables read by the runtime *when the first
parallel region is entered*, which means this function has to run before any
kernel does, and that whatever it writes is what the solver gets. There is no
API to query it afterwards and no way to change it later in the process.

That makes ``os.environ`` the return value, in practice. ``nthreads`` is
returned too, but four of the five variables the function sets are never read
back by gprMax — they exist purely to be seen by libgomp. So the tests below
assert on the environment rather than on the return value, and every one of
them goes through ``monkeypatch``: a leaked ``OMP_NUM_THREADS`` would change
how the *rest of the test session* runs, not merely this file.

Three behaviours are worth naming before reading the tests:

* **The thread count is chosen from physical cores, not logical ones.**
  Hyperthreads share an execution unit, so running two OpenMP threads on one
  core makes an FDTD kernel slower, not faster. This is the single most
  performance-relevant line in the utilities package.
* **An existing ``OMP_NUM_THREADS`` wins.** A user, a scheduler or an MPI
  launcher may have set it deliberately; gprMax defers.
* **Windows Subsystem for Linux gets special treatment.** ``OMP_PLACES`` and
  ``OMP_PROC_BIND`` are *deleted* there, because binding threads to cores
  under WSL hangs (microsoft/WSL#785).

``clean_omp_environment`` in ``conftest.py`` removes any inherited ``OMP_*``
before each test, so a developer who exports one in their shell sees the same
results as CI.
"""

import os

import pytest

from gprMax.utilities.host_info import set_omp_threads


class TestTheThreadCount:
    """How many threads, and where the number comes from."""

    def test_it_defaults_to_the_physical_core_count(self, install_host_config):
        """Not the logical count — hyperthreads hurt an FDTD kernel."""
        install_host_config()

        assert set_omp_threads() == 6

    def test_the_default_is_written_to_the_environment(self, install_host_config):
        install_host_config()

        set_omp_threads()

        assert os.environ["OMP_NUM_THREADS"] == "6"

    def test_an_explicit_count_is_used(self, install_host_config):
        install_host_config()

        assert set_omp_threads(4) == 4

    def test_an_explicit_count_is_written_to_the_environment(self, install_host_config):
        install_host_config()

        set_omp_threads(4)

        assert os.environ["OMP_NUM_THREADS"] == "4"

    def test_an_explicit_count_overrides_the_environment(self, install_host_config, monkeypatch):
        """``-n 2`` on the command line beats an inherited variable."""
        install_host_config()
        monkeypatch.setenv("OMP_NUM_THREADS", "8")

        assert set_omp_threads(2) == 2

    def test_an_inherited_count_is_respected(self, install_host_config, monkeypatch):
        """A scheduler or MPI launcher may have set it deliberately."""
        install_host_config()
        monkeypatch.setenv("OMP_NUM_THREADS", "3")

        assert set_omp_threads() == 3

    def test_an_inherited_count_is_returned_as_an_integer(self, install_host_config, monkeypatch):
        """It arrives as a string; callers do arithmetic with it."""
        install_host_config()
        monkeypatch.setenv("OMP_NUM_THREADS", "3")

        assert isinstance(set_omp_threads(), int)

    def test_an_inherited_count_is_not_rewritten(self, install_host_config, monkeypatch):
        install_host_config()
        monkeypatch.setenv("OMP_NUM_THREADS", "3")

        set_omp_threads()

        assert os.environ["OMP_NUM_THREADS"] == "3"

    def test_a_count_of_zero_falls_through_to_the_default(self, install_host_config):
        """``if nthreads:`` is a truthiness test, so ``0`` is not "no threads".

        Arguably right — zero OpenMP threads is meaningless — but it means
        the argument cannot be used to request a serial run.
        """
        install_host_config()

        assert set_omp_threads(0) == 6

    def test_an_empty_inherited_value_falls_through_to_the_default(
        self, install_host_config, monkeypatch
    ):
        """``os.environ.get`` returns ``""``, which is falsy.

        Without this branch the ``int("")`` would raise.
        """
        install_host_config()
        monkeypatch.setenv("OMP_NUM_THREADS", "")

        assert set_omp_threads() == 6

    def test_the_physical_core_count_is_read_from_the_global_config(self, install_host_config):
        """Not probed again — the value ``get_host_info`` found at startup."""
        install_host_config(physicalcores=13)

        assert set_omp_threads() == 13


class TestTheFixedSettings:
    """Three variables set on every platform, for every run."""

    def test_dynamic_thread_adjustment_is_disabled(self, install_host_config):
        """The runtime must not shrink the team mid-solve.

        An FDTD timestep is a fixed amount of work; a varying team size makes
        per-iteration timings meaningless and can leave cores idle.
        """
        install_host_config()

        set_omp_threads()

        assert os.environ["OMP_DYNAMIC"] == "FALSE"

    def test_places_are_cores(self, install_host_config):
        install_host_config()

        set_omp_threads()

        assert os.environ["OMP_PLACES"] == "cores"

    def test_threads_are_bound_to_their_places(self, install_host_config):
        """Without binding the OS migrates threads and destroys cache locality."""
        install_host_config()

        set_omp_threads()

        assert os.environ["OMP_PROC_BIND"] == "TRUE"

    def test_the_three_are_set_regardless_of_the_thread_count(self, install_host_config):
        """They precede the ``nthreads`` branching, so no path skips them."""
        install_host_config()

        set_omp_threads(1)

        assert os.environ["OMP_DYNAMIC"] == "FALSE"
        assert os.environ["OMP_PLACES"] == "cores"
        assert os.environ["OMP_PROC_BIND"] == "TRUE"


class TestTheWaitPolicyOnMacOs:
    """``OMP_WAIT_POLICY`` — set only on ``darwin``, and only there."""

    @pytest.fixture(autouse=True)
    def on_macos(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "darwin")

    def test_apple_silicon_gets_a_passive_wait(self, install_host_config):
        """Apple's tuning guide: spinning threads steal power budget.

        On an efficiency-core design a busy-wait does not merely waste a core,
        it lowers the clock available to the cores doing real work.
        """
        install_host_config(cpuID="Apple M2 Pro")

        set_omp_threads()

        assert os.environ["OMP_WAIT_POLICY"] == "PASSIVE"

    def test_an_intel_mac_gets_an_active_wait(self, install_host_config):
        """Spinning is the faster choice on a conventional x86 Mac."""
        install_host_config(cpuID="Intel(R) Core(TM) i9-9880H")

        set_omp_threads()

        assert os.environ["OMP_WAIT_POLICY"] == "ACTIVE"

    def test_the_branch_is_chosen_by_a_substring_of_the_cpu_name(self, install_host_config):
        """``"Apple" in cpuID`` — the only signal available."""
        install_host_config(cpuID="Apple")

        set_omp_threads()

        assert os.environ["OMP_WAIT_POLICY"] == "PASSIVE"

    def test_an_unknown_cpu_gets_the_active_wait(self, install_host_config):
        """The case that actually occurs on Apple silicon.

        ``machdep.cpu.brand_string`` does not exist on M-series chips, so
        ``get_host_info`` leaves ``cpuID`` as ``"unknown"`` — and this function
        then chooses ``ACTIVE``, the opposite of what the hardware wants. The
        two defects compound; written up together in
        ``notes/bugs/host-info-apple-silicon-cpuid.md``.
        """
        install_host_config(cpuID="unknown")

        set_omp_threads()

        assert os.environ["OMP_WAIT_POLICY"] == "ACTIVE"


class TestTheWaitPolicyElsewhere:
    """Not set at all on Windows or Linux — the default is left alone."""

    @pytest.mark.parametrize("platform_name", ["win32", "linux"])
    def test_it_is_not_set(self, install_host_config, monkeypatch, platform_name):
        monkeypatch.setattr("sys.platform", platform_name)
        install_host_config()

        set_omp_threads()

        assert "OMP_WAIT_POLICY" not in os.environ

    def test_an_inherited_value_is_left_untouched(self, install_host_config, monkeypatch):
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.setenv("OMP_WAIT_POLICY", "PASSIVE")
        install_host_config()

        set_omp_threads()

        assert os.environ["OMP_WAIT_POLICY"] == "PASSIVE"


class TestWindowsSubsystemForLinux:
    """Binding hangs under WSL, so the binding variables are removed again."""

    def test_affinity_is_disabled(self, install_host_config):
        """microsoft/WSL#785 — thread affinity is not implemented there."""
        install_host_config(osversion="Linux 4.4.0-Microsoft")

        set_omp_threads()

        assert os.environ["KMP_AFFINITY"] == "disabled"

    def test_the_places_variable_is_removed(self, install_host_config):
        """Set unconditionally a few lines earlier, then deleted here."""
        install_host_config(osversion="Linux 4.4.0-Microsoft")

        set_omp_threads()

        assert "OMP_PLACES" not in os.environ

    def test_the_binding_variable_is_removed(self, install_host_config):
        install_host_config(osversion="Linux 4.4.0-Microsoft")

        set_omp_threads()

        assert "OMP_PROC_BIND" not in os.environ

    def test_the_thread_count_is_still_set(self, install_host_config):
        """Only the binding is dropped; parallelism is not."""
        install_host_config(osversion="Linux 4.4.0-Microsoft")

        assert set_omp_threads() == 6

    def test_dynamic_adjustment_is_still_disabled(self, install_host_config):
        install_host_config(osversion="Linux 4.4.0-Microsoft")

        set_omp_threads()

        assert os.environ["OMP_DYNAMIC"] == "FALSE"

    def test_the_detection_is_a_substring_of_the_os_version(self, install_host_config):
        """``"Microsoft" in osversion`` — capital M, and case-sensitive.

        WSL2 kernels report ``microsoft-standard-WSL2`` in lower case, so this
        check misses them entirely. That is arguably correct — WSL2 is a real
        VM and does support affinity — but it is accidental rather than
        intended. Recorded in the analogy doc's observations table.
        """
        install_host_config(osversion="5.15.0-microsoft-standard-WSL2")

        set_omp_threads()

        assert os.environ["OMP_PLACES"] == "cores"

    def test_an_ordinary_linux_keeps_its_binding(self, install_host_config):
        install_host_config(osversion="Linux-6.0.0-x86_64")

        set_omp_threads()

        assert os.environ["OMP_PROC_BIND"] == "TRUE"


class TestEnvironmentHygiene:
    """The function's whole effect is a side effect; pin its extent.

    Every test here patches ``sys.platform``, because the number of variables
    written is platform-dependent — macOS gets a fifth. Leaving it unpatched
    makes the count assertion pass on two runners and fail on the third, which
    is the failure mode this suite exists to remove.
    """

    def test_exactly_four_variables_are_written(self, install_host_config, monkeypatch):
        """A fifth would be a silent change to how the solver runs."""
        monkeypatch.setattr("sys.platform", "linux")
        install_host_config()
        before = set(os.environ)

        set_omp_threads()

        assert set(os.environ) - before == {
            "OMP_NUM_THREADS",
            "OMP_DYNAMIC",
            "OMP_PLACES",
            "OMP_PROC_BIND",
        }

    def test_windows_writes_the_same_four(self, install_host_config, monkeypatch):
        monkeypatch.setattr("sys.platform", "win32")
        install_host_config()
        before = set(os.environ)

        set_omp_threads()

        assert set(os.environ) - before == {
            "OMP_NUM_THREADS",
            "OMP_DYNAMIC",
            "OMP_PLACES",
            "OMP_PROC_BIND",
        }

    def test_macos_writes_a_fifth(self, install_host_config, monkeypatch):
        """``OMP_WAIT_POLICY`` — the one platform-conditional variable.

        Asserted as its own test rather than folded into the count above, so
        the difference between the platforms is stated rather than hidden in a
        conditional expectation.
        """
        monkeypatch.setattr("sys.platform", "darwin")
        install_host_config()
        before = set(os.environ)

        set_omp_threads()

        assert set(os.environ) - before == {
            "OMP_NUM_THREADS",
            "OMP_DYNAMIC",
            "OMP_PLACES",
            "OMP_PROC_BIND",
            "OMP_WAIT_POLICY",
        }

    def test_nothing_unrelated_is_removed(self, install_host_config, monkeypatch):
        monkeypatch.setattr("sys.platform", "linux")
        install_host_config()
        monkeypatch.setenv("A_BYSTANDER", "untouched")

        set_omp_threads()

        assert os.environ["A_BYSTANDER"] == "untouched"

    def test_calling_twice_is_idempotent(self, install_host_config, monkeypatch):
        """The context loop configures once per model in a multi-model run."""
        monkeypatch.setattr("sys.platform", "linux")
        install_host_config()

        set_omp_threads()
        first = dict(os.environ)
        set_omp_threads()

        assert dict(os.environ) == first

    def test_the_second_call_sees_its_own_first_call(self, install_host_config, monkeypatch):
        """A subtle consequence: the first call sets ``OMP_NUM_THREADS``.

        So the second takes the *inherited* branch rather than recomputing
        from the core count. Identical result here, but it means an explicit
        argument on a later call is the only way to change the number.
        """
        monkeypatch.setattr("sys.platform", "linux")
        install_host_config()
        set_omp_threads()

        assert set_omp_threads() == 6


pytestmark = pytest.mark.unit
