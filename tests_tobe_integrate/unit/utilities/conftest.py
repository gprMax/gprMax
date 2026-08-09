"""Shared fixtures for the utilities test suite.

``gprMax/utilities/`` is the layer that talks to everything outside Python:
the operating system, the shell, the environment, the terminal and the
logging machinery. That is exactly what makes it awkward to test and exactly
why it has never been tested. Four separate global surfaces have to be held
still before an assertion means anything:

**The shell.** ``get_host_info`` runs ``wmic`` / ``sysctl`` / ``lscpu`` /
``cat`` depending on ``sys.platform``. On any given CI runner two of the three
branches are dead code, and the third returns whatever that machine happens to
be. ``fake_subprocess`` replaces ``subprocess.check_output`` with a lookup
table keyed on the argv list, so all three branches run on all three runners
and return the same thing every time. It also makes the *interesting* cases
reachable: a machine that still has ``wmic`` cannot otherwise exercise the
PowerShell fallback that the student's own merged fix
(``ce2c456e``) added, and that fix was originally verified by temporarily
adding ``print`` statements inside each ``except`` block.

**The platform.** ``sys.platform`` is patched directly. It is a plain string
attribute, so this is safe and total — but it must be paired with
``fake_subprocess``, or a Windows runner told it is Linux will genuinely try
to run ``lscpu``.

**``os.environ``.** ``set_omp_threads`` writes five variables and deletes two.
Every test that touches it uses ``monkeypatch.setenv`` / ``delenv``, and
``clean_omp_environment`` removes any inherited ``OMP_*`` first — a developer
who exports ``OMP_NUM_THREADS`` in their shell would otherwise see different
results from CI.

**``config.sim_config``.** Four functions here read the global directly
(``print_host_info``, ``set_omp_threads``, ``mem_check_*``). Unlike
``tests/unit/config/``, this suite has no interest in the real class, so it
installs a ``SimpleNamespace`` — the same approach the other eight suites
take — and ``restore_sim_config`` puts the original back.

``psutil`` is deliberately *not* faked in ``get_host_info``: the three keys it
supplies (``logicalcores``, ``physicalcores``, ``ram``) are asserted for type
and relationship rather than value, which holds on any machine.
"""

import subprocess
from types import SimpleNamespace

import pytest

# The number of physical cores the fake host reports. Chosen to differ from
# the logical-core count so hyperthreading assertions are meaningful, and to
# be an implausible real value so a leaked real probe is obvious.
PHYSICAL_CORES = 6
LOGICAL_CORES = 12

# Total RAM of the fake host, 16 GiB.
HOST_RAM = 16 * 1024**3


@pytest.fixture(autouse=True)
def restore_sim_config():
    """Save and restore ``config.sim_config`` around every test.

    Several functions under test read the global at call time, and some
    tests replace it. Restoring keeps the rest of ``tests/unit/`` unaffected
    by ordering.
    """
    from gprMax import config

    saved = config.sim_config
    yield
    config.sim_config = saved


@pytest.fixture(autouse=True)
def clean_omp_environment(monkeypatch):
    """Remove any inherited OpenMP variables before each test.

    ``set_omp_threads`` branches on whether ``OMP_NUM_THREADS`` is already
    set, so a developer's shell export would change the result. ``monkeypatch``
    restores the real environment afterwards.
    """
    for name in (
        "OMP_NUM_THREADS",
        "OMP_WAIT_POLICY",
        "OMP_DYNAMIC",
        "OMP_PLACES",
        "OMP_PROC_BIND",
        "KMP_AFFINITY",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def install_host_config():
    """Install a stand-in ``sim_config`` carrying a host-info dictionary.

    A ``SimpleNamespace``, not a real ``SimulationConfig``: nothing here
    exercises the config class, and building one would drag in the host probes
    this suite is trying to avoid.
    """
    from gprMax import config

    def _install(**overrides):
        hostinfo = {
            "hostname": "test-host",
            "machineID": "Test Manufacturer Test Model",
            "sockets": 2,
            "cpuID": "Test CPU @ 1.00GHz",
            "osversion": "Test OS 1.0",
            "hyperthreading": False,
            "logicalcores": LOGICAL_CORES,
            "physicalcores": PHYSICAL_CORES,
            "ram": HOST_RAM,
        }
        hostinfo.update(overrides)
        config.sim_config = SimpleNamespace(
            hostinfo=hostinfo,
            general={"solver": "cpu"},
        )
        return config.sim_config

    return _install


@pytest.fixture
def fake_subprocess(monkeypatch):
    """Replace ``subprocess.check_output`` with a table keyed on argv.

    The table maps a tuple of arguments to the ``bytes`` the real command
    would print, or to an exception *instance* to be raised instead. Anything
    not in the table raises ``FileNotFoundError``, so a command the test did
    not anticipate fails loudly rather than reaching the real shell.

    Returns the recorder, whose ``calls`` list is the argv of every command
    attempted, in order. Several tests assert on that list rather than on the
    return value — the point of the wmic fallback is *which command runs*.
    """

    class Recorder:
        def __init__(self):
            self.calls = []
            self.table = {}

        def register(self, argv, result):
            self.table[tuple(argv)] = result

        def __call__(self, argv, *args, **kwargs):
            self.calls.append(list(argv))
            try:
                result = self.table[tuple(argv)]
            except KeyError:
                raise FileNotFoundError(
                    f"no fake registered for {list(argv)}"
                ) from None
            if isinstance(result, BaseException):
                raise result
            return result

    recorder = Recorder()
    monkeypatch.setattr(subprocess, "check_output", recorder)
    return recorder


@pytest.fixture
def fake_cpu_counts(monkeypatch):
    """Pin what ``psutil`` reports about cores and memory.

    ``get_host_info`` derives ``hyperthreading`` from whether the logical and
    physical counts differ, which on a real runner is whatever that machine
    happens to be. Called with no arguments the fake reports a hyperthreaded
    machine; pass equal counts for the other branch.
    """
    import psutil

    def _fake(physical=PHYSICAL_CORES, logical=LOGICAL_CORES, ram=HOST_RAM):
        monkeypatch.setattr(
            psutil,
            "cpu_count",
            lambda logical=True, _l=logical, _p=physical: _l if logical else _p,
        )
        monkeypatch.setattr(
            psutil, "virtual_memory", lambda: SimpleNamespace(total=ram)
        )

    return _fake


@pytest.fixture
def fake_platform(monkeypatch):
    """Pin the five ``platform`` lookups ``get_host_info`` makes.

    Applied automatically by the three host fixtures. Without it, the macOS
    branch running on a Windows runner would report ``"macOS ()"`` and the
    Linux branch would report the Windows platform string — deterministic per
    machine, but different between them.
    """
    import platform

    monkeypatch.setattr(platform, "node", lambda: "test-host")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(platform, "release", lambda: "11")
    monkeypatch.setattr(platform, "mac_ver", lambda: ("14.0", ("", "", ""), ""))
    monkeypatch.setattr(platform, "platform", lambda: "Linux-6.0.0-x86_64")


@pytest.fixture
def windows_host(monkeypatch, fake_subprocess, fake_platform, fake_cpu_counts):
    """A Windows machine whose ``wmic`` works, with every command registered.

    Returns the ``fake_subprocess`` recorder so a test can override individual
    entries — replacing one with ``FileNotFoundError`` is how the wmic-absent
    path is forced.

    The byte strings reproduce real ``wmic`` output, header line and all:
    ``wmic`` prints the column name first, which is why the parser splits on
    newlines and takes element ``[1]``. The PowerShell fallbacks print the
    value alone, which is why they are parsed with ``.strip()`` only.
    """
    monkeypatch.setattr("sys.platform", "win32")
    fake_cpu_counts()

    fake_subprocess.register(
        ["wmic", "csproduct", "get", "vendor"], b"Vendor\nTest Manufacturer\n"
    )
    fake_subprocess.register(
        ["wmic", "computersystem", "get", "model"], b"Model\nTest Model\n"
    )
    fake_subprocess.register(
        ["wmic", "cpu", "get", "Name"], b"Name\nTest CPU @ 1.00GHz\n"
    )
    return fake_subprocess


@pytest.fixture
def powershell_commands():
    """The three PowerShell argv lists the wmic fallbacks issue.

    Kept here rather than inline so the tests assert against one definition of
    the exact command, and so a change to the fix is a one-line update.
    """
    return {
        "vendor": [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance -ClassName Win32_ComputerSystemProduct"
            " | Select-Object -ExpandProperty Vendor",
        ],
        "model": [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance -ClassName Win32_ComputerSystem"
            " | Select-Object -ExpandProperty Model",
        ],
        "cpu": [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance -ClassName Win32_Processor"
            " | Select-Object -ExpandProperty Name",
        ],
    }


@pytest.fixture
def macos_host(monkeypatch, fake_subprocess, fake_platform, fake_cpu_counts):
    """A macOS machine with all three ``sysctl`` probes registered."""
    monkeypatch.setattr("sys.platform", "darwin")
    fake_cpu_counts()

    fake_subprocess.register(["sysctl", "-n", "hw.model"], b"TestMac1,1\n")
    fake_subprocess.register(["sysctl", "-n", "hw.packages"], b"2\n")
    fake_subprocess.register(
        ["sysctl", "-n", "machdep.cpu.brand_string"], b"Test CPU @ 1.00GHz\n"
    )
    return fake_subprocess


@pytest.fixture
def linux_host(monkeypatch, fake_subprocess, fake_platform, fake_cpu_counts):
    """A Linux machine with the DMI files, ``/proc/cpuinfo`` and ``lscpu``.

    Note the source reads the DMI files by shelling out to ``cat`` rather than
    opening them, which is why they appear here as subprocess entries.
    """
    monkeypatch.setattr("sys.platform", "linux")
    fake_cpu_counts()

    fake_subprocess.register(
        ["cat", "/sys/class/dmi/id/sys_vendor"], b"Test Manufacturer\n"
    )
    fake_subprocess.register(
        ["cat", "/sys/class/dmi/id/product_name"], b"Test Model\n"
    )
    fake_subprocess.register(
        ["cat", "/proc/cpuinfo"],
        b"processor\t: 0\nmodel name\t: Test CPU @ 1.00GHz\n"
        b"processor\t: 1\nmodel name\t: Test CPU @ 1.00GHz\n",
    )
    fake_subprocess.register(
        ["lscpu"],
        b"Architecture:        x86_64\n"
        b"Thread(s) per core:  2\n"
        b"Socket(s):           2\n",
    )
    return fake_subprocess


@pytest.fixture
def make_grid():
    """A stand-in grid for the memory-check functions.

    They call four methods and read two attributes; nothing else about a real
    ``FDTDGrid`` is touched, and building one would cost megabytes per test.
    """

    def _make(
        name="main",
        basic=1000,
        dispersive=2000,
        fractals=3000,
        snapshots=None,
        fractalvolumes=None,
        mem_use=0,
    ):
        return SimpleNamespace(
            name=name,
            mem_use=mem_use,
            snapshots=snapshots or [],
            fractalvolumes=fractalvolumes or [],
            mem_est_basic=lambda: basic,
            mem_est_dispersive=lambda: dispersive,
            mem_est_fractals=lambda: fractals,
        )

    return _make


@pytest.fixture
def install_model_config(monkeypatch):
    """Install a stand-in ``get_model_config`` returning a recording object.

    ``host_info.py`` does ``import gprMax.config as config`` and calls
    ``config.get_model_config()``, so patching the attribute on the module
    object is enough — unlike the ``from ... import`` bindings that
    ``gprMax/config.py`` itself uses for the host probes.

    The memory checks accumulate into ``get_model_config().mem_use``, so the
    returned object is what a test inspects afterwards.
    """
    from gprMax import config

    def _install(mem_use=0, maxpoles=0, device=None):
        model_config = SimpleNamespace(
            mem_use=mem_use,
            materials={"maxpoles": maxpoles},
            device=device,
        )
        monkeypatch.setattr(config, "get_model_config", lambda: model_config)
        return model_config

    return _install
