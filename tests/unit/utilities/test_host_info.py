"""``get_host_info`` — the probe that describes the machine gprMax runs on.

This is the most-shelled-out function in the package and, until now, the least
tested. It runs three or four external commands, parses their stdout with
string surgery, and returns nine keys that appear in the run banner, drive the
OpenMP thread count and gate the memory warnings.

**Why it is worth this much attention.** The Windows branch contains the
student's own merged contribution: Microsoft removed ``wmic`` in Windows 11
25H2, so ``subprocess.check_output(["wmic", ...])`` raises ``FileNotFoundError``
rather than ``CalledProcessError``, which the original ``except`` clause did
not catch — gprMax crashed on startup before printing anything. The fix
(``ce2c456e``) widened the clause and added a PowerShell ``Get-CimInstance``
fallback. ``gsocDocs/feats/setup-and-wmic-fix.rst`` records that it was
verified *by temporarily adding print statements inside each except block*.
These tests replace that with something a CI runner can check.

**Why the fakes are total.** Every ``subprocess.check_output`` call is served
from a table (``fake_subprocess``), ``sys.platform`` is patched, and so are
the five ``platform`` lookups and the two ``psutil`` counts. That is heavy —
but it is the only way to get three properties that matter:

* all three platform branches run on all three CI runners, instead of two
  being dead code on each;
* the wmic-absent path can be *forced*, which is impossible on a machine that
  still has wmic — the exact situation in which the bug was missed;
* the suite gives the same answer on every machine, so a failure is a code
  change and never a hardware difference.

A few tests pin behaviour that is defective rather than contractual. They say
so, and name the file in ``notes/bugs/``. They are here because the failure
modes are silent — a wrong socket count does not raise, it just prints a wrong
banner and, on a workstation, chooses the wrong number of OpenMP threads.
"""

import subprocess

import pytest

from gprMax.utilities.host_info import get_host_info, print_host_info

# The nine keys returned, in insertion order. Asserted as an ordered sequence
# rather than a set: ``print_host_info`` and the model banner both read them
# positionally in places, and a reordering would be invisible to a set check.
HOST_INFO_KEYS = [
    "hostname",
    "machineID",
    "sockets",
    "cpuID",
    "osversion",
    "hyperthreading",
    "logicalcores",
    "physicalcores",
    "ram",
]


class TestTheReturnedDictionary:
    """Shape and provenance of the result, independent of platform."""

    def test_exactly_nine_keys_are_returned(self, windows_host):
        assert list(get_host_info()) == HOST_INFO_KEYS

    def test_the_hostname_comes_from_the_platform_module(self, windows_host):
        """Not from any subprocess — the one field that never shells out."""
        assert get_host_info()["hostname"] == "test-host"

    def test_the_core_counts_come_from_psutil(self, windows_host):
        hostinfo = get_host_info()

        assert (hostinfo["logicalcores"], hostinfo["physicalcores"]) == (12, 6)

    def test_the_memory_total_comes_from_psutil(self, windows_host):
        assert get_host_info()["ram"] == 16 * 1024**3

    def test_the_physical_core_count_falls_back_to_the_logical_one(
        self, windows_host, fake_cpu_counts
    ):
        """``psutil.cpu_count(logical=False)`` returns ``None`` on some machines.

        Containers and some ARM hosts do not expose the topology. Without the
        fallback, ``set_omp_threads`` would set ``OMP_NUM_THREADS`` to
        ``"None"``.
        """
        fake_cpu_counts(physical=None, logical=8)

        assert get_host_info()["physicalcores"] == 8

    def test_a_zero_physical_core_count_also_falls_back(
        self, windows_host, fake_cpu_counts
    ):
        """``not 0`` is true, so zero takes the same branch as ``None``."""
        fake_cpu_counts(physical=0, logical=8)

        assert get_host_info()["physicalcores"] == 8


class TestWindowsWithWmic:
    """The Windows branch when ``wmic`` is present and working."""

    def test_the_manufacturer_is_read_from_wmic(self, windows_host):
        assert "Test Manufacturer" in get_host_info()["machineID"]

    def test_the_model_is_read_from_wmic(self, windows_host):
        assert "Test Model" in get_host_info()["machineID"]

    def test_the_machine_id_joins_manufacturer_and_model(self, windows_host):
        assert get_host_info()["machineID"] == "Test Manufacturer Test Model"

    def test_the_wmic_header_line_is_skipped(self, windows_host):
        """``wmic`` prints the column name first; element ``[1]`` is the value.

        Without the skip the banner would read ``Vendor Model``.
        """
        windows_host.register(
            ["wmic", "csproduct", "get", "vendor"], b"Vendor\nAcme Corp\n"
        )
        windows_host.register(
            ["wmic", "computersystem", "get", "model"], b"Model\nWidget 9000\n"
        )

        assert get_host_info()["machineID"] == "Acme Corp Widget 9000"

    def test_carriage_returns_are_stripped(self, windows_host):
        """Windows commands emit ``\\r\\n``; the parser splits on ``\\n`` only."""
        windows_host.register(
            ["wmic", "csproduct", "get", "vendor"],
            b"Vendor    \r\nTest Manufacturer    \r\n",
        )

        assert get_host_info()["machineID"].startswith("Test Manufacturer")

    def test_a_single_line_response_is_used_as_is(self, windows_host):
        """Some ``wmic`` builds omit the header; the parser handles both."""
        windows_host.register(
            ["wmic", "csproduct", "get", "vendor"], b"Test Manufacturer\n"
        )

        assert get_host_info()["machineID"] == "Test Manufacturer Test Model"

    def test_the_cpu_name_is_read_from_wmic(self, windows_host):
        assert get_host_info()["cpuID"] == "Test CPU @ 1.00GHz"

    def test_one_cpu_line_means_one_socket(self, windows_host):
        """Sockets are *counted*, not queried — one output line each."""
        assert get_host_info()["sockets"] == 1

    def test_two_cpu_lines_mean_two_sockets(self, windows_host):
        windows_host.register(
            ["wmic", "cpu", "get", "Name"],
            b"Name\nTest CPU @ 1.00GHz\nTest CPU @ 1.00GHz\n",
        )

        assert get_host_info()["sockets"] == 2

    def test_blank_lines_do_not_count_as_sockets(self, windows_host):
        """``wmic`` pads its output; a blank line is not a processor."""
        windows_host.register(
            ["wmic", "cpu", "get", "Name"],
            b"Name\nTest CPU @ 1.00GHz\n   \n\n",
        )

        assert get_host_info()["sockets"] == 1

    def test_internal_whitespace_in_the_cpu_name_is_collapsed(self, windows_host):
        """``wmic`` pads processor names with runs of spaces."""
        windows_host.register(
            ["wmic", "cpu", "get", "Name"],
            b"Name\nTest    CPU   @  1.00GHz\n",
        )

        assert get_host_info()["cpuID"] == "Test CPU @ 1.00GHz"

    def test_the_os_version_names_windows_and_its_bit_width(self, windows_host):
        assert get_host_info()["osversion"] == "Windows 11 (64-bit)"

    def test_a_thirty_two_bit_machine_is_reported_as_such(
        self, windows_host, monkeypatch
    ):
        import platform

        monkeypatch.setattr(platform, "machine", lambda: "x86")

        assert get_host_info()["osversion"].endswith("(32-bit)")

    def test_hyperthreading_is_detected_from_the_core_counts(self, windows_host):
        assert get_host_info()["hyperthreading"] is True

    def test_equal_core_counts_mean_no_hyperthreading(
        self, windows_host, fake_cpu_counts
    ):
        fake_cpu_counts(physical=8, logical=8)

        assert get_host_info()["hyperthreading"] is False


class TestWindowsWithoutWmic:
    """The reason this file exists: Windows 11 25H2 removed ``wmic``.

    ``subprocess.check_output`` raises ``FileNotFoundError`` — not
    ``CalledProcessError`` — when the executable does not exist. The original
    code caught only the latter, so gprMax died on import with a traceback
    from inside ``host_info``. Each test below forces that exact condition.
    """

    def test_a_missing_wmic_does_not_crash(self, windows_host, powershell_commands):
        """The regression the fix exists to prevent."""
        windows_host.register(
            ["wmic", "csproduct", "get", "vendor"], FileNotFoundError("wmic")
        )
        windows_host.register(powershell_commands["vendor"], b"Test Manufacturer\n")

        assert get_host_info()["machineID"] == "Test Manufacturer Test Model"

    def test_the_powershell_vendor_command_is_issued(
        self, windows_host, powershell_commands
    ):
        """The *exact* argv, not merely "something with powershell in it"."""
        windows_host.register(
            ["wmic", "csproduct", "get", "vendor"], FileNotFoundError("wmic")
        )
        windows_host.register(powershell_commands["vendor"], b"Test Manufacturer\n")

        get_host_info()

        assert powershell_commands["vendor"] in windows_host.calls

    def test_the_powershell_model_command_is_issued(
        self, windows_host, powershell_commands
    ):
        windows_host.register(
            ["wmic", "computersystem", "get", "model"], FileNotFoundError("wmic")
        )
        windows_host.register(powershell_commands["model"], b"Test Model\n")

        get_host_info()

        assert powershell_commands["model"] in windows_host.calls

    def test_the_powershell_cpu_command_is_issued(
        self, windows_host, powershell_commands
    ):
        windows_host.register(
            ["wmic", "cpu", "get", "Name"], FileNotFoundError("wmic")
        )
        windows_host.register(powershell_commands["cpu"], b"Test CPU @ 1.00GHz\n")

        get_host_info()

        assert powershell_commands["cpu"] in windows_host.calls

    def test_powershell_is_tried_only_after_wmic_fails(self, windows_host):
        """Order matters: ``wmic`` is faster, so it stays the first choice."""
        get_host_info()

        assert not any("powershell" in call[0] for call in windows_host.calls)

    def test_the_powershell_output_has_no_header_to_skip(
        self, windows_host, powershell_commands
    ):
        """``Select-Object -ExpandProperty`` prints the value alone.

        So the fallback parses with ``.strip()`` only. Feeding it wmic-shaped
        output with a header would produce the header as the answer — the
        asymmetry is why the two paths cannot share a parser.
        """
        windows_host.register(
            ["wmic", "csproduct", "get", "vendor"], FileNotFoundError("wmic")
        )
        windows_host.register(
            powershell_commands["vendor"], b"  Test Manufacturer  \r\n"
        )

        assert get_host_info()["machineID"] == "Test Manufacturer Test Model"

    def test_all_three_fall_back_independently(
        self, windows_host, powershell_commands
    ):
        """A machine with no wmic at all takes every fallback in one run."""
        for argv in (
            ["wmic", "csproduct", "get", "vendor"],
            ["wmic", "computersystem", "get", "model"],
            ["wmic", "cpu", "get", "Name"],
        ):
            windows_host.register(argv, FileNotFoundError("wmic"))
        windows_host.register(powershell_commands["vendor"], b"Test Manufacturer\n")
        windows_host.register(powershell_commands["model"], b"Test Model\n")
        windows_host.register(powershell_commands["cpu"], b"Test CPU @ 1.00GHz\n")

        hostinfo = get_host_info()

        assert hostinfo["machineID"] == "Test Manufacturer Test Model"
        assert hostinfo["cpuID"] == "Test CPU @ 1.00GHz"

    def test_the_socket_count_still_works_through_the_fallback(
        self, windows_host, powershell_commands
    ):
        """PowerShell prints one line per processor, so the count is unchanged."""
        windows_host.register(
            ["wmic", "cpu", "get", "Name"], FileNotFoundError("wmic")
        )
        windows_host.register(
            powershell_commands["cpu"],
            b"Test CPU @ 1.00GHz\nTest CPU @ 1.00GHz\n",
        )

        assert get_host_info()["sockets"] == 2

    def test_a_failing_wmic_also_falls_back(self, windows_host, powershell_commands):
        """The original clause caught ``CalledProcessError``; it still must.

        Widening the ``except`` must not have lost the case it already handled
        — a ``wmic`` that exists but exits non-zero.
        """
        windows_host.register(
            ["wmic", "csproduct", "get", "vendor"],
            subprocess.CalledProcessError(1, "wmic"),
        )
        windows_host.register(powershell_commands["vendor"], b"Test Manufacturer\n")

        assert get_host_info()["machineID"] == "Test Manufacturer Test Model"

    def test_both_probes_failing_leaves_the_field_unknown(
        self, windows_host, powershell_commands
    ):
        """The default set at the top of the function survives.

        A machine with neither wmic nor a working ``Get-CimInstance`` still
        gets a banner, just a vaguer one.
        """
        windows_host.register(
            ["wmic", "csproduct", "get", "vendor"], FileNotFoundError("wmic")
        )
        windows_host.register(
            powershell_commands["vendor"],
            subprocess.CalledProcessError(1, "powershell"),
        )

        assert get_host_info()["machineID"] == "unknown Test Model"

    def test_a_missing_powershell_is_not_caught(
        self, windows_host, powershell_commands
    ):
        """The inner ``except`` still catches only ``CalledProcessError``.

        So the very failure mode the outer clause was widened for is
        unhandled one level down: on a machine with neither ``wmic`` nor
        ``powershell.exe`` on ``PATH`` — a stripped Windows container, or a
        ``PATH`` that has lost ``System32`` — gprMax crashes exactly as it did
        before the fix. Pinned as the current behaviour; written up in
        ``notes/bugs/host-info-powershell-fallback-filenotfound.md``.
        """
        windows_host.register(
            ["wmic", "csproduct", "get", "vendor"], FileNotFoundError("wmic")
        )
        windows_host.register(
            powershell_commands["vendor"], FileNotFoundError("powershell")
        )

        with pytest.raises(FileNotFoundError):
            get_host_info()

    def test_no_cpu_information_at_all_leaves_the_socket_count_at_zero(
        self, windows_host, powershell_commands
    ):
        """``sockets`` is reset to ``0`` before the loop, not left ``"unknown"``.

        The banner then reads ``0 x unknown``, which is at least honest.
        """
        windows_host.register(
            ["wmic", "cpu", "get", "Name"], FileNotFoundError("wmic")
        )
        windows_host.register(
            powershell_commands["cpu"], subprocess.CalledProcessError(1, "powershell")
        )

        hostinfo = get_host_info()

        assert (hostinfo["sockets"], hostinfo["cpuID"]) == (0, "unknown")


class TestMacOs:
    """The ``darwin`` branch — three ``sysctl`` calls."""

    def test_the_manufacturer_is_hard_coded(self, macos_host):
        """No probe; Apple hardware has exactly one vendor."""
        assert get_host_info()["machineID"].startswith("Apple")

    def test_the_model_comes_from_sysctl(self, macos_host):
        assert get_host_info()["machineID"] == "Apple TestMac1,1"

    def test_the_socket_count_is_an_integer(self, macos_host):
        """``sysctl`` returns text; the conversion is what makes it usable."""
        hostinfo = get_host_info()

        assert hostinfo["sockets"] == 2 and isinstance(hostinfo["sockets"], int)

    def test_the_cpu_name_comes_from_the_brand_string(self, macos_host):
        assert get_host_info()["cpuID"] == "Test CPU @ 1.00GHz"

    def test_internal_whitespace_in_the_cpu_name_is_collapsed(self, macos_host):
        macos_host.register(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            b"Test    CPU   @  1.00GHz\n",
        )

        assert get_host_info()["cpuID"] == "Test CPU @ 1.00GHz"

    def test_the_os_version_reports_the_mac_release(self, macos_host):
        assert get_host_info()["osversion"] == "macOS (14.0)"

    def test_a_failing_model_probe_leaves_the_field_unknown(self, macos_host):
        macos_host.register(
            ["sysctl", "-n", "hw.model"], subprocess.CalledProcessError(1, "sysctl")
        )

        assert get_host_info()["machineID"] == "Apple unknown"

    def test_a_failing_cpu_probe_leaves_both_cpu_fields_unknown(self, macos_host):
        """The two probes share one ``try``, so one failure loses both."""
        macos_host.register(
            ["sysctl", "-n", "hw.packages"],
            subprocess.CalledProcessError(1, "sysctl"),
        )

        hostinfo = get_host_info()

        assert (hostinfo["sockets"], hostinfo["cpuID"]) == ("unknown", "unknown")

    def test_apple_silicon_has_no_brand_string(self, macos_host):
        """``machdep.cpu.brand_string`` does not exist on M-series chips.

        ``sysctl`` exits non-zero, so ``cpuID`` stays ``"unknown"`` on exactly
        the hardware gprMax users are most likely to be running today — and
        ``set_omp_threads`` then takes the ``ACTIVE`` wait-policy branch,
        which Apple's own tuning guide advises against for Apple silicon.
        Pinned as the current behaviour; written up in
        ``notes/bugs/host-info-apple-silicon-cpuid.md``.
        """
        macos_host.register(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            subprocess.CalledProcessError(1, "sysctl"),
        )

        assert get_host_info()["cpuID"] == "unknown"

    def test_a_missing_sysctl_is_not_caught(self, macos_host):
        """The same blind spot the wmic fix closed, still open here.

        ``except subprocess.CalledProcessError`` without ``FileNotFoundError``
        — one of four surviving instances outside the Windows branch. Pinned
        as the current behaviour; written up in
        ``notes/bugs/host-info-remaining-filenotfound-blind-spots.md``.
        """
        macos_host.register(["sysctl", "-n", "hw.model"], FileNotFoundError("sysctl"))

        with pytest.raises(FileNotFoundError):
            get_host_info()


class TestLinux:
    """The ``linux`` branch — DMI files, ``/proc/cpuinfo`` and ``lscpu``."""

    def test_the_manufacturer_comes_from_the_dmi_tree(self, linux_host):
        assert get_host_info()["machineID"].startswith("Test Manufacturer")

    def test_the_model_comes_from_the_dmi_tree(self, linux_host):
        assert get_host_info()["machineID"] == "Test Manufacturer Test Model"

    def test_the_cpu_name_comes_from_proc_cpuinfo(self, linux_host):
        assert get_host_info()["cpuID"] == "Test CPU @ 1.00GHz"

    def test_the_last_model_name_line_wins(self, linux_host):
        """The loop assigns without breaking, so it ends on the final core.

        Harmless on a homogeneous machine, which is every machine gprMax runs
        on, but worth pinning: it is not the *first* match.
        """
        linux_host.register(
            ["cat", "/proc/cpuinfo"],
            b"model name\t: First CPU\nmodel name\t: Last CPU\n",
        )

        assert get_host_info()["cpuID"] == "Last CPU"

    def test_the_socket_count_comes_from_lscpu(self, linux_host):
        assert get_host_info()["sockets"] == 2

    def test_hyperthreading_is_two_threads_per_core(self, linux_host):
        """Unlike the other two branches, this one ignores ``psutil``."""
        assert get_host_info()["hyperthreading"] is True

    def test_one_thread_per_core_means_no_hyperthreading(self, linux_host):
        linux_host.register(
            ["lscpu"], b"Thread(s) per core:  1\nSocket(s):           2\n"
        )

        assert get_host_info()["hyperthreading"] is False

    def test_the_os_version_is_the_platform_string(self, linux_host):
        assert get_host_info()["osversion"] == "Linux-6.0.0-x86_64"

    def test_the_locale_is_forced_to_english(self, linux_host, monkeypatch):
        """``lscpu`` translates its labels, and the parser matches on English.

        Asserted through the environment handed to the subprocess rather than
        through the output, since the fake cannot speak French.
        """
        captured = {}
        original = linux_host.__call__

        def spy(argv, *args, **kwargs):
            captured[tuple(argv)] = kwargs.get("env")
            return original(argv, *args, **kwargs)

        monkeypatch.setattr(subprocess, "check_output", spy)
        get_host_info()

        assert captured[("lscpu",)]["LANG"] == "en_US.utf8"

    def test_a_two_digit_socket_count_is_misread(self, linux_host):
        """``int(line.strip()[-1])`` reads only the **last character**.

        ``"Socket(s):  12"`` therefore parses as ``2``. The banner
        under-reports, and on a large multi-socket node — precisely the
        machines gprMax is run on — the figure is silently wrong. Pinned as
        the current behaviour; written up in
        ``notes/bugs/host-info-lscpu-last-character-parse.md``.
        """
        linux_host.register(
            ["lscpu"], b"Socket(s):           12\nThread(s) per core:  2\n"
        )

        assert get_host_info()["sockets"] == 2

    def test_a_failing_dmi_read_leaves_the_machine_id_unknown(self, linux_host):
        """Reading the DMI tree needs root on some distributions."""
        linux_host.register(
            ["cat", "/sys/class/dmi/id/sys_vendor"],
            subprocess.CalledProcessError(1, "cat"),
        )

        assert get_host_info()["machineID"] == "unknown unknown"

    def test_a_failing_lscpu_leaves_the_socket_count_unknown(self, linux_host):
        """``lscpu`` is not installed in every minimal container image."""
        linux_host.register(["lscpu"], subprocess.CalledProcessError(1, "lscpu"))

        assert get_host_info()["sockets"] == "unknown"


class TestAnUnrecognisedPlatform:
    """No ``else`` on the platform chain."""

    def test_an_unknown_platform_raises_an_unbound_local_error(
        self, monkeypatch, fake_subprocess, fake_platform, fake_cpu_counts
    ):
        """FreeBSD, Cygwin and AIX all reach the end with nothing assigned.

        ``machineID``, ``hyperthreading`` and ``osversion`` are only bound
        inside the three branches, so the dictionary construction raises
        ``UnboundLocalError`` — a failure that names a local variable rather
        than the unsupported platform. This is the same missing-terminal-``else``
        pattern found four other times in this PR's scope. Pinned as the
        current behaviour; written up in
        ``notes/bugs/host-info-no-terminal-else.md``.
        """
        monkeypatch.setattr("sys.platform", "freebsd13")
        fake_cpu_counts()

        with pytest.raises(UnboundLocalError):
            get_host_info()


class TestPrintHostInfo:
    """``print_host_info`` — the one line the user actually reads."""

    def test_the_hostname_is_printed(self, install_host_config, caplog):
        caplog.set_level(1)
        hostinfo = install_host_config().hostinfo

        print_host_info(hostinfo)

        assert "test-host" in caplog.text

    def test_the_machine_id_is_printed(self, install_host_config, caplog):
        caplog.set_level(1)
        hostinfo = install_host_config().hostinfo

        print_host_info(hostinfo)

        assert "Test Manufacturer Test Model" in caplog.text

    def test_the_socket_count_and_cpu_are_printed_together(
        self, install_host_config, caplog
    ):
        caplog.set_level(1)
        hostinfo = install_host_config().hostinfo

        print_host_info(hostinfo)

        assert "2 x Test CPU @ 1.00GHz" in caplog.text

    def test_the_memory_is_printed_in_human_units(
        self, install_host_config, caplog
    ):
        """16 GiB, not 17179869184."""
        caplog.set_level(1)
        hostinfo = install_host_config().hostinfo

        print_host_info(hostinfo)

        assert "16.0 GiB" in caplog.text

    def test_hyperthreading_is_mentioned_when_present(
        self, install_host_config, caplog
    ):
        caplog.set_level(1)
        hostinfo = install_host_config(hyperthreading=True).hostinfo

        print_host_info(hostinfo)

        assert "Hyper-Threading" in caplog.text

    def test_hyperthreading_is_not_mentioned_when_absent(
        self, install_host_config, caplog
    ):
        caplog.set_level(1)
        hostinfo = install_host_config(hyperthreading=False).hostinfo

        print_host_info(hostinfo)

        assert "Hyper-Threading" not in caplog.text

    def test_the_thread_count_line_reports_the_physical_cores(
        self, install_host_config, caplog
    ):
        """OpenMP is sized by physical cores, and the banner says so."""
        caplog.set_level(1)
        hostinfo = install_host_config().hostinfo

        print_host_info(hostinfo)

        assert "|--->OpenMP: 6 threads" in caplog.text

    def test_it_logs_at_the_basic_level(self, install_host_config, caplog):
        """Level 25, so ``--log-level 30`` suppresses the banner."""
        caplog.set_level(1)
        hostinfo = install_host_config().hostinfo

        print_host_info(hostinfo)

        assert {record.levelname for record in caplog.records} == {"BASIC"}

    def test_it_emits_exactly_two_lines(self, install_host_config, caplog):
        caplog.set_level(1)
        hostinfo = install_host_config().hostinfo

        print_host_info(hostinfo)

        assert len(caplog.records) == 2

    def test_the_first_two_fields_come_from_the_global_not_the_argument(
        self, install_host_config, caplog
    ):
        """A genuine inconsistency, pinned so it is not "fixed" by accident.

        ``hostname`` and ``machineID`` are read from
        ``config.sim_config.hostinfo``, while every other field comes from the
        ``hostinfo`` argument. Passing a different dictionary — as a caller
        reasonably might, to describe a remote node — silently mixes the two.
        Written up in ``notes/bugs/host-info-print-mixes-sources.md``.
        """
        caplog.set_level(1)
        install_host_config()
        other = {
            "hostname": "ignored",
            "machineID": "ignored",
            "sockets": 4,
            "cpuID": "Other CPU",
            "osversion": "Other OS",
            "hyperthreading": False,
            "logicalcores": 4,
            "physicalcores": 4,
            "ram": 1024,
        }

        print_host_info(other)

        assert "test-host" in caplog.text and "Other CPU" in caplog.text
