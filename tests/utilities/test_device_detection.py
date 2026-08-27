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

"""Finding, describing and reporting compute devices.

gprMax can solve on a CPU, on CUDA through pycuda, on OpenCL through pyopencl,
or on Apple Metal through pyobjc. Three of those four backends are optional
imports, and the code that finds them follows the same shape each time: a
``has_*`` predicate that swallows ``ImportError``, a ``detect_*`` that returns
a dictionary of device objects keyed by ID, and a ``print_*_info`` that turns
that dictionary into the indented tree the user sees at startup.

**The testing problem, and the way round it.** None of the three packages is
installed in the test environment, and installing them would not help: pycuda
imports fine on a machine with no NVIDIA card and then fails at
``drv.init()``, while pyopencl needs an ICD loader. So the modules are
*fabricated* — ``fake_module`` inserts a ``ModuleType`` into ``sys.modules``,
and the ``import pycuda.driver as drv`` inside the function under test picks
it up like any other import. That is what makes the success paths reachable
at all; without it only the "not installed" branch could ever be tested, which
is the branch that matters least.

The fabricated modules are deliberately minimal — a device object here has a
``name`` and a memory figure and nothing else, because that is all the code
touches. Anything more would be inventing an API contract that the real
libraries define.

Two of the tests pin defects. ``print_opencl_info`` leaves a local unbound for
any device that is neither CPU nor GPU, and ``detect_metal`` stores a ``None``
device when the Metal framework reports no hardware. Both are silent in the
common case, which is exactly why they need writing down.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from gprMax.utilities.host_info import (
    detect_cuda_gpus,
    detect_metal,
    detect_opencl,
    has_metal,
    has_pycuda,
    has_pyopencl,
    print_cuda_info,
    print_metal_info,
    print_opencl_info,
)


@pytest.fixture
def fake_module(monkeypatch):
    """Install a fabricated module into ``sys.modules``.

    ``monkeypatch.setitem`` removes it afterwards, so no later test — in this
    file or any other — sees a ``pycuda`` that is not really there.

    Submodules are registered under their dotted name *and* as an attribute of
    the parent, because ``import a.b as c`` consults both.
    """

    def _install(name, **attributes):
        module = ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, _, child = name.rpartition(".")
            monkeypatch.setattr(sys.modules[parent], child, module, raising=False)
        return module

    return _install


@pytest.fixture
def hide_module(monkeypatch):
    """Force an ``ImportError`` for a module that may genuinely be installed.

    A developer with pycuda on their machine would otherwise see different
    results from CI on the "not installed" tests.
    """

    def _hide(name):
        monkeypatch.setitem(sys.modules, name, None)

    return _hide


def _cuda_device(name="Test GPU", memory=8 * 1024**3):
    """A stand-in pycuda device: a callable name and a callable memory size."""
    return SimpleNamespace(name=lambda: name, total_memory=lambda: memory)


def _opencl_device(
    name="Test Device", memory=8 * 1024**3, platform="Test Platform", device_type=2
):
    """A stand-in pyopencl device: attributes, not methods, unlike pycuda."""
    return SimpleNamespace(
        name=name,
        global_mem_size=memory,
        type=device_type,
        platform=SimpleNamespace(name=platform),
    )


class TestTheAvailabilityPredicates:
    """``has_pycuda`` / ``has_pyopencl`` / ``has_metal`` — importable or not."""

    @pytest.mark.parametrize(
        "predicate, module",
        [(has_pycuda, "pycuda"), (has_pyopencl, "pyopencl"), (has_metal, "Metal")],
    )
    def test_an_installed_module_is_reported_present(self, fake_module, predicate, module):
        """Truthiness, not identity — see the shadowing test below."""
        fake_module(module)

        assert predicate()

    def test_two_of_the_three_return_the_module_instead_of_true(self, fake_module):
        """``import pycuda`` rebinds the local that held ``True``.

        Each predicate sets a local to ``True``, then does ``import <name>``
        *inside the same function* — and the import statement binds the module
        to that same name. So ``has_pycuda`` and ``has_pyopencl`` return a
        module object on success. ``has_metal`` escapes only by accident: its
        local is ``metal`` and the module is ``Metal``, and the case differs.

        Harmless today, because every caller uses the result in an ``if``. It
        matters the moment one is compared with ``True`` or serialised.
        Written up in ``notes/bugs/host-info-has-predicates-shadowing.md``.
        """
        fake_module("pycuda")
        fake_module("pyopencl")
        fake_module("Metal")

        assert isinstance(has_pycuda(), ModuleType)
        assert isinstance(has_pyopencl(), ModuleType)
        assert has_metal() is True

    @pytest.mark.parametrize(
        "predicate, module",
        [(has_pycuda, "pycuda"), (has_pyopencl, "pyopencl"), (has_metal, "Metal")],
    )
    def test_a_missing_module_is_reported_absent(self, hide_module, predicate, module):
        hide_module(module)

        assert predicate() is False

    @pytest.mark.parametrize("predicate", [has_pycuda, has_pyopencl, has_metal])
    def test_a_missing_module_does_not_raise(self, hide_module, predicate):
        """The whole point — these are called unconditionally at startup.

        A user with none of the three optional backends must still be able to
        run gprMax on a CPU.
        """
        for name in ("pycuda", "pyopencl", "Metal"):
            hide_module(name)

        assert predicate() is False

    @pytest.mark.parametrize("predicate", [has_pycuda, has_pyopencl, has_metal])
    def test_the_result_is_a_boolean(self, hide_module, predicate):
        """Callers use it in ``if``; a truthy module object would also work,
        but the annotation-free signature makes the type worth pinning.
        """
        hide_module("pycuda")
        hide_module("pyopencl")
        hide_module("Metal")

        assert isinstance(predicate(), bool)


class TestDetectCudaGpus:
    """``detect_cuda_gpus`` — pycuda's device list, keyed by device ID."""

    @pytest.fixture
    def cuda(self, fake_module, monkeypatch):
        """A fabricated pycuda with a configurable device count."""
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        parent = fake_module("pycuda", VERSION_TEXT="2024.1")

        def _install(count=2):
            devices = {ID: _cuda_device(f"Test GPU {ID}") for ID in range(count)}
            driver = fake_module(
                "pycuda.driver",
                init=lambda: None,
                Device=SimpleNamespace(count=lambda: count, __call__=None),
            )
            driver.Device = lambda ID: devices[ID]
            driver.Device.count = staticmethod(lambda: count)
            return devices

        _install.parent = parent
        return _install

    def test_no_pycuda_returns_an_empty_dictionary(self, hide_module):
        hide_module("pycuda")

        assert detect_cuda_gpus() == {}

    def test_no_pycuda_warns_with_installation_instructions(self, hide_module, caplog):
        """The user asked for ``-gpu``; silence would be unhelpful."""
        hide_module("pycuda")

        detect_cuda_gpus()

        assert "install pycuda" in caplog.text

    def test_every_device_is_returned(self, cuda):
        cuda(count=2)

        assert len(detect_cuda_gpus()) == 2

    def test_devices_are_keyed_by_their_identifier(self, cuda):
        cuda(count=3)

        assert sorted(detect_cuda_gpus()) == [0, 1, 2]

    def test_the_driver_is_initialised(self, cuda, fake_module):
        """``drv.init()`` must run before any other pycuda call."""
        cuda(count=1)
        calls = []
        sys.modules["pycuda.driver"].init = lambda: calls.append("init")

        detect_cuda_gpus()

        assert calls == ["init"]

    def test_no_devices_warns(self, cuda, caplog):
        """pycuda installed but no card — a common misconfiguration."""
        cuda(count=0)

        detect_cuda_gpus()

        assert "No NVIDIA CUDA-Enabled GPUs detected" in caplog.text

    def test_no_devices_returns_an_empty_dictionary(self, cuda):
        cuda(count=0)

        assert detect_cuda_gpus() == {}

    def test_the_visible_devices_variable_restricts_the_list(self, cuda, monkeypatch):
        """Schedulers set ``CUDA_VISIBLE_DEVICES``; gprMax must honour it.

        Ignoring it on a shared node would mean grabbing another job's GPU.
        """
        cuda(count=4)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")

        assert sorted(detect_cuda_gpus()) == [1, 2]

    def test_a_single_visible_device_is_parsed(self, cuda, monkeypatch):
        cuda(count=4)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

        assert sorted(detect_cuda_gpus()) == [3]

    def test_the_visible_devices_variable_is_ignored_when_there_are_none(
        self, cuda, monkeypatch, caplog
    ):
        """The zero-device check comes first, so the warning still fires."""
        cuda(count=0)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

        detect_cuda_gpus()

        assert "No NVIDIA CUDA-Enabled GPUs detected" in caplog.text


class TestDetectOpencl:
    """``detect_opencl`` — a flat dictionary across all platforms."""

    @pytest.fixture
    def opencl(self, fake_module):
        def _install(platforms):
            fake_module(
                "pyopencl",
                VERSION_TEXT="2024.1",
                get_platforms=lambda: platforms,
                device_type=SimpleNamespace(
                    to_string=lambda value: {1: "CPU", 2: "GPU"}.get(value, "ACCELERATOR")
                ),
            )

        return _install

    @staticmethod
    def _platform(*devices):
        return SimpleNamespace(get_devices=lambda: list(devices))

    def test_no_pyopencl_returns_an_empty_dictionary(self, hide_module):
        hide_module("pyopencl")

        assert detect_opencl() == {}

    def test_no_pyopencl_warns_with_installation_instructions(self, hide_module, caplog):
        hide_module("pyopencl")

        detect_opencl()

        assert "install pyopencl" in caplog.text

    def test_every_device_is_returned(self, opencl):
        opencl([self._platform(_opencl_device(), _opencl_device())])

        assert len(detect_opencl()) == 2

    def test_devices_are_numbered_from_zero(self, opencl):
        opencl([self._platform(_opencl_device(), _opencl_device())])

        assert sorted(detect_opencl()) == [0, 1]

    def test_devices_across_platforms_share_one_numbering(self, opencl):
        """A machine with an integrated GPU and a discrete one has two
        platforms; the user selects a device by a single ID, not a pair.
        """
        opencl(
            [
                self._platform(_opencl_device("First")),
                self._platform(_opencl_device("Second")),
            ]
        )

        devices = detect_opencl()

        assert [devices[0].name, devices[1].name] == ["First", "Second"]

    def test_no_platforms_returns_an_empty_dictionary(self, opencl):
        opencl([])

        assert detect_opencl() == {}

    def test_a_failing_platform_query_warns(self, opencl, fake_module, caplog):
        """A bare ``except`` — an ICD loader that raises is not fatal."""
        fake_module(
            "pyopencl",
            get_platforms=lambda: (_ for _ in ()).throw(RuntimeError("no ICD")),
        )

        detect_opencl()

        assert "No OpenCL-capable platforms detected" in caplog.text

    def test_a_failing_platform_query_returns_an_empty_dictionary(self, fake_module):
        fake_module(
            "pyopencl",
            get_platforms=lambda: (_ for _ in ()).throw(RuntimeError("no ICD")),
        )

        assert detect_opencl() == {}


class TestDetectMetal:
    """``detect_metal`` — one device, or a placeholder for one."""

    def test_no_metal_returns_an_empty_dictionary(self, hide_module):
        hide_module("Metal")

        assert detect_metal() == {}

    def test_no_metal_warns_with_installation_instructions(self, hide_module, caplog):
        hide_module("Metal")

        detect_metal()

        assert "install pyobjc" in caplog.text

    def test_the_system_default_device_is_returned(self, fake_module):
        device = SimpleNamespace(name=lambda: "Apple M2")
        fake_module("Metal", MTLCreateSystemDefaultDevice=lambda: device)

        assert detect_metal() == {0: device}

    def test_it_is_keyed_at_zero(self, fake_module):
        """Metal exposes one system default; there is no device list."""
        fake_module("Metal", MTLCreateSystemDefaultDevice=lambda: SimpleNamespace())

        assert list(detect_metal()) == [0]

    def test_no_hardware_still_produces_an_entry(self, fake_module):
        """``MTLCreateSystemDefaultDevice`` returns ``None`` with no GPU.

        Upstream now guards ``devs[0] = device`` with ``if device is not None``,
        so a ``None`` device produces an empty dictionary rather than the
        previously-pinned ``{0: None}``.
        """
        fake_module("Metal", MTLCreateSystemDefaultDevice=lambda: None)

        assert detect_metal() == {}


class TestPrintCudaInfo:
    """``print_cuda_info`` — one line per GPU, under a heading."""

    @pytest.fixture(autouse=True)
    def pycuda(self, fake_module):
        fake_module("pycuda", VERSION_TEXT="2024.1")

    def test_a_heading_is_printed(self, caplog):
        caplog.set_level(1)

        print_cuda_info({})

        assert "|--->CUDA:" in caplog.text

    def test_the_device_name_is_printed(self, caplog):
        caplog.set_level(1)

        print_cuda_info({0: _cuda_device("Test GPU")})

        assert "Test GPU" in caplog.text

    def test_the_device_identifier_is_printed(self, caplog):
        """The number the user passes to ``-gpu``."""
        caplog.set_level(1)

        print_cuda_info({3: _cuda_device()})

        assert "Device 3" in caplog.text

    def test_the_memory_is_humanised(self, caplog):
        caplog.set_level(1)

        print_cuda_info({0: _cuda_device(memory=8 * 1024**3)})

        assert "8.0 GiB" in caplog.text

    def test_internal_whitespace_in_the_name_is_collapsed(self, caplog):
        """Driver-reported names are padded."""
        caplog.set_level(1)

        print_cuda_info({0: _cuda_device("Test    GPU")})

        assert "Test GPU" in caplog.text

    def test_one_line_per_device(self, caplog):
        caplog.set_level(1)

        print_cuda_info({0: _cuda_device(), 1: _cuda_device()})

        assert len(caplog.records) == 3  # heading + two device lines (no count when few)

    def test_an_empty_dictionary_prints_only_the_heading(self, caplog):
        caplog.set_level(1)

        print_cuda_info({})

        assert len(caplog.records) == 1  # upstream: no count line when no devices


class TestPrintOpenclInfo:
    """``print_opencl_info`` — devices grouped under their platform."""

    @pytest.fixture(autouse=True)
    def pyopencl(self, fake_module):
        fake_module(
            "pyopencl",
            VERSION_TEXT="2024.1",
            device_type=SimpleNamespace(
                to_string=lambda value: {1: "CPU", 2: "GPU"}.get(value, "ACCELERATOR")
            ),
        )

    def test_a_heading_is_printed(self, caplog):
        caplog.set_level(1)

        print_opencl_info({})

        assert "|--->OpenCL:" in caplog.text

    def test_the_platform_name_is_printed(self, caplog):
        caplog.set_level(1)

        print_opencl_info({0: _opencl_device(platform="Test Platform")})

        assert "Platform: Test Platform" in caplog.text

    def test_the_device_type_is_printed(self, caplog):
        caplog.set_level(1)

        print_opencl_info({0: _opencl_device(device_type=2)})

        assert "GPU" in caplog.text

    def test_a_cpu_device_is_labelled_as_such(self, caplog):
        """OpenCL on a CPU is a supported gprMax configuration."""
        caplog.set_level(1)

        print_opencl_info({0: _opencl_device(device_type=1)})

        assert "CPU" in caplog.text

    def test_the_device_name_and_memory_are_printed(self, caplog):
        caplog.set_level(1)

        print_opencl_info({0: _opencl_device(name="Test Device", memory=4 * 1024**3)})

        assert "Test Device" in caplog.text and "4.0 GiB" in caplog.text

    def test_one_platform_heading_for_several_devices_on_it(self, caplog):
        """The grouping is the reason the loop tracks the previous platform."""
        caplog.set_level(1)

        print_opencl_info(
            {
                0: _opencl_device(name="A", platform="Same"),
                1: _opencl_device(name="B", platform="Same"),
            }
        )

        assert caplog.text.count("|--->Platform:") == 1

    def test_a_second_platform_gets_its_own_heading(self, caplog):
        caplog.set_level(1)

        print_opencl_info(
            {
                0: _opencl_device(name="A", platform="First"),
                1: _opencl_device(name="B", platform="Second"),
            }
        )

        assert caplog.text.count("|--->Platform:") == 2

    def test_an_empty_dictionary_prints_only_the_heading(self, caplog):
        caplog.set_level(1)

        print_opencl_info({})

        assert len(caplog.records) == 1  # upstream: no count line when no devices

    def test_a_device_that_is_neither_cpu_nor_gpu_uses_its_reported_type(self, caplog):
        """Accelerators are reported without assuming they are CPUs or GPUs."""
        caplog.set_level(1)

        print_opencl_info({0: _opencl_device(device_type=8)})

        assert "ACCELERATOR" in caplog.text


class TestPrintMetalInfo:
    """``print_metal_info`` — the shortest of the three."""

    def test_a_heading_is_printed(self, caplog):
        caplog.set_level(1)

        print_metal_info({})

        assert "|--->Apple Metal:" in caplog.text

    def test_the_device_name_is_printed(self, caplog):
        caplog.set_level(1)

        print_metal_info({0: SimpleNamespace(name=lambda: "Apple M2 Pro")})

        assert "Apple M2 Pro" in caplog.text

    def test_the_device_identifier_is_printed(self, caplog):
        caplog.set_level(1)

        print_metal_info({0: SimpleNamespace(name=lambda: "Apple M2")})

        assert "Device 0" in caplog.text

    def test_internal_whitespace_in_the_name_is_collapsed(self, caplog):
        caplog.set_level(1)

        print_metal_info({0: SimpleNamespace(name=lambda: "Apple    M2")})

        assert "Apple M2" in caplog.text

    def test_no_memory_figure_is_reported(self, caplog):
        """Unlike CUDA and OpenCL — Metal uses unified memory, so a
        per-device figure would repeat the host's RAM.
        """
        caplog.set_level(1)

        print_metal_info({0: SimpleNamespace(name=lambda: "Apple M2")})

        assert "GiB" not in caplog.text

    def test_it_logs_at_the_basic_level(self, caplog):
        caplog.set_level(1)

        print_metal_info({0: SimpleNamespace(name=lambda: "Apple M2")})

        assert {record.levelname for record in caplog.records} == {"BASIC"}


pytestmark = pytest.mark.unit
