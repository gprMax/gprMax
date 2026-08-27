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

"""Tests for portable extension compiler settings."""

from pathlib import Path

import pytest

from build_config import build_jobs, compiler_configuration, find_libomp_prefix


def _fake_libomp(prefix: Path) -> Path:
    (prefix / "include").mkdir(parents=True)
    (prefix / "lib").mkdir(parents=True)
    (prefix / "include" / "omp.h").touch()
    (prefix / "lib" / "libomp.dylib").touch()
    return prefix


@pytest.mark.unit
def test_build_jobs_default_is_conservative(monkeypatch):
    monkeypatch.setattr("build_config.os.cpu_count", lambda: 64)

    assert build_jobs(environ={}) == 2


@pytest.mark.unit
def test_build_jobs_accepts_positive_override():
    assert build_jobs(environ={"GPRMAX_BUILD_JOBS": "4"}) == 4


@pytest.mark.unit
@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_build_jobs_rejects_invalid_override(value):
    with pytest.raises(ValueError, match="positive integer"):
        build_jobs(environ={"GPRMAX_BUILD_JOBS": value})


@pytest.mark.unit
def test_linux_release_flags_are_portable():
    config = compiler_configuration(platform_name="linux", environ={})

    assert config.compile_args == ("-O3", "-w", "-fopenmp")
    assert config.linker_args == ("-fopenmp",)
    assert "-march=native" not in config.compile_args


@pytest.mark.unit
def test_linux_native_optimisation_requires_explicit_opt_in():
    config = compiler_configuration(
        platform_name="linux",
        environ={"GPRMAX_BUILD_NATIVE": "yes"},
    )

    assert config.compile_args[-1] == "-march=native"


@pytest.mark.unit
def test_windows_uses_msvc_openmp_without_host_specific_flags():
    config = compiler_configuration(platform_name="win32", environ={})

    assert config.compile_args == ("/O2", "/openmp", "/w")
    assert config.linker_args == ()


@pytest.mark.unit
def test_windows_rejects_unsupported_native_build_request():
    with pytest.raises(ValueError, match="not supported"):
        compiler_configuration(
            platform_name="win32",
            environ={"GPRMAX_BUILD_NATIVE": "1"},
        )


@pytest.mark.unit
def test_macos_uses_clang_and_libomp(tmp_path):
    prefix = _fake_libomp(tmp_path / "libomp")
    config = compiler_configuration(
        platform_name="darwin",
        machine="arm64",
        environ={},
        libomp_prefix=prefix,
    )

    assert config.compile_args == ("-O3", "-w", "-Xpreprocessor", "-fopenmp")
    assert config.libraries == ("omp",)
    assert config.include_dirs == (str(prefix / "include"),)
    assert config.library_dirs == (str(prefix / "lib"),)
    assert config.linker_args == (f"-Wl,-rpath,{prefix / 'lib'}",)
    assert not any("macosx-version-min" in arg for arg in config.compile_args + config.linker_args)


@pytest.mark.unit
def test_macos_native_arm_build_is_explicit(tmp_path):
    prefix = _fake_libomp(tmp_path / "libomp")
    config = compiler_configuration(
        platform_name="darwin",
        machine="arm64",
        environ={"GPRMAX_BUILD_NATIVE": "true"},
        libomp_prefix=prefix,
    )

    assert config.compile_args[-1] == "-mcpu=native"


@pytest.mark.unit
def test_macos_reports_actionable_missing_libomp_error(tmp_path):
    with pytest.raises(RuntimeError, match="brew install libomp"):
        compiler_configuration(
            platform_name="darwin",
            environ={},
            libomp_prefix=tmp_path / "missing",
        )


@pytest.mark.unit
def test_explicit_libomp_prefix_takes_priority(tmp_path, monkeypatch):
    configured = _fake_libomp(tmp_path / "configured")
    monkeypatch.setattr(
        "build_config.shutil.which",
        lambda executable: pytest.fail(f"unexpected lookup for {executable}"),
    )

    assert (
        find_libomp_prefix(
            environ={"GPRMAX_LIBOMP_PREFIX": str(configured)},
            python_prefix=tmp_path / "python",
        )
        == configured.resolve()
    )
