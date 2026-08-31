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

"""Smoke test executed from an installed binary wheel, outside the source tree."""

from __future__ import annotations

import importlib
import importlib.machinery
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="gprmax-matplotlib-"))

import gprMax
from gprMax.examples import copy_examples, list_examples

CYTHON_MODULES = (
    "eigenmode_dft",
    "eigenmode_source",
    "fields_updates_dispersive",
    "fields_updates_hsg",
    "fields_updates_normal",
    "fractals_generate",
    "geometry_outputs",
    "geometry_primitives",
    "impedance_surface",
    "network_port",
    "ntff",
    "plane_wave",
    "pml_build",
    "pml_updates_electric_HORIPML",
    "pml_updates_electric_MRIPML",
    "pml_updates_magnetic_HORIPML",
    "pml_updates_magnetic_MRIPML",
    "sar_averaging",
    "snapshots",
    "symmetry_boundaries",
    "symmetry_boundaries_dispersive",
    "symmetry_boundaries_dispersive_complex",
    "virtual_waveguide",
    "yee_cell_build",
    "yee_cell_setget_rigid",
)


def _assert_compiled_extensions_load() -> None:
    extension_suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    for name in CYTHON_MODULES:
        module = importlib.import_module(f"gprMax.cython.{name}")
        assert module.__file__ is not None
        assert module.__file__.endswith(extension_suffixes), module.__file__


def _assert_only_wheel_payload_is_installed() -> None:
    package = Path(gprMax.__file__).resolve().parent
    cython_dir = package / "cython"
    assert not list(cython_dir.glob("*.pyx"))
    assert not list(cython_dir.glob("*.c"))
    assert not list(package.rglob("*.pxd"))


def _assert_examples_are_available(workspace: Path) -> None:
    categories = dict(list_examples())
    assert categories.get("gpr", 0) > 0
    assert categories.get("antennas", 0) > 0

    copied = copy_examples(workspace)
    assert (copied / "examples" / "gpr" / "basic" / "cylinder_Ascan_2D.in").is_file()


def _assert_matlab_utilities_are_available() -> None:
    toolboxes = Path(importlib.import_module("toolboxes").__file__).resolve().parent
    matlab = toolboxes / "Utilities" / "MATLAB"
    assert (matlab / "gprmax_read_h5.m").is_file()
    assert (matlab / "gprmax_h5_to_mat.m").is_file()
    assert (matlab / "gprmax_h5_get.m").is_file()
    assert (matlab / "plot_Ascan.m").is_file()
    assert (matlab / "plot_Bscan.m").is_file()


def _run_tiny_cpu_model(output: Path) -> None:
    scene = gprMax.Scene()
    scene.add(gprMax.Discretisation(p1=(0.001, 0.001, 0.001)))
    scene.add(gprMax.Domain(p1=(0.01, 0.01, 0.01)))
    scene.add(gprMax.PMLThickness(thickness=0))
    scene.add(gprMax.TimeWindow(iterations=300))
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=5e9, id="pulse"))
    scene.add(gprMax.HertzianDipole(p1=(0.005, 0.005, 0.005), polarisation="z", waveform_id="pulse"))
    scene.add(gprMax.Rx(p1=(0.006, 0.005, 0.005), outputs=["Ez"]))

    gprMax.run(
        scenes=[scene],
        outputfile=str(output),
        hide_progress_bars=True,
        log_level=30,
    )
    assert output.with_suffix(".h5").is_file()


def main() -> None:
    _assert_compiled_extensions_load()
    _assert_only_wheel_payload_is_installed()
    _assert_matlab_utilities_are_available()
    with tempfile.TemporaryDirectory(prefix="gprmax-wheel-") as directory:
        root = Path(directory)
        _assert_examples_are_available(root / "workspace")
        _run_tiny_cpu_model(root / "smoke")


if __name__ == "__main__":
    main()
