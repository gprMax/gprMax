# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, and John Hartley
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

import glob
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from jinja2 import Environment, FileSystemLoader
from setuptools import Extension, find_packages, setup

# Check Python version
MIN_PYTHON_VERSION = (3, 7)
if sys.version_info[:2] < MIN_PYTHON_VERSION:
    sys.exit("\nExited: Requires Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} or newer!\n")

# Importing gprMax _version__.py before building can cause issues.
with open("gprMax/_version.py", "r") as fd:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', fd.read(), re.MULTILINE)[1]


def build_dispersive_material_templates():
    """Function to generate Cython .pyx file for dispersive media update.
    Jinja2 templates are used to render the various dispersive update
    functions.
    """

    iswin = sys.platform == "win32"

    env = Environment(
        loader=FileSystemLoader(os.path.join("gprMax", "cython")),
    )

    template = env.get_template("fields_updates_dispersive_template.jinja")

    # Render dispersive template for different types
    r = template.render(
        functions=[
            # templates for Double precision and dispersive materials with
            # real susceptibility functions
            {
                "name_a": "update_electric_dispersive_multipole_A_double_real",
                "name_b": "update_electric_dispersive_multipole_B_double_real",
                "name_a_1": "update_electric_dispersive_1pole_A_double_real",
                "name_b_1": "update_electric_dispersive_1pole_B_double_real",
                "field_type": "double",
                "dispersive_type": "double",
                "iswin": iswin,
            },
            # templates for Float precision and dispersive materials with
            # real susceptibility functions
            {
                "name_a": "update_electric_dispersive_multipole_A_float_real",
                "name_b": "update_electric_dispersive_multipole_B_float_real",
                "name_a_1": "update_electric_dispersive_1pole_A_float_real",
                "name_b_1": "update_electric_dispersive_1pole_B_float_real",
                "field_type": "float",
                "dispersive_type": "float",
                "iswin": iswin,
            },
            # templates for Double precision and dispersive materials with
            # complex susceptibility functions
            {
                "name_a": "update_electric_dispersive_multipole_A_double_complex",
                "name_b": "update_electric_dispersive_multipole_B_double_complex",
                "name_a_1": "update_electric_dispersive_1pole_A_double_complex",
                "name_b_1": "update_electric_dispersive_1pole_B_double_complex",
                "field_type": "double",
                "dispersive_type": "double complex",
                # c function to take real part of complex double type
                "real_part": "creal",
                "iswin": iswin,
            },
            # templates for Float precision and dispersive materials with
            # complex susceptibility functions
            {
                "name_a": "update_electric_dispersive_multipole_A_float_complex",
                "name_b": "update_electric_dispersive_multipole_B_float_complex",
                "name_a_1": "update_electric_dispersive_1pole_A_float_complex",
                "name_b_1": "update_electric_dispersive_1pole_B_float_complex",
                "field_type": "float",
                "dispersive_type": "float complex",
                # c function to take real part of complex double type
                "real_part": "crealf",
                "iswin": iswin,
            },
        ]
    )

    with open(os.path.join("gprMax", "cython", "fields_updates_dispersive.pyx"), "w") as f:
        f.write(r)


# Generate Cython file for dispersive materials update functions
cython_disp_file = os.path.join("gprMax", "cython", "fields_updates_dispersive.pyx")
if not os.path.isfile(cython_disp_file):
    build_dispersive_material_templates()

# Process 'build' command line argument
if "build" in sys.argv:
    print("Running 'build_ext --inplace'")
    sys.argv.remove("build")
    sys.argv.append("build_ext")
    sys.argv.append("--inplace")

# Build a list of all the files that need to be Cythonized looking in gprMax
# directory
cythonfiles = []
for root, dirs, files in os.walk(os.path.join(os.getcwd(), "gprMax"), topdown=True):
    for file in files:
        if file.endswith(".pyx"):
            cythonfiles.append(os.path.relpath(os.path.join(root, file)))

# Process 'cleanall' command line argument
if "cleanall" in sys.argv:
    for file in cythonfiles:
        filebase = os.path.splitext(file)[0]
        # Remove Cython C files
        if os.path.isfile(f"{filebase}.c"):
            try:
                os.remove(f"{filebase}.c")
                print(f"Removed: {filebase}.c")
            except OSError:
                print(f"Could not remove: {filebase}.c")
        # Remove compiled Cython modules
        libfile = glob.glob(os.path.join(os.getcwd(), os.path.splitext(file)[0]) + "*.pyd") + glob.glob(
            os.path.join(os.getcwd(), os.path.splitext(file)[0]) + "*.so"
        )
        if libfile:
            libfile = libfile[0]
            try:
                os.remove(libfile)
                print(f"Removed: {os.path.abspath(libfile)}")
            except OSError:
                print(f"Could not remove: {os.path.abspath(libfile)}")

    # Remove build, dist, egg and __pycache__ directories
    shutil.rmtree(Path.cwd().joinpath("build"), ignore_errors=True)
    shutil.rmtree(Path.cwd().joinpath("dist"), ignore_errors=True)
    shutil.rmtree(Path.cwd().joinpath("gprMax.egg-info"), ignore_errors=True)
    for p in Path.cwd().rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)
        print(f"Removed: {p}")

    # Remove 'gprMax/cython/fields_updates_dispersive.jinja' if its there
    if os.path.isfile(cython_disp_file):
        os.remove(cython_disp_file)

    # Now do a normal clean
    sys.argv[1] = "clean"  # this is what distutils understands

else:
    # Compiler options - Windows
    if sys.platform == "win32":
        # No static linking as no static version of OpenMP library;
        # /w disables warnings
        compile_args = ["/O2", "/openmp", "/w"]
        linker_args = []
        libraries = []

    elif sys.platform == "darwin":
        # Check for Intel or Apple M series CPU
        cpuID = (
            subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True, stderr=subprocess.STDOUT)
            .decode("utf-8")
            .strip()
        )
        cpuID = " ".join(cpuID.split())
        if "Apple" in cpuID:
            gccbasepath = "/opt/homebrew/bin/"
        else:
            gccbasepath = "/usr/local/bin/"
        gccpath = glob.glob(gccbasepath + "gcc-[0-9][0-9]")
        if gccpath:
            # Use newest gcc found
            os.environ["CC"] = gccpath[-1].split(os.sep)[-1]
            if "Apple" in cpuID:
                rpath = "/opt/homebrew/opt/gcc/lib/gcc/" + gccpath[-1].split(os.sep)[-1][-1] + "/"
        else:
            raise (
                f"Cannot find gcc in {gccbasepath}. gprMax requires gcc "
                + "to be installed - easily done through the Homebrew package "
                + "manager (http://brew.sh). Note: gcc with OpenMP support "
                + "is required."
            )

        # Set minimum supported macOS deployment target to installed macOS version
        MIN_MACOS_VERSION = platform.mac_ver()[0]
        try:
            os.environ["MACOSX_DEPLOYMENT_TARGET"]
            del os.environ["MACOSX_DEPLOYMENT_TARGET"]
        except:
            pass
        os.environ["MIN_SUPPORTED_MACOSX_DEPLOYMENT_TARGET"] = MIN_MACOS_VERSION
        # Sometimes worth testing with '-fstrict-aliasing', '-fno-common'
        compile_args = ["-O3", "-w", "-fopenmp", "-march=native", f"-mmacosx-version-min={MIN_MACOS_VERSION}"]
        linker_args = ["-fopenmp", f"-mmacosx-version-min={MIN_MACOS_VERSION}"]
        libraries = ["gomp"]

    elif sys.platform == "linux":
        compile_args = ["-O3", "-w", "-fopenmp", "-march=native"]
        linker_args = ["-fopenmp"]
        libraries = []

    # Build list of all the extensions - Cython source files
    extensions = []
    for file in cythonfiles:
        tmp = os.path.splitext(file)
        extension = Extension(
            tmp[0].replace(os.sep, "."),
            [tmp[0] + tmp[1]],
            language="c",
            include_dirs=[np.get_include()],
            extra_compile_args=compile_args,
            extra_link_args=linker_args,
            libraries=libraries,
        )
        extensions.append(extension)

    # Cythonize - build .c files
    extensions = cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "embedsignature": True,
            "language_level": 3,
        },
        nthreads=None,
        annotate=False,
    )

    # Parse long_description from README.rst file.
    with open("README.rst", "r", encoding="utf-8") as fd:
        long_description = fd.read()

    setup(
        name="gprMax",
        version=version,
        author="Craig Warren, Antonis Giannopoulos, and John Hartley",
        url="http://www.gprmax.com",
        description="Electromagnetic Modelling Software based on the " + "Finite-Difference Time-Domain (FDTD) method",
        long_description=long_description,
        long_description_content_type="text/x-rst",
        license="GPLv3+",
        python_requires=f">{str(MIN_PYTHON_VERSION[0])}.{str(MIN_PYTHON_VERSION[1])}",
        install_requires=[
            "colorama",
            "cython",
            "h5py",
            "jinja2",
            "matplotlib",
            "numpy",
            "psutil",
            "scipy",
            "terminaltables",
            "tqdm",
        ],
        ext_modules=extensions,
        packages=find_packages(),
        include_package_data=True,
        include_dirs=[np.get_include()],
        zip_safe=False,
        classifiers=[
            "Environment :: Console",
            "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
            "Operating System :: MacOS",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX :: Linux",
            "Programming Language :: Cython",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering",
        ],
    )
