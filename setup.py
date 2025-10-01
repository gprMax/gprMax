# Copyright (C) 2015-2023: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
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

from setuptools import setup, Extension  # type: ignore[import-not-found]

try:
    import numpy as np
except ImportError:
    raise ImportError('gprMax requires the NumPy package.')

import glob
import os
import pathlib
import re
import subprocess
import shutil
import sys

# Importing _version__.py before building can cause issues.
with open('gprMax/_version.py', 'r') as fd:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        fd.read(), re.MULTILINE).group(1)

# Parse package name from init file. Importing __init__.py / gprMax will break as gprMax depends on compiled .pyx files.
with open('gprMax/__init__.py', 'r') as fd:
    packagename = re.search(r'^__name__\s*=\s*[\'"]([^\'"]*)[\'"]',
                            fd.read(), re.MULTILINE).group(1)

packages = [packagename, 'tests', 'tools', 'user_libs']

# Parse long_description from README.rst file.
with open('README.rst', 'r', encoding='utf-8') as fd:
    long_description = fd.read()

# Python version guard (align with pyproject.toml requires-python >=3.7)
if sys.version_info < (3, 7):
    sys.exit('\nExited: Requires Python 3.7 or newer.\n')

# Process 'build' command line argument
if 'build' in sys.argv:
    print("Running 'build_ext --inplace'")
    sys.argv.remove('build')
    sys.argv.append('build_ext')
    sys.argv.append('--inplace')

# Process '--no-cython' command line argument - either Cythonize or just compile the .c files
if '--no-cython' in sys.argv:
    USE_CYTHON = False
    sys.argv.remove('--no-cython')
else:
    USE_CYTHON = True

# Build a list of all the files that need to be Cythonized looking in gprMax directory
cythonfiles = []
for root, dirs, files in os.walk(os.path.join(os.getcwd(), packagename), topdown=True):
    for file in files:
        if file.endswith('.pyx'):
            cythonfiles.append(os.path.relpath(os.path.join(root, file)))

# Process 'cleanall' command line argument - cleanup Cython files
if 'cleanall' in sys.argv:
    USE_CYTHON = False
    for file in cythonfiles:
        filebase = os.path.splitext(file)[0]
        # Remove Cython C files
        if os.path.isfile(filebase + '.c'):
            try:
                os.remove(filebase + '.c')
                print('Removed: {}'.format(filebase + '.c'))
            except OSError:
                print('Could not remove: {}'.format(filebase + '.c'))
        # Remove compiled Cython modules
        libfile = glob.glob(os.path.join(os.getcwd(), os.path.splitext(file)[0]) + '*.pyd') + glob.glob(os.path.join(os.getcwd(), os.path.splitext(file)[0]) + '*.so')
        if libfile:
            libfile = libfile[0]
            try:
                os.remove(libfile)
                print('Removed: {}'.format(os.path.abspath(libfile)))
            except OSError:
                print('Could not remove: {}'.format(os.path.abspath(libfile)))
    # Remove build, dist, egg and __pycache__ directories
    shutil.rmtree(os.path.join(os.getcwd(), 'build'), ignore_errors=True)
    shutil.rmtree(os.path.join(os.getcwd(), 'dist'), ignore_errors=True)
    shutil.rmtree(os.path.join(os.getcwd(), 'gprMax.egg-info'), ignore_errors=True)
    for p in pathlib.Path(os.getcwd()).rglob('__pycache__'):
        shutil.rmtree(p, ignore_errors=True)
        print('Removed: {}'.format(p))
    # Now do a normal clean
    sys.argv[1] = 'clean'  # this is what distutils understands


# Set compiler options
# Windows
if sys.platform == 'win32':
    compile_args = ['/O2', '/openmp', '/w']  # No static linking as no static version of OpenMP library; /w disables warnings
    linker_args = []
    extra_objects = []
    libraries=[]
# Mac OS X - needs gcc (usually via HomeBrew) because the default compiler LLVM (clang) does not support OpenMP
#          - with gcc -fopenmp option implies -pthread
elif sys.platform == 'darwin':
    # Check for Intel or Apple M series CPU
    cpuID = subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True, stderr=subprocess.STDOUT).decode('utf-8').strip()
    cpuID = ' '.join(cpuID.split())
    if 'Apple' in cpuID:
        gccpath = glob.glob('/opt/homebrew/bin/gcc-[4-9]*')
        gccpath += glob.glob('/opt/homebrew/bin/gcc-[10-11]*')
        if gccpath:
            # Use newest gcc found
            os.environ['CC'] = gccpath[-1].split(os.sep)[-1]
            rpath = '/opt/homebrew/opt/gcc/lib/gcc/' + gccpath[-1].split(os.sep)[-1][-1] + '/'
        else:
            raise('Cannot find gcc 4-10 in /opt/homebrew/bin. gprMax requires gcc to be installed - easily done through the Homebrew package manager (http://brew.sh). Note: gcc with OpenMP support is required.')
    else:
        gccpath = glob.glob('/usr/local/bin/gcc-[4-9]*')
        gccpath += glob.glob('/usr/local/bin/gcc-[10-11]*')
        if gccpath:
            # Use newest gcc found
            os.environ['CC'] = gccpath[-1].split(os.sep)[-1]
            rpath = '/usr/local/opt/gcc/lib/gcc/' + gccpath[-1].split(os.sep)[-1][-1] + '/'
        else:
            raise('Cannot find gcc 4-10 in /usr/local/bin. gprMax requires gcc to be installed - easily done through the Homebrew package manager (http://brew.sh). Note: gcc with OpenMP support is required.')
    compile_args = ['-O3', '-w', '-fopenmp', '-march=native']  # Sometimes worth testing with '-fstrict-aliasing', '-fno-common'
    linker_args = ['-fopenmp', '-Wl,-rpath,' + rpath]
    # Homebrew fournit GCC avec libgomp (runtime OpenMP GNU). L'ancienne configuration essayait de lier iomp5 (runtime Intel),
    # absent sur la plupart des installations standard et cause une erreur: ld: library 'iomp5' not found.
    # Avec -fopenmp, GCC ajoute déjà automatiquement libgomp, donc on n'a pas besoin de préciser libraries.
    # Si on voulait être explicite on pourrait utiliser ['gomp'], mais on laisse vide pour portabilité.
    libraries = []
    extra_objects = []
# Linux
elif sys.platform == 'linux':
    compile_args = ['-O3', '-w', '-fopenmp', '-march=native']
    linker_args = ['-fopenmp']
    extra_objects = []
    libraries=[]

# Optional: Intel OpenMP runtime (iomp5) support
# ----------------------------------------------
# Historically the project listed iomp5 which produced link failures on
# standard GCC + libgomp installs. It is now optional so default builds rely
# on the platform OpenMP runtime while still permitting Intel toolchains
# (icx/icc/icl) to request iomp5.
# Enable by exporting GPRMAX_USE_INTEL_OPENMP=1 before installation, or by
# setting CC=icx/icc/icl for auto detection. Warning: if libiomp5.* is not
# present on the system the link step will fail.
use_intel_omp_env = os.environ.get('GPRMAX_USE_INTEL_OPENMP', '').lower() in ('1', 'true', 'yes')
cc_lower = os.environ.get('CC', '').lower()
detected_intel_cc = any(tok in cc_lower for tok in ('icx', 'icc', 'icl'))
if use_intel_omp_env or detected_intel_cc:
    # Évite les doublons si déjà ajouté
    if 'iomp5' not in libraries and sys.platform != 'win32':  # Sous Windows Intel fournit généralement la DLL automatiquement
        libraries.append('iomp5')
        print("[gprMax] Activation du runtime Intel OpenMP: liaison avec iomp5 (détection compilateur Intel ou variable GPRMAX_USE_INTEL_OPENMP).")

# Build a list of all the extensions
extensions = []
for file in cythonfiles:
    tmp = os.path.splitext(file)
    if USE_CYTHON:
        fileext = tmp[1]
    else:
        fileext = '.c'
    extension = Extension(tmp[0].replace(os.sep, '.'),
                          [tmp[0] + fileext],
                          language='c',
                          include_dirs=[np.get_include()],
                          libraries=libraries,
                          extra_compile_args=compile_args,
                          extra_link_args=linker_args,
                          extra_objects=extra_objects)
    extensions.append(extension)

# Cythonize (build .c files)
if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions,
                           compiler_directives={
                               'boundscheck': False,
                               'wraparound': False,
                               'initializedcheck': False,
                               'embedsignature': True,
                               'language_level': 3
                           },
                           annotate=False)

setup(
    name=packagename,
    version=version,
    author='Craig Warren and Antonis Giannopoulos',
    url='http://www.gprmax.com',
    # Les métadonnées (description, license, classifiers, long_description) sont définies dans pyproject.toml.
    python_requires=">=3.7",
    install_requires=[
        # Core runtime dependencies (Jupyter moved to optional extras in pyproject.toml)
        "colorama",
        "cython",
        "h5py",
        "matplotlib",
        "numpy",
        "psutil",
        "scipy",
        "terminaltables",
        "tqdm",
    ],
    ext_modules=extensions,
    packages=packages,
    include_package_data=True,
    include_dirs=[np.get_include()],
    zip_safe=False,
)
