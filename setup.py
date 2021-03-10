# Copyright (C) 2015-2021: The University of Edinburgh
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

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

try:
    import numpy as np
except ImportError:
    raise ImportError('gprMax requires the NumPy package.')

import glob
import os
import re
import shutil
import sys
from pathlib import Path

# SetupTools Required to make package
import setuptools
from jinja2 import Environment, PackageLoader, select_autoescape


def build_dispersive_material_templates():
    """Function to generate Cython .pyx files for dispersive media update.
        Jinja2 templates are used to render the various dispersive update
        functions.
    """

    env = Environment(
        loader=PackageLoader(__name__, 'gprMax/templates'),
    )

    template = env.get_template('fields_updates_dispersive_template')

    # Render dispersive template for different types
    r = template.render(
        functions=[
            # templates for Double precision and dispersive materials with
            # real susceptibility functions
            {
                'name_a': 'update_electric_dispersive_multipole_A_double_real',
                'name_b': 'update_electric_dispersive_multipole_B_double_real',
                'name_a_1': 'update_electric_dispersive_1pole_A_double_real',
                'name_b_1': 'update_electric_dispersive_1pole_B_double_real',
                'field_type': 'double',
                'dispersive_type': 'double'
            },
            # templates for Float precision and dispersive materials with
            # real susceptibility functions
            {
                'name_a': 'update_electric_dispersive_multipole_A_float_real',
                'name_b': 'update_electric_dispersive_multipole_B_float_real',
                'name_a_1': 'update_electric_dispersive_1pole_A_float_real',
                'name_b_1': 'update_electric_dispersive_1pole_B_float_real',
                'field_type': 'float',
                'dispersive_type': 'float'
            },
            # templates for Double precision and dispersive materials with
            # complex susceptibility functions
            {
                'name_a': 'update_electric_dispersive_multipole_A_double_complex',
                'name_b': 'update_electric_dispersive_multipole_B_double_complex',
                'name_a_1': 'update_electric_dispersive_1pole_A_double_complex',
                'name_b_1': 'update_electric_dispersive_1pole_B_double_complex',
                'field_type': 'double',
                'dispersive_type': 'double complex',
                # c function to take real part of complex double type
                'real_part': 'creal'
            },
            # templates for Float precision and dispersive materials with
            # complex susceptibility functions
            {
                'name_a': 'update_electric_dispersive_multipole_A_float_complex',
                'name_b': 'update_electric_dispersive_multipole_B_float_complex',
                'name_a_1': 'update_electric_dispersive_1pole_A_float_complex',
                'name_b_1': 'update_electric_dispersive_1pole_B_float_complex',
                'field_type': 'float',
                'dispersive_type': 'float complex',
                # c function to take real part of complex double type
                'real_part': 'crealf'
            }]
    )

    with open('gprMax/cython/fields_updates_dispersive.pyx', 'w') as f:
        f.write(r)

# Generate Cython file for dispersive materials update functions
build_dispersive_material_templates()

# Importing _version__.py before building can cause issues.
with open('gprMax/_version.py', 'r') as fd:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        fd.read(), re.MULTILINE).group(1)

# Parse package name from init file. Importing __init__.py / gprMax will break 
# as gprMax depends on compiled .pyx files.
with open('gprMax/__init__.py', 'r') as fd:
    packagename = re.search(r'^__name__\s*=\s*[\'"]([^\'"]*)[\'"]',
                            fd.read(), re.MULTILINE).group(1)

packages = [packagename, 'tests', 'tools', 'user_libs']

# Parse long_description from README.rst file.
with open('README.rst','r') as fd:
    long_description = fd.read()

# Python version
if sys.version_info[:2] < (3, 7):
    sys.exit('\nExited: Requires Python 3.7 or newer!\n')

# Process 'build' command line argument
if 'build' in sys.argv:
    print("Running 'build_ext --inplace'")
    sys.argv.remove('build')
    sys.argv.append('build_ext')
    sys.argv.append('--inplace')

# Process '--no-cython' command line argument - either Cythonize or just compile 
# the .c files
if '--no-cython' in sys.argv:
    USE_CYTHON = False
    sys.argv.remove('--no-cython')
else:
    USE_CYTHON = True

# Build a list of all the files that need to be Cythonized looking in gprMax 
# directory
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
                print(f'Removed: {filebase + ".c"}')
            except OSError:
                print(f'Could not remove: {filebase + ".c"}')
        # Remove compiled Cython modules
        libfile = glob.glob(os.path.join(os.getcwd(), 
                            os.path.splitext(file)[0]) + '*.pyd') + glob.glob(os.path.join(os.getcwd(), 
                            os.path.splitext(file)[0]) + '*.so')
        if libfile:
            libfile = libfile[0]
            try:
                os.remove(libfile)
                print(f'Removed: {os.path.abspath(libfile)}')
            except OSError:
                print(f'Could not remove: {os.path.abspath(libfile)}')
    # Remove build, dist, egg and __pycache__ directories
    shutil.rmtree(Path.cwd().joinpath('build'), ignore_errors=True)
    shutil.rmtree(Path.cwd().joinpath('dist'), ignore_errors=True)
    shutil.rmtree(Path.cwd().joinpath('gprMax.egg-info'), ignore_errors=True)
    for p in Path.cwd().rglob('__pycache__'):
        shutil.rmtree(p, ignore_errors=True)
        print(f'Removed: {p}')
    # Now do a normal clean
    sys.argv[1] = 'clean'  # this is what distutils understands

# Set compiler options
# Windows
if sys.platform == 'win32':
    compile_args = ['/O2', '/openmp', '/w']  # No static linking as no static version of OpenMP library; /w disables warnings
    linker_args = []
    extra_objects = []
    libraries=[]
# macOS - needs gcc (usually via HomeBrew) because the default compiler LLVM 
#           (clang) does not support OpenMP. With gcc -fopenmp option implies -pthread
elif sys.platform == 'darwin':
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
    libraries=['iomp5', 'pthread']
    extra_objects = []
# Linux
elif sys.platform == 'linux':
    compile_args = ['-O3', '-w', '-fopenmp', '-march=native']
    linker_args = ['-fopenmp']
    extra_objects = []
    libraries=[]

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
                          extra_compile_args=compile_args,
                          extra_link_args=linker_args,
                          libraries=libraries,
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


setup(name=packagename,
      version=version,
      author='Craig Warren, Antonis Giannopoulos, and John Hartley',
      url='http://www.gprmax.com',
      description='Electromagnetic Modelling Software based on the Finite-Difference Time-Domain (FDTD) method',
      long_description=long_description,
      long_description_content_type="text/x-rst",
      license='GPLv3+',
      classifiers=[
          'Environment :: Console',
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Cython',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering'
      ],
      #requirements
      python_requires=">3.7",
      install_requires=[
          "colorama",
          "cython",
          "h5py",
          "jupyter",
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
      include_dirs=[np.get_include()])
