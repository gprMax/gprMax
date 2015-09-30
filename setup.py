# Copyright (C) 2015: The University of Edinburgh
#            Authors: Craig Warren and Antonis Giannopoulos
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
    raise ImportError('The numpy package is required to build gprMax.')

import glob, os, re, shutil, sys

# Main package name
packagename = 'gprMax'

# Read version number from gprMax/gprMax.py
version = re.search('^__version__\s*=\s*\'(.*)\'',
                    open(os.path.join(packagename, 'gprMax.py')).read(),
                    re.M).group(1)

# Mac OS X - need to install gcc (via HomeBrew) and set it to be used. This is required because the default compiler LLVM (clang) does not support OpenMP
if sys.platform == 'darwin':
    # Try to find a HomeBrew installed gcc
    os.environ['CC'] = glob.glob('/usr/local/bin/gcc-[4-5]*')[0].split(os.sep)[-1]

# Either Cythonize or just compile the .c files if --no-cython is given
if '--no-cython' in sys.argv:
    USE_CYTHON = False
    sys.argv.remove('--no-cython')
else:
    USE_CYTHON = True

# Build a list of all the files that need to be Cythonized looking in gprMax directory and user_libs
cythonfiles = []
for root, dirs, files in os.walk(os.path.join(os.getcwd(), packagename)):
    for file in files:
        if file.endswith('.pyx'):
            cythonfiles.append(os.path.join(packagename, file))
for root, dirs, files in os.walk(os.path.join(os.getcwd(), 'user_libs')):
    for file in files:
        if file.endswith('.pyx'):
            cythonfiles.append(os.path.join('user_libs', file))

# Option to cleanup Cython files
if 'cleanall' in sys.argv:
    USE_CYTHON = False
    print('Deleting Cython files...')
    for file in cythonfiles:
        tmp = os.path.splitext(file)
        cfile = tmp[0] + '.c'
        if sys.platform == 'win32':
            libfile = tmp[0] + '.pyd'
        else:
            libfile = tmp[0] + '.so'
        try:
            os.remove(cfile)
        except OSError:
            pass
        try:
            os.remove(libfile)
        except OSError:
            pass
        shutil.rmtree('build', ignore_errors=True)
    # Now do a normal clean
    sys.argv[1] = 'clean'  # this is what distutils understands

# Build a list of all the extensions
if sys.platform == 'win32':
    compile_args = ['/O2', '/openmp', '/w']
    linker_args = ['/openmp']
else:
    compile_args = ['-O3', '-fopenmp', '-w']
    linker_args = ['-fopenmp']

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
                          extra_link_args=linker_args)
    extensions.append(extension)

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions,
                           compiler_directives={
                               'boundscheck': False,
                               'wraparound': False,
                               'embedsignature': True,
                               'language_level': 3
                           },
                           annotate=False)

setup(name=packagename,
      version=version,
      author='Craig Warren and Antonis Giannopoulos',
      url='http://www.gprmax.com',
      description='Electromagnetic Modelling Software based on the Finite-Difference Time-Domain (FDTD) method',
      license='GPLv3+',
      classifiers=[
                   'Environment :: Console',
                   'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows :: Windows 7',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Cython',
                   'Programming Language :: Python :: 3 :: Only'
                   ],
      ext_modules=extensions,
      include_dirs=[np.get_include()])
