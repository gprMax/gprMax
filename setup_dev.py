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

from setuptools import setup, Extension
#import py2exe, os, sys, re
import os, sys, re
from cx_Freeze import setup, Executable

#sys.argv.append('py2exe')

# Main package name
packagename = 'gprMax'

# Read version number from gprMax/gprMax.py
version = re.search('^__version__\s*=\s*\'(.*)\'',
                    open(os.path.join(packagename, 'gprMax.py')).read(),
                    re.M).group(1)

includes = []
include_files = []
excludes = []
packages = ['gprMax']

options =   {
            'build_exe':    {
            'path': [],
            'includes': includes,
            'include_files': include_files,
            'excludes': excludes,
            'packages': packages,
            'optimize': 2,
                            }
            }

executables = [
               Executable('gprMax/gprMax.py'),
               ]

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
      options=options,
      executables=executables
      )

#setup(name=packagename,
#      version=version,
#      author='Craig Warren and Antonis Giannopoulos',
#      url='http://www.gprmax.com',
#      description='Electromagnetic Modelling Software based on the Finite-Difference Time-Domain (FDTD) method',
#      license='GPLv3+',
#      classifiers=[
#                   'Environment :: Console',
#                   'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
#                   'Operating System :: MacOS :: MacOS X',
#                   'Operating System :: Microsoft :: Windows :: Windows 7',
#                   'Operating System :: POSIX :: Linux',
#                   'Programming Language :: Cython',
#                   'Programming Language :: Python :: 3 :: Only'
#                   ],
#      console=[{'script':'gprMax\gprMax.py'}],
#      options = {"py2exe": {"compressed": False,
#                            "optimize": 2,
#                            "includes": includes,
#                            "excludes": excludes,
#                            "packages": packages,
#                            "dll_excludes": dll_excludes,
#                            "bundle_files": 1,
#                            }
#                },
#      zipfile = None,
#      )
