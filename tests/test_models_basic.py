# Copyright (C) 2015-2017: The University of Edinburgh
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

import datetime
import os
import sys
from time import perf_counter

from colorama import init, Fore, Style
init()
import h5py
import numpy as np
np.seterr(invalid='raise')
import matplotlib.pyplot as plt

if sys.platform == 'linux':
    plt.switch_backend('agg')

from gprMax.gprMax import api
from gprMax.exceptions import GeneralError
from tests.analytical_solutions import hertzian_dipole_fs

"""Compare field outputs

    Usage:
        cd gprMax
        python -m tests.test_models_basic
"""

basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_basic')

# List of available test models
testmodels = ['hertzian_dipole_fs_analytical', '2D_ExHyHz', '2D_EyHxHz', '2D_EzHxHy', 'cylinder_Ascan_2D', 'hertzian_dipole_fs', 'hertzian_dipole_hs', 'hertzian_dipole_dispersive']

# Select a specific model if desired
#testmodels = [testmodels[0], testmodels[1], testmodels[2], testmodels[3], testmodels[4], testmodels[5]]
#testmodels = [testmodels[5]]
testresults = dict.fromkeys(testmodels)
path = '/rxs/rx1/'

starttime = perf_counter()

for i, model in enumerate(testmodels):

    testresults[model] = {}

    # Run model
    api(os.path.join(basepath, model + os.path.sep + model + '.in'))

    # Special case for analytical comparison
    if model == 'hertzian_dipole_fs_analytical':
        # Get output for model file
        filetest = h5py.File(os.path.join(basepath, model + os.path.sep + model + '.out'), 'r')
        testresults[model]['Test version'] = filetest.attrs['gprMax']

        # Get available field output component names
        outputstest = list(filetest[path].keys())

        # Arrays for storing time
        floattype = filetest[path + outputstest[0]].dtype
        timetest = np.zeros((filetest.attrs['Iterations']), dtype=floattype)
        timetest = np.arange(0, filetest.attrs['dt'] * filetest.attrs['Iterations'], filetest.attrs['dt']) / 1e-9
        timeref = timetest

        # Arrays for storing field data
        datatest = np.zeros((filetest.attrs['Iterations'], len(outputstest)), dtype=floattype)
        print(datatest.shape)
        for ID, name in enumerate(outputstest):
            datatest[:, ID] = filetest[path + str(name)][:]

        # Tx/Rx position to feed to analytical solution
        rxpos = filetest[path].attrs['Position']
        txpos = filetest['/srcs/src1/'].attrs['Position']
        rxposrelative = ((rxpos[0] - txpos[0]), (rxpos[1] - txpos[1]), (rxpos[2] - txpos[2]))

        # Analytical solution of a dipole in free space
        dataref = hertzian_dipole_fs(filetest.attrs['Iterations'], filetest.attrs['dt'], filetest.attrs['dx, dy, dz'], rxposrelative)

        filetest.close()

        # Diffs
        datadiffs = np.zeros(datatest.shape, dtype=floattype)
        for i in range(len(outputstest)):
            max = np.amax(np.abs(dataref[:, i]))
            try:
                datadiffs[:, i] = ((np.abs(dataref[:, i] - datatest[:, i])) / max) * 100
            except FloatingPointError:
                print('FloatingPointError')
                datadiffs[:, i] = 0

        # Register test passed
        threshold = 2  # Percent
        if np.amax(np.amax(datadiffs)) < 2:
            testresults[model]['Pass'] = True
        else:
            testresults[model]['Pass'] = False
        testresults[model]['Max diff'] = np.amax(np.amax(datadiffs))

    else:
        # Get output for model and reference files
        fileref = h5py.File(os.path.join(basepath, model + os.path.sep + model + '_ref.out'), 'r')
        filetest = h5py.File(os.path.join(basepath, model + os.path.sep + model + '.out'), 'r')
        testresults[model]['Ref version'] = fileref.attrs['gprMax']
        testresults[model]['Test version'] = filetest.attrs['gprMax']

        # Get available field output component names
        outputsref = list(fileref[path].keys())
        outputstest = list(filetest[path].keys())
        if outputsref != outputstest:
            raise GeneralError('Field output components do not match reference solution')

        # Check that type of float used to store fields matches
        if filetest[path + outputstest[0]].dtype != fileref[path + outputsref[0]].dtype:
            raise GeneralError('Type of floating point number does not match reference solution')
        else:
            floattype = fileref[path + outputsref[0]].dtype

        # Array for storing time
        timeref = np.zeros((fileref.attrs['Iterations']), dtype=floattype)
        timeref = np.arange(0, fileref.attrs['dt'] * fileref.attrs['Iterations'], fileref.attrs['dt']) / 1e-9
        timetest = np.zeros((filetest.attrs['Iterations']), dtype=floattype)
        timetest = np.arange(0, filetest.attrs['dt'] * filetest.attrs['Iterations'], filetest.attrs['dt']) / 1e-9

        # Get available field output component names
        outputsref = list(fileref[path].keys())
        outputstest = list(filetest[path].keys())
        if outputsref != outputstest:
            raise GeneralError('Field output components do not match reference solution')

        # Arrays for storing field data
        dataref = np.zeros((fileref.attrs['Iterations'], len(outputsref)), dtype=floattype)
        datatest = np.zeros((filetest.attrs['Iterations'], len(outputstest)), dtype=floattype)
        for ID, name in enumerate(outputsref):
            dataref[:, ID] = fileref[path + str(name)][:]
            datatest[:, ID] = filetest[path + str(name)][:]

        fileref.close()
        filetest.close()

        # Diffs
        datadiffs = np.zeros(datatest.shape, dtype=floattype)
        for i in range(len(outputstest)):
            max = np.nanmax(np.abs(dataref[:, i]))
            try:
                datadiffs[:, i] = ((np.abs(dataref[:, i] - datatest[:, i])) / max) * 100
            except FloatingPointError:
                print('FloatingPointError')
                datadiffs[:, i] = 0

        # Register test passed
        if not np.any(datadiffs):
            testresults[model]['Pass'] = True
        else:
            testresults[model]['Pass'] = False
        testresults[model]['Max diff'] = np.amax(np.amax(datadiffs))

    # Plot datasets
    fig1, ((ex1, hx1), (ey1, hy1), (ez1, hz1)) = plt.subplots(nrows=3, ncols=2, sharex=False, sharey='col', subplot_kw=dict(xlabel='Time [ns]'), num=model + '.in', figsize=(20, 10), facecolor='w', edgecolor='w')
    ex1.plot(timetest, datatest[:, 0], 'r', lw=2, label=model)
    ex1.plot(timeref, dataref[:, 0], 'g', lw=2, ls='--', label=model + '(Ref)')
    ey1.plot(timetest, datatest[:, 1], 'r', lw=2, label=model)
    ey1.plot(timeref, dataref[:, 1], 'g', lw=2, ls='--', label=model + '(Ref)')
    ez1.plot(timetest, datatest[:, 2], 'r', lw=2, label=model)
    ez1.plot(timeref, dataref[:, 2], 'g', lw=2, ls='--', label=model + '(Ref)')
    hx1.plot(timetest, datatest[:, 3], 'r', lw=2, label=model)
    hx1.plot(timeref, dataref[:, 3], 'g', lw=2, ls='--', label=model + '(Ref)')
    hy1.plot(timetest, datatest[:, 4], 'r', lw=2, label=model)
    hy1.plot(timeref, dataref[:, 4], 'g', lw=2, ls='--', label=model + '(Ref)')
    hz1.plot(timetest, datatest[:, 5], 'r', lw=2, label=model)
    hz1.plot(timeref, dataref[:, 5], 'g', lw=2, ls='--', label=model + '(Ref)')
    ylabels = ['$E_x$, field strength [V/m]', '$H_x$, field strength [A/m]', '$E_y$, field strength [V/m]', '$H_y$, field strength [A/m]', '$E_z$, field strength [V/m]', '$H_z$, field strength [A/m]']
    for i, ax in enumerate(fig1.axes):
        ax.set_ylabel(ylabels[i])
        ax.set_xlim(0, np.amax(timetest))
        ax.grid()
        ax.legend()

    # Plot diffs
    fig2, ((ex2, hx2), (ey2, hy2), (ez2, hz2)) = plt.subplots(nrows=3, ncols=2, sharex=False, sharey='col', subplot_kw=dict(xlabel='Time [ns]'), num='Diffs: ' + model + '.in', figsize=(20, 10), facecolor='w', edgecolor='w')
    ex2.plot(timeref, datadiffs[:, 0], 'r', lw=2, label='Ex')
    ey2.plot(timeref, datadiffs[:, 1], 'r', lw=2, label='Ey')
    ez2.plot(timeref, datadiffs[:, 2], 'r', lw=2, label='Ez')
    hx2.plot(timeref, datadiffs[:, 3], 'r', lw=2, label='Hx')
    hy2.plot(timeref, datadiffs[:, 4], 'r', lw=2, label='Hy')
    hz2.plot(timeref, datadiffs[:, 5], 'r', lw=2, label='Hz')
    ylabels = ['$E_x$, difference [%]', '$H_x$, difference [%]', '$E_y$, difference [%]', '$H_y$, difference [%]', '$E_z$, difference [%]', '$H_z$, difference [%]']
    for i, ax in enumerate(fig2.axes):
        ax.set_ylabel(ylabels[i])
        ax.set_xlim(0, np.amax(timetest))
        ax.grid()

    # Save a PDF/PNG of the figure
    savename = os.path.join(basepath, model + os.path.sep + model)
    #fig1.savefig(savename + '.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
    #fig2.savefig(savename + '_diffs.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
    fig1.savefig(savename + '.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    fig2.savefig(savename + '_diffs.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

stoptime = perf_counter()

# Summary of results
passed = 0
for name, data in testresults.items():
    if 'analytical' in name:
        if data['Pass']:
            print(Fore.GREEN + "Test '{}.in' using v.{} compared to analytical solution passed. Maximum difference = {}%".format(name, data['Test version'], data['Max diff']) + Style.RESET_ALL)
            passed += 1
        else:
            print(Fore.RED + "Test '{}.in' using v.{} compared to analytical solution failed. Maximum difference = {}%".format(name, data['Test version'], data['Max diff']) + Style.RESET_ALL)
    else:
        if data['Pass']:
            print(Fore.GREEN + "Test '{}.in' using v.{} compared to reference solution using v.{} passed. Maximum difference = {}%".format(name, data['Test version'], data['Ref version'], data['Max diff']) + Style.RESET_ALL)
            passed += 1
        else:
            print(Fore.RED + "Test '{}.in' using v.{} compared to reference solution using v.{} failed. Maximum difference = {}%".format(name, data['Test version'], data['Ref version'], data['Max diff']) + Style.RESET_ALL)
print('{} of {} tests passed successfully in [HH:MM:SS]: {}'.format(passed, len(testmodels), datetime.timedelta(seconds=int(stoptime - starttime))))
