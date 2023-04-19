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

import os
import sys

from colorama import init, Fore, Style
init()
import h5py
import numpy as np
import matplotlib.pyplot as plt

if sys.platform == 'linux':
    plt.switch_backend('agg')

from gprMax.gprMax import api
from gprMax.exceptions import GeneralError
from tests.analytical_solutions import hertzian_dipole_fs

"""Compare field outputs

    Usage:
        cd gprMax
        python -m tests.test_models
"""

basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_')
# basepath += 'basic'
# basepath += 'advanced'
basepath += 'pmls'

# List of available basic test models
# testmodels = ['hertzian_dipole_fs_analytical', '2D_ExHyHz', '2D_EyHxHz', '2D_EzHxHy', 'cylinder_Ascan_2D', 'hertzian_dipole_fs', 'hertzian_dipole_hs', 'hertzian_dipole_dispersive', 'magnetic_dipole_fs', 'pmls']

# List of available advanced test models
# testmodels = ['antenna_GSSI_1500_fs', 'antenna_MALA_1200_fs']

# List of available PML models
testmodels = ['pml_x0', 'pml_y0', 'pml_z0', 'pml_xmax', 'pml_ymax', 'pml_zmax', 'pml_3D_pec_plate']

# Select a specific model if desired
# testmodels = testmodels[:-1]
testmodels = [testmodels[6]]
testresults = dict.fromkeys(testmodels)
path = '/rxs/rx1/'

# Minimum value of difference to plot (dB)
plotmin = -160

for i, model in enumerate(testmodels):

    testresults[model] = {}

    # Run model
    inputfile = os.path.join(basepath, model + os.path.sep + model + '.in')
    api(inputfile, gpu=[None])

    # Special case for analytical comparison
    if model == 'hertzian_dipole_fs_analytical':
        # Get output for model file
        filetest = h5py.File(os.path.join(basepath, model + os.path.sep + model + '.out'), 'r')
        testresults[model]['Test version'] = filetest.attrs['gprMax']

        # Get available field output component names
        outputstest = list(filetest[path].keys())

        # Arrays for storing time
        floattype = filetest[path + outputstest[0]].dtype
        timetest = np.linspace(0, (filetest.attrs['Iterations'] - 1) * filetest.attrs['dt'], num=filetest.attrs['Iterations']) / 1e-9
        timeref = timetest

        # Arrays for storing field data
        datatest = np.zeros((filetest.attrs['Iterations'], len(outputstest)), dtype=floattype)
        for ID, name in enumerate(outputstest):
            datatest[:, ID] = filetest[path + str(name)][:]
            if np.any(np.isnan(datatest[:, ID])):
                raise GeneralError('Test data contains NaNs')

        # Tx/Rx position to feed to analytical solution
        rxpos = filetest[path].attrs['Position']
        txpos = filetest['/srcs/src1/'].attrs['Position']
        rxposrelative = ((rxpos[0] - txpos[0]), (rxpos[1] - txpos[1]), (rxpos[2] - txpos[2]))

        # Analytical solution of a dipole in free space
        dataref = hertzian_dipole_fs(filetest.attrs['Iterations'], filetest.attrs['dt'], filetest.attrs['dx_dy_dz'], rxposrelative)

        filetest.close()

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
            print(Fore.RED + 'WARNING: Type of floating point number in test model ({}) does not match type in reference solution ({})\n'.format(filetest[path + outputstest[0]].dtype, fileref[path + outputsref[0]].dtype) + Style.RESET_ALL)
        floattyperef = fileref[path + outputsref[0]].dtype
        floattypetest = filetest[path + outputstest[0]].dtype

        # Arrays for storing time
        timeref = np.zeros((fileref.attrs['Iterations']), dtype=floattyperef)
        timeref = np.linspace(0, (fileref.attrs['Iterations'] - 1) * fileref.attrs['dt'], num=fileref.attrs['Iterations']) / 1e-9
        timetest = np.zeros((filetest.attrs['Iterations']), dtype=floattypetest)
        timetest = np.linspace(0, (filetest.attrs['Iterations'] - 1) * filetest.attrs['dt'], num=filetest.attrs['Iterations']) / 1e-9

        # Arrays for storing field data
        dataref = np.zeros((fileref.attrs['Iterations'], len(outputsref)), dtype=floattyperef)
        datatest = np.zeros((filetest.attrs['Iterations'], len(outputstest)), dtype=floattypetest)
        for ID, name in enumerate(outputsref):
            dataref[:, ID] = fileref[path + str(name)][:]
            datatest[:, ID] = filetest[path + str(name)][:]
            if np.any(np.isnan(datatest[:, ID])):
                raise GeneralError('Test data contains NaNs')

        fileref.close()
        filetest.close()

    # Diffs
    datadiffs = np.zeros(datatest.shape, dtype=np.float64)
    for i in range(len(outputstest)):
        max = np.amax(np.abs(dataref[:, i]))
        datadiffs[:, i] = np.divide(np.abs(dataref[:, i] - datatest[:, i]), max, out=np.zeros_like(dataref[:, i]), where=max != 0)  # Replace any division by zero with zero

        # Calculate power (ignore warning from taking a log of any zero values)
        with np.errstate(divide='ignore'):
            datadiffs[:, i] = 20 * np.log10(datadiffs[:, i])
        # Replace any NaNs or Infs from zero division
        datadiffs[:, i][np.invert(np.isfinite(datadiffs[:, i]))] = 0

    # Store max difference
    maxdiff = np.amax(np.amax(datadiffs))
    testresults[model]['Max diff'] = maxdiff

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
    ylabels = ['$E_x$, difference [dB]', '$H_x$, difference [dB]', '$E_y$, difference [dB]', '$H_y$, difference [dB]', '$E_z$, difference [dB]', '$H_z$, difference [dB]']
    for i, ax in enumerate(fig2.axes):
        ax.set_ylabel(ylabels[i])
        ax.set_xlim(0, np.amax(timetest))
        ax.set_ylim([plotmin, np.amax(np.amax(datadiffs))])
        ax.grid()

    # Save a PDF/PNG of the figure
    savename = os.path.join(basepath, model + os.path.sep + model)
    # fig1.savefig(savename + '.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
    # fig2.savefig(savename + '_diffs.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
    fig1.savefig(savename + '.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    fig2.savefig(savename + '_diffs.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)

# Summary of results
for name, data in sorted(testresults.items()):
    if 'analytical' in name:
        print(Fore.CYAN + "Test '{}.in' using v.{} compared to analytical solution. Max difference {:.2f}dB.".format(name, data['Test version'], data['Max diff']) + Style.RESET_ALL)
    else:
        print(Fore.CYAN + "Test '{}.in' using v.{} compared to reference solution using v.{}. Max difference {:.2f}dB.".format(name, data['Test version'], data['Ref version'], data['Max diff']) + Style.RESET_ALL)
