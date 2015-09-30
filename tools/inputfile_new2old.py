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

import argparse, os, sys
import numpy as np

from gprMax.exceptions import CmdInputError

"""Converts new to old style input files."""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Converts new style input file to old style input file.', usage='cd gprMax; python -m tools.inputfile_old2new inputfile')
parser.add_argument('inputfile', help='name of input file including path')
args = parser.parse_args()

inputfile = args.inputfile

with open(inputfile, 'r') as f:
    # Strip out any newline characters and comments that must begin with double hashes
    inputlines = [line.rstrip() for line in f]

# New file name base
try:
    newfile = inputfile.split('.')[0]
except:
    pass
newfile += '_oldstyle'

print("Attempting to convert inputfile '{}' to use old syntax...\n".format(inputfile))

newcommands = ['#add_dispersion_lorenz', '#add_dispersion_drude', '#fractal_box', '#soil_peplinski', '#python', '#end_python', '#add_surface_roughness', '#add_surface_water', '#add_grass', '#magnetic_dipole', '#pml_cfs', '#cylindrical_sector', '#time_step_limit_type']
materials = {}
debyes = []
waveforms = []
badwaveforms = ['gaussiandotdot', 'ricker']
hertziandipoles = []
voltagesources = []
analysiscmds = []
ompthreadscheck = False
pmlcheck = False
messagescheck = False

lindex = 0
while(lindex < len(inputlines)):
    
    if inputlines[lindex].startswith('#') and not inputlines[lindex].startswith('##'):
        cmd = inputlines[lindex].split(':')
        cmdname = cmd[0].lower()
        params = cmd[1].split()

        if cmdname == '#domain':
            domain = (float(params[0]), float(params[1]), float(params[2]))
            lindex += 1

        elif cmdname == '#dx_dy_dz':
            dx_dy_dz = (float(params[0]), float(params[1]), float(params[2]))
            lindex += 1
        
        elif cmdname == '#time_window':
            if '.' in params[0] or 'e' in params[0].lower():
                timewindow = float(params[0])
            else:
                timewindow = int(params[0])
            lindex += 1
        
        elif cmdname == '#messages':
            messagescheck = True

        elif cmdname == '#voltage_source':
            voltagesources.append(inputlines[lindex])
            inputlines.pop(lindex)

        elif cmdname == '#hertzian_dipole':
            hertziandipoles.append(inputlines[lindex])
            inputlines.pop(lindex)

        elif cmdname == '#rx':
            analysiscmds.append(inputlines[lindex])
            inputlines.pop(lindex)

        elif cmdname == '#waveform':
            waveforms.append(inputlines[lindex])
            inputlines.pop(lindex)

        elif cmdname == '#material':
            materials[lindex] = inputlines[lindex]
            lindex += 1
        
        elif cmdname == '#add_dispersion_debye':
            debyes.append(inputlines[lindex])
            inputlines.pop(lindex)

        elif cmdname == '#triangle':
            # Syntax of command: #triangle: x1 y1 z1 x2 y2 z2 x3 y3 z3 thickness ID
            if float(params[9]) == 0:
                replacement = '#triangle: {} {} {} {} {} {} {} {} {} {}'.format(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[10])
            else:
                replacement = '#wedge: {} {} {} {} {} {} {} {} {} {} {}'.format(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10])
            print("Command '{}', replaced with '{}'".format(inputlines[lindex], replacement))
            inputlines.pop(lindex)
            inputlines.insert(lindex, replacement)
            lindex += 1

        elif cmdname == '#cylinder':
            # Syntax of command: #cylinder: x1 y1 z1 x2 y2 z2 radius ID
            replacement = '#cylinder_new: {} {} {} {} {} {} {} {}'.format(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7])
            print("Command '{}', replaced with '{}'".format(inputlines[lindex], replacement))
            inputlines.pop(lindex)
            inputlines.insert(lindex, replacement)
            lindex += 1

        elif cmdname == '#pml_cells':
            pmlcheck = True
            # Syntax of command: #pml_cells: xminus yminus zminus xplus yplus zplus or #pml_cells: i1 (assumes latter behaviour)
            replacement = '#pml_layers: {}'.format(params[0])
            print("Command '{}', replaced with '{}'".format(inputlines[lindex], replacement))
            inputlines.pop(lindex)
            inputlines.insert(lindex, replacement)
            lindex += 1
        
        elif cmdname == '#num_threads':
            ompthreadscheck = True
            # Set number of threads to number of physical CPU cores, i.e. avoid hyperthreading with OpenMP for now
            if sys.platform == 'darwin':
                nthreads = int(os.popen('sysctl hw.physicalcpu').readlines()[0].split(':')[1].strip())
            elif sys.platform == 'win32':
                # Consider using wmi tools to check hyperthreading on Windows
                nthreads = os.cpu_count()
            elif 'linux' in sys.platform:
                lscpu = os.popen('lscpu').readlines()
                cpusockets = [item for item in lscpu if item.startswith('Socket(s)')]
                cpusockets = int(cpusockets[0].split(':')[1].strip())
                corespersocket = [item for item in lscpu if item.startswith('Core(s) per socket')]
                corespersocket = int(corespersocket[0].split(':')[1].strip())
                nthreads = cpusockets * corespersocket
            else:
                nthreads = os.cpu_count()
            # Syntax of command: #num_threads: nthreads
            replacement = '#num_of_procs: {}'.format(nthreads)
            print("Command '{}', replaced with '{}'".format(inputlines[lindex], replacement))
            inputlines.pop(lindex)
            analysiscmds.append(replacement)
            lindex += 1

        elif cmdname == '#snapshot':
            # Syntax of command: #snapshot: x1 y1 z1 x2 y2 z2 dx dy dz time filename
            replacement = '#snapshot: {} {} {} {} {} {} {} {} {} {} {} v'.format(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10])
            print("Command '{}', replaced with '{}'".format(inputlines[lindex], replacement))
            inputlines.pop(lindex)
            analysiscmds.append(replacement)
            lindex += 1

        elif cmdname == '#geometry_view':
            # Syntax of command: #geometry_vtk: x1 y1 z1 x2 y2 z2 dx dy dz filename type
            replacement = '#geometry_vtk: {} {} {} {} {} {} {} {} {} {} {}'.format(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10])
            print("Command '{}', replaced with '{}'".format(inputlines[lindex], replacement))
            inputlines.pop(lindex)
            inputlines.insert(lindex, replacement)
            lindex += 1

        elif cmdname in newcommands:
            print("Command '{}' cannot be used in the old version of gprMax.".format(inputlines[lindex]))
            inputlines.pop(lindex)

        else:
            lindex += 1

    else:
        lindex +=1

# If messages not found set to on
if not messagescheck:
    cmd = '#messages: y'
    print("Command '{}' added.".format(cmd))
    inputlines.append(cmd)

# Detect number of CPUs/cores on machine and set OpenMP threads if not already set
if not ompthreadscheck:
    # Set number of threads to number of physical CPU cores, i.e. avoid hyperthreading with OpenMP for now
    if sys.platform == 'darwin':
        nthreads = int(os.popen('sysctl hw.physicalcpu').readlines()[0].split(':')[1].strip())
    elif sys.platform == 'win32':
        # Consider using wmi tools to check hyperthreading on Windows
        nthreads = os.cpu_count()
    elif 'linux' in sys.platform:
        lscpu = os.popen('lscpu').readlines()
        cpusockets = [item for item in lscpu if item.startswith('Socket(s)')]
        cpusockets = int(cpusockets[0].split(':')[1].strip())
        corespersocket = [item for item in lscpu if item.startswith('Core(s) per socket')]
        corespersocket = int(corespersocket[0].split(':')[1].strip())
        nthreads = cpusockets * corespersocket
    else:
        nthreads = os.cpu_count()
    cmd = '#num_of_procs: {}'.format(nthreads)
    print("Command '{}' added.".format(cmd))
    inputlines.append(cmd)

# Add the default PML if not already set
if not pmlcheck:
    cmd1 = '#abc_type: pml'
    cmd2 = '#pml_layers: 10'
    print("Commands '{}' and '{}' added.".format(cmd1, cmd2))
    inputlines.append(cmd1)
    inputlines.append(cmd2)

# Process materials
for position, material in materials.items():
    params = material.split()
    debye = next((debye for debye in debyes if debye.split()[-1] == params[5]), None)
    if debye:
        if len(debye.split()) > 5:
            raise CmdInputError("Command '{}' cannot be used in the old version of gprMax as it only supports materials with a single Debye pole.".format(''.join(debye)))
        medium = '#medium: {} {} {} {} {} {} {}'.format(float(params[1]) + float(debye.split()[2]), params[1], float(debye.split()[3]), params[2], params[3], params[4], params[5])
        print("Commands '{}' and '{}', replaced with '{}'".format(material, debye, medium))
    else:
        medium = '#medium: {} 0 0 {} {} {} {}'.format(params[1], params[2], params[3], params[4], params[5])
        print("Command '{}', replaced with '{}'".format(material, medium))
    inputlines[position] = medium


# Create #analysis block
outputfile = newfile.split(os.sep)
analysis = '#analysis: 1 {} b'.format(outputfile[-1] + '.out')
analysiscmds.insert(0, analysis)


# Convert #hertzian_dipole and #waveform to #tx and #hertzian_dipole
for source in hertziandipoles:
    params = source.split()
    if len(params) > 6:
        ID = params[7]
        tx = '#tx: {} {} {} {} {} {} {}'.format(params[1], params[2], params[3], params[4], ID, params[5], params[6])
    else:
        ID = params[5]
        tx = '#tx: {} {} {} {} {} {} {}'.format(params[1], params[2], params[3], params[4], ID, 0, timewindow)

    waveform = next(waveform for waveform in waveforms if waveform.split()[4] == ID)
    waveformparams = waveform.split()
    if waveformparams[1] is badwaveforms:
        raise CmdInputError("Waveform types {} are not compatible between new and old versions of gprMax.".format(''.join(badwaveforms)))
    elif waveformparams[1] == 'gaussiandotnorm':
        waveformparams[1] = 'ricker'
    hertzian = '#hertzian_dipole: {} {} {} {}'.format(waveformparams[2], waveformparams[3], waveformparams[1], waveformparams[4])

    print("Commands '{}' and '{}', replaced with '{}' and '{}'".format(source, waveform, tx, hertzian))
    inputlines.append(hertzian)
    analysiscmds.append(tx)


# Convert #voltage_source and #waveform to #tx and #voltage_source
for source in voltagesources:
    params = source.split()
    if len(params) > 7:
        ID = params[8]
        tx = '#tx: {} {} {} {} {} {} {}'.format(params[1], params[2], params[3], params[4], ID, params[6], params[7])
    else:
        ID = params[6]
        tx = '#tx: {} {} {} {} {} {} {}'.format(params[1], params[2], params[3], params[4], ID, 0, timewindow)
    
    waveform = next(waveform for waveform in waveforms if waveform.split()[4] == ID)
    waveformparams = waveform.split()
    if waveformparams[1] is badwaveforms:
        raise CmdInputError("Waveform types {} are not compatible between new and old versions of gprMax.".format(''.join(badwaveforms)))
    elif waveformparams[1] == 'gaussiandotnorm':
        waveformparams[1] = 'ricker'
    voltagesource = '#voltage_source: {} {} {} {} {}'.format(waveformparams[2], waveformparams[3], waveformparams[1], params[5], waveformparams[4])

    print("Commands '{}' and '{}', replaced with '{}' and '{}'".format(source, waveform, tx, voltagesource))
    inputlines.append(voltagesource)
    analysiscmds.append(tx)

inputlines += analysiscmds
inputlines.append('#end_analysis:')


# Write new input file
newinputfile = newfile + '.in'

with open(newinputfile, 'w') as f:
    for line in inputlines:
        f.write('{}\n'.format(line))

print("\nWritten new input file: '{}'".format(newinputfile))
