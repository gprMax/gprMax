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

import argparse

from gprMax.exceptions import CmdInputError

"""Converts old to new style input files."""

# Parse command line arguments
parser = argparse.ArgumentParser(description='Converts old style input file to new style input file.', usage='cd gprMax; python -m tools.inputfile_new2old inputfile')
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
    newfile = inputfile
newfile += '_v3syntax'

print("Attempting to convert inputfile '{}' to use new syntax...\n".format(inputfile))

model2D = False
txs = []
badwaveforms = ['gaussiandot', 'gaussiandotdot']
linesources = []
voltagesources = []
hertziandipoles = []
transmissionlines = []

lindex = 0
while(lindex < len(inputlines)):

    if inputlines[lindex].startswith('#') and not inputlines[lindex].startswith('##'):
        cmd = inputlines[lindex].split(':')
        cmdname = cmd[0].lower()
        params = cmd[1].split()

        if cmdname == '#dx_dy':
            model2D = True
            # Syntax of old command: #dx_dy: x y
            replacement = '#dx_dy_dz: {:g} {:g} {:g}'.format(float(params[0]), float(params[1]), float(params[1]))
            print("Command '{}', replaced with '{}'".format(inputlines[lindex], replacement))
            inputlines.pop(lindex)
            inputlines.insert(lindex, replacement)
            lindex += 1
            dx_dy_dz = (float(params[0]), float(params[1]), float(params[1]))

        elif cmdname == '#dx_dy_dz':
            dx_dy_dz = (float(params[0]), float(params[1]), float(params[2]))
            lindex += 1

        else:
            lindex += 1

    else:
        lindex += 1

lindex = 0
while(lindex < len(inputlines)):

    if inputlines[lindex].startswith('#') and not inputlines[lindex].startswith('##'):
        cmd = inputlines[lindex].split(':')
        cmdname = cmd[0].lower()
        params = cmd[1].split()

        if cmdname == '#domain':
            if model2D:
                # Syntax of old command: #domain: x y
                replacement = '#domain: {:g} {:g} {:g}'.format(float(params[0]), float(params[1]), dx_dy_dz[2])
                print("Command '{}', replaced with '{}'".format(inputlines[lindex], replacement))
                inputlines.pop(lindex)
                inputlines.insert(lindex, replacement)
                domain = (float(params[0]), float(params[1]), dx_dy_dz[2])
            else:
                domain = (float(params[0]), float(params[1]), float(params[2]))

            lindex += 1

        elif cmdname == '#time_window':
            params = params[0].lower()
            if '.' in params or 'e' in params:
                timewindow = float(params)
            else:
                timewindow = int(params)
            lindex += 1

        elif cmdname == '#num_of_procs':
            # Syntax of old command: #num_of_procs: nthreads
            replacement = '#num_threads: {}'.format(params[0])
            print("Command '{}', replaced with '{}'".format(inputlines[lindex], replacement))
            inputlines.pop(lindex)
            inputlines.insert(lindex, replacement)
            lindex += 1

        elif cmdname == '#tx':
            txs.append(inputlines[lindex])
            lindex += 1

        elif cmdname == '#line_source':
            linesources.append(inputlines[lindex])
            lindex += 1

        elif cmdname == '#voltage_source':
            voltagesources.append(inputlines[lindex])
            lindex += 1

        elif cmdname == '#hertzian_dipole':
            hertziandipoles.append(inputlines[lindex])
            lindex += 1
        
        elif cmdname == '#transmission_line':
            transmissionlines.append(inputlines[lindex])
            lindex += 1

        elif cmdname == '#rx':
            if model2D:
                # Syntax of old command: #rx: x1 y1
                replacement = '#rx: {} {} {}'.format(params[0], params[1], 0)
                print("Command '{}', replaced with '{}'".format(inputlines[lindex], replacement))
                inputlines.pop(lindex)
                inputlines.insert(lindex, replacement)
            lindex += 1

        elif cmdname == '#rx_box':
            if model2D:
                # Syntax of old command: #rx_box: x1 y1 x2 y2 dx dy
                replacement = '#rx_array: {} {} {} {} {} {} {} {} {}'.format(params[0], params[1], 0, params[2], params[3], dx_dy_dz[2], params[4], params[5], dx_dy_dz[2])
            else:
                # Syntax of old command: #rx_box: x1 y1 z1 x2 y2 z2 dx dy dz
                replacement = '#rx_array: {} {} {} {} {} {} {} {} {}'.format(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8])
            print("Command '{}', replaced with '{}'".format(inputlines[lindex], replacement))
            inputlines.pop(lindex)
            inputlines.insert(lindex, replacement)
            lindex += 1

        elif cmdname == '#tx_steps':
            if model2D:
                # Syntax of old command: #tx_steps: dx dy
                replacement = '#src_steps: {} {} {}'.format(params[0], params[1], 0)
            else:
                # Syntax of old command: #tx_steps: dx dy dz
                replacement = '#src_steps: {} {} {}'.format(params[0], params[1], params[2])
            print("Command '{}', replaced with '{}'".format(inputlines[lindex], replacement))
            inputlines.pop(lindex)
            inputlines.insert(lindex, replacement)
            lindex += 1

        elif cmdname == '#rx_steps':
            if model2D:
                # Syntax of old command: #rx_steps: dx dy
                replacement = '#rx_steps: {} {} {}'.format(params[0], params[1], 0)
            else:
                # Syntax of old command: #rx_steps: dx dy dz
                replacement = '#rx_steps: {} {} {}'.format(params[0], params[1], params[2])
            print("Command '{}', replaced with '{}'".format(inputlines[lindex], replacement))
            inputlines.pop(lindex)
            inputlines.insert(lindex, replacement)
            lindex += 1

        elif cmdname == '#medium':
            # Syntax of old command: #medium: e_rs e_inf tau sig_e mu_r sig_m ID
            replacement = '#material: {} {} {} {} {}'.format(params[0], params[3], params[4], params[5], params[6])
            print("Command '{}', replaced with '{}'".format(inputlines[lindex], replacement))
            inputlines.pop(lindex)
            inputlines.insert(lindex, replacement)
            if float(params[1]) > 0:
                replacement = '#add_dispersion_debye: 1 {} {} {}'.format(float(params[0]) - float(params[1]), params[2], params[6])
                print("Command '{}' added.".format(replacement))
                inputlines.insert(lindex + 1, replacement)
            lindex += 1

        elif cmdname == '#box':
            if model2D:
                # Syntax of old command: #box: x1 y1 x2 y2 ID
                replacement = '#box: {} {} {} {} {} {} {}'.format(params[0], params[1], 0, params[2], params[3], dx_dy_dz[2], params[4])
                print("Command '{}', replaced with '{}'".format(inputlines[lindex], replacement))
                inputlines.pop(lindex)
                inputlines.insert(lindex, replacement)
            lindex += 1

        elif cmdname == '#triangle':
            if model2D:
                # Syntax of old command: #triangle: x1 y1 x2 y2 x3 y3 ID
                replacement = '#triangle: {} {} {} {} {} {} {} {} {} {} {}'.format(params[0], params[1], 0, params[2], params[3], 0, params[4], params[5], 0, dx_dy_dz[2], params[6])
            else:
                # Syntax of old command: #triangle: x1 y1 z1 x2 y2 z2 x3 y3 z3 ID
                replacement = '#triangle: {} {} {} {} {} {} {} {} {} {} {}'.format(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], 0, params[9])

            print("Command '{}', replaced with '{}'".format(inputlines[lindex], replacement))
            inputlines.pop(lindex)
            inputlines.insert(lindex, replacement)
            lindex += 1

        elif cmdname == '#wedge':
            # Syntax of old command: #wedge: x1 y1 z1 x2 y2 z2 x3 y3 z3 thickness ID
            replacement = '#triangle: {} {} {} {} {} {} {} {} {} {} {}'.format(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10])
            print("Command '{}', replaced with '{}'".format(inputlines[lindex], replacement))
            inputlines.pop(lindex)
            inputlines.insert(lindex, replacement)
            lindex += 1

        elif cmdname == '#bowtie':
            print("Command '{}', is no longer supported. You can create the bowtie shape using two triangle commands.".format(inputlines[lindex], replacement))
            inputlines.pop(lindex)

        elif cmdname == '#cylinder':
            if model2D:
                # Syntax of old command: #cylinder: x y radius ID
                replacement = '#cylinder: {} {} {} {} {} {} {} {}'.format(params[0], params[1], 0, params[0], params[1], dx_dy_dz[2], params[2], params[3])
            else:
                # Syntax of old command: #cylinder: axis axis_start axis_stop f1 f2 radius ID
                if params[0] == 'x':
                    replacement = '#cylinder: {} {} {} {} {} {} {} {}'.format(params[1], params[3], params[4], params[2], params[3], params[4], params[5], params[6])
                elif params[0] == 'y':
                    replacement = '#cylinder: {} {} {} {} {} {} {} {}'.format(params[3], params[1], params[4], params[3], params[2], params[4], params[5], params[6])
                elif params[0] == 'z':
                    replacement = '#cylinder: {} {} {} {} {} {} {} {}'.format(params[3], params[4], params[1], params[3], params[4], params[2], params[5], params[6])

            print("Command '{}', replaced with '{}'".format(inputlines[lindex], replacement))
            inputlines.pop(lindex)
            inputlines.insert(lindex, replacement)
            lindex += 1

        elif cmdname == '#cylinder_new':
            # Syntax of old command: #cylinder_new: x1 y1 z1 x2 y2 z2 radius ID
            replacement = '#cylinder: {} {} {} {} {} {} {} {}'.format(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7])
            print("Command '{}', replaced with '{}'".format(inputlines[lindex], replacement))
            inputlines.pop(lindex)
            inputlines.insert(lindex, replacement)
            lindex += 1

        elif cmdname == '#cylindrical_segment':
            print("Command '{}' has been removed as it is no longer supported. You can create a cylindrical segment by using a #box to cut through a #cylinder.".format(inputlines[lindex]))
            inputlines.pop(lindex)

        elif cmdname in ['#x_segment', '#y_segment']:
            print("Command '{}' has been removed. A circular segment can be created by using the #cylinder command and cutting it with a #box. Alternatively the #cylindrical_sector command maybe useful.".format(inputlines[lindex]))
            inputlines.pop(lindex)

        elif cmdname == '#media_file':
            print("Command '{}' has is no longer supported. Please include your materials using the #material command directly in the input file.".format(inputlines[lindex]))
            inputlines.pop(lindex)

        elif cmdname == '#pml_layers':
            # Syntax of old command: #pml_layers: num_layers
            replacement = '#pml_cells: {}'.format(params[0])
            print("Command '{}', replaced with '{}'".format(inputlines[lindex], replacement))
            inputlines.pop(lindex)
            inputlines.insert(lindex, replacement)
            lindex += 1

        elif cmdname in ['#abc_order', '#abc_type', 'abc_optimisation_angles', '#abc_mixing_parameters', '#abc_stability_factors']:
            print("Command '{}' has been removed as Higdon Absorbing Boundary Conditions (ABC) are no longer supported. The default ABC is the (better performing) Perfectly Matched Layer (PML).".format(inputlines[lindex]))
            inputlines.pop(lindex)

        elif cmdname == '#analysis':
            # Syntax of old command: #analysis: num_model_runs outputfile outputfiletype
            if int(params[0]) > 1:
                extra = " To run a model multiple times use the command line option -n, e.g. gprMax {} -n {}".format(inputfile, int(params[0]))
            else:
                extra = ''
            print("Command '{}' has been removed as it is no longer required.{}".format(inputlines[lindex], extra))
            inputlines.pop(lindex)

        elif cmdname in ['#end_analysis', '#number_of_media', '#nips_number']:
            print("Command '{}' has been removed as it is no longer required.".format(inputlines[lindex]))
            inputlines.pop(lindex)

        elif cmdname == '#snapshot':
            if model2D:
                # Syntax of old command: #snapshot: i1 x1 y1 x2 y2 dx dy time filename type
                replacement = '#snapshot: {} {} {} {} {} {} {} {} {} {} {}'.format(params[1], params[2], 0, params[3], params[4], dx_dy_dz[2], params[5], params[6], dx_dy_dz[2], params[7], params[8])
            else:
                # Syntax of old command: #snapshot: i1 x1 y1 z1 x2 y2 z2 dx dy dz time filename type
                replacement = '#snapshot: {} {} {} {} {} {} {} {} {} {} {}'.format(params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11])
            print("Command '{}', replaced with '{}'".format(inputlines[lindex], replacement))
            inputlines.pop(lindex)
            inputlines.insert(lindex, replacement)
            lindex += 1

        elif cmdname == '#geometry_file':
            # Syntax of old command: #geometry_file: filename
            if params[0].endswith('.geo'):
                params = params[0].split('.')
            replacement = '#geometry_view: 0 0 0 {} {} {} {} {} {} {} n'.format(domain[0], domain[1], domain[2], dx_dy_dz[0], dx_dy_dz[1], dx_dy_dz[2], params[0])
            print("Command '{}', replaced with '{}'. This is a geometry view of the entire domain, sampled at the spatial resolution of the model, using the per Yee cell option (n). You may want to consider taking a smaller geometry view or using a coarser sampling. You may also want to use the per Yee cell edge option (f) to view finer details.".format(inputlines[lindex], replacement))
            inputlines.pop(lindex)
            inputlines.insert(lindex, replacement)
            lindex += 1

        elif cmdname == '#geometry_vtk':
            # Syntax of old command: #geometry_vtk: x1 y1 z1 x2 y2 z2 dx dy dz filename type
            replacement = '#geometry_view: {} {} {} {} {} {} {} {} {} {} {}'.format(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10])
            print("Command '{}', replaced with '{}'".format(inputlines[lindex], replacement))
            inputlines.pop(lindex)
            inputlines.insert(lindex, replacement)
            lindex += 1

        elif cmdname in ['#plane_wave', '#thin_wire', '#huygens_surface']:
            raise CmdInputError("Command '{}' has not yet implemented in the new version of gprMax. For now please continue to use the old version.".format(inputlines[lindex]))

        else:
            lindex += 1

    else:
        lindex += 1

# Convert separate #line_source and associated #tx to #waveform and #hertzian_dipole
for source in linesources:
    params = source.split()
    if params[3] is badwaveforms:
        raise CmdInputError("Waveform types {} are not compatible between new and old versions of gprMax.".format(''.join(badwaveforms)))
    elif params[3] == 'ricker':
        params[3] = 'gaussiandotnorm'
    waveform = '#waveform: {} {} {} {}'.format(params[3], params[1], params[2], params[4])
    tx = next(tx for tx in txs if tx.split()[3] == params[4])
    hertziantx = tx.split()
    if float(hertziantx[4]) > 0 or float(hertziantx[5]) != timewindow:
        hertzian = '#hertzian_dipole: z {} {} {} {} {} {}'.format(hertziantx[1], hertziantx[2], 0, hertziantx[3], hertziantx[4], hertziantx[5])
    else:
        hertzian = '#hertzian_dipole: z {} {} {} {}'.format(hertziantx[1], hertziantx[2], 0, hertziantx[3])

    print("Commands '{}' and '{}', replaced with '{}' and '{}'".format(source, tx, waveform, hertzian))
    inputlines.remove(source)
    inputlines.remove(tx)
    inputlines.append(waveform)
    inputlines.append(hertzian)

# Convert separate #hertzian_dipole and associated #tx to #waveform and #hertzian_dipole
for source in hertziandipoles:
    params = source.split()
    if params[3] is badwaveforms:
        raise CmdInputError("Waveform types {} are not compatible between new and old versions of gprMax.".format(''.join(badwaveforms)))
    elif params[3] == 'ricker':
        params[3] = 'gaussiandotnorm'
    waveform = '#waveform: {} {} {} {}'.format(params[3], params[1], params[2], params[4])
    tx = next(tx for tx in txs if tx.split()[5] == params[4])
    hertziantx = tx.split()
    if float(hertziantx[6]) > 0 or float(hertziantx[7]) != timewindow:
        hertzian = '#hertzian_dipole: {} {} {} {} {} {} {}'.format(hertziantx[1], hertziantx[2], hertziantx[3], hertziantx[4], hertziantx[5], hertziantx[6], hertziantx[7])
    else:
        hertzian = '#hertzian_dipole: {} {} {} {} {}'.format(hertziantx[1], hertziantx[2], hertziantx[3], hertziantx[4], hertziantx[5])

    print("Commands '{}' and '{}', replaced with '{}' and '{}'".format(source, tx, waveform, hertzian))
    inputlines.remove(source)
    inputlines.remove(tx)
    inputlines.append(waveform)
    inputlines.append(hertzian)

# Convert separate #voltage_source and associated #tx to #waveform and #voltage_source
for source in voltagesources:
    params = source.split()
    if params[3] is badwaveforms:
        raise CmdInputError("Waveform types {} are not compatible between new and old versions of gprMax.".format(''.join(badwaveforms)))
    elif params[3] == 'ricker':
        params[3] = 'gaussiandotnorm'
    waveform = '#waveform: {} {} {} {}'.format(params[3], params[1], params[2], params[5])
    tx = next(tx for tx in txs if tx.split()[5] == params[5])
    voltagesourcetx = tx.split()
    if float(voltagesourcetx[6]) > 0 or float(voltagesourcetx[7]) != timewindow:
        voltagesource = '#voltage_source: {} {} {} {} {} {} {} {}'.format(voltagesourcetx[1], voltagesourcetx[2], voltagesourcetx[3], voltagesourcetx[4], params[4], voltagesourcetx[5], voltagesourcetx[6], voltagesourcetx[7])
    else:
        voltagesource = '#voltage_source: {} {} {} {} {} {}'.format(voltagesourcetx[1], voltagesourcetx[2], voltagesourcetx[3], voltagesourcetx[4], params[4], voltagesourcetx[5])

    print("Commands '{}' and '{}', replaced with '{}' and '{}'".format(source, tx, waveform, voltagesource))
    inputlines.remove(source)
    inputlines.remove(tx)
    inputlines.append(waveform)
    inputlines.append(voltagesource)

# Convert separate #transmission_line and associated #tx to #waveform and #transmission_line
for source in transmissionlines:
    params = source.split()
    if params[3] is badwaveforms:
        raise CmdInputError("Waveform types {} are not compatible between new and old versions of gprMax.".format(''.join(badwaveforms)))
    elif params[3] == 'ricker':
        params[3] = 'gaussiandotnorm'
    waveform = '#waveform: {} {} {} {}'.format(params[3], params[1], params[2], params[6])
    tx = next(tx for tx in txs if tx.split()[5] == params[6])
    transmissionlinetx = tx.split()
    if float(transmissionlinetx[6]) != 0 or float(transmissionlinetx[7]) < timewindow:
        transmissionline = '#transmission_line: {} {} {} {} {} {} {} {}'.format(transmissionlinetx[1], transmissionlinetx[2], transmissionlinetx[3], transmissionlinetx[4], params[5], transmissionlinetx[5], transmissionlinetx[6], transmissionlinetx[7])
    else:
        transmissionline = '#transmission_line: {} {} {} {} {} {}'.format(transmissionlinetx[1], transmissionlinetx[2], transmissionlinetx[3], transmissionlinetx[4], params[5], transmissionlinetx[5])

    print("Commands '{}' and '{}', replaced with '{}' and '{}'".format(source, tx, waveform, transmissionline))
    inputlines.remove(source)
    inputlines.remove(tx)
    inputlines.append(waveform)
    inputlines.append(transmissionline)

# Write new input file
newinputfile = newfile + '.in'

with open(newinputfile, 'w') as f:
    for line in inputlines:
        f.write('{}\n'.format(line))

print("\nWritten new input file: '{}'".format(newinputfile))
