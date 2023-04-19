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
from io import StringIO

from gprMax.exceptions import CmdInputError


def process_python_include_code(inputfile, usernamespace):
    """Looks for and processes any Python code found in the input file.
    It will ignore any lines that are comments, i.e. begin with a
    double hash (##), and any blank lines. It will also ignore any
    lines that do not begin with a hash (#) after it has processed
    Python commands. It will also process any include file commands
    and insert the contents of the included file at that location.

    Args:
        inputfile (object): File object for input file.
        usernamespace (dict): Namespace that can be accessed by user
                in any Python code blocks in input file.

    Returns:
        processedlines (list): Input commands after Python processing.
    """

    # Strip out any newline characters and comments that must begin with double hashes
    inputlines = [line.rstrip() for line in inputfile if(not line.startswith('##') and line.rstrip('\n'))]

    # Rewind input file in preparation for any subsequent reading function
    inputfile.seek(0)

    # List to hold final processed commands
    processedlines = []

    x = 0
    while(x < len(inputlines)):

        # Process any Python code
        if(inputlines[x].startswith('#python:')):

            # String to hold Python code to be executed
            pythoncode = ''
            x += 1
            while not inputlines[x].startswith('#end_python:'):
                # Add all code in current code block to string
                pythoncode += inputlines[x] + '\n'
                x += 1
                if x == len(inputlines):
                    raise CmdInputError('Cannot find the end of the Python code block, i.e. missing #end_python: command.')
            # Compile code for faster execution
            pythoncompiledcode = compile(pythoncode, '<string>', 'exec')
            # Redirect stdout to a text stream
            sys.stdout = result = StringIO()
            # Execute code block & make available only usernamespace
            exec(pythoncompiledcode, usernamespace)
            # String containing buffer of executed code
            codeout = result.getvalue().split('\n')
            result.close()

            # Reset stdio
            sys.stdout = sys.__stdout__

            # Separate commands from any other generated output
            hashcmds = []
            pythonout = []
            for line in codeout:
                if line.startswith('#'):
                    hashcmds.append(line + '\n')
                elif line:
                    pythonout.append(line)

            # Add commands to a list
            processedlines.extend(hashcmds)

            # Print any generated output that is not commands
            if pythonout:
                print('Python messages (from stdout/stderr): {}\n'.format(pythonout))

        # Add any other commands to list
        elif(inputlines[x].startswith('#')):
            # Add gprMax command to list
            inputlines[x] += ('\n')
            processedlines.append(inputlines[x])

        x += 1

    # Process any include file commands
    processedlines = process_include_files(processedlines, inputfile)

    return processedlines


def process_include_files(hashcmds, inputfile):
    """
    Looks for and processes any include file commands and insert
        the contents of the included file at that location.

    Args:
        hashcmds (list): Input commands.
        inputfile (object): File object for input file.

    Returns:
        processedincludecmds (list): Input commands after processing
            any include file commands.
    """

    processedincludecmds = []
    x = 0
    while x < len(hashcmds):
        if hashcmds[x].startswith('#include_file:'):
            includefile = hashcmds[x].split()

            if len(includefile) != 2:
                raise CmdInputError('#include_file requires exactly one parameter')

            includefile = includefile[1]

            # See if file exists at specified path and if not try input file directory
            if not os.path.isfile(includefile):
                includefile = os.path.join(os.path.dirname(inputfile.name), includefile)

            with open(includefile, 'r') as f:
                # Strip out any newline characters and comments that must begin with double hashes
                includelines = [includeline.rstrip() + '\n' for includeline in f if(not includeline.startswith('##') and includeline.rstrip('\n'))]

            # Add lines from include file
            processedincludecmds.extend(includelines)

        else:
            processedincludecmds.append(hashcmds[x])

        x += 1

    return processedincludecmds


def write_processed_file(processedlines, appendmodelnumber, G):
    """
    Writes an input file after any Python code and include commands
    in the original input file have been processed.

    Args:
        processedlines (list): Input commands after after processing any
            Python code and include commands.
        appendmodelnumber (str): Text to append to filename.
        G (class): Grid class instance - holds essential parameters describing the model.
    """

    processedfile = os.path.join(G.inputdirectory, os.path.splitext(G.inputfilename)[0] + appendmodelnumber + '_processed.in')

    with open(processedfile, 'w') as f:
        for item in processedlines:
            f.write('{}'.format(item))

    print('Written input commands, after processing any Python code and include commands, to file: {}\n'.format(processedfile))


def check_cmd_names(processedlines, checkessential=True):
    """
    Checks the validity of commands, i.e. are they gprMax commands,
        and that all essential commands are present.

    Args:
        processedlines (list): Input commands after Python processing.
        checkessential (boolean): Perform check to see that all essential commands are present.

    Returns:
        singlecmds (dict): Commands that can only occur once in the model.
        multiplecmds (dict): Commands that can have multiple instances in the model.
        geometry (list): Geometry commands in the model.
    """

    # Dictionaries of available commands
    # Essential commands neccessary to run a gprMax model
    essentialcmds = ['#domain', '#dx_dy_dz', '#time_window']

    # Commands that there should only be one instance of in a model
    singlecmds = dict.fromkeys(['#domain', '#dx_dy_dz', '#time_window', '#title', '#messages', '#num_threads', '#time_step_stability_factor', '#pml_formulation', '#pml_cells', '#excitation_file', '#src_steps', '#rx_steps', '#taguchi', '#end_taguchi', '#output_dir'], None)

    # Commands that there can be multiple instances of in a model - these will be lists within the dictionary
    multiplecmds = {key: [] for key in ['#geometry_view', '#geometry_objects_write', '#material', '#soil_peplinski', '#add_dispersion_debye', '#add_dispersion_lorentz', '#add_dispersion_drude', '#waveform', '#voltage_source', '#hertzian_dipole', '#magnetic_dipole', '#transmission_line', '#rx', '#rx_array', '#snapshot', '#pml_cfs', '#include_file']}

    # Geometry object building commands that there can be multiple instances
    # of in a model - these will be lists within the dictionary
    geometrycmds = ['#geometry_objects_read', '#edge', '#plate', '#triangle', '#box', '#sphere', '#cylinder', '#cylindrical_sector', '#fractal_box', '#add_surface_roughness', '#add_surface_water', '#add_grass']
    # List to store all geometry object commands in order from input file
    geometry = []

    # Check if command names are valid, if essential commands are present, and
    # add command parameters to appropriate dictionary values or lists
    countessentialcmds = 0
    lindex = 0
    while(lindex < len(processedlines)):
        cmd = processedlines[lindex].split(':')
        cmdname = cmd[0]
        cmdparams = cmd[1]

        # Check if there is space between command name and parameters, i.e.
        # check first character of parameter string. Ignore case when there
        # are no parameters for a command, e.g. for #taguchi:
        if ' ' not in cmdparams[0] and len(cmdparams.strip('\n')) != 0:
            raise CmdInputError('There must be a space between the command name and parameters in ' + processedlines[lindex])

        # Check if command name is valid
        if cmdname not in essentialcmds and cmdname not in singlecmds and cmdname not in multiplecmds and cmdname not in geometrycmds:
            raise CmdInputError('Your input file contains an invalid command: ' + cmdname)

        # Count essential commands
        if cmdname in essentialcmds:
            countessentialcmds += 1

        # Assign command parameters as values to dictionary keys
        if cmdname in singlecmds:
            if singlecmds[cmdname] is None:
                singlecmds[cmdname] = cmd[1].strip(' \t\n')
            else:
                raise CmdInputError('You can only have a single instance of ' + cmdname + ' in your model')

        elif cmdname in multiplecmds:
            multiplecmds[cmdname].append(cmd[1].strip(' \t\n'))

        elif cmdname in geometrycmds:
            geometry.append(processedlines[lindex].strip(' \t\n'))

        lindex += 1

    if checkessential:
        if (countessentialcmds < len(essentialcmds)):
            raise CmdInputError('Your input file is missing essential commands required to run a model. Essential commands are: ' + ', '.join(essentialcmds))

    return singlecmds, multiplecmds, geometry
