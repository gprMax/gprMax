# Copyright (C) 2015-2025: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, John Hartley, 
#                          and Nathan Mannall
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

import logging
import sys
from io import StringIO
from pathlib import Path

import gprMax.config as config

from .hash_cmds_geometry import process_geometrycmds
from .hash_cmds_multiuse import process_multicmds
from .hash_cmds_singleuse import process_singlecmds

logger = logging.getLogger(__name__)


def process_python_include_code(inputfile, usernamespace):
    """Looks for and processes any Python code found in the input file.
    It will ignore any lines that are comments, i.e. begin with a
    double hash (##), and any blank lines. It will also ignore any
    lines that do not begin with a hash (#) after it has processed
    Python commands. It will also process any include file commands
    and insert the contents of the included file at that location.

    Args:
        inputfile: file object for input file.
        usernamespace: namespace that can be accessed by user in any Python code
                        blocks in input file.

    Returns:
        processedlines: list of input commands after Python processing.
    """

    # Strip out any newline characters and comments that must begin with double hashes
    inputlines = [
        line.rstrip() for line in inputfile if (not line.startswith("##") and line.rstrip("\n"))
    ]

    # Rewind input file in preparation for any subsequent reading function
    inputfile.seek(0)

    # List to hold final processed commands
    processedlines = []

    x = 0
    while x < len(inputlines):
        # Process any Python code
        if inputlines[x].startswith("#python:"):
            logger.warning(
                "#python blocks are deprecated and will be removed in "
                + "the next release of gprMax. Please convert your "
                + "model to use our Python API instead.\n"
            )
            # String to hold Python code to be executed
            pythoncode = ""
            x += 1
            while not inputlines[x].startswith("#end_python:"):
                # Add all code in current code block to string
                pythoncode += inputlines[x] + "\n"
                x += 1
                if x == len(inputlines):
                    logger.exception(
                        "Cannot find the end of the Python code "
                        + "block, i.e. missing #end_python: command."
                    )
                    raise SyntaxError
            # Compile code for faster execution
            pythoncompiledcode = compile(pythoncode, "<string>", "exec")
            # Redirect stdout to a text stream
            sys.stdout = result = StringIO()
            # Execute code block & make available only usernamespace
            exec(pythoncompiledcode, usernamespace)
            # String containing buffer of executed code
            codeout = result.getvalue().split("\n")
            result.close()

            # Reset stdio
            sys.stdout = sys.__stdout__

            # Separate commands from any other generated output
            hashcmds = []
            pythonout = []
            for line in codeout:
                if line.startswith("#"):
                    hashcmds.append(line + "\n")
                elif line:
                    pythonout.append(line)

            # Add commands to a list
            processedlines.extend(hashcmds)

            # Print any generated output that is not commands
            if pythonout:
                logger.info(f"Python messages (from stdout/stderr): {pythonout}\n")

        # Add any other commands to list
        elif inputlines[x].startswith("#"):
            # Add gprMax command to list
            inputlines[x] += "\n"
            processedlines.append(inputlines[x])

        x += 1

    # Process any include file commands
    processedlines = process_include_files(processedlines)

    return processedlines


def process_include_files(hashcmds):
    """Looks for and processes any include file commands and insert
        the contents of the included file at that location.

    Args:
        hashcmds: list of input commands.

    Returns:
        processedincludecmds: list of input commands after processing any
                                include file commands.
    """

    processedincludecmds = []
    x = 0
    while x < len(hashcmds):
        if hashcmds[x].startswith("#include_file:"):
            includefile = hashcmds[x].split()

            if len(includefile) != 2:
                logger.exception("#include_file requires exactly one parameter")
                raise ValueError

            includefile = includefile[1]

            # See if file exists at specified path and if not try input file directory
            includefile = Path(includefile)
            if not includefile.exists():
                includefile = Path(config.sim_config.input_file_path.parent, includefile)

            with open(includefile, "r") as f:
                # Strip out any newline characters and comments that must begin with double hashes
                includelines = [
                    includeline.rstrip() + "\n"
                    for includeline in f
                    if (not includeline.startswith("##") and includeline.rstrip("\n"))
                ]

            # Add lines from include file
            processedincludecmds.extend(includelines)

        else:
            processedincludecmds.append(hashcmds[x])

        x += 1

    return processedincludecmds


def write_processed_file(processedlines):
    """Writes an input file after any Python code and include commands
        in the original input file have been processed.

    Args:
        processedlines: list of input commands after after processing any
                        Python code and include commands.
    """

    parts = config.get_model_config().output_file_path.parts
    processedfile = Path(*parts[:-1], parts[-1] + "_processed.in")

    with open(processedfile, "w") as f:
        for item in processedlines:
            f.write(f"{item}")

    logger.info(
        f"Written input commands, after processing any Python code and include commands, to file: {processedfile}\n"
    )


def check_cmd_names(processedlines, checkessential=True):
    """Checks the validity of commands, i.e. are they gprMax commands,
        and that all essential commands are present.

    Args:
        processedlines: list of input commands after Python processing.
        checkessential: boolean to check for essential commands or not.

    Returns:
        singlecmds: dict of commands that can only occur once in the model.
        multiplecmds: dict of commands that can have multiple instances in the
                        model.
        geometry: list of geometry commands in the model.
    """

    # Dictionaries of available commands
    # Essential commands neccessary to run a gprMax model
    essentialcmds = ["#domain", "#dx_dy_dz", "#time_window"]

    # Commands that there should only be one instance of in a model
    singlecmds = dict.fromkeys(
        [
            "#domain",
            "#dx_dy_dz",
            "#time_window",
            "#title",
            "#omp_threads",
            "#time_step_stability_factor",
            "#pml_formulation",
            "#pml_cells",
            "#src_steps",
            "#rx_steps",
            "#output_dir",
        ],
        None,
    )

    # Commands that there can be multiple instances of in a model
    # - these will be lists within the dictionary
    multiplecmds = {
        key: []
        for key in [
            "#geometry_view",
            "#geometry_objects_write",
            "#material",
            "#material_range",
            "#material_list",
            "#soil_peplinski",
            "#add_dispersion_debye",
            "#add_dispersion_lorentz",
            "#add_dispersion_drude",
            "#waveform",
            "#voltage_source",
            "#hertzian_dipole",
            "#magnetic_dipole",
            "#transmission_line",
            "#plane_wave_angles",
            "#plane_wave_axial",
            "#plane_wave_vector",
            "#excitation_file",
            "#rx",
            "#rx_array",
            "#snapshot",
            "#pml_cfs",
            "#include_file",
        ]
    }

    # Geometry object building commands that there can be multiple instances
    # of in a model - these will be lists within the dictionary
    geometrycmds = [
        "#geometry_objects_read",
        "#edge",
        "#plate",
        "#triangle",
        "#box",
        "#sphere",
        "#ellipsoid",
        "#cone",
        "#cylinder",
        "#cylindrical_sector",
        "#fractal_box",
        "#add_surface_roughness",
        "#add_surface_water",
        "#add_grass",
    ]
    # List to store all geometry object commands in order from input file
    geometry = []

    # Check if command names are valid, if essential commands are present, and
    # add command parameters to appropriate dictionary values or lists
    countessentialcmds = 0
    lindex = 0
    while lindex < len(processedlines):
        cmd = processedlines[lindex].split(":")

        # Check the command name and parameters were both found
        if len(cmd) < 2:
            logger.error(
                f"Unable to identify command and parameters in '{processedlines[lindex].strip()}'."
                " There must be a colon ':' between the command name and parameters."
            )
            exit(1)

        cmdname = cmd[0]
        cmdparams = cmd[1]

        # Check if there is space between command name and parameters, i.e.
        # check first character of parameter string. Ignore case when there
        # are no parameters for a command, e.g. for #taguchi:
        if " " not in cmdparams[0] and len(cmdparams.strip("\n")) != 0:
            logger.exception(
                "There must be a space between the command name "
                + "and parameters in "
                + processedlines[lindex]
            )
            raise SyntaxError

        # Check if command name is valid
        if (
            cmdname not in essentialcmds
            and cmdname not in singlecmds
            and cmdname not in multiplecmds
            and cmdname not in geometrycmds
        ):
            logger.exception("Your input file contains an invalid command: " + cmdname)
            raise SyntaxError

        # Count essential commands
        if cmdname in essentialcmds:
            countessentialcmds += 1

        # Assign command parameters as values to dictionary keys
        if cmdname in singlecmds:
            if singlecmds[cmdname] is None:
                singlecmds[cmdname] = cmd[1].strip(" \t\n")
            else:
                logger.exception(
                    "You can only have a single instance of " + cmdname + " in your model"
                )
                raise SyntaxError

        elif cmdname in multiplecmds:
            multiplecmds[cmdname].append(cmd[1].strip(" \t\n"))

        elif cmdname in geometrycmds:
            geometry.append(processedlines[lindex].strip(" \t\n"))

        lindex += 1

    if checkessential and countessentialcmds < len(essentialcmds):
        logger.exception(
            "Your input file is missing essential commands "
            + "required to run a model. Essential commands are: "
            + ", ".join(essentialcmds)
        )
        raise SyntaxError

    return singlecmds, multiplecmds, geometry


def get_user_objects(processedlines, checkessential=True):
    """Make a list of all user objects.

    Args:
        processedlines: list of input commands after Python processing.
        checkessential: boolean to check for essential commands or not.

    Returns:
        user_objs: list of all user objects.
    """

    # Check validity of command names and that essential commands are present
    parsed_commands = check_cmd_names(processedlines, checkessential=checkessential)

    # Process parameters for commands that can only occur once in the model
    single_user_objs = process_singlecmds(parsed_commands[0])

    # Process parameters for commands that can occur multiple times in
    # the model
    multiple_user_objs = process_multicmds(parsed_commands[1])

    # Process geometry commands in the order they were given
    geometry_user_objs = process_geometrycmds(parsed_commands[2])

    user_objs = single_user_objs + multiple_user_objs + geometry_user_objs

    return user_objs


def parse_hash_commands(scene):
    """Parse user hash commands and add them to the scene.

    Args:
        scene: Scene object.

    Returns:
        scene: Scene object.
    """

    with open(config.sim_config.input_file_path) as inputfile:
        usernamespace = config.get_model_config().get_usernamespace()

        # Read input file and process any Python and include file commands
        processedlines = process_python_include_code(inputfile, usernamespace)

        # Print constants/variables in user-accessable namespace
        uservars = ""
        for key, value in sorted(usernamespace.items()):
            if key != "__builtins__":
                uservars += f"{key}: {value}, "
        logger.info(
            f"Constants/variables used/available for Python scripting: " + f"{{{uservars[:-2]}}}\n"
        )

        # Write a file containing the input commands after Python or include
        # file commands have been processed
        if config.sim_config.args.write_processed:
            write_processed_file(processedlines)

        user_objs = get_user_objects(processedlines, checkessential=True)
        for user_obj in user_objs:
            scene.add(user_obj)

        return scene
