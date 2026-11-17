"""Include traversal shared by model parsing and non-executing preflight.

This module deliberately has no simulation/configuration dependencies: study
preflight runs before a SimulationConfig exists.
"""

from pathlib import Path

from gprMax.utilities.input_io import read_input_text


def expand_include_lines(lines, base_dir, stack=()):
    """Expand each occurrence in order, relative to its containing file.

    An active-path stack detects cycles without suppressing repeated includes.
    Included Python blocks remain unsupported. Top-level Python is handled by
    the caller (executed by parsing, skipped by preflight).
    """
    expanded = []
    for line in lines:
        if not line.startswith("#include_file:"):
            expanded.append(line)
            continue
        tokens = line.split()
        if len(tokens) != 2:
            raise ValueError("#include_file requires exactly one parameter")
        path = Path(tokens[1])
        if not path.is_absolute():
            path = Path(base_dir) / path
        path = path.resolve()
        if path in stack:
            chain = " -> ".join(str(item) for item in (*stack, path))
            raise ValueError(f"#include_file cycle detected: {chain}")
        if len(stack) >= 100:
            raise ValueError(f"#include_file nesting exceeds 100 files at {path}")
        included = [
            item.rstrip() + "\n"
            for item in read_input_text(path).splitlines()
            if not item.startswith("##") and item.strip()
        ]
        if any(item.startswith(("#python:", "#end_python:")) for item in included):
            raise SyntaxError(f"Python blocks in included files are unsupported: {path}")
        expanded.extend(expand_include_lines(included, path.parent, (*stack, path)))
    return expanded


def static_hash_commands(inputfile):
    """Read literal commands without executing top-level legacy Python.

    Match the parser's column-zero command convention. Commands printed by
    Python cannot participate in preflight; the parser still expands their
    includes later using the same traversal.
    """
    if inputfile is None:
        return []
    path = Path(inputfile).expanduser().resolve()
    commands = []
    in_python = False
    for line in read_input_text(path).splitlines():
        if line.startswith("#python:"):
            in_python = True
        elif line.startswith("#end_python:"):
            in_python = False
        elif not in_python and line.startswith("#") and not line.startswith("##"):
            commands.append(line.rstrip() + "\n")
    if in_python:
        raise SyntaxError("Cannot find the end of the Python code block: missing #end_python: command.")
    return expand_include_lines(commands, path.parent, (path,))
