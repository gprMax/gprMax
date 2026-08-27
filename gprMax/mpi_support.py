# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
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

"""Runtime boundary for gprMax MPI functionality."""

from types import ModuleType


class MPIUnavailableError(RuntimeError):
    """Raised when a requested MPI feature cannot initialise mpi4py."""


def require_mpi(feature: str = "MPI functionality") -> ModuleType:
    """Import and return :mod:`mpi4py.MPI` for a requested MPI feature."""

    try:
        from mpi4py import MPI
    except (ImportError, RuntimeError) as exc:
        raise MPIUnavailableError(
            f"{feature} requires a working MPI runtime and the mpi4py package. "
            "Install an MPI implementation supported by your platform, then "
            "install mpi4py."
        ) from exc

    return MPI
