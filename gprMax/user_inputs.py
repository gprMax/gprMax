from .exceptions import CmdInputError
from .subgrids.base import SubGridBase
from .utilities import round_value

import numpy as np
from colorama import init
from colorama import Fore
from colorama import Style

init()

"""Module contains classes to handle points supplied by a user. The
classes implement a common interface such that geometry building objects
such as Box or triangle do not need to have any knowledge which grid to which they
are rounding continuous points or checking the point is within the grid.
Additionally all logic related to rounding points etc is encapulsated here.
"""


def create_user_input_points(grid):
    """Return a point checker class based on the grid supplied."""
    if isinstance(grid, SubGridBase):
        return SubgridUserInput(grid)
    else:
        return MainGridUserInput(grid)


class UserInput:

    """Base class to handle (x, y, z) points supplied by the user."""

    def __init__(self, grid):
        self.grid = grid

    def point_within_bounds(self, p, cmd_str, name):
        try:
            self.grid.within_bounds(p)
        except ValueError as err:
            v = ['x', 'y', 'z']
            # Discretisation
            dl = getattr(self.grid, 'd' + err.args[0])
            # Incorrect index
            i = p[v.index(err.args[0])]
            if name:
                fs = "'{}' the {} {}-coordinate {:g} is not within the model domain"
                s = fs.format(cmd_str, err.args[0], name, i * dl)
            else:
                fs = "'{}' {}-coordinate {:g} is not within the model domain"
                s = fs.format(cmd_str, err.args[0], i * dl)
            raise CmdInputError(s)

    def discretise_point(self, p):
        """Function to get the index of a continuous point with the grid."""
        rv = np.vectorize(round_value)
        return rv(p / self.grid.dl)

    def round_to_grid(self, p):
        """Function to get the nearest continuous point on the grid from a continuous point in space."""
        return self.discretise_point(p) * self.grid.dl


class MainGridUserInput(UserInput):
    """Class to handle (x, y, z) points supplied by the user in the main grid."""

    def __init__(self, grid):
        super().__init__(grid)

    def check_point(self, p, cmd_str, name=''):
        p = self.discretise_point(p)
        self.point_within_bounds(p, cmd_str, name)
        return p

    def check_src_rx_point(self, p, cmd_str, name=''):
        p = self.check_point(p, cmd_str, name)

        if self.grid.within_pml(p):
            s = """WARNING: '{}' sources and receivers should not normally be \
            positioned within the PML."""
            fs = s.format(cmd_str)
            print(Fore.RED + fs + Style.RESET_ALL)

        return p

    def check_box_points(self, p1, p2, cmd_str):
        p1 = self.check_point(p1, cmd_str, name='lower')
        p2 = self.check_point(p2, cmd_str, name='upper')

        if np.greater(p1, p2).any():
            s = """'{}' the lower coordinates should be less than the upper \
            coordinates"""
            raise CmdInputError(s.format(cmd_str))

        return p1, p2

    def check_tri_points(self, p1, p2, p3, cmd_str):
        p1 = self.check_point(p1, cmd_str, name='vertex_1')
        p2 = self.check_point(p2, cmd_str, name='vertex_2')
        p3 = self.check_point(p3, cmd_str, name='vertex_3')

        return p1, p2, p3


class SubgridUserInput(MainGridUserInput):
    """Class to handle (x, y, z) points supplied by the user in the sub grid."""

    def __init__(self, grid):
        super().__init__(grid)

        # defines the region exposed to the user
        self.inner_bound = np.array([
                                    grid.pmlthickness['x0'] + grid.pml_separation,
                                    grid.pmlthickness['y0'] + grid.pml_separation,
                                    grid.pmlthickness['z0'] + grid.pml_separation])

        self.outer_bound = np.subtract([grid.nx, grid.ny, grid.nz],
                                       self.inner_bound)

    def translate_to_gap(self, p):
        """Function to translate the user input point to the real point in the
        subgrid"""
        return np.add(p, self.inner_bound)

    def discretise_point(self, p):
        """"Function to discretise a point. Does not provide any checks. The
        user enters coordinates relative to self.inner_bound. This function
        translate the user point to the correct index for building objects"""
        p = super().discretise_point(p)
        p_t = self.translate_to_gap(p)
        return p_t

    def round_to_grid(self, p):
        p_t = self.discretise_point(p)
        p_m = p_t * self.grid.dl
        return p_m

    def check_point(self, p, cmd_str, name=''):
        p_t = super().check_point(p, cmd_str, name)

        # Provide user within a warning if they have placed objects within
        # the OS non-working region.
        if (np.less(p_t, self.inner_bound).any() or
            np.greater(p_t, self.outer_bound).any()):
                s = """WARNING: '{}' This object traverses the Outer Surface. This \
                is an advanced feature."""
                print(Fore.RED + s.format(cmd_str) + Style.RESET_ALL)

        return p_t
