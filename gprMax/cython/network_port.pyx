# cython: cdivision=True
# Copyright (C) 2026: The University of Edinburgh, United Kingdom
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <http://www.gnu.org/licenses/>.

from gprMax.config cimport float_or_double, float_or_double_complex


cpdef double update_rational_network_terminal(
    int i,
    int j,
    int k,
    double dl,
    double area,
    double source_coefficient,
    double denominator,
    double alpha,
    double conductance,
    double capacitance,
    double dt,
    double voltage_old,
    double generator_old,
    double generator_new,
    double generator_half,
    float_or_double_complex[::1] exp_half,
    float_or_double_complex[::1] coeff_half_new,
    float_or_double_complex[::1] coeff_half_old,
    float_or_double_complex[::1] exp_full,
    float_or_double_complex[::1] coeff_full_new,
    float_or_double_complex[::1] coeff_full_old,
    float_or_double_complex[::1] states,
    float_or_double[:, :, ::1] electric,
):
    """Correct one provisional electric edge and advance its sparse states."""

    cdef Py_ssize_t pole_index
    cdef double history
    cdef double voltage_new
    cdef double current_half
    cdef double input_old
    cdef double input_new
    cdef float_or_double_complex state_old

    input_old = voltage_old - generator_old
    history = (
        (conductance / 2.0 - capacitance / dt) * voltage_old
        - conductance * generator_half
        - capacitance * (generator_new - generator_old) / dt
    )
    for pole_index in range(states.shape[0]):
        state_old = (
            exp_half[pole_index] * states[pole_index]
            + coeff_half_old[pole_index] * input_old
            - coeff_half_new[pole_index] * generator_new
        )
        history = history + state_old.real

    electric[i, j, k] = (
        electric[i, j, k] + source_coefficient * history / area
    ) / denominator
    voltage_new = -dl * electric[i, j, k]
    current_half = alpha * voltage_new + history
    input_new = voltage_new - generator_new

    for pole_index in range(states.shape[0]):
        state_old = states[pole_index]
        states[pole_index] = (
            exp_full[pole_index] * state_old
            + coeff_full_new[pole_index] * input_new
            + coeff_full_old[pole_index] * input_old
        )

    return current_half
