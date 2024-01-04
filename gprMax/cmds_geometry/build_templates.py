# Copyright (C) 2015-2024: The University of Edinburgh, United Kingdom
#                 Authors: Craig Warren, Antonis Giannopoulos, and John Hartley
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

from jinja2 import Environment, PackageLoader

env = Environment(
    loader=PackageLoader(__name__, "templates"),
)

template = env.get_template("fields_updates_dispersive_template")

r = template.render(
    functions=[
        # name, double, real
        {
            "name_a": "update_electric_dispersive_multipole_A_double_real",
            "name_b": "update_electric_dispersive_multipole_B_double_real",
            "name_a_1": "update_electric_dispersive_1pole_A_double_real",
            "name_b_1": "update_electric_dispersive_1pole_B_double_real",
            "field_type": "double",
            "dispersive_type": "double",
        },
        # name, float, real
        {
            "name_a": "update_electric_dispersive_multipole_A_float_real",
            "name_b": "update_electric_dispersive_multipole_B_float_real",
            "name_a_1": "update_electric_dispersive_1pole_A_float_real",
            "name_b_1": "update_electric_dispersive_1pole_B_float_real",
            "field_type": "float",
            "dispersive_type": "float",
        },
        # name, double, complex
        {
            "name_a": "update_electric_dispersive_multipole_A_double_complex",
            "name_b": "update_electric_dispersive_multipole_B_double_complex",
            "name_a_1": "update_electric_dispersive_1pole_A_double_complex",
            "name_b_1": "update_electric_dispersive_1pole_B_double_complex",
            "field_type": "double",
            "dispersive_type": "double complex",
            "real_part": "creal",
        },
        # name, float, complex
        {
            "name_a": "update_electric_dispersive_multipole_A_float_complex",
            "name_b": "update_electric_dispersive_multipole_B_float_complex",
            "name_a_1": "update_electric_dispersive_1pole_A_float_complex",
            "name_b_1": "update_electric_dispersive_1pole_B_float_complex",
            "field_type": "float",
            "dispersive_type": "float complex",
            "real_part": "crealf",
        },
    ]
)

with open("cython/dispersive_updates_test.pyx", "w") as f:
    f.write(r)
