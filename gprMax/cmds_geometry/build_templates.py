from jinja2 import Environment, PackageLoader, select_autoescape
env = Environment(
    loader=PackageLoader(__name__, 'templates'),
)

template = env.get_template('fields_updates_dispersive_template')

r = template.render(
    functions=[
        # name, double, real
        {
            'name_a': 'update_electric_dispersive_multipole_A_double_real',
            'name_b': 'update_electric_dispersive_multipole_B_double_real',
            'name_a_1': 'update_electric_dispersive_1pole_A_double_real',
            'name_b_1': 'update_electric_dispersive_1pole_B_double_real',
            'field_type': 'double',
            'dispersive_type': 'double'
        },
        # name, float, real
        {
            'name_a': 'update_electric_dispersive_multipole_A_float_real',
            'name_b': 'update_electric_dispersive_multipole_B_float_real',
            'name_a_1': 'update_electric_dispersive_1pole_A_float_real',
            'name_b_1': 'update_electric_dispersive_1pole_B_float_real',
            'field_type': 'float',
            'dispersive_type': 'float'
        },
        # name, double, complex
        {
            'name_a': 'update_electric_dispersive_multipole_A_double_complex',
            'name_b': 'update_electric_dispersive_multipole_B_double_complex',
            'name_a_1': 'update_electric_dispersive_1pole_A_double_complex',
            'name_b_1': 'update_electric_dispersive_1pole_B_double_complex',
            'field_type': 'double',
            'dispersive_type': 'double complex',
            'real_part': 'creal'
        },
        # name, float, complex
        {
            'name_a': 'update_electric_dispersive_multipole_A_float_complex',
            'name_b': 'update_electric_dispersive_multipole_B_float_complex',
            'name_a_1': 'update_electric_dispersive_1pole_A_float_complex',
            'name_b_1': 'update_electric_dispersive_1pole_B_float_complex',
            'field_type': 'float',
            'dispersive_type': 'float complex',
            'real_part': 'crealf'
        }]
)

f = open('cython/dispersive_updates_test.pyx', 'w')
f.write(r)
f.close()
