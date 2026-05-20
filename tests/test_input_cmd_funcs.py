# import sys
# import unittest

# from contextlib import contextmanager
# from io import StringIO


# @contextmanager
# def captured_output():
#     new_out, new_err = StringIO(), StringIO()
#     old_out, old_err = sys.stdout, sys.stderr
#     try:
#         sys.stdout, sys.stderr = new_out, new_err
#         yield sys.stdout, sys.stderr
#     finally:
#         sys.stdout, sys.stderr = old_out, old_err


# from gprMax.input_cmd_funcs import *


# class My_input_cmd_funcs_test(unittest.TestCase):

#     def assert_output(self, out, expected_out):
#         """helper function"""
#         output = out.getvalue().strip()
#         self.assertEqual(output, expected_out)

#     # --- existing tests ---

#     def test_rx(self):
#         with captured_output() as (out, err):
#             rx(0, 0, 0)
#         self.assert_output(out, '#rx: 0 0 0')

#     def test_rx2(self):
#         with captured_output() as (out, err):
#             rx(0, 1, 2, 'id')
#         self.assert_output(out, '#rx: 0 1 2 id')

#     def test_rx3(self):
#         with captured_output() as (out, err):
#             rx(2, 1, 0, 'idd', ['Ex'])
#         self.assert_output(out, '#rx: 2 1 0 idd Ex')

#     def test_rx4(self):
#         with captured_output() as (out, err):
#             rx(2, 1, 0, 'id', ['Ex', 'Ez'])
#         self.assert_output(out, '#rx: 2 1 0 id Ex Ez')

#     def test_rx_rotate_exception(self):
#         with self.assertRaises(ValueError):
#             rx(2, 1, 0, 'id', ['Ex', 'Ez'], polarisation='x', rotate90origin=(1, 1))

#     def test_rx_rotate_success(self):
#         with captured_output() as (out, err):
#             rx(2, 1, 0, 'id', ['Ex', 'Ez'], polarisation='x', rotate90origin=(1, 1), dxdy=(0, 0))
#         self.assert_output(out, '#rx: 1 2 0 id Ex Ez')

#     def test_rx_rotate_success2(self):
#         with captured_output() as (out, err):
#             rx(2, 1, 0, 'id', ['Ex', 'Ez'], polarisation='y', rotate90origin=(1, 1), dxdy=(0, 0))
#         self.assert_output(out, '#rx: 1 2 0 id Ex Ez')

#     def test_src_steps(self):
#         with captured_output() as (out, err):
#             src_steps()
#         self.assert_output(out, '#src_steps: 0 0 0')

#     def test_src_steps2(self):
#         with captured_output() as (out, err):
#             src_steps(42, 43, 44.2)
#         self.assert_output(out, '#src_steps: 42 43 44.2')

#     def test_rx_steps(self):
#         with captured_output() as (out, err):
#             rx_steps()
#         self.assert_output(out, '#rx_steps: 0 0 0')

#     def test_rx_steps2(self):
#         with captured_output() as (out, err):
#             rx_steps(42, 43, 44.2)
#         self.assert_output(out, '#rx_steps: 42 43 44.2')

#     # --- new tests ---

#     def test_waveform(self):
#         with captured_output() as (out, err):
#             waveform('gaussian', 1, 1e9, 'mywaveform')
#         self.assert_output(out, '#waveform: gaussian 1 1e+09 mywaveform')

#     def test_hertzian_dipole(self):
#         with captured_output() as (out, err):
#             hertzian_dipole('z', 0, 0, 0, 'mywaveform')
#         self.assert_output(out, '#hertzian_dipole: z 0 0 0 mywaveform')

#     def test_magnetic_dipole(self):
#         with captured_output() as (out, err):
#             magnetic_dipole('z', 0, 0, 0, 'mywaveform')
#         self.assert_output(out, '#magnetic_dipole: z 0 0 0 mywaveform')

#     def test_transmission_line(self):
#         with captured_output() as (out, err):
#             transmission_line('z', 0, 0, 0, 50, 'mywaveform')
#         self.assert_output(out, '#transmission_line: z 0 0 0 50 mywaveform')

#     def test_box(self):
#         with captured_output() as (out, err):
#             box(0, 0, 0, 1, 1, 1, 'free_space')
#         self.assert_output(out, '#box: 0 0 0 1 1 1 free_space')

#     def test_cylinder(self):
#         with captured_output() as (out, err):
#             cylinder(0, 0, 0, 1, 1, 1, 0.1, 'free_space')
#         self.assert_output(out, '#cylinder: 0 0 0 1 1 1 0.1 free_space')

#     def test_sphere(self):
#         with captured_output() as (out, err):
#             sphere(0, 0, 0, 0.1, 'free_space')
#         self.assert_output(out, '#sphere: 0 0 0 0.1 free_space')


# if __name__ == '__main__':
#     unittest.main()
import sys
import unittest

from contextlib import contextmanager
from io import StringIO


@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


from gprMax.input_cmd_funcs import *


class My_input_cmd_funcs_test(unittest.TestCase):
    def assert_output(self, out, expected_out):
        """helper function"""
        output = out.getvalue().strip()
        self.assertEqual(output, expected_out)

    def test_rx(self):
        with captured_output() as (out, err):
            rx(0, 0, 0)
        self.assert_output(out, '#rx: 0 0 0')

    def test_rx2(self):
        with captured_output() as (out, err):
            rx(0, 1, 2, 'id')
        self.assert_output(out, '#rx: 0 1 2 id')

    def test_rx3(self):
        with captured_output() as (out, err):
            rx(2, 1, 0, 'idd', ['Ex'])
        self.assert_output(out, '#rx: 2 1 0 idd Ex')

    def test_rx4(self):
        with captured_output() as (out, err):
            rx(2, 1, 0, 'id', ['Ex', 'Ez'])
        self.assert_output(out, '#rx: 2 1 0 id Ex Ez')

    def test_rx_rotate_exception(self):
        with self.assertRaises(ValueError):
            rx(2, 1, 0, 'id', ['Ex', 'Ez'], polarisation='x', rotate90origin=(1, 1))

    def test_rx_rotate_success(self):
        with captured_output() as (out, err):
            rx(2, 1, 0, 'id', ['Ex', 'Ez'], polarisation='x', rotate90origin=(1, 1), dxdy=(0, 0))
        self.assert_output(out, '#rx: 1 2 0 id Ex Ez')

    def test_rx_rotate_success2(self):
        with captured_output() as (out, err):
            rx(2, 1, 0, 'id', ['Ex', 'Ez'], polarisation='y', rotate90origin=(1, 1), dxdy=(0, 0))
        self.assert_output(out, '#rx: 1 2 0 id Ex Ez')

    def test_src_steps(self):
        with captured_output() as (out, err):
            src_steps()
        self.assert_output(out, '#src_steps: 0 0 0')

    def test_src_steps2(self):
        with captured_output() as (out, err):
            src_steps(42, 43, 44.2)
        self.assert_output(out, '#src_steps: 42 43 44.2')

    def test_rx_steps(self):
        with captured_output() as (out, err):
            rx_steps()
        self.assert_output(out, '#rx_steps: 0 0 0')

    def test_rx_steps2(self):
        with captured_output() as (out, err):
            rx_steps(42, 43, 44.2)
        self.assert_output(out, '#rx_steps: 42 43 44.2')

    def test_waveform(self):
        with captured_output() as (out, err):
            waveform('gaussian', 1, 1e9, 'mywaveform')
        # Note: {:g} format outputs '1e+09' on Python 3.14+
        self.assert_output(out, '#waveform: gaussian 1 1e+09 mywaveform')

    def test_hertzian_dipole(self):
        with captured_output() as (out, err):
            hertzian_dipole('z', 0, 0, 0, 'mywaveform')
        self.assert_output(out, '#hertzian_dipole: z 0 0 0 mywaveform')

    def test_magnetic_dipole(self):
        with captured_output() as (out, err):
            magnetic_dipole('z', 0, 0, 0, 'mywaveform')
        self.assert_output(out, '#magnetic_dipole: z 0 0 0 mywaveform')

    def test_transmission_line(self):
        with captured_output() as (out, err):
            transmission_line('z', 0, 0, 0, 50, 'mywaveform')
        self.assert_output(out, '#transmission_line: z 0 0 0 50 mywaveform')

    def test_box(self):
        with captured_output() as (out, err):
            box(0, 0, 0, 1, 1, 1, 'free_space')
        self.assert_output(out, '#box: 0 0 0 1 1 1 free_space')

    def test_cylinder(self):
        with captured_output() as (out, err):
            cylinder(0, 0, 0, 1, 1, 1, 0.1, 'free_space')
        self.assert_output(out, '#cylinder: 0 0 0 1 1 1 0.1 free_space')

    def test_sphere(self):
        with captured_output() as (out, err):
            sphere(0, 0, 0, 0.1, 'free_space')
        self.assert_output(out, '#sphere: 0 0 0 0.1 free_space')


if __name__ == '__main__':
    unittest.main()