import sys
import unittest

# http://stackoverflow.com/a/17981937/1942837
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

# end stack copy

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
            rx(2, 1, 0, 'id', ['Ex', 'Ez'], polarisation='x', rotate90origin=(1, 1))  # no dxdy given

    def test_rx_rotate_success(self):
        with captured_output() as (out, err):
            rx(2, 1, 0, 'id', ['Ex', 'Ez'], polarisation='x', rotate90origin=(1, 1), dxdy=(0, 0))
        self.assert_output(out, '#rx: 1 2 0 id Ex Ez')  # note: x, y swapped

    def test_rx_rotate_success2(self):
        with captured_output() as (out, err):
            rx(2, 1, 0, 'id', ['Ex', 'Ez'], polarisation='y', rotate90origin=(1, 1), dxdy=(0, 0))
        self.assert_output(out, '#rx: 1 2 0 id Ex Ez')  # note: x, y swapped

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

if __name__ == '__main__':
    unittest.main()
