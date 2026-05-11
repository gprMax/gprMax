import os
import sys
import tempfile
import unittest

# http://stackoverflow.com/a/17981937/1942837
from contextlib import contextmanager
from io import StringIO
from unittest.mock import patch

import numpy as np


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
from gprMax.exceptions import CmdInputError
from gprMax.geometry_outputs import GeometryView
from gprMax.input_cmds_file import check_cmd_names
from gprMax.input_cmds_multiuse import process_multicmds
from gprMax.materials import Material


class DummyProgressBar(object):
    def __init__(self):
        self.total = 0

    def update(self, n=1):
        self.total += n


class DummyGrid(object):
    def __init__(self):
        self.messages = False
        self.nx = 10
        self.ny = 10
        self.nz = 10
        self.dx = 0.1
        self.dy = 0.1
        self.dz = 0.1
        self.geometryviews = []

    def calculate_coord(self, coord, val):
        return int(round(float(val) / getattr(self, 'd' + coord)))

    def within_bounds(self, **kwargs):
        for coord, val in kwargs.items():
            if val < 0 or val > getattr(self, 'n' + coord):
                raise ValueError(coord)


def process_geometry_view_cmd(command):
    G = DummyGrid()
    processedlines = [command + '\n']
    _, multicmds, _ = check_cmd_names(processedlines, checkessential=False)
    process_multicmds(multicmds, G)
    return G


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

    def test_geometry_view_export_properties(self):
        with captured_output() as (out, err):
            geometry_view(0, 0, 0, 1, 1, 1, 0.1, 0.1, 0.1, 'geo', export_properties=True)
        self.assert_output(out, '#geometry_view: 0 0 0 1 1 1 0.1 0.1 0.1 geo n p')

    def test_geometry_view_p_flag_parses_for_normal_view(self):
        G = process_geometry_view_cmd('#geometry_view: 0 0 0 1 1 1 0.1 0.1 0.1 geo n p')

        self.assertEqual(len(G.geometryviews), 1)
        self.assertTrue(G.geometryviews[0].export_properties)
        self.assertEqual(G.geometryviews[0].fileext, '.vti')

    def test_geometry_view_p_flag_rejects_fine_view(self):
        with self.assertRaises(CmdInputError):
            process_geometry_view_cmd('#geometry_view: 0 0 0 1 1 1 0.1 0.1 0.1 geo f p')

    def test_geometry_view_rejects_unknown_twelfth_parameter(self):
        with self.assertRaises(CmdInputError):
            process_geometry_view_cmd('#geometry_view: 0 0 0 1 1 1 0.1 0.1 0.1 geo n x')

    def test_geometry_view_property_files_are_written(self):
        G = DummyGrid()
        G.nx = 2
        G.ny = 2
        G.nz = 2
        G.inputdirectory = ''
        G.pmls = []
        G.hertziandipoles = []
        G.magneticdipoles = []
        G.voltagesources = []
        G.transmissionlines = []
        G.rxs = []

        pec = Material(0, 'pec')
        pec.se = float('inf')
        free_space = Material(1, 'free_space')
        soil = Material(2, 'soil')
        soil.er = 9.0
        soil.se = 0.01
        G.materials = [pec, free_space, soil]

        G.solid = np.ones((2, 2, 2), dtype=np.uint32)

        def define_geometry(*args):
            solid_geometry = args[-3]
            srcs_pml_geometry = args[-2]
            rxs_geometry = args[-1]
            solid_geometry[:] = np.array([1, 2, 2, 1, 0, 2, 1, 2], dtype=np.uint32)
            srcs_pml_geometry[:] = 0
            rxs_geometry[:] = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            geometryview = GeometryView(0, 0, 0, 2, 2, 2, 1, 1, 1, 'geo', '.vti', export_properties=True)
            geometryview.filename = os.path.join(tmpdir, 'geo.vti')

            with patch('gprMax.geometry_outputs.define_normal_geometry', side_effect=define_geometry):
                geometryview.write_vtk(G, DummyProgressBar())

            for suffix, arrayname in (('_relative_permittivity', 'relative_permittivity'), ('_conductivity', 'conductivity')):
                filepath = os.path.join(tmpdir, 'geo' + suffix + '.vti')
                geometryview.write_property_vti(G, DummyProgressBar(), geometryview.solid_geometry, geometryview.srcs_pml_geometry, geometryview.rxs_geometry, arrayname, filepath)
                self.assertTrue(os.path.exists(filepath))
                with open(filepath, 'rb') as f:
                    data = f.read()
                self.assertIn(('Name="{}"'.format(arrayname)).encode(), data)
                self.assertIn(b'<DataArray type="Float32"', data)

if __name__ == '__main__':
    unittest.main()
