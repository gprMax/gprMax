# Copyright (C) 2015-2023: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
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

"""Tests for the --save option in plotting tools (plot_Ascan, plot_Bscan, etc.)."""

import os
import subprocess
import sys
import tempfile
import unittest


def _run_plot_tool(module_name, args):
    """Run a tools plot module with given args; returns (returncode, stdout, stderr)."""
    cmd = [sys.executable, '-m', 'tools.' + module_name] + args
    proc = subprocess.run(
        cmd,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc.returncode, proc.stdout, proc.stderr


class TestPlotSave(unittest.TestCase):
    """Verify that plotting tools create files when --save is used."""

    def test_plot_source_wave_save_creates_file(self):
        """plot_source_wave with --save creates the output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, 'subdir', 'waveform.png')
            code, out, err = _run_plot_tool('plot_source_wave', [
                'ricker', '1', '1e9', '6e-9', '1.926e-12', '--save', outpath
            ])
            self.assertEqual(code, 0, 'plot_source_wave failed: {}'.format(err or out))
            self.assertTrue(os.path.isfile(outpath), 'Expected file at {}'.format(outpath))
            self.assertGreater(os.path.getsize(outpath), 0, 'Output file should not be empty')

    def test_plot_Ascan_save_creates_file(self):
        """plot_Ascan with --save creates the output file."""
        outfile = os.path.join(os.path.dirname(__file__), 'models_basic', 'cylinder_Ascan_2D', 'cylinder_Ascan_2D.out')
        if not os.path.isfile(outfile):
            self.skipTest('Test output file not found: {}'.format(outfile))

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, 'plots', 'ascan.png')
            code, out, err = _run_plot_tool('plot_Ascan', [
                outfile, '--outputs', 'Ez', '--save', outpath
            ])
            self.assertEqual(code, 0, 'plot_Ascan failed: {}'.format(err or out))
            self.assertTrue(os.path.isfile(outpath), 'Expected file at {}'.format(outpath))
            self.assertGreater(os.path.getsize(outpath), 0, 'Output file should not be empty')

    def test_plot_Bscan_save_creates_file(self):
        """plot_Bscan with --save creates the output file (requires merged B-scan .out)."""
        # plot_Bscan expects 2D data (traces x time); use merged file if present
        merged = os.path.join(os.path.dirname(__file__), 'models_basic', 'cylinder_Bscan_2D_merged', 'cylinder_Bscan_2D_merged.out')
        if not os.path.isfile(merged):
            merged = os.path.join(os.path.dirname(__file__), 'user_models', 'cylinder_Bscan_2D_merged.out')
        if not os.path.isfile(merged):
            self.skipTest('Merged B-scan output file not found (run outputfiles_merge first)')

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, 'bscan.png')
            code, out, err = _run_plot_tool('plot_Bscan', [
                merged, 'Ez', '--save', outpath
            ])
            self.assertEqual(code, 0, 'plot_Bscan failed: {}'.format(err or out))
            self.assertTrue(os.path.isfile(outpath), 'Expected file at {}'.format(outpath))
            self.assertGreater(os.path.getsize(outpath), 0, 'Output file should not be empty')


if __name__ == '__main__':
    unittest.main()
