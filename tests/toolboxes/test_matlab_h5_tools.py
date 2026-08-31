# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
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
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.

"""Integration tests for the MATLAB HDF5 reader and MAT-file converter."""

from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np
import pytest


MATLAB_TOOLS = Path(__file__).parents[2] / "toolboxes" / "Utilities" / "MATLAB"


def _matlab_path(path):
    return str(path).replace("'", "''")


def _write_fixture(path, offset=0):
    with h5py.File(path, "w") as output:
        output.attrs["Title"] = "MATLAB conversion fixture"
        output.attrs["Iterations"] = 4
        output.attrs["dt"] = 1e-11
        output.attrs["geometry_fixed"] = True
        receiver = output.create_group("rxs/rx1")
        receiver.attrs["Position"] = (0.1, 0.2, 0.3)
        receiver.attrs["Name"] = "test receiver"
        ex = receiver.create_dataset("Ex", data=np.arange(4, dtype=np.float32) + offset)
        ex.attrs["SampleInterval"] = 1e-11
        ex.attrs["TimeSampleOffset"] = -0.5e-11
        ez = receiver.create_dataset(
            "Ez",
            data=np.arange(12, dtype=np.float32).reshape(4, 3) + offset,
        )
        ez.attrs["SampleInterval"] = 1e-11
        ez.attrs["TimeSampleOffset"] = 0.0
        trace_receiver = output.create_group("trace_metadata/rxs/rx1")
        trace_receiver.create_dataset(
            "Position",
            data=np.asarray([[0.1, 0.2, 0.3], [0.12, 0.2, 0.3], [0.15, 0.2, 0.3]]),
        )
        port = output.create_group("port-feed")
        port.attrs["empty_faces"] = np.asarray([], dtype="S5")
        port.create_dataset(
            "S",
            data=(np.arange(12, dtype=np.float32).reshape(3, 2, 2) + 1j * (offset + 1)).astype(np.complex64),
        )
        port.create_dataset("valid", data=np.ones((3, 2, 2), dtype=np.uint8))
        port.create_dataset("enabled", data=np.ones((3,), dtype=np.bool_))
        receive = output.create_group("ports/receive")
        voltage = receive.create_dataset(
            "Vtotal",
            data=np.arange(12, dtype=np.float64).reshape(4, 3) + offset,
        )
        voltage.attrs["SampleInterval"] = 1e-11
        voltage.attrs["TimeSampleOffset"] = 0.5e-11
        voltage.attrs["Units"] = "V"
        generator = receive.create_dataset("Vgenerator", data=np.arange(4, dtype=np.float64) + offset)
        generator.attrs["SampleInterval"] = 1e-11
        generator.attrs["TimeSampleOffset"] = 0.5e-11
        generator.attrs["Units"] = "V"
        trace_port = output.create_group("trace_metadata/ports/receive")
        trace_port.create_dataset(
            "Position",
            data=np.asarray([[0.1, 0.2, 0.3], [0.12, 0.2, 0.3], [0.15, 0.2, 0.3]]),
        )
        output.create_group("unselected").create_dataset("large", data=np.ones(100))


@pytest.mark.integration
@pytest.mark.slow
def test_matlab_reader_and_converter_round_trip(tmp_path):
    matlab = shutil.which("matlab")
    if matlab is None:
        pytest.skip("MATLAB is not installed")

    first = tmp_path / "case-one.h5"
    second = tmp_path / "case0x2Done.h5"
    single_mat = tmp_path / "single-output.mat"
    batch_mat = tmp_path / "batch-output.mat"
    _write_fixture(first)
    _write_fixture(second, offset=100)

    script = f"""
addpath('{_matlab_path(MATLAB_TOOLS)}');
g = gprmax_read_h5('{_matlab_path(first)}', ...
    'Paths', ["/rxs", "/port-feed"]);
assert(isequal(size(g.data.rxs.rx1.Ez), [4 3]));
assert(isa(g.data.rxs.rx1.Ez, 'single'));
portName = matlab.lang.makeValidName('port-feed', 'ReplacementStyle', 'hex');
assert(~isreal(g.data.(portName).S));
assert(isa(g.data.(portName).S, 'single'));
assert(isequal(size(g.data.(portName).S), [3 2 2]));
assert(isa(g.data.(portName).valid, 'uint8'));
assert(islogical(g.data.(portName).enabled));
assert(islogical(g.header.geometry_fixed) && g.header.geometry_fixed);
assert(~isfield(g.data, 'unselected'));
[resolvedS, resolvedMetadata] = gprmax_h5_get(g, '/port-feed/S');
assert(isequal(resolvedS, g.data.(portName).S));
assert(resolvedMetadata.hdf5_path == "/port-feed/S");
portMetadata = g.metadata([g.metadata.hdf5_path] == "/port-feed");
emptyAttribute = portMetadata.attributes( ...
    [portMetadata.attributes.name] == "empty_faces");
assert(isempty(emptyAttribute.value));

gprmax_h5_to_mat('{_matlab_path(first)}', ...
    'OutputFile', '{_matlab_path(single_mat)}', ...
    'Paths', '/port-feed');
singleInfo = whos('-file', '{_matlab_path(single_mat)}');
expectedSingle = matlab.lang.makeValidName( ...
    'case-one', 'ReplacementStyle', 'hex');
assert(strcmp(singleInfo.name, expectedSingle));
loadedSingle = load('{_matlab_path(single_mat)}');
assert(~isreal(loadedSingle.(expectedSingle).data.(portName).S));

gprmax_h5_to_mat( ...
    {{'{_matlab_path(first)}', '{_matlab_path(second)}'}}, ...
    'OutputFile', '{_matlab_path(batch_mat)}', ...
    'VariableName', 'comparison cases', ...
    'Paths', '/rxs/rx1/Ez');
batchInfo = whos('-file', '{_matlab_path(batch_mat)}');
assert(strcmp(batchInfo.name, 'comparisonCases'));
loadedBatch = load('{_matlab_path(batch_mat)}');
runNames = fieldnames(loadedBatch.comparisonCases.runs);
assert(numel(runNames) == 2);
assert(~strcmp(runNames{{1}}, runNames{{2}}));
assert(loadedBatch.comparisonCases.runs.(runNames{{1}}).data.rxs.rx1.Ez(1) == 0);
assert(loadedBatch.comparisonCases.runs.(runNames{{2}}).data.rxs.rx1.Ez(1) == 100);

individualFiles = gprmax_h5_to_mat( ...
    {{'{_matlab_path(first)}', '{_matlab_path(second)}'}}, ...
    'Paths', '/rxs/rx1/Ez');
firstInfo = whos('-file', individualFiles(1));
secondInfo = whos('-file', individualFiles(2));
assert(~strcmp(firstInfo.name, secondInfo.name));
delete(individualFiles(1));
delete(individualFiles(2));

plotDirectory = '{_matlab_path(tmp_path / "plots")}';
mkdir(plotDirectory);
[ascanFigures, traces] = plot_Ascan('{_matlab_path(first)}', ...
    'Receiver', 1, 'Outputs', 'Ex', 'FFT', true, ...
    'Visible', false, 'Save', true, 'OutputDirectory', plotDirectory);
assert(numel(ascanFigures) == 1);
assert(isequal(traces.rx1.Ex.values, single((0:3)')));
assert(abs(traces.rx1.Ex.time(1) + 0.5e-11) < eps);
close(ascanFigures);

[portFigure, portTrace] = plot_Ascan('{_matlab_path(first)}', ...
    'Path', '/ports/receive', 'Outputs', 'Vgenerator', 'Visible', false);
portFields = fieldnames(portTrace);
assert(numel(portFields) == 1);
assert(portTrace.(portFields{{1}}).Vgenerator.units == "V");
close(portFigure);

bscanFile = fullfile(plotDirectory, 'field_bscan.png');
[bscanFigure, bscan] = plot_Bscan('{_matlab_path(first)}', 'Ez', ...
    'Visible', false, 'OutputFile', bscanFile);
assert(isequal(size(bscan.values), [4 3]));
assert(bscan.positions_used);
assert(max(abs(bscan.coordinate - [0; 0.02; 0.05])) < 1e-12);
close(bscanFigure);

voltageFile = fullfile(plotDirectory, 'voltage_bscan.png');
[voltageFigure, voltageScan] = plot_Bscan( ...
    '{_matlab_path(first)}', 'Vtotal', 'Path', '/ports/receive', ...
    'Visible', false, 'OutputFile', voltageFile);
assert(voltageScan.units == "V");
assert(abs(voltageScan.time(1) - 0.5e-11) < eps);
close(voltageFigure);
assert(isfile(bscanFile) && isfile(voltageFile));

try
    gprmax_h5_to_mat('{_matlab_path(first)}', ...
        'OutputFile', '{_matlab_path(single_mat)}');
    error('gprMax:test:ExpectedOutputProtection', ...
        'Existing output was unexpectedly overwritten.');
catch exception
    assert(strcmp(exception.identifier, 'gprMax:MATLAB:OutputExists'));
end
"""
    script_path = tmp_path / "run_gprmax_matlab_tests.m"
    script_path.write_text(script, encoding="utf-8")
    completed = subprocess.run(
        [matlab, "-batch", f"run('{_matlab_path(script_path)}')"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
