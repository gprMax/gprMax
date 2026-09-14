"""Merging must not damage inputs or publish partial outputs on failure."""

import os

import h5py
import numpy as np
import pytest

from gprMax.toolboxes.Utilities import outputfiles_merge as merger

pytestmark = pytest.mark.unit


def _inputs(tmp_path):
    paths = [tmp_path / f"input{index}.h5" for index in range(2)]
    for index, path in enumerate(paths):
        with h5py.File(path, "w") as output:
            output.attrs.update(Iterations=8, dt=1e-10, nrx=1)
            receiver = output.create_group("rxs/rx1")
            receiver.attrs.update(Name="probe", Position=(0.02,) * 3)
            receiver["Ez"] = np.arange(8.0) + index
    return paths


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
@pytest.mark.parametrize("input_index", [0, 1])
@pytest.mark.parametrize("removefiles", [False, True])
def test_merge_rejects_all_input_aliases_without_writes(tmp_path, alias, input_index, removefiles):
    inputs = _inputs(tmp_path)
    before = [path.read_bytes() for path in inputs]
    destination = tmp_path / "merged.h5"
    if alias == "same":
        destination = inputs[input_index]
    else:
        try:
            if alias == "symlink":
                destination.symlink_to(inputs[input_index])
            else:
                os.link(inputs[input_index], destination)
        except OSError as error:
            pytest.skip(f"Filesystem does not permit {alias}: {error}")
    with pytest.raises(ValueError, match="must not overwrite an input"):
        merger.merge_files(inputs, destination, removefiles=removefiles)
    assert [path.read_bytes() for path in inputs] == before


@pytest.mark.parametrize("exists", [False, True])
@pytest.mark.parametrize("removefiles", [False, True])
def test_failed_merge_preserves_destination_inputs_and_cleans_staging(tmp_path, monkeypatch, exists, removefiles):
    inputs = _inputs(tmp_path)
    before = [path.read_bytes() for path in inputs]
    destination = tmp_path / "merged.h5"
    previous = b"previous successful output"
    if exists:
        destination.write_bytes(previous)
    original = merger._write_trace_metadata

    def fail_second_column(*args):
        if args[2] == 1:
            raise OSError("injected write failure")
        return original(*args)

    monkeypatch.setattr(merger, "_write_trace_metadata", fail_second_column)
    with pytest.raises(OSError, match="injected write failure"):
        merger.merge_files(inputs, destination, removefiles=removefiles)
    assert [path.read_bytes() for path in inputs] == before
    assert destination.read_bytes() == previous if exists else not destination.exists()
    assert not list(tmp_path.glob(".merged.h5.*"))


def test_successful_merge_publishes_then_removes_inputs(tmp_path):
    inputs = _inputs(tmp_path)
    destination = tmp_path / "merged.h5"
    destination.write_bytes(b"replace me")
    result = merger.merge_files(inputs, destination, removefiles=True)
    assert result == destination
    assert not any(path.exists() for path in inputs)
    with h5py.File(destination) as output:
        np.testing.assert_array_equal(output["rxs/rx1/Ez"][:], np.arange(8.0)[:, None] + [0, 1])
    assert not list(tmp_path.glob(".merged.h5.*"))
