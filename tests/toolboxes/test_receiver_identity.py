"""Identity-routing regressions: same receivers, deliberately permuted rxN."""

from dataclasses import replace

import h5py
import numpy as np
import pytest

from gprMax.toolboxes.Utilities.receiver_identity import (
    ReceiverIdentity,
    match_receiver,
    receiver_catalogue,
    select_receiver,
)
from gprMax.toolboxes.Utilities.outputfiles_merge import get_output_data, merge_files
from gprMax.toolboxes.Utilities.outputfiles_trace import collect_traces
from gprMax.toolboxes.Marimo.h5_reader import load_file, list_receivers as marimo_receivers
from gprMax.toolboxes.Marimo.trace_matrix import stack_traces, process_trace
from gprMax.toolboxes.SFCW.processing import load_receiver, list_receivers
from gprMax.toolboxes.FMCW.processing import Chirp, process_channel, process_incident_referenced_channel


def write_output(path, names=("zulu", "alpha"), *, new=False, shift=0, subgrid=False):
    with h5py.File(path, "w") as output:
        output.attrs.update(
            Iterations=32,
            dt=2e-11,
            nrx=len(names),
            nsrc=1,
            dx_dy_dz=(0.001,) * 3,
            nx_ny_nz=(100,) * 3,
            Title="identity",
        )
        grid = output.create_group("subgrids/fine") if subgrid else output
        if subgrid:
            grid.attrs.update(dict(output.attrs))
            output.attrs["nrx"] = 0
        if new:
            grid.attrs.update(
                ReceiverOrderSchemaVersion=1, ReceiverOrder="construction", ReceiverLayout="same-test-declarations"
            )
        src = grid.create_group("srcs/src1")
        src.attrs.update(Position=(shift, 0, 0), Type="HertzianDipole")
        excitation = src.create_group("excitation")
        excitation.attrs.update(SampleInterval=2e-11, TimeSampleOffset=0.0)
        excitation["samples"] = np.r_[1.0, np.zeros(31)]
        for index, name in enumerate(names):
            rx = grid.create_group(f"rxs/rx{index+1}")
            marker = 10 if name == "zulu" else 1
            rx.attrs.update(Position=(shift + marker, 0, 0), GridPosition=(marker, 0, 0))
            if name is not None:
                rx.attrs["Name"] = name
            if new:
                rx.attrs.update(BuildIndex=index, NameKind="user")
            for comp in ("Ex", "Ey", "Ez"):
                ds = rx.create_dataset(comp, data=np.r_[0.0, 0.0, marker, np.zeros(29)])
                ds.attrs.update(SampleInterval=2e-11, TimeSampleOffset=0.0)
    return path


@pytest.fixture
def permuted(tmp_path):
    return (write_output(tmp_path / "new.h5", new=True), write_output(tmp_path / "old.h5", ("alpha", "zulu"), shift=3))


def test_named_selection_and_natural_discovery(permuted, tmp_path):
    from gprMax.toolboxes.Optimisation.quantities import read_receiver

    for path in permuted:
        assert load_receiver(path, "name:zulu", "Ez").samples[2] == 10
        assert get_output_data(path, 1, "Ez", receiver_name="zulu")[0][2] == 10
        assert read_receiver(path, "zulu", "Ez").values[2] == 10
    many = write_output(tmp_path / "many.h5", tuple(f"r{i}" for i in range(13)))
    assert list(list_receivers(many)) == [f"/rxs/rx{i}" for i in range(1, 14)]
    assert marimo_receivers(load_file(many)) == [f"rx{i}" for i in range(1, 14)]


@pytest.mark.parametrize("reverse", (False, True))
def test_merge_and_export_follow_identity_and_positions(permuted, tmp_path, reverse):
    files = permuted[::-1] if reverse else permuted
    expected = 1 if reverse else 10
    traces, *_ = collect_traces(files, 1, "Ez")
    assert [trace.samples[2] for trace in traces] == [expected, expected]
    merged = merge_files(files, tmp_path / "merged.h5")
    with h5py.File(merged) as output:
        np.testing.assert_array_equal(output["rxs/rx1/Ez"][2], [expected, expected])
        assert output["trace_metadata/rxs/rx1/InputReceiverPath"].asstr()[:].tolist() == ["rxs/rx1", "rxs/rx2"]
        np.testing.assert_array_equal(
            output["trace_metadata/rxs/rx1/Position"][:, 0], [record.receiver_position[0] for record in traces]
        )


def test_marimo_stack_and_live_reader_follow_first_identity(permuted):
    files = [load_file(path) for path in permuted]
    result = stack_traces(files, "Ez", "rx1")
    assert not result["warnings"]
    np.testing.assert_array_equal(result["matrix"][2], [10, 10])
    first = process_trace(files[0], "Ez", None, "rx1")
    second = process_trace(files[1], "Ez", 32, "rx1", expected_identity=first["identity"])
    assert second["receiver"] == "rx2"
    assert second["array"][2] == 10


def test_fmcw_background_and_incident_follow_identity(permuted):
    chirp = Chirp(0.1e9, 1.1e9, 1e-3, 16)
    target, background = permuted
    response = process_channel(target, chirp, background_filename=background, receiver_path="name:zulu", component="Ez")
    np.testing.assert_array_equal(response.response, 0)
    assert response.background.receiver.path == "/rxs/rx2/Ez"
    incident = process_incident_referenced_channel(target, background, chirp, receiver_path="/rxs/rx1", component="Ez")
    np.testing.assert_array_equal(incident.response, 0)
    # A different incident/reference point is legitimate only when explicitly selected.
    explicit = process_channel(
        target,
        chirp,
        background_filename=background,
        receiver_path="/rxs/rx1",
        component="Ez",
        background_receiver_path="/rxs/rx1",
    )
    assert np.max(np.abs(explicit.response)) > 8


@pytest.mark.parametrize("names", [("different", "other"), ("zulu", "zulu"), (None, None)])
def test_ambiguous_cross_file_inputs_fail_before_merge_write(permuted, tmp_path, names):
    bad = write_output(tmp_path / "bad.h5", names)
    destination = tmp_path / "must-survive.h5"
    destination.write_bytes(b"do not truncate")
    with pytest.raises(ValueError, match="identity"):
        merge_files([permuted[0], bad], destination)
    assert destination.read_bytes() == b"do not truncate"
    with pytest.raises(ValueError, match="identity"):
        collect_traces([permuted[0], bad], 1, "Ez")
    stack = stack_traces([load_file(permuted[0]), load_file(bad)], "Ez", "rx1")
    assert len(stack["warnings"]) == 1
    assert stack["matrix"].shape == (32, 1)


def test_construction_identity_tracks_moving_generated_names():
    first = ReceiverIdentity(
        "rxs/rx1", "Rx(1,0,0)", name_kind="generated", build_index=0, layout="layout", name_unique=True, count=2
    )
    second = replace(first, name="Rx(2,0,0)")
    crossed = replace(first, path="rxs/rx2", build_index=1)
    assert match_receiver(first, {second.path: second, crossed.path: crossed}) == second
    duplicate = replace(first, name="duplicate", name_kind="user", name_unique=False)
    assert match_receiver(duplicate, {duplicate.path: duplicate}) == duplicate
    with pytest.raises(ValueError, match="identity"):
        match_receiver(duplicate, {duplicate.path: replace(duplicate, layout="different")})


def test_study_identity_is_authoritative_and_unique():
    first = ReceiverIdentity("rxs/rx1", "old", "moving", study_unique=True)
    other = replace(first, path="rxs/rx2", name="new")
    assert match_receiver(first, {other.path: other}) == other
    with pytest.raises(ValueError, match="StudyID"):
        match_receiver(first, {first.path: first, other.path: other})


def test_subgrid_matching_stays_in_its_namespace(tmp_path):
    first = write_output(tmp_path / "a.h5", new=True, subgrid=True)
    second = write_output(tmp_path / "b.h5", ("alpha", "zulu"), subgrid=True)
    traces, *_ = collect_traces([first, second], 1, "Ez", grid_path="subgrids/fine")
    assert [trace.samples[2] for trace in traces] == [10, 10]
    merged = merge_files([first, second], tmp_path / "sub.h5")
    with h5py.File(merged) as output:
        np.testing.assert_array_equal(output["subgrids/fine/rxs/rx1/Ez"][2], [10, 10])


def test_duplicate_metadata_or_explicit_name_is_rejected(tmp_path):
    path = write_output(tmp_path / "dup.h5", ("same", "same"), new=True)
    with h5py.File(path, "r+") as output:
        catalogue = receiver_catalogue(output)
        with pytest.raises(ValueError, match="2 matches"):
            select_receiver(catalogue, "name:same")
        assert select_receiver(catalogue, "build:1").path == "rxs/rx2"
        output["rxs/rx2"].attrs["BuildIndex"] = 0
        with pytest.raises(ValueError, match="duplicate"):
            receiver_catalogue(output)


def test_legacy_comparison_matches_receiver_not_group_number(permuted):
    from testing.diff_output_files import diff_output_files

    _, differences = diff_output_files(*permuted)
    np.testing.assert_array_equal(differences, 0)


def test_reframe_resolves_paths_after_files_exist(permuted, monkeypatch):
    from types import SimpleNamespace

    pytest.importorskip("reframe")
    from reframe_tests.tests import regression_checks as checks
    import reframe.utility.sanity as sn

    monkeypatch.setattr(checks, "runtime", lambda: SimpleNamespace(system=SimpleNamespace(name="local")))
    captured = []

    def run(command):
        captured.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="", args=command)

    monkeypatch.setattr(checks.osext, "run_command", run)
    check = checks.ReceiverRegressionCheck(permuted[0], permuted[1], "rx1")
    assert sn.evaluate(check.run())
    assert captured[0][-4:] == [str(permuted[0]), str(permuted[1]), "rxs/rx1", "rxs/rx2"]
    assert check.objects == []

    from reframe.core.exceptions import SanityError

    monkeypatch.setattr(
        checks.osext,
        "run_command",
        lambda command: SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="h5diff could not read the file",
            args=command,
        ),
    )
    with pytest.raises(SanityError, match="could not read"):
        sn.evaluate(check.run())
    assert check.objects == []
