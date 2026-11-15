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

import json
import os
import warnings

import h5py
import numpy as np
import pytest
from matplotlib.figure import Figure

from gprMax.toolboxes.AntennaPatterns import initial_save, plot_fields
from gprMax.toolboxes.AntennaPatterns.initial_save import (
    main as initial_save_main,
    process_pattern,
)
from gprMax.toolboxes.AntennaPatterns.plot_fields import (
    load_pattern_data,
    main as plot_fields_main,
    plot_pattern,
)


def test_pattern_processing_and_plotting_workflow(tmp_path):
    outputfile = tmp_path / "pattern.h5"
    configfile = tmp_path / "pattern_config.json"
    patternfile = tmp_path / "pattern.npz"
    plotfile = tmp_path / "pattern.pdf"
    positions = ((1, 0, 1), (1, 0, 0), (1, 0, -1))

    with h5py.File(outputfile, "w") as output:
        output.attrs["Iterations"] = 8
        output.attrs["dt"] = 1e-10
        for index, position in enumerate(positions, start=1):
            receiver = output.create_group(f"rxs/rx{index}")
            receiver.attrs["Name"] = f"pattern_{index:03d}"
            receiver.attrs["Position"] = position
            receiver.create_dataset("Ex", data=np.linspace(0, index, 8))
            receiver.create_dataset("Ey", data=np.linspace(0, 2 * index, 8))
            receiver.create_dataset("Ez", data=np.linspace(0, 3 * index, 8))

    configfile.write_text(
        json.dumps(
            {
                "pattern": "E",
                "radii": [1],
                "theta_degrees": [-45, 0, 45],
                "origin": [0, 0, 0],
                "receiver_prefix": "pattern_",
                "relative_permittivity": 4,
                "relative_permeability": 1,
                "centre_frequency": 1e9,
                "antenna_dimension": 0.1,
                "impedance_scaling": True,
            }
        ),
        encoding="utf-8",
    )

    process_pattern(outputfile, configfile, patternfile)
    data = load_pattern_data(patternfile)
    plot_pattern(data, plotfile)

    assert data["patterns"].shape == (1, 3)
    assert np.all(np.isfinite(data["patterns"]))
    assert np.all(data["patterns"] >= 0)
    assert plotfile.is_file()
    assert plotfile.stat().st_size > 0

    # The writer may renumber public groups; physical angle order is defined
    # by these zero-padded Names, not by rxN. An ordinary feed is not a bin.
    with h5py.File(outputfile, "r+") as output:
        output.move("rxs/rx1", "rxs/temporary")
        output.move("rxs/rx3", "rxs/rx1")
        output.move("rxs/temporary", "rxs/rx3")
        feed = output.create_group("rxs/rx4")
        feed.attrs["Name"] = "ordinary_feed"
        feed.attrs["Position"] = (0, 0, 0)
        feed["Ez"] = np.ones(8) * 1e6
    process_pattern(outputfile, configfile, patternfile)
    np.testing.assert_array_equal(load_pattern_data(patternfile)["patterns"], data["patterns"])


def _pattern_data():
    return {
        "patterns": np.array([[1.0, 0.5, 0.1]]),
        "radii": np.array([0.1]),
        "theta_degrees": np.array([0.0, 90.0, 180.0]),
        "pattern": np.array("E"),
        "relative_permittivity": np.array(np.nan),
        "relative_permeability": np.array(1.0),
        "centre_frequency": np.array(1e9),
        "antenna_dimension": np.array(0.06),
    }


@pytest.mark.parametrize(
    ("minimum_db", "ring_step_db"),
    [(-72, 12), (-72, 10), (-80, 12), (-50, 12), (-60, 20)],
)
def test_plot_pattern_rings_stay_within_display_range(
    tmp_path, monkeypatch, minimum_db, ring_step_db
):
    captured = []
    monkeypatch.setattr(plot_fields.plt, "close", captured.append)

    plot_pattern(
        _pattern_data(),
        tmp_path / "pattern.png",
        minimum_db=minimum_db,
        ring_step_db=ring_step_db,
    )

    axes = captured[-1].axes[0]
    ticks = np.asarray(axes.get_yticks(), dtype=float)
    labels = [label.get_text() for label in axes.get_yticklabels()]

    assert axes.get_ylim() == (minimum_db, 0)
    assert np.all(ticks >= minimum_db) and np.all(ticks <= 0)
    assert ticks[-1] == 0
    np.testing.assert_allclose(np.diff(ticks), ring_step_db)
    assert labels[-1] == "0 dB"
    assert labels[:-1] == [f"{tick:g}" for tick in ticks[:-1]]


@pytest.mark.parametrize(
    ("minimum_db", "ring_step_db"),
    [(-72, 0), (-72, -12), (0, 12), (10, 12), (-72, np.inf), (-np.inf, 12)],
)
def test_plot_pattern_rejects_invalid_ring_settings(tmp_path, minimum_db, ring_step_db):
    with pytest.raises(ValueError):
        plot_pattern(
            _pattern_data(),
            tmp_path / "pattern.png",
            minimum_db=minimum_db,
            ring_step_db=ring_step_db,
        )


def _setup_pattern_case(tmp_path):
    outputfile = tmp_path / "pattern.h5"
    configfile = tmp_path / "pattern_config.json"
    positions = ((1, 0, 1), (1, 0, 0), (1, 0, -1))

    with h5py.File(outputfile, "w") as output:
        output.attrs["Iterations"] = 8
        output.attrs["dt"] = 1e-10
        for index, position in enumerate(positions, start=1):
            receiver = output.create_group(f"rxs/rx{index}")
            receiver.attrs["Name"] = f"pattern_{index:03d}"
            receiver.attrs["Position"] = position
            receiver.create_dataset("Ex", data=np.linspace(0, index, 8))
            receiver.create_dataset("Ey", data=np.linspace(0, 2 * index, 8))
            receiver.create_dataset("Ez", data=np.linspace(0, 3 * index, 8))

    configfile.write_text(
        json.dumps(
            {
                "pattern": "E",
                "radii": [1],
                "theta_degrees": [-45, 0, 45],
                "origin": [0, 0, 0],
                "receiver_prefix": "pattern_",
                "relative_permittivity": 4,
                "relative_permeability": 1,
                "centre_frequency": 1e9,
                "antenna_dimension": 0.1,
                "impedance_scaling": True,
            }
        ),
        encoding="utf-8",
    )
    return outputfile, configfile


def test_process_pattern_rejects_missing_output_or_config_file(tmp_path):
    outputfile, configfile = _setup_pattern_case(tmp_path)

    with pytest.raises(FileNotFoundError, match="gprMax output file does not exist"):
        process_pattern(tmp_path / "missing.h5", configfile, tmp_path / "out.npz")

    with pytest.raises(FileNotFoundError, match="Pattern metadata file does not exist"):
        process_pattern(outputfile, tmp_path / "missing.json", tmp_path / "out.npz")


def test_process_pattern_rejects_overwriting_input_file(tmp_path):
    outputfile, configfile = _setup_pattern_case(tmp_path)
    hard_output = tmp_path / "hard_output.npz"
    os.link(outputfile, hard_output)
    link_config = tmp_path / "link_config.npz"
    link_config.symlink_to(configfile)

    with pytest.raises(ValueError, match="must not overwrite an input file"):
        process_pattern(outputfile, configfile, hard_output)

    with pytest.raises(ValueError, match="must not overwrite an input file"):
        initial_save_main(
            [str(outputfile), "--config", str(configfile), "--output", str(link_config)]
        )


def test_process_pattern_preserves_numpy_filename_conventions(tmp_path):
    outputfile, configfile = _setup_pattern_case(tmp_path)
    original_hdf5 = outputfile.read_bytes()

    as_input_name = process_pattern(outputfile, configfile, outputfile)
    assert as_input_name == outputfile.with_name(f"{outputfile.name}.npz")
    assert as_input_name.is_file()
    assert outputfile.read_bytes() == original_hdf5

    uppercase_dest = initial_save_main(
        [str(outputfile), "--config", str(configfile), "--output", str(tmp_path / "out.NPZ")]
    )
    assert uppercase_dest == tmp_path / "out.NPZ.npz"
    assert uppercase_dest.is_file()
    assert not (tmp_path / "out.NPZ").exists()

    other_dest = process_pattern(outputfile, configfile, tmp_path / "out.other")
    assert other_dest == tmp_path / "out.other.npz"
    assert other_dest.is_file()
    assert not (tmp_path / "out.npz").exists()

    plain_dest = initial_save_main(
        [str(outputfile), "--config", str(configfile), "--output", str(tmp_path / "plain")]
    )
    assert plain_dest == tmp_path / "plain.npz"
    assert plain_dest.is_file()


def test_process_pattern_updates_symlink_to_extensionless_target(tmp_path):
    outputfile, configfile = _setup_pattern_case(tmp_path)
    target = tmp_path / "raw_target"
    target.write_bytes(b"stale")
    link = tmp_path / "pattern.npz"
    link.symlink_to(target)

    result = initial_save_main(
        [str(outputfile), "--config", str(configfile), "--output", str(link)]
    )

    assert result == link
    assert link.is_symlink()
    assert target.is_file()
    assert not (tmp_path / "raw_target.npz").exists()
    assert not (tmp_path / "pattern.npz.npz").exists()
    loaded = load_pattern_data(link)
    assert "patterns" in loaded


def test_process_pattern_preserves_destination_on_partial_write_failure(tmp_path, monkeypatch):
    outputfile, configfile = _setup_pattern_case(tmp_path)
    destination = tmp_path / "pattern.npz"
    sentinel = b"existing_pattern_data"
    destination.write_bytes(sentinel)

    def failing_savez(file, *args, **kwargs):
        file.write(b"partial-npz-bytes")
        file.flush()
        raise OSError("disk full")

    monkeypatch.setattr(initial_save.np, "savez", failing_savez)

    with pytest.raises(OSError, match="disk full"):
        process_pattern(outputfile, configfile, destination)

    assert destination.read_bytes() == sentinel


def test_process_pattern_preserves_destination_on_failure(tmp_path):
    outputfile, configfile = _setup_pattern_case(tmp_path)
    destination = tmp_path / "pattern.npz"
    sentinel = b"existing_pattern_data"
    destination.write_bytes(sentinel)

    config = json.loads(configfile.read_text(encoding="utf-8"))
    config["pattern"] = "INVALID"
    configfile.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Pattern must be 'E' or 'H'"):
        process_pattern(outputfile, configfile, destination)

    assert destination.read_bytes() == sentinel


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("relative_permittivity", -2.0, "relative_permittivity must be a finite positive number"),
        ("relative_permittivity", 0.0, "relative_permittivity must be a finite positive number"),
        ("relative_permeability", -1.0, "relative_permeability must be a finite positive number"),
        ("relative_permeability", 0.0, "relative_permeability must be a finite positive number"),
    ],
)
def test_process_pattern_rejects_non_positive_permittivity_or_permeability(
    tmp_path, field, value, match
):
    outputfile, configfile = _setup_pattern_case(tmp_path)
    config = json.loads(configfile.read_text(encoding="utf-8"))
    config[field] = value
    configfile.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        process_pattern(outputfile, configfile, tmp_path / "out.npz")


def test_load_pattern_data_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Pattern data file does not exist"):
        load_pattern_data(tmp_path / "missing.npz")


def test_plot_pattern_rejects_overwriting_input_file(tmp_path):
    outputfile, configfile = _setup_pattern_case(tmp_path)
    patternfile = tmp_path / "pattern.npz"
    process_pattern(outputfile, configfile, patternfile)

    with pytest.raises(ValueError, match="must not overwrite the input pattern data file"):
        plot_fields_main([str(patternfile), "--output", str(patternfile)])


def test_plot_pattern_handles_subunity_permittivity_without_warning(tmp_path, monkeypatch):
    data = _pattern_data()
    data["relative_permittivity"] = np.array(0.5)
    data["relative_permeability"] = np.array(1.0)
    plotfile = tmp_path / "subunity.png"

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        plot_pattern(data, plotfile)

    assert plotfile.is_file()
    assert plotfile.stat().st_size > 0


def test_antenna_patterns_cli_main(tmp_path):
    outputfile, configfile = _setup_pattern_case(tmp_path)
    patternfile = tmp_path / "cli_pattern.npz"
    plotfile = tmp_path / "cli_pattern.pdf"

    result_pattern = initial_save_main(
        [str(outputfile), "--config", str(configfile), "--output", str(patternfile)]
    )
    assert result_pattern == patternfile
    assert patternfile.is_file()

    result_plot = plot_fields_main([str(patternfile), "--output", str(plotfile)])
    assert result_plot == plotfile
    assert plotfile.is_file()


def test_plot_pattern_handles_extensionless_uppercase_and_symlink_targets(tmp_path):
    outputfile, configfile = _setup_pattern_case(tmp_path)
    patternfile = tmp_path / "pattern.npz"
    process_pattern(outputfile, configfile, patternfile)

    extensionless = plot_fields_main([str(patternfile), "--output", str(tmp_path / "polar")])
    assert extensionless == tmp_path / "polar.png"
    assert extensionless.read_bytes().startswith(b"\x89PNG")
    assert not (tmp_path / "polar").exists()

    uppercase = plot_fields_main([str(patternfile), "--output", str(tmp_path / "UPPER.PNG")])
    assert uppercase == tmp_path / "UPPER.PNG"
    assert uppercase.read_bytes().startswith(b"\x89PNG")
    assert not (tmp_path / "UPPER.PNG.png").exists()

    target = tmp_path / "plot_target"
    target.write_bytes(b"stale")
    link = tmp_path / "plot.png"
    link.symlink_to(target)

    symlink_result = plot_fields_main([str(patternfile), "--output", str(link)])
    assert symlink_result == link
    assert link.is_symlink()
    assert target.read_bytes().startswith(b"\x89PNG")
    assert not (tmp_path / "plot_target.png").exists()


def test_plot_pattern_preserves_destination_on_partial_write_failure(tmp_path, monkeypatch):
    destination = tmp_path / "plot.png"
    sentinel = b"existing_plot_bytes"
    destination.write_bytes(sentinel)

    def failing_savefig(self, fname, *args, **kwargs):
        with open(fname, "wb") as stream:
            stream.write(b"partial-png-bytes")
        raise OSError("disk full")

    monkeypatch.setattr(Figure, "savefig", failing_savefig)

    with pytest.raises(OSError, match="disk full"):
        plot_pattern(_pattern_data(), destination)

    assert destination.read_bytes() == sentinel
