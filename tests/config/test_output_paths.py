"""Where a model's files end up.

Every artefact a run produces — the ``.h5`` output file, the snapshot
directory, the geometry files — is derived from one attribute,
``ModelConfig.output_file_path``, computed once in ``set_output_file_path``.
Three inputs feed it, in strict priority order:

1. ``outputdir``, passed by the ``#output_dir:`` input-file command;
2. ``args.outputfile``, from the API or ``-o`` on the command line;
3. failing both, the *input* file path with its extension stripped.

Then two suffix operations are applied on top: the model number is appended to
the final path component (so a three-model run does not overwrite itself), and
``.h5`` is attached to give ``output_file_path_ext``.

The reason to test this in isolation is that a mistake here is invisible until
the very end of a run. Nothing reads ``output_file_path`` while the model is
solving; it is consumed when the last timestep has already been computed. A
wrong path does not crash a simulation, it loses one — or, worse, silently
overwrites the previous model's results because the model number was dropped.

Two of the tests below pin behaviour that is a known defect rather than a
contract, and say so in their docstrings. They exist because the current
behaviour is surprising and undocumented, and a reader hitting it needs to
find something in the suite that explains it.
"""

from pathlib import Path

import pytest


class TestTheDefaultPath:
    """No ``-o`` and no ``#output_dir:`` — the input file names the output."""

    def test_the_input_file_path_is_reused(self, make_model_config):
        model_config = make_model_config(inputfile="model.in")

        assert model_config.output_file_path == Path("model")

    def test_the_input_file_extension_is_stripped(self, make_model_config):
        model_config = make_model_config(inputfile="cylinder_Bscan.in")

        assert model_config.output_file_path.suffix == ""

    def test_the_directory_of_the_input_file_is_kept(self, make_model_config):
        model_config = make_model_config(inputfile="examples/deep/model.in")

        assert model_config.output_file_path == Path("examples/deep/model")

    def test_nothing_is_created_on_disk(self, make_model_config):
        """Computing the path must not touch the filesystem.

        ``ModelConfig`` is constructed before the model is built, and may be
        constructed for a run that never completes. Only the explicit
        ``outputdir`` branch is allowed to create anything.
        """
        model_config = make_model_config(inputfile="never-written.in")

        assert not model_config.output_file_path.exists()


class TestTheOutputFileArgument:
    """``-o`` / ``outputfile=`` overrides the input file name."""

    def test_the_argument_wins_over_the_input_file(self, make_model_config):
        model_config = make_model_config(inputfile="model.in", outputfile="results/run_a")

        assert model_config.output_file_path == Path("results/run_a")

    def test_an_extension_on_the_argument_is_stripped(self, make_model_config):
        """Users pass ``-o out.h5``; the ``.h5`` is added back later.

        Without the strip the file would be written as ``out.h5.h5``.
        """
        model_config = make_model_config(inputfile="model.in", outputfile="out.h5")

        assert model_config.output_file_path == Path("out")

    def test_the_argument_is_not_created_on_disk(self, make_model_config):
        model_config = make_model_config(inputfile="model.in", outputfile="out")

        assert not model_config.output_file_path.exists()


class TestTheOutputDirectory:
    """``#output_dir:`` — the only branch with a filesystem side effect."""

    def test_the_directory_is_created(self, make_model_config, tmp_path):
        model_config = make_model_config(inputfile="model.in")
        outputdir = tmp_path / "snapshots-and-outputs"

        model_config.set_output_file_path(str(outputdir))

        assert outputdir.is_dir()

    def test_the_input_file_stem_is_placed_inside_it(self, make_model_config, tmp_path):
        """Only the *stem* survives — the input file's own directory is dropped."""
        model_config = make_model_config(inputfile="deep/nested/model.in")
        outputdir = tmp_path / "out"

        model_config.set_output_file_path(str(outputdir))

        assert model_config.output_file_path == outputdir / "model"

    def test_an_existing_directory_is_accepted(self, make_model_config, tmp_path):
        """``exist_ok=True``, so re-running a multi-model simulation is fine."""
        model_config = make_model_config(inputfile="model.in")
        outputdir = tmp_path / "out"
        outputdir.mkdir()

        model_config.set_output_file_path(str(outputdir))

        assert model_config.output_file_path == outputdir / "model"

    def test_it_takes_priority_over_the_output_file_argument(self, make_model_config, tmp_path):
        model_config = make_model_config(inputfile="model.in", outputfile="ignored/elsewhere")
        outputdir = tmp_path / "wins"

        model_config.set_output_file_path(str(outputdir))

        assert model_config.output_file_path.parent == outputdir

    def test_a_missing_parent_directory_is_not_created(self, make_model_config, tmp_path):
        """``mkdir`` is called without ``parents=True``.

        ``#output_dir: results/2026/run_a`` therefore raises
        ``FileNotFoundError`` from deep inside configuration rather than
        reporting a bad path, even though the value came straight from user
        input. Pinned as the current behaviour, with the defect written up in
        ``notes/bugs/config-output-dir-no-parents.md``.
        """
        model_config = make_model_config(inputfile="model.in")
        nested = tmp_path / "missing" / "leaf"

        with pytest.raises(FileNotFoundError):
            model_config.set_output_file_path(str(nested))


class TestTheModelNumberSuffix:
    """A multi-model run must not overwrite itself."""

    def test_a_single_model_run_appends_nothing(self, make_model_config):
        model_config = make_model_config(model_num=0, inputfile="model.in", n=1)

        assert model_config.appendmodelnumber == ""

    def test_a_multi_model_run_appends_a_one_based_number(self, make_model_config):
        model_config = make_model_config(model_num=0, inputfile="model.in", n=3)

        assert model_config.appendmodelnumber == "1"

    @pytest.mark.parametrize("model_num, expected", [(0, "1"), (1, "2"), (9, "10")])
    def test_the_number_is_the_model_index_plus_one(self, make_model_config, model_num, expected):
        model_config = make_model_config(model_num=model_num, inputfile="model.in", n=10)

        assert model_config.appendmodelnumber == expected

    def test_the_number_lands_on_the_file_name_not_the_directory(self, make_model_config):
        model_config = make_model_config(model_num=1, inputfile="results/model.in", n=3)

        assert model_config.output_file_path == Path("results/model2")

    def test_consecutive_models_get_distinct_paths(self, install_sim_config):
        """The property that actually matters, asserted directly."""
        from gprMax import config

        sim_config = install_sim_config(inputfile="model.in", n=3)
        paths = {config.ModelConfig(n).output_file_path for n in range(3)}

        assert len(paths) == 3

    def test_the_number_survives_an_output_directory(self, make_model_config, tmp_path):
        model_config = make_model_config(model_num=2, inputfile="model.in", n=3)

        model_config.set_output_file_path(str(tmp_path))

        assert model_config.output_file_path == tmp_path / "model3"


class TestTheExtendedPath:
    """``output_file_path_ext`` — the name the ``.h5`` writer is handed."""

    def test_it_is_the_path_with_an_h5_suffix(self, make_model_config):
        model_config = make_model_config(inputfile="model.in")

        assert model_config.output_file_path_ext == Path("model.h5")

    def test_it_includes_the_model_number(self, make_model_config):
        model_config = make_model_config(model_num=1, inputfile="model.in", n=2)

        assert model_config.output_file_path_ext == Path("model2.h5")

    def test_it_is_recomputed_when_the_path_changes(self, make_model_config, tmp_path):
        """Both attributes are set together, so they cannot drift apart."""
        model_config = make_model_config(inputfile="model.in")

        model_config.set_output_file_path(str(tmp_path))

        assert model_config.output_file_path_ext == tmp_path / "model.h5"

    def test_a_dot_in_the_file_name_truncates_it(self, make_model_config):
        """``with_suffix`` treats everything after the last dot as an extension.

        ``v1.2_model.in`` becomes ``v1.h5``: ``.2_model`` is read as a suffix
        and replaced. Dots in file names are ordinary — version numbers,
        dimensions, dates — so this quietly writes two different models to the
        same file. Pinned as the current behaviour, with the defect written up
        in ``notes/bugs/config-output-path-with-suffix.md``.
        """
        model_config = make_model_config(inputfile="v1.2_model.in")

        assert model_config.output_file_path_ext == Path("v1.h5")


class TestTheSnapshotDirectory:
    """``set_snapshots_dir`` — a sibling directory, not a child."""

    def test_it_is_the_output_name_with_a_suffix(self, make_model_config):
        model_config = make_model_config(inputfile="model.in")

        assert model_config.set_snapshots_dir() == Path("model_snaps")

    def test_it_sits_beside_the_output_file(self, make_model_config):
        model_config = make_model_config(inputfile="results/model.in")

        assert model_config.set_snapshots_dir() == Path("results/model_snaps")

    def test_it_includes_the_model_number(self, make_model_config):
        """Each model in a B-scan gets its own snapshot directory."""
        model_config = make_model_config(model_num=1, inputfile="model.in", n=4)

        assert model_config.set_snapshots_dir() == Path("model2_snaps")

    def test_it_follows_an_output_directory(self, make_model_config, tmp_path):
        model_config = make_model_config(inputfile="model.in")

        model_config.set_output_file_path(str(tmp_path))

        assert model_config.set_snapshots_dir() == tmp_path / "model_snaps"

    def test_it_does_not_create_anything(self, make_model_config):
        """The name is computed here; the directory is made by the writer.

        Unlike ``set_output_file_path``'s ``outputdir`` branch, this one is
        pure, so calling it to *inspect* the path is safe.
        """
        model_config = make_model_config(inputfile="model.in")

        assert not model_config.set_snapshots_dir().exists()

    def test_it_is_recomputed_from_the_current_output_path(self, make_model_config, tmp_path):
        """Nothing is cached — the answer tracks later ``#output_dir:`` commands."""
        model_config = make_model_config(inputfile="model.in")
        before = model_config.set_snapshots_dir()

        model_config.set_output_file_path(str(tmp_path))
        after = model_config.set_snapshots_dir()

        assert before != after


pytestmark = pytest.mark.unit
