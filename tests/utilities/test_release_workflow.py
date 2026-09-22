"""Release policy and artifact checks without uploads or solver compilation."""

import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile

import pytest
import yaml


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def _usable_bash():
    candidate = shutil.which("bash")
    if candidate is None:
        return None
    try:
        result = subprocess.run(
            [candidate, "--version"], capture_output=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return candidate if result.returncode == 0 else None


BASH = _usable_bash()
spec = importlib.util.spec_from_file_location("release_checks", ROOT / "packaging/release_checks.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def workflow(name):
    # BaseLoader avoids interpreting GitHub's YAML 'on' key as a boolean.
    return yaml.load((ROOT / f".github/workflows/{name}.yml").read_text(), Loader=yaml.BaseLoader)


def wheel(directory, python="cp312", platform="manylinux_2_28_x86_64", version="4.0.0", metadata_version=None):
    tag = f"{python}-{python}-{platform}"
    path = directory / f"gprmax-{version}-{tag}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        info = f"gprmax-{version}.dist-info/"
        archive.writestr(info + "METADATA", f"Name: gprMax\nVersion: {metadata_version or version}\n")
        archive.writestr(info + "WHEEL", f"Wheel-Version: 1.0\nTag: {tag}\n")
    return path


@pytest.fixture
def distributions(tmp_path):
    root = tmp_path / "dist"
    root.mkdir()
    for python in checks.PYTHONS:
        for platform in ("manylinux_2_28_x86_64", "win_amd64", "macosx_11_0_x86_64", "macosx_11_0_arm64"):
            wheel(root, python, platform)
    with tarfile.open(root / "gprmax-4.0.0.tar.gz", "w:gz") as archive:
        data = b"Name: gprMax\nVersion: 4.0.0\n"
        info = tarfile.TarInfo("gprmax-4.0.0/PKG-INFO")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return root


@pytest.mark.parametrize("destination", ["dry-run", "testpypi", "pypi"])
@pytest.mark.parametrize("version", ["4.0.0", "4.0.0rc1", "4.0.1"])
def test_release_version_accepts_matching_tag(tmp_path, destination, version):
    path = tmp_path / "version.py"
    path.write_text(f'__version__ = "{version}"\nraise RuntimeError("must not execute")\n')
    assert checks.release_version(path, f"refs/tags/v.{version}", destination) == version


@pytest.mark.parametrize("destination", ["testpypi", "pypi"])
@pytest.mark.parametrize(
    "ref",
    [
        "refs/heads/master",
        "refs/heads/devel",
        "refs/heads/v.4.0.0",
        "refs/pull/1/merge",
        "refs/tags/v4.0.0",
        "refs/tags/4.0.0",
        "refs/tags/v.3.1.7",
        "refs/tags/v.4.0.1",
        "refs/tags/v.4.0.0rc1",
        "refs/tags/v.4.0.0-Caol-Ila",
    ],
)
def test_publishing_rejects_branch_pr_or_wrong_tag(tmp_path, destination, ref):
    path = tmp_path / "version.py"
    path.write_text('__version__ = "4.0.0"\n')
    with pytest.raises(ValueError, match=r"Publishing requires refs/tags/v\.4\.0\.0;"):
        checks.release_version(path, ref, destination)


@pytest.mark.parametrize("ref", ["refs/heads/master", "refs/heads/devel"])
def test_branch_dry_run_is_allowed(tmp_path, ref):
    path = tmp_path / "version.py"
    path.write_text('__version__ = "4.0.0"\n')
    assert checks.release_version(path, ref, "dry-run") == "4.0.0"


@pytest.mark.parametrize("version", ["v4.0.0", "v.4.0.0", "4.0.0+local", "1!4.0.0", "invalid"])
def test_rejects_non_public_or_noncanonical_version(tmp_path, version):
    path = tmp_path / "version.py"
    path.write_text(f"__version__ = {version!r}\n")
    with pytest.raises(ValueError):
        checks.release_version(path, f"refs/tags/v.{version}", "pypi")


@pytest.mark.parametrize(
    "version,destination,ref",
    [
        ("4.0.0", "pypi", "refs/tags/v.4.0.0"),
        ("4.0.0rc1", "testpypi", "refs/tags/v.4.0.0rc1"),
        ("4.0.0", "dry-run", "refs/heads/master"),
    ],
)
def test_version_cli_returns_package_version_not_tag(tmp_path, version, destination, ref):
    path = tmp_path / "version.py"
    path.write_text(f'__version__ = "{version}"\n')
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "packaging/release_checks.py"),
            "version",
            "--version-file",
            str(path),
            "--ref",
            ref,
            "--destination",
            destination,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert result.stdout.strip() == version


@pytest.mark.parametrize("version,destination", [("4.0.0", "pypi"), ("4.0.0rc1", "testpypi")])
def test_documented_publication_commands_use_traditional_tags(version, destination):
    instructions = (ROOT / "docs/source/releasing.rst").read_text(encoding="utf-8")
    assert f"--ref v.{version} -f destination={destination}" in instructions


def test_complete_release_has_hashes(distributions):
    manifest = checks.check_distributions(distributions, "4.0.0")
    assert manifest["version"] == "4.0.0"
    assert len(manifest["files"]) == 13
    for name, info in manifest["files"].items():
        assert info == {"size": (distributions / name).stat().st_size, "sha256": checks.sha256(distributions / name)}


@pytest.mark.parametrize("pattern", ["*cp311-cp311-win*.whl", "*.tar.gz"])
def test_missing_artifacts_stop_release(distributions, pattern):
    next(distributions.glob(pattern)).unlink()
    with pytest.raises(ValueError, match="Incomplete release"):
        checks.check_distributions(distributions, "4.0.0")


@pytest.mark.parametrize("version,metadata_version", [("4.0.1", None), ("4.0.0", "3.1.7")])
def test_wrong_version_stops_release(distributions, version, metadata_version):
    wheel(distributions, version=version, metadata_version=metadata_version)
    with pytest.raises(ValueError, match="Expected gprMax"):
        checks.check_distributions(distributions, "4.0.0")


@pytest.mark.parametrize("platform", ["linux_x86_64", "manylinux_2_28_aarch64", "macosx_11_0_universal2"])
def test_unsupported_platform_stops_release(distributions, platform):
    wheel(distributions, platform=platform)
    with pytest.raises(ValueError, match="non-portable wheel platform"):
        checks.check_distributions(distributions, "4.0.0")


def test_duplicate_matrix_entry_stops_release(distributions):
    wheel(distributions, platform="manylinux_2_34_x86_64")
    with pytest.raises(ValueError, match="Duplicate"):
        checks.check_distributions(distributions, "4.0.0")


@pytest.mark.parametrize(
    "platform",
    ["manylinux_2_28_x86_64.manylinux_2_34_x86_64", "manylinux_2_17_x86_64.manylinux2014_x86_64"],
)
def test_repaired_wheel_with_multiple_compatible_tags(distributions, platform):
    next(distributions.glob("*cp312-cp312-manylinux*.whl")).unlink()
    wheel(distributions, platform=platform)
    assert len(checks.check_distributions(distributions, "4.0.0")["files"]) == 13


def test_non_distribution_file_stops_release(distributions):
    (distributions / "notes.txt").write_text("not for PyPI")
    with pytest.raises(ValueError, match="Unexpected release file"):
        checks.check_distributions(distributions, "4.0.0")


def test_downloaded_wheel_must_match_original_bytes(distributions, tmp_path):
    manifest = checks.check_distributions(distributions, "4.0.0")
    target = tmp_path / "downloaded"
    target.mkdir()
    original = next(distributions.glob("*.whl"))
    copy = target / original.name
    copy.write_bytes(original.read_bytes())
    checks.check_downloaded(target, manifest)
    copy.write_bytes(copy.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="differs"):
        checks.check_downloaded(target, manifest)


def test_filename_and_internal_wheel_tags_must_agree(distributions):
    path = next(distributions.glob("*cp312-cp312-win*.whl"))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("gprmax-4.0.0.dist-info/METADATA", "Name: gprMax\nVersion: 4.0.0\n")
        archive.writestr("gprmax-4.0.0.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: cp311-cp311-win_amd64\n")
    with pytest.raises(ValueError, match="tags disagree"):
        checks.check_distributions(distributions, "4.0.0")


@pytest.mark.parametrize("python", ["cp310", "cp314"])
def test_unexpected_python_stops_release(distributions, python):
    wheel(distributions, python=python)
    with pytest.raises(ValueError, match="Python/ABI"):
        checks.check_distributions(distributions, "4.0.0")


def test_empty_download_fails(distributions, tmp_path):
    manifest = checks.check_distributions(distributions, "4.0.0")
    directory = tmp_path / "empty"
    directory.mkdir()
    with pytest.raises(ValueError, match="exactly one"):
        checks.check_downloaded(directory, manifest)


def test_cli_writes_manifest(distributions, tmp_path):
    manifest = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "packaging/release_checks.py"),
            "distributions",
            str(distributions),
            "--version",
            "4.0.0",
            "--manifest",
            str(manifest),
        ],
        check=True,
    )
    assert len(json.loads(manifest.read_text())["files"]) == 13


@pytest.mark.parametrize("version", ["4.0.0", "4.0.1"])
def test_pip_selects_only_an_exact_local_release_wheel(distributions, tmp_path, version):
    target = tmp_path / "downloaded"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--no-index",
            "--find-links",
            str(distributions),
            "--no-deps",
            "--only-binary=:all:",
            "--no-cache-dir",
            "--dest",
            str(target),
            f"gprMax=={version}",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if version == "4.0.0":
        assert result.returncode == 0, result.stdout + result.stderr
        checks.check_downloaded(target, checks.check_distributions(distributions, version))
    else:
        assert result.returncode != 0
        assert not list(target.glob("*.whl"))


def test_release_is_opt_in_and_cannot_cancel_an_upload():
    release = workflow("release")
    assert set(release["on"]) == {"workflow_dispatch"}
    assert release["on"]["workflow_dispatch"]["inputs"]["destination"]["default"] == "dry-run"
    assert release["permissions"] == {"contents": "read"}
    assert release["concurrency"]["cancel-in-progress"] == "false"
    assert release["jobs"]["preflight"]["if"] == "github.repository == 'gprMax/gprMax'"


def test_publishing_is_isolated_and_follows_validation():
    jobs = workflow("release")["jobs"]
    assert jobs["build"]["uses"] == "./.github/workflows/wheels.yml"
    assert jobs["build"]["needs"] == "preflight"
    assert jobs["validate"]["needs"] == ["preflight", "build"]
    assert jobs["verify-artifacts"]["needs"] == ["preflight", "validate"]
    assert jobs["testpypi"]["needs"] == ["validate", "verify-artifacts"]
    assert jobs["verify-testpypi"]["needs"] == ["preflight", "testpypi"]
    assert jobs["pypi"]["needs"] == ["validate", "verify-artifacts"]
    for name, job in jobs.items():
        if name not in {"testpypi", "pypi"}:
            assert "id-token" not in job.get("permissions", {})
            continue
        assert job["environment"]["name"] == name
        assert job["permissions"] == {"id-token": "write"}
        assert job["if"] == f"inputs.destination == '{name}'"
        assert len(job["steps"]) == 2
        download, publish = job["steps"]
        assert download["uses"].startswith("actions/download-artifact@")
        assert download["with"]["name"] == "release-distributions"
        assert publish["uses"].startswith("pypa/gh-action-pypi-publish@")
        assert not {"password", "skip-existing", "verify-metadata"}.intersection(publish["with"])


@pytest.mark.parametrize("destination", ["dry-run", "pypi", "testpypi"])
def test_destinations_have_independent_upload_paths(destination):
    jobs = workflow("release")["jobs"]

    def ancestors(name):
        needs = jobs[name].get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        return set(needs).union(*(ancestors(parent) for parent in needs))

    # Keep upload conditions explicit, so no skipped TestPyPI dependency can
    # skip the production job (or be bypassed with an unsafe always() guard).
    for upload in ("pypi", "testpypi"):
        assert jobs[upload]["if"] == f"inputs.destination == '{upload}'"
        dependencies = ancestors(upload)
        assert dependencies == {"preflight", "build", "validate", "verify-artifacts"}
        assert all("if" not in jobs[name] for name in dependencies - {"preflight"})

    assert "if" not in jobs["verify-testpypi"]
    assert "testpypi" in ancestors("verify-testpypi")
    # Dry runs never match an upload; publishing selects exactly one index.
    enabled = {name for name in ("pypi", "testpypi") if jobs[name]["if"] == f"inputs.destination == '{destination}'"}
    assert enabled == (set() if destination == "dry-run" else {destination})


def test_artifact_installation_is_checked_without_testpypi():
    job = workflow("release")["jobs"]["verify-artifacts"]
    assert "if" not in job  # Also required for dry-run rehearsals.
    downloads = [step["with"]["name"] for step in job["steps"] if "download-artifact@" in step.get("uses", "")]
    assert downloads == ["release-distributions", "release-manifest"]
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "--no-index --find-links dist --no-deps --only-binary=:all:" in commands
    assert "--no-cache-dir" in commands
    assert "release_checks.py downloaded downloaded --manifest release-manifest.json" in commands
    assert "pip install --index-url https://pypi.org/simple/ downloaded/*.whl" in commands
    assert 'cd "$RUNNER_TEMP"' in commands
    assert 'python "$GITHUB_WORKSPACE/tests/wheel_smoke.py"' in commands
    assert "test.pypi.org" not in commands
    assert "--extra-index-url" not in commands


def test_release_matrix_matches_build_configuration():
    wheels = workflow("wheels")
    assert "workflow_call" in wheels["on"]
    assert {entry["name"] for entry in wheels["jobs"]["wheels"]["strategy"]["matrix"]["include"]} == set(
        checks.PLATFORMS
    )
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert config["tool"]["cibuildwheel"]["build"] == [f"{python}-*" for python in checks.PYTHONS]


@pytest.mark.parametrize("name", ["pytest", "tests"])
def test_v4_checks_cover_stable_and_development_branches(name):
    triggers = workflow(name)["on"]
    assert {"master", "devel"} <= set(triggers["push"]["branches"])
    assert "pull_request" in triggers
    assert "workflow_dispatch" in triggers


def test_no_mixed_testpypi_dependency_index():
    job = workflow("release")["jobs"]["verify-testpypi"]
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "--extra-index-url" not in commands
    assert "--no-deps --only-binary=:all:" in commands
    assert "pip install --index-url https://pypi.org/simple/ downloaded/*.whl" in commands
    assert 'cd "$RUNNER_TEMP"' in commands


@pytest.mark.skipif(BASH is None, reason="Release jobs run on Linux with Bash")
def test_release_shell_steps_parse():
    for job in workflow("release")["jobs"].values():
        for step in job.get("steps", []):
            if "run" in step:
                # On Windows, a bare "bash" can launch the System32 WSL stub
                # instead of the Git Bash found on PATH. Bytes preserve the
                # workflow's LF line endings rather than translating to CRLF.
                subprocess.run([BASH, "-n"], input=step["run"].encode("utf-8"), check=True, timeout=10)


@pytest.mark.parametrize("bash", ["/opt/release tools/bash", "C:/Program Files/Git/bin/bash.exe"])
def test_release_shell_check_uses_resolved_executable_and_lf_bytes(monkeypatch, bash):
    script = "if true; then\n  echo 'caf\u00e9'\nfi\n"
    monkeypatch.setitem(globals(), "BASH", bash)
    monkeypatch.setitem(
        globals(),
        "workflow",
        lambda name: {"jobs": {"check": {"steps": [{"uses": "actions/checkout@v6"}, {"run": script}]}}},
    )
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))

    test_release_shell_steps_parse()

    assert calls == [([bash, "-n"], {"input": script.encode("utf-8"), "check": True, "timeout": 10})]
