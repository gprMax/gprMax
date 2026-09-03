"""Portable discovery of the independent STASIS validation libraries."""

import pytest

from testing.validation.validate_sar_spatial_averaging import _load_reference

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "library_name",
    ("libAvgSARStep{}.so", "libAvgSARStep{}.dylib", "AvgSARStep{}.dll", "libAvgSARStep{}.dll"),
)
def test_stasis_reference_discovers_platform_libraries(tmp_path, library_name):
    (tmp_path / "spatialAverageSAR.py").write_text("def spatialAverageSAR(): pass\n")
    build = tmp_path / "core" / "build"
    build.mkdir(parents=True)
    expected = tuple(build / library_name.format(step) for step in (1, 2))
    for path in expected:
        path.touch()

    reference, step1, step2 = _load_reference(tmp_path)

    assert callable(reference)
    assert (step1, step2) == tuple(path.resolve() for path in expected)


def test_stasis_reference_reports_missing_second_library(tmp_path):
    (tmp_path / "spatialAverageSAR.py").write_text("def spatialAverageSAR(): pass\n")
    (tmp_path / "AvgSARStep1.dll").touch()

    with pytest.raises(FileNotFoundError, match="AvgSARStep2"):
        _load_reference(tmp_path)
