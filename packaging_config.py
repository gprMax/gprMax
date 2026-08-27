"""Shared configuration for source distributions and binary wheels."""

from pathlib import Path

from setuptools import find_packages


EXAMPLES_PACKAGE = "gprMax._examples"
EXAMPLES_SOURCE = Path("examples")

# Developer and manual-validation packages remain in the project repository
# but are not installed into an ordinary user environment or binary wheel.
EXCLUDED_INSTALL_PACKAGES = (
    "examples",
    "examples.*",
    "reframe_tests",
    "reframe_tests.*",
    "testing",
    "testing.*",
)

# Keep all toolbox code and compact, runnable examples. Exclude only generated
# converter outputs and large reference datasets that are not needed to run a
# toolbox. These assets remain available from the project repository.
EXCLUDED_PACKAGE_DATA = {
    # Binary wheels use the extension modules produced by build_ext. Generated
    # C/Cython inputs remain in the sdist for source builds.
    "gprMax": ["config.pxd", "*.pxd"],
    "gprMax.cython": ["*.c", "*.pyx", "*.pxd", "*.jinja"],
    "toolboxes.STEPtoVoxel": ["examples/patch_antenna/output/*"],
    "toolboxes.STLtoVoxel": [
        "examples/bunny.vti",
        "examples/stl/Caribou_Lakes.stl",
        "examples/stl/Frenchman_Mountain.stl",
        "examples/stl/Mont_Blanc.stl",
        "examples/stl/Stanford_Bunny.h5",
        "examples/stl/Trinity_Alps.stl",
        "examples/stl/point_cloud/*",
    ],
}


def packaged_example_files():
    """Return example resources, excluding local interpreter artefacts."""

    return [
        path.relative_to(EXAMPLES_SOURCE).as_posix()
        for path in EXAMPLES_SOURCE.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
        and path.name != "__init__.py"
    ]


def distribution_packages():
    """Return user-facing packages installed by source and wheel builds."""

    return find_packages(exclude=EXCLUDED_INSTALL_PACKAGES) + [EXAMPLES_PACKAGE]
