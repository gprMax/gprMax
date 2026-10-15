"""Example 9: geometry-only tracked modes of a dielectric cylindrical guide."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from examples.features.eigenmode_tracking.gallery import build_scene as _build_scene, main

CASE = 'dielectric_cylindrical'


def build_scene(**kwargs):
    return _build_scene(CASE, **kwargs)


if __name__ == '__main__':
    main(default_case=CASE, default_output=Path(__file__).resolve().parent/'results')
