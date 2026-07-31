"""Regression tests for two materials.py bugs (Codex-reported):

1. RangeMaterial.calculate_properties() only appended a REUSED existing
   material's numID to self.matID when `iter == 0`. Any LATER bin
   (iter > 0) that happened to reuse an already-existing material (e.g.
   because its rounded properties collided with a material already in
   G.materials) appended nothing at all, leaving matID shorter than
   nbins and every subsequent bin's index into it wrong
   (fractal_box.py's `mixingmodel.matID[int(numberinbin)]` lookup).
   Fixed by appending the reused material's numID on every iteration
   where one is found, not just iter == 0.

2. ListMaterial.calculate_properties() accessed `material.numID` before
   checking whether `material` is None - a missing/misspelled material
   ID would raise AttributeError instead of reaching the intended
   ValueError with a clear "material(s) ... do not exist" message. Fixed
   by moving the None check first.
"""
import pytest

from gprMax.materials import ListMaterial, Material, RangeMaterial


def test_range_material_appends_one_entry_per_bin_even_when_a_later_bin_reuses_a_material():
    materials = []

    # bin0 (er=1.5) is new; bin1 (er=2.5) is pre-populated as an EXISTING
    # material below, so bin1's reuse happens at iter=1, not iter=0 - the
    # exact case the old `iter == 0` guard mishandled.
    existing = Material(numID=99, ID="|2.5000+0.0000+1.0000+0.0000|")
    materials.append(existing)

    class _Grid:
        pass

    grid = _Grid()
    grid.materials = materials

    rm = RangeMaterial(
        ID="range1",
        er_range=(1.0, 3.0),
        se_range=(0.0, 0.0),
        mr_range=(1.0, 1.0),
        sm_range=(0.0, 0.0),
    )
    rm.calculate_properties(2, grid)

    assert len(rm.matID) == 2
    assert rm.matID[1] == existing.numID  # bin1 correctly reused, not dropped
    # bin0 must be a genuinely new material, distinct from the reused one
    assert rm.matID[0] != existing.numID
    new_material = next(m for m in grid.materials if m.numID == rm.matID[0])
    assert new_material.ID == "|1.5000+0.0000+1.0000+0.0000|"


def test_list_material_missing_material_raises_valueerror_not_attributeerror():
    class _Grid:
        materials = []

    lm = ListMaterial(ID="list1", listofmaterials=["nonexistent_material"])

    with pytest.raises(ValueError):
        lm.calculate_properties(1, _Grid())


def test_list_material_existing_materials_resolve_correctly():
    existing = Material(numID=5, ID="my_material")

    class _Grid:
        materials = [existing]

    lm = ListMaterial(ID="list1", listofmaterials=["my_material"])
    lm.calculate_properties(1, _Grid())

    assert lm.matID == [5]
