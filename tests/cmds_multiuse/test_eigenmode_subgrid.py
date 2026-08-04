"""Validation for eigenmode commands placed on HSG subgrids."""

import pytest

import gprMax
from gprMax.subgrids.subgrid_hsg import SubGridHSG


@pytest.mark.parametrize(
    "command_type",
    (gprMax.EigenmodeSource, gprMax.EigenmodeRx),
    ids=("source", "receiver"),
)
def test_eigenmode_command_rejects_subgrid(command_type):
    subgrid = object.__new__(SubGridHSG)
    with pytest.raises(ValueError, match="currently supports only the main grid"):
        command_type().build(subgrid)
