"""Read saved terminal spectra across source families without reinterpreting them."""

import h5py
import numpy as np
import pytest

from gprMax.toolboxes.Optimisation import read_port


def write_terminal(file, path, reference=50):
    group = file.create_group(path)
    group["frequency"] = [1e9, 2e9, 3e9]
    group["S11"] = [0.2 + 0j, 0.1j, np.nan + 0j]
    group["Zin"] = [75 + 0j, np.nan + 0j, np.nan + 0j]
    group["valid_S11"] = [1, 1, 0]
    group["valid_Zin"] = [1, 0, 0]
    family = path.split("/")[-2]
    group.attrs["ReferenceImpedance" if family == "ports" else "Z0"] = reference
    group.attrs["IndependentFrequencyResolution"] = 1e9
    group.attrs["TailRelativeLevelDB"] = -80.0


@pytest.mark.parametrize("family", ["ports", "tls", "frills"])
@pytest.mark.parametrize("grid", ["/", "/subgrids/fine"])
def test_source_family_and_grid_keep_native_reference_and_masks(tmp_path, family, grid):
    path = tmp_path / "outputs.h5"
    with h5py.File(path, "w") as file:
        write_terminal(file, f"{grid.rstrip('/')}/{family}/feed", reference=75)
    for name in ("feed", f"{family}/feed"):
        data = read_port(path, name, grid=grid)
        assert data.reference_impedance == 75
        assert data.at(2e9) == 0.1j
        assert data.valid[1] and not data.impedance_valid[1]
        with pytest.raises(ValueError, match="invalid"):
            data.at(3e9)
        assert data.group == f"{grid.rstrip('/')}/{family}/feed"


def test_colliding_ids_require_explicit_family(tmp_path):
    path = tmp_path / "outputs.h5"
    with h5py.File(path, "w") as file:
        write_terminal(file, "ports/feed", reference=50)
        write_terminal(file, "frills/feed", reference=75)
    with pytest.raises(ValueError, match="disambiguate"):
        read_port(path, "feed")
    assert read_port(path, "frills/feed").reference_impedance == 75


def test_valid_frill_bin_cannot_hide_nonfinite_s11(tmp_path):
    path = tmp_path / "outputs.h5"
    with h5py.File(path, "w") as file:
        write_terminal(file, "frills/frill1")
        file["frills/frill1/valid_S11"][2] = 1
    with pytest.raises(ValueError, match="Nonfinite S11"):
        read_port(path, "frill1")
