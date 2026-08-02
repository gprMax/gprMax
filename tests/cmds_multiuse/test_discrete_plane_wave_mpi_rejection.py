"""Regression test for a confirmed correctness bug found by running real
MPI models (mpirun -n 2) and comparing to a serial reference: the discrete
plane wave's TFSF box is corrected with plain per-rank loops over local
array indices (apply_TFSF_conditions_electric/_magnetic in gprMax/sources.py),
with no awareness that the box may span multiple MPI ranks. A box larger
than one rank's sub-domain (the normal case - TFSF boxes are usually sized
to occupy most of the model) only gets corrected on whichever single rank
owns the source object; other ranks silently never apply it. No crash, no
warning - confirmed relative errors up to ~70x against a serial reference.

All three discrete-plane-wave commands (#plane_wave_angles,
#plane_wave_vector, #plane_wave_axial) now reject MPI outright at build
time, matching the existing pattern for #magnetic_frill_source and
#eigenmode_source (both of which reject MPI for analogous reasons - a
write footprint or box that cannot be safely split across rank
boundaries).
"""

import pytest

import gprMax
import gprMax.config as config


def _set_mpi(monkeypatch, mpi=True):
    monkeypatch.setattr(config, "sim_config", type("_SC", (), {})())
    config.sim_config.general = {"solver": "cpu"}
    config.sim_config.mpi = mpi


def test_plane_wave_angles_rejected_with_mpi(monkeypatch):
    _set_mpi(monkeypatch)
    dpw = gprMax.DiscretePlaneWaveAngles(
        p1=(0.01, 0.01, 0.01),
        p2=(0.04, 0.04, 0.04),
        theta=36.7,
        phi=63.4,
        psi=90.0,
        waveform_id="w",
    )
    with pytest.raises(ValueError, match="MPI"):
        dpw.build(grid=None)


def test_plane_wave_vector_rejected_with_mpi(monkeypatch):
    _set_mpi(monkeypatch)
    dpw = gprMax.DiscretePlaneWaveVector(
        p1=(0.01, 0.01, 0.01),
        p2=(0.04, 0.04, 0.04),
        m_vec=(1, 2, 3),
        psi=90.0,
        waveform_id="w",
    )
    with pytest.raises(ValueError, match="MPI"):
        dpw.build(grid=None)


def test_plane_wave_axial_rejected_with_mpi(monkeypatch):
    _set_mpi(monkeypatch)
    dpw = gprMax.DiscretePlaneWaveAxial(
        p1=(0.01, 0.01, 0.01),
        p2=(0.04, 0.04, 0.04),
        axis="z",
        psi=90.0,
        waveform_id="w",
    )
    with pytest.raises(ValueError, match="MPI"):
        dpw.build(grid=None)
