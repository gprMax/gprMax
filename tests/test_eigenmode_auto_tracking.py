"""Opt-in automatic tracking preserves legacy defaults and physical branches."""

from types import SimpleNamespace
from collections import defaultdict

import h5py
import numpy as np
import pytest
from scipy.sparse import diags

from gprMax.eigenmode_config import EigenmodeTrackingConfig
import gprMax.eigenmode_tracking as tracking_module
import gprMax.eigenmode_ports as port_module
import gprMax.config as config_module
from gprMax.eigenmode_tracking import (
    _verification_check,
    assess_anchor_quality,
    track_solver_bank,
    write_diagnostics,
)
from gprMax.sources import EigenmodeReceiver


def _solver(order=(0, 1), phase=(1.0, 1.0)):
    order = np.asarray(order, dtype=int)
    basis = np.eye(2, dtype=np.complex128)[:, order]
    basis = basis * np.asarray(phase)[None, :]
    eigenvalues = np.asarray((-4.0, -1.0))[order]
    neff = np.asarray((2.0, 1.0), dtype=np.complex128)[order]
    zeros = np.zeros_like(basis)
    solver = SimpleNamespace(
        num_modes=2,
        eigenvalues=eigenvalues,
        eigenvectors=basis.copy(),
        operator=diags((-4.0, -1.0), format="csr"),
        free_scalar_mask=np.ones(2, dtype=bool),
        polarization="TM",
        pec_a_mask=np.zeros(2, dtype=bool),
        Et=zeros.copy(),
        Ea=basis.copy(),
        Ew=zeros.copy(),
        Ht=basis.copy(),
        Ha=zeros.copy(),
        Hw=zeros.copy(),
        operator_neff=neff.copy(),
        beta=neff.copy(),
        complex_neff=neff.copy(),
        real_neff=np.real(neff),
        spatially_resolved=np.ones(2, dtype=bool),
        raw_powers=np.ones(2, dtype=np.complex128),
        forward_power_metrics=np.ones(2),
        power_valid=np.ones(2, dtype=bool),
        powers=np.ones(2),
        _set_modal_fields=lambda: None,
    )
    return solver


def _owner():
    return SimpleNamespace(
        tracking_config=EigenmodeTrackingConfig(extra_candidates=0),
        fallback_frequency=2.0,
        _tracking_impedance=1.0,
        port_index=3,
        degenerate=(),
        mode_polarizations={},
        normal_axis=2,
        invariant_axis=None,
    )


def _degenerate_solver(rotation):
    rotation = np.asarray(rotation, dtype=np.complex128)
    basis = np.eye(2, dtype=np.complex128) @ rotation
    zeros = np.zeros_like(basis)
    return SimpleNamespace(
        num_modes=2,
        N=2,
        eta0=1.0,
        k0=1.0,
        eigenvalues=np.asarray((-1.0, -1.0)),
        beta=np.asarray((1.0, 1.0)),
        eigenvectors=basis.copy(),
        operator=-diags((1.0, 1.0), format="csr"),
        free_scalar_mask=np.ones(2, dtype=bool),
        polarization="TM",
        pec_a_mask=np.zeros(2, dtype=bool),
        Et=zeros.copy(),
        Ea=basis.copy(),
        Ew=zeros.copy(),
        Ht=basis.copy(),
        Ha=zeros.copy(),
        Hw=zeros.copy(),
        field_residuals=np.zeros(2),
    )


def test_tracker_restores_swapped_candidate_order_and_phase():
    owner = _owner()
    solvers = (_solver(order=(1, 0), phase=(1j, -1)), _solver(), _solver())

    track_solver_bank(owner, (1.0, 2.0, 3.0), solvers, 2)

    for solver in solvers:
        np.testing.assert_allclose(solver.complex_neff, (2.0, 1.0))
    np.testing.assert_array_equal(owner.tracking_diagnostics["candidate_indices"][0], (1, 0))
    np.testing.assert_array_equal(owner.tracking_diagnostics["candidate_indices"][1], (0, 1))
    assert np.all(owner.tracking_diagnostics["numerical_valid"])


def test_tracking_configuration_defaults_are_opt_in_safe():
    config = EigenmodeTrackingConfig()

    assert config.residual_tolerance == 1e-8
    assert config.verification_overlap == 0.999
    assert config.max_depth == 8
    assert config.max_solves == 200


@pytest.mark.parametrize("interface", ("python", "hash"))
@pytest.mark.parametrize("tracking", (None, "legacy", "auto"))
@pytest.mark.parametrize("coordinator", (None, True, False))
def test_legacy_recommendation_at_port_setup(monkeypatch, caplog, interface, tracking, coordinator):
    from gprMax.grid.fdtd_grid import FDTDGrid
    from gprMax.hash_cmds_multiuse import process_multicmds
    from gprMax.user_objects.cmds_multiuse import EigenmodePort

    monkeypatch.setattr(
        config_module, "sim_config", SimpleNamespace(general={"solver": "cpu"}, mpi=False)
    )
    monkeypatch.setattr(config_module, "get_model_config", lambda: SimpleNamespace(mode="3D"))
    if interface == "python":
        options = {} if tracking is None else {"tracking": tracking}
        port = EigenmodePort(
            port=1, p1=(0, 0, 0.02), p2=(0.05, 0.05, 0.02),
            direction="+", modes=(1, 2), anchors="auto", **options,
        )
    else:
        commands = defaultdict(lambda: None)
        commands["#eigenmode_band"] = ["band 6e9 14e9 3"]
        commands["#eigenmode_excitation"] = ["1 1 auto n"]
        tail = "" if tracking is None else f" tracking={tracking}"
        commands["#eigenmode_port"] = [
            "1 0 0 0.02 0.05 0.05 0.02 + 1,2 auto" + tail
        ]
        port = next(obj for obj in process_multicmds(commands) if isinstance(obj, EigenmodePort))
    grid = FDTDGrid()
    grid.eigenmodeband = SimpleNamespace()
    if coordinator is not None:
        grid.is_coordinator = lambda: coordinator

    with caplog.at_level("WARNING", logger="gprMax.user_objects.cmds_multiuse"):
        port.build(grid)

    warnings = [record.message for record in caplog.records if "uses legacy mode tracking" in record.message]
    expected = tracking != "auto" and coordinator is not False
    assert len(warnings) == int(expected)
    if expected:
        assert 'Set tracking="auto"' in warnings[0]
        assert "tracking=auto in hash input" in warnings[0]
        assert "Eigenmode port 1" in warnings[0]
    assert grid.eigenmodeportdefs[1].tracking == (tracking or "legacy")
    assert grid.eigenmodeportdefs[1].anchors == "auto"


@pytest.mark.parametrize("phase", (1.0, 1j))
def test_assignment_compares_nonorthogonal_degenerate_spans(phase):
    """Repeated eigenvectors may span the same space without being orthogonal."""
    old = np.eye(3, dtype=complex)[:, :2]
    new = np.array(((1, 0.8), (0, 0.6), (0, 0)), dtype=complex) * phase
    values = np.array((-1.0, -1.0))
    assigned, scores = tracking_module._assign_step(
        old, new, values, values, EigenmodeTrackingConfig(), values
    )
    np.testing.assert_array_equal(assigned, (0, 1))
    np.testing.assert_allclose(scores, 1.0)


def test_assignment_rejects_a_different_degenerate_span():
    old = np.eye(3)[:, :2]
    new = np.eye(3)[:, 1:]
    values = np.array((-1.0, -1.0))
    assigned, _ = tracking_module._assign_step(
        old, new, values, values, EigenmodeTrackingConfig(), values
    )
    assert np.any(assigned < 0)


def test_degenerate_span_orthonormalization_rejects_rank_loss():
    with pytest.raises(ValueError, match="rank loss"):
        tracking_module._orthonormal(np.array(((1.0, 1.0), (0.0, 0.0))))


def test_verification_compares_degenerate_modes_as_a_subspace():
    base = _degenerate_solver(np.eye(2))
    rotation = np.asarray(((1, 1), (-1, 1)), dtype=float) / np.sqrt(2)
    variant = _degenerate_solver(rotation)

    check = _verification_check(
        base,
        0,
        variant,
        EigenmodeTrackingConfig(),
        kind="mesh",
        base_group=(0, 1),
    )

    assert check["passed"]
    assert check["overlap"] == pytest.approx(1.0)


def test_automatic_grouping_adds_a_hidden_degenerate_partner():
    owner = _owner()
    solver = _degenerate_solver(np.eye(2))

    mapping = track_solver_bank(owner, (2.0,), (solver,), 1)

    assert mapping.shape == (1, 2)
    assert owner._auto_tracking_mode_count == 2
    assert owner.degenerate == ((1, 2),)


def test_unbound_suspect_warns_without_discarding_anchor(monkeypatch):
    owner = _owner()
    owner.verification = "fast"
    owner.mpi_coordinator = True
    owner.tracking_diagnostics = {}
    owner.port_anchor_mode_valid = np.asarray([[True]])
    warnings = []
    monkeypatch.setattr(tracking_module.logger, "warning", warnings.append)

    records = assess_anchor_quality(owner, None, (2.0,), (_degenerate_solver(np.eye(2)),), (1,))

    assert records[0]["numerical_valid"]
    assert records[0]["confinement"] == "unbound_suspect"
    assert records[0]["artifact"] == "artificial_boundary_or_box_suspect"
    assert owner.port_anchor_mode_valid[0, 0]
    assert "remain in use" in warnings[0]


def test_tracking_diagnostics_are_versioned_and_persisted(tmp_path):
    owner = _owner()
    owner.tracking = "auto"
    owner.verification = "fast"
    owner.degenerate_diagnostics = []
    track_solver_bank(owner, (1.0, 2.0), (_solver(), _solver()), 2)
    owner.tracking_diagnostics.update(
        requested_frequencies=np.asarray((1.0, 2.0)),
        adaptive_frequencies=np.asarray((1.5,)),
        verification_solve_count=0,
        unresolved_intervals=((1.25, 1.5, (2,)),),
        quality=(
            {
                "frequency": 1.0,
                "mode": 1,
                "residual": 1e-12,
                "field_residual": 2e-12,
                "edge_fraction": 2e-3,
                "numerical_valid": True,
                "confinement": "unbound_suspect",
                "artifact": "artificial_boundary_or_box_suspect",
                "reasons": ("artificial_edge_participation",),
            },
        ),
    )
    path = tmp_path / "tracking.h5"

    with h5py.File(path, "w") as output:
        write_diagnostics(output, owner)
    with h5py.File(path, "r") as output:
        tracking = output["mode_tracking"]
        assert tracking.attrs["SchemaVersion"] == 1
        assert tracking.attrs["Tracking"] == "auto"
        np.testing.assert_allclose(tracking["adaptive_frequencies"], (1.5,))
        assert "combined_residuals" in tracking
        assert "field_residuals" in tracking
        np.testing.assert_allclose(tracking["unresolved_interval_lower_frequency"], (1.25,))
        assert tracking["unresolved_interval_modes"][0] == b"2"
        assert tracking["anchor_quality/confinement"][0] == b"unbound_suspect"


def test_passive_receiver_builds_automatic_tracking_bank(monkeypatch):
    """A receive-only second port must track before extracting its public bank."""
    receiver = object.__new__(EigenmodeReceiver)
    receiver.frequencies = (1.0, 2.0)
    receiver.frequency = 1.0
    receiver.fallback_frequency = 1.5
    receiver.degenerate = ()
    receiver.tracking = "auto"
    receiver.tracking_config = EigenmodeTrackingConfig(extra_candidates=0)
    receiver.mode_indices = (1, 2)
    receiver.mode_count = 2
    receiver.mode_polarizations = {}
    receiver.normal_axis = 2
    receiver.invariant_axis = None
    receiver.port_index = 2
    receiver.port_id = "port2"
    receiver.dft_start = 1.0
    receiver.dft_stop = 2.0
    receiver.dft_points = 2
    receiver.dft_frequencies = np.asarray((1.0, 2.0))
    receiver._tracking_extra_solves = 0
    receiver._extract_frequency_dependent_materials = lambda grid: None
    receiver._solve_eigenmode = lambda grid: setattr(receiver, "mode_solver", _solver())
    receiver._plot_eigenmode_fields = lambda: None
    prepared = {}

    def prepare(frequencies, solvers, mode_indices):
        prepared["tracking_count"] = receiver._auto_tracking_mode_count
        receiver.port_anchor_frequencies = tuple(frequencies)
        receiver.port_anchor_e = None
        receiver.port_anchor_h = None
        receiver.port_anchor_neff = None
        receiver.port_anchor_operator_neff = None
        receiver.port_anchor_mode_valid = None
        receiver.port_anchor_mode_reference_valid = None
        receiver.port_anchor_mode_propagating = None
        receiver.port_anchor_balanced_power = None
        receiver.port_mode_anchor_policies = ("explicit", "explicit")
        receiver.port_mode_solvers = tuple(solvers)

    receiver._prepare_port_anchor_bank = prepare
    monkeypatch.setattr(
        config_module,
        "sim_config",
        SimpleNamespace(em_consts={"z0": 1.0}),
    )
    monkeypatch.setattr(
        tracking_module,
        "assess_anchor_quality",
        lambda owner, grid, frequencies, solvers, modes: prepared.setdefault(
            "quality_modes", tuple(modes)
        ),
    )

    class DummyMonitor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def prepare(self, grid):
            prepared["monitor_prepared"] = True

    monkeypatch.setattr(port_module, "EigenmodePortMonitor", DummyMonitor)
    grid = SimpleNamespace(dl=np.ones(3), eigenmodeports=[])

    receiver.grid_init(grid)

    assert prepared == {
        "tracking_count": 2,
        "quality_modes": (1, 2),
        "monitor_prepared": True,
    }
    assert receiver.tracking_diagnostics["requested_frequencies"].tolist() == [1.0, 2.0]
    assert grid.eigenmodeports == [receiver.port_monitor]
