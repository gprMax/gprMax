"""Geometry-only export of tracked eigenmode bases, independent of plotting."""

from pathlib import Path

import h5py
import numpy as np

from gprMax._version import __version__


def write_eigenmode_fields(path, grid, ports=()):
    """Persist prepared monitor banks without accessing time-domain results.

    Arrays retain native transverse Yee staggering and the positive-normal
    basis. Direction describes the requested launch, not a sign applied here.
    """
    monitors = [m for m in grid.eigenmodeports if not ports or m.port_index in ports]
    missing = set(ports) - {m.port_index for m in monitors}
    if missing or not monitors:
        raise ValueError(f"Eigenmode field output has missing/unprepared ports: {sorted(missing)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as output:
        output.attrs.update(
            schema="gprmax-eigenmode-fields",
            schema_version=2,
            gprmax_version=__version__,
            coordinate_units="m",
            field_kind="tracked modal basis; not driven simulation fields",
            basis_orientation="positive normal",
            normalization="solver tracked anchor normalization",
        )
        for monitor in monitors:
            owner = monitor.owner
            group = output.create_group(f"ports/{monitor.port_index}")
            from gprMax.eigenmode_tracking import write_diagnostics

            write_diagnostics(group, owner)
            axes = tuple(owner.transverse_axes)
            starts = np.asarray(owner.transverse_start)
            counts = np.asarray(owner.transverse_stop) - starts
            spacing = np.asarray(grid.dl)[list(axes)]
            group.attrs.update(
                normal_axis=owner.normal_axis,
                transverse_axes=axes,
                direction=owner.direction,
                plane_m=owner.plane_index * grid.dl[owner.normal_axis],
                port_id=monitor.output_id,
                requested_anchor_policy=owner.requested_anchor_policy,
                resolved_anchor_policy=owner.resolved_anchor_policy,
                tracking=owner.tracking,
                verification=owner.verification,
            )
            group["frequencies"] = monitor.anchor_frequencies
            group["mode_indices"] = monitor.mode_indices
            group["neff"] = monitor.anchor_neff
            for name in (
                "anchor_mode_valid",
                "anchor_mode_reference_valid",
                "anchor_mode_propagating",
                "anchor_balanced_power",
            ):
                group[name] = getattr(monitor, name)
            if monitor.anchor_operator_neff is not None:
                group["operator_neff"] = monitor.anchor_operator_neff
            group["mode_anchor_policies"] = np.asarray(monitor.mode_anchor_policies, dtype="S")
            for family, bank in (("E", monitor.anchor_e), ("H", monitor.anchor_h)):
                for component in range(3):
                    values = np.asarray([[mode[component] for mode in anchor] for anchor in bank], dtype=np.complex128)
                    field = group.create_group(f"fields/{family}{'xyz'[component]}")
                    field.create_dataset("values", data=values, compression="gzip")
                    for dim, label in enumerate(("u", "v")):
                        n = values.shape[dim + 2]
                        if n not in (counts[dim], counts[dim] + 1):
                            raise ValueError("Unexpected transverse modal staggering")
                        offset = 0.5 if n == counts[dim] else 0.0
                        field[label] = (starts[dim] + np.arange(n) + offset) * spacing[dim]
    return path
