# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Compatibility entry point for the retired antenna-parameter plotter.

Use :mod:`toolboxes.Plotting.plot_port` for current gprMax outputs. The old
post-processing calculation has deliberately been removed: gprMax now stores
authoritative S11, input impedance, admittance, validity masks, and diagnostic
terminal quantities in its output file.
"""

import logging

from .plot_port import (  # noqa: F401
    FrequencyTrace,
    PortData,
    TimeTrace,
    discover_port_outputs,
    discover_terminal_outputs,
    main,
    plot_port_parameters,
    plot_port_signals,
    plot_port_validity,
    read_port_output,
    read_port_params,
    save_port_figures,
    select_port_paths,
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.warning(
        "plot_antenna_params has been retired; use "
        "python -m toolboxes.Plotting.plot_port instead"
    )
    raise SystemExit(main())
