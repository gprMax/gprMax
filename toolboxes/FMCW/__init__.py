# Copyright (C) 2026: The University of Edinburgh, United Kingdom
#
# This file is part of gprMax and is distributed under the GNU General Public
# License, version 3 or (at your option) any later version.

"""Frequency-modulated continuous-wave processing for gprMax outputs."""

from .processing import (
    ChannelResponse,
    Chirp,
    DerampedSweep,
    FastTimeResponse,
    interpolate_instrument_response,
    load_instrument_response,
    load_receiver_delay_response,
    process_channel,
    process_incident_referenced_channel,
    reconstruct_fast_time,
    synthesize_deramped_sweep,
    write_fmcw_output,
)

__all__ = [
    "ChannelResponse",
    "Chirp",
    "DerampedSweep",
    "FastTimeResponse",
    "interpolate_instrument_response",
    "load_instrument_response",
    "load_receiver_delay_response",
    "process_channel",
    "process_incident_referenced_channel",
    "reconstruct_fast_time",
    "synthesize_deramped_sweep",
    "write_fmcw_output",
]
