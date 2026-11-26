"""Shared protocol, radar log parser, and radar fall detection rules."""

from .board_log import RadarSample, parse_radar_payload
from .protocol import EVENT_KEYS, VALID_STATES, MonotonicClock, make_event, validate_event
from .rules import FallDetector, RadarFallConfig

__all__ = [
    "EVENT_KEYS",
    "VALID_STATES",
    "MonotonicClock",
    "make_event",
    "validate_event",
    "RadarSample",
    "parse_radar_payload",
    "RadarFallConfig",
    "FallDetector",
]
