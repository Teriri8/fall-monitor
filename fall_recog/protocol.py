"""Event contract and shared monotonic clock for Fall Detection System v2."""

from __future__ import annotations

import threading
import time
from typing import Any

EVENT_KEYS: tuple[str, ...] = (
    "device_id",
    "t_ms",
    "frame_id",
    "state",
    "action",
    "reason",
    "rng",
    "vel",
    "human",
)

VALID_STATES: tuple[str, ...] = ("normal", "suspect", "alarm")


class MonotonicClock:
    """Thread-safe PC monotonic clock starting at 0 ms on creation."""

    def __init__(self, start_s: float | None = None) -> None:
        self._start_s = time.monotonic() if start_s is None else float(start_s)
        self._lock = threading.Lock()

    @property
    def start_s(self) -> float:
        return self._start_s

    def now_s(self) -> float:
        """Return elapsed seconds since clock start."""
        with self._lock:
            return max(0.0, time.monotonic() - self._start_s)

    def now_ms(self) -> int:
        """Return elapsed integer milliseconds since clock start."""
        return int(round(self.now_s() * 1000.0))

    def to_ms(self, monotonic_timestamp_s: float) -> int:
        """Convert an absolute time.monotonic() timestamp into clock-relative ms."""
        return max(0, int(round((float(monotonic_timestamp_s) - self._start_s) * 1000.0)))


def _clean_number(value: float | int) -> float | int:
    fval = float(value)
    if abs(fval) < 1e-9:
        return 0
    if fval.is_integer():
        return int(fval)
    return round(fval, 4)


def make_event(
    *,
    device_id: str,
    t_ms: int,
    frame_id: int,
    state: str = "normal",
    action: str = "none",
    reason: str = "idle",
    rng: float | int = 0,
    vel: float | int = 0,
    human: int = 0,
) -> dict[str, Any]:
    """Build an event dict adhering strictly to the 9-key v2 contract."""
    state_str = str(state)
    if state_str not in VALID_STATES:
        raise ValueError(f"Invalid state {state_str!r}; expected one of {VALID_STATES}")

    return {
        "device_id": str(device_id),
        "t_ms": max(0, int(t_ms)),
        "frame_id": max(0, int(frame_id)),
        "state": state_str,
        "action": str(action),
        "reason": str(reason),
        "rng": _clean_number(rng),
        "vel": _clean_number(vel),
        "human": 1 if int(human) != 0 else 0,
    }


def validate_event(event: dict[str, Any], *, strict_keys: bool = True) -> dict[str, Any]:
    """Validate and normalize an incoming event payload."""
    if not isinstance(event, dict):
        raise ValueError("Event payload must be a JSON object")

    missing = [k for k in EVENT_KEYS if k not in event]
    if missing:
        raise ValueError(f"Missing required event keys: {missing}")

    if strict_keys:
        extra = [k for k in event.keys() if k not in EVENT_KEYS]
        if extra:
            raise ValueError(f"Unexpected extra event keys: {extra}")

    return make_event(
        device_id=event["device_id"],
        t_ms=event["t_ms"],
        frame_id=event["frame_id"],
        state=event["state"],
        action=event["action"],
        reason=event["reason"],
        rng=event["rng"],
        vel=event["vel"],
        human=event["human"],
    )
