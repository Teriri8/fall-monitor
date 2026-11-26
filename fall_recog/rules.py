"""Radar fall detection rule state machine (v1 compatible for PC UDP replay)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .board_log import RadarSample
from .protocol import make_event


@dataclass(slots=True)
class RadarFallConfig:
    """Thresholds for WS63 radar fall detection."""

    vel_fall: float = 0.80
    impact_rng: float = 0.35
    vel_still: float = 0.18
    suspect_sec: float = 3.0
    alarm_sec: float = 8.0
    rebound_rng: float = 0.40
    motion_cancel_sec: float = 1.0
    human_loss_reset_sec: float = 3.0
    baseline_alpha: float = 0.15


class FallDetector:
    """Rule-based fall state machine consuming RadarSample frames."""

    def __init__(
        self,
        config: RadarFallConfig | None = None,
        device_id: str = "ws63-1",
    ) -> None:
        self.cfg = config or RadarFallConfig()
        self.device_id = device_id
        self.reset()

    def reset(self) -> None:
        """Reset detector internal state back to normal."""
        self.state = "normal"
        self.action = "none"
        self.reason = "idle"
        self._baseline_rng: float | None = None
        self._prev_rng: float | None = None
        self._impact_detected = False
        self._impact_rng_val = 0.0
        self._impact_t_ms: int | None = None
        self._still_start_ms: int | None = None
        self._motion_start_ms: int | None = None
        self._human_lost_start_ms: int | None = None

    def _clear_fall_tracking(self, *, reason: str, action: str = "none") -> None:
        self.state = "normal"
        self.action = action
        self.reason = reason
        self._impact_detected = False
        self._impact_t_ms = None
        self._still_start_ms = None
        self._motion_start_ms = None

    def update(self, sample: RadarSample, t_ms: int) -> dict[str, Any]:
        """Advance radar fall state machine by one sample and emit a 9-key event."""
        cfg = self.cfg
        t_ms = max(0, int(t_ms))

        if sample.human == 0:
            if self._human_lost_start_ms is None:
                self._human_lost_start_ms = t_ms
            lost_sec = (t_ms - self._human_lost_start_ms) / 1000.0
            if lost_sec >= cfg.human_loss_reset_sec:
                self._clear_fall_tracking(reason="no_human", action="none")
                self._baseline_rng = None
                self._prev_rng = None
            return make_event(
                device_id=self.device_id,
                t_ms=t_ms,
                frame_id=sample.frame_id,
                state=self.state,
                action=self.action,
                reason=self.reason,
                rng=sample.rng,
                vel=sample.vel,
                human=0,
            )

        self._human_lost_start_ms = None

        if self._baseline_rng is None:
            self._baseline_rng = sample.rng
        if self._prev_rng is None:
            self._prev_rng = sample.rng

        delta_base = abs(sample.rng - self._baseline_rng)
        delta_step = abs(sample.rng - self._prev_rng)
        impact_hit = abs(sample.vel) >= cfg.vel_fall and max(delta_base, delta_step) >= cfg.impact_rng

        if impact_hit and self.state == "normal":
            self._impact_detected = True
            self._impact_rng_val = sample.rng
            self._impact_t_ms = t_ms
            self._still_start_ms = None
            self._motion_start_ms = None
            self.action = "fall_impact"
            self.reason = "impact_detected"

        if self._impact_detected:
            if abs(sample.vel) <= cfg.vel_still:
                self._motion_start_ms = None
                if self._still_start_ms is None:
                    self._still_start_ms = t_ms
                still_sec = (t_ms - self._still_start_ms) / 1000.0
                if still_sec >= cfg.alarm_sec:
                    self.state = "alarm"
                    self.action = "fall"
                    self.reason = "fast_then_still"
                elif still_sec >= cfg.suspect_sec:
                    self.state = "suspect"
                    self.action = "fall"
                    self.reason = "fast_then_still"
            else:
                rebounded = (
                    self._still_start_ms is not None
                    and abs(sample.rng - self._impact_rng_val) >= cfg.rebound_rng
                )
                if rebounded:
                    self._clear_fall_tracking(reason="rebound", action="stand")
                else:
                    if self._motion_start_ms is None:
                        self._motion_start_ms = t_ms
                    motion_sec = (t_ms - self._motion_start_ms) / 1000.0
                    if motion_sec >= cfg.motion_cancel_sec:
                        self._clear_fall_tracking(reason="moving", action="walk")
        else:
            if abs(sample.vel) <= cfg.vel_still:
                self.action = "still"
                self.reason = "idle"
            else:
                self.action = "move"
                self.reason = "moving"
            self._baseline_rng = (
                (1.0 - cfg.baseline_alpha) * self._baseline_rng
                + cfg.baseline_alpha * sample.rng
            )

        self._prev_rng = sample.rng

        return make_event(
            device_id=self.device_id,
            t_ms=t_ms,
            frame_id=sample.frame_id,
            state=self.state,
            action=self.action,
            reason=self.reason,
            rng=sample.rng,
            vel=sample.vel,
            human=1,
        )
