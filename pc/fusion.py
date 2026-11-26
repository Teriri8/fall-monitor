"""Dual-modal arbitration (Camera primary + WS63 radar veto/degradation)."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Any

from fall_recog.protocol import make_event


@dataclass(slots=True)
class FusionConfig:
    """Timing windows for camera-radar fusion."""

    window_s: float = 2.0
    radar_timeout_s: float = 5.0


class FusionArbiter:
    """2-of-2 / degraded arbitration over cam-1 and ws63-1 event streams.

    Rules (v2 Section 7.3):
      - Radar missing or timed out (> radar_timeout_s): DEGRADED mode, emit camera state.
      - Camera suspect/alarm AND radar suspect/alarm within window_s: upgrade to alarm.
      - Camera suspect/alarm AND radar normal: hold suspect, emit veto log.
      - Camera occluded (human=0, normal) AND radar suspect/alarm: pass through suspect.
    """

    def __init__(
        self,
        config: FusionConfig | None = None,
        device_id: str = "fusion-1",
        log_path: str | None = None,
    ) -> None:
        self.cfg = config or FusionConfig()
        self.device_id = device_id
        self.log_path = log_path
        if self.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)

        self._lock = threading.Lock()
        self._cam_event: dict[str, Any] | None = None
        self._cam_rx_s: float | None = None
        self._cam_suspect_s: float | None = None

        self._radar_event: dict[str, Any] | None = None
        self._radar_rx_s: float | None = None
        self._radar_suspect_s: float | None = None

        self._mode: str = "INIT"
        self._veto_active = False
        self._degraded_count = 0
        self._veto_count = 0
        self._last_event: dict[str, Any] | None = None
        self._seq = 0

    def _write_log(self, tag: str, payload: dict[str, Any]) -> None:
        if not self.log_path:
            return
        record = {"tag": tag, **payload}
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _is_radar_online(self, now_s: float) -> bool:
        if self._radar_rx_s is None:
            return False
        return (now_s - self._radar_rx_s) <= self.cfg.radar_timeout_s

    def _evaluate_locked(self, now_s: float) -> dict[str, Any]:
        now_s = max(0.0, float(now_s))
        t_ms = int(round(now_s * 1000.0))
        self._seq += 1

        cam = self._cam_event
        radar = self._radar_event
        cam_state = cam["state"] if cam else "normal"
        cam_action = cam["action"] if cam else "none"
        cam_reason = cam["reason"] if cam else "idle"
        cam_human = int(cam["human"]) if cam else 0
        frame_id = int(cam["frame_id"]) if cam else (int(radar["frame_id"]) if radar else self._seq)

        radar_online = self._is_radar_online(now_s)
        radar_state = radar["state"] if (radar and radar_online) else "normal"
        radar_rng = radar["rng"] if (radar and radar_online) else 0
        radar_vel = radar["vel"] if (radar and radar_online) else 0

        cam_alert = cam_state in ("suspect", "alarm")
        radar_recent_alert = (
            radar_online
            and (
                radar_state in ("suspect", "alarm")
                or (
                    self._radar_suspect_s is not None
                    and (now_s - self._radar_suspect_s) <= self.cfg.window_s
                )
            )
        )
        cam_recent_alert = cam_alert or (
            self._cam_suspect_s is not None
            and (now_s - self._cam_suspect_s) <= self.cfg.window_s
            and cam_state != "normal"
        )

        if not radar_online:
            if self._mode != "DEGRADED":
                self._degraded_count += 1
                self._write_log(
                    "DEGRADED",
                    {
                        "t_ms": t_ms,
                        "reason": "radar_timeout_or_absent",
                        "cam_state": cam_state,
                    },
                )
            self._mode = "DEGRADED"
            self._veto_active = False
            event = make_event(
                device_id=self.device_id,
                t_ms=t_ms,
                frame_id=frame_id,
                state=cam_state,
                action=cam_action,
                reason=f"degraded:{cam_reason}" if cam_state != "normal" else cam_reason,
                rng=0,
                vel=0,
                human=cam_human,
            )
            self._last_event = event
            return event

        self._mode = "FUSION"
        radar_human = int(radar["human"]) if radar else 0
        human = 1 if (cam_human or radar_human) else 0

        if (cam_alert and radar_recent_alert) or (cam_recent_alert and radar_state in ("suspect", "alarm")):
            self._veto_active = False
            state = "alarm"
            action = "fall"
            reason = "fusion_confirmed"
        elif cam_alert and radar_state == "normal":
            if not self._veto_active:
                self._veto_count += 1
                self._write_log(
                    "VETO",
                    {
                        "t_ms": t_ms,
                        "cam_state": cam_state,
                        "cam_reason": cam_reason,
                        "radar_state": radar_state,
                        "radar_rng": radar_rng,
                        "radar_vel": radar_vel,
                    },
                )
            self._veto_active = True
            state = "suspect"
            action = "fall"
            reason = "radar_veto"
        elif cam_state == "normal" and cam_human == 0 and radar_state in ("suspect", "alarm"):
            self._veto_active = False
            state = "suspect"
            action = "fall"
            reason = "radar_occluded_fallback"
        else:
            self._veto_active = False
            state = "normal"
            action = cam_action if cam else "none"
            reason = cam_reason if cam else "idle"

        event = make_event(
            device_id=self.device_id,
            t_ms=t_ms,
            frame_id=frame_id,
            state=state,
            action=action,
            reason=reason,
            rng=radar_rng,
            vel=radar_vel,
            human=human,
        )
        self._last_event = event
        return event

    def update_camera(self, event: dict[str, Any], now_s: float) -> dict[str, Any]:
        """Ingest a cam-1 event and return updated fusion-1 event."""
        with self._lock:
            self._cam_event = dict(event)
            self._cam_rx_s = float(now_s)
            if event.get("state") in ("suspect", "alarm"):
                self._cam_suspect_s = float(now_s)
            elif event.get("state") == "normal":
                self._cam_suspect_s = None
            return self._evaluate_locked(now_s)

    def update_radar(self, event: dict[str, Any], now_s: float) -> dict[str, Any]:
        """Ingest a ws63-1 event and return updated fusion-1 event."""
        with self._lock:
            self._radar_event = dict(event)
            self._radar_rx_s = float(now_s)
            if event.get("state") in ("suspect", "alarm"):
                self._radar_suspect_s = float(now_s)
            elif event.get("state") == "normal":
                self._radar_suspect_s = None
            return self._evaluate_locked(now_s)

    def tick(self, now_s: float) -> dict[str, Any]:
        """Re-evaluate fusion state (detects radar timeout even without new radar packets)."""
        with self._lock:
            return self._evaluate_locked(now_s)

    def reset(self, now_s: float) -> dict[str, Any]:
        """Reset transient alert history to normal."""
        with self._lock:
            self._cam_suspect_s = None
            self._radar_suspect_s = None
            self._veto_active = False
            if self._cam_event is not None:
                self._cam_event = make_event(
                    device_id="cam-1",
                    t_ms=int(round(now_s * 1000.0)),
                    frame_id=int(self._cam_event.get("frame_id", 0)),
                    state="normal",
                    action="none",
                    reason="manual_reset",
                    rng=0,
                    vel=0,
                    human=int(self._cam_event.get("human", 0)),
                )
            if self._radar_event is not None:
                self._radar_event = make_event(
                    device_id="ws63-1",
                    t_ms=int(round(now_s * 1000.0)),
                    frame_id=int(self._radar_event.get("frame_id", 0)),
                    state="normal",
                    action="none",
                    reason="manual_reset",
                    rng=self._radar_event.get("rng", 0),
                    vel=self._radar_event.get("vel", 0),
                    human=int(self._radar_event.get("human", 0)),
                )
            return self._evaluate_locked(now_s)

    def get_status(self, now_s: float) -> dict[str, Any]:
        with self._lock:
            online = self._is_radar_online(now_s)
            age = None if self._radar_rx_s is None else round(max(0.0, now_s - self._radar_rx_s), 2)
            return {
                "mode": "FUSION" if online else "DEGRADED",
                "radar_online": online,
                "radar_age_s": age,
                "veto_active": self._veto_active,
                "degraded_count": self._degraded_count,
                "veto_count": self._veto_count,
                "last_event": dict(self._last_event) if self._last_event else None,
            }
