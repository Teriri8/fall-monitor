"""Parser for WS63 radar log frames (text [RADAR_SAMPLE] and JSON)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_KV_PATTERN = re.compile(r"\b([a-zA-Z_]+)\s*[:=]\s*([-+]?\d+(?:\.\d+)?)")


@dataclass(slots=True)
class RadarSample:
    """Normalized single-frame radar measurement from WS63."""

    frame_id: int
    gear: int
    rng: float
    vel: float
    snr: float
    human: int
    lb: float | None = None
    hb: float | None = None
    hm: int | None = None
    raw: str = ""


def _to_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _to_int(val: Any, default: int = 0) -> int:
    try:
        return int(round(float(val)))
    except (TypeError, ValueError):
        return default


def _infer_human(
    *,
    human_val: Any | None,
    hm_val: int | None,
    gear_val: int,
    snr_val: float,
    rng_val: float,
) -> int:
    if human_val is not None:
        return 1 if _to_int(human_val, 0) > 0 else 0
    if hm_val is not None:
        return 1 if hm_val > 0 else 0
    if gear_val > 0 or (snr_val > 0.0 and rng_val > 0.0):
        return 1
    return 0


def parse_radar_payload(
    raw_payload: bytes | str,
    fallback_frame_id: int = 0,
) -> RadarSample | None:
    """Parse a WS63 UDP/serial radar payload into RadarSample.

    Supported formats:
    1) Text log: "[RADAR_SAMPLE] gear=2 rng=1.85 vel=-1.20 snr=18.5"
       (optionally with frame_id=..., lb=..., hb=..., hm=..., human=...)
    2) JSON object: {"gear": 2, "rng": 1.85, "vel": -1.2, "snr": 18.5, "human": 1}
    """
    if isinstance(raw_payload, bytes):
        text = raw_payload.decode("utf-8", errors="replace").strip()
    else:
        text = str(raw_payload).strip()

    if not text:
        return None

    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None

        recognized = {"gear", "rng", "vel", "snr", "hm", "human", "lb", "hb"}
        if not any(k in data for k in recognized):
            return None

        frame_id = _to_int(data.get("frame_id", data.get("fid")), fallback_frame_id)
        gear = _to_int(data.get("gear"), 0)
        rng = _to_float(data.get("rng"), 0.0)
        vel = _to_float(data.get("vel"), 0.0)
        snr = _to_float(data.get("snr"), 0.0)
        lb = _to_float(data["lb"]) if "lb" in data and data["lb"] is not None else None
        hb = _to_float(data["hb"]) if "hb" in data and data["hb"] is not None else None
        hm = _to_int(data["hm"]) if "hm" in data and data["hm"] is not None else None
        human = _infer_human(
            human_val=data.get("human"),
            hm_val=hm,
            gear_val=gear,
            snr_val=snr,
            rng_val=rng,
        )
        return RadarSample(
            frame_id=max(0, frame_id),
            gear=gear,
            rng=rng,
            vel=vel,
            snr=snr,
            human=human,
            lb=lb,
            hb=hb,
            hm=hm,
            raw=text,
        )

    kv: dict[str, str] = {}
    for key, val in _KV_PATTERN.findall(text):
        kv[key.lower()] = val

    if not any(k in kv for k in ("gear", "rng", "vel", "snr", "hm", "human")):
        return None

    frame_id = _to_int(kv.get("frame_id", kv.get("fid")), fallback_frame_id)
    gear = _to_int(kv.get("gear"), 0)
    rng = _to_float(kv.get("rng"), 0.0)
    vel = _to_float(kv.get("vel"), 0.0)
    snr = _to_float(kv.get("snr"), 0.0)
    lb = _to_float(kv["lb"]) if "lb" in kv else None
    hb = _to_float(kv["hb"]) if "hb" in kv else None
    hm = _to_int(kv["hm"]) if "hm" in kv else None
    human = _infer_human(
        human_val=kv.get("human"),
        hm_val=hm,
        gear_val=gear,
        snr_val=snr,
        rng_val=rng,
    )

    return RadarSample(
        frame_id=max(0, frame_id),
        gear=gear,
        rng=rng,
        vel=vel,
        snr=snr,
        human=human,
        lb=lb,
        hb=hb,
        hm=hm,
        raw=text,
    )
