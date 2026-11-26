"""Synchronized video + radar UDP session recorder for Section 10 ablation dataset."""

from __future__ import annotations

import csv
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCENARIO_TO_LABEL: dict[str, str] = {
    "fall": "fall",
    "sofa": "normal",
    "sit": "normal",
    "walk": "normal",
    "empty": "normal",
}

MANIFEST_FIELDS: tuple[str, ...] = (
    "clip_id",
    "label",
    "scenario",
    "video_path",
    "radar_log_path",
    "fall_start_s",
    "duration_s",
    "frames",
)


class SessionRecorder:
    """Records synchronized raw camera video (.mp4) and radar UDP frames (.jsonl)."""

    def __init__(self, base_dir: str = "datasets/recorded") -> None:
        self.base_dir = Path(base_dir)
        self.manifest_path = self.base_dir / "manifest.csv"
        self._lock = threading.Lock()
        self._active = False
        self._clip_id = ""
        self._label = "normal"
        self._scenario = "walk"
        self._video_path: Path | None = None
        self._radar_path: Path | None = None
        self._writer: cv2.VideoWriter | None = None
        self._radar_fh = None
        self._start_mono_s = 0.0
        self._frame_count = 0
        self._radar_count = 0
        self._fps = 15.0

    def is_recording(self) -> bool:
        with self._lock:
            return self._active

    def start(
        self,
        *,
        scenario: str = "fall",
        label: str | None = None,
        fps: float = 15.0,
    ) -> dict[str, Any]:
        """Start a new synchronized recording session."""
        scenario_clean = scenario.strip().lower()
        resolved_label = (label or SCENARIO_TO_LABEL.get(scenario_clean, "normal")).strip().lower()
        if resolved_label not in ("fall", "normal"):
            resolved_label = "normal"

        with self._lock:
            if self._active:
                return self._status_locked()

            self.base_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._clip_id = f"{scenario_clean}_{stamp}"
            self._scenario = scenario_clean
            self._label = resolved_label
            self._video_path = self.base_dir / f"{self._clip_id}.mp4"
            self._radar_path = self.base_dir / f"{self._clip_id}_radar.jsonl"
            self._writer = None
            self._radar_fh = self._radar_path.open("w", encoding="utf-8")
            self._start_mono_s = time.monotonic()
            self._frame_count = 0
            self._radar_count = 0
            self._fps = max(5.0, min(30.0, float(fps)))
            self._active = True
            return self._status_locked()

    def write_frame(self, frame: np.ndarray) -> None:
        """Append one BGR frame to the active video file."""
        with self._lock:
            if not self._active or self._video_path is None:
                return
            if self._writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(str(self._video_path), fourcc, self._fps, (w, h))
            if self._writer is not None and self._writer.isOpened():
                self._writer.write(frame)
                self._frame_count += 1

    def write_radar(self, raw_payload: str, sample_dict: dict[str, Any] | None = None) -> None:
        """Append one radar sample aligned to recording-relative t_ms."""
        with self._lock:
            if not self._active or self._radar_fh is None:
                return
            rel_ms = max(0, int(round((time.monotonic() - self._start_mono_s) * 1000.0)))
            record: dict[str, Any] = {"t_ms": rel_ms, "raw": raw_payload}
            if sample_dict:
                record.update(sample_dict)
            self._radar_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._radar_fh.flush()
            self._radar_count += 1

    def stop(self, fall_start_s: float = 0.0) -> dict[str, Any]:
        """Finalize video and radar log files and append entry to manifest.csv."""
        with self._lock:
            if not self._active:
                return {"active": False, "saved": None, "clip_count": self._count_manifest_rows()}

            duration_s = round(max(0.0, time.monotonic() - self._start_mono_s), 2)
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            if self._radar_fh is not None:
                self._radar_fh.close()
                self._radar_fh = None

            self._active = False
            video_rel = str(self._video_path).replace("\\", "/") if self._video_path else ""
            radar_rel = str(self._radar_path).replace("\\", "/") if self._radar_path else ""

            row = {
                "clip_id": self._clip_id,
                "label": self._label,
                "scenario": self._scenario,
                "video_path": video_rel,
                "radar_log_path": radar_rel,
                "fall_start_s": round(float(fall_start_s), 2),
                "duration_s": duration_s,
                "frames": self._frame_count,
            }
            self._append_manifest_locked(row)
            return {
                "active": False,
                "saved": row,
                "radar_frames": self._radar_count,
                "clip_count": self._count_manifest_rows(),
            }

    def _append_manifest_locked(self, row: dict[str, Any]) -> None:
        file_exists = self.manifest_path.exists() and self.manifest_path.stat().st_size > 0
        with self.manifest_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_FIELDS))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _count_manifest_rows(self) -> int:
        if not self.manifest_path.exists():
            return 0
        try:
            with self.manifest_path.open("r", encoding="utf-8") as fh:
                return max(0, sum(1 for _ in fh) - 1)
        except OSError:
            return 0

    def _status_locked(self) -> dict[str, Any]:
        elapsed = round(max(0.0, time.monotonic() - self._start_mono_s), 1) if self._active else 0.0
        return {
            "active": self._active,
            "clip_id": self._clip_id if self._active else None,
            "scenario": self._scenario if self._active else None,
            "label": self._label if self._active else None,
            "elapsed_s": elapsed,
            "frames": self._frame_count if self._active else 0,
            "radar_frames": self._radar_count if self._active else 0,
            "clip_count": self._count_manifest_rows(),
            "manifest_path": str(self.manifest_path).replace("\\", "/"),
        }

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_locked()


def save_snapshot(
    jpeg_bytes: bytes | None,
    *,
    tag: str,
    snapshots_dir: str = "logs/snapshots",
    max_keep: int = 30,
) -> dict[str, Any] | None:
    """Persist an annotated JPEG frame to logs/snapshots and prune old files."""
    if not jpeg_bytes:
        return None
    out_dir = Path(snapshots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in tag)[:24]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    name = f"{stamp}_{safe_tag}.jpg"
    target = out_dir / name
    target.write_bytes(jpeg_bytes)

    files = sorted(out_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in files[max_keep:]:
        try:
            stale.unlink()
        except OSError:
            pass

    return {
        "name": name,
        "tag": safe_tag,
        "url": f"/api/snapshots/{name}",
        "size": len(jpeg_bytes),
        "created_at": datetime.now().strftime("%H:%M:%S"),
    }


def list_snapshots(snapshots_dir: str = "logs/snapshots", limit: int = 8) -> list[dict[str, Any]]:
    """Return recent snapshot metadata sorted newest-first."""
    out_dir = Path(snapshots_dir)
    if not out_dir.exists():
        return []
    files = sorted(out_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    items: list[dict[str, Any]] = []
    for p in files:
        stem = p.stem
        parts = stem.split("_", 3)
        tag = parts[3] if len(parts) >= 4 else "event"
        ts = datetime.fromtimestamp(p.stat().st_mtime).strftime("%H:%M:%S")
        items.append(
            {
                "name": p.name,
                "tag": tag,
                "url": f"/api/snapshots/{p.name}",
                "size": p.stat().st_size,
                "created_at": ts,
            }
        )
    return items
