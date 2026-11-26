"""Offline ablation evaluator for Section 10 (Radar-only vs Camera-only vs Fusion)."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fall_recog.board_log import parse_radar_payload
from fall_recog.rules import FallDetector
from pc.camera_source import CameraFallDetector, LatestFrameGrabber, YoloPoseRunner
from pc.fusion import FusionArbiter


@dataclass
class ClipResult:
    clip_id: str
    label: str  # "fall" or "normal"
    alarmed: bool
    latency_s: float | None


def _summarize(name: str, results: list[ClipResult]) -> dict[str, Any]:
    falls = [r for r in results if r.label == "fall"]
    normals = [r for r in results if r.label != "fall"]
    tp = sum(1 for r in falls if r.alarmed)
    fp = sum(1 for r in normals if r.alarmed)
    recall = (tp / len(falls) * 100.0) if falls else 0.0
    fpr = (fp / len(normals) * 100.0) if normals else 0.0
    latencies = [r.latency_s for r in falls if r.alarmed and r.latency_s is not None]
    mean_lat = (sum(latencies) / len(latencies)) if latencies else None
    return {
        "name": name,
        "recall": f"{recall:.1f}% ({tp}/{len(falls)})",
        "fpr": f"{fpr:.1f}% ({fp}/{len(normals)})",
        "mean_latency": f"{mean_lat:.2f} s" if mean_lat is not None else "-",
    }


def _load_radar_events(radar_path: str | None) -> list[tuple[float, dict[str, Any]]]:
    if not radar_path:
        return []
    path = Path(radar_path)
    if not path.exists():
        return []
    det = FallDetector()
    out: list[tuple[float, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            t_s = idx * 0.622
            if line.startswith("{"):
                obj = json.loads(line)
                if "t_ms" in obj:
                    t_s = float(obj["t_ms"]) / 1000.0
            sample = parse_radar_payload(line, fallback_frame_id=idx)
            if sample is None:
                continue
            ev = det.update(sample, t_ms=int(round(t_s * 1000.0)))
            out.append((t_s, ev))
    return out


def evaluate_manifest(manifest_path: str, weights: str = "models/yolov8n-pose.pt") -> list[dict[str, Any]]:
    """Run radar-only, camera-only, and fusion evaluation over a JSON or CSV clip manifest."""
    path = Path(manifest_path)
    if path.suffix.lower() == ".json":
        clips = json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open("r", encoding="utf-8") as fh:
            clips = list(csv.DictReader(fh))

    pose_runner = YoloPoseRunner(weights=weights)
    radar_res: list[ClipResult] = []
    cam_res: list[ClipResult] = []
    fusion_res: list[ClipResult] = []

    for i, item in enumerate(clips):
        clip_id = str(item.get("clip_id", f"clip-{i+1}"))
        label = str(item.get("label", "normal")).lower()
        video_path = item.get("video_path")
        radar_path = item.get("radar_log_path")
        fall_start_s = float(item.get("fall_start_s", 0.0) or 0.0)

        radar_seq = _load_radar_events(radar_path)
        r_alarm_t = next((t for t, ev in radar_seq if ev["state"] == "alarm"), None)
        radar_res.append(
            ClipResult(
                clip_id=clip_id,
                label=label,
                alarmed=r_alarm_t is not None,
                latency_s=(r_alarm_t - fall_start_s) if r_alarm_t is not None else None,
            )
        )

        cam_det = CameraFallDetector()
        arbiter = FusionArbiter()
        c_alarm_t: float | None = None
        f_alarm_t: float | None = None

        if video_path and Path(video_path).exists():
            grabber = LatestFrameGrabber(source=video_path)
            grabber.start()
            radar_idx = 0
            try:
                while True:
                    frame, t_ms, fid = grabber.get_latest()
                    if frame is None:
                        break
                    t_s = t_ms / 1000.0
                    while radar_idx < len(radar_seq) and radar_seq[radar_idx][0] <= t_s:
                        rt, rev = radar_seq[radar_idx]
                        fev = arbiter.update_radar(rev, rt)
                        if fev["state"] == "alarm" and f_alarm_t is None:
                            f_alarm_t = rt
                        radar_idx += 1

                    obs, _ms = pose_runner.infer(frame)
                    cev, _metrics = cam_det.update(obs, t_ms=t_ms, frame_id=fid)
                    if cev["state"] == "alarm" and c_alarm_t is None:
                        c_alarm_t = t_s
                    fev = arbiter.update_camera(cev, t_s)
                    if fev["state"] == "alarm" and f_alarm_t is None:
                        f_alarm_t = t_s
            finally:
                grabber.stop()

        cam_res.append(
            ClipResult(
                clip_id=clip_id,
                label=label,
                alarmed=c_alarm_t is not None,
                latency_s=(c_alarm_t - fall_start_s) if c_alarm_t is not None else None,
            )
        )
        fusion_res.append(
            ClipResult(
                clip_id=clip_id,
                label=label,
                alarmed=f_alarm_t is not None,
                latency_s=(f_alarm_t - fall_start_s) if f_alarm_t is not None else None,
            )
        )

    return [
        _summarize("仅雷达", radar_res),
        _summarize("仅摄像头", cam_res),
        _summarize("融合（三取二）", fusion_res),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline ablation benchmark for Fall Detection v2")
    parser.add_argument("--manifest", required=True, help="CSV or JSON manifest of clips")
    parser.add_argument("--weights", default="models/yolov8n-pose.pt")
    args = parser.parse_args()

    rows = evaluate_manifest(args.manifest, weights=args.weights)
    print("| 配置 | 召回率 | 误报率 | 平均报警延迟 |")
    print("|---|---|---|---|")
    for row in rows:
        print(f"| {row['name']} | {row['recall']} | {row['fpr']} | {row['mean_latency']} |")


if __name__ == "__main__":
    main()
