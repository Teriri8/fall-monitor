"""Camera capture, YOLOv8n-pose inference, and 3-criterion fall detector."""

from __future__ import annotations

import argparse
import collections
import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from fall_recog.protocol import MonotonicClock, make_event

DEFAULT_USB_CAMERA_INDEX = 1  # Verified external USB camera ("HD video") via DirectShow

COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def parse_camera_source(source: str | int) -> tuple[str, int | str]:
    """Classify a camera source spec into ('usb'|'rtsp'|'http'|'file', value)."""
    if isinstance(source, int):
        return ("usb", source)
    text = str(source).strip()
    if text.lower().startswith("usb:"):
        return ("usb", int(text.split(":", 1)[1]))
    if text.isdigit():
        return ("usb", int(text))
    lower = text.lower()
    if lower.startswith("rtsp://"):
        return ("rtsp", text)
    if lower.startswith("http://") or lower.startswith("https://"):
        return ("http", text)
    return ("file", text)


def list_cameras(max_index: int = 4) -> list[dict[str, Any]]:
    """Probe available camera indices and match DirectShow names on Windows."""
    dshow_names: dict[int, str] = {}
    if os.name == "nt":
        ps_script = (
            "$code = @'\n"
            "using System; using System.Collections.Generic; using System.Runtime.InteropServices; using System.Runtime.InteropServices.ComTypes;\n"
            "public class DShowEnum {\n"
            "  [ComImport, Guid(\"62BE5D10-60EB-11d0-BD3B-00A0C911CE86\")] class CreateDevEnum {}\n"
            "  [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid(\"29840822-5B84-11D0-BD3B-00A0C911CE86\")]\n"
            "  interface ICreateDevEnum { [PreserveSig] int CreateClassEnumerator(ref Guid pType, out IEnumMoniker ppEnumMoniker, int dwFlags); }\n"
            "  [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid(\"55272A00-42CB-11CE-8135-00AA004BB851\")]\n"
            "  interface IPropertyBag { [PreserveSig] int Read([MarshalAs(UnmanagedType.LPWStr)] string pszPropName, out object pVar, IntPtr pErrorLog); [PreserveSig] int Write([MarshalAs(UnmanagedType.LPWStr)] string pszPropName, ref object pVar); }\n"
            "  public static List<string> List() {\n"
            "    var res = new List<string>(); Guid clsid = new Guid(\"860BB310-5D01-11d0-BD3B-00A0C911CE86\");\n"
            "    var dev = (ICreateDevEnum)new CreateDevEnum(); IEnumMoniker em;\n"
            "    if (dev.CreateClassEnumerator(ref clsid, out em, 0) != 0 || em == null) return res;\n"
            "    IMoniker[] m = new IMoniker[1]; Guid bg = typeof(IPropertyBag).GUID; int i = 0;\n"
            "    while (em.Next(1, m, IntPtr.Zero) == 0) {\n"
            "      object b; m[0].BindToStorage(null, null, ref bg, out b); var bag = (IPropertyBag)b;\n"
            "      object n = \"\"; try { bag.Read(\"FriendlyName\", out n, IntPtr.Zero); } catch {}\n"
            "      res.Add(i + \"|\" + n); i++; Marshal.ReleaseComObject(bag); Marshal.ReleaseComObject(m[0]);\n"
            "    } return res;\n"
            "  }\n"
            "}\n"
            "'@\n"
            "Add-Type -TypeDefinition $code; [DShowEnum]::List()"
        )
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=8,
            )
            if proc.returncode == 0:
                for line in proc.stdout.splitlines():
                    if "|" in line:
                        idx_s, name = line.split("|", 1)
                        if idx_s.strip().isdigit():
                            dshow_names[int(idx_s.strip())] = name.strip()
        except Exception:
            pass

    results: list[dict[str, Any]] = []
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        results.append(
            {
                "index": idx,
                "name": dshow_names.get(idx, f"Camera {idx}"),
                "opened": True,
                "readable": bool(ret and frame is not None),
                "width": w,
                "height": h,
            }
        )
    return results


@dataclass(slots=True)
class PoseObservation:
    """Single-person pose observation passed into CameraFallDetector."""

    bbox: tuple[float, float, float, float] | None  # x1, y1, x2, y2
    frame_w: int
    frame_h: int
    keypoints: np.ndarray | None = None  # shape (17, 3) -> x, y, conf


@dataclass(slots=True)
class CameraFallConfig:
    """Thresholds for v2 Section 7.1 & 7.2 three-criterion fall state machine."""

    wh_fall_thresh: float = 1.20
    wh_stand_thresh: float = 0.85
    torso_lying_min_deg: float = 55.0
    torso_high_angle_deg: float = 65.0
    torso_high_angle_wh: float = 0.95
    torso_stand_max_deg: float = 35.0
    hip_vy_fall_thresh: float = 0.45
    hip_vy_window_s: float = 0.40
    impact_memory_s: float = 1.20
    kp_conf_thresh: float = 0.30
    kp_loss_thresh: int = 6
    kp_prev_min: int = 8
    kp_loss_wh_jump: float = 0.35
    slow_slump_confirm_s: float = 1.00
    suspect_sec: float = 3.0
    alarm_sec: float = 8.0
    jitter_tolerance_s: float = 0.60
    recover_sec: float = 0.80
    missing_normal_frames: int = 15
    missing_alarm_hold_sec: float = 8.0


def _extract_midpoint(
    keypoints: np.ndarray | None,
    idx_a: int,
    idx_b: int,
    conf_thresh: float,
) -> tuple[float, float] | None:
    if keypoints is None or len(keypoints) <= max(idx_a, idx_b):
        return None
    pts: list[tuple[float, float]] = []
    for idx in (idx_a, idx_b):
        x, y, c = float(keypoints[idx, 0]), float(keypoints[idx, 1]), float(keypoints[idx, 2])
        if c >= conf_thresh and (x > 0.0 or y > 0.0):
            pts.append((x, y))
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def compute_torso_angle_deg(
    keypoints: np.ndarray | None,
    conf_thresh: float = 0.30,
) -> float | None:
    """Compute torso angle relative to vertical downward axis (0=upright, 90=horizontal)."""
    shoulder = _extract_midpoint(keypoints, 5, 6, conf_thresh)
    hip = _extract_midpoint(keypoints, 11, 12, conf_thresh)
    if shoulder is None or hip is None:
        return None
    dx = hip[0] - shoulder[0]
    dy = hip[1] - shoulder[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    return math.degrees(math.atan2(abs(dx), dy))


class CameraFallDetector:
    """Three-criterion fall detector (bbox ratio, torso angle, hip vertical velocity)."""

    def __init__(
        self,
        config: CameraFallConfig | None = None,
        device_id: str = "cam-1",
    ) -> None:
        self.cfg = config or CameraFallConfig()
        self.device_id = device_id
        self.reset()

    def reset(self) -> None:
        self.state = "normal"
        self.action = "none"
        self.reason = "idle"
        self._hip_history: collections.deque[tuple[float, float]] = collections.deque(maxlen=64)
        self._pose_history: collections.deque[tuple[float, float, int]] = collections.deque(maxlen=64)
        self._recent_impact_s: float | None = None
        self._recent_impact_reason: str = "fast_then_still"
        self._lying_start_s: float | None = None
        self._last_lying_seen_s: float | None = None
        self._stand_start_s: float | None = None
        self._missing_frames = 0
        self._missing_start_s: float | None = None

    def _compute_hip_vy(self, now_s: float, hip_y_norm: float) -> float:
        self._hip_history.append((now_s, hip_y_norm))
        while self._hip_history and (now_s - self._hip_history[0][0]) > 1.2:
            self._hip_history.popleft()

        best_t: float | None = None
        best_y: float | None = None
        best_err = 999.0
        for t_prev, y_prev in self._hip_history:
            dt = now_s - t_prev
            if 0.15 <= dt <= 0.70:
                err = abs(dt - self.cfg.hip_vy_window_s)
                if err < best_err:
                    best_err = err
                    best_t = t_prev
                    best_y = y_prev

        if best_t is None or best_y is None:
            return 0.0
        dt = now_s - best_t
        if dt <= 1e-6:
            return 0.0
        return (hip_y_norm - best_y) / dt

    def _check_keypoint_loss_jump(
        self,
        now_s: float,
        wh_ratio: float,
        visible_kps: int,
    ) -> bool:
        self._pose_history.append((now_s, wh_ratio, visible_kps))
        while self._pose_history and (now_s - self._pose_history[0][0]) > 1.5:
            self._pose_history.popleft()

        if visible_kps > self.cfg.kp_loss_thresh or wh_ratio < self.cfg.wh_fall_thresh:
            return False

        for t_prev, prev_wh, prev_kps in self._pose_history:
            if t_prev >= now_s:
                continue
            if prev_kps >= self.cfg.kp_prev_min and (wh_ratio - prev_wh) >= self.cfg.kp_loss_wh_jump:
                return True
        return False

    def update(
        self,
        obs: PoseObservation,
        t_ms: int,
        frame_id: int = 0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Update fall state machine from one frame's PoseObservation."""
        cfg = self.cfg
        t_ms = max(0, int(t_ms))
        now_s = t_ms / 1000.0

        if obs.bbox is None:
            self._missing_frames += 1
            if self._missing_start_s is None:
                self._missing_start_s = now_s
            human = 0 if self._missing_frames >= cfg.missing_normal_frames else 1

            if self.state in ("suspect", "alarm"):
                if (now_s - self._missing_start_s) >= cfg.missing_alarm_hold_sec:
                    self.state = "normal"
                    self.action = "none"
                    self.reason = "no_human"
                    self._lying_start_s = None
                    self._last_lying_seen_s = None
                    self._recent_impact_s = None
            else:
                if human == 0:
                    self.action = "none"
                    self.reason = "no_human"
                if (
                    self._last_lying_seen_s is not None
                    and (now_s - self._last_lying_seen_s) > cfg.jitter_tolerance_s
                ):
                    self._lying_start_s = None
                    self._last_lying_seen_s = None

            event = make_event(
                device_id=self.device_id,
                t_ms=t_ms,
                frame_id=frame_id,
                state=self.state,
                action=self.action,
                reason=self.reason,
                rng=0,
                vel=0,
                human=human,
            )
            metrics = {
                "human": human,
                "wh_ratio": None,
                "hw_ratio": None,
                "torso_angle_deg": None,
                "hip_vy": 0.0,
                "visible_kps": 0,
                "lying_elapsed_s": 0.0 if self._lying_start_s is None else round(now_s - self._lying_start_s, 2),
            }
            return event, metrics

        self._missing_frames = 0
        self._missing_start_s = None

        x1, y1, x2, y2 = obs.bbox
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        wh_ratio = bw / bh
        hw_ratio = bh / bw

        torso_angle_deg = compute_torso_angle_deg(obs.keypoints, cfg.kp_conf_thresh)
        hip_mid = _extract_midpoint(obs.keypoints, 11, 12, cfg.kp_conf_thresh)
        hip_y_px = hip_mid[1] if hip_mid is not None else (y1 + y2) * 0.5
        hip_y_norm = hip_y_px / max(1.0, float(obs.frame_h))
        hip_vy = self._compute_hip_vy(now_s, hip_y_norm)

        visible_kps = 0
        if obs.keypoints is not None and len(obs.keypoints) > 0:
            visible_kps = int(np.sum(obs.keypoints[:, 2] >= cfg.kp_conf_thresh))

        kp_loss_hit = self._check_keypoint_loss_jump(now_s, wh_ratio, visible_kps)

        if hip_vy >= cfg.hip_vy_fall_thresh:
            self._recent_impact_s = now_s
            self._recent_impact_reason = "fast_then_still"
        elif kp_loss_hit:
            self._recent_impact_s = now_s
            self._recent_impact_reason = "posture_change"

        is_standing = (
            wh_ratio <= cfg.wh_stand_thresh
            and (torso_angle_deg is None or torso_angle_deg <= cfg.torso_stand_max_deg)
        )
        is_lying = (
            (
                wh_ratio >= cfg.wh_fall_thresh
                and (torso_angle_deg is None or torso_angle_deg >= cfg.torso_lying_min_deg)
            )
            or (
                torso_angle_deg is not None
                and torso_angle_deg >= cfg.torso_high_angle_deg
                and wh_ratio >= cfg.torso_high_angle_wh
            )
        )

        if is_lying:
            if self._lying_start_s is None:
                self._lying_start_s = now_s
            self._last_lying_seen_s = now_s
            self._stand_start_s = None
            lying_elapsed = now_s - self._lying_start_s

            has_impact = (
                self._recent_impact_s is not None
                and (self._lying_start_s - self._recent_impact_s) <= cfg.impact_memory_s
            )
            fall_reason = self._recent_impact_reason if has_impact else "posture_change"
            eligible = has_impact or kp_loss_hit or lying_elapsed >= cfg.slow_slump_confirm_s

            if eligible and (lying_elapsed + 1e-6) >= cfg.alarm_sec:
                self.state = "alarm"
                self.action = "fall"
                self.reason = fall_reason
            elif eligible and (lying_elapsed + 1e-6) >= cfg.suspect_sec:
                self.state = "suspect"
                self.action = "fall"
                self.reason = fall_reason
            elif self.state == "normal":
                self.action = "lying" if lying_elapsed >= 0.3 else "move"
                self.reason = fall_reason if has_impact else "posture_change"
        else:
            lying_elapsed = 0.0
            if is_standing:
                if self._stand_start_s is None:
                    self._stand_start_s = now_s
                if (now_s - self._stand_start_s) >= cfg.recover_sec:
                    was_alert = self.state in ("suspect", "alarm")
                    self._lying_start_s = None
                    self._last_lying_seen_s = None
                    self._recent_impact_s = None
                    self.state = "normal"
                    self.action = "stand"
                    self.reason = "recovered" if was_alert else "idle"
            else:
                self._stand_start_s = None
                if (
                    self._lying_start_s is not None
                    and self._last_lying_seen_s is not None
                    and (now_s - self._last_lying_seen_s) <= cfg.jitter_tolerance_s
                ):
                    lying_elapsed = now_s - self._lying_start_s
                else:
                    self._lying_start_s = None
                    self._last_lying_seen_s = None
                    if self.state == "normal":
                        self.action = "move"
                        self.reason = "idle"

        event = make_event(
            device_id=self.device_id,
            t_ms=t_ms,
            frame_id=frame_id,
            state=self.state,
            action=self.action,
            reason=self.reason,
            rng=0,
            vel=0,
            human=1,
        )
        metrics = {
            "human": 1,
            "wh_ratio": round(wh_ratio, 3),
            "hw_ratio": round(hw_ratio, 3),
            "torso_angle_deg": None if torso_angle_deg is None else round(torso_angle_deg, 1),
            "hip_vy": round(hip_vy, 3),
            "visible_kps": visible_kps,
            "lying_elapsed_s": round(lying_elapsed, 2),
        }
        return event, metrics


class LatestFrameGrabber:
    """Low-latency camera reader keeping only the freshest frame for live streams."""

    def __init__(
        self,
        source: str | int = DEFAULT_USB_CAMERA_INDEX,
        clock: MonotonicClock | None = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self.kind, self.value = parse_camera_source(source)
        self.clock = clock or MonotonicClock()
        self.width = width
        self.height = height
        self.fps = fps
        self.is_live = self.kind != "file"

        self._lock = threading.Lock()
        self._detector_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_t_ms = 0
        self._frame_seq = 0
        self._file_fps = 30.0

    def _open_capture(self) -> cv2.VideoCapture:
        if self.kind == "rtsp":
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
            cap = cv2.VideoCapture(str(self.value), cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        if self.kind == "http":
            cap = cv2.VideoCapture(str(self.value))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        if self.kind == "usb":
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            cap = cv2.VideoCapture(int(self.value), backend)
            if not cap.isOpened() and backend != cv2.CAP_ANY:
                cap.release()
                cap = cv2.VideoCapture(int(self.value))
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        cap = cv2.VideoCapture(str(self.value))
        fps_val = cap.get(cv2.CAP_PROP_FPS)
        self._file_fps = fps_val if fps_val and fps_val > 1.0 else 30.0
        return cap

    def start(self) -> None:
        if not self.is_live:
            self._cap = self._open_capture()
            if not self._cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {self.value}")
            return

        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._cap = self._open_capture()
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open live camera source: {self.value}")
        self._thread = threading.Thread(target=self._reader_loop, name="frame-grabber", daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.5)
                try:
                    self._cap = self._open_capture()
                except Exception:
                    continue
                if not self._cap.isOpened():
                    continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                self._cap.release()
                self._cap = None
                time.sleep(0.3)
                continue

            now_ms = self.clock.now_ms()
            with self._lock:
                self._frame_seq += 1
                self._latest_frame = frame
                self._latest_t_ms = now_ms

    def get_latest(self) -> tuple[np.ndarray | None, int, int]:
        """Return (frame_copy, t_ms, frame_id) for live mode, or next sequential frame for file mode."""
        if not self.is_live:
            if self._cap is None:
                return (None, 0, self._frame_seq)
            ret, frame = self._cap.read()
            if not ret or frame is None:
                return (None, 0, self._frame_seq)
            self._frame_seq += 1
            t_ms = int(round((self._frame_seq - 1) * (1000.0 / self._file_fps)))
            return (frame, t_ms, self._frame_seq)

        with self._lock:
            if self._latest_frame is None:
                return (None, 0, 0)
            return (self._latest_frame.copy(), self._latest_t_ms, self._frame_seq)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class YoloPoseRunner:
    """Thin wrapper over Ultralytics YOLOv8n-pose selecting the primary tracked person."""

    def __init__(
        self,
        weights: str = "models/yolov8n-pose.pt",
        device: str | None = None,
        conf: float = 0.35,
        imgsz: int = 640,
    ) -> None:
        import torch
        from ultralytics import YOLO

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLO(weights)
        self.model.to(self.device)
        self._last_center: tuple[float, float] | None = None
        # Warm up CUDA/Tensor kernels once at init so the first live frame is fast
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, device=self.device, conf=self.conf, imgsz=self.imgsz, verbose=False)

    def infer(self, frame: np.ndarray) -> tuple[PoseObservation, float]:
        t0 = time.perf_counter()
        results = self.model.predict(
            frame,
            device=self.device,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0
        h, w = frame.shape[:2]

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return PoseObservation(bbox=None, frame_w=w, frame_h=h, keypoints=None), infer_ms

        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        kps_data = (
            results[0].keypoints.data.cpu().numpy()
            if results[0].keypoints is not None and results[0].keypoints.data is not None
            else None
        )

        best_idx = 0
        if len(boxes_xyxy) > 1:
            best_score = -1e9
            for i, box in enumerate(boxes_xyxy):
                bx1, by1, bx2, by2 = box
                bw = max(1.0, float(bx2 - bx1))
                bh = max(1.0, float(by2 - by1))
                area_norm = (bw * bh) / max(1.0, float(w * h))
                wh_bonus = 0.25 if (bw / bh) >= 1.15 else 0.0
                cx, cy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
                dist_penalty = 0.0
                if self._last_center is not None:
                    dist_penalty = math.hypot(cx - self._last_center[0], cy - self._last_center[1]) / max(w, h)
                score = area_norm + wh_bonus - 0.35 * dist_penalty
                if score > best_score:
                    best_score = score
                    best_idx = i

        sel_box = tuple(float(v) for v in boxes_xyxy[best_idx][:4])
        self._last_center = ((sel_box[0] + sel_box[2]) * 0.5, (sel_box[1] + sel_box[3]) * 0.5)
        sel_kps = kps_data[best_idx] if kps_data is not None and best_idx < len(kps_data) else None
        return PoseObservation(bbox=sel_box, frame_w=w, frame_h=h, keypoints=sel_kps), infer_ms


def annotate_frame(
    frame: np.ndarray,
    obs: PoseObservation,
    event: dict[str, Any],
    metrics: dict[str, Any],
    fps: float,
    infer_ms: float,
) -> np.ndarray:
    """Draw pose skeleton, bounding box, and telemetry HUD onto frame copy."""
    canvas = frame.copy()
    state = event["state"]
    color_map = {
        "normal": (46, 204, 113),
        "suspect": (0, 180, 255),
        "alarm": (48, 48, 255),
    }
    color = color_map.get(state, (200, 200, 200))

    if obs.bbox is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in obs.bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

    if obs.keypoints is not None:
        kps = obs.keypoints
        for a, b in COCO_SKELETON:
            if a < len(kps) and b < len(kps):
                xa, ya, ca = kps[a]
                xb, yb, cb = kps[b]
                if ca >= 0.3 and cb >= 0.3:
                    cv2.line(
                        canvas,
                        (int(round(xa)), int(round(ya))),
                        (int(round(xb)), int(round(yb))),
                        (255, 210, 60),
                        2,
                    )
        for kp in kps:
            x, y, c = kp
            if c >= 0.3:
                cv2.circle(canvas, (int(round(x)), int(round(y))), 4, color, -1)

    cv2.rectangle(canvas, (8, 8), (520, 96), (18, 22, 28), -1)
    cv2.rectangle(canvas, (8, 8), (520, 96), color, 2)
    line1 = f"CAM-1 | STATE: {state.upper()} ({event['reason']}) | FPS: {fps:.1f} | {infer_ms:.1f}ms"
    line2 = (
        f"w/h: {metrics.get('wh_ratio') or '-'} | "
        f"angle: {metrics.get('torso_angle_deg') if metrics.get('torso_angle_deg') is not None else '-'} deg | "
        f"hip_vy: {metrics.get('hip_vy', 0.0):+.2f} | "
        f"kps: {metrics.get('visible_kps', 0)}/17 | "
        f"lie: {metrics.get('lying_elapsed_s', 0.0):.1f}s"
    )
    cv2.putText(canvas, line1, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, line2, (16, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 236, 245), 1, cv2.LINE_AA)
    return canvas


class CameraSource:
    """Runs frame grabber + YOLOv8n-pose + CameraFallDetector in a worker thread."""

    def __init__(
        self,
        source: str | int = DEFAULT_USB_CAMERA_INDEX,
        weights: str = "models/yolov8n-pose.pt",
        device: str | None = None,
        clock: MonotonicClock | None = None,
        config: CameraFallConfig | None = None,
        on_event: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
        show_cv: bool = False,
        heartbeat_s: float = 0.5,
    ) -> None:
        self.clock = clock or MonotonicClock()
        self.grabber = LatestFrameGrabber(source=source, clock=self.clock)
        self.pose = YoloPoseRunner(weights=weights, device=device)
        self.detector = CameraFallDetector(config=config, device_id="cam-1")
        self.on_event = on_event
        self.show_cv = show_cv
        self.heartbeat_s = heartbeat_s

        self._lock = threading.Lock()
        self._detector_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_jpeg: bytes | None = None
        self._latest_event: dict[str, Any] | None = None
        self._latest_metrics: dict[str, Any] = {}

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.grabber.start()
        self._thread = threading.Thread(target=self._run_loop, name="camera-pose", daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        last_processed_id = -1
        last_emit_s = -1.0
        last_state: str | None = None
        fps_ema = 0.0
        prev_tick = time.perf_counter()

        while not self._stop_event.is_set():
            try:
                frame, t_ms, frame_id = self.grabber.get_latest()
                if frame is None or frame_id == last_processed_id:
                    if not self.grabber.is_live and last_processed_id > 0:
                        break
                    time.sleep(0.005)
                    continue
                last_processed_id = frame_id

                obs, infer_ms = self.pose.infer(frame)
                with self._detector_lock:
                    event, metrics = self.detector.update(obs, t_ms=t_ms, frame_id=frame_id)

                now_perf = time.perf_counter()
                dt = max(1e-4, now_perf - prev_tick)
                prev_tick = now_perf
                inst_fps = 1.0 / dt
                fps_ema = inst_fps if fps_ema == 0.0 else (0.85 * fps_ema + 0.15 * inst_fps)

                full_metrics = {
                    **metrics,
                    "fps": round(fps_ema, 1),
                    "infer_ms": round(infer_ms, 1),
                    "device": str(self.pose.device),
                }

                annotated = annotate_frame(frame, obs, event, full_metrics, fps_ema, infer_ms)
                ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                jpeg_bytes = buf.tobytes() if ok else None

                with self._lock:
                    if jpeg_bytes is not None:
                        self._latest_jpeg = jpeg_bytes
                    self._latest_event = event
                    self._latest_metrics = full_metrics

                now_s = t_ms / 1000.0
                state_changed = event["state"] != last_state
                if state_changed or (now_s - last_emit_s) >= self.heartbeat_s:
                    last_state = event["state"]
                    last_emit_s = now_s
                    if self.on_event is not None:
                        self.on_event(event, full_metrics)

                if self.show_cv:
                    cv2.imshow("Fall Detection v2 - cam-1", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self._stop_event.set()
                        break
            except Exception as exc:
                print(f"[CAM] loop error: {exc}")
                time.sleep(0.1)

        if self.show_cv:
            cv2.destroyAllWindows()

    def get_latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def get_latest_state(self) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        with self._lock:
            ev = dict(self._latest_event) if self._latest_event else None
            return ev, dict(self._latest_metrics)

    def reset(self) -> None:
        with self._detector_lock:
            self.detector.reset()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.grabber.stop()


def _main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8n-pose camera source for Fall Detection v2")
    parser.add_argument("--list-cameras", action="store_true", help="List available camera indices")
    parser.add_argument("--source", default=str(DEFAULT_USB_CAMERA_INDEX), help="USB index, rtsp://, http://, or video file")
    parser.add_argument("--weights", default="models/yolov8n-pose.pt")
    parser.add_argument("--device", default=None, help="cuda:0 or cpu")
    parser.add_argument("--show-cv", action="store_true", help="Show local OpenCV preview window")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds (0=forever)")
    args = parser.parse_args()

    if args.list_cameras:
        for cam in list_cameras():
            print(cam)
        return

    src = CameraSource(
        source=args.source,
        weights=args.weights,
        device=args.device,
        show_cv=args.show_cv,
        on_event=lambda ev, m: print(f"[CAM] {ev} | fps={m.get('fps')} infer={m.get('infer_ms')}ms"),
    )
    src.start()
    t0 = time.monotonic()
    try:
        while True:
            time.sleep(0.2)
            if args.duration > 0 and (time.monotonic() - t0) >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        src.stop()


if __name__ == "__main__":
    _main()
