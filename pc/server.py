"""FastAPI monitoring server for Fall Detection System v2 (HTTP + SSE + MJPEG)."""

from __future__ import annotations

import asyncio
import collections
import json
import os
import socket
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from fall_recog.protocol import MonotonicClock, make_event, validate_event
from pc.recorder import SessionRecorder, list_snapshots, save_snapshot

WEB_DIR = Path(__file__).resolve().parent / "web"


def get_lan_urls(port: int) -> list[str]:
    """Discover local/LAN HTTP URLs so mobile devices on the same hotspot can connect."""
    urls = [f"http://127.0.0.1:{port}"]
    seen = {"127.0.0.1"}
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        if ip and ip not in seen:
            seen.add(ip)
            urls.append(f"http://{ip}:{port}")
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127.") and ip not in seen:
                seen.add(ip)
                urls.append(f"http://{ip}:{port}")
    except OSError:
        pass
    return urls


def _make_placeholder_jpeg(text: str = "CAM-1 WAITING FOR VIDEO...") -> bytes:
    """Create a lightweight JPEG placeholder when no camera frame is available yet."""
    import cv2
    import numpy as np

    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (16, 22, 34)
    cv2.putText(
        img,
        text,
        (70, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (140, 165, 195),
        2,
        cv2.LINE_AA,
    )
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    return buf.tobytes() if ok else b""


class EventStore:
    """Thread-safe event repository, auto-snapshot trigger, and SSE fan-out."""

    def __init__(
        self,
        clock: MonotonicClock | None = None,
        fusion: Any | None = None,
        camera_source: Any | None = None,
        recorder: SessionRecorder | None = None,
        strict_9_keys: bool = True,
        log_path: str | None = None,
        snapshots_dir: str = "logs/snapshots",
        port: int = 8000,
    ) -> None:
        self.clock = clock or MonotonicClock()
        self.fusion = fusion
        self.camera_source = camera_source
        self.recorder = recorder
        self.strict_9_keys = strict_9_keys
        self.log_path = log_path
        self.snapshots_dir = snapshots_dir
        self.port = port
        if self.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)

        self._lock = threading.Lock()
        self._devices: dict[str, dict[str, Any]] = {}
        self._history: collections.deque[dict[str, Any]] = collections.deque(maxlen=200)
        self._camera_metrics: dict[str, Any] = {}
        self._subscribers: dict[asyncio.Queue[dict[str, Any]], asyncio.AbstractEventLoop] = {}
        self._last_snapshot_mono: dict[str, float] = {}
        self._profile_fallback: dict[str, Any] = {
            "profile": "standard",
            "suspect_sec": 3.0,
            "alarm_sec": 8.0,
            "jitter_tolerance_s": 1.2,
        }

    def _append_log(self, event: dict[str, Any]) -> None:
        if not self.log_path:
            return
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _broadcast_locked(self, event: dict[str, Any]) -> None:
        dead: list[asyncio.Queue[dict[str, Any]]] = []
        for queue, loop in list(self._subscribers.items()):
            if loop.is_closed():
                dead.append(queue)
                continue
            try:
                loop.call_soon_threadsafe(self._put_nowait_drop_oldest, queue, event)
            except RuntimeError:
                dead.append(queue)
        for q in dead:
            self._subscribers.pop(q, None)

    @staticmethod
    def _put_nowait_drop_oldest(queue: asyncio.Queue[dict[str, Any]], item: dict[str, Any]) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(item)

    def _maybe_capture_snapshot(self, ev: dict[str, Any]) -> None:
        """Auto-save annotated frame on alarm or radar_veto with a 5s cooldown."""
        if self.camera_source is None:
            return
        tag: str | None = None
        if ev["state"] == "alarm":
            tag = f"alarm_{ev['reason']}"
        elif ev["reason"] == "radar_veto":
            tag = "radar_veto"
        if tag is None:
            return

        now = time.monotonic()
        with self._lock:
            last_t = self._last_snapshot_mono.get(tag, -1e9)
            if (now - last_t) < 5.0:
                return
            self._last_snapshot_mono[tag] = now

        jpeg = self.camera_source.get_latest_jpeg()
        save_snapshot(jpeg, tag=tag, snapshots_dir=self.snapshots_dir)

    def ingest_event(
        self,
        raw_event: dict[str, Any],
        *,
        camera_metrics: dict[str, Any] | None = None,
        run_fusion: bool = False,
    ) -> dict[str, Any]:
        """Validate, store, optionally fuse, snapshot, log, and broadcast an event."""
        ev = validate_event(raw_event, strict_keys=self.strict_9_keys)
        fusion_ev: dict[str, Any] | None = None

        with self._lock:
            self._devices[ev["device_id"]] = ev
            self._history.appendleft(ev)
            if camera_metrics is not None:
                self._camera_metrics = dict(camera_metrics)
            self._append_log(ev)
            self._broadcast_locked(ev)

        if run_fusion and self.fusion is not None:
            now_s = self.clock.now_s()
            if ev["device_id"] == "cam-1":
                fusion_ev = self.fusion.update_camera(ev, now_s)
            elif ev["device_id"] == "ws63-1":
                fusion_ev = self.fusion.update_radar(ev, now_s)
            if fusion_ev is not None:
                with self._lock:
                    self._devices[fusion_ev["device_id"]] = fusion_ev
                    self._history.appendleft(fusion_ev)
                    self._append_log(fusion_ev)
                    self._broadcast_locked(fusion_ev)
                self._maybe_capture_snapshot(fusion_ev)
        elif ev["device_id"] == "cam-1":
            self._maybe_capture_snapshot(ev)

        return ev

    def update_camera_metrics(self, metrics: dict[str, Any]) -> None:
        with self._lock:
            self._camera_metrics = dict(metrics)

    def set_profile(
        self,
        profile: str,
        suspect_sec: float | None = None,
        alarm_sec: float | None = None,
    ) -> dict[str, Any]:
        if self.camera_source is not None and hasattr(self.camera_source, "set_profile"):
            res = self.camera_source.set_profile(profile, suspect_sec=suspect_sec, alarm_sec=alarm_sec)
            with self._lock:
                self._profile_fallback = dict(res)
            return res

        mode = profile.strip().lower()
        presets = {"standard": (3.0, 8.0), "demo": (1.0, 3.0)}
        with self._lock:
            if mode in presets:
                s_sec, a_sec = presets[mode]
                self._profile_fallback["profile"] = mode
                self._profile_fallback["suspect_sec"] = s_sec
                self._profile_fallback["alarm_sec"] = a_sec
            else:
                self._profile_fallback["profile"] = "custom"
            if suspect_sec is not None:
                self._profile_fallback["suspect_sec"] = max(0.3, float(suspect_sec))
            if alarm_sec is not None:
                self._profile_fallback["alarm_sec"] = max(
                    self._profile_fallback["suspect_sec"] + 0.5, float(alarm_sec)
                )
            return dict(self._profile_fallback)

    def get_profile(self) -> dict[str, Any]:
        if self.camera_source is not None and hasattr(self.camera_source, "get_profile"):
            return self.camera_source.get_profile()
        with self._lock:
            return dict(self._profile_fallback)

    def subscribe(self, loop: asyncio.AbstractEventLoop) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        with self._lock:
            self._subscribers[queue] = loop
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._subscribers.pop(queue, None)

    def snapshot(self) -> dict[str, Any]:
        now_s = self.clock.now_s()
        fusion_status = self.fusion.get_status(now_s) if self.fusion is not None else None
        recording_status = self.recorder.get_status() if self.recorder is not None else None
        profile_info = self.get_profile()
        snaps = list_snapshots(self.snapshots_dir, limit=8)
        with self._lock:
            return {
                "t_ms": self.clock.now_ms(),
                "devices": {k: dict(v) for k, v in self._devices.items()},
                "history": list(self._history)[:50],
                "camera_metrics": dict(self._camera_metrics),
                "fusion_status": fusion_status,
                "profile": profile_info,
                "recording": recording_status,
                "snapshots": snaps,
                "urls": get_lan_urls(self.port),
            }


def create_app(
    *,
    store: EventStore | None = None,
    camera_source: Any | None = None,
    udp_source: Any | None = None,
    fusion: Any | None = None,
    recorder: SessionRecorder | None = None,
    clock: MonotonicClock | None = None,
    snapshots_dir: str = "logs/snapshots",
    port: int = 8000,
) -> FastAPI:
    """Create the FastAPI app for Fall Detection v2."""
    shared_clock = clock or (store.clock if store else MonotonicClock())
    shared_fusion = fusion if fusion is not None else (store.fusion if store else None)
    shared_recorder = recorder if recorder is not None else (store.recorder if store else SessionRecorder())
    shared_store = store or EventStore(
        clock=shared_clock,
        fusion=shared_fusion,
        camera_source=camera_source,
        recorder=shared_recorder,
        snapshots_dir=snapshots_dir,
        port=port,
    )
    if shared_store.camera_source is None and camera_source is not None:
        shared_store.camera_source = camera_source
    if shared_store.recorder is None and shared_recorder is not None:
        shared_store.recorder = shared_recorder

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield

    app = FastAPI(title="Fall Detection v2 Monitor", version="2.1.0", lifespan=lifespan)
    app.state.store = shared_store
    app.state.camera_source = camera_source
    app.state.udp_source = udp_source
    app.state.fusion = shared_fusion
    app.state.recorder = shared_recorder
    app.state.clock = shared_clock

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/state")
    async def get_state() -> JSONResponse:
        if shared_fusion is not None:
            fusion_ev = shared_fusion.tick(shared_clock.now_s())
            with shared_store._lock:
                shared_store._devices[fusion_ev["device_id"]] = fusion_ev
        if camera_source is not None:
            _ev, metrics = camera_source.get_latest_state()
            if metrics:
                shared_store.update_camera_metrics(metrics)
        return JSONResponse(shared_store.snapshot())

    @app.post("/api/events")
    async def post_event(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
        try:
            ev = shared_store.ingest_event(payload, run_fusion=True)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JSONResponse({"ok": True, "event": ev})

    @app.post("/api/mode")
    async def set_mode(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}
        profile = str(body.get("profile", "standard"))
        suspect_sec = body.get("suspect_sec")
        alarm_sec = body.get("alarm_sec")
        updated = shared_store.set_profile(profile, suspect_sec=suspect_sec, alarm_sec=alarm_sec)
        return JSONResponse({"ok": True, "profile": updated, "snapshot": shared_store.snapshot()})

    @app.post("/api/record/start")
    async def start_recording(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}
        scenario = str(body.get("scenario", "fall"))
        label = body.get("label")
        fps = float(shared_store.snapshot().get("camera_metrics", {}).get("fps") or 15.0)
        status = shared_recorder.start(scenario=scenario, label=label, fps=fps)
        return JSONResponse({"ok": True, "recording": status})

    @app.post("/api/record/stop")
    async def stop_recording(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}
        fall_start_s = float(body.get("fall_start_s", 0.0) or 0.0)
        result = shared_recorder.stop(fall_start_s=fall_start_s)
        return JSONResponse({"ok": True, "result": result, "recording": shared_recorder.get_status()})

    @app.get("/api/snapshots")
    async def get_snapshots() -> JSONResponse:
        return JSONResponse({"snapshots": list_snapshots(shared_store.snapshots_dir, limit=20)})

    @app.get("/api/snapshots/{filename}")
    async def get_snapshot_file(filename: str) -> FileResponse:
        if Path(filename).name != filename or not filename.endswith(".jpg"):
            raise HTTPException(status_code=400, detail="Invalid snapshot filename")
        target = Path(shared_store.snapshots_dir) / filename
        if not target.exists():
            raise HTTPException(status_code=404, detail="Snapshot not found")
        return FileResponse(target, media_type="image/jpeg")

    @app.post("/api/reset")
    async def reset_alerts() -> JSONResponse:
        now_s = shared_clock.now_s()
        now_ms = shared_clock.now_ms()
        if camera_source is not None:
            camera_source.reset()
        if udp_source is not None:
            udp_source.reset()
        if shared_fusion is not None:
            fusion_ev = shared_fusion.reset(now_s)
            shared_store.ingest_event(fusion_ev, run_fusion=False)

        for dev_id in ("cam-1", "ws63-1"):
            if dev_id in shared_store.snapshot()["devices"]:
                prev = shared_store.snapshot()["devices"][dev_id]
                shared_store.ingest_event(
                    make_event(
                        device_id=dev_id,
                        t_ms=now_ms,
                        frame_id=int(prev.get("frame_id", 0)),
                        state="normal",
                        action="none",
                        reason="manual_reset",
                        rng=0,
                        vel=0,
                        human=int(prev.get("human", 0)),
                    ),
                    run_fusion=False,
                )
        return JSONResponse({"ok": True, "snapshot": shared_store.snapshot()})

    @app.post("/api/simulate/radar")
    async def simulate_radar(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}
        mode = str(body.get("mode", "normal")).lower()
        now_ms = shared_clock.now_ms()
        snap = shared_store.snapshot()
        prev_ws = snap["devices"].get("ws63-1", {})
        next_fid = int(prev_ws.get("frame_id", 0)) + 1

        if mode == "suspect":
            ev = make_event(
                device_id="ws63-1",
                t_ms=now_ms,
                frame_id=next_fid,
                state="suspect",
                action="fall",
                reason="fast_then_still",
                rng=1.55,
                vel=-1.25,
                human=1,
            )
        elif mode == "alarm":
            ev = make_event(
                device_id="ws63-1",
                t_ms=now_ms,
                frame_id=next_fid,
                state="alarm",
                action="fall",
                reason="fast_then_still",
                rng=1.52,
                vel=0.02,
                human=1,
            )
        else:
            ev = make_event(
                device_id="ws63-1",
                t_ms=now_ms,
                frame_id=next_fid,
                state="normal",
                action="still",
                reason="idle",
                rng=2.10,
                vel=0.03,
                human=1,
            )
        shared_store.ingest_event(ev, run_fusion=True)
        return JSONResponse({"ok": True, "event": ev, "snapshot": shared_store.snapshot()})

    @app.get("/api/events/stream")
    async def event_stream(request: Request) -> StreamingResponse:
        loop = asyncio.get_running_loop()
        queue = shared_store.subscribe(loop)

        async def generator():
            try:
                init_payload = json.dumps(
                    {"type": "snapshot", "data": shared_store.snapshot()},
                    ensure_ascii=False,
                )
                yield f"event: snapshot\ndata: {init_payload}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=10.0)
                        payload = json.dumps(item, ensure_ascii=False)
                        yield f"event: device_event\ndata: {payload}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                shared_store.unsubscribe(queue)

        return StreamingResponse(generator(), media_type="text/event-stream")

    @app.get("/video/cam-1")
    async def video_feed(request: Request) -> StreamingResponse:
        placeholder = _make_placeholder_jpeg()

        async def mjpeg_generator():
            while True:
                if await request.is_disconnected():
                    break
                frame_bytes = camera_source.get_latest_jpeg() if camera_source is not None else None
                chunk = frame_bytes if frame_bytes else placeholder
                if chunk:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(chunk)}\r\n\r\n".encode("ascii")
                        + chunk
                        + b"\r\n"
                    )
                await asyncio.sleep(0.05)

        return StreamingResponse(
            mjpeg_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def _main() -> None:
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    app = create_app(port=port)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    _main()
