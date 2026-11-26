"""One-click launcher for Fall Detection System v2 (Camera + WS63 UDP + Fusion + Web)."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from datetime import datetime

import uvicorn

from fall_recog.protocol import MonotonicClock
from pc.camera_source import DEFAULT_USB_CAMERA_INDEX, CameraSource, list_cameras
from pc.fusion import FusionArbiter
from pc.server import EventStore, create_app, get_lan_urls
from pc.udp_source import UdpRadarSource


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fall Detection v2 Unified Runner")
    parser.add_argument(
        "--source",
        default=str(DEFAULT_USB_CAMERA_INDEX),
        help="Camera source: USB index (default 1 = HD video), rtsp://..., http://..., or video path",
    )
    parser.add_argument("--weights", default="models/yolov8n-pose.pt", help="YOLOv8 pose weights path")
    parser.add_argument("--device", default=None, help="Inference device: cuda:0 or cpu (auto if omitted)")
    parser.add_argument("--host", default="0.0.0.0", help="Web server bind host")
    parser.add_argument("--port", type=int, default=8000, help="Web server port")
    parser.add_argument("--udp-port", type=int, default=9001, help="WS63 UDP listen port")
    parser.add_argument("--no-camera", action="store_true", help="Disable camera capture (API/UDP test mode)")
    parser.add_argument("--no-udp", action="store_true", help="Disable UDP radar listener")
    parser.add_argument("--show-cv", action="store_true", help="Also open local OpenCV preview window")
    parser.add_argument("--list-cameras", action="store_true", help="List available camera indices and exit")
    parser.add_argument(
        "--no-strict-9-keys",
        action="store_true",
        help="Allow extra keys in POST /api/events (default enforces exact 9-key contract)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_cameras:
        cams = list_cameras()
        if not cams:
            print("No readable camera devices found.")
        for c in cams:
            print(f"[{c['index']}] {c['name']} ({c['width']}x{c['height']}, readable={c['readable']})")
        return 0

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    event_log_path = os.path.join("logs", f"events_{run_tag}.jsonl")
    fusion_log_path = os.path.join("logs", f"fusion_{run_tag}.jsonl")

    clock = MonotonicClock()
    fusion = FusionArbiter(log_path=fusion_log_path)
    store = EventStore(
        clock=clock,
        fusion=fusion,
        strict_9_keys=not args.no_strict_9_keys,
        log_path=event_log_path,
        port=args.port,
    )

    camera_source: CameraSource | None = None
    udp_source: UdpRadarSource | None = None

    def on_cam_event(ev: dict, metrics: dict) -> None:
        store.ingest_event(ev, camera_metrics=metrics, run_fusion=True)

    def on_udp_event(ev: dict) -> None:
        store.ingest_event(ev, run_fusion=True)

    if not args.no_camera:
        camera_source = CameraSource(
            source=args.source,
            weights=args.weights,
            device=args.device,
            clock=clock,
            on_event=on_cam_event,
            show_cv=args.show_cv,
        )

    if not args.no_udp:
        udp_source = UdpRadarSource(
            host=args.host,
            port=args.udp_port,
            clock=clock,
            on_event=on_udp_event,
        )

    app = create_app(
        store=store,
        camera_source=camera_source,
        udp_source=udp_source,
        fusion=fusion,
        clock=clock,
        port=args.port,
    )

    stop_ticker = threading.Event()

    def fusion_ticker() -> None:
        while not stop_ticker.is_set():
            fusion_ev = fusion.tick(clock.now_s())
            with store._lock:
                store._devices[fusion_ev["device_id"]] = fusion_ev
            stop_ticker.wait(1.0)

    ticker_thread = threading.Thread(target=fusion_ticker, name="fusion-ticker", daemon=True)

    try:
        if udp_source is not None:
            udp_source.start()
            print(f"[OK] UDP radar listener started on 0.0.0.0:{args.udp_port}")
        if camera_source is not None:
            camera_source.start()
            print(f"[OK] Camera source started (source={args.source}, device={camera_source.pose.device})")
        ticker_thread.start()

        print("[OK] Web dashboard URLs (PC & Mobile):")
        for url in get_lan_urls(args.port):
            print(f"     -> {url}")
        print(f"[OK] Logs: {event_log_path} | {fusion_log_path}")

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        stop_ticker.set()
        ticker_thread.join(timeout=1.5)
        if camera_source is not None:
            camera_source.stop()
        if udp_source is not None:
            udp_source.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
