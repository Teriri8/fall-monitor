"""UDP receiver for WS63 radar frames, reusing fall_recog.FallDetector."""

from __future__ import annotations

import argparse
import socket
import threading
import time
from typing import Any, Callable

from fall_recog.board_log import RadarSample, parse_radar_payload
from fall_recog.protocol import MonotonicClock
from fall_recog.rules import FallDetector, RadarFallConfig


class UdpRadarSource:
    """Listens on a UDP port for WS63 radar frames and emits 9-key ws63-1 events."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9001,
        clock: MonotonicClock | None = None,
        config: RadarFallConfig | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        recorder: Any | None = None,
        device_id: str = "ws63-1",
    ) -> None:
        self.host = host
        self.port = int(port)
        self.clock = clock or MonotonicClock()
        self.detector = FallDetector(config=config, device_id=device_id)
        self.on_event = on_event
        self.recorder = recorder
        self.device_id = device_id

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self._rx_count = 0
        self._last_sample: RadarSample | None = None
        self._last_event: dict[str, Any] | None = None
        self._last_rx_s: float | None = None

    @property
    def rx_count(self) -> int:
        with self._lock:
            return self._rx_count

    @property
    def last_event(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._last_event) if self._last_event else None

    @property
    def last_rx_s(self) -> float | None:
        with self._lock:
            return self._last_rx_s

    def process_payload(self, raw_payload: bytes | str, t_ms: int | None = None) -> dict[str, Any] | None:
        """Parse a raw payload and advance the radar detector state machine."""
        with self._lock:
            next_id = self._rx_count + 1
        sample = parse_radar_payload(raw_payload, fallback_frame_id=next_id)
        if sample is None:
            return None

        now_s = self.clock.now_s()
        event_t_ms = self.clock.now_ms() if t_ms is None else int(t_ms)
        with self._lock:
            self._rx_count += 1
            event = self.detector.update(sample, event_t_ms)
            self._last_sample = sample
            self._last_event = event
            self._last_rx_s = now_s

        if self.recorder is not None and self.recorder.is_recording():
            self.recorder.write_radar(
                sample.raw,
                {
                    "gear": sample.gear,
                    "rng": sample.rng,
                    "vel": sample.vel,
                    "snr": sample.snr,
                    "human": sample.human,
                },
            )

        if self.on_event is not None:
            self.on_event(event)
        return event

    def reset(self) -> None:
        with self._lock:
            self.detector.reset()

    def start(self) -> None:
        """Bind UDP socket and start background listener thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.settimeout(0.5)
        self._sock = sock
        self._thread = threading.Thread(target=self._loop, name="udp-radar", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._sock is not None
        while not self._stop_event.is_set():
            try:
                data, _addr = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                self.process_payload(data)
            except Exception as exc:
                print(f"[UDP] parse/update error: {exc}")

    def stop(self) -> None:
        self._stop_event.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def send_udp_payload(host: str, port: int, payload: str) -> None:
    """Send a single UDP text/JSON radar payload (test helper)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(payload.encode("utf-8"), (host, port))


def _main() -> None:
    parser = argparse.ArgumentParser(description="WS63 UDP Radar receiver / simulator")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--simulate-fall", action="store_true", help="Send synthetic fall sequence to target host:port")
    parser.add_argument("--simulate-normal", action="store_true", help="Send synthetic normal frame to target host:port")
    parser.add_argument("--target", default="127.0.0.1")
    args = parser.parse_args()

    if args.simulate_normal:
        msg = "[RADAR_SAMPLE] gear=1 rng=2.10 vel=0.04 snr=18.2 hm=1"
        send_udp_payload(args.target, args.port, msg)
        print(f"Sent normal UDP frame to {args.target}:{args.port}: {msg}")
        return

    if args.simulate_fall:
        steps = [
            (0.0, "[RADAR_SAMPLE] gear=1 rng=2.20 vel=0.05 snr=18.0 hm=1"),
            (0.6, "[RADAR_SAMPLE] gear=2 rng=1.65 vel=-1.35 snr=21.5 hm=1"),
            (1.2, "[RADAR_SAMPLE] gear=2 rng=1.60 vel=0.05 snr=17.0 hm=1"),
            (2.5, "[RADAR_SAMPLE] gear=2 rng=1.58 vel=0.02 snr=16.5 hm=1"),
            (4.2, "[RADAR_SAMPLE] gear=2 rng=1.59 vel=0.01 snr=16.2 hm=1"),
        ]
        prev = 0.0
        for t_s, msg in steps:
            if t_s > prev:
                time.sleep(t_s - prev)
                prev = t_s
            send_udp_payload(args.target, args.port, msg)
            print(f"Sent t={t_s:.1f}s -> {msg}")
        return

    src = UdpRadarSource(
        host=args.host,
        port=args.port,
        on_event=lambda ev: print("[UDP EVENT]", ev),
    )
    src.start()
    print(f"Listening for WS63 UDP radar frames on {args.host}:{args.port} (Ctrl+C to quit)")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        src.stop()


if __name__ == "__main__":
    _main()
