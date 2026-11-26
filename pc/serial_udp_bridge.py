"""Fallback serial-to-UDP bridge for WS63 radar logs -> PC UDP 9001."""

from __future__ import annotations

import argparse
import socket
import sys
import time

from fall_recog.board_log import parse_radar_payload


def should_forward_line(line: str) -> bool:
    """Return True if a serial line parses as a valid WS63 radar sample."""
    return parse_radar_payload(line) is not None


def list_serial_ports() -> list[tuple[str, str]]:
    """Return available serial ports as (device, description)."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    return [(p.device, p.description) for p in list_ports.comports()]


def run_bridge(
    *,
    serial_port: str,
    baudrate: int = 115200,
    target_host: str = "127.0.0.1",
    target_port: int = 9001,
) -> None:
    """Continuously read WS63 serial lines and forward valid radar samples over UDP."""
    import serial

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sent = 0
    print(f"[BRIDGE] {serial_port}@{baudrate} -> udp://{target_host}:{target_port}")
    try:
        while True:
            try:
                with serial.Serial(serial_port, baudrate=baudrate, timeout=1.0) as ser:
                    print(f"[BRIDGE] Connected to {serial_port}")
                    while True:
                        raw = ser.readline()
                        if not raw:
                            continue
                        line = raw.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        if should_forward_line(line):
                            sock.sendto(line.encode("utf-8"), (target_host, target_port))
                            sent += 1
                            if sent == 1 or sent % 10 == 0:
                                print(f"[BRIDGE] forwarded #{sent}: {line}")
            except serial.SerialException as exc:
                print(f"[BRIDGE] Serial error ({exc}), retrying in 2s...")
                time.sleep(2.0)
    except KeyboardInterrupt:
        print(f"\n[BRIDGE] Stopped. Total forwarded: {sent}")
    finally:
        sock.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="WS63 Serial -> UDP 9001 fallback bridge")
    parser.add_argument("--list-ports", action="store_true", help="List available COM ports")
    parser.add_argument("--port", default="", help="Serial port name, e.g. COM3")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baudrate (default 115200)")
    parser.add_argument("--target-host", default="127.0.0.1", help="PC UDP host")
    parser.add_argument("--target-port", type=int, default=9001, help="PC UDP port")
    args = parser.parse_args()

    if args.list_ports:
        ports = list_serial_ports()
        if not ports:
            print("No serial ports found.")
        for dev, desc in ports:
            print(f"{dev}: {desc}")
        return 0

    if not args.port:
        ports = list_serial_ports()
        print("Please specify --port (e.g. --port COM3). Available ports:")
        for dev, desc in ports:
            print(f"  {dev}: {desc}")
        return 1

    run_bridge(
        serial_port=args.port,
        baudrate=args.baud,
        target_host=args.target_host,
        target_port=args.target_port,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
