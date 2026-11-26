"""Unit and integration tests for board_log parser, FallDetector, and UDP receiver."""

import time
import unittest

from fall_recog.board_log import parse_radar_payload
from fall_recog.protocol import EVENT_KEYS
from fall_recog.rules import FallDetector
from pc.serial_udp_bridge import should_forward_line
from pc.udp_source import UdpRadarSource, send_udp_payload


class TestUdpAndRadarRules(unittest.TestCase):
    def test_parse_text_and_json_formats(self) -> None:
        s1 = parse_radar_payload("[RADAR_SAMPLE] gear=2 rng=1.85 vel=-1.20 snr=18.5 hm=1", fallback_frame_id=7)
        self.assertIsNotNone(s1)
        assert s1 is not None
        self.assertEqual(s1.frame_id, 7)
        self.assertEqual(s1.gear, 2)
        self.assertAlmostEqual(s1.rng, 1.85)
        self.assertAlmostEqual(s1.vel, -1.20)
        self.assertEqual(s1.human, 1)

        s2 = parse_radar_payload('{"gear": 0, "rng": 0.0, "vel": 0.0, "snr": 2.0, "human": 0}', fallback_frame_id=8)
        self.assertIsNotNone(s2)
        assert s2 is not None
        self.assertEqual(s2.human, 0)

        self.assertIsNone(parse_radar_payload("random noise line"))

    def test_radar_fall_and_rebound_cancel(self) -> None:
        det = FallDetector()
        base = parse_radar_payload("[RADAR_SAMPLE] gear=1 rng=2.20 vel=0.04 snr=17.0 hm=1", 1)
        assert base is not None
        det.update(base, t_ms=0)

        impact = parse_radar_payload("[RADAR_SAMPLE] gear=2 rng=1.60 vel=-1.30 snr=21.0 hm=1", 2)
        assert impact is not None
        det.update(impact, t_ms=600)

        still = parse_radar_payload("[RADAR_SAMPLE] gear=2 rng=1.58 vel=0.03 snr=18.0 hm=1", 3)
        assert still is not None
        det.update(still, t_ms=1200)
        ev_susp = det.update(still, t_ms=4300)
        self.assertEqual(ev_susp["state"], "suspect")
        self.assertEqual(tuple(ev_susp.keys()), EVENT_KEYS)

        ev_alarm = det.update(still, t_ms=9500)
        self.assertEqual(ev_alarm["state"], "alarm")

        rebound = parse_radar_payload("[RADAR_SAMPLE] gear=1 rng=2.15 vel=0.65 snr=19.0 hm=1", 6)
        assert rebound is not None
        ev_cancel = det.update(rebound, t_ms=10200)
        self.assertEqual(ev_cancel["state"], "normal")
        self.assertEqual(ev_cancel["reason"], "rebound")

    def test_real_udp_loopback_roundtrip(self) -> None:
        received: list[dict] = []
        src = UdpRadarSource(host="127.0.0.1", port=0, on_event=received.append)
        src.start()
        try:
            assert src._sock is not None
            bound_port = src._sock.getsockname()[1]
            send_udp_payload("127.0.0.1", bound_port, "[RADAR_SAMPLE] gear=2 rng=1.75 vel=-0.95 snr=19.0 hm=1")
            deadline = time.monotonic() + 2.0
            while not received and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0]["device_id"], "ws63-1")
            self.assertEqual(tuple(received[0].keys()), EVENT_KEYS)
        finally:
            src.stop()

    def test_serial_bridge_line_filter(self) -> None:
        self.assertTrue(should_forward_line("[RADAR_SAMPLE] gear=2 rng=1.85 vel=-1.20 snr=18.5 hm=1"))
        self.assertTrue(should_forward_line('{"gear":1,"rng":2.0,"vel":0.1,"snr":16.0,"human":1}'))
        self.assertFalse(should_forward_line("booting ws63 wifi driver..."))


if __name__ == "__main__":
    unittest.main()
