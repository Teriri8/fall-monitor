"""Unit tests for FusionArbiter (confirm, veto, degraded, occluded fallback)."""

import tempfile
import unittest
from pathlib import Path

from fall_recog.protocol import make_event
from pc.fusion import FusionArbiter, FusionConfig


class TestFusionArbiter(unittest.TestCase):
    def test_degraded_mode_on_radar_absent_or_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_file = Path(tmp) / "fusion.jsonl"
            arb = FusionArbiter(
                config=FusionConfig(window_s=2.0, radar_timeout_s=5.0),
                log_path=str(log_file),
            )
            cam_suspect = make_event(
                device_id="cam-1",
                t_ms=1000,
                frame_id=10,
                state="suspect",
                action="fall",
                reason="fast_then_still",
                human=1,
            )
            fev = arb.update_camera(cam_suspect, now_s=1.0)
            self.assertEqual(fev["state"], "suspect")
            self.assertTrue(fev["reason"].startswith("degraded:"))
            self.assertEqual(arb.get_status(1.0)["mode"], "DEGRADED")
            self.assertIn("DEGRADED", log_file.read_text(encoding="utf-8"))

    def test_radar_veto_holds_suspect_and_logs_veto(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_file = Path(tmp) / "fusion.jsonl"
            arb = FusionArbiter(
                config=FusionConfig(window_s=2.0, radar_timeout_s=5.0),
                log_path=str(log_file),
            )
            radar_norm = make_event(
                device_id="ws63-1",
                t_ms=1000,
                frame_id=1,
                state="normal",
                action="still",
                reason="idle",
                rng=2.1,
                vel=0.02,
                human=1,
            )
            arb.update_radar(radar_norm, now_s=1.0)

            cam_alarm = make_event(
                device_id="cam-1",
                t_ms=1500,
                frame_id=20,
                state="alarm",
                action="fall",
                reason="posture_change",
                human=1,
            )
            fev = arb.update_camera(cam_alarm, now_s=1.5)
            self.assertEqual(fev["state"], "suspect")
            self.assertEqual(fev["reason"], "radar_veto")
            self.assertTrue(arb.get_status(1.5)["veto_active"])
            self.assertIn("VETO", log_file.read_text(encoding="utf-8"))

    def test_both_suspect_within_window_upgrades_to_alarm(self) -> None:
        arb = FusionArbiter(config=FusionConfig(window_s=2.0, radar_timeout_s=5.0))
        cam_suspect = make_event(
            device_id="cam-1",
            t_ms=2000,
            frame_id=30,
            state="suspect",
            action="fall",
            reason="fast_then_still",
            human=1,
        )
        radar_suspect = make_event(
            device_id="ws63-1",
            t_ms=2800,
            frame_id=5,
            state="suspect",
            action="fall",
            reason="fast_then_still",
            rng=1.5,
            vel=0.01,
            human=1,
        )
        arb.update_camera(cam_suspect, now_s=2.0)
        fev = arb.update_radar(radar_suspect, now_s=2.8)
        self.assertEqual(fev["state"], "alarm")
        self.assertEqual(fev["reason"], "fusion_confirmed")

    def test_occluded_camera_falls_back_to_radar_suspect(self) -> None:
        arb = FusionArbiter(config=FusionConfig(window_s=2.0, radar_timeout_s=5.0))
        cam_lost = make_event(
            device_id="cam-1",
            t_ms=1000,
            frame_id=10,
            state="normal",
            action="none",
            reason="no_human",
            human=0,
        )
        radar_suspect = make_event(
            device_id="ws63-1",
            t_ms=1200,
            frame_id=3,
            state="suspect",
            action="fall",
            reason="fast_then_still",
            rng=1.4,
            vel=0.02,
            human=1,
        )
        arb.update_camera(cam_lost, now_s=1.0)
        fev = arb.update_radar(radar_suspect, now_s=1.2)
        self.assertEqual(fev["state"], "suspect")
        self.assertEqual(fev["reason"], "radar_occluded_fallback")


if __name__ == "__main__":
    unittest.main()
