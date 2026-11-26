"""API integration tests for FastAPI server endpoints, mode switch, recorder, and snapshots."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from fall_recog.protocol import EVENT_KEYS, make_event
from pc.fusion import FusionArbiter
from pc.recorder import SessionRecorder
from pc.server import create_app


class _FakeCameraSource:
    def __init__(self) -> None:
        self._profile = {"profile": "standard", "suspect_sec": 3.0, "alarm_sec": 8.0, "jitter_tolerance_s": 1.2}
        self._jpeg = b"\xff\xd8\xff\xe0FAKEJPEGDATA\xff\xd9"

    def get_latest_jpeg(self) -> bytes:
        return self._jpeg

    def get_latest_state(self):
        return None, {"fps": 20.0}

    def reset(self) -> None:
        pass

    def set_profile(self, profile: str, suspect_sec=None, alarm_sec=None) -> dict:
        if profile == "demo":
            self._profile.update({"profile": "demo", "suspect_sec": 1.0, "alarm_sec": 3.0})
        else:
            self._profile.update({"profile": "standard", "suspect_sec": 3.0, "alarm_sec": 8.0})
        return dict(self._profile)

    def get_profile(self) -> dict:
        return dict(self._profile)


class TestServerApi(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.fusion = FusionArbiter()
        self.fake_cam = _FakeCameraSource()
        self.recorder = SessionRecorder(base_dir=str(self.tmp_dir / "recorded"))
        self.snap_dir = str(self.tmp_dir / "snapshots")
        self.app = create_app(
            fusion=self.fusion,
            camera_source=self.fake_cam,
            recorder=self.recorder,
            snapshots_dir=self.snap_dir,
        )
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        if self.recorder.is_recording():
            self.recorder.stop()
        self._tmp.cleanup()

    def test_root_event_contract_and_auto_snapshot(self) -> None:
        r_home = self.client.get("/")
        self.assertEqual(r_home.status_code, 200)
        self.assertIn("跌倒监测系统", r_home.text)

        bad_missing = {"device_id": "cam-1", "t_ms": 100}
        self.assertEqual(self.client.post("/api/events", json=bad_missing).status_code, 422)

        bad_extra = make_event(device_id="cam-1", t_ms=100, frame_id=1)
        bad_extra["extra_field"] = 123
        self.assertEqual(self.client.post("/api/events", json=bad_extra).status_code, 422)

        cam_ev = make_event(
            device_id="cam-1",
            t_ms=500,
            frame_id=12,
            state="suspect",
            action="fall",
            reason="fast_then_still",
            human=1,
        )
        self.assertEqual(self.client.post("/api/events", json=cam_ev).status_code, 200)
        self.assertEqual(self.client.post("/api/simulate/radar", json={"mode": "suspect"}).status_code, 200)

        r_state = self.client.get("/api/state")
        self.assertEqual(r_state.status_code, 200)
        body = r_state.json()
        self.assertIn("cam-1", body["devices"])
        self.assertIn("ws63-1", body["devices"])
        self.assertIn("fusion-1", body["devices"])
        self.assertEqual(body["devices"]["fusion-1"]["state"], "alarm")
        self.assertEqual(tuple(body["devices"]["cam-1"].keys()), EVENT_KEYS)

        # Alarm should have auto-captured a snapshot
        snaps = self.client.get("/api/snapshots").json()["snapshots"]
        self.assertGreaterEqual(len(snaps), 1)
        r_img = self.client.get(snaps[0]["url"])
        self.assertEqual(r_img.status_code, 200)
        self.assertTrue(r_img.content.startswith(b"\xff\xd8"))

        r_reset = self.client.post("/api/reset")
        self.assertEqual(r_reset.status_code, 200)
        self.assertEqual(r_reset.json()["snapshot"]["devices"]["fusion-1"]["state"], "normal")

    def test_mode_switch_and_synchronized_recording(self) -> None:
        r_mode = self.client.post("/api/mode", json={"profile": "demo"})
        self.assertEqual(r_mode.status_code, 200)
        self.assertEqual(r_mode.json()["profile"]["suspect_sec"], 1.0)
        self.assertEqual(r_mode.json()["profile"]["alarm_sec"], 3.0)

        r_start = self.client.post("/api/record/start", json={"scenario": "sofa"})
        self.assertEqual(r_start.status_code, 200)
        self.assertTrue(r_start.json()["recording"]["active"])
        self.assertEqual(r_start.json()["recording"]["label"], "normal")

        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.recorder.write_frame(dummy_frame)
        self.recorder.write_frame(dummy_frame)
        self.recorder.write_radar("[RADAR_SAMPLE] gear=1 rng=2.1 vel=0.02 snr=17.0 hm=1", {"gear": 1})

        r_stop = self.client.post("/api/record/stop", json={"fall_start_s": 0.0})
        self.assertEqual(r_stop.status_code, 200)
        saved = r_stop.json()["result"]["saved"]
        self.assertIsNotNone(saved)
        self.assertEqual(saved["frames"], 2)
        self.assertTrue(Path(saved["video_path"]).exists())
        self.assertTrue(Path(saved["radar_log_path"]).exists())
        self.assertTrue((self.tmp_dir / "recorded" / "manifest.csv").exists())


if __name__ == "__main__":
    unittest.main()
