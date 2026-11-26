"""API integration tests for FastAPI server endpoints and 9-key validation."""

import unittest

from fastapi.testclient import TestClient

from fall_recog.protocol import EVENT_KEYS, make_event
from pc.fusion import FusionArbiter
from pc.server import create_app


class TestServerApi(unittest.TestCase):
    def setUp(self) -> None:
        self.fusion = FusionArbiter()
        self.app = create_app(fusion=self.fusion)
        self.client = TestClient(self.app)

    def test_root_and_event_contract(self) -> None:
        r_home = self.client.get("/")
        self.assertEqual(r_home.status_code, 200)
        self.assertIn("跌倒监测系统 v2", r_home.text)

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
        r_post = self.client.post("/api/events", json=cam_ev)
        self.assertEqual(r_post.status_code, 200)

        r_sim = self.client.post("/api/simulate/radar", json={"mode": "suspect"})
        self.assertEqual(r_sim.status_code, 200)

        r_state = self.client.get("/api/state")
        self.assertEqual(r_state.status_code, 200)
        body = r_state.json()
        self.assertIn("cam-1", body["devices"])
        self.assertIn("ws63-1", body["devices"])
        self.assertIn("fusion-1", body["devices"])
        self.assertEqual(body["devices"]["fusion-1"]["state"], "alarm")
        self.assertEqual(tuple(body["devices"]["cam-1"].keys()), EVENT_KEYS)

        r_reset = self.client.post("/api/reset")
        self.assertEqual(r_reset.status_code, 200)
        self.assertEqual(r_reset.json()["snapshot"]["devices"]["fusion-1"]["state"], "normal")


if __name__ == "__main__":
    unittest.main()
