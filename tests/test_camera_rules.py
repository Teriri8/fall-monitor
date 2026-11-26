"""Unit tests for CameraFallDetector three-criterion state machine."""

import unittest

import numpy as np

from fall_recog.protocol import EVENT_KEYS
from pc.camera_source import CameraFallDetector, PoseObservation, compute_torso_angle_deg


def _make_kps(
    shoulder: tuple[float, float],
    hip: tuple[float, float],
    visible_count: int = 17,
) -> np.ndarray:
    kps = np.zeros((17, 3), dtype=np.float32)
    # Place shoulders at 5,6 and hips at 11,12
    kps[5] = [shoulder[0] - 15, shoulder[1], 0.9]
    kps[6] = [shoulder[0] + 15, shoulder[1], 0.9]
    kps[11] = [hip[0] - 12, hip[1], 0.9]
    kps[12] = [hip[0] + 12, hip[1], 0.9]
    for i in range(17):
        if i in (5, 6, 11, 12):
            continue
        kps[i] = [shoulder[0], shoulder[1], 0.8]
    if visible_count < 17:
        for i in range(17 - visible_count):
            idx = 16 - i
            kps[idx, 2] = 0.05
    return kps


class TestCameraRules(unittest.TestCase):
    def test_torso_angle_geometry(self) -> None:
        upright = _make_kps((300, 120), (300, 260))
        lying = _make_kps((200, 380), (360, 380))
        self.assertAlmostEqual(compute_torso_angle_deg(upright), 0.0, delta=1.0)
        self.assertAlmostEqual(compute_torso_angle_deg(lying), 90.0, delta=1.0)

    def test_standing_and_bending_stay_normal(self) -> None:
        det = CameraFallDetector()
        for i in range(25):
            t_ms = i * 200
            obs = PoseObservation(
                bbox=(250, 80, 370, 360),  # w/h = 120/280 = 0.43
                frame_w=640,
                frame_h=480,
                keypoints=_make_kps((310, 120), (310, 230)),
            )
            ev, _m = det.update(obs, t_ms=t_ms, frame_id=i + 1)
            self.assertEqual(tuple(ev.keys()), EVENT_KEYS)
            self.assertEqual(ev["state"], "normal")

        # Bending/squatting: w/h = 0.85, angle ~36 deg -> still normal over 4s
        for j in range(20):
            t_ms = 5000 + j * 200
            obs = PoseObservation(
                bbox=(250, 200, 370, 340),  # w/h = 120/140 = 0.857
                frame_w=640,
                frame_h=480,
                keypoints=_make_kps((345, 220), (295, 290)),  # atan2(50, 70) = 35.5 deg
            )
            ev, _m = det.update(obs, t_ms=t_ms, frame_id=30 + j)
            self.assertEqual(ev["state"], "normal")

    def test_rapid_fall_suspect_alarm_and_recover(self) -> None:
        det = CameraFallDetector()
        # Standing baseline (hip_y = 200/480 = 0.417)
        for i in range(5):
            obs = PoseObservation(
                bbox=(260, 80, 360, 340),
                frame_w=640,
                frame_h=480,
                keypoints=_make_kps((310, 120), (310, 200)),
            )
            det.update(obs, t_ms=i * 200, frame_id=i + 1)

        # Rapid hip drop over 0.4s: hip_y goes from 200 -> 390 (+0.396/0.4s = 0.99/s)
        drop_obs = PoseObservation(
            bbox=(200, 310, 440, 420),  # w/h = 240/110 = 2.18
            frame_w=640,
            frame_h=480,
            keypoints=_make_kps((220, 385), (380, 390)),
        )
        ev, m = det.update(drop_obs, t_ms=1200, frame_id=6)
        self.assertGreaterEqual(m["hip_vy"], 0.45)
        self.assertEqual(ev["state"], "normal")

        # Keep lying still: at t=4200ms (3.0s lying) -> suspect; at t=9200ms (8.0s lying) -> alarm
        seen_suspect = False
        seen_alarm = False
        for step in range(1, 45):
            t_ms = 1200 + step * 200
            ev, _ = det.update(drop_obs, t_ms=t_ms, frame_id=6 + step)
            if t_ms == 4200:
                self.assertEqual(ev["state"], "suspect")
                self.assertEqual(ev["reason"], "fast_then_still")
                seen_suspect = True
            if t_ms == 9200:
                self.assertEqual(ev["state"], "alarm")
                self.assertEqual(ev["reason"], "fast_then_still")
                seen_alarm = True
        self.assertTrue(seen_suspect and seen_alarm)

        # Stand back up for 1.0s -> recovers to normal
        stand_obs = PoseObservation(
            bbox=(260, 80, 360, 340),
            frame_w=640,
            frame_h=480,
            keypoints=_make_kps((310, 120), (310, 200)),
        )
        stand_reasons = []
        for k in range(6):
            ev, _ = det.update(stand_obs, t_ms=10200 + k * 200, frame_id=60 + k)
            stand_reasons.append(ev["reason"])
        self.assertEqual(ev["state"], "normal")
        self.assertIn("recovered", stand_reasons)

    def test_keypoint_loss_plus_bbox_jump_triggers_suspect(self) -> None:
        det = CameraFallDetector()
        stand_obs = PoseObservation(
            bbox=(260, 80, 360, 340),  # w/h = 0.38
            frame_w=640,
            frame_h=480,
            keypoints=_make_kps((310, 120), (310, 200), visible_count=15),
        )
        det.update(stand_obs, t_ms=0, frame_id=1)
        det.update(stand_obs, t_ms=200, frame_id=2)

        # Sudden keypoint loss (only 2 visible) + bbox w/h jumps to 1.8
        lost_obs = PoseObservation(
            bbox=(180, 340, 420, 470),  # w/h = 240/130 = 1.85
            frame_w=640,
            frame_h=480,
            keypoints=_make_kps((220, 390), (380, 395), visible_count=2),
        )
        for idx in range(18):
            ev, _ = det.update(lost_obs, t_ms=400 + idx * 200, frame_id=3 + idx)
        self.assertEqual(ev["state"], "suspect")
        self.assertEqual(ev["reason"], "posture_change")

    def test_brief_jitter_does_not_reset_lying_timer(self) -> None:
        det = CameraFallDetector()
        lying_obs = PoseObservation(
            bbox=(200, 320, 440, 430),
            frame_w=640,
            frame_h=480,
            keypoints=_make_kps((220, 385), (380, 390)),
        )
        det.update(lying_obs, t_ms=0, frame_id=1)
        det.update(lying_obs, t_ms=1500, frame_id=2)
        # Brief 0.4s missing bbox (< 0.6s jitter tolerance)
        det.update(PoseObservation(bbox=None, frame_w=640, frame_h=480), t_ms=1900, frame_id=3)
        ev, _ = det.update(lying_obs, t_ms=3100, frame_id=4)
        self.assertEqual(ev["state"], "suspect")


if __name__ == "__main__":
    unittest.main()
