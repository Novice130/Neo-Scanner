"""
Tests for frame differencing and page turn motion detector.
"""

import numpy as np
import pytest

from camscan.motion import PageTurnDetector


def test_motion_detector_lifecycle():
    detector = PageTurnDetector(
        motion_threshold=5.0,
        settle_time_s=0.5,
        cooldown_s=1.0,
    )

    h, w = 120, 160
    still_frame = np.full((h, w, 3), 200, dtype=np.uint8)

    # Initial frame
    trig, score, state = detector.process_frame(still_frame, timestamp=0.0)
    assert not trig
    assert state == PageTurnDetector.STATE_IDLE

    # Consecutive still frame
    trig, score, state = detector.process_frame(still_frame, timestamp=0.1)
    assert not trig
    assert score < 1.0
    assert state == PageTurnDetector.STATE_IDLE

    # Simulated motion (page turn): changing a significant region of pixels
    motion_frame = still_frame.copy()
    motion_frame[20:100, 20:140] = 50

    trig, score, state = detector.process_frame(motion_frame, timestamp=0.2)
    assert not trig
    assert score >= 5.0
    assert state == PageTurnDetector.STATE_MOTION

    # Page settles: frame becomes still again
    settled_frame = motion_frame.copy()
    trig, score, state = detector.process_frame(settled_frame, timestamp=0.3)
    assert not trig
    assert score < 5.0
    assert state == PageTurnDetector.STATE_SETTLING

    # Midway through settle window (0.2s elapsed < 0.5s required)
    trig, score, state = detector.process_frame(settled_frame, timestamp=0.5)
    assert not trig
    assert state == PageTurnDetector.STATE_SETTLING

    # Settle window completed (0.55s elapsed >= 0.5s)
    trig, score, state = detector.process_frame(settled_frame, timestamp=0.85)
    assert trig, "Should trigger capture when settle window elapses"
    assert state == PageTurnDetector.STATE_COOLDOWN

    # Subsequent frame during cooldown should NOT trigger
    trig, score, state = detector.process_frame(settled_frame, timestamp=1.2)
    assert not trig
    assert state == PageTurnDetector.STATE_COOLDOWN


def test_motion_detector_interrupted_settle():
    detector = PageTurnDetector(
        motion_threshold=5.0,
        settle_time_s=0.5,
        cooldown_s=1.0,
    )

    frame_a = np.full((120, 160, 3), 100, dtype=np.uint8)
    frame_b = np.full((120, 160, 3), 200, dtype=np.uint8)

    detector.process_frame(frame_a, timestamp=0.0)
    # Motion starts
    detector.process_frame(frame_b, timestamp=0.1)
    assert detector.state == PageTurnDetector.STATE_MOTION

    # Starts settling
    detector.process_frame(frame_b, timestamp=0.2)
    assert detector.state == PageTurnDetector.STATE_SETTLING

    # Interrupted by new motion before 0.5s
    detector.process_frame(frame_a, timestamp=0.4)
    assert detector.state == PageTurnDetector.STATE_MOTION

    # Must settle afresh
    detector.process_frame(frame_a, timestamp=0.5)
    assert detector.state == PageTurnDetector.STATE_SETTLING

    # Premature
    trig, _, _ = detector.process_frame(frame_a, timestamp=0.8)
    assert not trig

    # Full settle window (0.55s from 0.5)
    trig, _, _ = detector.process_frame(frame_a, timestamp=1.05)
    assert trig
