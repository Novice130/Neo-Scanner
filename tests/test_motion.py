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


def test_hand_detection_prevents_capture():
    detector = PageTurnDetector(
        motion_threshold=5.0,
        settle_time_s=0.5,
        cooldown_s=1.0,
    )
    h, w = 120, 160
    # Clean paper (light gray)
    paper_frame = np.full((h, w, 3), 220, dtype=np.uint8)

    # Frame with hand holding document (skin color: BGR 140, 170, 220)
    hand_frame = paper_frame.copy()
    hand_frame[40:100, 40:120] = (140, 170, 220)

    # Hand moves into frame
    trig, score, state = detector.process_frame(paper_frame, timestamp=0.0)
    trig, score, state = detector.process_frame(hand_frame, timestamp=0.1)
    assert state in (PageTurnDetector.STATE_MOTION, PageTurnDetector.STATE_HAND_DETECTED)
    assert detector.hand_detected

    # Hand holds book still for 1.0s (longer than settle_time_s 0.5s)
    trig, score, state = detector.process_frame(hand_frame, timestamp=0.5)
    assert not trig
    assert state == PageTurnDetector.STATE_HAND_DETECTED, "Must not trigger while hand is holding book"
    assert detector.hand_detected

    trig, score, state = detector.process_frame(hand_frame, timestamp=1.2)
    assert not trig
    assert state == PageTurnDetector.STATE_HAND_DETECTED, "Must remain blocked while hand is on page"

    # Hand leaves the frame -> paper is clear
    trig, score, state = detector.process_frame(paper_frame, timestamp=1.4)
    assert not detector.hand_detected
    assert state in (PageTurnDetector.STATE_MOTION, PageTurnDetector.STATE_SETTLING)

    # Paper settles without hands
    trig, score, state = detector.process_frame(paper_frame, timestamp=1.5)
    assert not trig
    assert state == PageTurnDetector.STATE_SETTLING

    # Settle time elapses -> cleanly triggers capture!
    trig, score, state = detector.process_frame(paper_frame, timestamp=2.1)
    assert trig, "Should trigger capture after hands are removed and page settles"
    assert state == PageTurnDetector.STATE_COOLDOWN


def test_duplicate_page_prevention():
    detector = PageTurnDetector(
        motion_threshold=5.0,
        settle_time_s=0.3,
        cooldown_s=0.5,
    )
    h, w = 120, 160
    page1 = np.full((h, w, 3), 200, dtype=np.uint8)
    motion_frame = page1.copy()
    motion_frame[20:100, 20:140] = 50

    detector.process_frame(page1, timestamp=0.0)
    detector.process_frame(motion_frame, timestamp=0.1)  # motion
    detector.process_frame(page1, timestamp=0.2)         # returning to still
    detector.process_frame(page1, timestamp=0.3)         # settling starts
    trig, _, state = detector.process_frame(page1, timestamp=0.7)
    assert trig, "First page must trigger"

    # Time passes beyond cooldown (cooldown_s=0.5, last_capture=0.7 -> now 1.5 is past cooldown)
    # Next, simulate turning/moving hand over the EXACT SAME page
    detector.process_frame(motion_frame, timestamp=1.5)
    detector.process_frame(page1, timestamp=1.6)
    detector.process_frame(page1, timestamp=1.7)         # settling starts

    # Settles again on the SAME page
    trig, _, state = detector.process_frame(page1, timestamp=2.1)
    assert not trig, "Should NOT trigger duplicate capture for unchanged page"
    assert state == PageTurnDetector.STATE_IDLE

    # Now turn to a genuinely DIFFERENT page (Page 2)
    page2 = np.full((h, w, 3), 100, dtype=np.uint8)
    page2[30:90, 30:130] = 240
    detector.process_frame(page2, timestamp=2.5)  # motion
    detector.process_frame(page2, timestamp=2.6)  # settling starts
    trig, _, state = detector.process_frame(page2, timestamp=3.0)
    assert trig, "New different page must trigger capture"



