"""
Module for detecting page turns and document settling using frame differencing.
"""

import time
import typing as t

import cv2
import numpy as np


class PageTurnDetector:
    """
    Detects page turn events using consecutive frame differencing.

    Lifecycle:
    1. IDLE: Document is stationary.
    2. MOTION: Motion detected above threshold (e.g., hand turning a page).
    3. SETTLING: Motion drops below threshold and stays quiet for a settle duration.
    4. TRIGGER: Settle duration achieved -> triggers capture!
    5. COOLDOWN: Short delay to prevent repeated captures of the same still page.
    """

    STATE_IDLE = "IDLE"
    STATE_MOTION = "MOTION"
    STATE_SETTLING = "SETTLING"
    STATE_COOLDOWN = "COOLDOWN"

    def __init__(
        self,
        motion_threshold: float = 3.0,
        settle_time_s: float = 0.8,
        cooldown_s: float = 2.0,
        diff_resolution: tuple[int, int] = (160, 120),
    ):
        """
        :param motion_threshold: Percentage of pixels changed (0.0 to 100.0)
        :param settle_time_s: Duration in seconds motion must stay below threshold
        :param cooldown_s: Cooldown after capture before another trigger can occur
        :param diff_resolution: Resolution to downscale for motion differencing
        """
        self.motion_threshold = motion_threshold
        self.settle_time_s = settle_time_s
        self.cooldown_s = cooldown_s
        self.diff_resolution = diff_resolution

        self._prev_gray = None
        self.state = self.STATE_IDLE
        self._settle_start = None
        self._last_capture_time = -float("inf")

    def reset(self):
        """Reset internal tracking state."""
        self._prev_gray = None
        self.state = self.STATE_IDLE
        self._settle_start = None
        self._last_capture_time = -float("inf")

    def process_frame(
        self, frame: np.ndarray, timestamp: t.Optional[float] = None
    ) -> tuple[bool, float, str]:
        """
        Process incoming camera frame.
        :param frame: BGR image from camera
        :param timestamp: Optional explicit timestamp (for deterministic testing)
        :return: (should_capture, motion_score, current_state)
        """
        now = timestamp if timestamp is not None else time.time()

        # Downscale and convert to grayscale for lightweight motion computation
        small = cv2.resize(frame, self.diff_resolution, interpolation=cv2.INTER_AREA)
        if len(small.shape) == 3:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        else:
            gray = small

        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._prev_gray is None:
            self._prev_gray = gray_blur
            return False, 0.0, self.state

        # Absolute frame differencing
        diff = cv2.absdiff(self._prev_gray, gray_blur)
        self._prev_gray = gray_blur

        # Threshold difference
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        motion_score = (float(np.count_nonzero(thresh)) / thresh.size) * 100.0

        # Check cooldown
        if now - self._last_capture_time < self.cooldown_s:
            self.state = self.STATE_COOLDOWN
            return False, motion_score, self.state

        should_trigger = False

        if self.state in (self.STATE_IDLE, self.STATE_COOLDOWN):
            if motion_score >= self.motion_threshold:
                self.state = self.STATE_MOTION
                self._settle_start = None

        elif self.state == self.STATE_MOTION:
            if motion_score < self.motion_threshold:
                # Motion ceased, begin settle window
                self.state = self.STATE_SETTLING
                self._settle_start = now
            # else stay in MOTION

        elif self.state == self.STATE_SETTLING:
            if motion_score >= self.motion_threshold:
                # Motion resumed before settle window finished
                self.state = self.STATE_MOTION
                self._settle_start = None
            else:
                elapsed = now - (self._settle_start or now)
                if elapsed >= self.settle_time_s:
                    # Settled! Trigger capture
                    should_trigger = True
                    self._last_capture_time = now
                    self.state = self.STATE_COOLDOWN
                    self._settle_start = None

        return should_trigger, motion_score, self.state
