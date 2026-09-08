"""
Module for detecting page turns, hand presence, and document settling using frame differencing and skin tone analysis.
"""

import time
import typing as t

import cv2
import numpy as np


class PageTurnDetector:
    """
    Detects page turn events using consecutive frame differencing,
    coupled with YCrCb skin-color detection to prevent snapping while
    hands/fingers are holding the book or in the frame.

    Lifecycle:
    1. IDLE: Document is stationary; waiting for a page turn.
    2. MOTION: Motion detected above threshold (turning page, moving hands).
    3. HAND_DETECTED: Hands/fingers are touching or holding the page.
    4. SETTLING: Hands have released the book; document is stationary for settle duration.
    5. TRIGGER: Settle duration achieved -> triggers capture!
    6. COOLDOWN: Short delay to prevent repeated captures of the same page.
    """

    STATE_IDLE = "IDLE"
    STATE_MOTION = "MOTION"
    STATE_HAND_DETECTED = "HAND_DETECTED"
    STATE_SETTLING = "SETTLING"
    STATE_COOLDOWN = "COOLDOWN"

    def __init__(
        self,
        motion_threshold: float = 3.0,
        settle_time_s: float = 0.8,
        cooldown_s: float = 2.0,
        diff_resolution: tuple[int, int] = (160, 120),
        skin_threshold: float = 0.025,
        auto_trigger_initial: bool = False,
    ):
        """
        :param motion_threshold: Percentage of pixels changed (0.0 to 100.0)
        :param settle_time_s: Duration in seconds motion must stay below threshold
        :param cooldown_s: Cooldown after capture before another trigger can occur
        :param diff_resolution: Resolution to downscale for motion differencing
        :param skin_threshold: Fraction of document area covered by skin to count as hand holding
        :param auto_trigger_initial: If True, settle and capture initial page without prior motion
        """
        self.motion_threshold = motion_threshold
        self.settle_time_s = settle_time_s
        self.cooldown_s = cooldown_s
        self.diff_resolution = diff_resolution
        self.skin_threshold = skin_threshold
        self.auto_trigger_initial = auto_trigger_initial

        self._prev_gray = None
        self.state = self.STATE_IDLE
        self._settle_start = None
        self._last_capture_time = -float("inf")

        # Hand / skin detection state
        self.hand_detected = False
        self.skin_ratio = 0.0

        # Duplicate page prevention
        self._last_captured_thumb: t.Optional[np.ndarray] = None

    def reset(self):
        """Reset internal tracking state and capture reference."""
        self._prev_gray = None
        self.state = self.STATE_IDLE
        self._settle_start = None
        self._last_capture_time = -float("inf")
        self.hand_detected = False
        self.skin_ratio = 0.0
        self._last_captured_thumb = None

    def arm_new_session(self, auto_trigger_initial: bool = True):
        """Arm the detector for a new session, allowing immediate capture of page 1."""
        self.reset()
        self.auto_trigger_initial = auto_trigger_initial

    def notify_captured(self, frame: np.ndarray):
        """Register a frame as captured to prevent duplicate captures of the same page."""
        try:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            self._last_captured_thumb = cv2.resize(gray, (64, 48), interpolation=cv2.INTER_AREA)
        except Exception:
            self._last_captured_thumb = None

    def detect_hand(
        self,
        frame: np.ndarray,
        contour: t.Optional[np.ndarray] = None,
    ) -> tuple[bool, float]:
        """
        Detect whether hands or fingers are present on/near the document using YCrCb skin masking.
        :param frame: BGR camera frame
        :param contour: Optional 4-point document contour
        :return: (is_hand_present, skin_ratio)
        """
        if frame is None or frame.size == 0:
            return False, 0.0

        # Work on downscaled region for speed
        if contour is not None and len(contour) >= 4:
            # Crop to bounding box of the detected document
            x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
            ih, iw = frame.shape[:2]
            x = max(0, min(x, iw - 1))
            y = max(0, min(y, ih - 1))
            w = max(1, min(w, iw - x))
            h = max(1, min(h, ih - y))
            roi = frame[y : y + h, x : x + w]
        else:
            roi = frame

        small_roi = cv2.resize(roi, (120, 90), interpolation=cv2.INTER_AREA)
        ycrcb = cv2.cvtColor(small_roi, cv2.COLOR_BGR2YCrCb)

        # Standard YCrCb skin chrominance range
        skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        skin_pixels = np.count_nonzero(skin_mask)
        skin_ratio = float(skin_pixels) / skin_mask.size

        self.skin_ratio = skin_ratio
        self.hand_detected = skin_ratio >= self.skin_threshold
        return self.hand_detected, skin_ratio

    def _is_duplicate_page(self, gray_blur: np.ndarray) -> bool:
        """Check if the current frame is visually identical to the last captured page."""
        if self._last_captured_thumb is None:
            return False
        cur_thumb = cv2.resize(gray_blur, (64, 48), interpolation=cv2.INTER_AREA)
        diff = cv2.absdiff(cur_thumb, self._last_captured_thumb)
        diff_ratio = (float(np.count_nonzero(diff > 25)) / diff.size) * 100.0
        return diff_ratio < 6.0

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: t.Optional[float] = None,
        contour: t.Optional[np.ndarray] = None,
        has_document: bool = True,
    ) -> tuple[bool, float, str]:
        """
        Process incoming camera frame.
        :param frame: BGR image from camera
        :param timestamp: Optional explicit timestamp (for deterministic testing)
        :param contour: Optional detected document contour for hand localization
        :param has_document: True if document is currently in view
        :return: (should_capture, motion_score, current_state)
        """
        now = timestamp if timestamp is not None else time.time()

        # 1. Downscale and convert to grayscale for lightweight motion computation
        small = cv2.resize(frame, self.diff_resolution, interpolation=cv2.INTER_AREA)
        if len(small.shape) == 3:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        else:
            gray = small

        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._prev_gray is None:
            self._prev_gray = gray_blur
            return False, 0.0, self.state

        # Absolute consecutive frame differencing
        diff = cv2.absdiff(self._prev_gray, gray_blur)
        self._prev_gray = gray_blur

        # Threshold difference
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        motion_score = (float(np.count_nonzero(thresh)) / thresh.size) * 100.0

        # 2. Hand / holding detection
        self.detect_hand(frame, contour)

        # 3. Check cooldown
        if now - self._last_capture_time < self.cooldown_s:
            self.state = self.STATE_COOLDOWN
            return False, motion_score, self.state

        should_trigger = False

        # 4. State evaluation
        if self.hand_detected:
            # Hands/fingers are on or holding the book -> pause settle timer
            self.state = self.STATE_HAND_DETECTED
            self._settle_start = None
            return False, motion_score, self.state

        if motion_score >= self.motion_threshold:
            # Document or page is in active motion
            self.state = self.STATE_MOTION
            self._settle_start = None
            return False, motion_score, self.state

        # Motion is quiet and NO hands detected
        if self.state in (self.STATE_MOTION, self.STATE_HAND_DETECTED):
            # Motion or hands just ceased -> begin settle countdown
            self.state = self.STATE_SETTLING
            self._settle_start = now

        elif self.state in (self.STATE_IDLE, self.STATE_COOLDOWN):
            # If armed for initial page capture and no page captured yet:
            if self.auto_trigger_initial and self._last_captured_thumb is None and has_document:
                self.state = self.STATE_SETTLING
                self._settle_start = now
            else:
                self.state = self.STATE_IDLE
                self._settle_start = None

        elif self.state == self.STATE_SETTLING:
            elapsed = now - (self._settle_start or now)
            if elapsed >= self.settle_time_s:
                # Settle window finished! Check if page is duplicate of last capture
                if self._is_duplicate_page(gray_blur):
                    # Same page as previous capture; return to IDLE without snapping
                    self.state = self.STATE_IDLE
                    self._settle_start = None
                    return False, motion_score, self.state

                # Clean, sharp, new page settled without hands!
                should_trigger = True
                self._last_capture_time = now
                self.state = self.STATE_COOLDOWN
                self._settle_start = None
                self.auto_trigger_initial = False
                self.notify_captured(frame)

        return should_trigger, motion_score, self.state


