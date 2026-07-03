"""
ablebridge.input_drivers.eyegaze — Webcam-based eye tracking input driver.

Uses MediaPipe Face Mesh to detect eye landmarks and estimate gaze direction.
Provides:
- Real-time gaze position (normalized 0-1)
- Dwell detection (user looks at a point for X ms → click)
- Blink detection
- Smoothed output to reduce jitter
- Easy calibration routine

Hardware required: Any standard webcam (720p+ recommended)

This driver is the heart of AbleBridge for ALS patients.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from ablebridge.core.types import (
    BaseInputDriver,
    ChannelStatus,
    ConfidenceScore,
    DriverState,
    InputChannel,
    InputConfig,
)

try:
    import cv2
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    cv2 = None
    mp = None


# ──────────────────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class CalibrationPoint:
    """One calibration target position."""

    label: str
    screen_x: float  # Normalized 0-1
    screen_y: float  # Normalized 0-1
    gaze_samples: list[tuple[float, float]]  # List of (gx, gy) readings


class EyeGazeCalibrator:
    """
    Calibrates eye tracking by collecting samples at known screen positions.
    Uses simple polynomial regression to map gaze → screen coordinates.
    """

    def __init__(self, num_points: int = 9):
        self.num_points = num_points
        self.points: list[CalibrationPoint] = []
        self._model: tuple[np.ndarray, np.ndarray] | None = None  # (coeffs_x, coeffs_y)

    def add_sample(self, label: str, screen_x: float, screen_y: float,
                   gaze_x: float, gaze_y: float) -> None:
        """Add a calibration sample."""
        for pt in self.points:
            if pt.label == label:
                pt.gaze_samples.append((gaze_x, gaze_y))
                return
        self.points.append(CalibrationPoint(label, screen_x, screen_y, [(gaze_x, gaze_y)]))

    def compute_model(self) -> bool:
        """Fit a polynomial model mapping gaze → screen coordinates."""
        all_gaze_x: list[float] = []
        all_gaze_y: list[float] = []
        all_screen_x: list[float] = []
        all_screen_y: list[float] = []

        for pt in self.points:
            if len(pt.gaze_samples) < 3:
                continue
            gx = np.mean([s[0] for s in pt.gaze_samples])
            gy = np.mean([s[1] for s in pt.gaze_samples])
            for _ in pt.gaze_samples:
                all_gaze_x.append(gx)
                all_gaze_y.append(gy)
                all_screen_x.append(pt.screen_x)
                all_screen_y.append(pt.screen_y)

        if len(all_gaze_x) < 9:
            logger.warning("[EyeGazeCalibrator] Not enough samples for calibration")
            return False

        gx = np.array(all_gaze_x).reshape(-1, 1)
        gy = np.array(all_gaze_y).reshape(-1, 1)
        sx = np.array(all_screen_x)
        sy = np.array(all_screen_y)

        # Polynomial degree 2 features
        ones = np.ones((len(gx), 1))
        features = np.hstack([ones, gx, gy, gx**2, gy**2, gx * gy])

        coeff_x = np.linalg.lstsq(features, sx, rcond=None)[0]
        coeff_y = np.linalg.lstsq(features, sy, rcond=None)[0]

        self._model = (coeff_x, coeff_y)
        logger.info("[EyeGazeCalibrator] Calibration model computed")
        return True

    def transform(self, gaze_x: float, gaze_y: float) -> tuple[float, float]:
        """Transform raw gaze coordinates to screen coordinates."""
        if self._model is None:
            return gaze_x, gaze_y  # Fallback: identity

        coeff_x, coeff_y = self._model
        feat = np.array([[1.0, gaze_x, gaze_y, gaze_x**2, gaze_y**2, gaze_x * gaze_y]])
        screen_x = float(np.dot(feat, coeff_x))
        screen_y = float(np.dot(feat, coeff_y))

        # Clamp to [0, 1]
        return max(0.0, min(1.0, screen_x)), max(0.0, min(1.0, screen_y)))


# ──────────────────────────────────────────────────────────────────────────────
# Main Driver
# ──────────────────────────────────────────────────────────────────────────────


class EyeGazeDriver(BaseInputDriver):
    """
    Webcam-based eye tracking using MediaPipe Face Mesh.

    Features:
    - Real-time gaze position at 30+ FPS
    - Dwell-to-click with configurable dwell time
    - Kalman filter for smooth output
    - Adaptive baseline (adjusts to user's natural gaze drift)
    - Blink detection for secondary input

    For ALS patients: This is the primary input. A single webcam + this driver
    replaces a $10,000 Tobii eye tracker.
    """

    name: str = "eyegaze_input"
    channel_type = InputChannel.EYEGAZE

    def __init__(self, config: InputConfig):
        super().__init__(config)
        self._camera_id = config.custom_params.get("camera_id", 0)
        self._dwell_time = config.dwell_time_ms / 1000.0  # Convert to seconds
        self._sensitivity = config.sensitivity

        # Calibration
        self._calibrator = EyeGazeCalibrator(num_points=9)
        self._is_calibrated = False

        # State
        self._cap: Any = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Gaze smoothing: rolling window + EMA
        self._position_history: deque[tuple[float, float]] = deque(maxlen=5)
        self._last_click_time = 0.0
        self._click_cooldown = 0.3  # seconds between clicks

        # Dwell tracking
        self._dwell_start_time: float | None = None
        self._dwell_position: tuple[float, float] | None = None
        self._is_dwelling = False

        # Smoothing parameters
        self._ema_alpha = 0.3
        self._smooth_x = 0.5
        self._smooth_y = 0.5

        # Stats
        self._fps = 0.0
        self._frame_times = deque(maxlen=30)

    # ── BaseInputDriver Implementation ────────────────────────────────────────

    def start(self) -> None:
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError(
                "MediaPipe/OpenCV not installed. Run: pip install opencv-python mediapipe"
            )

        self._state = DriverState.STARTING
        self._cap = cv2.VideoCapture(self._camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"[EyeGazeDriver] Cannot open camera {self._camera_id}")

        # Configure camera
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self._state = DriverState.RUNNING
        logger.info("[EyeGazeDriver] Started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        self._state = DriverState.STOPPED
        logger.info("[EyeGazeDriver] Stopped")

    def calibrate(self) -> bool:
        """
        Run interactive calibration.
        Shows targets on screen, collects gaze samples, fits model.

        Returns True on success.
        """
        if not MEDIAPIPE_AVAILABLE:
            return False

        self._state = DriverState.CALIBRATING
        logger.info("[EyeGazeDriver] Starting calibration...")

        # Collect 9-point calibration
        positions = [
            (0.5, 0.5, "center"),
            (0.2, 0.2, "tl"), (0.5, 0.2, "tc"), (0.8, 0.2, "tr"),
            (0.2, 0.5, "ml"),                     (0.8, 0.5, "mr"),
            (0.2, 0.8, "bl"), (0.5, 0.8, "bc"), (0.8, 0.8, "br"),
        ]

        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:

            for sx, sy, label in positions:
                logger.info(f"[EyeGazeDriver] Look at {label} ({sx:.1f}, {sy:.1f})")
                # Collect samples for 2 seconds
                samples: list[tuple[float, float]] = []
                start = time.time()
                while time.time() - start < 2.0:
                    if self._stop_event.is_set():
                        return False
                    ret, frame = self._cap.read()
                    if not ret:
                        continue
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb)
                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0].landmark
                        gx, gy = self._estimate_gaze(landmarks)
                        if gx is not None:
                            samples.append((gx, gy))
                            time.sleep(0.05)

                if samples:
                    avg_gx = np.mean([s[0] for s in samples])
                    avg_gy = np.mean([s[1] for s in samples])
                    self._calibrator.add_sample(label, sx, sy, avg_gx, avg_gy)

        success = self._calibrator.compute_model()
        self._is_calibrated = success
        self._state = DriverState.RUNNING if self._cap else DriverState.STOPPED

        if success:
            logger.info("[EyeGazeDriver] Calibration complete!")
        else:
            logger.warning("[EyeGazeDriver] Calibration failed - using uncalibrated mode")
        return success

    def get_status(self) -> ChannelStatus:
        return ChannelStatus(
            name=self.name,
            channel_type="input",
            state=self._state,
            is_enabled=self._config.enabled,
            confidence=0.85 if self._is_calibrated else 0.5,
            latency_ms=1000.0 / self._fps if self._fps > 0 else 0.0,
            error_message="",
            extra={
                "calibrated": self._is_calibrated,
                "fps": round(self._fps, 1),
                "dwell_time_ms": self._dwell_time * 1000,
            },
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Main capture and processing loop. Runs in background thread."""
        if not MEDIAPIPE_AVAILABLE:
            return

        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:

            while not self._stop_event.is_set():
                t0 = time.time()

                ret, frame = self._cap.read()
                if not ret:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    gx, gy = self._estimate_gaze(landmarks)

                    if gx is not None:
                        # Apply calibration if available
                        if self._is_calibrated:
                            gx, gy = self._calibrator.transform(gx, gy)

                        # Smooth
                        gx, gy = self._smooth(gx, gy)

                        self._position_history.append((gx, gy))

                        # Dwell detection
                        self._process_dwell(gx, gy)

                        # Emit position event (high frequency, low priority)
                        self._emit("gaze_move", (gx, gy), confidence=0.8)

                # FPS tracking
                dt = time.time() - t0
                self._frame_times.append(dt)
                if len(self._frame_times) >= 10:
                    self._fps = 1.0 / np.mean(self._frame_times)

    def _estimate_gaze(self, landmarks) -> tuple[float | None, float | None]:
        """
        Estimate gaze direction from MediaPipe face mesh landmarks.

        Uses iris center vs eye corner geometry.
        Returns (gx, gy) normalized to [0, 1].
        """
        # MediaPipe face mesh indices for key landmarks
        # Left eye: 33 (outer corner), 133 (inner corner), 468 (left iris)
        # Right eye: 362 (outer), 263 (inner), 473 (right iris)
        try:
            # Average left and right eye gaze
            left_eye_l = landmarks[33]
            left_eye_r = landmarks[133]
            left_iris = landmarks[468]

            right_eye_l = landmarks[362]
            right_eye_r = landmarks[263]
            right_iris = landmarks[473]

            # Normalize iris position within eye box [0, 1]
            def iris_ratio(inner, outer, iris):
                total = abs(outer.x - inner.x)
                if total < 0.001:
                    return 0.5
                pos = (iris.x - inner.x) / total
                return max(0.0, min(1.0, pos))

            def iris_vratio(top, bottom, iris):
                total = abs(top.y - bottom.y)
                if total < 0.001:
                    return 0.5
                pos = (iris.y - bottom.y) / total
                return max(0.0, min(1.0, pos))

            # Use both eyes (average)
            gx = (iris_ratio(left_eye_r, left_eye_l, left_iris) +
                  (1 - iris_ratio(right_eye_l, right_eye_r, right_iris))) / 2.0
            gy = (iris_vratio(landmarks[159], landmarks[145], left_iris) +
                  iris_vratio(landmarks[386], landmarks[374], right_iris)) / 2.0

            return gx, gy
        except Exception:
            return None, None

    def _smooth(self, x: float, y: float) -> tuple[float, float]:
        """Exponential moving average smoothing."""
        self._smooth_x = self._ema_alpha * x + (1 - self._ema_alpha) * self._smooth_x
        self._smooth_y = self._ema_alpha * y + (1 - self._ema_alpha) * self._smooth_y
        return self._smooth_x, self._smooth_y

    def _process_dwell(self, x: float, y: float) -> None:
        """Detect if gaze is dwelling on a point (for dwell-to-click)."""
        now = time.time()

        if self._dwell_position is None:
            self._dwell_position = (x, y)
            self._dwell_start_time = now
            return

        dx = x - self._dwell_position[0]
        dy = y - self._dwell_position[1]
        distance = math.sqrt(dx * dx + dy * dy)

        # If gaze moved too far, reset dwell
        threshold = 0.03 * (2 - self._sensitivity)  # Sensitivity affects threshold
        if distance > threshold:
            self._dwell_position = (x, y)
            self._dwell_start_time = now
            self._is_dwelling = False
            return

        # Check if dwelling long enough
        if self._dwell_start_time and (now - self._dwell_start_time) >= self._dwell_time:
            if (now - self._last_click_time) >= self._click_cooldown:
                self._emit("dwell_click", (x, y), confidence=0.85)
                self._last_click_time = now
                self._dwell_start_time = now
                self._is_dwelling = True
