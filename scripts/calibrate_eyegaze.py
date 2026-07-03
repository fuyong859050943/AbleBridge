"""
calibrate_eyegaze.py — Standalone eye tracking calibration tool.

Run this before using EyeGazeDriver for the first time.
Usage: python scripts/calibrate_eyegaze.py

This opens a simple window with calibration targets.
Look at each target for 2 seconds. The system records your eye position
and builds a mapping model.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cv2
    import mediapipe as mp
    import numpy as np
except ImportError as e:
    print(f"Error: Missing dependencies. Run: pip install opencv-python mediapipe numpy")
    print(f"Details: {e}")
    sys.exit(1)


# ── Calibration Points ─────────────────────────────────────────────────────────
CALIBRATION_POINTS = [
    (0.5, 0.5, "Center"),
    (0.1, 0.1, "Top-Left"),
    (0.5, 0.1, "Top-Center"),
    (0.9, 0.1, "Top-Right"),
    (0.1, 0.5, "Middle-Left"),
    (0.9, 0.5, "Middle-Right"),
    (0.1, 0.9, "Bottom-Left"),
    (0.5, 0.9, "Bottom-Center"),
    (0.9, 0.9, "Bottom-Right"),
]

SAMPLE_DURATION = 2.0  # seconds per point
SAMPLE_INTERVAL = 0.05  # seconds between samples


def get_screen_resolution():
    """Get the primary screen resolution."""
    try:
        import tkinter as tk
        root = tk.Tk()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1920, 1080  # Default fallback


def collect_samples(camera_id: int = 0) -> list[dict]:
    """
    Collect eye tracking samples at calibration points.
    Returns a list of dicts: {label, screen_x, screen_y, gaze_x, gaze_y}
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        sys.exit(1)

    samples: list[dict] = []

    print("\n🎯 Eye Tracking Calibration")
    print("=" * 50)
    print(f"Look directly at each target. Stay still.")
    print(f"Press SPACE or ENTER to start.\n")
    print(f"Camera: {camera_id}")
    print(f"Points: {len(CALIBRATION_POINTS)}")
    print()

    input("Press Enter when ready...")

    for screen_x, screen_y, label in CALIBRATION_POINTS:
        print(f"\n▶  Look at: {label} ({screen_x:.0%}, {screen_y:.0%})")
        print(f"   Collecting samples for {SAMPLE_DURATION:.0f}s...", end="", flush=True)

        gaze_readings: list[tuple[float, float]] = []
        start_time = time.time()

        while time.time() - start_time < SAMPLE_DURATION:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # Estimate gaze
                try:
                    left_eye_l = landmarks[33]
                    left_eye_r = landmarks[133]
                    left_iris = landmarks[468]

                    right_eye_l = landmarks[362]
                    right_eye_r = landmarks[263]
                    right_iris = landmarks[473]

                    def iris_ratio(inner, outer, iris):
                        total = abs(outer.x - inner.x)
                        if total < 0.001:
                            return 0.5
                        return max(0.0, min(1.0, (iris.x - inner.x) / total))

                    gx = (iris_ratio(left_eye_r, left_eye_l, left_iris) +
                          (1 - iris_ratio(right_eye_l, right_eye_r, right_iris))) / 2.0
                    gy = (iris_ratio(landmarks[159], landmarks[145], left_iris) +
                          iris_ratio(landmarks[386], landmarks[374], right_iris)) / 2.0

                    gaze_readings.append((gx, gy))
                except Exception:
                    pass

            # Draw target indicator on preview
            preview_x = int(screen_x * frame.shape[1])
            preview_y = int(screen_y * frame.shape[0])
            cv2.circle(frame, (preview_x, preview_y), 20, (0, 255, 0), -1)
            cv2.putText(frame, label, (preview_x + 25, preview_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            cv2.imshow("Calibration Preview", frame)
            cv2.waitKey(1)

        print(f" ✓ Got {len(gaze_readings)} readings")

        if gaze_readings:
            avg_gx = np.mean([r[0] for r in gaze_readings])
            avg_gy = np.mean([r[1] for r in gaze_readings])
            samples.append({
                "label": label,
                "screen_x": screen_x,
                "screen_y": screen_y,
                "gaze_x": float(avg_gx),
                "gaze_y": float(avg_gy),
            })

    cap.release()
    cv2.destroyAllWindows()

    return samples


def fit_model(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a polynomial regression model mapping gaze → screen coordinates.
    Returns (coeff_x, coeff_y).
    """
    if len(samples) < 4:
        print("Error: Not enough calibration samples")
        return None, None

    gx = np.array([s["gaze_x"] for s in samples]).reshape(-1, 1)
    gy = np.array([s["gaze_y"] for s in samples]).reshape(-1, 1)
    sx = np.array([s["screen_x"] for s in samples])
    sy = np.array([s["screen_y"] for s in samples])

    # Polynomial features: [1, gx, gy, gx^2, gy^2, gx*gy]
    ones = np.ones((len(gx), 1))
    features = np.hstack([ones, gx, gy, gx**2, gy**2, gx * gy])

    coeff_x = np.linalg.lstsq(features, sx, rcond=None)[0]
    coeff_y = np.linalg.lstsq(features, sy, rcond=None)[0]

    return coeff_x, coeff_y


def save_model(coeff_x: np.ndarray, coeff_y: np.ndarray, path: Path | str) -> None:
    """Save the calibration model to a file."""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "coeff_x": coeff_x.tolist(),
        "coeff_y": coeff_y.tolist(),
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"\n✅ Calibration model saved to: {path}")


def main():
    print("=" * 50)
    print("AbleBridge — Eye Tracking Calibration")
    print("=" * 50)

    samples = collect_samples()
    print(f"\n📊 Collected {len(samples)} calibration samples")

    if len(samples) < 4:
        print("❌ Calibration failed. Please try again.")
        sys.exit(1)

    print("\n🔧 Fitting calibration model...")
    coeff_x, coeff_y = fit_model(samples)

    if coeff_x is None:
        sys.exit(1)

    # Save to default location
    model_path = Path("config/eyegaze_calibration.json")
    save_model(coeff_x, coeff_y, model_path)

    # Print sample accuracy
    print("\n📈 Calibration Accuracy (sample → estimated):")
    for s in samples:
        feat = np.array([[1.0, s["gaze_x"], s["gaze_y"],
                          s["gaze_x"]**2, s["gaze_y"]**2,
                          s["gaze_x"] * s["gaze_y"]]])
        est_x = float(np.dot(feat, coeff_x))
        est_y = float(np.dot(feat, coeff_y))
        err = abs(est_x - s["screen_x"]) + abs(est_y - s["screen_y"])
        status = "✓" if err < 0.1 else "~" if err < 0.2 else "✗"
        print(f"  {status} {s['label']:15s}  error: {err:.3f}")

    print("\n🎉 Calibration complete!")
    print("The EyeGazeDriver will now use this calibration automatically.")


if __name__ == "__main__":
    main()
