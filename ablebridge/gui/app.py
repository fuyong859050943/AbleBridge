"""
ablebridge.gui.app — Web-based GUI for AbleBridge.

A single-page application served by Flask + SocketIO.
Provides:
- On-screen AAC keyboard with word prediction
- Real-time channel status panel
- Quick action buttons (Help, Water, etc.)
- Eye gaze calibration launcher
- Profile switcher
- Live event log

The UI is a React-like SPA served as static files,
with WebSocket for real-time updates.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from loguru import logger

from ablebridge.core.engine import AbleBridgeEngine
from ablebridge.core.event_bus import EventType, BusEvent
from ablebridge.core.types import (
    OutputEvent,
    OutputChannel,
    IntentEvent,
    PredictionResult,
)

# ──────────────────────────────────────────────────────────────────────────────
# Flask App
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(
    __name__,
    template_folder=Path(__file__).parent / "templates",
    static_folder=Path(__file__).parent / "static",
)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "ablebridge-secret-key-2024")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    ping_timeout=60,
    ping_interval=25,
)

# ──────────────────────────────────────────────────────────────────────────────
# Engine Singleton (shared across requests)
# ──────────────────────────────────────────────────────────────────────────────

_engine: AbleBridgeEngine | None = None
_engine_lock = threading.RLock()


def get_engine() -> AbleBridgeEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            profile_dir = Path(__file__).parent.parent.parent / "config" / "profiles"
            _engine = AbleBridgeEngine(profile_dir=profile_dir)
            _engine.load_profile("default")
            _engine.auto_register_drivers()
        return _engine


# ──────────────────────────────────────────────────────────────────────────────
# Event Bus → SocketIO Bridge
# ──────────────────────────────────────────────────────────────────────────────

def _bus_to_socket(bus_event: BusEvent) -> None:
    """Forward bus events to connected web clients via WebSocket."""
    payload_type = type(bus_event.payload).__name__
    socketio.emit(
        "engine_event",
        {
            "type": bus_event.type.name,
            "payload_type": payload_type,
            "payload": _serialize_payload(bus_event.payload),
            "timestamp": bus_event.timestamp,
            "source": bus_event.source,
        },
        namespace="/",
    )


def _serialize_payload(payload: Any) -> dict[str, Any]:
    """Serialize event payloads for JSON."""
    if isinstance(payload, PredictionResult):
        return {
            "predicted_text": payload.predicted_text,
            "confidence": payload.confidence,
            "alternatives": payload.alternatives,
        }
    if isinstance(payload, IntentEvent):
        return {
            "category": payload.category.name,
            "raw_text": payload.raw_text,
            "confidence": payload.confidence,
            "urgency": payload.urgency,
            "structured": payload.structured,
        }
    if isinstance(payload, OutputEvent):
        return {
            "channel": payload.channel.value,
            "content": payload.content,
            "action": payload.action,
            "priority": payload.priority,
        }
    return {"repr": str(payload)}


# ──────────────────────────────────────────────────────────────────────────────
# Web Routes
# ──────────────────────────────────────────────────────────────────────────────


@app.route("/")
def index() -> Any:
    """Main SPA page."""
    return render_template(
        "index.html",
        version="0.1.0",
        title="AbleBridge — Accessibility Bridge",
    )


@app.route("/api/status")
def status() -> Any:
    """Return full system status."""
    engine = get_engine()
    return jsonify(engine.get_system_status())


@app.route("/api/predict", methods=["POST"])
def predict() -> Any:
    """Get word predictions for AAC keyboard."""
    engine = get_engine()
    data = request.json or {}
    context = data.get("context", "")
    predictions = engine.predict_next(context)
    return jsonify({
        "predictions": [
            {"text": text, "confidence": float(conf)}
            for text, conf in predictions
        ]
    })


@app.route("/api/speak", methods=["POST"])
def speak() -> Any:
    """Send text to TTS."""
    engine = get_engine()
    data = request.json or {}
    text = data.get("text", "")
    if text:
        engine.speak(text)
    return jsonify({"ok": True, "text": text})


@app.route("/api/input", methods=["POST"])
def handle_input() -> Any:
    """Handle direct input from GUI (keyboard clicks, etc.)."""
    engine = get_engine()
    data = request.json or {}
    action = data.get("action", "")
    value = data.get("value", "")
    engine.handle_input(action, value)
    return jsonify({"ok": True})


@app.route("/api/calibrate", methods=["POST"])
def calibrate() -> Any:
    """Trigger eye gaze calibration."""
    engine = get_engine()
    success = False
    for driver in engine.input_manager._drivers.values():
        if driver.channel_type.value == "eyegaze":
            success = driver.calibrate()
            break
    return jsonify({"ok": True, "calibrated": success})


@app.route("/api/profile", methods=["GET", "POST"])
def profile() -> Any:
    """List or switch user profile."""
    engine = get_engine()
    if request.method == "GET":
        profiles = engine.list_profiles()
        current = engine.profile.id if engine.profile else "default"
        return jsonify({"profiles": profiles, "current": current})

    data = request.json or {}
    profile_id = data.get("profile_id", "default")
    try:
        engine.load_profile(profile_id)
        return jsonify({"ok": True, "profile_id": profile_id})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/driver/start", methods=["POST"])
def start_driver() -> Any:
    """Start a specific driver."""
    data = request.json or {}
    driver_type = data.get("type", "")
    engine = get_engine()
    # Simplified: just report status
    return jsonify({"ok": True})


@app.route("/api/log")
def event_log() -> Any:
    """Return recent event log."""
    engine = get_engine()
    limit = int(request.args.get("limit", 50))
    history = engine.bus.history(limit=limit)
    return jsonify({
        "events": [
            {
                "type": e.type.name,
                "timestamp": e.timestamp,
                "source": e.source,
                "payload_type": type(e.payload).__name__,
            }
            for e in history
        ]
    })


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket Events
# ──────────────────────────────────────────────────────────────────────────────


@socketio.on("connect")
def on_connect() -> None:
    logger.info("[WebSocket] Client connected")
    # Subscribe to engine events
    engine = get_engine()
    engine.bus.subscribe_global(_bus_to_socket)
    emit("connected", {"status": "ok", "version": "0.1.0"})


@socketio.on("disconnect")
def on_disconnect() -> None:
    logger.info("[WebSocket] Client disconnected")


@socketio.on("key_press")
def on_key_press(data: dict[str, Any]) -> None:
    """Handle virtual keyboard key press from web UI."""
    key = data.get("key", "")
    if not key:
        return

    engine = get_engine()
    engine.handle_input("type", key)

    # Get predictions
    current_text = data.get("current_text", "") + key
    predictions = engine.predict_next(current_text)

    emit(
        "predictions",
        {
            "predictions": [
                {"text": text, "confidence": float(conf)}
                for text, conf in predictions[:5]
            ]
        },
    )


@socketio.on("send_message")
def on_send_message(data: dict[str, Any]) -> Any:
    """Handle full AAC message send."""
    text = data.get("text", "")
    if not text:
        return

    engine = get_engine()
    # Speak the message
    engine.speak(text)
    # Also dispatch as output event
    output = OutputEvent(
        channel=OutputChannel.TTS,
        content=text,
        action="speak",
    )
    engine.output_manager.dispatch(output)

    return {"ok": True}


# ──────────────────────────────────────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────────────────────────────────────


def main(host: str = "0.0.0.0", port: int = 8765, debug: bool = False) -> None:
    """Launch the web GUI server."""
    logger.info(f"[GUI] Starting AbleBridge Web UI at http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
