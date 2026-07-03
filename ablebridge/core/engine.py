"""
ablebridge.core.engine — The central orchestrator of AbleBridge.

The Engine is the single entry point that coordinates:
1. Loading and managing input/output drivers
2. Routing input events through the AI intent engine
3. Dispatching output events to the right channels
4. Managing user profiles and configuration
5. Running the event bus as the central nervous system

Architecture:
    [Drivers] → [Event Bus] → [Input Manager] → [Intent Engine]
                                                    ↓
    [Output Manager] ← [Predictions] ← [Adaptive Learning]
          ↓
    [Drivers]
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from ablebridge.core.event_bus import EventBus, EventType, BusEvent
from ablebridge.core.profile import ProfileManager
from ablebridge.core.types import (
    AbleBridgeEngine,
    BaseInputDriver,
    BaseOutputDriver,
    ChannelStatus,
    DriverState,
    InputChannel,
    InputEvent,
    InputConfig,
    IntentEngine,
    IntentEvent,
    OutputChannel,
    OutputConfig,
    OutputEvent,
    PredictionEngine,
    UserProfile,
)


class InputManager:
    """
    Manages all registered input drivers and routes their events.
    Handles multi-channel fusion (combining signals from multiple inputs).
    """

    def __init__(self, bus: EventBus):
        self._drivers: dict[InputChannel, BaseInputDriver] = {}
        self._bus = bus
        self._lock = threading.RLock()
        self._sequence = 0

    def register(self, driver: BaseInputDriver) -> None:
        """Register an input driver."""
        with self._lock:
            if driver.channel_type in self._drivers:
                logger.warning(
                    f"[InputManager] Replacing existing driver for {driver.channel_type.value}"
                )
            driver._callback = self._on_input_event
            self._drivers[driver.channel_type] = driver
        logger.info(f"[InputManager] Registered input driver: {driver.name}")

    def unregister(self, channel: InputChannel) -> None:
        """Stop and remove a driver."""
        with self._lock:
            if channel in self._drivers:
                self._drivers[channel].stop()
                del self._drivers[channel]
                logger.info(f"[InputManager] Unregistered: {channel.value}")

    def start_all(self) -> None:
        """Start all registered drivers."""
        with self._lock:
            for driver in self._drivers.values():
                try:
                    driver.start()
                except Exception:
                    logger.exception(f"[InputManager] Failed to start {driver.name}")

    def stop_all(self) -> None:
        """Stop all drivers."""
        with self._lock:
            for driver in self._drivers.values():
                try:
                    driver.stop()
                except Exception:
                    logger.exception(f"[InputManager] Failed to stop {driver.name}")

    def get_status(self) -> dict[InputChannel, ChannelStatus]:
        """Get status of all input channels."""
        with self._lock:
            return {ch: d.get_status() for ch, d in self._drivers.items()}

    def _on_input_event(self, event: InputEvent) -> None:
        """Internal callback — emit enriched event to the bus."""
        self._sequence += 1
        event.sequence_id = self._sequence
        self._bus.publish_input(event, source=f"input_manager:{event.channel.value}")


class OutputManager:
    """
    Manages all registered output drivers and dispatches output events.
    Routes events to the appropriate channels based on event metadata.
    """

    def __init__(self, bus: EventBus):
        self._drivers: dict[OutputChannel, BaseOutputDriver] = {}
        self._bus = bus
        self._lock = threading.RLock()

    def register(self, driver: BaseOutputDriver) -> None:
        """Register an output driver."""
        with self._lock:
            if driver.channel_type in self._drivers:
                logger.warning(
                    f"[OutputManager] Replacing existing driver for {driver.channel_type.value}"
                )
            self._drivers[driver.channel_type] = driver
        logger.info(f"[OutputManager] Registered output driver: {driver.name}")

    def unregister(self, channel: OutputChannel) -> None:
        """Stop and remove a driver."""
        with self._lock:
            if channel in self._drivers:
                self._drivers[channel].stop()
                del self._drivers[channel]

    def start_all(self) -> None:
        """Start all output drivers."""
        with self._lock:
            for driver in self._drivers.values():
                try:
                    driver.start()
                except Exception:
                    logger.exception(f"[OutputManager] Failed to start {driver.name}")

    def stop_all(self) -> None:
        """Stop all drivers."""
        with self._lock:
            for driver in self._drivers.values():
                try:
                    driver.stop()
                except Exception:
                    logger.exception(f"[OutputManager] Failed to stop {driver.name}")

    def dispatch(self, event: OutputEvent) -> dict[OutputChannel, bool]:
        """
        Dispatch an output event to all applicable channels.
        Returns a dict of channel → success.
        """
        results: dict[OutputChannel, bool] = {}
        with self._lock:
            channels_to_notify = [
                (ch, d) for ch, d in self._drivers.items()
                if event.channel == ch or event.channel == OutputChannel.CUSTOM
            ]
            # If event is for a specific channel, only notify that one
            if event.channel != OutputChannel.CUSTOM:
                channels_to_notify = [
                    (ch, d)
                    for ch, d in self._drivers.items()
                    if ch == event.channel
                ]
                # Also broadcast to screen for all text events
                if (
                    event.channel != OutputChannel.SCREEN
                    and event.content
                    and len(event.content) > 0
                ):
                    if OutputChannel.SCREEN in self._drivers:
                        channels_to_notify.append(
                            (OutputChannel.SCREEN, self._drivers[OutputChannel.SCREEN])
                        )

            for channel, driver in channels_to_notify:
                try:
                    success = driver.send(event)
                    results[channel] = success
                    if not success:
                        logger.warning(
                            f"[OutputManager] Driver {driver.name} returned failure"
                        )
                except Exception:
                    logger.exception(f"[OutputManager] Error dispatching to {channel.value}")
                    results[channel] = False
        return results

    def get_status(self) -> dict[OutputChannel, ChannelStatus]:
        """Get status of all output channels."""
        with self._lock:
            return {ch: d.get_status() for ch, d in self._drivers.items()}


class AbleBridgeEngine:
    """
    The main AbleBridge engine.

    This is the top-level coordinator that:
    - Owns the event bus (central nervous system)
    - Manages input and output managers
    - Routes events through the AI pipeline
    - Exposes a clean API for GUI and CLI

    Usage:
        engine = AbleBridgeEngine(profile_path="config/profiles/me.yaml")
        engine.load_profile("my_profile")
        engine.start()
        # ... engine runs ...
        engine.stop()
    """

    def __init__(
        self,
        profile_dir: str | Path = "config/profiles",
        log_level: str = "INFO",
    ):
        # ── Core ──────────────────────────────────────────────────────────────
        self._bus = EventBus()
        self._profile_mgr = ProfileManager(Path(profile_dir))
        self._profile: UserProfile | None = None
        self._running = False
        self._lock = threading.RLock()

        # ── Managers ──────────────────────────────────────────────────────────
        self._input_mgr = InputManager(self._bus)
        self._output_mgr = OutputManager(self._bus)

        # ── AI Engines ────────────────────────────────────────────────────────
        self._intent_engine: IntentEngine | None = None
        self._prediction_engine: PredictionEngine | None = None

        # ── Session ───────────────────────────────────────────────────────────
        self._session_id = f"session_{int(time.time())}"
        self._input_seq = 0
        self._output_seq = 0

        # ── Bind internal event handlers ────────────────────────────────────
        self._bus.subscribe(EventType.INPUT, self._on_input, priority=50)
        self._bus.subscribe(EventType.INTENT_RESOLVED, self._on_intent, priority=50)
        self._bus.subscribe(EventType.CALIBRATION_START, self._on_calibration_start)

        logger.info("[Engine] AbleBridge engine initialized")

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def bus(self) -> EventBus:
        return self._bus

    @property
    def profile(self) -> UserProfile | None:
        return self._profile

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def input_manager(self) -> InputManager:
        return self._input_mgr

    @property
    def output_manager(self) -> OutputManager:
        return self._output_mgr

    @property
    def intent_engine(self) -> IntentEngine | None:
        return self._intent_engine

    @property
    def prediction_engine(self) -> PredictionEngine | None:
        return self._prediction_engine

    @property
    def session_id(self) -> str:
        return self._session_id

    # ── Profile Management ────────────────────────────────────────────────────

    def list_profiles(self) -> list[str]:
        """Return all available profile names."""
        return self._profile_mgr.list_profiles()

    def load_profile(self, profile_id: str) -> UserProfile:
        """
        Load a user profile by ID and configure the engine accordingly.
        This hot-swaps drivers, AI settings, and UI preferences.
        """
        profile = self._profile_mgr.load(profile_id)
        self._apply_profile(profile)
        return profile

    def save_profile(self, profile: UserProfile | None = None) -> None:
        """Save the current (or provided) profile to disk."""
        profile = profile or self._profile
        if profile:
            profile.updated_at = time.time()
            self._profile_mgr.save(profile)

    def _apply_profile(self, profile: UserProfile) -> None:
        """Apply a profile: configure drivers, AI, UI settings."""
        with self._lock:
            self._profile = profile

            # Stop existing drivers
            self._input_mgr.stop_all()
            self._output_mgr.stop_all()

            # Initialize AI engines based on profile
            if profile.ai.intent_confidence_threshold > 0:
                try:
                    from ablebridge.ai.intent import OllamaIntentEngine
                    self._intent_engine = OllamaIntentEngine(
                        base_url=profile.ai.base_url,
                        model=profile.ai.model,
                        threshold=profile.ai.intent_confidence_threshold,
                    )
                    logger.info(f"[Engine] Intent engine ready: {profile.ai.model}")
                except Exception:
                    logger.warning("[Engine] Could not initialize Ollama, using mock intent")
                    from ablebridge.ai.intent import MockIntentEngine
                    self._intent_engine = MockIntentEngine()

            if profile.ai.prediction_enabled:
                try:
                    from ablebridge.ai.predictor import AdaptivePredictionEngine
                    self._prediction_engine = AdaptivePredictionEngine(profile=profile)
                    logger.info("[Engine] Prediction engine ready")
                except Exception:
                    logger.warning("[Engine] Prediction engine unavailable")

            self._bus.publish(BusEvent(EventType.PROFILE_CHANGED, profile))

    # ── Driver Registration ────────────────────────────────────────────────────

    def register_input_driver(self, driver: BaseInputDriver) -> None:
        """Register an input driver."""
        self._input_mgr.register(driver)

    def register_output_driver(self, driver: BaseOutputDriver) -> None:
        """Register an output driver."""
        self._output_mgr.register(driver)

    def auto_register_drivers(self, profile: UserProfile | None = None) -> None:
        """
        Automatically discover and register all available drivers
        based on installed packages and profile settings.
        """
        profile = profile or self._profile
        if not profile:
            logger.warning("[Engine] No profile loaded, cannot auto-register")
            return

        # ── Keyboard Input (always available) ─────────────────────────────
        if profile.inputs.get("keyboard", InputConfig()).enabled:
            try:
                from ablebridge.input_drivers.keyboard import KeyboardDriver
                cfg = profile.inputs.get("keyboard", InputConfig())
                self.register_input_driver(KeyboardDriver(cfg))
            except Exception:
                logger.exception("[Engine] Could not register keyboard driver")

        # ── EyeGaze Input ───────────────────────────────────────────────────
        if profile.inputs.get("eyegaze", InputConfig()).enabled:
            try:
                from ablebridge.input_drivers.eyegaze import EyeGazeDriver
                cfg = profile.inputs.get("eyegaze", InputConfig())
                self.register_input_driver(EyeGazeDriver(cfg))
            except Exception:
                logger.exception("[Engine] Could not register eye gaze driver")

        # ── Voice Input ────────────────────────────────────────────────────
        if profile.inputs.get("voice", InputConfig()).enabled:
            try:
                from ablebridge.input_drivers.voice import VoiceDriver
                cfg = profile.inputs.get("voice", InputConfig())
                self.register_input_driver(VoiceDriver(cfg))
            except Exception:
                logger.exception("[Engine] Could not register voice driver")

        # ── Switch Input ────────────────────────────────────────────────────
        if profile.inputs.get("switch", InputConfig()).enabled:
            try:
                from ablebridge.input_drivers.switch_driver import SwitchDriver
                cfg = profile.inputs.get("switch", InputConfig())
                self.register_input_driver(SwitchDriver(cfg))
            except Exception:
                logger.exception("[Engine] Could not register switch driver")

        # ── TTS Output (always available) ─────────────────────────────────
        try:
            from ablebridge.output_drivers.tts import TTSDriver
            cfg = profile.outputs.get("tts", OutputConfig())
            self.register_output_driver(TTSDriver(cfg))
        except Exception:
            logger.exception("[Engine] Could not register TTS driver")

        # ── Smart Home Output ──────────────────────────────────────────────
        if profile.outputs.get("smarthome", OutputConfig()).enabled:
            try:
                from ablebridge.output_drivers.smarthome import SmartHomeDriver
                cfg = profile.outputs.get("smarthome", OutputConfig())
                self.register_output_driver(SmartHomeDriver(cfg))
            except Exception:
                logger.exception("[Engine] Could not register smart home driver")

        logger.info("[Engine] Auto-registration complete")

    # ── Engine Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the engine and all registered drivers."""
        with self._lock:
            if self._running:
                logger.warning("[Engine] Already running")
                return
            self._running = True
            self._session_id = f"session_{int(time.time())}"

            self._input_mgr.start_all()
            self._output_mgr.start_all()

            self._bus.publish(BusEvent(EventType.ENGINE_START, self._session_id))
            logger.info(f"[Engine] Started (session={self._session_id})")

    def stop(self) -> None:
        """Stop the engine and all drivers."""
        with self._lock:
            if not self._running:
                return
            self._running = False

            self._input_mgr.stop_all()
            self._output_mgr.stop_all()

            if self._intent_engine:
                self._intent_engine.close()
            if self._prediction_engine:
                self._prediction_engine.close()

            self._bus.publish(BusEvent(EventType.ENGINE_STOP, self._session_id))
            logger.info("[Engine] Stopped")

    def restart(self) -> None:
        """Restart the engine."""
        self.stop()
        time.sleep(0.5)
        self.start()

    # ── Direct API ─────────────────────────────────────────────────────────────

    def speak(self, text: str, priority: int = 0) -> None:
        """Direct TTS output (convenience method)."""
        event = OutputEvent(
            channel=OutputChannel.TTS,
            content=text,
            action="speak",
            priority=priority,
        )
        self._output_mgr.dispatch(event)

    def predict_next(self, context: str) -> list[tuple[str, float]]:
        """
        Get text predictions for the given context string.
        Used by the GUI's AAC keyboard.
        """
        if self._prediction_engine:
            result = self._prediction_engine.predict(context)
            return [(r.predicted_text, r.confidence) for r in [result] if r.confidence > 0]
        return []

    def handle_input(self, action: str, value: Any) -> IntentEvent | None:
        """
        Directly handle an input event (used by GUI / testing).
        Returns the resolved IntentEvent.
        """
        event = InputEvent(
            channel=InputChannel.KEYBOARD,
            action=action,
            raw_value=value,
            session_id=self._session_id,
        )
        self._bus.publish_input(event, source="gui")
        return None  # Intent is resolved asynchronously via the bus

    def get_system_status(self) -> dict[str, Any]:
        """Return full system status for the UI status panel."""
        return {
            "running": self._running,
            "session_id": self._session_id,
            "profile_id": self._profile.id if self._profile else None,
            "inputs": {
                ch.value: st.model_dump()
                for ch, st in self._input_mgr.get_status().items()
            },
            "outputs": {
                ch.value: st.model_dump()
                for ch, st in self._output_mgr.get_status().items()
            },
            "bus_stats": self._bus.stats(),
            "ai": {
                "intent": (
                    self._intent_engine.__class__.__name__
                    if self._intent_engine else "none"
                ),
                "prediction": (
                    self._prediction_engine.__class__.__name__
                    if self._prediction_engine else "none"
                ),
            },
        }

    # ── Internal Event Handlers ───────────────────────────────────────────────

    def _on_input(self, bus_event: BusEvent) -> None:
        """Handle incoming input events: route to AI pipeline."""
        if not isinstance(bus_event.payload, InputEvent):
            return
        input_event: InputEvent = bus_event.payload
        input_event.session_id = self._session_id

        # Route to intent engine
        if self._intent_engine:
            try:
                intent = self._intent_engine.process(input_event)
                if intent and intent.confidence >= (
                    self._profile.ai.intent_confidence_threshold if self._profile else 0.6
                ):
                    self._bus.publish(
                        BusEvent(
                            EventType.INTENT_RESOLVED,
                            intent,
                            source="intent_engine",
                        )
                    )
            except Exception:
                logger.exception("[Engine] Intent engine error")

        # Route to prediction engine
        if self._prediction_engine and input_event.action in ("type", "dwell"):
            try:
                result = self._prediction_engine.predict(
                    str(input_event.raw_value or "")
                )
                self._bus.publish(
                    BusEvent(EventType.PREDICTION_UPDATED, result, source="prediction")
                )
            except Exception:
                logger.exception("[Engine] Prediction engine error")

    def _on_intent(self, bus_event: BusEvent) -> None:
        """Handle resolved intents: dispatch to output channels."""
        if not isinstance(bus_event.payload, IntentEvent):
            return
        intent: IntentEvent = bus_event.payload

        # Dispatch to output
        output_event = OutputEvent(
            channel=OutputChannel.TTS,
            content=intent.raw_text or intent.structured.get("response", ""),
            action="speak",
            priority=100 if intent.urgency > 0.7 else intent.urgency * 10,
            metadata={"intent": intent.category.name, "intent_data": intent.structured},
        )
        self._output_mgr.dispatch(output_event)

    def _on_calibration_start(self, bus_event: BusEvent) -> None:
        """Handle calibration start event."""
        logger.info("[Engine] Calibration started")


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point: `ablebridge-core`"""
    import argparse

    parser = argparse.ArgumentParser(description="AbleBridge Core Engine")
    parser.add_argument("--profile", default="default", help="Profile ID to load")
    parser.add_argument("--profile-dir", default="config/profiles", help="Profile directory")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    args = parser.parse_args()

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=args.log_level)

    engine = AbleBridgeEngine(profile_dir=args.profile_dir, log_level=args.log_level)
    engine.load_profile(args.profile)
    engine.auto_register_drivers()
    engine.start()

    try:
        logger.info("[Engine] Running. Press Ctrl+C to stop.")
        while engine.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("[Engine] Interrupted")
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
