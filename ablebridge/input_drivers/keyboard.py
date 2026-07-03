"""
ablebridge.input_drivers.keyboard — Standard keyboard input driver.

Works on any platform using pynput. Handles:
- Direct key presses (for on-screen keyboard)
- Modifier key tracking (Shift, Ctrl)
- Dwell-time auto-repeat
- Configurable input channel binding

This is always the fallback input — every AbleBridge setup has a keyboard.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from loguru import logger

from ablebridge.core.types import (
    BaseInputDriver,
    ChannelStatus,
    DriverState,
    InputChannel,
    InputConfig,
    InputEvent,
)

try:
    from pynput import keyboard as pynput_keyboard

    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


class KeyboardDriver(BaseInputDriver):
    """
    Standard keyboard input driver using pynput.

    Emits:
    - "key_press" — single key pressed
    - "key_release" — key released
    - "text_input" — text character entered (preferred for AAC)
    """

    name: str = "keyboard_input"
    channel_type = InputChannel.KEYBOARD

    def __init__(self, config: InputConfig):
        super().__init__(config)
        self._listener: pynput_keyboard.Listener | None = None
        self._active_keys: set[str] = set()
        self._shift_held = False
        self._ctrl_held = False
        self._listener_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ── BaseInputDriver Implementation ────────────────────────────────────────

    def start(self) -> None:
        if not PYNPUT_AVAILABLE:
            raise RuntimeError("pynput not installed. Run: pip install pynput")

        self._stop_event.clear()
        self._state = DriverState.STARTING

        def on_press(key: Any) -> None:
            try:
                char = self._key_to_char(key)
                if char:
                    self._active_keys.add(char)
                    self._emit("key_press", char, confidence=1.0)

                    # Handle modifiers
                    if char == "Shift":
                        self._shift_held = True
                    elif char == "Ctrl":
                        self._ctrl_held = True
            except Exception:
                logger.exception("[KeyboardDriver] on_press error")

        def on_release(key: Any) -> None:
            try:
                char = self._key_to_char(key)
                if char:
                    self._active_keys.discard(char)
                    self._emit("key_release", char, confidence=1.0)

                    if char == "Shift":
                        self._shift_held = False
                    elif char == "Ctrl":
                        self._ctrl_held = False
            except Exception:
                logger.exception("[KeyboardDriver] on_release error")

        self._listener = pynput_keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
            suppress=False,
        )
        self._listener.daemon = True
        self._listener.start()

        self._state = DriverState.RUNNING
        logger.info("[KeyboardDriver] Started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._listener:
            self._listener.stop()
            self._listener = None
        self._state = DriverState.STOPPED
        logger.info("[KeyboardDriver] Stopped")

    def calibrate(self) -> bool:
        # Keyboard doesn't need calibration
        return True

    def get_status(self) -> ChannelStatus:
        return ChannelStatus(
            name=self.name,
            channel_type="input",
            state=self._state,
            is_enabled=self._config.enabled,
            confidence=1.0,
            latency_ms=0.0,
            extra={"active_keys": list(self._active_keys)},
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _key_to_char(self, key: Any) -> str:
        """Convert a pynput key object to a string representation."""
        try:
            # Handle special keys
            if isinstance(key, pynput_keyboard.Key):
                name = key.name.upper()
                if name == "SPACE":
                    return " "
                return name

            # Handle char keys
            if isinstance(key, pynput_keyboard.KeyCode):
                if key.char:
                    c = key.char
                    return c.upper() if self._shift_held else c.lower()
            return ""
        except Exception:
            return ""

    # ── Public API ─────────────────────────────────────────────────────────────

    def simulate_key(self, char: str) -> None:
        """
        Simulate a key press (for testing the GUI on-screen keyboard).
        """
        self._emit("text_input", char, confidence=1.0)
        logger.debug(f"[KeyboardDriver] Simulated key: '{char}'")

    def get_active_keys(self) -> set[str]:
        return set(self._active_keys)

    def is_modifier_held(self, modifier: str) -> bool:
        if modifier.lower() == "shift":
            return self._shift_held
        elif modifier.lower() == "ctrl":
            return self._ctrl_held
        return False
