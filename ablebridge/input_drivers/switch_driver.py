"""
ablebridge.input_drivers.switch_driver — Switch/scanner input driver.

Supports:
- USB/Bluetooth button switches (single or dual)
- Sip-and-puff controllers
- GPIO-connected buttons (Raspberry Pi)
- Keyboard-as-switch (for testing)

For scanning users: implements automatic scanning through on-screen boards,
with adjustable scan speed and item highlighting.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from loguru import logger

from ablebridge.core.types import (
    BaseInputDriver,
    ChannelStatus,
    DriverState,
    InputChannel,
    InputConfig,
)

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class SwitchDriver(BaseInputDriver):
    """
    Switch input driver supporting multiple switch types.

    Modes:
    1. Direct mode: Each switch maps directly to an action (e.g., left/right)
    2. Scanning mode: Auto-scan through options, switch press selects current
    3. Step mode: Switch press advances to next option, long-press selects

    Common hardware:
    - Jelly Bead / Buddy Button (~$30)
    - Sip-and-Puff ($200-500)
    - ATOM Switch (~$50)
    """

    name: str = "switch_input"
    channel_type = InputChannel.SWITCH

    def __init__(
        self,
        config: InputConfig,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 9600,
    ):
        super().__init__(config)
        self._port = port
        self._baudrate = baudrate
        self._scan_rate = config.scan_rate_hz  # Scans per second
        self._dwell_time = config.dwell_time_ms / 1000.0

        # State
        self._serial: serial.Serial | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Switch states
        self._switches: dict[str, bool] = {
            "switch_a": False,
            "switch_b": False,
        }
        self._last_press_time: dict[str, float] = {}
        self._debounce_time = 0.05  # 50ms debounce

        # Scanning
        self._scan_position = 0
        self._is_scanning = False
        self._scan_options: list[str] = []
        self._scan_lock = threading.Lock()

        # Stats
        self._press_count = 0

    # ── BaseInputDriver Implementation ────────────────────────────────────────

    def start(self) -> None:
        self._state = DriverState.STARTING
        self._stop_event.clear()

        # Try to open serial connection (for hardware switches)
        if SERIAL_AVAILABLE and not self._port.startswith("keyboard"):
            try:
                self._serial = serial.Serial(
                    self._port,
                    self._baudrate,
                    timeout=0.1,
                )
                logger.info(f"[SwitchDriver] Connected to {self._port}")
            except Exception as e:
                logger.warning(f"[SwitchDriver] Could not open {self._port}: {e}")
                logger.info("[SwitchDriver] Running in keyboard-simulation mode")
                self._serial = None

        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        self._state = DriverState.RUNNING
        logger.info("[SwitchDriver] Started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._serial:
            self._serial.close()
            self._serial = None
        self._state = DriverState.STOPPED
        logger.info("[SwitchDriver] Stopped")

    def calibrate(self) -> bool:
        """Calibration for switch: sensitivity / debounce tuning."""
        return True  # Switches are generally self-calibrating

    def get_status(self) -> ChannelStatus:
        return ChannelStatus(
            name=self.name,
            channel_type="input",
            state=self._state,
            is_enabled=self._config.enabled,
            confidence=1.0,
            latency_ms=10.0,
            extra={
                "switches": dict(self._switches),
                "scan_position": self._scan_position,
                "is_scanning": self._is_scanning,
                "press_count": self._press_count,
            },
        )

    # ── Scanning Interface ────────────────────────────────────────────────────

    def set_scan_options(self, options: list[str]) -> None:
        """Set the list of items to scan through."""
        with self._scan_lock:
            self._scan_options = list(options)
            self._scan_position = 0

    def start_scanning(self) -> None:
        """Start automatic scanning."""
        self._is_scanning = True
        self._scan_position = 0

    def stop_scanning(self) -> None:
        """Stop automatic scanning."""
        self._is_scanning = False

    def get_current_scan_item(self) -> str | None:
        """Get the currently highlighted scan item."""
        with self._scan_lock:
            if self._scan_options and 0 <= self._scan_position < len(self._scan_options):
                return self._scan_options[self._scan_position]
        return None

    # ── Internal ─────────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """Polling loop for switch states. Runs in background thread."""
        scan_interval = 1.0 / max(self._scan_rate, 0.5)
        last_scan_time = 0.0

        while not self._stop_event.is_set():
            now = time.time()

            # Check hardware switch
            switch_pressed = self._read_hardware_switch()

            # Check for debounce
            for sw, state in switch_pressed.items():
                if state and (now - self._last_press_time.get(sw, 0)) > self._debounce_time:
                    self._handle_switch_press(sw)
                    self._last_press_time[sw] = now

            # Handle scanning
            if self._is_scanning:
                if now - last_scan_time >= scan_interval:
                    last_scan_time = now
                    self._advance_scan()

            time.sleep(0.01)

    def _read_hardware_switch(self) -> dict[str, bool]:
        """Read current switch states from hardware."""
        result = dict(self._switches)

        if self._serial and self._serial.is_open:
            try:
                line = self._serial.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    # Format: "SW1:1,SW2:0" or "PRESS" (single switch)
                    if line == "PRESS" or line == "1":
                        result["switch_a"] = True
                    elif "SW1" in line:
                        parts = line.split(",")
                        for p in parts:
                            if ":" in p:
                                k, v = p.split(":")
                                if k in result:
                                    result[k] = v.strip() == "1"
            except Exception:
                pass

        return result

    def _handle_switch_press(self, switch: str) -> None:
        """Handle a switch press event."""
        self._press_count += 1
        self._switches[switch] = True

        if self._is_scanning:
            # In scanning mode: press selects current item
            item = self.get_current_scan_item()
            if item:
                self._emit("scan_select", item, confidence=1.0)
        else:
            # Direct mode: each switch has a direct action
            self._emit("switch_press", switch, confidence=1.0)

    def _advance_scan(self) -> None:
        """Advance to the next scan item."""
        with self._scan_lock:
            if not self._scan_options:
                return
            self._scan_position = (self._scan_position + 1) % len(self._scan_options)
            item = self._scan_options[self._scan_position]

        self._emit("scan_highlight", item, confidence=1.0)
