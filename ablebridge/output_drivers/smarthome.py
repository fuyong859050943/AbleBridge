"""
ablebridge.output_drivers.smarthome — Smart home control output driver.

Integrates with Home Assistant (MQTT), OpenHAB, and generic MQTT devices.
Allows users to control lights, thermostats, door locks, TVs, and more
using their preferred input channel.

For wheelchair users: also supports power wheelchair control APIs
(Quantum, Permobil, Invacare) when available.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from ablebridge.core.types import (
    BaseOutputDriver,
    ChannelStatus,
    DriverState,
    OutputChannel,
    OutputConfig,
    OutputEvent,
)

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False


class DeviceType(Enum):
    LIGHT = "light"
    SWITCH = "switch"
    THERMOSTAT = "thermostat"
    LOCK = "lock"
    COVER = "cover"  # Blinds, curtains
    MEDIA = "media"
    WHEELCHAIR = "wheelchair"
    CUSTOM = "custom"


@dataclass
class SmartDevice:
    """A single smart home device."""

    id: str
    name: str
    type: DeviceType
    mqtt_topic: str  # e.g. "home/living_room/light/set"
    mqtt_payload_on: str = "ON"
    mqtt_payload_off: str = "OFF"
    supports_brightness: bool = False
    supports_color: bool = False
    supports_temperature: bool = False


class SmartHomeDriver(BaseOutputDriver):
    """
    Smart home control via MQTT / Home Assistant API.

    Allows control of:
    - Lights (on/off, brightness, color)
    - Thermostats (temperature setpoint)
    - Door locks (lock/unlock)
    - Media (play/pause, volume)
    - Wheelchair (if API available)

    All actions are triggered by AI intent resolution or direct UI.
    """

    name: str = "smarthome_output"
    channel_type = OutputChannel.SMART_HOME

    def __init__(
        self,
        config: OutputConfig,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        username: str = "",
        password: str = "",
    ):
        super().__init__(config)
        self._broker_host = broker_host
        self._broker_port = broker_port
        self._username = username
        self._password = password

        self._client: mqtt.Client | None = None
        self._connected = False
        self._devices: dict[str, SmartDevice] = {}
        self._lock = threading.Lock()

    # ── BaseOutputDriver Implementation ────────────────────────────────────────

    def start(self) -> None:
        self._state = DriverState.STARTING

        if MQTT_AVAILABLE:
            self._client = mqtt.Client(
                client_id=f"ablebridge_{int(time.time())}",
                protocol=mqtt.MQTTv311,
            )
            if self._username:
                self._client.username_pw_set(self._username, self._password)

            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_message = self._on_message

            try:
                self._client.connect(self._broker_host, self._broker_port, keepalive=60)
                self._client.loop_start()
                logger.info(f"[SmartHomeDriver] Connecting to {self._broker_host}:{self._broker_port}")
            except Exception as e:
                logger.warning(f"[SmartHomeDriver] Could not connect to MQTT broker: {e}")
        else:
            logger.warning("[SmartHomeDriver] paho-mqtt not installed. Run: pip install paho-mqtt")

        # Register default demo devices
        self._register_demo_devices()
        self._state = DriverState.RUNNING

    def stop(self) -> None:
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
        self._connected = False
        self._state = DriverState.STOPPED
        logger.info("[SmartHomeDriver] Stopped")

    def send(self, event: OutputEvent) -> bool:
        """Handle a smart home control event."""
        if event.action == "device_control":
            return self._handle_device_control(event)
        elif event.action == "toggle":
            return self._handle_toggle(event)
        elif event.action == "set_brightness":
            return self._handle_brightness(event)
        elif event.action == "set_temperature":
            return self._handle_temperature(event)
        return False

    def get_status(self) -> ChannelStatus:
        return ChannelStatus(
            name=self.name,
            channel_type="output",
            state=self._state,
            is_enabled=self._config.enabled,
            confidence=0.95 if self._connected else 0.0,
            extra={
                "connected": self._connected,
                "devices": [d.name for d in self._devices.values()],
                "broker": f"{self._broker_host}:{self._broker_port}",
            },
        )

    # ── MQTT Callbacks ────────────────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc: int) -> None:
        if rc == 0:
            self._connected = True
            logger.info("[SmartHomeDriver] MQTT connected")
            # Subscribe to device state topics
            for device in self._devices.values():
                if device.type == DeviceType.MEDIA:
                    client.subscribe(device.mqtt_topic.replace("/set", "/state"))
        else:
            logger.warning(f"[SmartHomeDriver] MQTT connection failed: rc={rc}")

    def _on_disconnect(self, client, userdata, rc: int) -> None:
        self._connected = False
        logger.warning(f"[SmartHomeDriver] MQTT disconnected: rc={rc}")

    def _on_message(self, client, userdata, msg) -> None:
        """Handle incoming MQTT messages (device state updates)."""
        try:
            topic = msg.topic
            payload = msg.payload.decode("utf-8")
            logger.debug(f"[SmartHomeDriver] {topic}: {payload}")
        except Exception:
            pass

    # ── Smart Home Actions ────────────────────────────────────────────────────

    def register_device(self, device: SmartDevice) -> None:
        """Register a smart device."""
        with self._lock:
            self._devices[device.id] = device
        logger.info(f"[SmartHomeDriver] Registered device: {device.name}")

    def _handle_device_control(self, event: OutputEvent) -> bool:
        device_id = event.metadata.get("device_id", "")
        action = event.metadata.get("action", "toggle")
        with self._lock:
            device = self._devices.get(device_id)

        if not device:
            # Try to find by name
            for d in self._devices.values():
                if d.name.lower() in (event.content or "").lower():
                    device = d
                    break

        if not device:
            logger.warning(f"[SmartHomeDriver] Device not found: {device_id}")
            return False

        return self._publish(device.mqtt_topic, event.metadata.get("payload", "TOGGLE"))

    def _handle_toggle(self, event: OutputEvent) -> bool:
        state = event.metadata.get("state", "TOGGLE")
        topic = event.metadata.get("topic", "")
        if not topic:
            return False
        return self._publish(topic, state)

    def _handle_brightness(self, event: OutputEvent) -> bool:
        brightness = event.metadata.get("brightness", 100)
        topic = event.metadata.get("topic", "")
        if not topic:
            return False
        return self._publish(topic, json.dumps({"brightness": brightness}))

    def _handle_temperature(self, event: OutputEvent) -> bool:
        temp = event.metadata.get("temperature", 22)
        topic = event.metadata.get("topic", "")
        if not topic:
            return False
        return self._publish(topic, json.dumps({"temperature": temp}))

    def _publish(self, topic: str, payload: str) -> bool:
        """Publish an MQTT message."""
        if not self._client or not self._connected:
            logger.warning(f"[SmartHomeDriver] Not connected, cannot publish to {topic}")
            return False
        try:
            result = self._client.publish(topic, payload, qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug(f"[SmartHomeDriver] Published to {topic}: {payload}")
                return True
            else:
                logger.warning(f"[SmartHomeDriver] Publish failed: rc={result.rc}")
                return False
        except Exception as e:
            logger.error(f"[SmartHomeDriver] Publish error: {e}")
            return False

    def _register_demo_devices(self) -> None:
        """Register a set of demo devices for testing."""
        self.register_device(SmartDevice(
            id="light_living",
            name="Living Room Light",
            type=DeviceType.LIGHT,
            mqtt_topic="home/living_room/light/set",
            supports_brightness=True,
        ))
        self.register_device(SmartDevice(
            id="light_bedroom",
            name="Bedroom Light",
            type=DeviceType.LIGHT,
            mqtt_topic="home/bedroom/light/set",
            supports_brightness=True,
        ))
        self.register_device(SmartDevice(
            id="lock_front",
            name="Front Door",
            type=DeviceType.LOCK,
            mqtt_topic="home/front_door/lock/set",
            mqtt_payload_on="LOCK",
            mqtt_payload_off="UNLOCK",
        ))
        self.register_device(SmartDevice(
            id="tv_living",
            name="TV",
            type=DeviceType.MEDIA,
            mqtt_topic="home/living_room/tv/command",
        ))

    # ── Public API ─────────────────────────────────────────────────────────────

    def list_devices(self) -> list[SmartDevice]:
        """List all registered devices."""
        with self._lock:
            return list(self._devices.values())

    def device(self, device_id: str) -> SmartDevice | None:
        """Get a device by ID."""
        with self._lock:
            return self._devices.get(device_id)

    def turn_on(self, device_id: str) -> bool:
        """Turn on a device."""
        with self._lock:
            device = self._devices.get(device_id)
        if device:
            return self._publish(device.mqtt_topic, device.mqtt_payload_on)
        return False

    def turn_off(self, device_id: str) -> bool:
        """Turn off a device."""
        with self._lock:
            device = self._devices.get(device_id)
        if device:
            return self._publish(device.mqtt_topic, device.mqtt_payload_off)
        return False
