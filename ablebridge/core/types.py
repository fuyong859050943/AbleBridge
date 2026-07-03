"""
ablebridge.core.types — Shared type definitions for the entire system.

All input drivers, output drivers, AI modules, and the core engine
share these types to ensure type-safe, consistent communication.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Generic, TypeVar

# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────


class InputChannel(Enum):
    """All supported input channel types."""

    KEYBOARD = "keyboard"
    EYEGAZE = "eyegaze"
    VOICE = "voice"
    SWITCH = "switch"
    HEADTRACK = "headtrack"
    KEYBOARD_SCAN = "keyboard_scan"
    BREATH = "breath"
    TOUCH = "touch"
    BCI = "bci"  # Brain-Computer Interface (OpenBCI, etc.)
    EMG = "emg"  # Electromyography (残存肌肉信号)
    CUSTOM = "custom"


class OutputChannel(Enum):
    """All supported output channel types."""

    TTS = "tts"  # Text-to-Speech
    SCREEN = "screen"
    SCREENREADER = "screenreader"
    VIBRATION = "vibration"
    SMART_HOME = "smart_home"
    SMS = "sms"
    MESSAGING = "messaging"
    WHEELCHAIR = "wheelchair"
    BRAILLE = "braille"
    NOTIFICATION = "notification"
    CUSTOM = "custom"


class IntentCategory(Enum):
    """High-level intent categories the AI engine can classify."""

    COMMUNICATE = auto()  # General communication / AAC
    CONTROL_DEVICE = auto()  # Control smart home, TV, lights
    EMERGENCY = auto()  # Call for help, emergency alert
    NAVIGATION = auto()  # Move wheelchair, navigate
    EMOTION = auto()  # Express emotion, pain level, mood
    NEED = auto()  # Basic needs: water, bathroom, food
    QUESTION = auto()  # Ask a question
    CONFIRM = auto()  # Yes/No confirmation
    CUSTOM = auto()


class DriverState(Enum):
    """Lifecycle state of a driver."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    CALIBRATING = "calibrating"


class ConfidenceLevel(Enum):
    """Human-readable confidence tier."""

    HIGH = "high"  # ≥ 0.85
    MEDIUM = "medium"  # ≥ 0.60
    LOW = "low"  # ≥ 0.40
    VERY_LOW = "very_low"  # < 0.40


# ──────────────────────────────────────────────────────────────────────────────
# Score Types
# ──────────────────────────────────────────────────────────────────────────────

T = TypeVar("T")


@dataclass(frozen=True)
class ConfidenceScore(Generic[T]):
    """
    A confidence-scored value from any input channel.
    Used throughout the system to represent graded signal quality.
    """

    value: T
    confidence: float = field(default=0.0)  # 0.0–1.0
    channel: InputChannel = InputChannel.CUSTOM
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def level(self) -> ConfidenceLevel:
        if self.confidence >= 0.85:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.60:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.40:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.VERY_LOW

    def is_usable(self, threshold: float = 0.50) -> bool:
        return self.confidence >= threshold


# ──────────────────────────────────────────────────────────────────────────────
# Event Types
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class InputEvent:
    """
    Unified representation of any user input event.
    All input drivers emit this type after normalization.
    """

    channel: InputChannel
    timestamp: float = field(default_factory=time.time)
    action: str = ""  # e.g. "dwell", "click", "type", "speak"
    raw_value: Any = None  # Raw driver-specific data
    parsed: ConfidenceScore[Any] | None = None
    sequence_id: int = 0  # Monotonic sequence number per channel
    session_id: str = ""  # For cross-channel correlation


@dataclass
class OutputEvent:
    """Unified representation of any output event dispatched by the engine."""

    channel: OutputChannel
    timestamp: float = field(default_factory=time.time)
    content: str = ""
    confidence: float = 1.0
    action: str = "display"  # e.g. "speak", "vibrate", "alert"
    metadata: dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher = more urgent (emergency=100)


@dataclass
class IntentEvent:
    """
    Result of the AI intent engine's analysis of one or more input events.
    """

    category: IntentCategory
    raw_text: str = ""
    structured: dict[str, Any] = field(default_factory=dict)
    # Parsed fields for common intents
    entities: dict[str, str] = field(default_factory=dict)
    sentiment: float = 0.0  # -1.0 to 1.0
    urgency: float = 0.0  # 0.0 to 1.0
    confidence: float = 0.0
    sources: list[InputChannel] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""


@dataclass
class PredictionResult:
    """
    A single prediction from the adaptive learning engine.
    """

    predicted_text: str = ""
    predicted_action: str = ""
    confidence: float = 0.0
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    model_id: str = ""
    timestamp: float = field(default_factory=time.time)


# ──────────────────────────────────────────────────────────────────────────────
# Channel Status
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ChannelStatus:
    """Runtime status snapshot of a single input or output channel."""

    name: str
    channel_type: str = ""  # "input" or "output"
    state: DriverState = DriverState.STOPPED
    is_enabled: bool = False
    confidence: float = 0.0
    latency_ms: float = 0.0
    error_message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# User Profile
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class InputConfig:
    """Configuration for a single input channel within a user profile."""

    enabled: bool = False
    dwell_time_ms: int = 500
    scan_rate_hz: float = 2.0
    sensitivity: float = 0.5  # 0.0–1.0
    custom_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputConfig:
    """Configuration for a single output channel within a user profile."""

    enabled: bool = False
    volume: float = 1.0  # 0.0–1.0
    rate: float = 1.0  # e.g. speech rate multiplier
    pitch: float = 1.0
    custom_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AIConfig:
    """Configuration for AI components."""

    provider: str = "ollama"  # "ollama" | "openai" | "mock"
    model: str = "llama3.2:latest"
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    local: bool = True
    temperature: float = 0.7
    max_tokens: int = 256
    prediction_enabled: bool = True
    adaptive_learning_enabled: bool = True
    intent_confidence_threshold: float = 0.60


@dataclass
class UserProfile:
    """
    Complete user profile containing all channel configs and AI settings.
    Loaded from YAML on startup, persisted on change.
    """

    id: str = "default"
    name: str = "Default User"
    description: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    inputs: dict[str, InputConfig] = field(default_factory=dict)
    outputs: dict[str, OutputConfig] = field(default_factory=dict)
    ai: AIConfig = field(default_factory=AIConfig)
    ui: dict[str, Any] = field(
        default_factory=lambda: {
            "theme": "light",
            "font_size": 18,
            "grid_layout": "aac_standard",
            "language": "en",
            "aac_preset": "standard",
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# Abstract Base Driver Interface
# ──────────────────────────────────────────────────────────────────────────────

Callback = Callable[[InputEvent | OutputEvent], None]


class BaseInputDriver(ABC):
    """
    Abstract base class for all input drivers.
    Implement this to add a new input channel.
    """

    name: str = "base_input"
    channel_type: InputChannel = InputChannel.CUSTOM

    def __init__(self, config: InputConfig, event_callback: Callback | None = None):
        self.config = config
        self._callback = event_callback
        self._state = DriverState.STOPPED
        self._seq = 0

    @property
    def state(self) -> DriverState:
        return self._state

    @abstractmethod
    def start(self) -> None:
        """Start the driver. Must be non-blocking or run in a thread."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop and clean up the driver."""
        ...

    @abstractmethod
    def calibrate(self) -> bool:
        """Run calibration routine. Returns True on success."""
        ...

    @abstractmethod
    def get_status(self) -> ChannelStatus:
        """Return current runtime status."""
        ...

    def _emit(self, action: str, raw_value: Any, confidence: float = 1.0) -> None:
        """Helper to emit a normalized InputEvent."""
        self._seq += 1
        event = InputEvent(
            channel=self.channel_type,
            action=action,
            raw_value=raw_value,
            parsed=ConfidenceScore(
                value=raw_value,
                confidence=confidence,
                channel=self.channel_type,
            ),
            sequence_id=self._seq,
        )
        if self._callback:
            self._callback(event)


class BaseOutputDriver(ABC):
    """
    Abstract base class for all output drivers.
    Implement this to add a new output channel.
    """

    name: str = "base_output"
    channel_type: OutputChannel = OutputChannel.CUSTOM

    def __init__(self, config: OutputConfig):
        self.config = config
        self._state = DriverState.STOPPED

    @property
    def state(self) -> DriverState:
        return self._state

    @abstractmethod
    def send(self, event: OutputEvent) -> bool:
        """Send an output event. Returns True on success."""
        ...

    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    @abstractmethod
    def get_status(self) -> ChannelStatus:
        ...
