"""
AbleBridge — A universal AI-powered accessibility bridge.

Architecture:
    User → [Input Drivers] → [Core Engine] → [AI Intent] → [Output Drivers] → Device/Action

Mission:
    Build bridges, not walls. Every input channel deserves every output channel.
"""

__version__ = "0.1.0"
__author__ = "AbleBridge Community"
__license__ = "MIT"

from ablebridge.core.types import (
    InputEvent,
    OutputEvent,
    IntentEvent,
    PredictionResult,
    UserProfile,
    ChannelStatus,
    ConfidenceScore,
)
from ablebridge.core.engine import AbleBridgeEngine

__all__ = [
    "__version__",
    "AbleBridgeEngine",
    "InputEvent",
    "OutputEvent",
    "IntentEvent",
    "PredictionResult",
    "UserProfile",
    "ChannelStatus",
    "ConfidenceScore",
]
