"""
ablebridge.core.event_bus — Pub/sub event bus for loose coupling between modules.

The event bus is the nervous system of AbleBridge. All components communicate
through typed events, never directly. This enables hot-swapping drivers,
adding new AI models, and changing UI without touching other parts.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Generic, TypeVar

from loguru import logger

from ablebridge.core.types import (
    InputEvent,
    InputChannel,
    IntentEvent,
    IntentCategory,
    OutputEvent,
    OutputChannel,
    PredictionResult,
    ChannelStatus,
)


class EventType(Enum):
    """All event types that flow through the bus."""

    # Input events (from drivers → engine)
    INPUT = auto()
    INPUT_START = auto()
    INPUT_STOP = auto()
    INPUT_ERROR = auto()

    # Engine events (internal)
    INTENT_RESOLVED = auto()
    PREDICTION_UPDATED = auto()
    PROFILE_CHANGED = auto()

    # Output events (from engine → drivers)
    OUTPUT_DISPATCH = auto()
    OUTPUT_ERROR = auto()

    # System events
    CALIBRATION_START = auto()
    CALIBRATION_COMPLETE = auto()
    CALIBRATION_FAILED = auto()
    DRIVER_REGISTERED = auto()
    DRIVER_STATE_CHANGE = auto()
    ENGINE_START = auto()
    ENGINE_STOP = auto()
    HEARTBEAT = auto()


@dataclass
class BusEvent:
    """
    A typed envelope for all events on the bus.
    """

    type: EventType
    payload: Any = None
    timestamp: float = field(default_factory=time.time)
    source: str = ""  # Module name that emitted the event
    session_id: str = ""


# Type alias for event handlers
Handler = Callable[[BusEvent], None]
TPayload = TypeVar("TPayload")


class EventBus:
    """
    Thread-safe, async-capable publish/subscribe event bus.

    Features:
    - Typed subscriptions (subscribe to specific EventType, not all events)
    - Synchronous (thread) and asynchronous (asyncio) handler support
    - Dead-letter queue for unhandled events
    - Event history ring buffer for debugging
    - Handler metadata (priority, one-shot)

    Usage:
        bus = EventBus()
        bus.subscribe(EventType.INPUT, my_handler)
        bus.publish(BusEvent(EventType.INPUT, some_event))
    """

    def __init__(self, history_size: int = 500):
        self._subscribers: dict[EventType, list[tuple[int, Handler, str]]] = {}
        # priority (lower=higher priority), handler, name
        self._global_handlers: list[tuple[int, Handler, str]] = []
        self._history: list[BusEvent] = []
        self._history_size = history_size
        self._lock = threading.RLock()
        self._dead_letter_queue: list[BusEvent] = []
        self._stats = {"published": 0, "handled": 0, "dead_letter": 0}

    # ── Subscription ──────────────────────────────────────────────────────────

    def subscribe(
        self,
        event_type: EventType,
        handler: Handler,
        priority: int = 100,
        name: str = "",
    ) -> None:
        """
        Subscribe to a specific event type.

        Args:
            event_type: The event type to listen for
            handler: Callable that takes a BusEvent
            priority: Lower number = higher priority (runs first)
            name: Optional name for this subscription (for debugging)
        """
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append((priority, handler, name))
            self._subscribers[event_type].sort(key=lambda x: x[0])
        logger.debug(f"[EventBus] Subscribed '{name or handler}' → {event_type.name}")

    def subscribe_global(
        self, handler: Handler, priority: int = 100, name: str = ""
    ) -> None:
        """Subscribe to ALL events (use sparingly)."""
        with self._lock:
            self._global_handlers.append((priority, handler, name))
            self._global_handlers.sort(key=lambda x: x[0])

    def unsubscribe(self, event_type: EventType, handler: Handler) -> bool:
        """Remove a handler. Returns True if found and removed."""
        with self._lock:
            if event_type not in self._subscribers:
                return False
            before = len(self._subscribers[event_type])
            self._subscribers[event_type] = [
                (p, h, n) for p, h, n in self._subscribers[event_type] if h != handler
            ]
            removed = before - len(self._subscribers[event_type])
            return removed > 0

    # ── Publishing ────────────────────────────────────────────────────────────

    def publish(self, event: BusEvent) -> None:
        """
        Publish an event to all subscribers. Thread-safe.
        Unhandled events go to the dead-letter queue.
        """
        self._stats["published"] += 1

        # Record to history
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._history_size:
                self._history.pop(0)

        # Collect handlers to call (snapshot under lock)
        handlers_to_call: list[Handler] = []
        with self._lock:
            for _, handler, name in self._global_handlers:
                handlers_to_call.append(handler)
            for _, handler, name in self._subscribers.get(event.type, []):
                handlers_to_call.append(handler)

        # Call handlers outside the lock
        handled = False
        for handler in handlers_to_call:
            try:
                handler(event)
                self._stats["handled"] += 1
                handled = True
            except Exception:
                logger.exception(f"[EventBus] Handler error for {event.type.name}")

        if not handled:
            self._dead_letter_queue.append(event)
            self._stats["dead_letter"] += 1

    def publish_input(self, event: InputEvent, source: str = "") -> None:
        """Convenience: publish an InputEvent."""
        self.publish(BusEvent(EventType.INPUT, event, source=source))

    def publish_output(self, event: OutputEvent, source: str = "") -> None:
        """Convenience: publish an OutputEvent."""
        self.publish(BusEvent(EventType.OUTPUT_DISPATCH, event, source=source))

    # ── History & Debug ───────────────────────────────────────────────────────

    def history(self, event_type: EventType | None = None, limit: int = 50) -> list[BusEvent]:
        """Return recent event history."""
        with self._lock:
            events = self._history
            if event_type:
                events = [e for e in events if e.type == event_type]
            return list(events[-limit:])

    def dead_letters(self) -> list[BusEvent]:
        """Return all unhandled events."""
        return list(self._dead_letter_queue)

    def stats(self) -> dict[str, int]:
        """Return bus statistics."""
        return dict(self._stats)


# ──────────────────────────────────────────────────────────────────────────────
# Async Event Bus (for asyncio-based GUI)
# ──────────────────────────────────────────────────────────────────────────────


class AsyncEventBus(EventBus):
    """
    Async-aware event bus that also supports asyncio coroutines as handlers.
    Use this for the GUI / web layer; the core engine can use either.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._async_queue: asyncio.Queue[BusEvent | None] = asyncio.Queue()
        self._runner_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the async dispatch loop."""
        self._runner_task = asyncio.create_task(self._run_dispatcher())

    async def stop(self) -> None:
        """Stop the async dispatch loop."""
        if self._runner_task:
            self._runner_task.cancel()
            try:
                await self._runner_task
            except asyncio.CancelledError:
                pass

    async def _run_dispatcher(self) -> None:
        """Async loop that dispatches events from the queue."""
        while True:
            try:
                event = await self._async_queue.get()
                if event is None:  # Sentinel for shutdown
                    break
                self._dispatch_sync(event)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[AsyncEventBus] Dispatch error")

    async def publish_async(self, event: BusEvent) -> None:
        """Publish from an async context (non-blocking)."""
        await self._async_queue.put(event)

    def _dispatch_sync(self, event: BusEvent) -> None:
        """Synchronous dispatch for events pushed from async context."""
        self._stats["published"] += 1
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._history_size:
                self._history.pop(0)
        handlers_to_call: list[Handler] = []
        with self._lock:
            for _, handler, name in self._global_handlers:
                handlers_to_call.append(handler)
            for _, handler, name in self._subscribers.get(event.type, []):
                handlers_to_call.append(handler)
        for handler in handlers_to_call:
            try:
                # Try to await if it's a coroutine
                result = handler(event)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
                self._stats["handled"] += 1
            except Exception:
                logger.exception(f"[AsyncEventBus] Handler error for {event.type.name}")
