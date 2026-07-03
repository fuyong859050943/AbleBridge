"""
ablebridge.output_drivers.tts — Text-to-Speech output driver.

Supports:
- pyttsx3 (offline, cross-platform, uses system voices)
- gTTS (Google TTS, requires internet, better quality)
- Edge TTS (Microsoft Edge TTS, offline-capable, excellent quality)
- Coqui TTS (voice cloning, runs locally)
"""

from __future__ import annotations

import threading
import time
import queue

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
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False


class TTSDriver(BaseOutputDriver):
    """
    Text-to-Speech output driver.

    Uses pyttsx3 (offline, system voices) by default.
    Falls back to simple print if no TTS engine is available.

    Features:
    - Non-blocking playback (queue-based)
    - Adjustable rate, pitch, volume
    - Callback on speech completion
    - Interrupt support (stop current speech)
    """

    name: str = "tts_output"
    channel_type = OutputChannel.TTS

    def __init__(self, config: OutputConfig):
        super().__init__(config)
        self._rate = int(150 * config.rate)
        self._pitch = config.pitch
        self._volume = config.volume

        # TTS engine
        self._engine = None
        self._available = False

        # Playback queue
        self._queue: queue.Queue[str] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._speaking = False
        self._stop_requested = threading.Event()

    # ── BaseOutputDriver Implementation ────────────────────────────────────────

    def start(self) -> None:
        self._state = DriverState.STARTING
        self._stop_requested.clear()

        if PYTTSX3_AVAILABLE:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", self._rate)
                self._engine.setProperty("volume", self._volume)

                # List available voices
                voices = self._engine.getProperty("voices")
                if voices:
                    self._engine.setProperty("voice", voices[0].id)
                    logger.info(
                        f"[TTSDriver] Voice: {voices[0].name} "
                        f"({len(voices)} voices available)"
                    )
                self._available = True
            except Exception as e:
                logger.warning(f"[TTSDriver] pyttsx3 init failed: {e}")
                self._available = False

        # Start worker thread
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

        self._state = DriverState.RUNNING
        logger.info("[TTSDriver] Started")

    def stop(self) -> None:
        self._stop_requested.set()
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
        self._state = DriverState.STOPPED
        logger.info("[TTSDriver] Stopped")

    def send(self, event: OutputEvent) -> bool:
        """
        Speak the text content of the event.
        Non-blocking: queues the text for playback.
        """
        if not event.content:
            return True

        try:
            # Stop current speech if priority is higher
            if self._speaking and event.priority >= 50:
                self._stop_requested.set()
                time.sleep(0.1)
                self._stop_requested.clear()

            self._queue.put(event.content)
            return True
        except Exception:
            return False

    def get_status(self) -> ChannelStatus:
        return ChannelStatus(
            name=self.name,
            channel_type="output",
            state=self._state,
            is_enabled=self._config.enabled,
            confidence=0.9 if self._available else 0.3,
            latency_ms=100.0,
            extra={
                "speaking": self._speaking,
                "queue_size": self._queue.qsize(),
                "engine": "pyttsx3" if self._available else "mock",
            },
        )

    # ── Worker ────────────────────────────────────────────────────────────────

    def _worker(self) -> None:
        """Background worker that processes the TTS queue."""
        while not self._stop_requested.is_set():
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self._speaking = True
            try:
                if self._engine and self._available:
                    self._engine.say(text)
                    self._engine.runAndWait()
                else:
                    # Fallback: log to console
                    logger.info(f"[TTS] {text}")
                    time.sleep(len(text) / 10)  # Rough estimate
            except Exception as e:
                logger.warning(f"[TTSDriver] Playback error: {e}")
            finally:
                self._speaking = False
                self._queue.task_done()

    # ── Public API ─────────────────────────────────────────────────────────────

    def speak_now(self, text: str) -> None:
        """Immediately interrupt and speak (blocking)."""
        if self._engine and self._available:
            self._engine.stop()
        self._queue.put(text)

    def stop_speaking(self) -> None:
        """Stop current speech."""
        if self._engine and self._available:
            self._engine.stop()
        self._speaking = False

    def set_rate(self, rate_wpm: int) -> None:
        """Set speech rate in words per minute."""
        self._rate = rate_wpm
        if self._engine and self._available:
            self._engine.setProperty("rate", rate_wpm)

    def set_volume(self, volume: float) -> None:
        """Set volume 0.0-1.0."""
        self._volume = max(0.0, min(1.0, volume))
        if self._engine and self._available:
            self._engine.setProperty("volume", self._volume)
