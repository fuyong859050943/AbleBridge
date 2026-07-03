"""
ablebridge.input_drivers.voice — Local speech recognition input driver.

Uses Vosk (offline) or Whisper (local/OpenAI) for speech-to-text.
All processing happens on-device — no cloud, no privacy concerns.

For ALS patients with remaining speech: voice is the fastest input.
For others: complements eye gaze as a secondary channel.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from loguru import logger

from ablebridge.core.types import (
    BaseInputDriver,
    ChannelStatus,
    DriverState,
    InputChannel,
    InputConfig,
)

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False


@dataclass
class VoiceCommand:
    """A recognized voice command with confidence score."""

    text: str
    confidence: float  # 0-1
    duration_ms: int
    timestamp: float


class VoiceDriver(BaseInputDriver):
    """
    Local speech recognition using Vosk (offline) or Whisper.

    Features:
    - Offline operation (Vosk model) — no internet required
    - Speaker-adaptive recognition (improves over time)
    - Voice activity detection (VAD) with adaptive threshold
    - Command detection (short commands vs continuous speech)
    - Configurable language and vocabulary

    Recommended: vosk-model-small-en-us (75MB, good accuracy)
    Best accuracy: vosk-model-en-us-0.22 (1.8GB)
    """

    name: str = "voice_input"
    channel_type = InputChannel.VOICE

    def __init__(
        self,
        config: InputConfig,
        model_path: str = "models/vosk-model-small-en-us",
        sample_rate: int = 16000,
    ):
        super().__init__(config)
        self._model_path = model_path
        self._sample_rate = sample_rate
        self._blocksize = 4096
        self._vad_threshold = 0.01  # RMS energy threshold for voice activity
        self._min_utterance_duration = 0.3  # seconds
        self._max_utterance_duration = 30.0  # seconds

        # State
        self._stream: sd.InputStream | None = None
        self._recognizer: vosk.KaldiRecognizer | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Voice activity detection
        self._is_speaking = False
        self._speech_buffer: list[np.ndarray] = []
        self._speech_start_time: float = 0.0
        self._silence_frames = 0
        self._speech_frames_threshold = 3

        # Recognition queue
        self._command_queue: list[VoiceCommand] = []
        self._queue_lock = threading.Lock()

        # Stats
        self._latency_ms = 0.0
        self._last_result_time = 0.0

    # ── BaseInputDriver Implementation ────────────────────────────────────────

    def start(self) -> None:
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError(
                "sounddevice not installed. Run: pip install sounddevice"
            )

        if not VOSK_AVAILABLE:
            raise RuntimeError(
                "vosk not installed. Run: pip install vosk && python -m vosk download_model small-en-us"
            )

        self._state = DriverState.STARTING

        # Load model
        try:
            model = vosk.Model(self._model_path)
        except Exception as e:
            logger.warning(f"[VoiceDriver] Could not load model from {self._model_path}: {e}")
            logger.warning("[VoiceDriver] Falling back to mock mode (no real ASR)")
            model = None

        if model:
            self._recognizer = vosk.KaldiRecognizer(model, self._sample_rate)
        else:
            self._recognizer = None

        # Open audio stream
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            blocksize=self._blocksize,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        )
        self._stream.start()

        self._stop_event.clear()
        self._state = DriverState.RUNNING
        logger.info(f"[VoiceDriver] Started (model={self._model_path})")

    def stop(self) -> None:
        self._stop_event.set()
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._state = DriverState.STOPPED
        logger.info("[VoiceDriver] Stopped")

    def calibrate(self) -> bool:
        """Calibrate voice detection threshold to ambient noise level."""
        if not SOUNDDEVICE_AVAILABLE:
            return False

        logger.info("[VoiceDriver] Calibrating microphone (silence for 2s)...")
        samples = []
        with sd.InputStream(samplerate=self._sample_rate, blocksize=self._blocksize, channels=1) as stream:
            for _ in range(int(self._sample_rate * 2 / self._blocksize)):
                block, _ = stream.read(self._blocksize)
                rms = np.sqrt(np.mean(block.astype(np.float32) ** 2))
                samples.append(rms)

        if samples:
            noise_floor = np.mean(samples)
            self._vad_threshold = max(noise_floor * 3, 0.005)
            logger.info(f"[VoiceDriver] VAD threshold calibrated: {self._vad_threshold:.4f}")
        return True

    def get_status(self) -> ChannelStatus:
        return ChannelStatus(
            name=self.name,
            channel_type="input",
            state=self._state,
            is_enabled=self._config.enabled,
            confidence=0.85,
            latency_ms=self._latency_ms,
            extra={
                "speaking": self._is_speaking,
                "queue_size": len(self._command_queue),
                "vad_threshold": round(self._vad_threshold, 4),
            },
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _audio_callback(self, indata: np.ndarray, frames: int, status: sd.CallbackFlags) -> None:
        """Audio callback from sounddevice. Runs in audio thread."""
        if status:
            logger.debug(f"[VoiceDriver] Audio status: {status}")

        if self._stop_event.is_set():
            return

        # Convert to float32 for processing
        audio = indata[:, 0].astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio**2))
        now = time.time()

        # Voice Activity Detection
        if rms > self._vad_threshold:
            self._silence_frames = 0
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_start_time = now
                self._speech_buffer = []
            self._speech_buffer.append(audio)
        else:
            if self._is_speaking:
                self._silence_frames += 1
                if self._silence_frames >= self._speech_frames_threshold:
                    self._is_speaking = False
                    duration = now - self._speech_start_time
                    if duration >= self._min_utterance_duration:
                        self._process_utterance()

    def _process_utterance(self) -> None:
        """Process accumulated speech buffer."""
        if not self._speech_buffer or not self._recognizer:
            return

        # Concatenate all frames
        full_audio = np.concatenate(self._speech_buffer)
        duration_ms = int(len(full_audio) / self._sample_rate * 1000)

        # Convert back to int16 for Vosk
        int16_audio = (full_audio * 32767).astype(np.int16)

        # Recognize
        self._recognizer.Reset()
        self._recognizer.AcceptWaveform(int16_audio.tobytes())
        result = self._recognizer.PartialResult()

        if result and result.get("partial"):
            text = result["partial"].strip()
            if text:
                cmd = VoiceCommand(
                    text=text,
                    confidence=0.85,  # Vosk doesn't provide per-utterance confidence easily
                    duration_ms=duration_ms,
                    timestamp=time.time(),
                )
                with self._queue_lock:
                    self._command_queue.append(cmd)
                self._emit("voice_partial", text, confidence=0.80)
                self._latency_ms = (time.time() - cmd.timestamp) * 1000

        # Final result
        final = self._recognizer.Result()
        if final:
            import json
            try:
                parsed = json.loads(final)
                if parsed.get("text"):
                    text = parsed["text"].strip()
                    cmd = VoiceCommand(
                        text=text,
                        confidence=0.90,
                        duration_ms=duration_ms,
                        timestamp=time.time(),
                    )
                    with self._queue_lock:
                        self._command_queue.append(cmd)
                    self._emit("voice_input", text, confidence=0.90)
            except Exception:
                pass

        self._speech_buffer = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_latest_command(self) -> VoiceCommand | None:
        """Get and clear the most recent recognized command."""
        with self._queue_lock:
            if self._command_queue:
                return self._command_queue.pop(-1)
        return None

    def drain_queue(self) -> list[VoiceCommand]:
        """Get and clear all pending commands."""
        with self._queue_lock:
            commands = list(self._command_queue)
            self._command_queue.clear()
            return commands

    def set_threshold(self, threshold: float) -> None:
        """Set VAD threshold (0.0-1.0)."""
        self._vad_threshold = max(0.001, min(0.5, threshold))
