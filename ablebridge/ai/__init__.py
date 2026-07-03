"""
ablebridge.ai.intent — AI Intent Engine.

Routes input events through a local LLM (Ollama) to determine:
1. What the user is trying to communicate
2. What action to take
3. What response to give (for AAC)

The intent engine is the "brain" of AbleBridge.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any

import requests
from loguru import logger

from ablebridge.core.types import InputEvent, IntentEvent, IntentCategory


# ──────────────────────────────────────────────────────────────────────────────
# Base Intent Engine
# ──────────────────────────────────────────────────────────────────────────────


class IntentEngine(ABC):
    """
    Abstract base for intent engines.
    Implement this to add a new AI backend (OpenAI, Anthropic, local, etc.)
    """

    name: str = "base_intent"

    @abstractmethod
    def process(self, event: InputEvent) -> IntentEvent | None:
        """Process an input event and return a resolved intent."""
        ...

    @abstractmethod
    def chat(self, message: str, context: str = "") -> tuple[str, float]:
        """
        Simple chat interface. Returns (response_text, confidence).
        Used for AAC conversation mode.
        """
        ...

    def close(self) -> None:
        """Clean up resources."""
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Ollama Intent Engine (local LLM)
# ──────────────────────────────────────────────────────────────────────────────


SYSTEM_PROMPT = """You are the intent understanding module of AbleBridge,
an AI accessibility bridge for people with disabilities.

Your task is to classify user input and extract structured intent.
Always be helpful, patient, and supportive.

Respond ONLY with valid JSON in this format:
{
  "category": "COMMUNICATE|CONTROL_DEVICE|EMERGENCY|NAVIGATION|EMOTION|NEED|QUESTION|CONFIRM",
  "raw_text": "the user's message",
  "response": "a helpful response (for AAC conversation mode)",
  "entities": {"device": "...", "action": "...", "target": "..."},
  "urgency": 0.0-1.0,
  "sentiment": -1.0 to 1.0,
  "confidence": 0.0-1.0
}

Categories:
- COMMUNICATE: General conversation, greeting, statement
- CONTROL_DEVICE: Smart home control (lights, TV, thermostat)
- EMERGENCY: Help request, distress, urgent need
- NAVIGATION: Move wheelchair, get somewhere
- EMOTION: Express feelings, mood, pain level
- NEED: Basic needs (water, food, bathroom, medication)
- QUESTION: User is asking a question
- CONFIRM: Yes/No confirmation

If the input is unclear, respond with confidence below 0.5.
"""


class OllamaIntentEngine(IntentEngine):
    """
    Local LLM intent engine using Ollama.

    Advantages:
    - 100% offline (no data leaves the device)
    - Free (no API costs)
    - Fast (especially with quantization)
    - Private (perfect for medical/personal data)

    Setup:
        1. Install Ollama: https://ollama.ai
        2. Pull a model: ollama pull llama3.2
        3. Start server: ollama serve  (automatic on first use)
    """

    name = "ollama_intent"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:latest",
        threshold: float = 0.60,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._threshold = threshold
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                available = [m["name"] for m in models]
                if self._model in available or any(
                    self._model.split(":")[0] in m for m in available
                ):
                    logger.info(f"[OllamaIntent] Model '{self._model}' is available")
                    return True
                logger.warning(
                    f"[OllamaIntent] Model '{self._model}' not found. "
                    f"Available: {available}. Pull it with: ollama pull {self._model}"
                )
            return False
        except Exception as e:
            logger.warning(f"[OllamaIntent] Ollama not available: {e}")
            return False

    def process(self, event: InputEvent) -> IntentEvent | None:
        """Classify an input event and return structured intent."""
        if not self._available:
            return None

        text = self._extract_text(event)
        if not text:
            return None

        try:
            response_text, confidence = self.chat(text)
            parsed = self._parse_llm_response(response_text, text)
            parsed.sources = [event.channel]
            return parsed
        except Exception:
            logger.exception("[OllamaIntent] Error processing intent")
            return None

    def chat(self, message: str, context: str = "") -> tuple[str, float]:
        """Send a message to the LLM and get a response."""
        if not self._available:
            return "", 0.0

        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {message}"
        if context:
            full_prompt = f"Context: {context}\n\n{full_prompt}"

        payload = {
            "model": self._model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }

        try:
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=30,
            )
            if resp.status_code == 200:
                result = resp.json()
                return result.get("response", ""), result.get("confidence", 0.85)
            else:
                logger.warning(f"[OllamaIntent] HTTP {resp.status_code}: {resp.text[:100]}")
                return "", 0.0
        except Exception:
            return "", 0.0

    def _extract_text(self, event: InputEvent) -> str:
        """Extract text content from an input event."""
        if isinstance(event.raw_value, str):
            return event.raw_value
        if isinstance(event.raw_value, tuple) and len(event.raw_value) == 2:
            # Gaze position or similar
            return ""
        return str(event.raw_value or "")

    def _parse_llm_response(self, llm_text: str, original_text: str) -> IntentEvent:
        """Parse LLM JSON response into an IntentEvent."""
        # Try to extract JSON from the response
        try:
            # Find JSON block
            start = llm_text.find("{")
            end = llm_text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = llm_text[start:end]
                data = json.loads(json_str)

                category_str = data.get("category", "COMMUNICATE").upper()
                try:
                    category = IntentCategory[category_str]
                except KeyError:
                    category = IntentCategory.COMMUNICATE

                return IntentEvent(
                    category=category,
                    raw_text=original_text,
                    structured=data,
                    entities=data.get("entities", {}),
                    sentiment=data.get("sentiment", 0.0),
                    urgency=data.get("urgency", 0.0),
                    confidence=data.get("confidence", 0.5),
                    sources=[],
                )
        except Exception as e:
            logger.debug(f"[OllamaIntent] Could not parse JSON: {e}")

        # Fallback: return as general communication
        return IntentEvent(
            category=IntentCategory.COMMUNICATE,
            raw_text=original_text,
            structured={"response": llm_text or original_text},
            confidence=0.5,
        )

    def close(self) -> None:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Mock Intent Engine (for testing / no LLM)
# ──────────────────────────────────────────────────────────────────────────────


class MockIntentEngine(IntentEngine):
    """
    Mock intent engine for testing without Ollama.
    Provides basic keyword-based intent classification.
    """

    name = "mock_intent"

    EMERGENCY_WORDS = ["help", "emergency", "urgent", "pain", "call 911", "救我", "救命"]
    NEED_WORDS = ["water", "bathroom", "toilet", "food", "hungry", "thirsty", "medicine", "困", "饿", "渴"]
    DEVICE_WORDS = ["light", "tv", "music", "door", "lock", "temperature", "灯", "空调", "电视"]
    EMOTION_WORDS = ["happy", "sad", "tired", "angry", "confused", "feel", "mood", "疼", "累", "开心"]

    def process(self, event: InputEvent) -> IntentEvent | None:
        text = self._extract_text(event)
        if not text:
            return None

        category, confidence = self._classify_keyword(text)

        return IntentEvent(
            category=category,
            raw_text=text,
            structured={"response": f"I understood: {text}"},
            confidence=confidence,
            urgency=1.0 if category == IntentCategory.EMERGENCY else 0.3,
        )

    def chat(self, message: str, context: str = "") -> tuple[str, float]:
        """Mock chat: echo back with basic acknowledgement."""
        if not message.strip():
            return "", 0.0

        # Keyword-based responses
        lower = message.lower()
        if any(w in lower for w in self.EMERGENCY_WORDS):
            return "I understand you need help. I'm here for you. What do you need?", 0.85
        elif any(w in lower for w in self.NEED_WORDS):
            return "I understand you have a need. I'm checking how to help.", 0.80
        elif any(w in lower for w in self.DEVICE_WORDS):
            return "I can help you control the device. Which one would you like?", 0.75
        elif any(w in lower for w in self.EMOTION_WORDS):
            return "I hear how you're feeling. Thank you for sharing with me.", 0.80
        elif "?" in message or message.strip().endswith("?"):
            return "That's a great question. Let me help you think about that.", 0.70
        elif len(message.split()) <= 3:
            return f"Got it: '{message}'", 0.60

        return (
            f"I hear you saying: '{message}'. "
            "Tell me more and I'll do my best to help.",
            0.60,
        )

    def _extract_text(self, event: InputEvent) -> str:
        if isinstance(event.raw_value, str):
            return event.raw_value
        return ""

    def _classify_keyword(self, text: str) -> tuple[IntentCategory, float]:
        """Simple keyword-based classification."""
        lower = text.lower()
        if any(w in lower for w in self.EMERGENCY_WORDS):
            return IntentCategory.EMERGENCY, 0.90
        elif any(w in lower for w in self.NEED_WORDS):
            return IntentCategory.NEED, 0.85
        elif any(w in lower for w in self.DEVICE_WORDS):
            return IntentCategory.CONTROL_DEVICE, 0.80
        elif any(w in lower for w in self.EMOTION_WORDS):
            return IntentCategory.EMOTION, 0.75
        elif "?" in text:
            return IntentCategory.QUESTION, 0.70
        return IntentCategory.COMMUNICATE, 0.60
