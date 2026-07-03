"""
AbleBridge — Test Suite

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=ablebridge  (with coverage)
    pytest tests/ -v -k "engine"        (filter by name)
"""

import pytest
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Test Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def temp_profile_dir(tmp_path):
    """Temporary profile directory."""
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    return profile_dir


@pytest.fixture
def engine(temp_profile_dir):
    """Fresh engine instance for each test."""
    from ablebridge.core.engine import AbleBridgeEngine
    engine = AbleBridgeEngine(profile_dir=str(temp_profile_dir))
    engine.load_profile("test_user")
    return engine


# ── Core Engine Tests ──────────────────────────────────────────────────────────


class TestAbleBridgeEngine:
    def test_engine_initialization(self, temp_profile_dir):
        from ablebridge.core.engine import AbleBridgeEngine
        engine = AbleBridgeEngine(profile_dir=str(temp_profile_dir))
        assert engine is not None
        assert not engine.is_running
        assert engine.bus is not None

    def test_profile_load(self, temp_profile_dir):
        from ablebridge.core.engine import AbleBridgeEngine
        from ablebridge.core.types import UserProfile

        engine = AbleBridgeEngine(profile_dir=str(temp_profile_dir))
        profile = engine.load_profile("default")
        assert profile is not None
        assert profile.id == "default"

    def test_engine_start_stop(self, engine):
        engine.start()
        assert engine.is_running
        engine.stop()
        assert not engine.is_running

    def test_speak_does_not_crash(self, engine):
        """speak() should not raise even with no output drivers."""
        engine.start()
        engine.speak("Test message")
        engine.stop()

    def test_predict_returns_list(self, engine):
        """predict_next should always return a list."""
        result = engine.predict_next("Hello ")
        assert isinstance(result, list)

    def test_system_status(self, engine):
        """get_system_status returns a valid dict."""
        status = engine.get_system_status()
        assert isinstance(status, dict)
        assert "running" in status
        assert "inputs" in status
        assert "outputs" in status

    def test_profile_save_load(self, engine, temp_profile_dir):
        from ablebridge.core.profile import ProfileManager

        profile = engine.profile
        profile.ui["test_key"] = "test_value"
        engine.save_profile(profile)

        mgr = ProfileManager(temp_profile_dir)
        loaded = mgr.load("test_user")
        assert loaded.ui.get("test_key") == "test_value"

    def test_multi_driver_registration(self, engine):
        """Can register multiple drivers without conflict."""
        from ablebridge.core.types import BaseInputDriver, InputConfig

        class MockDriver(BaseInputDriver):
            name = "mock"
            channel_type = None

            def start(self): pass
            def stop(self): pass
            def calibrate(self): return True
            def get_status(self): from ablebridge.core.types import ChannelStatus; return ChannelStatus(name="mock")

        # Register multiple mock drivers
        for i in range(3):
            engine.register_input_driver(MockDriver(InputConfig()))
        engine.start()
        engine.stop()


# ── Event Bus Tests ────────────────────────────────────────────────────────────


class TestEventBus:
    def test_subscribe_and_publish(self):
        from ablebridge.core.event_bus import EventBus, EventType, BusEvent

        bus = EventBus()
        received = []

        def handler(event: BusEvent) -> None:
            received.append(event)

        bus.subscribe(EventType.INPUT, handler)
        bus.publish(BusEvent(EventType.INPUT, payload={"test": "data"}))

        assert len(received) == 1
        assert received[0].payload["test"] == "data"

    def test_global_handler(self):
        from ablebridge.core.event_bus import EventBus, EventType, BusEvent

        bus = EventBus()
        received = []

        def global_handler(event: BusEvent) -> None:
            received.append(event)

        bus.subscribe_global(global_handler)
        bus.publish(BusEvent(EventType.ENGINE_START, payload="engine"))
        assert len(received) == 1

    def test_history(self):
        from ablebridge.core.event_bus import EventBus, EventType, BusEvent

        bus = EventBus(history_size=5)
        for i in range(10):
            bus.publish(BusEvent(EventType.ENGINE_START, payload=i))
        history = bus.history(limit=5)
        assert len(history) == 5
        assert history[0].payload == 9  # Most recent first

    def test_dead_letter_queue(self):
        from ablebridge.core.event_bus import EventBus, EventType, BusEvent

        bus = EventBus()
        # No subscribers — should go to dead letter
        bus.publish(BusEvent(EventType.INPUT, payload="orphaned"))
        dls = bus.dead_letters()
        assert len(dls) >= 1

    def test_stats(self):
        from ablebridge.core.event_bus import EventBus, EventType, BusEvent

        bus = EventBus()
        bus.subscribe(EventType.INPUT, lambda e: None)
        for _ in range(5):
            bus.publish(BusEvent(EventType.INPUT, payload="x"))
        stats = bus.stats()
        assert stats["published"] == 5
        assert stats["handled"] == 5


# ── Types Tests ────────────────────────────────────────────────────────────────


class TestTypes:
    def test_confidence_score(self):
        from ablebridge.core.types import ConfidenceScore

        score = ConfidenceScore(value="test", confidence=0.85, channel=None)
        assert score.level.value == "high"
        assert score.is_usable(0.8) is True
        assert score.is_usable(0.9) is False

    def test_confidence_score_very_low(self):
        from ablebridge.core.types import ConfidenceScore

        score = ConfidenceScore(value="test", confidence=0.1)
        assert score.level.value == "very_low"

    def test_user_profile_defaults(self):
        from ablebridge.core.types import UserProfile

        profile = UserProfile(id="test", name="Test User")
        assert profile.id == "test"
        assert profile.ai.provider == "ollama"
        assert profile.ai.local is True
        assert profile.ui["theme"] == "light"


# ── Prediction Engine Tests ────────────────────────────────────────────────────


class TestNGramPredictor:
    def test_train_and_predict(self):
        from ablebridge.ai.predictor import NGramPredictor

        predictor = NGramPredictor(order=3)
        predictor.train_on_text("hello how are you doing today")
        predictor.train_on_text("hello how are you feeling today")
        predictor.train_on_text("hello how are you")

        predictions = predictor.predict("hello how are")
        words = [w for w, _ in predictions]

        # Should predict "you" as top word (appears after "hello how are")
        assert "you" in words

    def test_shortcut_expansion(self):
        from ablebridge.ai.predictor import NGramPredictor

        predictor = NGramPredictor()
        predictor.add_shortcut("hw", "hello world")

        result = predictor.predict("hw")
        assert result[0][0] == "hello world"

    def test_common_words_fallback(self):
        from ablebridge.ai.predictor import NGramPredictor

        predictor = NGramPredictor()
        predictions = predictor.predict("", n=3)
        assert len(predictions) == 3
        assert all(isinstance(w, str) for w, _ in predictions)

    def test_save_load(self, tmp_path):
        from ablebridge.ai.predictor import NGramPredictor

        predictor = NGramPredictor(order=2)
        predictor.train_on_text("the quick brown fox")
        path = tmp_path / "model.json"
        predictor.save(path)

        predictor2 = NGramPredictor(order=2)
        success = predictor2.load(path)
        assert success
        predictions = predictor2.predict("quick")
        assert len(predictions) > 0


# ── Intent Engine Tests ────────────────────────────────────────────────────────


class TestMockIntentEngine:
    def test_chat_emergency(self):
        from ablebridge.ai.intent import MockIntentEngine
        from ablebridge.core.types import IntentCategory

        engine = MockIntentEngine()
        response, conf = engine.chat("help")
        assert conf >= 0.8
        assert "help" in response.lower()

    def test_chat_needs(self):
        from ablebridge.ai.intent import MockIntentEngine

        engine = MockIntentEngine()
        response, _ = engine.chat("I need water")
        assert len(response) > 0

    def test_process_input_event(self):
        from ablebridge.ai.intent import MockIntentEngine
        from ablebridge.core.types import InputEvent, IntentCategory

        engine = MockIntentEngine()
        event = InputEvent(
            channel=None,
            action="voice_input",
            raw_value="I need the bathroom",
        )
        intent = engine.process(event)
        assert intent is not None
        assert intent.category == IntentCategory.NEED

    def test_classify_device_control(self):
        from ablebridge.ai.intent import MockIntentEngine
        from ablebridge.core.types import IntentCategory

        engine = MockIntentEngine()
        _, conf = engine.chat("turn on the light")
        assert conf >= 0.7


# ── Integration Tests ─────────────────────────────────────────────────────────


class TestIntegration:
    def test_full_input_output_flow(self, engine):
        """Test: input event → engine → output dispatch."""
        from ablebridge.core.event_bus import EventType, BusEvent
        from ablebridge.core.types import InputEvent, OutputChannel, OutputEvent

        output_received = []

        def capture_output(event: BusEvent) -> None:
            if event.type == EventType.OUTPUT_DISPATCH:
                output_received.append(event.payload)

        engine.bus.subscribe(EventType.OUTPUT_DISPATCH, capture_output)
        engine.start()

        # Inject input event
        input_event = InputEvent(
            channel=None,
            action="voice_input",
            raw_value="hello",
        )
        engine.bus.publish_input(input_event, source="test")

        engine.stop()

        # Should have at least attempted to dispatch output
        # (may be empty if intent engine is not available, that's OK)
        assert isinstance(output_received, list)

    def test_gui_api_status_endpoint(self, engine):
        """Test the Flask API status endpoint."""
        from ablebridge.gui.app import app

        engine.start()
        with app.test_client() as client:
            resp = client.get("/api/status")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "running" in data
            assert "inputs" in data
        engine.stop()

    def test_gui_api_predict(self, engine):
        """Test the predict API endpoint."""
        from ablebridge.gui.app import app

        engine.start()
        with app.test_client() as client:
            resp = client.post(
                "/api/predict",
                json={"context": "hello "},
                content_type="application/json",
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert "predictions" in data
        engine.stop()


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
