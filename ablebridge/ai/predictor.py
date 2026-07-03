"""
ablebridge.ai.predictor — Adaptive prediction engine.

Provides:
1. Text completion predictions (for AAC typing)
2. Phrase prediction based on context
3. Quick phrase shortcuts (e.g., "HW" → "How are you?")
4. Per-user adaptive learning (language model fine-tuned to user's patterns)

Uses n-gram language models + fuzzy matching for speed,
with optional LLM enhancement via Ollama.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger

from ablebridge.core.types import PredictionResult, UserProfile

# ──────────────────────────────────────────────────────────────────────────────
# Word-level N-gram Predictor (fast, no GPU required)
# ──────────────────────────────────────────────────────────────────────────────


class NGramPredictor:
    """
    Fast word-level N-gram language model for text prediction.

    Uses a weighted combination of unigram, bigram, and trigram probabilities
    for real-time predictions without GPU.

    Features:
    - Trainable on user's own text (adapts to vocabulary)
    - Quick phrase shortcuts
    - Case-aware predictions
    - Handles punctuation gracefully
    """

    # Common quick phrases (shortcut → expanded)
    DEFAULT_SHORTCUTS: dict[str, str] = {
        "brb": "be right back",
        "ty": "thank you",
        "np": "no problem",
        "tyt": "take your time",
        "tb": "thinking about you",
        "hw": "hello, how are you?",
        "gm": "good morning",
        "gn": "good night",
        "tyvm": "thank you very much",
        "pls": "please",
        "thx": "thanks",
        "bc": "because",
        "wb": "welcome back",
        "idk": "I don't know",
        "imo": "in my opinion",
        "afaik": "as far as I know",
        "btw": "by the way",
        "fyi": "for your information",
        "lol": "ha ha",
        "omg": "oh my goodness",
        # Emergency quick phrases
        "help": "I need help please",
        "water": "I would like some water please",
        "bathroom": "I need to use the bathroom",
        "pain": "I am in pain",
        "stop": "please stop",
        "yes": "yes",
        "no": "no",
    }

    # Most common English words (fallback predictions)
    COMMON_WORDS: list[str] = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
        "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    ]

    def __init__(self, order: int = 3):
        """
        Args:
            order: N-gram order (1=unigram, 2=bigram, 3=trigram). Higher = more accurate but more memory.
        """
        self._order = order
        self._ngrams: list[dict[tuple[str, ...], Counter]] = [Counter() for _ in range(order)]
        self._word_counts: Counter = Counter()
        self._vocab: set[str] = set()
        self._shortcuts: dict[str, str] = dict(self.DEFAULT_SHORTCUTS)
        self._sentence_starter_probs: Counter = Counter()
        self._total_words = 0
        self._user_data_path: Path | None = None

        # Fill common words into unigram model
        for word in self.COMMON_WORDS:
            self._word_counts[word] = 1

    # ── Training ───────────────────────────────────────────────────────────────

    def train_on_text(self, text: str) -> None:
        """Add text to the training corpus."""
        words = self._tokenize(text)
        if not words:
            return

        self._sentence_starter_probs[words[0]] += 1

        for n in range(1, self._order + 1):
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i : i + n])
                self._ngrams[n - 1][ngram] += 1

        for w in words:
            self._word_counts[w] += 1
            self._vocab.add(w)
            self._total_words += 1

    def train_from_file(self, path: Path | str) -> int:
        """
        Train from a text file (one sentence per line).
        Returns the number of lines trained on.
        """
        path = Path(path)
        if not path.exists():
            return 0

        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.train_on_text(line)
                    count += 1

        logger.info(f"[NGramPredictor] Trained on {count} lines from {path}")
        return count

    def save(self, path: Path | str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "order": self._order,
            "shortcuts": self._shortcuts,
            "word_counts": dict(self._word_counts),
            "sentence_starters": dict(self._sentence_starter_probs),
            "ngrams": [
                {str(k): dict(v) for k, v in ng.items()}
                for ng in self._ngrams
            ],
            "total_words": self._total_words,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        logger.info(f"[NGramPredictor] Model saved to {path}")

    def load(self, path: Path | str) -> bool:
        """Load model from disk."""
        path = Path(path)
        if not path.exists():
            return False

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            self._order = data["order"]
            self._shortcuts = data.get("shortcuts", dict(self.DEFAULT_SHORTCUTS))
            self._word_counts = Counter(data["word_counts"])
            self._sentence_starter_probs = Counter(data["sentence_starters"])
            self._total_words = data.get("total_words", 0)
            self._ngrams = [
                Counter({tuple(json.loads(k)): Counter(v) for k, v in ng.items()})
                for ng in data["ngrams"]
            ]
            self._vocab = set(self._word_counts.keys())
            logger.info(f"[NGramPredictor] Model loaded from {path}")
            return True
        except Exception:
            logger.exception(f"[NGramPredictor] Failed to load model from {path}")
            return False

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict(self, context: str, n: int = 5) -> list[tuple[str, float]]:
        """
        Predict the next word(s) given context.
        Returns list of (word, probability) tuples, sorted by confidence.
        """
        context = context.strip()

        # Check shortcuts first
        for shortcut, expansion in self._shortcuts.items():
            if shortcut.lower() == context.lower():
                return [(expansion, 0.99)] + [(w, 0.0) for w in range(n - 1)]

        words = self._tokenize(context)
        if not words:
            # Return most common words as suggestions
            return [(w, self._word_counts.get(w, 1) / max(self._total_words, 1))
                    for w in self.COMMON_WORDS[:n]]

        predictions: list[tuple[str, float]] = []

        # Trigram model (if we have 2+ words)
        if len(words) >= 2 and self._order >= 3:
            trigram = tuple(words[-2:])
            if trigram in self._ngrams[2]:
                following = self._ngrams[2][trigram]
                total = sum(following.values())
                for word, count in following.most_common(n):
                    prob = count / total if total > 0 else 0
                    predictions.append((word, prob))

        # Bigram model (if we have 1+ words)
        if len(predictions) < n and words:
            bigram = tuple(words[-1:]) if words else ()
            if bigram in self._ngrams[1]:
                following = self._ngrams[1][bigram]
                total = sum(following.values())
                for word, count in following.most_common(n * 2):
                    if not any(p[0] == word for p in predictions):
                        prob = count / total if total > 0 else 0
                        predictions.append((word, prob))

        # Unigram model (fill remaining)
        while len(predictions) < n:
            for word, count in self._word_counts.most_common(len(self._COMMON_WORDS) * 2):
                if not any(p[0] == word for p in predictions):
                    prob = math.log(count + 1) / math.log(max(self._total_words, 1) + 1)
                    predictions.append((word, prob))
                    break
            else:
                break

        # Normalize and sort
        predictions = [(w, max(0.0, min(1.0, p))) for w, p in predictions]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]

    def predict_phrase(self, prefix: str, n: int = 3) -> list[str]:
        """Predict a complete phrase (multiple words) given a prefix."""
        words = self._tokenize(prefix)
        if not words:
            return []

        # Find phrases that start with the given words
        results: list[tuple[str, float]] = []

        for n_idx in range(min(self._order, len(words) + 5)):
            for ngram, counter in self._ngrams[n_idx].items():
                if ngram[: len(words)] == tuple(words):
                    phrase = " ".join(ngram)
                    score = sum(counter.values()) / self._total_words
                    results.append((phrase, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [phrase for phrase, _ in results[:n]]

    def _tokenize(self, text: str) -> list[str]:
        """Split text into words, preserving punctuation markers."""
        tokens = re.findall(r"[\w']+|[.,!?;:]", text.lower())
        # Mark sentence-ending punctuation
        return tokens

    # ── Quick Phrases ─────────────────────────────────────────────────────────

    def add_shortcut(self, shortcut: str, expansion: str) -> None:
        """Add a user-defined quick phrase shortcut."""
        self._shortcuts[shortcut.lower()] = expansion

    def remove_shortcut(self, shortcut: str) -> bool:
        """Remove a shortcut. Returns True if it existed."""
        if shortcut.lower() in self._shortcuts:
            del self._shortcuts[shortcut.lower()]
            return True
        return False

    def list_shortcuts(self) -> dict[str, str]:
        """List all shortcuts."""
        return dict(self._shortcuts)


# ──────────────────────────────────────────────────────────────────────────────
# Adaptive Prediction Engine (full wrapper)
# ──────────────────────────────────────────────────────────────────────────────


class AdaptivePredictionEngine:
    """
    Top-level prediction engine that combines:
    1. N-gram model (fast, local)
    2. LLM enhancement via Ollama (smarter context-aware predictions)
    3. Per-user learning from interaction history

    This is what the GUI calls for AAC word suggestions.
    """

    name = "adaptive_prediction"

    def __init__(self, profile: UserProfile | None = None):
        self._profile = profile
        self._ngram = NGramPredictor(order=3)
        self._llm_available = False

        # Load pre-trained common phrases
        default_path = Path("config/phrases.txt")
        if default_path.exists():
            self._ngram.train_from_file(default_path)

        # Train on user's history if available
        if profile:
            history_path = Path(f"data/{profile.id}/phrases.txt")
            if history_path.exists():
                self._ngram.train_from_file(history_path)

        # Check Ollama availability
        if profile and profile.ai.local:
            try:
                import requests
                resp = requests.get(
                    f"{profile.ai.base_url}/api/tags",
                    timeout=3,
                )
                self._llm_available = resp.status_code == 200
            except Exception:
                self._llm_available = False

    def predict(self, context: str) -> PredictionResult:
        """
        Main prediction interface.
        Returns top prediction with alternatives.
        """
        if not context.strip():
            return PredictionResult(
                predicted_text=self._ngram.COMMON_WORDS[0],
                confidence=0.1,
                alternatives=[(w, 0.0) for w in self._ngram.COMMON_WORDS[1:5]],
                model_id="ngram",
            )

        # Fast N-gram prediction
        top_predictions = self._ngram.predict(context, n=5)
        top_word, top_conf = top_predictions[0] if top_predictions else ("", 0.0)

        # LLM enhancement (async, for richer predictions)
        llm_text = ""
        if self._llm_available and len(context) > 5:
            try:
                llm_text, _ = self._llm_predict(context)
            except Exception:
                pass

        result = PredictionResult(
            predicted_text=top_word,
            confidence=top_conf,
            alternatives=[
                (w, float(c)) for w, c in top_predictions[1:]
            ],
            model_id="ngram",
        )

        if llm_text and len(llm_text) > len(top_word):
            result.predicted_action = llm_text
            result.confidence = max(result.confidence, 0.7)

        return result

    def _llm_predict(self, context: str) -> tuple[str, float]:
        """Get prediction from LLM."""
        try:
            import requests
            payload = {
                "model": self._profile.ai.model if self._profile else "llama3.2",
                "prompt": f"Given the partial sentence: '{context}'\nPredict the most likely next word or phrase to complete it. Reply with ONLY the predicted text, nothing else.",
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 20},
            }
            resp = requests.post(
                f"{self._profile.ai.base_url}/api/generate",
                json=payload,
                timeout=10,
            )
            if resp.status_code == 200:
                text = resp.json().get("response", "").strip()
                return text, 0.75
        except Exception:
            pass
        return "", 0.0

    def learn(self, text: str) -> None:
        """Learn from user input to improve future predictions."""
        self._ngram.train_on_text(text)

        # Persist to user data
        if self._profile:
            data_dir = Path(f"data/{self._profile.id}")
            data_dir.mkdir(parents=True, exist_ok=True)
            history_path = data_dir / "phrases.txt"
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")

    def close(self) -> None:
        # Save model on close
        if self._profile:
            path = Path(f"data/{self._profile.id}/ngram_model.json")
            self._ngram.save(path)
