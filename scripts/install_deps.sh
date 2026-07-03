#!/bin/bash
# AbleBridge — Dependency Installation Script
# For Linux/macOS. For Windows, use PowerShell.

set -e

echo "🤝 AbleBridge — Installing Dependencies"
echo "========================================"

# Detect OS
OS="$(uname -s)"
echo "Detected OS: $OS"

# ── System Dependencies ─────────────────────────────────────────────────────────
if [ "$OS" = "Linux" ]; then
    if command -v apt-get &> /dev/null; then
        echo "[Linux] Installing system packages..."
        sudo apt-get update
        sudo apt-get install -y \
            python3-dev python3-pip \
            portaudio19-dev libasound2-dev \
            libusb-1.0-0-dev libudev-dev \
            ffmpeg 2>/dev/null || true
    elif command -v brew &> /dev/null; then
        echo "[macOS] Installing system packages..."
        brew install portaudio
    fi
fi

# ── Python Packages ────────────────────────────────────────────────────────────
echo "[Python] Installing pip packages..."
pip3 install --upgrade pip

# Core
pip3 install pydantic pydantic-settings loguru typer rich flask flask-socketio flask-cors jinja2 numpy sounddevice scipy

# Optional: uncomment to install all extras
# pip3 install -e ".[all]"

# Eye tracking (MediaPipe)
echo "[Optional] Installing MediaPipe for eye tracking..."
pip3 install opencv-python mediapipe 2>/dev/null || echo "[Skip] MediaPipe not available"

# Voice recognition (Vosk)
echo "[Optional] Installing Vosk for speech recognition..."
pip3 install vosk 2>/dev/null || echo "[Skip] Vosk not available"
echo "[Optional] Download a Vosk model with: python -m vosk download_model small-en-us"

# TTS
echo "[Optional] Installing TTS packages..."
pip3 install pyttsx3 2>/dev/null || echo "[Skip] pyttsx3 not available"

# Smart Home
echo "[Optional] Installing MQTT..."
pip3 install paho-mqtt 2>/dev/null || echo "[Skip] paho-mqtt not available"

# ── Ollama (for local LLM) ─────────────────────────────────────────────────────
echo ""
echo "🤖 AI Setup: Ollama (for local LLM intent engine)"
if command -v ollama &> /dev/null; then
    echo "Ollama is installed: $(ollama --version)"
    echo "Pulling default model (llama3.2)..."
    ollama pull llama3.2 2>/dev/null || echo "[Skip] Model pull failed (run manually: ollama pull llama3.2)"
else
    echo "[Info] Ollama not found. To install:"
    echo "  curl -fsSL https://ollama.ai/install.sh | sh"
    echo "Then: ollama pull llama3.2"
fi

# ── Vosk Model ────────────────────────────────────────────────────────────────
echo ""
echo "🎤 Speech Model: Vosk"
MODEL_DIR="$HOME/.cache/vosk"
if [ -d "$MODEL_DIR" ]; then
    echo "Vosk models found in $MODEL_DIR"
else
    echo "[Info] To download a Vosk model:"
    echo "  python -m vosk download_model small-en-us"
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "Quick Start:"
echo "  python -m ablebridge.gui          # Launch web GUI"
echo "  python -m ablebridge.core         # Run core engine"
echo "  ablebridge status                 # Check system status"
