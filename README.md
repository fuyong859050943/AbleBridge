# AbleBridge 🤝

**A universal, AI-powered accessibility bridge that connects every input and output channel — so no one gets left behind.**

> "Disability is not a problem of the person, but a mismatch between the person and their environment." — That's why we build bridges.

---

## 🎯 What is AbleBridge?

AbleBridge is a **local-first, AI-powered accessibility middleware** that unifies heterogeneous input/output channels for people with disabilities. Instead of siloed apps that solve one problem, AbleBridge provides a **plugin-based architecture** where any input can talk to any output, with AI bridging the gap.

**Who is it for?**
- 🧠 **ALS / Motor Neuron Disease** — Eye gaze, residual micro-movements, breath control
- 🦽 **Spinal Cord Injury / Cerebral Palsy** — Switch scanning, sip-and-puff, head tracking
- 👁️ **Visual Impairment** — Screen reader integration, voice command, haptic feedback
- 👂 **Deaf / Hard of Hearing** — Real-time captioning, visual alerts, vibration patterns
- 🗣️ **Aphasia / Speech Disability** — AAC with personalized voice synthesis
- 🧩 **Cognitive / Learning Disability** — Simplified UI, prediction, visual cues
- 🧠 **Autism / Sensory Processing** — Social cue augmentation, predictable interaction

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      USER INTERFACE LAYER                      │
│   ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│   │On-Screen │  │  AAC Board    │  │  Smart Home Control   │  │
│   │ Keyboard │  │  (Grids/Text) │  │  (Visual Dashboard)   │  │
│   └────┬─────┘  └──────┬───────┘  └──────────┬─────────────┘  │
│        │               │                      │               │
│        └───────────────┼──────────────────────┘               │
│                        ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              ABLEBRIDGE CORE ENGINE                       │  │
│  │                                                           │  │
│  │  ┌──────────────┐     ┌────────────────────────────┐    │  │
│  │  │ Intent Engine │────▶│ Adaptive Learning Engine  │    │  │
│  │  │  (Local LLM)  │     │   (Per-user preference)    │    │  │
│  │  └───────┬───────┘     └────────────────────────────┘    │  │
│  │          │                                                │  │
│  │  ┌───────┴───────────────────────────────────────┐       │  │
│  │  │           CHANNEL ORCHESTRATOR                │       │  │
│  │  │  Multi-channel fusion + confidence scoring   │       │  │
│  │  └───────┬────────────────┬──────────────────────┘       │  │
│  │          │                │                               │  │
│  │  ┌───────▼───────┐ ┌──────▼──────┐                       │  │
│  │  │ INPUT MANAGER │ │ OUTPUT MGR  │                       │  │
│  │  └───────┬───────┘ └──────┬──────┘                       │  │
│  └──────────┼────────────────┼───────────────────────────────┘  │
│             │                │                                 │
│  ┌──────────▼───────┐ ┌──────▼──────┐  ┌─────────────────┐   │
│  │ INPUT DRIVERS     │ │OUTPUT DRIVERS│  │ INTEGRATIONS    │   │
│  │                  │ │             │  │                 │   │
│  │ 👁️ EyeGazeDriver │ │ TTS Driver  │  │ HomeAssistant   │   │
│  │ 🎤 VoiceDriver   │ │ Screen Reader│  │ Wheelchair API  │   │
│  │ 🔘 SwitchDriver  │ │ Vibrate Driver│ │ OpenAI/LocalLLM │   │
│  │ ⌨️ KeyboardDriver│ │ Visual Driver│  │ WhatsApp/Discord│   │
│  │ 🖐️ HeadTrackDriver│ │ Alert Driver │  │ OpenBCI (BCI)   │   │
│  │ 💨 BreathDriver  │ │ Haptic Driver│  │ MQTT/SmartHome  │   │
│  └──────────────────┘ └─────────────┘  └─────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🧠 AI-Powered Intelligence
- **Intent Prediction** — Local LLM (Ollama) predicts what the user wants before they finish typing/speaking
- **Multi-channel Fusion** — Combines eye gaze + voice + switch inputs simultaneously, AI picks highest-confidence signal
- **Adaptive Learning** — Learns each user's vocabulary, behavior patterns, and preferred interaction style over time
- **Context-Aware Suggestions** — Understands conversation context for smarter AAC predictions

### 🔌 Universal Input Support
| Driver | Technology | Latency | Cost |
|--------|-----------|---------|------|
| EyeGaze | MediaPipe + webcam | <50ms | ~$0 |
| Voice | Vosk / Whisper (local) | <200ms | $0 |
| Switch | GPIO / USB / Bluetooth | <10ms | ~$15 |
| HeadTrack | webcam + ML | <50ms | $0 |
| Breath | Pressure sensor | <20ms | ~$30 |
| Keyboard | Standard + scanning | <10ms | $0 |

### 🔊 Rich Output Modes
- **Personalized TTS** — Clone user's voice from 30 seconds of audio
- **AAC Boards** — Dynamic grid boards with semantic prediction
- **Visual / Haptic** — Color-coded feedback, vibration patterns
- **Smart Home** — Direct control of lights, TV, wheelchair, door locks

### 🏠 100% Local & Private
- All AI inference runs **on-device** (Ollama + llama.cpp)
- No cloud dependencies for core functionality
- User data never leaves the device
- GDPR-compliant by design

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- 8GB+ RAM (16GB recommended for local LLM)
- Webcam (for eye tracking / head tracking)
- Optional: USB switch, Arduino, MQTT broker

### Install

```bash
git clone https://github.com/YOUR_USERNAME/AbleBridge.git
cd AbleBridge

# Install dependencies
pip install -e .
# or on macOS/Linux:
pip3 install -e .

# Run the GUI
python -m ablebridge.gui
```

### First-Time Setup

```bash
# Launch setup wizard
python -m ablebridge.setup

# Configure your inputs
python -m ablebridge.config --add-input eyegaze
python -m ablebridge.config --add-input voice

# Run with a specific profile
python -m ablebridge.core --profile myprofile
```

### Web UI (Default)

Open browser: `http://localhost:8765`

```
┌──────────────────────────────────────────────────────┐
│  AbleBridge GUI                          [⚙️] [📊]  │
├──────────────────────────────────────────────────────┤
│                                                      │
│   ┌─────────────────────────────────────────────┐   │
│   │         ON-SCREEN KEYBOARD                   │   │
│   │  Q W E R T Y U I O P                        │   │
│   │   A S D F G H J K L                         │   │
│   │  ⇧  Z X C V B N M  ⌫                       │   │
│   └─────────────────────────────────────────────┘   │
│                                                      │
│   ┌──────────────────┐  ┌────────────────────────┐  │
│   │  AAC PREDICTION  │  │   CHANNEL STATUS       │  │
│   │  > Hello, how    │  │   👁️ Eye: 94%  🟢     │  │
│   │    are you? [🔊] │  │   🎤 Voice: 78% 🟢     │  │
│   │                  │  │   🔘 Switch: --        │  │
│   └──────────────────┘  └────────────────────────┘  │
│                                                      │
│   ┌─────────────────────────────────────────────┐   │
│   │  QUICK ACTIONS                               │   │
│   │  [Help Me] [Call] [Water] [Bathroom] [Yes] [No] │
│   └─────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
ablebridge/
├── ablebridge/              # Main Python package
│   ├── __init__.py
│   ├── core/                # Core engine
│   │   ├── engine.py        # Main orchestrator
│   │   ├── event_bus.py     # Pub/sub event system
│   │   ├── profile.py       # User profile management
│   │   └── types.py         # Shared type definitions
│   ├── input_drivers/       # Input channel drivers
│   │   ├── base.py          # Abstract base driver
│   │   ├── eyegaze.py       # MediaPipe webcam eye tracking
│   │   ├── voice.py         # Local ASR (Vosk/Whisper)
│   │   ├── switch.py        # USB/GPIO switch input
│   │   ├── keyboard.py      # Standard keyboard
│   │   └── headtrack.py     # Head movement tracking
│   ├── output_drivers/      # Output channel drivers
│   │   ├── base.py          # Abstract base driver
│   │   ├── tts.py           # Text-to-speech
│   │   ├── voicelink.py     # Voice cloning + TTS
│   │   ├── screenreader.py  # Screen reader output
│   │   └── smarthome.py     # Home automation
│   ├── ai/                  # AI modules
│   │   ├── intent.py        # Intent understanding
│   │   ├── predictor.py     # Text/action prediction
│   │   └── adaptive.py      # Per-user learning
│   ├── gui/                 # Web-based GUI
│   │   ├── app.py           # Flask app
│   │   ├── routes.py        # API routes
│   │   └── static/          # Frontend assets
│   ├── integrations/        # Third-party integrations
│   │   ├── homeassistant.py
│   │   ├── ollama.py        # Local LLM
│   │   └── whatsapp.py
│   ├── config/              # Configuration
│   │   └── settings.py
│   └── utils/               # Utilities
│       ├── audio.py         # Audio processing
│       ├── calibration.py   # Eye tracking calibration
│       └── logger.py
├── tests/                   # Test suite
├── docs/                    # Documentation
├── scripts/                 # Setup scripts
│   ├── install_deps.sh
│   └── calibrate_eyegaze.py
├── pyproject.toml
├── README.md
├── LICENSE (MIT)
└── CONTRIBUTING.md
```

---

## 🔧 Configuration Example

```yaml
# config/profiles/default.yaml
profile:
  name: "Default"
  inputs:
    eyegaze:
      enabled: true
      camera_id: 0
      dwell_time: 500ms
      calibration_required: true
    voice:
      enabled: true
      model: "vosk-model-small-en-us"
      threshold: 0.6
    switch:
      enabled: false
      port: "/dev/ttyUSB0"
  outputs:
    tts:
      enabled: true
      engine: "pyttsx3"  # or "gtts", "coqui"
      voice_clone: false
      rate: 150
      pitch: 1.0
    smarthome:
      enabled: false
      mqtt_broker: "localhost:1883"
  ai:
    intent:
      provider: "ollama"
      model: "llama3.2:latest"
      local: true
    prediction:
      enabled: true
      model: "phi3-mini"
  ui:
    theme: "light"
    font_size: 18
    grid_layout: "aac_standard"
```

---

## 🤝 Contributing

We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting PRs.

### Good First Issues
- Add a new input driver (Leap Motion, facial gestures)
- Add a new output driver (Braille display, Telegram bot)
- Improve the eye tracking calibration UI
- Add more language support for TTS
- Write integration tests for specific drivers

---

## 📜 License

MIT License — see [LICENSE](LICENSE).  
This means **anyone can use, modify, and distribute** AbleBridge, including for **commercial purposes**, as long as they include the copyright notice.

---

## 🌟 Roadmap

| Version | Milestone | Status |
|---------|-----------|--------|
| v0.1 | Core engine + Keyboard/Switch I/O + Basic TTS | ✅ **You are here** |
| v0.2 | EyeGaze driver + MediaPipe integration | 🔨 In Progress |
| v0.3 | Voice driver (Vosk ASR) + AAC prediction | 🔨 Planned |
| v0.4 | Ollama LLM intent engine + adaptive learning | 🔨 Planned |
| v0.5 | Voice cloning (Coqui TTS) + Smart Home | 🔨 Planned |
| v0.6 | First public beta + documentation | 🔨 Planned |
| v1.0 | Production-ready + community governance | 🔨 Planned |

---

## 💬 Community

- **GitHub Issues** — Bug reports, feature requests
- **Discussions** — Q&A, ideas, show-and-tell
- **Discord** — Real-time chat (coming soon)

---

*Built with ❤️ for the disability community. Every human deserves to be heard.*
