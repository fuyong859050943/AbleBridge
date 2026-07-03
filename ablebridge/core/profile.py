"""
ablebridge.core.profile — User profile persistence.

Profiles are YAML files that store complete configuration:
inputs, outputs, AI settings, UI preferences.
"""

from __future__ import annotations

import time
from pathlib import Path

import yaml
from loguru import logger
from pydantic import ValidationError

from ablebridge.core.types import UserProfile, InputConfig, OutputConfig, AIConfig


class ProfileManager:
    """
    Load, save, and list user profiles from the filesystem.
    Profiles are stored as YAML files in the profile directory.
    """

    DEFAULT_PROFILE = """
profile:
  id: default
  name: Default User
  description: Default accessibility profile for AbleBridge
  inputs:
    keyboard:
      enabled: true
      dwell_time_ms: 0
      scan_rate_hz: 2.0
      sensitivity: 1.0
    eyegaze:
      enabled: false
      dwell_time_ms: 500
      scan_rate_hz: 2.0
      sensitivity: 0.7
      custom_params:
        camera_id: 0
    voice:
      enabled: false
      dwell_time_ms: 0
      scan_rate_hz: 1.0
      sensitivity: 0.6
    switch:
      enabled: false
      dwell_time_ms: 300
      scan_rate_hz: 2.0
      sensitivity: 1.0
      custom_params:
        port: /dev/ttyUSB0
  outputs:
    tts:
      enabled: true
      volume: 1.0
      rate: 1.0
      pitch: 1.0
    smarthome:
      enabled: false
      volume: 1.0
      rate: 1.0
      pitch: 1.0
  ai:
    provider: ollama
    model: llama3.2:latest
    base_url: http://localhost:11434
    local: true
    temperature: 0.7
    max_tokens: 256
    prediction_enabled: true
    adaptive_learning_enabled: true
    intent_confidence_threshold: 0.60
  ui:
    theme: light
    font_size: 18
    grid_layout: aac_standard
    language: en
    aac_preset: standard
"""

    def __init__(self, profile_dir: Path | str = "config/profiles"):
        self._dir = Path(profile_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        # Write default profile if none exist
        if not list(self._dir.glob("*.yaml")) and not list(self._dir.glob("*.yml")):
            self._write_default_profile()

    def _write_default_profile(self) -> None:
        """Write the built-in default profile."""
        default_path = self._dir / "default.yaml"
        content = yaml.safe_load(self.DEFAULT_PROFILE)
        with open(default_path, "w", encoding="utf-8") as f:
            yaml.dump(content, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"[ProfileManager] Created default profile at {default_path}")

    def _profile_path(self, profile_id: str) -> Path:
        return self._dir / f"{profile_id}.yaml"

    def list_profiles(self) -> list[str]:
        """Return all available profile IDs."""
        return [
            p.stem for p in self._dir.glob("*.yaml")
        ] + [
            p.stem for p in self._dir.glob("*.yml")
        ]

    def load(self, profile_id: str) -> UserProfile:
        """Load a profile by ID. Creates default if not found."""
        path = self._profile_path(profile_id)
        if not path.exists():
            logger.warning(f"[ProfileManager] Profile '{profile_id}' not found, creating default")
            return self._create_default(profile_id)

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        try:
            data = raw.get("profile", raw) if isinstance(raw, dict) else {}
            return UserProfile(**data)
        except ValidationError as e:
            logger.error(f"[ProfileManager] Profile validation error: {e}")
            return self._create_default(profile_id)

    def save(self, profile: UserProfile) -> None:
        """Save a profile to disk."""
        path = self._profile_path(profile.id)
        data = {"profile": profile.model_dump()}
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"[ProfileManager] Saved profile '{profile.id}' to {path}")

    def _create_default(self, profile_id: str) -> UserProfile:
        """Create and save a default profile."""
        profile = UserProfile(id=profile_id)
        self.save(profile)
        return profile

    def delete(self, profile_id: str) -> bool:
        """Delete a profile. Returns False if it doesn't exist."""
        path = self._profile_path(profile_id)
        if path.exists():
            path.unlink()
            return True
        return False
