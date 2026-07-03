"""
ablebridge.cli — Command-line interface.

Usage:
    ablebridge gui          # Launch web GUI
    ablebridge core         # Run core engine (no GUI)
    ablebridge calibrate    # Run eye tracking calibration
    ablebridge status       # Show system status
    ablebridge predict      # Test prediction engine
    ablebridge voicespeak   # Test TTS
    ablebridge driver list  # List available drivers
    ablebridge driver test <driver>  # Test a specific driver
    ablebridge profile list # List profiles
    ablebridge profile new <name>    # Create new profile
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from ablebridge import __version__
from ablebridge.core.engine import AbleBridgeEngine
from ablebridge.core.types import InputChannel, OutputChannel

app = typer.Typer(
    name="ablebridge",
    help="AbleBridge — Universal AI Accessibility Bridge",
    add_completion=False,
)
console = Console()


def _get_engine(profile_dir: str = "config/profiles") -> AbleBridgeEngine:
    engine = AbleBridgeEngine(profile_dir=profile_dir)
    engine.load_profile("default")
    return engine


@app.command()
def gui(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8765, help="Port to bind"),
    profile: str = typer.Option("default", help="Profile ID"),
    profile_dir: str = typer.Option("config/profiles", help="Profile directory"),
) -> None:
    """Launch the web-based GUI."""
    from ablebridge.gui.app import main as gui_main

    console.print(Panel.fit(
        f"[bold]AbleBridge v{__version__}[/bold]\n"
        "Universal AI Accessibility Bridge\n\n"
        f"🌐 GUI: http://localhost:{port}\n"
        f"📁 Profile: {profile}",
        border_style="green",
    ))

    engine = AbleBridgeEngine(profile_dir=profile_dir)
    engine.load_profile(profile)
    engine.auto_register_drivers()
    engine.start()

    try:
        gui_main(host=host, port=port)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
    finally:
        engine.stop()


@app.command()
def core(
    profile: str = typer.Option("default", help="Profile ID"),
    profile_dir: str = typer.Option("config/profiles", help="Profile directory"),
) -> None:
    """Run the core engine without GUI (for headless servers / Raspberry Pi)."""
    console.print(f"[bold]Starting AbleBridge Core[/bold] — profile: {profile}")

    engine = AbleBridgeEngine(profile_dir=profile_dir)
    engine.load_profile(profile)
    engine.auto_register_drivers()
    engine.start()

    console.print(f"[green]Engine running. Session: {engine.session_id}[/green]")
    console.print("Press Ctrl+C to stop.\n")

    try:
        while engine.is_running:
            time.sleep(5)
            status = engine.get_system_status()
            running_inputs = sum(
                1 for s in status["inputs"].values()
                if s.get("state") == "running"
            )
            running_outputs = sum(
                1 for s in status["outputs"].values()
                if s.get("state") == "running"
            )
            console.print(
                f"  [{time.strftime('%H:%M:%S')}] "
                f"Inputs: {running_inputs} | Outputs: {running_outputs} | "
                f"Bus events: {status['bus_stats']['published']}"
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")
    finally:
        engine.stop()
        console.print("[green]Stopped.[/green]")


@app.command()
def status(
    profile_dir: str = typer.Option("config/profiles", help="Profile directory"),
) -> None:
    """Show system status (inputs, outputs, AI engines)."""
    engine = AbleBridgeEngine(profile_dir=profile_dir)
    engine.load_profile("default")

    status = engine.get_system_status()

    # Inputs table
    table = Table(title="Input Channels")
    table.add_column("Channel", style="cyan")
    table.add_column("State", style="yellow")
    table.add_column("Confidence", style="green")
    table.add_column("Latency", style="magenta")

    for ch_id, ch_status in status["inputs"].items():
        state = ch_status.get("state", "unknown")
        conf = f"{ch_status.get('confidence', 0) * 100:.0f}%"
        lat = f"{ch_status.get('latency_ms', 0):.0f}ms"
        table.add_row(ch_id, state, conf, lat)

    console.print(table)
    console.print()

    # Outputs table
    table2 = Table(title="Output Channels")
    table2.add_column("Channel", style="cyan")
    table2.add_column("State", style="yellow")
    for ch_id, ch_status in status["outputs"].items():
        table2.add_row(ch_id, ch_status.get("state", "unknown"))
    console.print(table2)
    console.print()

    # AI engines
    ai = status.get("ai", {})
    console.print(Panel(
        f"[bold]Intent Engine:[/bold] {ai.get('intent', 'None')}\n"
        f"[bold]Prediction:[/bold] {ai.get('prediction', 'None')}",
        title="AI Engines",
        border_style="blue",
    ))


@app.command()
def predict(
    text: str = typer.Argument("", help="Text to get predictions for"),
    profile_dir: str = typer.Option("config/profiles", help="Profile directory"),
) -> None:
    """Test the prediction engine."""
    engine = _get_engine(profile_dir)
    engine.auto_register_drivers()
    engine.start()

    if not text:
        console.print("[yellow]Enter text to predict:[/yellow]")
        text = typer.prompt("Text")

    predictions = engine.predict_next(text)
    console.print(f"\n[bold]Predictions for:[/bold] '{text}'")
    for word, conf in predictions:
        bar = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
        console.print(f"  {word:<20} {bar} {conf:.2%}")

    engine.stop()


@app.command()
def test_tts(
    text: str = typer.Option("Hello! AbleBridge is working.", help="Text to speak"),
    profile_dir: str = typer.Option("config/profiles", help="Profile directory"),
) -> None:
    """Test the TTS output."""
    engine = _get_engine(profile_dir)
    engine.auto_register_drivers()
    engine.start()
    console.print(f"[green]Speaking:[/green] '{text}'")
    engine.speak(text)
    time.sleep(3)
    engine.stop()


@app.command()
def driver_list() -> None:
    """List all available input and output drivers."""
    table = Table(title="Available Drivers")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Status", style="yellow")

    # Input drivers
    for ch in InputChannel:
        table.add_row("input", ch.value, "✅ Available")
    for ch in OutputChannel:
        table.add_row("output", ch.value, "✅ Available")

    console.print(table)


@app.command()
def profile_list(
    profile_dir: str = typer.Option("config/profiles", help="Profile directory"),
) -> None:
    """List all user profiles."""
    engine = AbleBridgeEngine(profile_dir=profile_dir)
    profiles = engine.list_profiles()
    console.print("[bold]Available profiles:[/bold]")
    for p in profiles:
        console.print(f"  • {p}")


@app.command()
def profile_new(
    name: str = typer.Argument(..., help="Profile name"),
    profile_dir: str = typer.Option("config/profiles", help="Profile directory"),
) -> None:
    """Create a new user profile."""
    from ablebridge.core.profile import ProfileManager
    from ablebridge.core.types import UserProfile

    mgr = ProfileManager(Path(profile_dir))
    profile = UserProfile(id=name, name=name)
    mgr.save(profile)
    console.print(f"[green]Created profile:[/green] {name}")


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point (used by pyproject.toml)
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    app()


if __name__ == "__main__":
    main()
