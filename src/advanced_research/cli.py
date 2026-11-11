"""Command line interface for Advanced Research framework."""

import asyncio
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path
import yaml

from .core.context import ContextInjector, Priority
from .core.integrations import (
    IntegrationsManager,
    LRSIntegration,
    OpencodeIntegration,
    NeuralBlitzIntegration,
)
from .core.lrs_context import LRSContextInjector

app = typer.Typer(help="Advanced Research Framework CLI")
console = Console()


@app.command()
def init(
    config_file: Path = typer.Option(
        Path("config.yaml"), "--config", "-c", help="Configuration file path"
    ),
) -> None:
    """Initialize the Advanced Research framework."""
    console.print("[bold green]Initializing Advanced Research Framework[/bold green]")

    # Create default config if it doesn't exist
    if not config_file.exists():
        default_config = {
            "lrs": {
                "enabled": False,
                "mode": "mock",
                "endpoint": "http://localhost:8080/xapi/",
            },
            "opencode": {
                "enabled": False,
                "mode": "mock",
                "workspace": "advanced-research",
            },
            "neuralblitz": {
                "enabled": False,
                "mode": "mock",
                "backend": "jax",
                "model": {
                    "hidden_dim": 256,
                    "num_layers": 4,
                },
            },
        }

        with open(config_file, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)

        console.print(f"[green]Created default configuration at {config_file}[/green]")

    # Load configuration
    with open(config_file) as f:
        config = yaml.safe_load(f)

    console.print(Panel(f"Configuration loaded from {config_file}", title="Config"))

    # Initialize integrations
    integrations = IntegrationsManager(config)
    asyncio.run(integrations.initialize_all())

    console.print("[yellow]Configured integrations initialized (mock mode is simulated, not connected).[/yellow]")


@app.command()
def context_demo(
    user_id: str = typer.Option(
        "demo-user", "--user", "-u", help="User ID for tracking"
    ),
) -> None:
    """Demonstrate the context injection system with LRS integration."""
    console.print("[bold blue]Context Injection System Demo[/bold blue]")

    # Create LRS integration
    lrs_config = {"enabled": False, "mode": "mock", "endpoint": "http://localhost:8080/xapi/"}
    lrs_integration = LRSIntegration(lrs_config)

    # Create enhanced context injector
    injector = LRSContextInjector(lrs_integration)
    injector.set_user(user_id)

    # Add sample context blocks
    injector.add_context(
        "research_goal",
        "Develop geometric deep learning architectures for understanding complex manifolds in high-dimensional spaces.",
        Priority.HIGH,
        tags=["research", "geometry"],
    )

    injector.add_context(
        "current_approach",
        "Using Riemannian geometry principles to design attention mechanisms that respect manifold structure.",
        Priority.MEDIUM,
        tags=["approach", "riemannian"],
    )

    injector.add_context(
        "mathematical_foundation",
        "Key concepts: geodesics, curvature tensors, parallel transport, exponential maps.",
        Priority.HIGH,
        tags=["math", "foundation"],
    )

    # Display context
    context = injector.get_context()
    console.print(Panel(context, title="Generated Context"))

    # Display statistics
    stats = injector.get_statistics()

    table = Table(title="Context Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in stats.items():
        table.add_row(key.replace("_", " ").title(), str(value))

    console.print(table)


@app.command()
def integrations_status(
    config_file: Path = typer.Option(
        Path("config.yaml"), "--config", "-c", help="Configuration file path"
    ),
) -> None:
    """Show status of all integrations."""
    console.print("[bold yellow]Integration Status[/bold yellow]")

    if not config_file.exists():
        console.print(f"[red]Configuration file {config_file} not found[/red]")
        return

    with open(config_file) as f:
        config = yaml.safe_load(f)

    integrations = IntegrationsManager(config)

    table = Table(title="Integration Status")
    table.add_column("Integration", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Configuration", style="blue")

    # LRS Status
    lrs_status = "✓ Enabled" if config.get("lrs", {}).get("enabled") else "✗ Disabled"
    lrs_config = f"Endpoint: {config.get('lrs', {}).get('endpoint', 'N/A')}"
    table.add_row("LRS Agents", lrs_status, lrs_config)

    # Opencode Status
    opencode_status = (
        "✓ Enabled" if config.get("opencode", {}).get("enabled") else "✗ Disabled"
    )
    opencode_config = f"Workspace: {config.get('opencode', {}).get('workspace', 'N/A')}"
    table.add_row("Opencode", opencode_status, opencode_config)

    # NeuralBlitz Status
    neuralblitz_status = (
        "✓ Enabled" if config.get("neuralblitz", {}).get("enabled") else "✗ Disabled"
    )
    neuralblitz_config = (
        f"Backend: {config.get('neuralblitz', {}).get('backend', 'N/A')}"
    )
    table.add_row("NeuralBlitz v50", neuralblitz_status, neuralblitz_config)

    console.print(table)


@app.command()
def test_integrations(
    config_file: Path = typer.Option(
        Path("config.yaml"), "--config", "-c", help="Configuration file path"
    ),
) -> None:
    """Test all integrations."""
    console.print("[bold magenta]Testing Integrations[/bold magenta]")

    if not config_file.exists():
        console.print(f"[red]Configuration file {config_file} not found[/red]")
        return

    with open(config_file) as f:
        config = yaml.safe_load(f)

    integrations = IntegrationsManager(config)

    async def run_checks() -> None:
        try:
            await integrations.initialize_all()
            for name in ("lrs", "opencode", "neuralblitz"):
                integration = integrations.get_integration(name)
                if integration is None or not integration.enabled:
                    console.print(f"[yellow]{name}: disabled[/yellow]")
                elif integration.mode == "mock":
                    console.print(f"[yellow]{name}: MOCK only; no network write, persistence, or measured computation[/yellow]")
                else:
                    console.print(f"[red]{name}: real adapter unavailable[/red]")
                    raise RuntimeError(f"{name} does not have a real provider adapter")
        finally:
            await integrations.shutdown_all()

    try:
        asyncio.run(run_checks())
    except Exception as exc:
        console.print(f"[red]Integration verification failed: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print("[yellow]Integration mode verification complete; real connectivity was not tested.[/yellow]")

def main() -> None:
    app()
