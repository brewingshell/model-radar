from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

import click
import typer
from rich.console import Console

from distill.codec import parse_snapshot
from distill.config import ConfigError, load_config
from distill.pipeline import load_published_snapshot, run_pipeline
from distill.publisher import PublicationError, PublicationLockError, Publisher
from distill.source import RequiredSourceError
from distill.tui import SnapshotApp, plain_top

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def run(
    config: Annotated[Path, typer.Option(..., "--config")],
    output: Annotated[Path | None, typer.Option("--output")] = None,
    source: Annotated[str | None, typer.Option("--source")] = None,
    generated_at: Annotated[str | None, typer.Option("--generated-at")] = None,
    no_publish: Annotated[bool, typer.Option("--no-publish")] = False,
) -> None:
    try:
        settings = load_config(config)
        if output is not None:
            settings = settings.model_copy(update={"output": str(output)})
        timestamp = datetime.fromisoformat(generated_at) if generated_at else None
        snapshot = run_pipeline(
            settings, source_override=source, generated_at=timestamp, publish=not no_publish
        )
        console.print(f"{snapshot.status}: {len(snapshot.models)} models")
        if snapshot.status == "degraded":
            raise typer.Exit(code=3)
    except ConfigError as exc:
        console.print(str(exc), style="red")
        raise typer.Exit(code=78) from exc
    except RequiredSourceError as exc:
        console.print(str(exc), style="red")
        raise typer.Exit(code=69) from exc
    except PublicationLockError as exc:
        console.print(str(exc), style="red")
        raise typer.Exit(code=75) from exc
    except (PublicationError, ValueError, OSError) as exc:
        console.print(str(exc), style="red")
        raise typer.Exit(code=4) from exc
    except Exception as exc:
        console.print(str(exc), style="red")
        raise typer.Exit(code=1) from exc


@app.command()
def top(
    output: Annotated[Path, typer.Option("--output")] = Path("out"),
    snapshot: Annotated[Path | None, typer.Option("--snapshot")] = None,
    plain: Annotated[bool, typer.Option("--plain")] = False,
    once: Annotated[bool, typer.Option("--once")] = False,
) -> None:
    path = snapshot or (
        output / "snapshot.json"
        if (output / "snapshot.json").exists()
        else output / "current" / "snapshot.json"
    )
    try:
        parsed = parse_snapshot(path.read_bytes())
        if plain or once or not sys.stdout.isatty():
            console.print(plain_top(parsed))
        else:
            SnapshotApp(parsed).run()
    except (OSError, ValueError) as exc:
        console.print(str(exc), style="red")
        raise typer.Exit(code=4) from exc


@app.command()
def validate(snapshot: Path) -> None:
    try:
        parse_snapshot(snapshot.read_bytes())
        console.print("valid")
    except (OSError, ValueError) as exc:
        console.print(str(exc), style="red")
        raise typer.Exit(code=4) from exc


@app.command()
def render(
    snapshot: Annotated[Path, typer.Option(..., "--snapshot")],
    output: Annotated[Path, typer.Option("--output")] = Path("out"),
) -> None:
    try:
        Publisher(output).publish(parse_snapshot(snapshot.read_bytes()))
        console.print(f"published to {output / 'distill.html'}")
    except (OSError, ValueError, PublicationError, PublicationLockError) as exc:
        console.print(str(exc), style="red")
        raise typer.Exit(code=4) from exc


@app.command()
def inspect(output: Annotated[Path, typer.Option("--output")] = Path("out")) -> None:
    snapshot = load_published_snapshot(output)
    console.print(snapshot.model_dump_json(indent=2))


@app.command()
def sources(config: Annotated[Path, typer.Option(..., "--config")]) -> None:
    settings = load_config(config)
    for source in settings.sources:
        console.print(
            f"{source.name}: {source.kind} required={source.required} enabled={source.enabled}"
        )


def main() -> None:
    try:
        app(standalone_mode=False)
    except click.exceptions.UsageError as exc:
        console.print(str(exc), style="red")
        raise SystemExit(64) from exc
