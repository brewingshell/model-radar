from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

try:
    from typer._click.exceptions import UsageError as _UsageError
except ImportError:  # pragma: no cover - fallback for unvendored typer
    from click.exceptions import UsageError as _UsageError  # type: ignore[assignment]

from model_radar.codec import parse_snapshot
from model_radar.config import ConfigError, load_config
from model_radar.models import ModelRecord, Snapshot
from model_radar.pipeline import load_published_snapshot, run_pipeline
from model_radar.publisher import PublicationError, PublicationLockError, Publisher
from model_radar.source import RequiredSourceError
from model_radar.tui import SnapshotApp, plain_top

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
        console.print(f"published to {output / 'model-radar.html'}")
    except (OSError, ValueError, PublicationError, PublicationLockError) as exc:
        console.print(str(exc), style="red")
        raise typer.Exit(code=4) from exc


def _find_models(snapshot: Snapshot, key: str) -> list[ModelRecord]:
    for attribute in ("model_id", "slug", "name"):
        matches = [model for model in snapshot.models if getattr(model, attribute) == key]
        if matches:
            return matches
    return []


@app.command()
def inspect(
    model_id: Annotated[str | None, typer.Argument()] = None,
    output: Annotated[Path, typer.Option("--output")] = Path("out"),
) -> None:
    try:
        snapshot = load_published_snapshot(output)
    except (OSError, ValueError) as exc:
        console.print(str(exc), style="red")
        raise typer.Exit(code=4) from exc
    if model_id is None:
        console.print(snapshot.model_dump_json(indent=2))
        return
    matches = _find_models(snapshot, model_id)
    if not matches:
        console.print(f"model not found: {model_id}", style="red")
        raise typer.Exit(code=4)
    for index, model in enumerate(matches):
        if index:
            console.print()
        console.print(
            json.dumps(
                {
                    "model": model.model_dump(mode="json"),
                    "provenance": model.provenance,
                    "field_provenance": model.field_provenance,
                },
                indent=2,
            )
        )


@app.command()
def sources(config: Annotated[Path, typer.Option(..., "--config")]) -> None:
    try:
        settings = load_config(config)
    except ConfigError as exc:
        console.print(str(exc), style="red")
        raise typer.Exit(code=78) from exc
    for source in settings.sources:
        console.print(
            f"{source.name}: {source.kind} required={source.required} enabled={source.enabled}"
        )


def main() -> None:
    try:
        result = app(standalone_mode=False)
    except _UsageError as exc:
        console.print(str(exc), style="red")
        raise SystemExit(64) from exc
    if isinstance(result, int) and result != 0:
        raise SystemExit(result)
