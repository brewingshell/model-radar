from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from model_radar.analysis import (
    _benchmark_family_key,
    _model_family_name,
    model_source_url,
    model_type,
    notable_new_models,
)
from model_radar.codec import canonical_json, parse_snapshot, snapshot_json
from model_radar.models import ModelRecord, Snapshot
from model_radar.render import render_html


class PublicationLockError(RuntimeError):
    pass


class PublicationError(RuntimeError):
    pass


class Publisher:
    def __init__(self, output: str | Path, max_html_bytes: int = 2_000_000) -> None:
        self.output = Path(output)
        self.max_html_bytes = max_html_bytes
        self.current = self.output / "current"
        self.history = self.output / "history"
        self.lock = self.output / ".publish.lock"

    def publish(self, snapshot: Snapshot) -> dict[str, object]:
        self.output.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise PublicationLockError("publication lock already exists") from exc
        try:
            os.close(fd)
            previous = self._previous_snapshot(snapshot)
            oldest = self._oldest_snapshot(snapshot)
            changes = _summarize_changes(snapshot, previous, oldest)
            published = snapshot.model_copy(update={"changes": changes})
            stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=self.output))
            try:
                json_bytes = snapshot_json(published)
                html_bytes = render_html(published)
                if not json_bytes:
                    raise PublicationError("artifact validation failed")
                if len(html_bytes) > self.max_html_bytes:
                    raise PublicationError("HTML exceeds the configured safety limit")
                files = {"snapshot.json": json_bytes, "model-radar.html": html_bytes}
                manifest_entries: dict[str, dict[str, object]] = {}
                for name, content in files.items():
                    path = stage / name
                    path.write_bytes(content)
                    _fsync(path)
                    manifest_entries[name] = {
                        "size": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                manifest: dict[str, object] = {
                    "schema_version": published.schema_version,
                    "snapshot_id": published.snapshot_id,
                    "files": manifest_entries,
                }
                manifest_path = stage / "manifest.json"
                manifest_path.write_bytes(canonical_json(manifest))
                _fsync(manifest_path)
                self._validate_artifacts(stage, published)
                _fsync(stage)
                self._publish_root_files(stage, files, manifest_path)
                self._store_history(published, json_bytes)
                self._refresh_current_directory(stage)
                shutil.rmtree(stage, ignore_errors=True)
                return manifest
            except Exception as exc:
                shutil.rmtree(stage, ignore_errors=True)
                raise PublicationError(str(exc)) from exc
        finally:
            self.lock.unlink(missing_ok=True)

    def _prior_snapshots(self, current: Snapshot) -> list[tuple[datetime, Snapshot]]:
        """Every retained snapshot from before the current calendar day."""
        current_date = current.generated_at.date()
        snapshots: dict[datetime, Snapshot] = {}
        paths: list[Path] = []
        if self.history.exists():
            paths.extend(self.history.glob("*.json"))
        root_snapshot = self.output / "snapshot.json"
        if root_snapshot.exists():
            paths.append(root_snapshot)
        for path in paths:
            try:
                snapshot = parse_snapshot(path.read_bytes())
            except (OSError, ValueError):
                continue
            if snapshot.generated_at.date() < current_date:
                snapshots[snapshot.generated_at] = snapshot
        return sorted(snapshots.items())

    def _previous_snapshot(self, current: Snapshot) -> Snapshot | None:
        prior = self._prior_snapshots(current)
        return prior[-1][1] if prior else None

    def _oldest_snapshot(self, current: Snapshot) -> Snapshot | None:
        prior = self._prior_snapshots(current)
        return prior[0][1] if prior else None

    def _publish_root_files(
        self, stage: Path, files: dict[str, bytes], manifest_path: Path
    ) -> None:
        for name, content in {**files, "manifest.json": manifest_path.read_bytes()}.items():
            temporary = self.output / f".{name}.next"
            temporary.write_bytes(content)
            _fsync(temporary)
            os.replace(temporary, self.output / name)
        _fsync(self.output)

    def _store_history(self, snapshot: Snapshot, json_bytes: bytes) -> None:
        self.history.mkdir(parents=True, exist_ok=True)
        path = self.history / f"{snapshot.generated_at.date().isoformat()}.json"
        temporary = self.history / f".{snapshot.generated_at.date().isoformat()}.next"
        temporary.write_bytes(json_bytes)
        _fsync(temporary)
        os.replace(temporary, path)
        cutoff = snapshot.generated_at.date() - timedelta(days=13)
        by_date: dict[date, tuple[datetime, Path]] = {}
        for candidate in self.history.glob("*.json"):
            try:
                stored = parse_snapshot(candidate.read_bytes())
            except (OSError, ValueError):
                continue
            stored_date = stored.generated_at.date()
            if stored_date < cutoff:
                candidate.unlink(missing_ok=True)
                continue
            existing = by_date.get(stored_date)
            if existing is None or stored.generated_at > existing[0]:
                by_date[stored_date] = (stored.generated_at, candidate)
        for stored_date, (_, candidate) in by_date.items():
            for duplicate in self.history.glob("*.json"):
                if duplicate == candidate:
                    continue
                try:
                    duplicate_snapshot = parse_snapshot(duplicate.read_bytes())
                except (OSError, ValueError):
                    continue
                if duplicate_snapshot.generated_at.date() == stored_date:
                    duplicate.unlink(missing_ok=True)
        _fsync(self.history)

    def _refresh_current_directory(self, stage: Path) -> None:
        if self.current.is_symlink():
            self.current.unlink()
        self.current.mkdir(parents=True, exist_ok=True)
        for name in ("snapshot.json", "model-radar.html", "manifest.json"):
            shutil.copy2(self.output / name, self.current / name)

    def _validate_artifacts(self, stage: Path, snapshot: Snapshot) -> None:
        parsed = parse_snapshot((stage / "snapshot.json").read_bytes())
        if parsed != snapshot:
            raise PublicationError("snapshot round-trip validation failed")
        manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("snapshot_id") != snapshot.snapshot_id:
            raise PublicationError("manifest snapshot ID does not match snapshot")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise PublicationError("manifest files are invalid")
        for name, metadata in files.items():
            path = stage / name
            if not path.is_file() or not isinstance(metadata, dict):
                raise PublicationError(f"manifest entry is invalid: {name}")
            content = path.read_bytes()
            if (
                metadata.get("size") != len(content)
                or metadata.get("sha256") != hashlib.sha256(content).hexdigest()
            ):
                raise PublicationError(f"manifest checksum failed: {name}")
        html = (stage / "model-radar.html").read_text(encoding="utf-8")
        if "Content-Security-Policy" not in html:
            raise PublicationError("HTML is missing Content-Security-Policy")
        if len(html.encode("utf-8")) > self.max_html_bytes:
            raise PublicationError("HTML exceeds the configured safety limit")


_LEADERBOARD_VIEWS: tuple[tuple[str, str, str], ...] = (
    ("performance-top5", "Performance", "intelligence_index"),
    ("performance-per-token-top5", "Value per token", "aa_token_dollar_efficiency"),
    ("org-copilot-best-top10", "Org Copilot best", "intelligence_index"),
    ("org-copilot-per-token-top10", "Org Copilot per token", "copilot_token_efficiency"),
    ("tiny-llm-top10", "Tiny LLM", "intelligence_index"),
    ("mini-llm-top10", "Mini LLM", "intelligence_index"),
)


def _leaderboard_value(model: ModelRecord, metric: str) -> float | None:
    if metric == "intelligence_index":
        return model.intelligence_index
    score = model.scores.get(metric)
    return score.value if score and score.value is not None else None


def _view_leader(snapshot: Snapshot, view_id: str, metric: str) -> tuple[ModelRecord, float] | None:
    by_id = {model.model_id: model for model in snapshot.models}
    view = next((item for item in snapshot.views if item.view_id == view_id), None)
    if view is None:
        return None
    for model_id in view.model_ids:
        model = by_id.get(model_id)
        if model is None or model_type(model) != "llm":
            continue
        value = _leaderboard_value(model, metric)
        if value is not None:
            return model, value
    return None


def _leaderboard_updates(current: Snapshot, previous: Snapshot) -> list[str]:
    updates: list[str] = []
    for view_id, label, metric in _LEADERBOARD_VIEWS:
        current_leader = _view_leader(current, view_id, metric)
        previous_leader = _view_leader(previous, view_id, metric)
        if current_leader is None or previous_leader is None:
            continue
        challenger, challenger_value = current_leader
        incumbent, incumbent_value = previous_leader
        if challenger.model_id == incumbent.model_id:
            continue
        if _benchmark_family_key(challenger.name) == _benchmark_family_key(incumbent.name):
            continue
        delta = challenger_value - incumbent_value
        if delta <= 0:
            continue
        current_name = _display_family(challenger)
        previous_name = _display_family(incumbent)
        updates.append(
            f"{current_name} overtook {previous_name} on {label}, "
            f"{challenger_value:g} vs {incumbent_value:g} ({delta:+g})."
        )
    return updates[:4]


_FAMILY_PREFIX = re.compile(r"^[A-Za-z0-9 .&-]+:\s*")


def _display_family(model: ModelRecord) -> str:
    return _model_family_name(_FAMILY_PREFIX.sub("", model.name))


def _example(text: str, url: str | None = None) -> dict[str, str | None]:
    return {"text": text, "url": url}


def _example_text(example: object) -> str:
    if isinstance(example, dict):
        return str(example.get("text", ""))
    return str(example)


def _new_model_item(
    current: Snapshot,
    baseline: Snapshot,
    kind: str,
    label: str,
    limit: int = 10,
    exclude: set[str] | None = None,
) -> dict[str, Any] | None:
    baseline_names = {model.name for model in baseline.models}
    new_models = [model for model in current.models if model.name not in baseline_names]
    if not new_models:
        return None
    candidates = new_models
    if exclude:
        candidates = [
            model for model in new_models if _model_family_name(model.name) not in exclude
        ]
    notable = [
        _example(_model_family_name(model.name), model_source_url(model))
        for model in notable_new_models(candidates, limit=limit)
    ]
    if not notable:
        return None
    date = baseline.generated_at.date().isoformat()
    detail = f"{len(new_models)} models added since {date}."
    return {
        "kind": kind,
        "label": label,
        "count": len(new_models),
        "detail": detail,
        "examples": notable,
        "since": date,
    }


def _summarize_changes(
    current: Snapshot,
    previous: Snapshot | None,
    oldest: Snapshot | None = None,
) -> dict[str, Any]:
    if previous is None:
        detail = "First retained snapshot; future runs will show changes here."
        return {
            "summary": [detail],
            "items": [
                {
                    "kind": "first",
                    "label": "First snapshot",
                    "detail": detail,
                    "examples": [],
                }
            ],
            "previous_snapshot_id": None,
        }
    summary: list[str] = []
    items: list[dict[str, Any]] = []
    daily_item = _new_model_item(current, previous, "new", "New since last snapshot")
    daily_families = (
        {_example_text(item) for item in daily_item["examples"]} if daily_item else set()
    )
    window_item = None
    if (
        oldest is not None
        and oldest.generated_at.date() < previous.generated_at.date()
        and daily_item is not None
    ):
        window_item = _new_model_item(
            current,
            oldest,
            "new-window",
            "New in retained history",
            exclude=daily_families,
        )
    if window_item is not None:
        names = ", ".join(_example_text(item) for item in window_item["examples"][:4])
        summary.append(f"{window_item['detail']} Notable: {names}.")
        items.append(window_item)
    if daily_item is not None:
        names = ", ".join(_example_text(item) for item in daily_item["examples"][:4])
        summary.append(f"{daily_item['detail']} Notable: {names}.")
        items.append(daily_item)
    leaderboard = _leaderboard_updates(current, previous)
    if leaderboard:
        details = [_example(detail) for detail in leaderboard]
        summary.extend(leaderboard)
        items.append(
            {
                "kind": "leaderboard",
                "label": "Leaderboard update",
                "count": len(leaderboard),
                "detail": (
                    "1 leaderboard changed hands."
                    if len(leaderboard) == 1
                    else f"{len(leaderboard)} leaderboards changed hands."
                ),
                "examples": details,
            }
        )
    if not summary:
        detail = "No material model or shortlist changes were detected."
        summary.append(detail)
        items.append(
            {
                "kind": "quiet",
                "label": "No changes",
                "detail": detail,
                "examples": [],
            }
        )
    return {
        "summary": summary[:8],
        "items": items[:8],
        "previous_snapshot_id": previous.snapshot_id,
    }


def _fsync(path: Path) -> None:
    if path.is_dir():
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    else:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
