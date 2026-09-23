from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from model_radar.codec import canonical_json, parse_snapshot, snapshot_json
from model_radar.models import Snapshot
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
            changes = _summarize_changes(snapshot, previous)
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

    def _previous_snapshot(self, current: Snapshot) -> Snapshot | None:
        candidates: list[tuple[datetime, Path]] = []
        current_date = current.generated_at.date()
        if self.history.exists():
            for path in self.history.glob("*.json"):
                try:
                    snapshot = parse_snapshot(path.read_bytes())
                    if snapshot.generated_at.date() < current_date:
                        candidates.append((snapshot.generated_at, path))
                except (OSError, ValueError):
                    continue
        root_snapshot = self.output / "snapshot.json"
        if root_snapshot.exists():
            try:
                snapshot = parse_snapshot(root_snapshot.read_bytes())
                if snapshot.generated_at.date() < current_date:
                    candidates.append((snapshot.generated_at, root_snapshot))
            except (OSError, ValueError):
                pass
        if not candidates:
            return None
        path = max(candidates, key=lambda item: item[0])[1]
        try:
            return parse_snapshot(path.read_bytes())
        except (OSError, ValueError):
            return None

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


def _summarize_changes(current: Snapshot, previous: Snapshot | None) -> dict[str, Any]:
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
    current_names = {model.name for model in current.models}
    previous_names = {model.name for model in previous.models}
    new_names = sorted(current_names - previous_names)
    summary: list[str] = []
    items: list[dict[str, Any]] = []
    if new_names:
        detail = f"{len(new_names)} model records entered the catalog."
        summary.append(f"{detail} Examples: {', '.join(new_names[:4])}.")
        items.append(
            {
                "kind": "new",
                "label": "New models",
                "count": len(new_names),
                "detail": detail,
                "examples": new_names[:6],
            }
        )
    current_sources = current.summary.get("source_record_count", 0)
    previous_sources = previous.summary.get("source_record_count", 0)
    if current_sources != previous_sources:
        detail = f"Source records moved from {previous_sources} to {current_sources}."
        summary.append(detail)
        items.append(
            {
                "kind": "source",
                "label": "Source records",
                "detail": detail,
                "examples": [],
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
