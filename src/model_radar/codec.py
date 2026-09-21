from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from model_radar.models import Snapshot


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def snapshot_json(snapshot: Snapshot) -> bytes:
    payload = snapshot.model_dump(mode="json", exclude_none=False)
    return canonical_json(payload)


def parse_snapshot(data: bytes) -> Snapshot:
    return Snapshot.model_validate(json.loads(data))


def utc_now() -> datetime:
    return datetime.now(UTC)
