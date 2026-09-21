from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.constructor import DuplicateKeyError

from model_radar.models import AppConfig


class ConfigError(ValueError):
    """Configuration cannot be loaded or validated."""


def load_config(path: str | Path, environ: dict[str, str] | None = None) -> AppConfig:
    env = os.environ if environ is None else environ
    yaml = YAML(typ="safe")
    yaml.allow_duplicate_keys = False
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            data: Any = yaml.load(handle) or {}
    except (OSError, DuplicateKeyError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    if not isinstance(data, dict):
        raise ConfigError("configuration root must be a mapping")
    try:
        data = _apply_environment(data, env)
        models_path = Path(path).with_name("models.json")
        if models_path.is_file():
            with models_path.open("r", encoding="utf-8") as handle:
                models_data = json.load(handle)
            if not isinstance(models_data, dict) or not isinstance(
                models_data.get("model_types"), dict
            ):
                raise ConfigError("models.json must contain a model_types mapping")
            data["model_type_tabs"] = models_data["model_types"]
        org_path = Path(path).with_name("org.json")
        if org_path.is_file():
            with org_path.open("r", encoding="utf-8") as handle:
                org_data = json.load(handle)
            if not isinstance(org_data, dict) or not isinstance(org_data.get("models"), list):
                raise ConfigError("org.json must contain a models list")
            copilot = data.setdefault("copilot", {})
            if not isinstance(copilot, dict):
                raise ConfigError("copilot configuration must be a mapping")
            copilot["models"] = org_data["models"]
            if isinstance(org_data.get("source"), str):
                copilot["source"] = org_data["source"]
        return AppConfig.model_validate(data)
    except (ValidationError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ConfigError(str(exc)) from exc


def _apply_environment(data: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    result = dict(data)
    if "MODEL_RADAR_OUTPUT" in env:
        result["output"] = env["MODEL_RADAR_OUTPUT"]
    if "MODEL_RADAR_MAX_PAGES" in env:
        result.setdefault("fetch", {})["max_pages"] = int(env["MODEL_RADAR_MAX_PAGES"])
    if "MODEL_RADAR_MAX_RECORDS" in env:
        result.setdefault("fetch", {})["max_records"] = int(env["MODEL_RADAR_MAX_RECORDS"])
    return result
