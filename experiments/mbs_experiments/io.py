from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

TERMINAL_STATUSES = {"ok", "skipped", "timeout", "no_plan", "unsupported", "error"}


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        json.dump(record, f, sort_keys=True)
        f.write("\n")


def write_jsonl_atomic(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "w") as f:
        for record in records:
            json.dump(record, f, sort_keys=True)
            f.write("\n")
    os.replace(tmp_path, path)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield payload


def result_key(record: Mapping[str, Any], fields: Sequence[str]) -> tuple[Any, ...]:
    return tuple(record.get(field) for field in fields)


def load_existing_keys(path: Path, fields: Sequence[str]) -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    for record in iter_jsonl(path):
        status = str(record.get("status", ""))
        if status in TERMINAL_STATUSES:
            keys.add(result_key(record, fields))
    return keys


def make_error_record(base: Mapping[str, Any], exc: BaseException) -> dict[str, Any]:
    record = dict(base)
    record.update(
        {
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    )
    return record
