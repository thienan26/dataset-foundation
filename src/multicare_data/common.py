from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


class PipelineError(ValueError):
    """An actionable integrity or configuration failure."""

    def __init__(self, message: str, attempt_count: int | None = None):
        super().__init__(message)
        self.attempt_count = attempt_count


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if immutable:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
    else:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(data, encoding="utf-8")
        os.replace(temp, path)


def read_rows(path: Path, optional: bool = False) -> list[dict]:
    if optional and not path.exists():
        return []
    if not path.exists():
        raise PipelineError(f"Missing artifact: {path}. Run the preceding stage first.")
    return pq.read_table(path).to_pylist()


def iter_parquet_rows(path: Path, columns: list[str] | None = None, batch_size: int = 1024):
    """Read Parquet in bounded batches for source tables too large for a full in-memory read."""
    if not path.exists():
        raise PipelineError(f"Missing artifact: {path}. Run the preceding stage first.")
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        yield from batch.to_pylist()


def write_rows(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows) if rows else pa.table({c: pa.array([], type=pa.string())
                                                           for c in (columns or ["case_id"])})
    temp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temp, compression="zstd")
    os.replace(temp, path)


def safe_path(root: Path, relative: str) -> Path:
    # Reject absolute paths, Windows drive prefixes and traversal on every platform.
    normalized = relative.replace("\\", "/")
    if not normalized or normalized.startswith("/") or ":" in normalized:
        raise PipelineError(f"Unsafe relative path: {relative!r}")
    if ".." in normalized.split("/"):
        raise PipelineError(f"Unsafe relative path: {relative!r}")
    target = (root / normalized).resolve()
    if not target.is_relative_to(root.resolve()):
        raise PipelineError(f"Path escapes root: {relative!r}")
    return target


def unique_index(rows: list[dict], key: str) -> dict:
    result = {}
    for row in rows:
        if not row.get(key) or row[key] in result:
            raise PipelineError(f"Empty or duplicate {key}: {row.get(key)}")
        result[row[key]] = row
    return result


class Project:
    def __init__(self, root: str | Path = "."):
        self.root = Path(root).resolve()

    def path(self, relative: str) -> Path:
        return safe_path(self.root, relative)

    def config(self, name: str) -> dict:
        path = self.path(f"configs/{name}.yaml")
        if not path.exists():
            raise PipelineError(f"Missing {path}. Use --root to select the project repository.")
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def rows(self, relative: str, optional: bool = False) -> list[dict]:
        return read_rows(self.path(relative), optional)

    def iter_rows(self, relative: str, columns: list[str] | None = None, batch_size: int = 1024):
        return iter_parquet_rows(self.path(relative), columns, batch_size)

    def write(self, relative: str, rows: list[dict], columns: list[str] | None = None) -> None:
        write_rows(self.path(relative), rows, columns)

    def config_hashes(self) -> dict:
        return {p.name: file_hash(p) for p in sorted(self.path("configs").glob("*.yaml"))}

