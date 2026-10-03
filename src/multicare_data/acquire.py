from __future__ import annotations

import json
import os
import shutil
import stat
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .common import PipelineError, Project, file_hash, now, read_json, safe_path, write_json


def get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "multicare-data/0.1 research"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def download(url: str, path: Path, expected_size: int | None = None,
             checksum: str | None = None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)

    def validate(target):
        if expected_size is not None and target.stat().st_size != expected_size:
            raise PipelineError(f"Size mismatch: {target.name}")
        if checksum:
            algorithm, value = checksum.split(":", 1)
            if file_hash(target, algorithm) != value:
                raise PipelineError(f"Publisher checksum mismatch: {target.name}")

    if path.exists():
        validate(path)  # An existing raw file is never silently replaced.
    else:
        partial = path.with_suffix(path.suffix + ".part")
        for attempt in range(3):
            offset = partial.stat().st_size if partial.exists() else 0
            if expected_size is not None and offset > expected_size:
                partial.unlink()
                offset = 0
            if expected_size is not None and offset == expected_size:
                break
            headers = {"User-Agent": "multicare-data/0.1 research"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as response:
                    append = offset > 0 and response.status == 206
                    if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise PipelineError("Invalid HTTP resume range")
                    with partial.open("ab" if append else "wb") as stream:
                        shutil.copyfileobj(response, stream, 1024 * 1024)
                break
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        validate(partial)
        os.replace(partial, path)
    return {"path": path.name, "url": url, "size": path.stat().st_size,
            "publisher_checksum": checksum, "sha256": file_hash(path)}


def resolve_latest(project: Project) -> dict:
    cfg = project.config("source")
    record = get_json(f"https://zenodo.org/api/records/{cfg['concept_record_id']}")
    latest_link = record.get("links", {}).get("latest")
    if latest_link:
        record = get_json(latest_link)
    return {"record_id": str(record["id"]), "version": record["metadata"]["version"],
            "pinned_record_id": cfg["record_id"],
            "matches_pin": str(record["id"]) == str(cfg["record_id"])}


def acquire(project: Project, tables_only: bool = False) -> dict:
    cfg = project.config("source")
    record_path = project.path("data/raw/zenodo_record.json")
    if record_path.exists():
        record = read_json(record_path)
    else:
        record = get_json(cfg["api_url"])
        if str(record["id"]) != str(cfg["record_id"]) or record["metadata"]["version"] != cfg["version"]:
            raise PipelineError("Zenodo identity does not match the pinned release")
        write_json(record_path, record, immutable=True)
    required = list(cfg["required_tables"])
    if not tables_only:
        required += cfg["archive_names"]
        required += ["abstracts.parquet"]
    files = {f["key"]: f for f in record["files"]}
    previous = project.path("data/raw/source_lock.json")
    lock = read_json(previous) if previous.exists() else {
        "dataset": cfg["dataset"], "version": cfg["version"], "record_id": cfg["record_id"],
        "doi": cfg["doi"], "license": cfg["license"], "downloaded_at": now(), "files": [],
        "record_sha256": file_hash(record_path), "source_config_sha256": file_hash(project.path("configs/source.yaml")),
    }
    if lock["version"] != cfg["version"] or lock["record_id"] != cfg["record_id"]:
        raise PipelineError("Raw directory belongs to another release; use a separate project root")
    locked = {f["path"]: f for f in lock["files"]}
    for name in required:
        if name not in files:
            raise PipelineError(f"Publisher missing required file: {name}")
        item = files[name]
        path = safe_path(project.path("data/raw/multicare"), name)
        print(f"Acquire {name} ({item['size'] / 1e6:.1f} MB)", flush=True)
        entry = download(item["links"]["self"], path, item["size"], item["checksum"])
        if name in locked and locked[name]["sha256"] != entry["sha256"]:
            raise PipelineError(f"Immutable source changed: {name}")
        locked[name] = entry
        lock["files"] = [locked[k] for k in sorted(locked)]
        lock["complete"] = set(cfg["required_tables"] + cfg["archive_names"]).issubset(locked)
        write_json(previous, lock)
    return {"files": len(locked), "complete": lock["complete"], "version": cfg["version"]}


def verify_source(project: Project, require_images: bool = False) -> dict:
    cfg = project.config("source")
    lock = read_json(project.path("data/raw/source_lock.json"))
    if lock["record_id"] != cfg["record_id"] or lock["version"] != cfg["version"]:
        raise PipelineError("Source pin mismatch")
    if lock["source_config_sha256"] != file_hash(project.path("configs/source.yaml")):
        raise PipelineError("Source config changed after acquisition")
    if lock["record_sha256"] != file_hash(project.path("data/raw/zenodo_record.json")):
        raise PipelineError("Publisher metadata changed")
    entries = {f["path"]: f for f in lock["files"]}
    unique_files = entries
    if len(unique_files) != len(lock["files"]):
        raise PipelineError("Duplicate entries in source lock")
    required = cfg["required_tables"] + (cfg["archive_names"] if require_images else [])
    if not set(required).issubset(entries):
        raise PipelineError("Source acquisition is incomplete; run acquire")
    if not lock.get("license"):
        raise PipelineError("Unknown source license")
    for item in entries.values():
        path = safe_path(project.path("data/raw/multicare"), item["path"])
        if not path.exists() or path.stat().st_size != item["size"] or file_hash(path) != item["sha256"]:
            raise PipelineError(f"Source integrity failure: {path}")
    return {"status": "PASS", "files_verified": len(entries), "complete": lock["complete"]}


def extract_images(project: Project) -> dict:
    verify_source(project, require_images=True)
    target = project.path("data/raw/multicare/images")
    count = 0
    for archive_name in project.config("source")["archive_names"]:
        with zipfile.ZipFile(project.path(f"data/raw/multicare/{archive_name}")) as archive:
            for entry in archive.infolist():
                path = safe_path(target, entry.filename)
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise PipelineError("Symlinks are not permitted in source archives")
                if entry.is_dir():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                # Verify existing files against the ZIP CRC by streaming the source again.
                with archive.open(entry) as source:
                    if path.exists():
                        import hashlib
                        h = hashlib.sha256()
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            h.update(block)
                        if h.hexdigest() != file_hash(path):
                            raise PipelineError(f"Extracted source changed: {path}")
                    else:
                        partial = path.with_suffix(path.suffix + ".part")
                        with partial.open("wb") as dest:
                            shutil.copyfileobj(source, dest)
                        os.replace(partial, path)
                count += 1
    return {"images_extracted": count}
