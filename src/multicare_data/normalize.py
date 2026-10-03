from __future__ import annotations

import csv
import os
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

from .acquire import verify_source
from .common import PipelineError, Project, canonical, digest, write_json

ARTICLE_SCHEMA = pa.schema([
    ("article_id", pa.string()), ("title", pa.string()), ("pmcid", pa.string()),
    ("pmid", pa.string()), ("doi", pa.string()), ("year", pa.string()), ("license", pa.string()),
    ("source_url", pa.string()), ("keywords", pa.list_(pa.string())), ("mesh_terms", pa.list_(pa.string())),
    ("publication_metadata_json", pa.string()),
])
CASE_SCHEMA = pa.schema([
    ("case_id", pa.string()), ("article_id", pa.string()), ("raw_case_text", pa.string()),
    ("raw_text_sha256", pa.string()), ("age", pa.float64()), ("gender", pa.string()),
    ("patient_link_id", pa.string()),
])
IMAGE_SCHEMA = pa.schema([
    ("image_id", pa.string()), ("case_id", pa.string()), ("article_id", pa.string()),
    ("caption", pa.string()), ("source_image_id", pa.string()), ("filename", pa.string()),
    ("path", pa.string()), ("license", pa.string()), ("image_modality", pa.string()),
    ("metadata_json", pa.string()),
])
REJECT_SCHEMA = pa.schema([("image_id", pa.string()), ("case_id", pa.string()), ("reason", pa.string())])


def _parquet_rows(path, columns=None, batch_size=4096):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        yield from batch.to_pylist()


def _write_stream(path, records, schema, batch_size=4096):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    writer = None
    batch = []
    count = 0
    try:
        for record in records:
            batch.append(record)
            if len(batch) >= batch_size:
                table = pa.Table.from_pylist(batch, schema=schema)
                writer = writer or pq.ParquetWriter(temp, schema, compression="zstd")
                writer.write_table(table)
                count += len(batch)
                batch.clear()
                del table
        if batch:
            table = pa.Table.from_pylist(batch, schema=schema)
            writer = writer or pq.ParquetWriter(temp, schema, compression="zstd")
            writer.write_table(table)
            count += len(batch)
        if writer:
            writer.close()
        else:
            pq.write_table(pa.Table.from_pylist([], schema=schema), temp, compression="zstd")
        os.replace(temp, path)
    except Exception:
        if writer:
            writer.close()
        raise
    return count


def _metadata_record(source):
    nested = source.get("article_metadata")
    records = nested if isinstance(nested, list) else [nested if isinstance(nested, dict) else source]
    yield from ((source, row) for row in records)


def _case_record(source):
    nested = source.get("cases")
    records = nested if isinstance(nested, list) else [nested if isinstance(nested, dict) else source]
    yield from ((source, row) for row in records)


def normalize(project: Project) -> dict:
    verify_source(project)
    raw_root = project.path("data/raw/multicare")
    article_map = {}

    def articles():
        for source in _parquet_rows(raw_root / "metadata.parquet"):
            for parent, row in _metadata_record(source):
                article_id = parent.get("article_id") or row.get("article_id") or row.get("pmcid")
                if not article_id or article_id in article_map:
                    raise PipelineError(f"Missing or duplicate MultiCaRe article ID: {article_id}")
                license_value = row.get("license") or "UNKNOWN"
                article_map[article_id] = license_value
                yield {"article_id": article_id, "title": row.get("title") or "",
                       "pmcid": row.get("pmcid") or article_id, "pmid": str(row.get("pmid") or ""),
                       "doi": row.get("doi") or "", "year": str(row.get("year") or ""),
                       "license": license_value, "source_url": row.get("link") or "",
                       "keywords": row.get("keywords") or [], "mesh_terms": row.get("mesh_terms") or [],
                       "publication_metadata_json": canonical(row)}

    article_count = _write_stream(project.path("data/normalized/articles.parquet"), articles(), ARTICLE_SCHEMA)
    case_map = {}

    def cases():
        for source in _parquet_rows(raw_root / "cases.parquet"):
            for parent, row in _case_record(source):
                cid = row.get("case_id")
                if not cid or cid in case_map:
                    raise PipelineError(f"Missing or duplicate MultiCaRe case ID: {cid}")
                aid = parent.get("article_id") or row.get("article_id") or cid.split("_")[0]
                if aid not in article_map:
                    raise PipelineError(f"Orphan case: {cid}")
                text = row.get("case_text")
                if not isinstance(text, str):
                    raise PipelineError(f"Non-string case text: {cid}")
                case_map[cid] = aid
                yield {"case_id": cid, "article_id": aid, "raw_case_text": text,
                       "raw_text_sha256": digest(text), "age": row.get("age"),
                       "gender": row.get("gender"), "patient_link_id": None}

    case_count = _write_stream(project.path("data/normalized/cases.parquet"), cases(), CASE_SCHEMA)

    # Image relationships are a small ID index; read nested image tables in bounded batches.
    original_map = {}
    for source in _parquet_rows(raw_root / "case_images.parquet", columns=["article_id", "case_images"]):
        groups = source.get("case_images")
        groups = groups if isinstance(groups, list) else [groups] if isinstance(groups, dict) else []
        for group in groups:
            cid = group.get("case_id")
            for image in group.get("case_image_list") or []:
                image_id = image["image_id"]
                if image_id not in original_map:
                    original_map[image_id] = cid
                else:
                    previous = original_map[image_id]
                    if isinstance(previous, set):
                        previous.add(cid)
                    elif previous != cid:
                        original_map[image_id] = {previous, cid}

    seen_images = set()
    rejected = Counter()

    def accepted_image_records():
        # Stream large captions/labels metadata in bounded batches, not one giant DataFrame.
        with (raw_root / "captions_and_labels.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                cid, filename, image_id = row["patient_id"], row["file"], row["file_id"]
                if cid not in case_map:
                    continue
                source_cases = original_map.get(row["main_image"])
                if source_cases != cid and not (isinstance(source_cases, set) and cid in source_cases):
                    continue
                if image_id in seen_images:
                    raise PipelineError(f"Duplicate MultiCaRe image ID: {image_id}")
                seen_images.add(image_id)
                aid = case_map[cid]
                yield {"image_id": image_id, "case_id": cid, "article_id": aid,
                       "caption": row.get("caption") or "", "source_image_id": row["main_image"],
                       "filename": filename,
                       "path": f"data/raw/multicare/images/{filename[:4]}/{filename[:5]}/{filename}",
                       "license": row.get("license") or article_map[aid],
                       "image_modality": row.get("image_subtype") or row.get("image_type") or "unknown",
                       "metadata_json": canonical(row)}

    def rejected_image_records():
        # Re-read the CSV rather than retaining another 100k-row list in memory.
        with (raw_root / "captions_and_labels.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                cid, image_id = row["patient_id"], row["file_id"]
                if cid not in case_map:
                    rejected["ORPHAN_IMAGE_CASE"] += 1
                    yield {"image_id": image_id, "case_id": cid, "reason": "ORPHAN_IMAGE_CASE"}
                    continue
                source_cases = original_map.get(row["main_image"])
                if source_cases != cid and not (isinstance(source_cases, set) and cid in source_cases):
                    rejected["IMAGE_RELATIONSHIP_UNVERIFIED"] += 1
                    yield {"image_id": image_id, "case_id": cid,
                           "reason": "IMAGE_RELATIONSHIP_UNVERIFIED"}

    image_count = _write_stream(project.path("data/normalized/images.parquet"),
                                accepted_image_records(), IMAGE_SCHEMA)
    rejected_count = _write_stream(project.path("data/normalized/rejected_images.parquet"),
                                   rejected_image_records(), REJECT_SCHEMA)
    summary = {"articles": article_count, "cases": case_count, "images": image_count,
               "rejected_images": rejected_count, "rejected_by_reason": dict(rejected),
               "source_version": project.config("source")["version"],
               "source_schema": "nested article_metadata/cases/case_images flattened to normalized entities"}
    write_json(project.path("reports/normalization.json"), summary)
    return summary

