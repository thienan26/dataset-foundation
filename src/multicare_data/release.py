from __future__ import annotations

import shutil
import tempfile
from collections import Counter
from pathlib import Path

from .common import PipelineError, Project, digest, file_hash, read_json, write_json, write_rows
from .evaluate import evaluate
from .sample import single_targets


def freeze_taxonomy(project: Project) -> dict:
    cfg = project.config("release")
    feasibility = project.rows("reports/disease_feasibility.parquet")
    classes = [r for r in feasibility if r["status"] == "in_v001"]
    if len(classes) < cfg["minimum_classes"]:
        raise PipelineError(f"Need at least {cfg['minimum_classes']} feasible classes; found {len(classes)}")
    destination = project.path(f"taxonomy/frozen/{cfg['taxonomy_version']}.json")
    value = {"version": cfg["taxonomy_version"], "source_version": project.config("source")["version"],
             "scope_ontology": project.config("scope")["ontology"],
             "scope_ontology_commit": project.config("scope")["ontology_commit"],
             "scope": {"infectious": True, "tropical_annotation": "curated only; otherwise unknown"},
             "classes": {r["concept_id"]: {"name": r["disease"], "usable_cases": r["usable_cases"],
                                             "independent_groups": r["independent_groups"]}
                         for r in sorted(classes, key=lambda r: r["concept_id"])}}
    catalog = {r["concept_id"]: r for r in project.rows("data/catalog/disease_catalog.parquet")}
    for cid in value["classes"]:
        value["classes"][cid]["aliases"] = catalog[cid]["aliases"]
    write_json(destination, value, immutable=True)
    return {"taxonomy": str(destination.relative_to(project.root)), "classes": len(classes)}


def manifest_rows(project: Project) -> list[dict]:
    samples = {r["case_id"]: r for r in single_targets(project)}
    candidates = {r["case_id"]: r for r in project.rows("data/qc/feasible_candidates.parquet")}
    assignments = {r["case_id"]: r for r in project.rows("data/splits/assignments.parquet")}
    selected = {r["case_id"]: r for r in project.rows("data/qc/selected_images.parquet")}
    adjudication = {r["case_id"]: r for r in project.rows("data/reviews/adjudication.parquet")}
    humans = {r["case_id"]: r for r in project.rows("data/reviews/human_reviews.parquet", optional=True)}
    output = []
    for cid, row in candidates.items():
        assignment, image = assignments.get(cid), selected.get(cid)
        if not assignment or not image:
            raise PipelineError(f"Missing split/image in final candidate {cid}")
        review = adjudication.get(cid, {})
        human = humans.get(cid, {})
        source = samples[cid]
        image_path = Path(image["path"])
        output.append({**source, "sample_id": digest(f"{project.config('release')['version']}:{cid}"),
                       "selected_image_id": image["image_id"], "image_caption_audit_only": image["caption"],
                       "image_modality": image["image_modality"], "image_relevance": image["image_relevance"],
                       "image_license": image["license"], "image_sha256": image["image_sha256"],
                       "pixel_sha256": image["pixel_sha256"], "perceptual_hash": image["perceptual_hash"],
                       "image_path": f"images/{digest(cid)}{image_path.suffix.lower()}",
                       "llm_a_decision": review.get("llm_a_decision"),
                       "llm_b_decision": review.get("llm_b_decision"),
                       "review_status": review.get("review_status"),
                       "human_decision": human.get("human_decision", "not_sampled"),
                       "partition": assignment["partition"], "fold": assignment["fold"],
                       "split_group": assignment["split_group"],
                       "raw_text_sha256": source["raw_text_sha256"],
                       "clinical_text_sha256": row["clinical_text_sha256"],
                       "source_version": project.config("source")["version"],
                       "source_record_id": project.config("source")["record_id"],
                       "source_license": source["source_license"]})
    return output


def freeze(project: Project, version: str) -> dict:
    cfg = project.config("release")
    if version != cfg["version"]:
        raise PipelineError(f"Expected release version {cfg['version']}; edit the versioned config for a new release")
    target = project.path(f"data/releases/{version}")
    if target.exists():
        raise PipelineError(f"Release is immutable and already exists: {target}")
    audit = evaluate(project)
    if audit["status"] != "PASS":
        raise PipelineError(f"Release gate failed: {audit['failed_checks']}")
    manifest = manifest_rows(project)
    taxonomy_path = project.path(f"taxonomy/frozen/{cfg['taxonomy_version']}.json")
    if not taxonomy_path.exists():
        raise PipelineError("Freeze the feasible taxonomy first")
    release_parent = target.parent
    release_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{version}-", dir=release_parent))
    try:
        write_rows(stage / "manifest.parquet", manifest)
        write_rows(stage / "development.parquet", [r for r in manifest if r["partition"] == "development"])
        write_rows(stage / "frozen_test.parquet", [r for r in manifest if r["partition"] == "frozen_test"])
        write_rows(stage / "folds.parquet", [r for r in manifest if r["partition"] == "development"])
        write_rows(stage / "screening.parquet", project.rows("data/screening/labels.parquet"))
        write_rows(stage / "scope_screening.parquet", project.rows("data/screening/scope.parquet"))
        write_rows(stage / "out_of_scope_candidates.parquet", project.rows("data/candidates/out_of_scope_candidates.parquet"))
        write_rows(stage / "quarantine.parquet", project.rows("data/qc/quarantine.parquet", optional=True))
        write_rows(stage / "image_manifest.parquet", [{k: r.get(k) for k in ["sample_id", "selected_image_id", "image_path",
                    "image_sha256", "pixel_sha256", "perceptual_hash", "image_modality", "image_relevance", "image_license"]}
                    for r in manifest])
        write_rows(stage / "duplicate_groups.parquet", project.rows("data/groups/groups.parquet"))
        for name, source in (("taxonomy.json", taxonomy_path),
                             ("source_lock.json", project.path("data/raw/source_lock.json"))):
            shutil.copy2(source, stage / name)
        write_json(stage / "dataset_card.json", {
            "name": version, "source": project.config("source"), "task": "single-label multimodal infectious disease",
            "unit": "one clinical case paired with one independently reviewed image",
            "samples": len(manifest), "classes": sorted({r["diagnosis_label"] for r in manifest}),
            "partition_samples": dict(Counter(r["partition"] for r in manifest)),
            "review_models": project.config("review"), "licenses": sorted({r["source_license"] for r in manifest}),
            "image_licenses": sorted({r["image_license"] for r in manifest}),
            "aggregate_license": "CC BY-NC-SA 4.0; honor the individual article/image license ledger",
            "citation": "Nievas Offidani, M. MultiCaRe, version 3.0.1. Zenodo record 20416562. https://doi.org/10.5281/zenodo.20416562",
            "limitations": ["Silver labels require human adjudication/audit; model decisions are not clinical ground truth.",
                            "The source contains open-access case reports and is not population-representative.",
                            "Tropical status remains unknown unless explicitly curated."],
        })
        review_dir = stage / "reviews"
        review_dir.mkdir()
        for name, source in (("reviewer_a.parquet", "data/reviews/reviewer_a.parquet"),
                             ("reviewer_b.parquet", "data/reviews/reviewer_b.parquet"),
                             ("human_reviews.parquet", "data/reviews/human_reviews.parquet"),
                             ("adjudication.parquet", "data/reviews/adjudication.parquet")):
            rows = project.rows(source, optional=True)
            if rows:
                write_rows(review_dir / name, rows)
        quality = stage / "quality"
        quality.mkdir()
        for source in project.path("reports").glob("*.json"):
            shutil.copy2(source, quality / source.name)
        licenses = sorted({(r["article_id"], r["source_license"], r["selected_image_id"], r["image_license"])
                           for r in manifest})
        with (stage / "LICENSES.csv").open("w", encoding="utf-8", newline="") as stream:
            import csv
            writer = csv.writer(stream)
            writer.writerow(["article_id", "article_license", "image_id", "image_license"])
            writer.writerows(licenses)
        copied, checksums = set(), {}
        if cfg["copy_selected_images"]:
            all_images = {r["image_id"]: r for r in project.rows("data/normalized/images.parquet")}
            for row in manifest:
                if row["selected_image_id"] in copied:
                    continue
                source_image = project.path(all_images[row["selected_image_id"]]["path"])
                destination = stage / row["image_path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                if file_hash(source_image) != row["image_sha256"]:
                    raise PipelineError(f"Selected image changed while freezing: {source_image.name}")
                shutil.copy2(source_image, destination)
                if file_hash(destination) != row["image_sha256"]:
                    raise PipelineError(f"Copied image checksum mismatch: {destination.name}")
                checksums[row["image_path"]] = row["image_sha256"]
                copied.add(row["selected_image_id"])
        audit = evaluate(project)
        if audit["status"] != "PASS":
            raise PipelineError("Release gate changed during freeze")
        for path in sorted(stage.rglob("*")):
            if path.is_file() and path.name != "checksums.json":
                checksums[str(path.relative_to(stage)).replace("\\", "/")] = file_hash(path)
        write_json(stage / "checksums.json", {"algorithm": "SHA-256", "files": checksums})
        write_json(stage / "quality/release_audit.json", audit)
        # Include the copied audit in the checksum inventory before committing the directory.
        checksums["quality/release_audit.json"] = file_hash(stage / "quality/release_audit.json")
        write_json(stage / "checksums.json", {"algorithm": "SHA-256", "files": checksums})
        stage.rename(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return verify_release(project, version)


def verify_release(project: Project, version: str) -> dict:
    if version != project.config("release")["version"]:
        raise PipelineError("Requested version differs from the configured immutable release")
    root = project.path(f"data/releases/{version}")
    if not root.is_dir():
        raise PipelineError(f"Release does not exist: {root}")
    lock = read_json(root / "checksums.json")
    errors = []
    for relative, expected in lock["files"].items():
        path = project.path(f"data/releases/{version}/{relative}")
        if not path.is_file() or file_hash(path) != expected:
            errors.append(relative)
    audit = read_json(root / "quality/release_audit.json")
    if audit.get("status") != "PASS":
        errors.append("quality/release_audit.json:status")
    manifest = project.rows(f"data/releases/{version}/manifest.parquet")
    if not manifest:
        errors.append("manifest.parquet:empty")
    result = {"status": "PASS" if not errors else "FAIL", "version": version,
              "files_verified": len(lock["files"]), "samples": len(manifest), "errors": errors}
    if errors:
        raise PipelineError(f"Frozen release verification failed: {errors}")
    return result
